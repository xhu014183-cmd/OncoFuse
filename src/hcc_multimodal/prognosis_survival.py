"""Deterministic penalized Cox training and locked external validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .prognosis_models import (
    CoxModelBundle,
    PublicPrognosisCohortArtifact,
    PublicPrognosisRecord,
    SurvivalEndpointRecord,
)

PENALIZER_GRID = (0.001, 0.01, 0.1, 1.0)
FEATURE_SETS: dict[str, list[str]] = {
    "clinical_core": ["age_years", "female", "afp_ng_ml"],
    "imaging_core": [
        "lesion_count",
        "total_tumor_volume_ml",
        "max_lesion_extent_mm",
        "largest_lesion_sphericity",
    ],
    "fused_core": [
        "age_years",
        "female",
        "afp_ng_ml",
        "lesion_count",
        "total_tumor_volume_ml",
        "max_lesion_extent_mm",
        "largest_lesion_sphericity",
    ],
    "fused_extended": [
        "age_years",
        "female",
        "afp_ng_ml",
        "lesion_count",
        "total_tumor_volume_ml",
        "max_lesion_extent_mm",
        "largest_lesion_sphericity",
        "albumin_g_dl",
        "bilirubin_mg_dl",
        "inr",
        "alt_iu_l",
        "creatinine_mg_dl",
    ],
    "fused_extended_no_albumin": [
        "age_years",
        "female",
        "afp_ng_ml",
        "lesion_count",
        "total_tumor_volume_ml",
        "max_lesion_extent_mm",
        "largest_lesion_sphericity",
        "bilirubin_mg_dl",
        "inr",
        "alt_iu_l",
        "creatinine_mg_dl",
    ],
}
TRANSFORMS: dict[str, Literal["identity", "log1p"]] = {
    "age_years": "identity",
    "female": "identity",
    "afp_ng_ml": "log1p",
    "lesion_count": "log1p",
    "total_tumor_volume_ml": "log1p",
    "max_lesion_extent_mm": "log1p",
    "largest_lesion_sphericity": "identity",
    "albumin_g_dl": "identity",
    "bilirubin_mg_dl": "log1p",
    "inr": "identity",
    "alt_iu_l": "log1p",
    "creatinine_mg_dl": "log1p",
}


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _load_cohort(path: str | Path) -> PublicPrognosisCohortArtifact:
    source = Path(path)
    if source.is_dir():
        source = source / "prognosis-cohort.json"
    return PublicPrognosisCohortArtifact.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def _transform(value: float, method: str) -> float:
    if not math.isfinite(value):
        raise ValueError("Model features must be finite")
    if method == "log1p":
        if value < 0:
            raise ValueError("log1p feature values must be non-negative")
        return math.log1p(value)
    return value


def _raw_matrix(
    records: Sequence[PublicPrognosisRecord], feature_names: list[str]
) -> np.ndarray:
    rows: list[list[float]] = []
    for record in records:
        row: list[float] = []
        for name in feature_names:
            value = getattr(record, name)
            if value is None:
                raise ValueError(f"Patient {record.patient_id} is missing required feature {name}")
            row.append(_transform(float(value), TRANSFORMS[name]))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _standardize(
    matrix: np.ndarray,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fitted_means = matrix.mean(axis=0) if means is None else means
    fitted_scales = matrix.std(axis=0, ddof=0) if scales is None else scales
    if np.any(~np.isfinite(fitted_scales)) or np.any(fitted_scales <= 1e-12):
        raise ValueError("At least one model feature has zero variance")
    return (matrix - fitted_means) / fitted_scales, fitted_means, fitted_scales


def _cox_objective(
    beta: np.ndarray,
    matrix: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    penalizer: float,
) -> tuple[float, np.ndarray]:
    eta = matrix @ beta
    log_likelihood = 0.0
    gradient = np.zeros_like(beta)
    for event_time in np.unique(durations[events == 1]):
        event_mask = (durations == event_time) & (events == 1)
        event_count = int(event_mask.sum())
        risk_mask = durations >= event_time
        risk_eta = eta[risk_mask]
        shift = float(risk_eta.max())
        weights = np.exp(risk_eta - shift)
        denominator = float(weights.sum())
        log_likelihood += float(eta[event_mask].sum()) - event_count * (
            shift + math.log(denominator)
        )
        weighted_mean = (weights[:, None] * matrix[risk_mask]).sum(axis=0) / denominator
        gradient += matrix[event_mask].sum(axis=0) - event_count * weighted_mean
    objective = -log_likelihood + 0.5 * penalizer * float(beta @ beta)
    objective_gradient = -gradient + penalizer * beta
    return objective, objective_gradient


def _cox_hessian(
    beta: np.ndarray,
    matrix: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    penalizer: float,
) -> np.ndarray:
    eta = matrix @ beta
    hessian = np.eye(matrix.shape[1], dtype=float) * penalizer
    for event_time in np.unique(durations[events == 1]):
        event_count = int(((durations == event_time) & (events == 1)).sum())
        risk_matrix = matrix[durations >= event_time]
        risk_eta = eta[durations >= event_time]
        weights = np.exp(risk_eta - float(risk_eta.max()))
        probabilities = weights / float(weights.sum())
        weighted_mean = (probabilities[:, None] * risk_matrix).sum(axis=0)
        weighted_second = risk_matrix.T @ (probabilities[:, None] * risk_matrix)
        hessian += event_count * (
            weighted_second - np.outer(weighted_mean, weighted_mean)
        )
    return hessian


def _newton_coefficients(
    matrix: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    penalizer: float,
) -> np.ndarray | None:
    beta = np.zeros(matrix.shape[1], dtype=float)
    for _ in range(100):
        objective, gradient = _cox_objective(
            beta, matrix, durations, events, penalizer
        )
        if float(np.linalg.norm(gradient, ord=np.inf)) < 1e-7:
            return beta
        hessian = _cox_hessian(beta, matrix, durations, events, penalizer)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        directional_derivative = float(gradient @ step)
        accepted = False
        scale = 1.0
        for _ in range(30):
            candidate = beta - scale * step
            candidate_objective, _ = _cox_objective(
                candidate, matrix, durations, events, penalizer
            )
            if candidate_objective <= objective - 1e-4 * scale * directional_derivative:
                beta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return None
        if float(np.linalg.norm(scale * step)) < 1e-9 * (
            1.0 + float(np.linalg.norm(beta))
        ):
            _, final_gradient = _cox_objective(
                beta, matrix, durations, events, penalizer
            )
            return (
                beta
                if float(np.linalg.norm(final_gradient, ord=np.inf)) < 1e-5
                else None
            )
    return None


def _fit_coefficients(
    matrix: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    penalizer: float,
) -> np.ndarray:
    if int(events.sum()) < 2:
        raise ValueError("Cox training requires at least two observed events")
    objective = lambda beta: _cox_objective(
        beta, matrix, durations, events, penalizer
    )
    result = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "maxls": 30, "ftol": 1e-11, "gtol": 1e-8},
    )
    if result.success and np.isfinite(result.x).all():
        return np.asarray(result.x, dtype=float)
    if np.isfinite(result.x).all():
        _, gradient = objective(np.asarray(result.x, dtype=float))
        if float(np.linalg.norm(gradient, ord=np.inf)) < 1e-5:
            return np.asarray(result.x, dtype=float)
    messages = [f"L-BFGS-B: {result.message}"]
    newton = _newton_coefficients(matrix, durations, events, penalizer)
    if newton is not None and np.isfinite(newton).all():
        return newton
    messages.append("damped Newton: did not converge")
    start = (
        np.asarray(result.x, dtype=float)
        if np.isfinite(result.x).all()
        else np.zeros(matrix.shape[1], dtype=float)
    )
    result = minimize(
        objective,
        start,
        method="BFGS",
        jac=True,
        options={"maxiter": 3000, "gtol": 1e-7},
    )
    if result.success and np.isfinite(result.x).all():
        return np.asarray(result.x, dtype=float)
    if np.isfinite(result.x).all():
        _, gradient = objective(np.asarray(result.x, dtype=float))
        if float(np.linalg.norm(gradient, ord=np.inf)) < 1e-5:
            return np.asarray(result.x, dtype=float)
    messages.append(f"BFGS: {result.message}")
    raise RuntimeError("Penalized Cox optimization failed; " + "; ".join(messages))


def concordance_index(
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> float:
    concordant = 0.0
    comparable = 0
    for first in range(len(durations)):
        for second in range(first + 1, len(durations)):
            if durations[first] < durations[second] and events[first] == 1:
                earlier, later = first, second
            elif durations[second] < durations[first] and events[second] == 1:
                earlier, later = second, first
            else:
                continue
            comparable += 1
            if risks[earlier] > risks[later]:
                concordant += 1.0
            elif risks[earlier] == risks[later]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError("Concordance index has no comparable patient pairs")
    return concordant / comparable


def _baseline_hazard(
    matrix: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    beta: np.ndarray,
) -> tuple[list[float], list[float]]:
    relative = np.exp(np.clip(matrix @ beta, -50, 50))
    cumulative = 0.0
    times: list[float] = []
    hazards: list[float] = []
    for event_time in np.unique(durations[events == 1]):
        event_count = int(((durations == event_time) & (events == 1)).sum())
        risk_sum = float(relative[durations >= event_time].sum())
        cumulative += event_count / risk_sum
        times.append(float(event_time))
        hazards.append(float(cumulative))
    return times, hazards


def _event_stratified_folds(events: np.ndarray, folds: int, seed: int) -> list[list[int]]:
    if folds < 2 or len(events) < folds:
        raise ValueError("Not enough patients for the requested cross-validation folds")
    rng = np.random.default_rng(seed)
    groups = [np.where(events == value)[0].copy() for value in (1, 0)]
    for group in groups:
        rng.shuffle(group)
    result: list[list[int]] = [[] for _ in range(folds)]
    for group in groups:
        for index, patient_index in enumerate(group.tolist()):
            result[index % folds].append(int(patient_index))
    return [sorted(fold) for fold in result]


def _select_penalizer(
    raw: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    seed: int,
) -> tuple[float, dict[str, float]]:
    folds = _event_stratified_folds(events, 5, seed)
    scores: dict[str, float] = {}
    all_indices = np.arange(len(events))
    for penalizer in PENALIZER_GRID:
        fold_scores: list[float] = []
        for validation in folds:
            validation_indices = np.asarray(validation, dtype=int)
            training_indices = np.setdiff1d(all_indices, validation_indices)
            train, means, scales = _standardize(raw[training_indices])
            validation_matrix, _, _ = _standardize(
                raw[validation_indices], means, scales
            )
            beta = _fit_coefficients(
                train,
                durations[training_indices],
                events[training_indices],
                penalizer,
            )
            fold_scores.append(
                concordance_index(
                    durations[validation_indices],
                    events[validation_indices],
                    validation_matrix @ beta,
                )
            )
        scores[str(penalizer)] = float(np.mean(fold_scores))
    selected = max(PENALIZER_GRID, key=lambda value: (scores[str(value)], -value))
    return selected, scores


def _bootstrap_c_index(
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(durations), size=len(durations))
        try:
            samples.append(
                concordance_index(durations[indices], events[indices], risks[indices])
            )
        except ValueError:
            continue
    if not samples:
        return None
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _ph_checks(
    matrix: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    beta: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    eta = matrix @ beta
    residuals: list[np.ndarray] = []
    log_times: list[float] = []
    for index in np.where(events == 1)[0]:
        risk_mask = durations >= durations[index]
        weights = np.exp(np.clip(eta[risk_mask] - eta[risk_mask].max(), -50, 50))
        expected = (weights[:, None] * matrix[risk_mask]).sum(axis=0) / weights.sum()
        residuals.append(matrix[index] - expected)
        log_times.append(math.log(float(durations[index])))
    values = np.asarray(residuals)
    checks: list[dict[str, Any]] = []
    for column, name in enumerate(feature_names):
        result = stats.spearmanr(log_times, values[:, column])
        p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else None
        checks.append(
            {
                "feature": name,
                "spearman_rho": float(result.statistic),
                "p_value": p_value,
                "violation_flag": p_value is not None and p_value < 0.01,
                "interpretation": "screening diagnostic only; no post-hoc model respecification",
            }
        )
    return checks


def _bundle_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _train_one(
    records: list[PublicPrognosisRecord],
    endpoints: list[SurvivalEndpointRecord],
    *,
    specification: str,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[CoxModelBundle, dict[str, Any], dict[str, Any]]:
    feature_names = FEATURE_SETS[specification]
    endpoint_by_id = {item.patient_id: item for item in endpoints}
    durations = np.asarray([endpoint_by_id[item.patient_id].duration_days for item in records])
    events = np.asarray([endpoint_by_id[item.patient_id].event for item in records], dtype=int)
    raw = _raw_matrix(records, feature_names)
    outer_folds = _event_stratified_folds(events, 5, seed)
    out_of_fold = np.full(len(records), np.nan, dtype=float)
    outer_choices: list[dict[str, Any]] = []
    all_indices = np.arange(len(records))
    for fold_index, validation in enumerate(outer_folds):
        validation_indices = np.asarray(validation, dtype=int)
        training_indices = np.setdiff1d(all_indices, validation_indices)
        penalizer, tuning = _select_penalizer(
            raw[training_indices],
            durations[training_indices],
            events[training_indices],
            seed=seed + 100 + fold_index,
        )
        train, means, scales = _standardize(raw[training_indices])
        validation_matrix, _, _ = _standardize(raw[validation_indices], means, scales)
        beta = _fit_coefficients(
            train,
            durations[training_indices],
            events[training_indices],
            penalizer,
        )
        out_of_fold[validation_indices] = validation_matrix @ beta
        outer_choices.append(
            {"fold": fold_index + 1, "penalizer": penalizer, "inner_scores": tuning}
        )
    if not np.isfinite(out_of_fold).all():
        raise RuntimeError("Nested cross-validation did not score every development patient")
    final_penalizer, final_tuning = _select_penalizer(
        raw, durations, events, seed=seed + 999
    )
    standardized, means, scales = _standardize(raw)
    beta = _fit_coefficients(
        standardized, durations, events, final_penalizer
    )
    risks = standardized @ beta
    event_times, cumulative_hazard = _baseline_hazard(
        standardized, durations, events, beta
    )
    model_name = (
        "fused_extended"
        if specification == "fused_extended_no_albumin"
        else specification
    )
    stable_payload = {
        "model_name": model_name,
        "specification": specification,
        "feature_names": feature_names,
        "transformations": {name: TRANSFORMS[name] for name in feature_names},
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": beta.tolist(),
        "penalizer": final_penalizer,
        "baseline_event_times_days": event_times,
        "baseline_cumulative_hazard": cumulative_hazard,
        "training_n": len(records),
        "training_event_n": int(events.sum()),
        "seed": seed,
    }
    model_hash = _bundle_hash(stable_payload)
    bundle = CoxModelBundle(
        model_id=f"hcc-tace-os-{specification}-v1",
        model_name=model_name,  # type: ignore[arg-type]
        external_validation_cohort=(
            None if model_name == "fused_extended" else "hcc_tace_seg"
        ),
        feature_names=feature_names,
        transformations={name: TRANSFORMS[name] for name in feature_names},
        means=means.tolist(),
        scales=scales.tolist(),
        coefficients=beta.tolist(),
        penalizer=final_penalizer,
        baseline_event_times_days=event_times,
        baseline_cumulative_hazard=cumulative_hazard,
        development_reference_risks=sorted(float(value) for value in risks),
        median_risk_threshold=float(np.median(risks)),
        training_n=len(records),
        training_event_n=int(events.sum()),
        cross_validation_seed=seed,
        model_hash=model_hash,
        limitations=[
            "Retrospective public-cohort research model; no prospective clinical validation",
            "Risk index is relative and must not be translated into individual life expectancy",
            "Expert/public segmentation is required; no automatic segmentation is included",
        ],
    )
    internal = {
        "model_id": bundle.model_id,
        "specification": specification,
        "n": len(records),
        "event_n": int(events.sum()),
        "nested_oof_c_index": concordance_index(durations, events, out_of_fold),
        "bootstrap_95_ci": _bootstrap_c_index(
            durations,
            events,
            out_of_fold,
            iterations=bootstrap_iterations,
            seed=seed + 2000,
        ),
        "outer_penalizer_choices": outer_choices,
        "final_penalizer": final_penalizer,
        "full_development_tuning_scores": final_tuning,
        "proportional_hazards_screen": _ph_checks(
            standardized, durations, events, beta, feature_names
        ),
        "performance_claim_permitted": False,
    }
    split = {
        "model_id": bundle.model_id,
        "folds": [
            {
                "fold": index + 1,
                "validation_patient_ids": [records[item].patient_id for item in fold],
            }
            for index, fold in enumerate(outer_folds)
        ],
    }
    return bundle, internal, split


def train_prognosis_models(
    cohort_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 1729,
    bootstrap_iterations: int = 1000,
) -> Path:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    cohort = _load_cohort(cohort_path)
    cohort_ids = {
        *(item.cohort_id for item in cohort.feature_records),
        *(item.cohort_id for item in cohort.endpoint_records),
    }
    if cohort_ids != {"waw_tace"}:
        raise ValueError(
            "Training requires the development-only artifact; use development-cohort.json"
        )
    development = [item for item in cohort.feature_records if item.cohort_id == "waw_tace"]
    development_ids = {item.patient_id for item in development}
    endpoints = [
        item for item in cohort.endpoint_records if item.patient_id in development_ids
    ]
    if len(development) < 20:
        raise ValueError("At least 20 WAW-TACE patients are required for model training")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "development_cohort": "waw_tace",
        "models": {},
    }
    splits: dict[str, Any] = {"seed": seed, "models": {}}
    file_names = {
        "clinical_core": "model-bundle-clinical.json",
        "imaging_core": "model-bundle-imaging.json",
        "fused_core": "model-bundle-fused.json",
        "fused_extended": "model-bundle-fused-extended.json",
        "fused_extended_no_albumin": "model-bundle-fused-extended-no-albumin.json",
    }
    for index, specification in enumerate(file_names):
        bundle, internal, split = _train_one(
            development,
            endpoints,
            specification=specification,
            seed=seed + index * 10000,
            bootstrap_iterations=bootstrap_iterations,
        )
        bundle.write_json(output / file_names[specification])
        validation["models"][specification] = internal
        splits["models"][specification] = split
    _write_json(output / "internal-validation.json", validation)
    _write_json(output / "split-manifest.json", splits)
    (output / "model-card.md").write_text(
        "# HCC TACE Overall-Survival Model Card\n\n"
        "## 中文\n\n"
        "开发队列为WAW-TACE，锁定外部验证队列为HCC-TACE-Seg。临床、影像和"
        "融合Cox模型只使用预先指定的低维基线特征；GLM和DeepSeek不参与风险"
        "预测。WAW扩展模型仅作内部探索，去除白蛋白模型用于单位修正敏感性"
        "分析。本模型仅供研究，不能用于临床预后或治疗决策。\n\n"
        "## English\n\n"
        "Development: WAW-TACE. Locked external test: HCC-TACE-Seg.\n\n"
        "The clinical, imaging and fused penalized Cox models use pre-specified, "
        "low-dimensional baseline features. GLM and DeepSeek outputs are not predictors.\n\n"
        "The extended WAW-only model is exploratory and has no external validation. "
        "The no-albumin sensitivity model addresses the source unit inconsistency.\n\n"
        "Research use only; not for clinical prognosis or treatment decisions.\n",
        encoding="utf-8",
    )
    return output / "internal-validation.json"


def score_records(
    bundle: CoxModelBundle,
    records: Sequence[PublicPrognosisRecord],
) -> np.ndarray:
    raw = _raw_matrix(records, bundle.feature_names)
    matrix, _, _ = _standardize(
        raw,
        np.asarray(bundle.means, dtype=float),
        np.asarray(bundle.scales, dtype=float),
    )
    return matrix @ np.asarray(bundle.coefficients, dtype=float)


def _baseline_at(bundle: CoxModelBundle, horizon_days: float) -> float:
    hazard = 0.0
    for time, value in zip(
        bundle.baseline_event_times_days, bundle.baseline_cumulative_hazard
    ):
        if time > horizon_days:
            break
        hazard = value
    return hazard


def _km_survival(durations: np.ndarray, events: np.ndarray, horizon: float) -> float:
    survival = 1.0
    for time in np.unique(durations[(events == 1) & (durations <= horizon)]):
        at_risk = int((durations >= time).sum())
        observed = int(((durations == time) & (events == 1)).sum())
        if at_risk:
            survival *= 1.0 - observed / at_risk
    return float(survival)


def _calibration(
    bundle: CoxModelBundle,
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    horizon: float,
) -> list[dict[str, Any]]:
    quantiles = np.quantile(risks, [0.25, 0.5, 0.75])
    groups = np.digitize(risks, quantiles, right=True)
    baseline = _baseline_at(bundle, horizon)
    rows: list[dict[str, Any]] = []
    for group in range(4):
        selected = groups == group
        if not selected.any():
            continue
        predicted = np.exp(-baseline * np.exp(np.clip(risks[selected], -50, 50)))
        rows.append(
            {
                "quartile": group + 1,
                "n": int(selected.sum()),
                "mean_predicted_survival": float(predicted.mean()),
                "kaplan_meier_survival": _km_survival(
                    durations[selected], events[selected], horizon
                ),
            }
        )
    return rows


def _paired_delta_interval(
    durations: np.ndarray,
    events: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    point = concordance_index(durations, events, first) - concordance_index(
        durations, events, second
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(durations), size=len(durations))
        try:
            samples.append(
                concordance_index(durations[indices], events[indices], first[indices])
                - concordance_index(
                    durations[indices], events[indices], second[indices]
                )
            )
        except ValueError:
            continue
    return {
        "delta_c_index": point,
        "bootstrap_95_ci": (
            [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
            if samples
            else None
        ),
    }


def _plot_calibration(
    rows_by_horizon: dict[str, list[dict[str, Any]]], target: Path
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional research dependency
        return None
    figure, axes = plt.subplots(1, len(rows_by_horizon), figsize=(5 * len(rows_by_horizon), 4))
    if len(rows_by_horizon) == 1:
        axes = [axes]
    for axis, (horizon, rows) in zip(axes, rows_by_horizon.items()):
        predicted = [item["mean_predicted_survival"] for item in rows]
        observed = [item["kaplan_meier_survival"] for item in rows]
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.scatter(predicted, observed)
        axis.set(title=f"{horizon}-day calibration", xlabel="Predicted survival", ylabel="KM survival", xlim=(0, 1), ylim=(0, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return str(target)


def _plot_external_forest(results: dict[str, Any], target: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional research dependency
        return None
    order = ["clinical_core", "imaging_core", "fused_core"]
    labels = ["Clinical core", "Imaging core", "Fused core"]
    points = np.asarray([float(results[name]["c_index"]) for name in order])
    intervals = [results[name]["bootstrap_95_ci"] for name in order]
    lower = np.asarray(
        [float(interval[0]) if interval is not None else point for point, interval in zip(points, intervals)]
    )
    upper = np.asarray(
        [float(interval[1]) if interval is not None else point for point, interval in zip(points, intervals)]
    )
    positions = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(
        points,
        positions,
        xerr=np.vstack((points - lower, upper - points)),
        fmt="o",
        color="#1f618d",
        capsize=4,
    )
    axis.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.set(
        yticks=positions,
        yticklabels=labels,
        xlabel="Harrell C-index (patient bootstrap 95% CI)",
        title="Locked HCC-TACE-Seg external validation",
        xlim=(0.0, 1.0),
    )
    axis.invert_yaxis()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(target)


def evaluate_external_prognosis(
    cohort_path: str | Path,
    models_dir: str | Path,
    output_dir: str | Path,
    *,
    unlock_external: bool,
    bootstrap_iterations: int = 1000,
    seed: int = 1729,
) -> Path:
    if not unlock_external:
        raise ValueError("External validation requires explicit --unlock-external")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    unlock_audit = output / "external-unlock-audit.json"
    if unlock_audit.exists():
        raise RuntimeError("External validation has already been opened in this output directory")
    cohort = _load_cohort(cohort_path)
    cohort_ids = {
        *(item.cohort_id for item in cohort.feature_records),
        *(item.cohort_id for item in cohort.endpoint_records),
    }
    if cohort_ids != {"hcc_tace_seg"}:
        raise ValueError(
            "External evaluation requires external-test-cohort.json only"
        )
    records = [item for item in cohort.feature_records if item.cohort_id == "hcc_tace_seg"]
    record_ids = {item.patient_id for item in records}
    endpoints = [item for item in cohort.endpoint_records if item.patient_id in record_ids]
    if len(records) < 20:
        raise ValueError("At least 20 prepared HCC-TACE-Seg patients are required")
    endpoint_by_id = {item.patient_id: item for item in endpoints}
    durations = np.asarray([endpoint_by_id[item.patient_id].duration_days for item in records])
    events = np.asarray([endpoint_by_id[item.patient_id].event for item in records], dtype=int)
    model_files = {
        "clinical_core": "model-bundle-clinical.json",
        "imaging_core": "model-bundle-imaging.json",
        "fused_core": "model-bundle-fused.json",
    }
    model_root = Path(models_dir).resolve()
    results: dict[str, Any] = {}
    risks: dict[str, np.ndarray] = {}
    model_hashes: dict[str, str] = {}
    figure_paths: dict[str, str] = {}
    for index, (name, file_name) in enumerate(model_files.items()):
        path = model_root / file_name
        bundle = CoxModelBundle.model_validate_json(path.read_text(encoding="utf-8"))
        if bundle.development_cohort != "waw_tace":
            raise ValueError("External models must be frozen WAW-TACE development bundles")
        model_risks = score_records(bundle, records)
        risks[name] = model_risks
        model_hashes[name] = bundle.model_hash
        calibration = {
            str(horizon): _calibration(bundle, durations, events, model_risks, horizon)
            for horizon in (365.0, 730.0)
        }
        results[name] = {
            "model_id": bundle.model_id,
            "model_hash": bundle.model_hash,
            "n": len(records),
            "event_n": int(events.sum()),
            "c_index": concordance_index(durations, events, model_risks),
            "bootstrap_95_ci": _bootstrap_c_index(
                durations,
                events,
                model_risks,
                iterations=bootstrap_iterations,
                seed=seed + index,
            ),
            "calibration": calibration,
            "risk_group_threshold_source": "WAW-TACE development median",
            "high_risk_n": int((model_risks >= bundle.median_risk_threshold).sum()),
            "performance_claim_permitted": False,
        }
        calibration_figure = _plot_calibration(
            calibration,
            output / "figures" / f"calibration-{name}.png",
        )
        if calibration_figure is not None:
            figure_paths[f"calibration_{name}"] = str(
                Path(calibration_figure).relative_to(output)
            )
        bundle.model_copy(update={"external_validation_status": "evaluated"}).write_json(path)
    comparisons = {
        "fused_minus_clinical": _paired_delta_interval(
            durations,
            events,
            risks["fused_core"],
            risks["clinical_core"],
            iterations=bootstrap_iterations,
            seed=seed + 100,
        ),
        "fused_minus_imaging": _paired_delta_interval(
            durations,
            events,
            risks["fused_core"],
            risks["imaging_core"],
            iterations=bootstrap_iterations,
            seed=seed + 200,
        ),
    }
    forest_figure = _plot_external_forest(
        results, output / "figures" / "external-validation-forest.png"
    )
    if forest_figure is not None:
        figure_paths["external_validation_forest"] = str(
            Path(forest_figure).relative_to(output)
        )
    artifact = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "development_cohort": "waw_tace",
        "external_test_cohort": "hcc_tace_seg",
        "external_data_used_for_tuning": False,
        "results": results,
        "comparisons": comparisons,
        "figures": figure_paths,
        "interpretation": (
            "All results are reported regardless of direction or statistical uncertainty; "
            "no post-external-test tuning is permitted."
        ),
    }
    result_path = _write_json(output / "external-validation.json", artifact)
    _write_json(output / "model-comparison.json", comparisons)
    _write_json(
        unlock_audit,
        {
            "opened_at": datetime.now(UTC).isoformat(),
            "cohort": "hcc_tace_seg",
            "patient_n": len(records),
            "model_hashes": model_hashes,
            "authorization": "explicit_cli_unlock",
        },
    )
    return result_path


def score_single_record(
    bundle: CoxModelBundle,
    record: PublicPrognosisRecord,
) -> tuple[float, float, str]:
    risk = float(score_records(bundle, [record])[0])
    reference = np.asarray(bundle.development_reference_risks, dtype=float)
    percentile = float(100.0 * np.mean(reference <= risk))
    group = (
        "at_or_above_development_median"
        if risk >= bundle.median_risk_threshold
        else "lower_than_development_median"
    )
    return risk, percentile, group


def score_feature_mapping(
    bundle: CoxModelBundle,
    values: dict[str, float],
) -> tuple[float, float, str]:
    """Score one non-cohort case without manufacturing a public-cohort identity."""
    transformed: list[float] = []
    for name in bundle.feature_names:
        if name not in values:
            raise ValueError(f"Missing model feature: {name}")
        transformed.append(_transform(float(values[name]), bundle.transformations[name]))
    raw = np.asarray([transformed], dtype=float)
    matrix, _, _ = _standardize(
        raw,
        np.asarray(bundle.means, dtype=float),
        np.asarray(bundle.scales, dtype=float),
    )
    risk = float(matrix[0] @ np.asarray(bundle.coefficients, dtype=float))
    reference = np.asarray(bundle.development_reference_risks, dtype=float)
    percentile = float(100.0 * np.mean(reference <= risk))
    group = (
        "at_or_above_development_median"
        if risk >= bundle.median_risk_threshold
        else "lower_than_development_median"
    )
    return risk, percentile, group


__all__ = [
    "FEATURE_SETS",
    "PENALIZER_GRID",
    "concordance_index",
    "evaluate_external_prognosis",
    "score_feature_mapping",
    "score_records",
    "score_single_record",
    "train_prognosis_models",
]
