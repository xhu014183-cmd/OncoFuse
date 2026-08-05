from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from .schemas import (
    ArtifactModel,
    JsonModel,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
)


class EvaluationEndpoint(JsonModel):
    name: Literal["radiographic_progression_within_followup_window"]
    followup_window_days: int = Field(gt=0)
    positive_label: str
    preregistration_id: str | None = None


class CohortRecord(JsonModel):
    patient_id: str
    split: Literal["train", "validation", "test"]
    center: str
    scanner_group: str = "unknown"
    baseline_date: str
    followup_date: str
    endpoint_label: int = Field(ge=0, le=1)
    image_only_score: float | None = Field(default=None, ge=0, le=1)
    lab_only_score: float | None = Field(default=None, ge=0, le=1)
    rule_fusion_score: float | None = Field(default=None, ge=0, le=1)
    galad_score: float | None = Field(default=None, ge=0, le=1)
    treatment_events: list[dict[str, Any]] = Field(default_factory=list)
    missing_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def temporal_order(self) -> CohortRecord:
        if date.fromisoformat(self.baseline_date) >= date.fromisoformat(self.followup_date):
            raise ValueError("followup_date must occur after baseline_date")
        return self


class EvaluationCohort(ArtifactModel):
    cohort_id: str
    endpoint: EvaluationEndpoint
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    records: list[CohortRecord]

    @model_validator(mode="after")
    def patient_split_isolation(self) -> EvaluationCohort:
        splits: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            splits[record.patient_id].add(record.split)
            interval = (date.fromisoformat(record.followup_date) - date.fromisoformat(record.baseline_date)).days
            if interval > self.endpoint.followup_window_days:
                raise ValueError(
                    f"Patient {record.patient_id} follow-up interval {interval} exceeds endpoint "
                    f"window {self.endpoint.followup_window_days}"
                )
        leaked = {patient: values for patient, values in splits.items() if len(values) > 1}
        if leaked:
            raise ValueError(f"Patients occur in multiple dataset splits: {leaked}")
        return self


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def _point_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    predictions = scores >= 0.5
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predictions & positive))
    true_negative = int(np.sum(~predictions & negative))
    false_positive = int(np.sum(predictions & negative))
    false_negative = int(np.sum(~predictions & positive))
    sensitivity = true_positive / (true_positive + false_negative) if positive.any() else None
    specificity = true_negative / (true_negative + false_positive) if negative.any() else None
    ppv = true_positive / (true_positive + false_positive) if predictions.any() else None
    npv = true_negative / (true_negative + false_negative) if (~predictions).any() else None
    brier = float(np.mean((scores - labels) ** 2)) if len(labels) else None
    bins = np.minimum((scores * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.any():
            calibration_error += float(
                selected.mean() * abs(scores[selected].mean() - labels[selected].mean())
            )
    return {
        "auroc": _auroc(labels, scores),
        "auprc": _average_precision(labels, scores),
        "sensitivity_at_0_5": sensitivity,
        "specificity_at_0_5": specificity,
        "ppv_at_0_5": ppv,
        "npv_at_0_5": npv,
        "brier_score": brier,
        "expected_calibration_error_10_bin": calibration_error,
    }


def _bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        metrics = _point_metrics(labels[indices], scores[indices])
        for name, value in metrics.items():
            if value is not None:
                values[name].append(value)
    intervals: dict[str, list[float] | None] = {}
    for name in _point_metrics(labels, scores):
        samples = values.get(name, [])
        intervals[name] = (
            [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
            if samples
            else None
        )
    return intervals


def _evaluate_records(
    records: list[CohortRecord],
    score_field: str,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    available = [record for record in records if getattr(record, score_field) is not None]
    labels = np.asarray([record.endpoint_label for record in available], dtype=int)
    scores = np.asarray([getattr(record, score_field) for record in available], dtype=float)
    point = _point_metrics(labels, scores) if len(available) else {
        "auroc": None,
        "auprc": None,
        "sensitivity_at_0_5": None,
        "specificity_at_0_5": None,
        "brier_score": None,
        "expected_calibration_error_10_bin": None,
    }
    intervals = (
        _bootstrap_intervals(labels, scores, iterations=bootstrap_iterations, seed=seed)
        if len(available) >= 2
        else {name: None for name in point}
    )
    return {
        "n": len(available),
        "positive_n": int(labels.sum()) if len(labels) else 0,
        "negative_n": int((labels == 0).sum()) if len(labels) else 0,
        "missing_n": len(records) - len(available),
        "metrics": point,
        "bootstrap_95_ci": intervals,
        "performance_claim_permitted": False,
    }


def evaluate_cohort(
    cohort: EvaluationCohort,
    *,
    split: Literal["validation", "test"] = "test",
    bootstrap_iterations: int = 1000,
    seed: int = 1729,
) -> dict[str, Any]:
    records = [record for record in cohort.records if record.split == split]
    baselines = {
        name: _evaluate_records(
            records,
            field,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + index,
        )
        for index, (name, field) in enumerate(
            (
                ("image_only", "image_only_score"),
                ("lab_only", "lab_only_score"),
                ("rule_fusion", "rule_fusion_score"),
                ("galad", "galad_score"),
            )
        )
    }
    strata: dict[str, Any] = {}
    for field in ("center", "scanner_group"):
        groups: dict[str, list[CohortRecord]] = defaultdict(list)
        for record in records:
            groups[getattr(record, field)].append(record)
        strata[field] = {
            group: {
                name: _evaluate_records(
                    group_records,
                    score_field,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + index,
                )
                for index, (name, score_field) in enumerate(
                    (
                        ("image_only", "image_only_score"),
                        ("lab_only", "lab_only_score"),
                        ("rule_fusion", "rule_fusion_score"),
                        ("galad", "galad_score"),
                    )
                )
            }
            for group, group_records in sorted(groups.items())
        }
    missing_reasons = _counter(
        reason for record in records for reason in record.missing_reasons
    )
    insufficient = len(records) < 20 or len({record.endpoint_label for record in records}) < 2
    warnings = []
    if insufficient:
        warnings.append(
            "The selected split is too small or lacks both classes; metrics are interface checks only"
        )
    if cohort.endpoint.preregistration_id is None:
        warnings.append("Endpoint has no preregistration identifier")
    quality_status: QualityStatus = "warning" if warnings else "pass"
    return {
        "schema_version": cohort.schema_version,
        "pipeline_version": cohort.pipeline_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [
            SourceReference(
                source_id=cohort.cohort_id,
                source_type="patient_level_evaluation_cohort",
                data_origin="user_supplied",
            ).to_dict()
        ],
        "cohort_id": cohort.cohort_id,
        "split": split,
        "endpoint": cohort.endpoint.to_dict(),
        "patient_count": len({record.patient_id for record in records}),
        "record_count": len(records),
        "bootstrap": {"iterations": bootstrap_iterations, "seed": seed, "unit": "patient"},
        "quality": QualityEvidence(
            status=quality_status,
            checks=[
                QualityCheck(
                    check_id="PATIENT_SPLIT_ISOLATION",
                    status="pass",
                    message="No patient appears in more than one split",
                ),
                QualityCheck(
                    check_id="ENDPOINT_CLASS_SUPPORT",
                    status="warning" if insufficient else "pass",
                    message="Both endpoint classes and minimum sample size checked",
                ),
            ],
            warnings=warnings,
        ).to_dict(),
        "missing_reasons": missing_reasons,
        "baselines": baselines,
        "stratified_results": strata,
        "intended_use": (
            "research evaluation interface; results do not establish clinical performance"
        ),
    }


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def evaluate_cohort_file(
    path: str | Path,
    output_path: str | Path,
    *,
    split: Literal["validation", "test"] = "test",
    bootstrap_iterations: int = 1000,
    seed: int = 1729,
) -> Path:
    source = Path(path)
    cohort = EvaluationCohort.model_validate_json(source.read_text(encoding="utf-8"))
    result = evaluate_cohort(
        cohort,
        split=split,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
