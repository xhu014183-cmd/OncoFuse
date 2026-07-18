from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import csv
import io
import json

import numpy as np

from .evaluation import _bootstrap_intervals, _point_metrics
from .research_cohort import (
    _configuration_hash,
    _file_hash,
    load_research_manifest,
    load_research_protocol,
)
from .research_models import (
    AdjudicationSet,
    AdjudicationValidationReport,
    EvaluationScope,
    ExternalTestUnlockAudit,
    FollowupRunRecord,
    PatientEndpointResult,
    ReviewLabel,
    ResearchCaseRun,
    ResearchEvaluationArtifact,
    ResearchRunManifest,
)
from .schemas import QualityCheck, QualityEvidence, QualityStatus, SourceReference


BASELINE_FIELDS = {
    "image_only": "image_only_score",
    "lab_only": "lab_only_score",
    "rule_fusion": "rule_fusion_score",
}


def load_adjudications(path: str | Path) -> AdjudicationSet:
    return AdjudicationSet.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _cohen_kappa(adjudications: AdjudicationSet) -> tuple[float | None, float | None, float | None]:
    if not adjudications.records:
        return None, None, None
    labels: tuple[ReviewLabel, ...] = (
        "progression",
        "no_progression",
        "indeterminate",
    )
    first = [record.reviewers[0].label for record in adjudications.records]
    second = [record.reviewers[1].label for record in adjudications.records]
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
    expected = sum(
        (first.count(label) / len(first)) * (second.count(label) / len(second))
        for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    adjudication_rate = sum(record.adjudicator is not None for record in adjudications.records) / len(first)
    return observed, kappa, adjudication_rate


def _validate_adjudication_data(
    run: ResearchRunManifest,
    adjudications: AdjudicationSet,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if adjudications.cohort_id != run.cohort_id:
        errors.append("Adjudication cohort_id does not match the research run")
    expected_split = "development" if adjudications.scope == "development" else "external_test"
    run_cases = {case.patient_id: case for case in run.cases}
    expected_keys: set[tuple[str, str]] = set()
    for case in run.cases:
        if case.split != expected_split:
            continue
        for followup in case.followups:
            if followup.disposition in {"included", "indeterminate"}:
                expected_keys.add((case.patient_id, followup.study_id))
    actual_keys = {(record.patient_id, record.study_id) for record in adjudications.records}
    for record in adjudications.records:
        found_case = run_cases.get(record.patient_id)
        if found_case is None:
            errors.append(f"Unknown adjudicated patient: {record.patient_id}")
            continue
        if found_case.split != expected_split:
            errors.append(
                f"Adjudication scope leak: {record.patient_id} belongs to {found_case.split}, "
                f"not {expected_split}"
            )
        valid_studies = {followup.study_id for followup in found_case.followups}
        if record.study_id not in valid_studies:
            errors.append(f"Unknown follow-up study: {record.patient_id}/{record.study_id}")
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"Missing adjudications for {len(missing)} eligible follow-up studies")
    if extra:
        errors.append(f"Adjudications include {len(extra)} ineligible follow-up studies")
    if not adjudications.records:
        warnings.append("Adjudication set is empty")
    return errors, warnings


def validate_adjudications(
    run_path: str | Path,
    adjudications_path: str | Path,
    output_path: str | Path,
    *,
    external_unlock_audit: str | Path | None = None,
) -> Path:
    run = ResearchRunManifest.model_validate_json(Path(run_path).read_text(encoding="utf-8"))
    adjudications = load_adjudications(adjudications_path)
    if adjudications.scope == "external_test":
        if external_unlock_audit is None:
            raise PermissionError(
                "External-test adjudications require an existing external unlock audit"
            )
        audit = ExternalTestUnlockAudit.model_validate_json(
            Path(external_unlock_audit).read_text(encoding="utf-8")
        )
        if (
            audit.cohort_id != run.cohort_id
            or audit.protocol_hash != run.protocol_hash
            or audit.manifest_hash != run.manifest_hash
            or audit.configuration_hash != run.configuration_hash
        ):
            raise PermissionError("External unlock audit does not match the locked research run")
    errors, warnings = _validate_adjudication_data(run, adjudications)
    agreement, kappa, adjudication_rate = _cohen_kappa(adjudications)
    report = AdjudicationValidationReport(
        cohort_id=run.cohort_id,
        scope=adjudications.scope,
        valid=not errors,
        record_count=len(adjudications.records),
        raw_agreement=agreement,
        cohen_kappa=kappa,
        adjudication_rate=adjudication_rate,
        errors=errors,
        warnings=warnings,
        sources=[
            SourceReference(
                source_id=Path(adjudications_path).name,
                source_type="blinded_adjudication_set",
                data_origin="user_supplied",
                deidentified=True,
            )
        ],
    )
    target = Path(output_path)
    report.write_json(target)
    return target


def _missing_pattern(record: Any) -> str:
    missing = [
        name
        for name, field in BASELINE_FIELDS.items()
        if getattr(record, field) is None
    ]
    return "none" if not missing else "+".join(missing)


def _endpoint_for_case(
    case: ResearchCaseRun,
    adjudication_lookup: dict[tuple[str, str], Any],
) -> PatientEndpointResult:
    usable = [
        followup
        for followup in case.followups
        if (case.patient_id, followup.study_id) in adjudication_lookup
    ]
    usable.sort(key=lambda item: (item.days_after_treatment, item.study_id))
    labeled = [
        (followup, adjudication_lookup[(case.patient_id, followup.study_id)].final_label)
        for followup in usable
    ]
    selected: FollowupRunRecord | None
    label: Literal[0, 1] | None
    endpoint_status: Literal["positive", "negative", "indeterminate"]
    positives = [item for item in labeled if item[1] == "progression" and item[0].days_after_treatment <= 180]
    if positives:
        selected, _ = positives[0]
        label = 1
        endpoint_status = "positive"
        reasons = ["FIRST_ADJUDICATED_PROGRESSION_WITHIN_180_DAYS"]
    else:
        qualifying_negatives = [
            item
            for item in labeled
            if item[1] == "no_progression" and 150 <= item[0].days_after_treatment <= 210
        ]
        indeterminate_before_window = any(
            label_value == "indeterminate" and followup.days_after_treatment <= 180
            for followup, label_value in labeled
        )
        late_progression = any(
            label_value == "progression" and followup.days_after_treatment > 180
            for followup, label_value in labeled
        )
        all_early_nonprogression = all(
            label_value == "no_progression"
            for followup, label_value in labeled
            if followup.days_after_treatment <= 180
        )
        if (
            qualifying_negatives
            and all_early_nonprogression
            and not indeterminate_before_window
            and not late_progression
        ):
            selected, _ = min(
                qualifying_negatives,
                key=lambda item: (abs(item[0].days_after_treatment - 180), item[0].days_after_treatment),
            )
            label = 0
            endpoint_status = "negative"
            reasons = ["ADEQUATE_NONPROGRESSION_ASCERTAINMENT_150_TO_210_DAYS"]
        else:
            selected = usable[-1] if usable else None
            label = None
            endpoint_status = "indeterminate"
            reasons = []
            if late_progression:
                reasons.append("FIRST_PROGRESSION_ONLY_AFTER_180_DAYS")
            if indeterminate_before_window:
                reasons.append("INDETERMINATE_REVIEW_WITHIN_180_DAYS")
            if not qualifying_negatives:
                reasons.append("INSUFFICIENT_NEGATIVE_FOLLOWUP_ASCERTAINMENT")
            if not usable:
                reasons.append("NO_ADJUDICATED_FOLLOWUP")

    if selected is None:
        image_score = lab_score = fusion_score = None
        study_id = None
        selected_days = None
        scanner_group = "unknown"
        missing_pattern = "image_only+lab_only+rule_fusion"
    else:
        image_score = selected.image_only_score
        lab_score = selected.lab_only_score
        fusion_score = selected.rule_fusion_score
        study_id = selected.study_id
        selected_days = selected.days_after_treatment
        scanner_group = selected.scanner_group
        missing_pattern = _missing_pattern(selected)
    return PatientEndpointResult(
        patient_id=case.patient_id,
        center_id=case.center_id,
        split=case.split,
        label=label,
        endpoint_status=endpoint_status,
        selected_study_id=study_id,
        selected_days_after_treatment=selected_days,
        reason_codes=sorted(set(reasons)),
        image_only_score=image_score,
        lab_only_score=lab_score,
        rule_fusion_score=fusion_score,
        scanner_group=scanner_group,
        treatment_type=case.treatment_type,
        missing_pattern=missing_pattern,
    )


def _metric_bundle(
    records: list[PatientEndpointResult],
    field: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    available = [record for record in records if record.label is not None and getattr(record, field) is not None]
    labels = np.asarray([record.label for record in available], dtype=int)
    scores = np.asarray([getattr(record, field) for record in available], dtype=float)
    if len(available):
        metrics = _point_metrics(labels, scores)
        intervals = _bootstrap_intervals(labels, scores, iterations=iterations, seed=seed)
    else:
        empty_labels = np.asarray([], dtype=int)
        empty_scores = np.asarray([], dtype=float)
        metrics = _point_metrics(empty_labels, empty_scores)
        intervals = {name: None for name in metrics}
    positive_n = int(labels.sum()) if len(labels) else 0
    negative_n = int((labels == 0).sum()) if len(labels) else 0
    return {
        "n": len(available),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "missing_n": len(records) - len(available),
        "metrics": metrics,
        "bootstrap_95_ci": intervals,
    }


def _paired_metric_differences(
    records: list[PatientEndpointResult],
    first_field: str,
    second_field: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    available = [
        record
        for record in records
        if record.label is not None
        and getattr(record, first_field) is not None
        and getattr(record, second_field) is not None
    ]
    if not available:
        return {"n": 0, "point_difference": {}, "bootstrap_95_ci": {}}
    labels = np.asarray([record.label for record in available], dtype=int)
    first = np.asarray([getattr(record, first_field) for record in available], dtype=float)
    second = np.asarray([getattr(record, second_field) for record in available], dtype=float)
    first_metrics = _point_metrics(labels, first)
    second_metrics = _point_metrics(labels, second)
    names = ("auprc", "auroc", "brier_score")
    point: dict[str, float | None] = {}
    for name in names:
        first_value = first_metrics[name]
        second_value = second_metrics[name]
        point[name] = (
            first_value - second_value
            if first_value is not None and second_value is not None
            else None
        )
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        left = _point_metrics(labels[indices], first[indices])
        right = _point_metrics(labels[indices], second[indices])
        for name in names:
            left_value = left[name]
            right_value = right[name]
            if left_value is not None and right_value is not None:
                samples[name].append(left_value - right_value)
    intervals = {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        if values
        else None
        for name, values in samples.items()
    }
    return {"n": len(available), "point_difference": point, "bootstrap_95_ci": intervals}


def _evaluate_all_baselines(
    records: list[PatientEndpointResult],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        name: _metric_bundle(
            records,
            field,
            iterations=iterations,
            seed=seed + index,
        )
        for index, (name, field) in enumerate(BASELINE_FIELDS.items())
    }


def _stratified_results(
    records: list[PatientEndpointResult],
    *,
    iterations: int,
    seed: int,
    minimum_n: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("center_id", "scanner_group", "treatment_type", "missing_pattern"):
        groups: dict[str, list[PatientEndpointResult]] = defaultdict(list)
        for record in records:
            groups[str(getattr(record, field))].append(record)
        output[field] = {
            group: {
                "n": len(group_records),
                "inferential": len(group_records) >= minimum_n
                and len({item.label for item in group_records if item.label is not None}) == 2,
                "baselines": _evaluate_all_baselines(
                    group_records,
                    iterations=iterations,
                    seed=seed,
                ),
            }
            for group, group_records in sorted(groups.items())
        }
    return output


def _sensitivity_analysis(
    records: list[PatientEndpointResult],
    assumed_label: int,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    updated = [
        record.model_copy(update={"label": assumed_label})
        if record.label is None
        else record
        for record in records
    ]
    return _evaluate_all_baselines(updated, iterations=iterations, seed=seed)


def _write_research_report(artifact: ResearchEvaluationArtifact, path: Path) -> None:
    lines = [
        "# HCC multicenter research evaluation",
        "",
        f"- Cohort: {artifact.cohort_id}",
        f"- Scope: {artifact.scope}",
        f"- Primary metric: {artifact.primary_metric}",
        f"- Endpoint counts: {json.dumps(artifact.endpoint_counts, sort_keys=True)}",
        f"- Exploratory only: {artifact.exploratory_only}",
        "",
        "## Deterministic baselines",
        "",
        "| Baseline | N | AUPRC (95% CI) | AUROC | Sens. | Spec. | PPV | NPV | Brier | ECE |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("image_only", "lab_only", "rule_fusion"):
        result = artifact.baselines[name]
        metrics = result["metrics"]
        auprc_interval = result["bootstrap_95_ci"].get("auprc")
        lines.append(
            f"| {name} | {result['n']} | {metrics['auprc']} ({auprc_interval}) | "
            f"{metrics['auroc']} | "
            f"{metrics['sensitivity_at_0_5']} | {metrics['specificity_at_0_5']} | "
            f"{metrics['ppv_at_0_5']} | {metrics['npv_at_0_5']} | "
            f"{metrics['brier_score']} | {metrics['expected_calibration_error_10_bin']} |"
        )
    lines.extend(
        [
            "",
            "## Adjudication",
            "",
            f"- Raw agreement: {artifact.adjudication.get('raw_agreement')}",
            f"- Cohen's kappa: {artifact.adjudication.get('cohen_kappa')}",
            f"- Third-review rate: {artifact.adjudication.get('adjudication_rate')}",
            "",
            "## Paired baseline differences",
            "",
            f"```json\n{json.dumps(artifact.paired_differences, sort_keys=True, indent=2)}\n```",
            "",
            "## Missingness and sensitivity",
            "",
            f"- Missing patterns: {json.dumps(artifact.missingness.get('patterns', {}), sort_keys=True)}",
            "- Indeterminate endpoints are excluded from the primary analysis and evaluated "
            "under both positive and negative boundary assumptions in evaluation.json.",
            "",
            "## Interpretation boundary",
            "",
            artifact.intended_use,
            "",
            "> Research use only; this report does not establish clinical utility and contains no diagnostic or treatment recommendation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_research_cohort(
    protocol_path: str | Path,
    manifest_path: str | Path,
    run_path: str | Path,
    adjudications_path: str | Path,
    output_dir: str | Path,
    *,
    scope: EvaluationScope,
    unlock_external: bool = False,
    bootstrap_iterations_override: int | None = None,
) -> Path:
    protocol = load_research_protocol(protocol_path)
    manifest = load_research_manifest(manifest_path)
    run = ResearchRunManifest.model_validate_json(Path(run_path).read_text(encoding="utf-8"))
    if protocol.cohort_id != manifest.cohort_id or run.cohort_id != protocol.cohort_id:
        raise ValueError("Protocol, manifest, and research run cohort IDs must match")
    if (
        run.protocol_hash != _file_hash(protocol_path)
        or run.manifest_hash != _file_hash(manifest_path)
        or run.configuration_hash != _configuration_hash(protocol_path)
    ):
        raise ValueError("Research run lock does not match protocol, manifest, or rule configuration")
    labels_path = Path(adjudications_path).resolve()
    run_directory = Path(run_path).resolve().parent
    if labels_path.is_relative_to(run_directory):
        raise ValueError("Adjudication labels must be physically separate from feature-run artifacts")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unlock_path: Path | None = None
    if scope == "external_test":
        if not unlock_external:
            raise PermissionError("External test evaluation requires --unlock-external")
        unlock_path = output / "external_test_unlock.json"
        if unlock_path.exists():
            raise PermissionError("External test labels have already been unlocked in this output directory")
        unlock = ExternalTestUnlockAudit(
            cohort_id=run.cohort_id,
            protocol_hash=run.protocol_hash,
            manifest_hash=run.manifest_hash,
            configuration_hash=run.configuration_hash,
            git_commit=run.git_commit,
            external_test_center_ids=run.external_test_center_ids,
            unlocked_at=datetime.now(timezone.utc),
            sources=[
                SourceReference(
                    source_id=Path(run_path).name,
                    source_type="locked_research_run",
                )
            ],
        )
        unlock.write_json(unlock_path)
    elif unlock_external:
        raise ValueError("--unlock-external is valid only for external_test scope")

    adjudications = load_adjudications(adjudications_path)
    if adjudications.scope != scope:
        raise ValueError("Adjudication scope does not match requested evaluation scope")
    adjudication_report_path = validate_adjudications(
        run_path,
        adjudications_path,
        output / "adjudication_validation.json",
        external_unlock_audit=unlock_path,
    )
    adjudication_report = AdjudicationValidationReport.model_validate_json(
        adjudication_report_path.read_text(encoding="utf-8")
    )
    if not adjudication_report.valid:
        raise ValueError(f"Adjudication validation failed: {adjudication_report.errors}")

    expected_split = "development" if scope == "development" else "external_test"
    adjudication_lookup = {
        (record.patient_id, record.study_id): record for record in adjudications.records
    }
    endpoint_results = [
        _endpoint_for_case(case, adjudication_lookup)
        for case in run.cases
        if case.split == expected_split
    ]
    endpoint_results.sort(key=lambda item: item.patient_id)
    counts: dict[str, int] = dict(
        Counter(result.endpoint_status for result in endpoint_results)
    )
    for status in ("positive", "negative", "indeterminate"):
        counts.setdefault(status, 0)
    main_records = [record for record in endpoint_results if record.label is not None]
    iterations = bootstrap_iterations_override or protocol.bootstrap_iterations
    baselines = _evaluate_all_baselines(
        main_records,
        iterations=iterations,
        seed=protocol.bootstrap_seed,
    )
    paired = {
        "image_minus_lab": _paired_metric_differences(
            main_records,
            "image_only_score",
            "lab_only_score",
            iterations=iterations,
            seed=protocol.bootstrap_seed + 10,
        ),
        "fusion_minus_image": _paired_metric_differences(
            main_records,
            "rule_fusion_score",
            "image_only_score",
            iterations=iterations,
            seed=protocol.bootstrap_seed + 20,
        ),
        "fusion_minus_lab": _paired_metric_differences(
            main_records,
            "rule_fusion_score",
            "lab_only_score",
            iterations=iterations,
            seed=protocol.bootstrap_seed + 30,
        ),
    }
    strata = _stratified_results(
        main_records,
        iterations=iterations,
        seed=protocol.bootstrap_seed + 100,
        minimum_n=protocol.minimum_inferential_stratum_size,
    )
    sensitivity = {
        "indeterminate_as_positive": _sensitivity_analysis(
            endpoint_results,
            1,
            iterations=iterations,
            seed=protocol.bootstrap_seed + 200,
        ),
        "indeterminate_as_negative": _sensitivity_analysis(
            endpoint_results,
            0,
            iterations=iterations,
            seed=protocol.bootstrap_seed + 300,
        ),
    }
    missingness = {
        "patterns": dict(Counter(record.missing_pattern for record in endpoint_results)),
        "score_missing": {
            name: sum(getattr(record, field) is None for record in endpoint_results)
            for name, field in BASELINE_FIELDS.items()
        },
    }
    exploratory = scope == "development" or counts["positive"] < 20 or counts["negative"] < 20
    warnings = []
    if exploratory:
        warnings.append(
            "Results are exploratory because this is development scope or the external test "
            "contains fewer than 20 positive or 20 negative endpoints"
        )
    quality_status: QualityStatus = "warning" if warnings else "pass"
    artifact = ResearchEvaluationArtifact(
        cohort_id=run.cohort_id,
        scope=scope,
        bootstrap_iterations=iterations,
        protocol_hash=run.protocol_hash,
        manifest_hash=run.manifest_hash,
        configuration_hash=run.configuration_hash,
        git_commit=run.git_commit,
        quality=QualityEvidence(
            status=quality_status,
            checks=[
                QualityCheck(
                    check_id="LOCKED_RUN_HASHES",
                    status="pass",
                    message="Protocol, manifest, rules, and run hashes match",
                ),
                QualityCheck(
                    check_id="PATIENT_SCOPE_ISOLATION",
                    status="pass",
                    message=f"Only {expected_split} patients were evaluated",
                ),
                QualityCheck(
                    check_id="MINIMUM_CLASS_SUPPORT",
                    status="warning" if exploratory else "pass",
                    message=(
                        f"Endpoint support: {counts['positive']} positive and "
                        f"{counts['negative']} negative"
                    ),
                ),
            ],
            warnings=warnings,
        ),
        adjudication={
            "raw_agreement": adjudication_report.raw_agreement,
            "cohen_kappa": adjudication_report.cohen_kappa,
            "adjudication_rate": adjudication_report.adjudication_rate,
            "record_count": adjudication_report.record_count,
        },
        endpoint_counts=counts,
        case_dispositions=endpoint_results,
        baselines=baselines,
        paired_differences=paired,
        stratified_results=strata,
        sensitivity_analyses=sensitivity,
        missingness=missingness,
        exploratory_only=exploratory,
        intended_use=(
            "multicenter research validation only; not for diagnosis, staging, prognosis, "
            "treatment selection, or a clinical performance claim"
        ),
        sources=[
            SourceReference(
                source_id=Path(run_path).name,
                source_type="locked_research_run",
            ),
            SourceReference(
                source_id=Path(adjudications_path).name,
                source_type="blinded_adjudication_set",
                data_origin="user_supplied",
                deidentified=True,
            ),
        ],
    )
    evaluation_path = output / "evaluation.json"
    artifact.write_json(evaluation_path)
    (output / "case_dispositions.json").write_text(
        json.dumps(
            {
                "schema_version": artifact.schema_version,
                "pipeline_version": artifact.pipeline_version,
                "generated_at": artifact.generated_at.isoformat(),
                "cohort_id": artifact.cohort_id,
                "counts": counts,
                "cases": [item.to_dict() for item in endpoint_results],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "missingness.json").write_text(
        json.dumps(
            {
                "schema_version": artifact.schema_version,
                "pipeline_version": artifact.pipeline_version,
                "generated_at": artifact.generated_at.isoformat(),
                "cohort_id": artifact.cohort_id,
                "missingness": missingness,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    missing_table = io.StringIO(newline="")
    writer = csv.DictWriter(
        missing_table,
        fieldnames=[
            "patient_id",
            "endpoint_status",
            "missing_pattern",
            "image_only_missing",
            "lab_only_missing",
            "rule_fusion_missing",
            "reason_codes",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for item in endpoint_results:
        writer.writerow(
            {
                "patient_id": item.patient_id,
                "endpoint_status": item.endpoint_status,
                "missing_pattern": item.missing_pattern,
                "image_only_missing": item.image_only_score is None,
                "lab_only_missing": item.lab_only_score is None,
                "rule_fusion_missing": item.rule_fusion_score is None,
                "reason_codes": "|".join(item.reason_codes),
            }
        )
    (output / "missing_data.csv").write_text(missing_table.getvalue(), encoding="utf-8")
    _write_research_report(artifact, output / "research_report.md")
    return evaluation_path
