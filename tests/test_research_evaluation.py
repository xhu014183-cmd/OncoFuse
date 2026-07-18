from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from hcc_multimodal.research_cohort import _configuration_hash, _file_hash
from hcc_multimodal.research_evaluation import (
    evaluate_research_cohort,
    validate_adjudications,
)
from hcc_multimodal.research_models import (
    AdjudicationRecord,
    AdjudicationReview,
    AdjudicationSet,
    FollowupRunRecord,
    IndexTreatment,
    ResearchCase,
    ResearchCaseRun,
    ResearchCohortManifest,
    ResearchProtocol,
    ResearchRunManifest,
    ReviewLabel,
    StudyReference,
)


REVIEW_TIME = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _review(reviewer: str, label: ReviewLabel) -> AdjudicationReview:
    return AdjudicationReview(
        reviewer_id_hash=reviewer,
        label=label,
        reason_codes=["BLINDED_IMAGE_REVIEW"],
        reviewed_at=REVIEW_TIME,
    )


def _adjudication(
    patient_id: str,
    study_id: str,
    label: ReviewLabel,
    *,
    disagreement: bool = False,
) -> AdjudicationRecord:
    if disagreement:
        alternative: ReviewLabel = (
            "no_progression" if label == "progression" else "progression"
        )
        return AdjudicationRecord(
            patient_id=patient_id,
            study_id=study_id,
            reviewers=[_review("reviewer-a", label), _review("reviewer-b", alternative)],
            adjudicator=_review("reviewer-c", label),
            final_label=label,
        )
    return AdjudicationRecord(
        patient_id=patient_id,
        study_id=study_id,
        reviewers=[_review("reviewer-a", label), _review("reviewer-b", label)],
        final_label=label,
    )


def _study(patient_id: str, days: int, *, baseline: bool = False) -> StudyReference:
    treatment_date = date(2026, 1, 15)
    study_date = date(2026, 1, 1) if baseline else treatment_date + timedelta(days=days)
    suffix = "BASE" if baseline else f"FU_{days}"
    return StudyReference(
        study_id=suffix,
        study_date=study_date,
        phase="portal_venous",
        ct_dir=f"{patient_id}/{suffix}/ct",
        seg_file=f"{patient_id}/{suffix}/seg.dcm",
        study_instance_uid=f"1.2.840.{patient_id}.{days}.1",
        series_instance_uid=f"1.2.840.{patient_id}.{days}.2",
        seg_series_instance_uid=f"1.2.840.{patient_id}.{days}.3",
        frame_of_reference_uid=f"1.2.840.{patient_id}.{days}.4",
        scanner_group="thin" if patient_id.endswith(("POS", "NEG")) else "thick",
        registration_status="not_applicable" if baseline else "verified",
    )


def _write_locked_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    definitions = [
        ("D_POS", "CENTER_DEV", 100, 1.0, 1.0, 1.0),
        ("D_NEG", "CENTER_DEV", 170, 0.0, 0.0, 0.0),
        ("D_LATE", "CENTER_DEV", 190, 1.0, 0.5, 0.75),
        ("D_SHORT", "CENTER_DEV", 100, 0.0, None, None),
        ("D_UNK", "CENTER_DEV", 160, None, 0.5, None),
        ("E_POS", "CENTER_EXT", 120, 1.0, 1.0, 1.0),
    ]
    protocol = ResearchProtocol(
        cohort_id="LOCKED_SYNTHETIC",
        external_test_center_ids=["CENTER_EXT"],
        split_salt="locked-synthetic-salt",
    )
    manifest_cases: list[ResearchCase] = []
    run_cases: list[ResearchCaseRun] = []
    for patient_id, center, days, image, lab, fusion in definitions:
        baseline = _study(patient_id, -14, baseline=True)
        followup = _study(patient_id, days)
        manifest_cases.append(
            ResearchCase(
                patient_id=patient_id,
                center_id=center,
                treatment=IndexTreatment(
                    treatment_date=date(2026, 1, 15),
                    treatment_type="TACE" if patient_id != "D_UNK" else "ablation",
                ),
                baseline=baseline,
                followups=[followup],
                labs_file=f"{patient_id}/labs.json",
            )
        )
        run_cases.append(
            ResearchCaseRun(
                patient_id=patient_id,
                center_id=center,
                split="external_test" if center == "CENTER_EXT" else "development",
                treatment_date=date(2026, 1, 15),
                treatment_type="TACE" if patient_id != "D_UNK" else "ablation",
                disposition="included",
                reason_codes=[],
                followups=[
                    FollowupRunRecord(
                        study_id=followup.study_id,
                        study_date=followup.study_date,
                        days_after_treatment=days,
                        scanner_group=followup.scanner_group,
                        disposition="included" if fusion is not None else "indeterminate",
                        reason_codes=[],
                        image_only_score=image,
                        lab_only_score=lab,
                        rule_fusion_score=fusion,
                    )
                ],
            )
        )

    protocol_path = tmp_path / "protocol.yaml"
    manifest_path = tmp_path / "manifest.json"
    protocol.write_json(protocol_path)
    ResearchCohortManifest(
        cohort_id=protocol.cohort_id,
        data_root="data",
        cases=manifest_cases,
    ).write_json(manifest_path)
    feature_dir = tmp_path / "features"
    run_path = feature_dir / "research_run_manifest.json"
    ResearchRunManifest(
        cohort_id=protocol.cohort_id,
        protocol_hash=_file_hash(protocol_path),
        manifest_hash=_file_hash(manifest_path),
        configuration_hash=_configuration_hash(protocol_path),
        git_commit="2082649",
        external_test_center_ids=["CENTER_EXT"],
        cases=run_cases,
        counts={"included": len(run_cases)},
    ).write_json(run_path)

    labels_dir = tmp_path / "labels"
    development_path = labels_dir / "development.json"
    external_path = labels_dir / "external.json"
    AdjudicationSet(
        cohort_id=protocol.cohort_id,
        scope="development",
        records=[
            _adjudication("D_POS", "FU_100", "progression", disagreement=True),
            _adjudication("D_NEG", "FU_170", "no_progression"),
            _adjudication("D_LATE", "FU_190", "progression"),
            _adjudication("D_SHORT", "FU_100", "no_progression"),
            _adjudication("D_UNK", "FU_160", "indeterminate"),
        ],
    ).write_json(development_path)
    AdjudicationSet(
        cohort_id=protocol.cohort_id,
        scope="external_test",
        records=[_adjudication("E_POS", "FU_120", "progression")],
    ).write_json(external_path)
    return protocol_path, manifest_path, run_path, development_path, external_path


def test_endpoint_rules_metrics_and_deterministic_outputs(tmp_path: Path):
    protocol, manifest, run, labels, _ = _write_locked_fixture(tmp_path)
    validation_path = validate_adjudications(
        run,
        labels,
        tmp_path / "adjudication-validation.json",
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["raw_agreement"] == pytest.approx(0.8)
    assert validation["adjudication_rate"] == pytest.approx(0.2)

    first_path = evaluate_research_cohort(
        protocol,
        manifest,
        run,
        labels,
        tmp_path / "evaluation-a",
        scope="development",
        bootstrap_iterations_override=20,
    )
    second_path = evaluate_research_cohort(
        protocol,
        manifest,
        run,
        labels,
        tmp_path / "evaluation-b",
        scope="development",
        bootstrap_iterations_override=20,
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first["endpoint_counts"] == {"indeterminate": 3, "negative": 1, "positive": 1}
    dispositions = {item["patient_id"]: item for item in first["case_dispositions"]}
    assert dispositions["D_POS"]["endpoint_status"] == "positive"
    assert dispositions["D_NEG"]["endpoint_status"] == "negative"
    assert "FIRST_PROGRESSION_ONLY_AFTER_180_DAYS" in dispositions["D_LATE"]["reason_codes"]
    assert "INSUFFICIENT_NEGATIVE_FOLLOWUP_ASCERTAINMENT" in dispositions["D_SHORT"]["reason_codes"]
    assert "INDETERMINATE_REVIEW_WITHIN_180_DAYS" in dispositions["D_UNK"]["reason_codes"]
    assert first["baselines"]["rule_fusion"]["metrics"]["auprc"] == 1.0
    assert first["bootstrap_iterations"] == 20
    assert set(first["paired_differences"]) == {
        "fusion_minus_image",
        "fusion_minus_lab",
        "image_minus_lab",
    }
    assert set(first["sensitivity_analyses"]) == {
        "indeterminate_as_negative",
        "indeterminate_as_positive",
    }
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second


def test_external_labels_require_one_time_unlock_and_scope_isolation(tmp_path: Path):
    protocol, manifest, run, _, external_labels = _write_locked_fixture(tmp_path)
    output = tmp_path / "external-evaluation"
    with pytest.raises(PermissionError, match="--unlock-external"):
        evaluate_research_cohort(
            protocol,
            manifest,
            run,
            external_labels,
            output,
            scope="external_test",
        )

    result_path = evaluate_research_cohort(
        protocol,
        manifest,
        run,
        external_labels,
        output,
        scope="external_test",
        unlock_external=True,
        bootstrap_iterations_override=10,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["endpoint_counts"]["positive"] == 1
    assert {item["center_id"] for item in result["case_dispositions"]} == {"CENTER_EXT"}
    unlock = json.loads((output / "external_test_unlock.json").read_text(encoding="utf-8"))
    assert unlock["explicit_authorization"] is True
    with pytest.raises(PermissionError, match="already been unlocked"):
        evaluate_research_cohort(
            protocol,
            manifest,
            run,
            external_labels,
            output,
            scope="external_test",
            unlock_external=True,
            bootstrap_iterations_override=10,
        )


def test_adjudication_contract_blocks_missing_third_reader_and_duplicates():
    with pytest.raises(ValidationError, match="third adjudicator"):
        AdjudicationRecord(
            patient_id="P1",
            study_id="S1",
            reviewers=[
                _review("reviewer-a", "progression"),
                _review("reviewer-b", "no_progression"),
            ],
            final_label="progression",
        )
    record = _adjudication("P1", "S1", "progression")
    with pytest.raises(ValidationError, match="Duplicate adjudication"):
        AdjudicationSet(
            cohort_id="C1",
            scope="development",
            records=[record, record],
        )


def test_labels_must_be_physically_separate_from_feature_directory(tmp_path: Path):
    protocol, manifest, run, labels, _ = _write_locked_fixture(tmp_path)
    leaked_labels = run.parent / "labels.json"
    leaked_labels.write_bytes(labels.read_bytes())
    with pytest.raises(ValueError, match="physically separate"):
        evaluate_research_cohort(
            protocol,
            manifest,
            run,
            leaked_labels,
            tmp_path / "evaluation",
            scope="development",
            bootstrap_iterations_override=5,
        )
