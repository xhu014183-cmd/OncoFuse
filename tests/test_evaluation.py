from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcc_multimodal.evaluation import (
    EvaluationCohort,
    evaluate_cohort,
    evaluate_cohort_file,
)


def _payload(record_count: int = 24) -> dict:
    return {
        "cohort_id": "SYNTHETIC_INTERFACE_CHECK",
        "endpoint": {
            "name": "radiographic_progression_within_followup_window",
            "followup_window_days": 180,
            "positive_label": "radiographic progression signal within 180 days",
            "preregistration_id": "SYNTHETIC-NOT-A-REGISTRY-ID",
        },
        "inclusion_criteria": ["Deidentified paired research record"],
        "exclusion_criteria": ["Missing endpoint label"],
        "records": [
            {
                "patient_id": f"P{index:03d}",
                "split": "test",
                "center": "A" if index % 2 else "B",
                "scanner_group": "thin" if index % 3 else "thick",
                "baseline_date": "2026-01-01",
                "followup_date": "2026-04-01",
                "endpoint_label": index % 2,
                "image_only_score": 0.8 if index % 2 else 0.2,
                "lab_only_score": 0.7 if index % 2 else 0.3,
                "rule_fusion_score": 0.9 if index % 2 else 0.1,
                "missing_reasons": [],
            }
            for index in range(record_count)
        ],
    }


def test_patient_level_evaluation_reports_three_deterministic_baselines():
    cohort = EvaluationCohort.model_validate(_payload())
    result = evaluate_cohort(cohort, bootstrap_iterations=50, seed=7)
    assert result["patient_count"] == 24
    assert set(result["baselines"]) == {"image_only", "lab_only", "rule_fusion"}
    assert result["baselines"]["rule_fusion"]["metrics"]["auroc"] == 1.0
    assert result["baselines"]["rule_fusion"]["bootstrap_95_ci"]["auroc"] == [1.0, 1.0]
    assert result["baselines"]["rule_fusion"]["performance_claim_permitted"] is False
    assert set(result["stratified_results"]) == {"center", "scanner_group"}


def test_patient_split_leakage_is_rejected():
    payload = _payload(2)
    leaked = dict(payload["records"][0])
    leaked["split"] = "validation"
    payload["records"].append(leaked)
    with pytest.raises(ValueError, match="multiple dataset splits"):
        EvaluationCohort.model_validate(payload)


def test_followup_outside_preregistered_window_is_rejected():
    payload = _payload(2)
    payload["records"][0]["followup_date"] = "2027-01-01"
    with pytest.raises(ValueError, match="exceeds endpoint window"):
        EvaluationCohort.model_validate(payload)


def test_small_single_class_cohort_is_marked_interface_only():
    payload = _payload(4)
    for record in payload["records"]:
        record["endpoint_label"] = 0
    cohort = EvaluationCohort.model_validate(payload)
    result = evaluate_cohort(cohort, bootstrap_iterations=10)
    assert result["quality"]["status"] == "warning"
    assert result["baselines"]["image_only"]["metrics"]["auroc"] is None


def test_evaluation_file_round_trip(tmp_path: Path):
    source = tmp_path / "cohort.json"
    target = tmp_path / "evaluation.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    result_path = evaluate_cohort_file(
        source,
        target,
        bootstrap_iterations=10,
        seed=3,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_path == target
    assert result["cohort_id"] == "SYNTHETIC_INTERFACE_CHECK"
    assert result["bootstrap"] == {"iterations": 10, "seed": 3, "unit": "patient"}
