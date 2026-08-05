from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.galad import (
    calculate_galad_from_values,
    calculate_galad_score,
    load_galad_coefficients,
    normalize_sex,
)


def _write_lab_json(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labs.json"
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    return path


def _galad_rows(
    afp: str = "400",
    afp_unit: str = "ng/mL",
    l3: str = "15",
    dcp: str = "1000",
    dcp_unit: str = "mAU/mL",
) -> list[dict]:
    return [
        {"analyte": "AFP", "value": afp, "unit": afp_unit, "observed_at": "2026-07-01"},
        {"analyte": "AFP-L3%", "value": l3, "unit": "%", "observed_at": "2026-07-01"},
        {"analyte": "PIVKA-II", "value": dcp, "unit": dcp_unit, "observed_at": "2026-07-01"},
    ]


def test_sex_normalization():
    assert normalize_sex("男") == "male"
    assert normalize_sex("Male") == "male"
    assert normalize_sex("F") == "female"
    assert normalize_sex("女性") == "female"
    assert normalize_sex("unknown") is None
    assert normalize_sex(None) is None


def test_high_risk_patient_scores_high():
    result = calculate_galad_from_values(
        400, 15, 1000, age_years=60, sex="男", patient_id="GALAD-HIGH"
    )
    assert result.status == "calculated"
    assert result.score is not None and result.score > 0.7
    assert result.risk_tier == "HIGH"
    assert result.missing_inputs == []
    # contributing factor shares are a partition of unity
    assert math.isclose(sum(result.contributing_factors.values()), 1.0)
    assert set(result.term_contributions) == {
        "age_years",
        "sex_male",
        "log10_afp_ng_ml",
        "afp_l3_pct",
        "log10_dcp_mau_ml",
    }
    assert "not for diagnosis" in result.intended_use


def test_low_risk_patient_scores_low():
    result = calculate_galad_from_values(
        5, 5, 20, age_years=45, sex="female", patient_id="GALAD-LOW"
    )
    assert result.status == "calculated"
    assert result.score is not None and result.score < 0.3
    assert result.risk_tier == "LOW"


def test_score_is_monotonic_in_markers():
    low = calculate_galad_from_values(5, 5, 20, age_years=50, sex="male", patient_id="P1")
    high = calculate_galad_from_values(500, 30, 2000, age_years=50, sex="male", patient_id="P2")
    assert low.score is not None and high.score is not None
    assert high.score > low.score


def test_missing_marker_is_fail_closed():
    result = calculate_galad_from_values(
        None, 15, 1000, age_years=60, sex="male", patient_id="GALAD-MISS"
    )
    assert result.status == "incomplete"
    assert result.score is None
    assert result.risk_tier is None
    assert result.missing_inputs == ["AFP"]


def test_multiple_missing_inputs_are_all_listed():
    result = calculate_galad_from_values(
        None, None, 1000, age_years=None, sex="未知", patient_id="GALAD-MISS2"
    )
    assert result.status == "incomplete"
    assert result.missing_inputs == ["sex", "age_years", "AFP", "AFP-L3%"]


def test_zero_or_negative_values_are_rejected_not_logged():
    # log10(0) would blow up; fail closed instead
    result = calculate_galad_from_values(0, 15, 1000, age_years=60, sex="male", patient_id="GALAD-ZERO")
    assert result.status == "incomplete"
    assert "AFP" in result.missing_inputs
    negative = calculate_galad_from_values(10, 15, -5, age_years=60, sex="male", patient_id="GALAD-NEG")
    assert negative.status == "incomplete"
    assert "DCP" in negative.missing_inputs


def test_afp_l3_percentage_bounds():
    result = calculate_galad_from_values(10, 150, 100, age_years=60, sex="male", patient_id="GALAD-L3")
    assert result.status == "incomplete"
    assert "AFP-L3%" in result.missing_inputs


def test_score_from_parsed_lab_evidence(tmp_path: Path):
    path = _write_lab_json(tmp_path, _galad_rows())
    labs = parse_laboratory_report(path, patient_id="GALAD-PIPE")
    result = calculate_galad_score(labs, age_years=60, sex="男")
    assert result.status == "calculated"
    assert result.risk_tier == "HIGH"
    assert result.inputs is not None
    assert result.inputs.afp_ng_ml == pytest.approx(400)
    assert result.inputs.dcp_mau_ml == pytest.approx(1000)
    assert result.evidence_refs == [
        "GALAD-PIPE:labs:AFP@2026-07-01",
        "GALAD-PIPE:labs:AFP-L3%@2026-07-01",
        "GALAD-PIPE:labs:DCP@2026-07-01",
    ]


def test_unit_conversion_feeds_score(tmp_path: Path):
    # DCP reported in AU/L must be normalized to mAU/mL before scoring
    # (project convention: 1 AU/L == 1 mAU/mL)
    rows = _galad_rows(dcp="1000", dcp_unit="AU/L")
    path = _write_lab_json(tmp_path, rows)
    labs = parse_laboratory_report(path, patient_id="GALAD-UNIT")
    result = calculate_galad_score(labs, age_years=60, sex="male")
    assert result.status == "calculated"
    assert result.inputs is not None
    assert result.inputs.dcp_mau_ml == pytest.approx(1000)


def test_missing_analyte_in_lab_file_is_incomplete(tmp_path: Path):
    rows = [row for row in _galad_rows() if row["analyte"] != "AFP-L3%"]
    path = _write_lab_json(tmp_path, rows)
    labs = parse_laboratory_report(path, patient_id="GALAD-NOL3")
    result = calculate_galad_score(labs, age_years=60, sex="male")
    assert result.status == "incomplete"
    assert result.missing_inputs == ["AFP-L3%"]
    assert result.evidence_refs == [
        "GALAD-NOL3:labs:AFP@2026-07-01",
        "GALAD-NOL3:labs:DCP@2026-07-01",
    ]


def test_non_numeric_inputs_fail_closed_not_raise():
    result = calculate_galad_from_values(
        "abc", 15, 1000, age_years="old", sex="male", patient_id="GALAD-NAN"
    )
    assert result.status == "incomplete"
    assert result.score is None
    assert set(result.missing_inputs) == {"age_years", "AFP"}


def test_non_contemporaneous_markers_are_fail_closed(tmp_path: Path):
    rows = _galad_rows()
    rows[2]["observed_at"] = "2025-12-01"  # DCP 7 months before the others
    path = _write_lab_json(tmp_path, rows)
    labs = parse_laboratory_report(path, patient_id="GALAD-WINDOW")
    result = calculate_galad_score(labs, age_years=60, sex="male")
    assert result.status == "incomplete"
    assert "contemporaneous_window" in result.missing_inputs


def test_markers_within_date_window_calculate(tmp_path: Path):
    rows = _galad_rows()
    rows[2]["observed_at"] = "2026-05-01"  # 61 days before the others
    path = _write_lab_json(tmp_path, rows)
    labs = parse_laboratory_report(path, patient_id="GALAD-WINDOW-OK")
    result = calculate_galad_score(labs, age_years=60, sex="male")
    assert result.status == "calculated"


def test_date_span_check_can_be_disabled(tmp_path: Path):
    rows = _galad_rows()
    rows[2]["observed_at"] = "2025-12-01"
    path = _write_lab_json(tmp_path, rows)
    labs = parse_laboratory_report(path, patient_id="GALAD-NOWINDOW")
    result = calculate_galad_score(
        labs, age_years=60, sex="male", max_analyte_date_span_days=None
    )
    assert result.status == "calculated"


def test_coefficient_config_validation(tmp_path: Path):
    bad = tmp_path / "bad_coeffs.yaml"
    bad.write_text("version: galad-bad\nintercept: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terms"):
        load_galad_coefficients(bad)
    config = load_galad_coefficients()
    assert config["version"] == "galad-v1"


def test_default_coefficients_match_published_galad():
    """Default config must reproduce the published GALAD equation exactly."""
    # GALAD (Johnson 2014): Z = -10.08 + 0.09*age + 1.67*male
    #   + 2.34*log10(AFP) + 0.04*AFP-L3% + 1.33*log10(DCP)
    result = calculate_galad_from_values(
        100, 10, 100, age_years=60, sex="male", patient_id="GALAD-REF"
    )
    expected_z = -10.08 + 0.09 * 60 + 1.67 + 2.34 * 2 + 0.04 * 10 + 1.33 * 2
    assert result.logit == pytest.approx(expected_z, abs=1e-9)
    assert result.score == pytest.approx(1.0 / (1.0 + math.exp(-expected_z)))
    assert "Johnson et al. 2014" in result.intended_use


def test_c_galad_alternative_parameter_set():
    from hcc_multimodal.galad import DEFAULT_COEFFICIENTS_PATH

    c_galad = DEFAULT_COEFFICIENTS_PATH.with_name("galad_coefficients.c_galad.v1.yaml")
    result = calculate_galad_from_values(
        100, 10, 100, age_years=60, sex="male", patient_id="CGALAD-REF",
        coefficients_path=c_galad,
    )
    expected_z = -11.501 + 0.099 * 60 + 0.733 + 0.840 * 2 + 0.073 * 10 + 2.346 * 2
    assert result.coefficient_version == "c-galad-v1"
    assert result.logit == pytest.approx(expected_z, abs=1e-9)


def test_custom_coefficients_are_respected(tmp_path: Path):
    custom = tmp_path / "coeffs.yaml"
    custom.write_text(
        "\n".join(
            [
                "version: galad-test-v9",
                "intercept: 0.0",
                "terms:",
                "  age_years: 0.0",
                "  sex_male: 0.0",
                "  log10_afp_ng_ml: 1.0",
                "  afp_l3_pct: 0.0",
                "  log10_dcp_mau_ml: 0.0",
                "risk_tiers:",
                "  low_below: 0.3",
                "  intermediate_below: 0.7",
            ]
        ),
        encoding="utf-8",
    )
    result = calculate_galad_from_values(
        10, 0, 10, age_years=50, sex="female", patient_id="GALAD-CUSTOM",
        coefficients_path=custom,
    )
    assert result.status == "calculated"
    # logit = log10(10) = 1 -> score = 1 / (1 + e^-1)
    assert result.score == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert result.coefficient_version == "galad-test-v9"


def test_galad_baseline_in_cohort_evaluation():
    """A synthetic HCC/control cohort scored by GALAD feeds evaluation.py."""
    from hcc_multimodal.evaluation import EvaluationCohort, evaluate_cohort

    records = []
    for index in range(40):
        is_case = index % 2 == 0
        result = calculate_galad_from_values(
            400 if is_case else 5,
            15 if is_case else 5,
            1000 if is_case else 20,
            age_years=60 if is_case else 45,
            sex="male" if is_case else "female",
            patient_id=f"GALAD-C{index:03d}",
        )
        assert result.score is not None
        records.append(
            {
                "patient_id": f"GALAD-C{index:03d}",
                "split": "test",
                "center": "demo",
                "baseline_date": "2026-01-01",
                "followup_date": "2026-04-01",
                "endpoint_label": 1 if is_case else 0,
                "galad_score": result.score,
            }
        )
    cohort = EvaluationCohort.model_validate(
        {
            "cohort_id": "GALAD_SYNTHETIC_INTERFACE_CHECK",
            "endpoint": {
                "name": "radiographic_progression_within_followup_window",
                "followup_window_days": 180,
                "positive_label": "synthetic HCC case",
            },
            "inclusion_criteria": ["Synthetic demonstration record"],
            "exclusion_criteria": ["Missing GALAD inputs"],
            "records": records,
        }
    )
    report = evaluate_cohort(cohort, bootstrap_iterations=50, seed=11)
    galad = report["baselines"]["galad"]
    assert galad["n"] == 40
    assert galad["metrics"]["auroc"] == 1.0
    assert galad["metrics"]["sensitivity_at_0_5"] == 1.0
    assert galad["metrics"]["specificity_at_0_5"] == 1.0
    assert galad["performance_claim_permitted"] is False
