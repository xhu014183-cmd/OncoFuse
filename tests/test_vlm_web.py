import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.prognosis_models import CoxModelBundle
from hcc_multimodal.vlm_web import (
    _clinical_report,
    _deterministic_impression,
    _galad_result,
    _guideline_hint,
    _Handler,
    _joint_interpretation,
    _lab_series,
    _longitudinal_imaging,
    _risk_tier,
)


def test_impression_elevated_and_rising_markers():
    metadata = (
        "Imaging measurements (deterministic): lesion_count=1; "
        "total_volume_ml=12.808; max_extent_mm=34.141; quality=pass"
    )
    labs = [
        {"marker": "AFP", "latest_above_reference": True, "trajectory": "persistent_rising"},
        {"marker": "DCP", "latest_above_reference": True, "trajectory": "persistent_rising"},
    ]

    text = _deterministic_impression(metadata, labs)

    assert "AFP、DCP" in text
    assert "进展信号一致" in text


def test_impression_no_lesion():
    text = _deterministic_impression("lesion_count=0; quality=pass", [])
    assert "未检出" in text


def test_impression_normal_markers_with_lesion():
    labs = [
        {"marker": "AFP", "latest_above_reference": False, "trajectory": "stable_normal"}
    ]
    text = _deterministic_impression("lesion_count=1; quality=pass", labs)
    assert "未见检验-影像一致信号" in text


def test_guideline_hint_high_risk_combination():
    metadata = "lesion_count=1; max_extent_mm=34.141"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 85.3,
            "latest_unit": "ng/mL",
            "latest_above_reference": True,
            "trajectory": "persistent_rising",
        },
        {
            "marker": "DCP",
            "latest_value": 68.0,
            "latest_unit": "mAU/mL",
            "latest_above_reference": True,
            "trajectory": "persistent_rising",
        },
    ]

    text = _guideline_hint(metadata, labs)

    assert "高风险特征组合" in text


def test_guideline_hint_low_risk_small_lesion():
    metadata = "lesion_count=1; max_extent_mm=12.0"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 6.0,
            "latest_unit": "ng/mL",
            "latest_above_reference": False,
            "trajectory": "stable_normal",
        }
    ]

    text = _guideline_hint(metadata, labs)

    assert "倾向性提示较低" in text


def test_guideline_hint_no_lesion():
    assert "无法应用" in _guideline_hint("lesion_count=0", [])


def test_joint_interpretation_contains_both_modalities():
    metadata = "lesion_count=1; max_extent_mm=34.141"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 85.3,
            "latest_unit": "ng/mL",
            "latest_above_reference": True,
            "trajectory": "persistent_rising",
        }
    ]

    text = _joint_interpretation(
        metadata, labs, "impression-text", "guideline-text"
    )

    assert "影像侧" in text
    assert "检验侧" in text
    assert "impression-text" in text
    assert "guideline-text" in text


def test_lab_series_builds_points_and_reference(tmp_path: Path):
    path = tmp_path / "labs.txt"
    path.write_text(
        "1998-09-30 AFP 6 ng/mL 0-7\n1998-12-29 AFP 85.3 ng/mL 0-7\n",
        encoding="utf-8",
    )
    labs = parse_laboratory_report(path, patient_id="CASE")

    series = _lab_series(labs)

    assert "AFP" in series
    assert len(series["AFP"]["points"]) == 2
    assert series["AFP"]["points"][-1]["value"] == 85.3
    assert series["AFP"]["upper_reference"] == 7.0


def test_risk_tier_high_for_large_lesion_and_elevated_dcp():
    metadata = "lesion_count=1; max_extent_mm=34.141"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 85.3,
            "latest_above_reference": True,
            "trajectory": "persistent_rising",
        },
        {
            "marker": "DCP",
            "latest_value": 68.0,
            "latest_above_reference": True,
            "trajectory": "persistent_rising",
        },
    ]

    assert _risk_tier(metadata, labs)["level"] == "high"


def test_risk_tier_medium_for_large_lesion_only():
    metadata = "lesion_count=1; max_extent_mm=34.141"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 6.0,
            "latest_above_reference": False,
            "trajectory": "stable_normal",
        }
    ]

    assert _risk_tier(metadata, labs)["level"] == "medium"


def test_risk_tier_low_for_small_lesion_and_normal_markers():
    metadata = "lesion_count=1; max_extent_mm=12.0"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 6.0,
            "latest_above_reference": False,
            "trajectory": "stable_normal",
        }
    ]

    assert _risk_tier(metadata, labs)["level"] == "low"


def test_clinical_report_concordant_progression():
    metadata = "lesion_count=1; total_volume_ml=12.808; max_extent_mm=34.141; quality=pass"
    lab_summary = [
        {"marker": "AFP", "latest_value": 85.3, "latest_unit": "ng/mL", "latest_above_reference": True, "trajectory": "persistent_rising"},
        {"marker": "DCP", "latest_value": 68.0, "latest_unit": "mAU/mL", "latest_above_reference": True, "trajectory": "persistent_rising"},
    ]
    lab_series = {
        "AFP": {"points": [], "upper_reference": 7.0},
        "DCP": {"points": [], "upper_reference": 40.0},
    }

    report = _clinical_report(
        metadata, lab_summary, lab_series, ["well-defined margins"], "high"
    )

    assert "进展信号一致" in report["assessment"]
    assert "参考 7.0 以下" in report["laboratory"]
    assert "多期增强 MRI" in report["suggestion"]
    assert "边界清楚" in report["imaging"]
    assert "规则引擎" not in str(report)


def test_clinical_report_discordant():
    metadata = "lesion_count=1; max_extent_mm=12.0"
    lab_summary = [
        {
            "marker": "AFP",
            "latest_value": 6.0,
            "latest_unit": "ng/mL",
            "latest_above_reference": False,
            "trajectory": "stable_normal",
        }
    ]
    lab_series = {"AFP": {"points": [], "upper_reference": 7.0}}

    report = _clinical_report(metadata, lab_summary, lab_series, [], "low")

    assert "不一致" in report["assessment"]


def test_clinical_report_no_lesion():
    report = _clinical_report("lesion_count=0", [], {}, [], "low")
    assert "未检出" in report["assessment"]


def test_clinical_report_longitudinal_progression():
    metadata = "lesion_count=2; total_volume_ml=1.632; max_extent_mm=22.5"
    labs = [
        {
            "marker": "AFP",
            "latest_value": 85.3,
            "latest_unit": "ng/mL",
            "latest_above_reference": True,
            "trajectory": "persistent_rising",
        }
    ]
    longitudinal = {
        "baseline": {"volume": 0.692, "lesion_count": 1},
        "followup": {"volume": 1.632, "lesion_count": 2},
        "compare": {"volume_change_pct": 135.8, "new_lesion_signal": True},
    }

    report = _clinical_report(
        metadata,
        labs,
        {"AFP": {"points": [], "upper_reference": 7.0}},
        [],
        "high",
        longitudinal,
    )

    assert "病灶体积由 0.69 mL 变为 1.63 mL（+135.8%）" in report["longitudinal"]
    assert "可见新发病灶" in report["longitudinal"]
    assert "进展信号一致" in report["longitudinal"]


def test_longitudinal_imaging_synthetic_pair():
    inputs = Path(__file__).resolve().parent.parent / "demo-output" / "input"
    if not (inputs / "baseline_ct.nii.gz").exists():
        pytest.skip("synthetic longitudinal inputs are not present")

    result = _longitudinal_imaging(
        inputs / "baseline_ct.nii.gz",
        inputs / "baseline_tumor_mask.nii.gz",
        inputs / "followup_ct.nii.gz",
        inputs / "followup_tumor_mask.nii.gz",
        baseline_date="2026-01-15",
        followup_date="2026-07-15",
        patient_id="DEMO",
    )

    compare = result["compare"]
    assert compare["volume_change_pct"] is not None
    assert compare["volume_change_pct"] > 0
    assert compare["new_lesion_signal"] is not None


def test_galad_calculated_with_full_inputs(tmp_path: Path):
    path = tmp_path / "labs.txt"
    path.write_text(
        "2026-01-15 AFP 6 ng/mL 0-7\n"
        "2026-07-20 AFP 96 ng/mL 0-7\n"
        "2026-07-20 AFP-L3% 12 % 0-10\n"
        "2026-01-15 DCP 20 mAU/mL 0-40\n"
        "2026-07-20 DCP 80 mAU/mL 0-40\n",
        encoding="utf-8",
    )
    labs = parse_laboratory_report(path, patient_id="CASE")

    result = _galad_result(
        labs, age_years=60.0, sex="male", patient_id="CASE"
    )

    assert result["status"] == "calculated"
    assert result["score"] is not None
    assert result["risk_tier"] in {"LOW", "INTERMEDIATE", "HIGH"}


def test_galad_incomplete_without_age(tmp_path: Path):
    path = tmp_path / "labs.txt"
    path.write_text(
        "2026-07-20 AFP 96 ng/mL 0-7\n"
        "2026-07-20 AFP-L3% 12 % 0-10\n"
        "2026-07-20 DCP 80 mAU/mL 0-40\n",
        encoding="utf-8",
    )
    labs = parse_laboratory_report(path, patient_id="CASE")

    result = _galad_result(labs, age_years=None, sex="male", patient_id="CASE")

    assert result["status"] == "incomplete"
    assert "age_years" in result["missing_inputs"]


LAB_TEXT = """2026-01-15 AFP 6.0 ng/mL 0-7
2026-04-15 AFP 18.0 ng/mL 0-7
2026-07-15 AFP 85.3 ng/mL 0-7
2026-01-15 DCP 25.0 mAU/mL 0-40
2026-07-15 DCP 68.0 mAU/mL 0-40
"""


def _b64(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode("ascii")


def _handler() -> "_Handler":
    return _Handler.__new__(_Handler)


def _web_prognosis_model() -> CoxModelBundle:
    names = [
        "age_years",
        "female",
        "afp_ng_ml",
        "lesion_count",
        "total_tumor_volume_ml",
        "max_lesion_extent_mm",
        "largest_lesion_sphericity",
    ]
    return CoxModelBundle(
        model_id="web-test-fused-v1",
        model_name="fused_core",
        feature_names=names,
        transformations={
            name: "log1p"
            if name
            in {
                "afp_ng_ml",
                "lesion_count",
                "total_tumor_volume_ml",
                "max_lesion_extent_mm",
            }
            else "identity"
            for name in names
        },
        means=[60, 0.5, 4, 1, 2, 3, 0.7],
        scales=[10, 0.5, 2, 1, 2, 1, 0.1],
        coefficients=[0.1] * 7,
        penalizer=0.1,
        baseline_event_times_days=[365],
        baseline_cumulative_hazard=[0.2],
        development_reference_risks=[-1, 0, 1],
        median_risk_threshold=0,
        training_n=233,
        training_event_n=168,
        cross_validation_seed=1729,
        model_hash="c" * 64,
        external_validation_status="evaluated",
        limitations=["research only"],
    )


def _disable_llm(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("External LLM configuration is unavailable")

    monkeypatch.setattr("hcc_multimodal.vlm_llm.call_vlm_llm", unavailable)


def test_interpret_single_timepoint(tmp_path: Path, monkeypatch):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "case")
    _disable_llm(monkeypatch)
    request = {
        "files": {
            "ct": {"name": "ct.nii.gz", "data_b64": _b64(paths["followup_image"])},
            "mask": {"name": "mask.nii.gz", "data_b64": _b64(paths["followup_mask"])},
            "labs": {"name": "labs.txt", "content": LAB_TEXT},
        },
        "options": {
            "fusion_modes": ["auditable"],
            "patient_label": "CASE",
            "study_date": "2026-07-15",
            "phase": "portal_venous",
        },
    }
    result = _handler()._interpret(request)
    assert result["ok"] is True
    interp = result["interpretation"]
    assert interp["study_date"] == "2026-07-15"
    assert interp["risk_tier"]["level"] in {"low", "medium", "high"}
    assert interp["slices"]
    assert result["web_demo"]["arms"]["auditable"]["audit_status"] == "pass"


def test_prognosis_endpoint_uses_separate_contract(tmp_path: Path):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "case")
    request = {
        "files": {
            "ct": {"data_b64": _b64(paths["followup_image"])},
            "mask": {"data_b64": _b64(paths["followup_mask"])},
            "model": {"content": _web_prognosis_model().to_json()},
        },
        "options": {
            "age_years": 60,
            "sex": "male",
            "afp_ng_ml": 80,
            "phase": "portal_venous",
            "study_date": "2026-07-15",
        },
    }
    result = _handler()._prognosis(request)
    assert result["ok"] is True
    assert result["report"]["report_type"] == "hcc_tace_os_research_prognosis"
    assert result["prognostic_evidence"]["external_validation_status"] == "evaluated"
    assert result["artifact_index"]["controlled-prognosis-report.json"]["available"] is True


def test_interpret_longitudinal_mode(tmp_path: Path, monkeypatch):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "case")
    _disable_llm(monkeypatch)
    request = {
        "files": {
            "ct": {"name": "ct.nii.gz", "data_b64": _b64(paths["baseline_image"])},
            "mask": {"name": "mask.nii.gz", "data_b64": _b64(paths["baseline_mask"])},
            "ct2": {"name": "ct2.nii.gz", "data_b64": _b64(paths["followup_image"])},
            "mask2": {"name": "mask2.nii.gz", "data_b64": _b64(paths["followup_mask"])},
            "labs": {"name": "labs.txt", "content": LAB_TEXT},
        },
        "options": {
            "fusion_modes": ["auditable"],
            "patient_label": "CASE",
            "study_date": "2026-01-15",
            "study_date2": "2026-07-15",
        },
    }
    result = _handler()._interpret(request)
    assert result["ok"] is True
    interp = result["interpretation"]
    assert interp["longitudinal"] is not None
    assert interp["longitudinal"]["compare"]["volume_change_pct"] is not None
    assert interp["slices_baseline"]


def test_interpret_missing_files_raises():
    with pytest.raises(ValueError, match="required"):
        _handler()._interpret({"files": {}, "options": {}})


def test_interpret_requires_auditable_arm(tmp_path: Path, monkeypatch):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "case")
    _disable_llm(monkeypatch)
    request = {
        "files": {
            "ct": {"name": "ct.nii.gz", "data_b64": _b64(paths["followup_image"])},
            "mask": {"name": "mask.nii.gz", "data_b64": _b64(paths["followup_mask"])},
            "labs": {"name": "labs.txt", "content": LAB_TEXT},
        },
        "options": {
            "fusion_modes": ["open"],
            "study_date": "2026-07-15",
        },
    }
    with pytest.raises(ValueError, match="auditable"):
        _handler()._interpret(request)


def test_interpret_allows_missing_mask_without_zero_lesion_claim(tmp_path: Path):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "case")
    request = {
        "files": {
            "ct": {"name": "ct.nii.gz", "data_b64": _b64(paths["followup_image"])},
            "labs": {"name": "labs.txt", "content": LAB_TEXT},
        },
        "options": {
            "fusion_modes": ["auditable"],
            "patient_label": "CASE",
            "study_date": "2026-07-15",
        },
    }
    result = _handler()._interpret(request)
    assert result["ok"] is True
    lion = result["pipeline"]["lion_inspired"]
    assert lion["patient_evidence"]["lesion_count"] is None
    assert "不得将其解释为影像无病灶" in result["interpretation"][
        "clinical_impression"
    ]


def test_health_exposes_model_status_without_secrets(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "secret-zhipu")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek")
    handler = _handler()
    handler.path = "/api/health"
    handler.server = SimpleNamespace(generated_at="2026-08-24T00:00:00Z")
    captured = {}

    def capture(payload, *, status):
        captured["payload"] = payload
        captured["status"] = status

    handler._send_json = capture
    handler.do_GET()
    serialized = json.dumps(captured)
    assert captured["status"] == 200
    assert captured["payload"]["models"]["zhipu"]["configured"] is True
    assert "secret-zhipu" not in serialized
    assert "secret-deepseek" not in serialized
