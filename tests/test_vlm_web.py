from pathlib import Path

from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.vlm_web import (
    _clinical_report,
    _deterministic_impression,
    _guideline_hint,
    _joint_interpretation,
    _lab_series,
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
