from __future__ import annotations

from pathlib import Path

from hcc_multimodal.cli import run_demo
from hcc_multimodal.deepseek import lock_report_fields, validate_report
from hcc_multimodal.fusion import fuse_cross_sectional_evidence, fuse_evidence
from hcc_multimodal.imaging import compare_imaging, measure_nifti
from hcc_multimodal.labs import load_lab_evidence
from hcc_multimodal.preview import create_overlay_montage
from hcc_multimodal.prompting import build_report_prompt
from hcc_multimodal.reporting import render_human_markdown, write_human_markdown
from hcc_multimodal.synthetic import generate_synthetic_case
from hcc_multimodal.tcia import write_composite_labs


def _case(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path)
    baseline = measure_nifti(
        paths["baseline_image"], paths["baseline_mask"],
        patient_id="DEMO_HCC_001", study_date="2026-01-15",
    )
    followup = measure_nifti(
        paths["followup_image"], paths["followup_mask"],
        patient_id="DEMO_HCC_001", study_date="2026-07-15",
    )
    return paths, baseline, followup


def test_measurement_detects_two_followup_lesions(tmp_path: Path):
    _, baseline, followup = _case(tmp_path)
    assert baseline.lesion_count == 1
    assert followup.lesion_count == 2
    assert followup.total_tumor_volume_ml > baseline.total_tumor_volume_ml
    assert followup.quality.image_mask_aligned is True


def test_longitudinal_comparison_finds_new_lesion(tmp_path: Path):
    _, baseline, followup = _case(tmp_path)
    comparison = compare_imaging(baseline, followup)
    assert comparison.new_lesion_signal is True
    assert comparison.category == "imaging_progression_signal"
    assert "NEW_LESION_GEOMETRIC_SIGNAL" in comparison.reason_codes


def test_afp_dcp_fusion_is_concordant(tmp_path: Path):
    paths, baseline, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    assert labs.markers["AFP"].direction == "rising"
    assert labs.markers["DCP"].latest_above_upper is True
    verdict = fuse_evidence(labs, compare_imaging(baseline, followup))
    assert verdict.state == "concordant_progression_signal"
    assert verdict.modality_concordance == "high"
    assert verdict.requires_clinician_review is True


def test_end_to_end_writes_evidence_files(tmp_path: Path):
    verdict_path = run_demo(tmp_path / "output")
    assert verdict_path.exists()
    assert (tmp_path / "output" / "baseline_imaging_evidence.json").exists()
    assert (tmp_path / "output" / "llm_prompt.json").exists()
    assert (tmp_path / "output" / "llm_prompt.txt").exists()
    assert (tmp_path / "output" / "multimodal_case_evidence.json").exists()
    assert (tmp_path / "output" / "baseline_image_embedding.npy").exists()
    assert (tmp_path / "output" / "followup_image_embedding_evidence.json").exists()
    assert "concordant_progression_signal" in verdict_path.read_text(encoding="utf-8")
    prompt = (tmp_path / "output" / "llm_prompt.txt").read_text(encoding="utf-8")
    for forbidden in (
        "voxel_count",
        "centroid_world_mm",
        "mean_image_intensity",
        "image_embedding",
        "artifact_sha256",
        "image_file",
    ):
        assert forbidden not in prompt


def test_iwiki_lab_shape_is_supported(tmp_path: Path):
    import json

    payload = {
        "patientInfo": {"patientId": "IWIKI_HCC_001"},
        "testResultInfos": [
            {"inspectionDate": "2026-01-01", "systemTestItemName": "AFP", "result": "5.0", "unit": "ng/mL", "referenceRange": "0-7"},
            {"inspectionDate": "2026-04-01", "systemTestItemName": "AFP", "result": "50.0", "unit": "ng/mL", "referenceRange": "0-7"},
            {"inspectionDate": "2026-01-01", "systemTestItemName": "DCP", "result": "20", "unit": "mAU/mL", "referenceRange": "0-40"},
            {"inspectionDate": "2026-04-01", "systemTestItemName": "PIVKA-II", "result": "60", "unit": "mAU/mL", "referenceRange": "0-40"}
        ]
    }
    path = tmp_path / "iwiki_labs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    evidence = load_lab_evidence(path)
    assert evidence.patient_id == "IWIKI_HCC_001"
    assert evidence.markers["AFP"].direction == "rising"
    assert evidence.markers["DCP"].latest_value == 60


def test_cross_sectional_fusion_is_explicitly_illustrative(tmp_path: Path):
    paths, _, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    verdict = fuse_cross_sectional_evidence(labs, followup)
    assert verdict.state == "cross_sectional_lesion_marker_signal"
    assert verdict.modality_concordance == "illustrative_only"
    assert "unrelated synthetic labs" in verdict.intended_use


def test_overlay_preview_is_generated(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path)
    preview = create_overlay_montage(
        paths["followup_image"],
        paths["followup_mask"],
        tmp_path / "overlay.png",
    )
    assert preview.exists()
    assert preview.stat().st_size > 1000


def test_public_lab_scenarios_change_fusion_result(tmp_path: Path):
    _, _, imaging = _case(tmp_path / "images")
    states = {}
    for scenario_id in (
        "dual_marker_rising",
        "afp_negative_dcp_rising",
        "markers_normal",
    ):
        path = write_composite_labs(
            tmp_path / f"{scenario_id}.json",
            imaging.patient_id,
            scenario_id,
        )
        labs = load_lab_evidence(path)
        states[scenario_id] = fuse_cross_sectional_evidence(labs, imaging).state
    assert states["dual_marker_rising"] == "cross_sectional_lesion_marker_signal"
    assert states["afp_negative_dcp_rising"] == "cross_sectional_lesion_marker_signal"
    assert states["markers_normal"] == "segmented_lesion_without_marker_signal"


def test_prompt_preserves_facts_and_safety_boundaries(tmp_path: Path):
    paths, _, imaging = _case(tmp_path / "prompt")
    labs = load_lab_evidence(paths["labs"])
    verdict = fuse_cross_sectional_evidence(labs, imaging)
    bundle = build_report_prompt(
        imaging_evidence=imaging.to_dict(),
        lab_evidence=labs,
        verdict=verdict,
        scenario_id="test_scenario",
        report_type="cross_sectional_test",
        data_relationship="Real public image plus declared synthetic PoC labs",
    )
    prompt = bundle["user_message"]
    assert "85.3" in prompt
    assert "68.0" in prompt
    assert verdict.state in prompt
    assert "Do not infer, diagnose, stage" in bundle["system_message"]
    assert "Return exactly one JSON object" in bundle["system_message"]
    assert bundle["schema_version"] == "1.0.0"
    assert bundle["evidence_payload"]["fusion_verdict"]["state"] == verdict.state
    assert bundle["required_quality_items"]
    assert "embedding" not in prompt.lower()
    assert "image_file" not in prompt


def test_deepseek_report_contract_validation():
    verdict = {
        "patient_id": "P1",
        "state": "cross_sectional_lesion_marker_signal",
        "modality_concordance": "illustrative_only",
        "requires_clinician_review": True,
        "quality_status": "warning",
        "intended_use": "research demonstration",
        "supporting_evidence": [],
        "conflicting_evidence": [],
        "missing_evidence": [],
    }
    report = {
        "case_id": "P1",
        "scenario_id": "dual_marker_rising",
        "report_type": "cross_sectional_public_imaging_poc",
        "data_scope": "real image plus synthetic labs",
        "imaging_summary": [],
        "laboratory_summary": [],
        "multimodal_assessment": {
            "state": "cross_sectional_lesion_marker_signal",
            "concordance": "illustrative_only",
            "summary": "PoC only",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "missing_evidence": [],
        },
        "uncertainty": [],
        "data_quality_status": "warning",
        "quality_and_limits": [],
        "review_required": True,
        "intended_use": "research demonstration",
        "disclaimer": "Research use only; not for diagnosis, staging, prognosis, or treatment decisions.",
    }
    validation = validate_report(report, verdict, "dual_marker_rising")
    assert validation["valid"] is True
    assert validation["errors"] == []


def test_locked_fields_mismatch_is_blocked():
    verdict = {
        "patient_id": "P1",
        "state": "state_locked",
        "modality_concordance": "illustrative_only",
        "requires_clinician_review": True,
        "quality_status": "warning",
        "intended_use": "exact English boundary",
    }
    report = {
        "case_id": "P1",
        "scenario_id": "s1",
        "multimodal_assessment": {
            "state": "state_locked",
            "concordance": "illustrative_only",
        },
        "review_required": True,
        "intended_use": "翻译后的边界",
    }
    import pytest

    with pytest.raises(ValueError, match="Locked report fields do not match"):
        lock_report_fields(report, verdict, "s1")


def test_human_markdown_hides_machine_codes_in_main_result():
    report = {
        "case_id": "P1",
        "scenario_id": "dual_marker_rising",
        "report_type": "poc",
        "data_scope": "真实公开影像与合成检验场景。",
        "imaging_summary": ["专家分割 mask 标注1个 Mass 区域。"],
        "laboratory_summary": ["AFP升高。", "DCP升高。"],
        "multimodal_assessment": {
            "state": "cross_sectional_lesion_marker_signal",
            "concordance": "illustrative_only",
            "summary": "两类证据在本PoC中共同形成信号。",
            "supporting_evidence": ["影像与标志物均提供支持。"],
            "conflicting_evidence": [],
            "missing_evidence": ["缺少真实配对检验。"],
        },
        "uncertainty": ["缺少真实配对检验。"],
        "data_quality_status": "warning",
        "quality_and_limits": ["不用于诊断。"],
        "review_required": True,
        "intended_use": "research demonstration",
        "disclaimer": "Research use only; not for diagnosis, staging, prognosis, or treatment decisions.",
    }
    markdown = render_human_markdown(report)
    assert markdown.startswith("# HCC 多模态研究证据摘要")
    assert "**横断面影像标注与合成标志物共同形成信号。**" in markdown
    assert "未记录显式冲突" in markdown
    assert "<summary>机器审计字段</summary>" in markdown
    assert "真实公开影像与合成检验场景。" in markdown


def test_human_markdown_writer_persists_utf8_report(tmp_path: Path):
    report = {
        "case_id": "P1",
        "scenario_id": "s1",
        "report_type": "research",
        "data_scope": "Synthetic evidence.",
        "imaging_summary": [],
        "laboratory_summary": [],
        "multimodal_assessment": {
            "state": "insufficient_evidence",
            "concordance": "insufficient",
            "summary": "Evidence is insufficient.",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "missing_evidence": [],
        },
        "uncertainty": [],
        "data_quality_status": "warning",
        "quality_and_limits": [],
        "review_required": True,
        "intended_use": "research",
        "disclaimer": "Research use only; not for diagnosis, staging, prognosis, or treatment decisions.",
    }
    target = write_human_markdown(report, tmp_path / "report.md")
    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith("# HCC 多模态研究证据摘要")
