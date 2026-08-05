from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from hcc_multimodal.fusion import fuse_cross_sectional_evidence, fuse_evidence
from hcc_multimodal.galad import calculate_galad_from_values
from hcc_multimodal.imaging import compare_imaging, measure_nifti
from hcc_multimodal.labs import load_lab_evidence
from hcc_multimodal.synthetic import generate_synthetic_case


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


def _galad_high():
    return calculate_galad_from_values(
        400, 15, 1000, age_years=60, sex="male", patient_id="DEMO_HCC_001"
    )


def _galad_low():
    return calculate_galad_from_values(
        5, 5, 20, age_years=45, sex="female", patient_id="DEMO_HCC_001"
    )


def test_rule_b_high_galad_with_segmented_mass(tmp_path: Path):
    paths, _, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    verdict = fuse_cross_sectional_evidence(labs, followup, galad=_galad_high())
    assert verdict.state == "high_concordance_hcc_suspect"
    assert "HIGH_CONCORDANCE_HCC_SUSPECT" in verdict.reason_codes
    trace = next(t for t in verdict.rule_traces if t.rule_id == "FUSION.GALAD.HIGH_RISK_IMAGING_MASS")
    assert trace.status == "fired"
    assert any(source.source_type == "galad_style_risk_score" for source in verdict.sources)


def test_rule_a_high_galad_with_occult_imaging(tmp_path: Path):
    paths, _, _ = _case(tmp_path / "case")
    image = nib.load(paths["followup_image"])
    empty_path = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), image.affine), empty_path)
    occult = measure_nifti(
        paths["followup_image"], empty_path,
        patient_id="DEMO_HCC_001", study_date="2026-07-15",
    )
    assert occult.lesion_count == 0
    labs = load_lab_evidence(paths["labs"])
    verdict = fuse_cross_sectional_evidence(labs, occult, galad=_galad_high())
    assert verdict.state == "galad_high_risk_occult_imaging"
    assert "TRIGGER_HIGH_SENSITIVITY_IMAGING" in verdict.reason_codes
    trace = next(t for t in verdict.rule_traces if t.rule_id == "FUSION.GALAD.HIGH_RISK_OCCULT_IMAGING")
    assert trace.status == "fired"


def test_low_galad_falls_back_to_legacy_states(tmp_path: Path):
    paths, _, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    verdict = fuse_cross_sectional_evidence(labs, followup, galad=_galad_low())
    assert verdict.state == "cross_sectional_lesion_marker_signal"
    assert "HIGH_CONCORDANCE_HCC_SUSPECT" not in verdict.reason_codes
    trace = next(t for t in verdict.rule_traces if t.rule_id == "FUSION.GALAD.HIGH_RISK_IMAGING_MASS")
    assert trace.status == "not_fired"


def test_incomplete_galad_is_reported_as_missing(tmp_path: Path):
    paths, _, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    incomplete = calculate_galad_from_values(
        None, 15, 1000, age_years=60, sex="male", patient_id="DEMO_HCC_001"
    )
    verdict = fuse_cross_sectional_evidence(labs, followup, galad=incomplete)
    assert verdict.state == "cross_sectional_lesion_marker_signal"
    assert any("GALAD" in note for note in verdict.missing_evidence)


def test_longitudinal_fusion_records_galad_tier(tmp_path: Path):
    paths, baseline, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    comparison = compare_imaging(baseline, followup)
    verdict = fuse_evidence(labs, comparison, galad=_galad_high())
    assert verdict.state == "concordant_progression_signal"
    assert "GALAD_HIGH_RISK_TIER" in verdict.reason_codes
    trace = next(t for t in verdict.rule_traces if t.rule_id == "FUSION.GALAD.RISK_TIER")
    assert trace.status == "fired"
    assert any("GALAD" in note for note in verdict.supporting_evidence)


def test_fusion_without_galad_is_unchanged(tmp_path: Path):
    paths, baseline, followup = _case(tmp_path)
    labs = load_lab_evidence(paths["labs"])
    comparison = compare_imaging(baseline, followup)
    verdict = fuse_evidence(labs, comparison)
    assert verdict.state == "concordant_progression_signal"
    assert "GALAD_HIGH_RISK_TIER" not in verdict.reason_codes
    assert all(not t.rule_id.startswith("FUSION.GALAD") for t in verdict.rule_traces)
