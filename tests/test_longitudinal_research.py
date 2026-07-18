from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from hcc_multimodal.fusion import fuse_evidence
from hcc_multimodal.imaging import compare_imaging, measure_nifti
from hcc_multimodal.labs import load_lab_evidence
from hcc_multimodal.synthetic import generate_synthetic_case


def _evidence(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path)
    baseline = measure_nifti(
        paths["baseline_image"],
        paths["baseline_mask"],
        patient_id="DEMO_HCC_001",
        study_date="2026-01-15",
        phase="synthetic_single_phase",
    )
    followup = measure_nifti(
        paths["followup_image"],
        paths["followup_mask"],
        patient_id="DEMO_HCC_001",
        study_date="2026-07-15",
        phase="synthetic_single_phase",
    )
    return paths, baseline, followup


def test_hungarian_output_explicitly_accounts_for_every_lesion(tmp_path: Path):
    _, baseline, followup = _evidence(tmp_path)
    comparison = compare_imaging(
        baseline,
        followup,
        registration_status="verified",
    )
    statuses = [item.status for item in comparison.matched_lesions]
    assert statuses.count("matched") == 1
    assert statuses.count("new") == 1
    assert comparison.method == "hungarian-centroid-bbox-volume"
    assert comparison.threshold_version == "longitudinal-matching-v1"
    assert all(item.total_cost is not None for item in comparison.matched_lesions if item.status == "matched")


def test_split_and_merge_candidates_are_not_mislabeled_as_new(tmp_path: Path):
    shape = (30, 30, 20)
    image = np.zeros(shape, dtype=np.float32)
    baseline_mask = np.zeros(shape, dtype=np.uint8)
    followup_mask = np.zeros(shape, dtype=np.uint8)
    baseline_mask[10:14, 10:14, 8:12] = 1
    followup_mask[7:10, 10:13, 8:11] = 1
    followup_mask[15:18, 10:13, 8:11] = 1
    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    baseline_path = tmp_path / "baseline.nii.gz"
    followup_path = tmp_path / "followup.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(baseline_mask, affine), baseline_path)
    nib.save(nib.Nifti1Image(followup_mask, affine), followup_path)
    baseline = measure_nifti(
        image_path, baseline_path, patient_id="P1", study_date="2026-01-01",
    )
    followup = measure_nifti(
        image_path, followup_path, patient_id="P1", study_date="2026-02-01",
    )
    split = compare_imaging(baseline, followup, registration_status="verified")
    assert "split_candidate" in {item.status for item in split.matched_lesions}
    assert split.new_lesion_signal is False

    merged = compare_imaging(
        followup,
        baseline.model_copy(update={"study_date": "2026-03-01"}),
        registration_status="verified",
    )
    assert "merge_candidate" in {item.status for item in merged.matched_lesions}


def test_failed_registration_blocks_quantitative_analysis(tmp_path: Path):
    _, baseline, followup = _evidence(tmp_path)
    comparison = compare_imaging(baseline, followup, registration_status="failed")
    assert comparison.quality.status == "fail"
    assert comparison.category == "imaging_unavailable"
    assert comparison.new_lesion_signal is None
    assert comparison.volume_change_pct is None
    assert comparison.matched_lesions == []


def test_known_phase_mismatch_blocks_direct_matching(tmp_path: Path):
    _, baseline, followup = _evidence(tmp_path)
    mismatched = followup.model_copy(update={"phase": "arterial"})
    comparison = compare_imaging(baseline, mismatched, registration_status="verified")
    assert comparison.quality.status == "fail"
    assert comparison.category == "imaging_unavailable"
    assert any(check.check_id == "PHASE_COMPATIBILITY" and check.status == "fail" for check in comparison.quality.checks)


def test_empty_mask_is_unavailable_and_cannot_enter_longitudinal_analysis(tmp_path: Path):
    paths, baseline, _ = _evidence(tmp_path / "case")
    image = nib.load(paths["followup_image"])
    empty_path = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), image.affine), empty_path)
    empty = measure_nifti(
        paths["followup_image"],
        empty_path,
        patient_id="DEMO_HCC_001",
        study_date="2026-07-15",
    )
    assert empty.quality.status == "unavailable"
    comparison = compare_imaging(baseline, empty, registration_status="verified")
    assert comparison.quality.status == "fail"
    assert comparison.category == "imaging_unavailable"


def test_fusion_uses_insufficient_not_moderate_default(tmp_path: Path):
    paths, baseline, _ = _evidence(tmp_path)
    comparison = compare_imaging(
        baseline,
        baseline.model_copy(update={"study_date": "2026-07-15"}),
        registration_status="verified",
    )
    labs = load_lab_evidence(paths["labs"], index_time="2026-07-15")
    stable_labs = labs.model_copy(
        update={
            "markers": {
                name: marker.model_copy(update={"direction": "stable", "latest_above_upper": False})
                for name, marker in labs.markers.items()
            }
        }
    )
    verdict = fuse_evidence(stable_labs, comparison)
    assert verdict.state == "insufficient_evidence"
    assert verdict.modality_concordance == "insufficient"
    assert "INSUFFICIENT_CONCORDANT_EVIDENCE" in verdict.reason_codes


def test_intervening_treatment_blocks_concordance(tmp_path: Path):
    paths, baseline, followup = _evidence(tmp_path)
    comparison = compare_imaging(baseline, followup, registration_status="verified")
    labs = load_lab_evidence(paths["labs"], index_time="2026-07-15")
    verdict = fuse_evidence(
        labs,
        comparison,
        treatment_events=[{"date": "2026-04-01", "type": "research_context_event"}],
    )
    assert verdict.state == "indeterminate_after_intervening_treatment"
    assert verdict.modality_concordance == "insufficient"
    assert "INTERVENING_TREATMENT" in verdict.reason_codes
