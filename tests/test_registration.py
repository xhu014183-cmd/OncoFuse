from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from hcc_multimodal.imaging import compare_imaging, measure_nifti
from hcc_multimodal.registration import register_volumes, simpleitk_available
from hcc_multimodal.synthetic import _sphere

SHAPE = (64, 64, 32)
AFFINE = np.diag([1.5, 1.5, 2.5, 1.0])
PHYSICAL_SHIFT = np.array([6.0, -4.5, 5.0])

sitk_required = pytest.mark.skipif(not simpleitk_available(), reason="SimpleITK not installed")


def _write_shifted_case(tmp_path) -> dict[str, object]:
    """One synthetic study duplicated with a known physical affine shift."""
    rng = np.random.default_rng(7)
    image = rng.normal(50.0, 6.0, size=SHAPE).astype(np.float32)
    mask = _sphere(SHAPE, (28, 32, 16), 5)
    image[mask] += 30.0
    shifted = AFFINE.copy()
    shifted[:3, 3] = PHYSICAL_SHIFT
    payloads = {
        "baseline_image": (image, AFFINE),
        "baseline_mask": (mask.astype(np.uint8), AFFINE),
        "followup_image": (image, shifted),
        "followup_mask": (mask.astype(np.uint8), shifted),
    }
    paths: dict[str, object] = {}
    for name, (data, affine) in payloads.items():
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[name] = path
    return paths


@sitk_required
def test_register_volumes_recovers_physical_shift(tmp_path) -> None:
    paths = _write_shifted_case(tmp_path)
    outcome = register_volumes(
        paths["baseline_image"],
        paths["baseline_mask"],
        paths["followup_image"],
        paths["followup_mask"],
        tmp_path / "registration",
    )
    assert outcome.evidence.status == "verified"
    assert outcome.image_path is not None and outcome.image_path.exists()
    assert outcome.mask_path is not None and outcome.mask_path.exists()
    assert outcome.evidence.centroid_residual_mm is not None
    assert outcome.evidence.centroid_residual_mm < 2.0
    # SimpleITK reports the fixed->moving translation in its internal LPS frame
    # (NIfTI RAS x/y axes are negated), so +[6, -4.5, 5] RAS becomes [-6, 4.5, 5].
    expected_lps = [-PHYSICAL_SHIFT[0], -PHYSICAL_SHIFT[1], PHYSICAL_SHIFT[2]]
    assert outcome.evidence.translation_mm == pytest.approx(expected_lps, abs=2.0)
    assert outcome.evidence.coordinate_frame == "itk_lps"
    assert outcome.evidence.transform_matrix is not None
    assert outcome.evidence.metric_value is not None


@sitk_required
def test_registered_followup_feeds_compare_imaging(tmp_path) -> None:
    paths = _write_shifted_case(tmp_path)
    outcome = register_volumes(
        paths["baseline_image"],
        paths["baseline_mask"],
        paths["followup_image"],
        paths["followup_mask"],
        tmp_path / "registration",
    )
    assert outcome.image_path is not None and outcome.mask_path is not None
    baseline = measure_nifti(
        paths["baseline_image"], paths["baseline_mask"],
        patient_id="P1", study_date="2026-01-01",
    )
    followup = measure_nifti(
        outcome.image_path, outcome.mask_path,
        patient_id="P1", study_date="2026-02-01",
    )
    longitudinal = compare_imaging(
        baseline,
        followup,
        registration_status="verified",
        registration=outcome.evidence,
    )
    assert longitudinal.registration_status == "verified"
    assert longitudinal.registration is not None
    assert longitudinal.registration.status == "verified"
    matched = [match for match in longitudinal.matched_lesions if match.status == "matched"]
    assert matched, "expected the resampled lesion to match the baseline lesion"
    assert matched[0].centroid_distance_mm is not None
    assert matched[0].centroid_distance_mm < 2.0


@sitk_required
def test_register_volumes_fails_closed_on_missing_input(tmp_path) -> None:
    paths = _write_shifted_case(tmp_path)
    outcome = register_volumes(
        tmp_path / "missing_image.nii.gz",
        paths["baseline_mask"],
        paths["followup_image"],
        paths["followup_mask"],
        tmp_path / "registration",
    )
    assert outcome.evidence.status == "failed"
    assert outcome.image_path is None
    assert outcome.mask_path is None
    assert outcome.evidence.warnings


def test_register_volumes_unavailable_without_simpleitk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("hcc_multimodal.registration.simpleitk_available", lambda: False)
    paths = _write_shifted_case(tmp_path)
    outcome = register_volumes(
        paths["baseline_image"],
        paths["baseline_mask"],
        paths["followup_image"],
        paths["followup_mask"],
        tmp_path / "registration",
    )
    assert outcome.evidence.status == "unavailable"
    assert outcome.image_path is None
    assert outcome.mask_path is None
    assert outcome.evidence.warnings
