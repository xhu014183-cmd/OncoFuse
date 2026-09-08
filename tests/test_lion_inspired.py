from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hcc_multimodal.lion_inspired import PrecomputedMaskLionBackend


def _volumes(tmp_path: Path) -> tuple[Path, Path]:
    image = np.zeros((24, 24, 24), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2:7, 2:7, 2:7] = 1
    mask[14:20, 14:20, 14:20] = 1
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    image_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    return image_path, mask_path


def test_two_components_create_lesion_and_patient_evidence(tmp_path: Path):
    image, mask = _volumes(tmp_path)
    evidence = PrecomputedMaskLionBackend().infer(
        image_path=image,
        mask_path=mask,
        mask_role="public_reference",
        patient_id="P1",
        study_date="2026-07-20",
        phase="portal_venous",
    )
    assert [item.lesion_id for item in evidence.lesion_evidence] == [
        "LESION_001",
        "LESION_002",
    ]
    assert evidence.patient_evidence.lesion_count == 2
    assert evidence.patient_evidence.total_tumor_volume_ml == pytest.approx(0.341)
    assert evidence.patient_evidence.diagnostic_capability == "not_available"
    payload = evidence.to_dict()
    assert "malignancy_probability" not in str(payload)
    assert "hcc_probability" not in str(payload)


def test_missing_seg_is_unavailable_not_zero_lesions(tmp_path: Path):
    image, _ = _volumes(tmp_path)
    evidence = PrecomputedMaskLionBackend().infer(
        image_path=image,
        mask_path=None,
        mask_role=None,
        patient_id="P1",
        study_date="2026-07-20",
        phase="unknown",
    )
    assert evidence.status == "unavailable"
    assert evidence.patient_evidence.lesion_count is None
    assert evidence.lesion_evidence == []
    assert "absence" in " ".join(evidence.limitations).casefold()


def test_empty_mask_fails_closed_without_negative_conclusion(tmp_path: Path):
    image, mask = _volumes(tmp_path)
    volume = nib.load(str(mask))
    nib.save(nib.Nifti1Image(np.zeros(volume.shape, dtype=np.uint8), volume.affine), mask)
    evidence = PrecomputedMaskLionBackend().infer(
        image_path=image,
        mask_path=mask,
        mask_role="user_supplied",
        patient_id="P1",
        study_date="2026-07-20",
        phase="unknown",
    )
    assert evidence.status == "unavailable"
    assert evidence.patient_evidence.lesion_count is None


def test_misaligned_mask_is_rejected(tmp_path: Path):
    image, mask = _volumes(tmp_path)
    volume = nib.load(str(mask))
    shifted = volume.affine.copy()
    shifted[0, 3] = 4.0
    nib.save(nib.Nifti1Image(np.asarray(volume.dataobj), shifted), mask)
    with pytest.raises(ValueError, match="affine|align"):
        PrecomputedMaskLionBackend().infer(
            image_path=image,
            mask_path=mask,
            mask_role="user_supplied",
            patient_id="P1",
            study_date="2026-07-20",
            phase="unknown",
        )
