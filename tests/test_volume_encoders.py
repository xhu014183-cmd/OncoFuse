from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hcc_multimodal.synthetic import generate_synthetic_case
from hcc_multimodal.volume_encoders import (
    EncoderUnavailableError,
    VolumeInput,
    available_volume_encoders,
    create_volume_encoder,
    encode_nifti_volume,
    preprocess_m3d_volume,
    preprocess_physical_roi,
)


def test_available_volume_encoders_is_stable_and_lazy():
    before = {name: name in sys.modules for name in ("torch", "transformers", "monai")}
    assert available_volume_encoders() == ("m3d-clip", "statistical-v1")
    assert {name: name in sys.modules for name in before} == before


def test_create_volume_encoder_rejects_unknown_name():
    with pytest.raises(ValueError, match="available encoders"):
        create_volume_encoder("unknown")


def test_statistical_encoder_is_deterministic_and_auditable(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    first, first_path = encode_nifti_volume(
        paths["baseline_image"],
        paths["baseline_mask"],
        patient_id="DEMO_HCC_001",
        study_date="2026-01-15",
        encoder_name="statistical-v1",
        artifact_path=tmp_path / "first.npy",
    )
    second, second_path = encode_nifti_volume(
        paths["baseline_image"],
        paths["baseline_mask"],
        patient_id="DEMO_HCC_001",
        study_date="2026-01-15",
        encoder_name="statistical-v1",
        artifact_path=tmp_path / "second.npy",
    )
    first_vector = np.load(first_path, allow_pickle=False)
    second_vector = np.load(second_path, allow_pickle=False)

    assert first.embedding_dimension == 25
    assert first.embedding_dimension == first_vector.size
    assert first.artifact_sha256 == second.artifact_sha256
    assert np.array_equal(first_vector, second_vector)
    assert np.isfinite(first_vector).all()
    assert first.artifact_file == "first.npy"
    assert str(tmp_path) not in first.to_json()
    assert first.intended_use.startswith("deterministic integration baseline")


def test_statistical_encoder_distinguishes_volumes(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    _, baseline_path = encode_nifti_volume(
        paths["baseline_image"],
        paths["baseline_mask"],
        patient_id="P1",
        study_date="2026-01-15",
        encoder_name="statistical-v1",
        artifact_path=tmp_path / "baseline.npy",
    )
    _, followup_path = encode_nifti_volume(
        paths["followup_image"],
        paths["followup_mask"],
        patient_id="P1",
        study_date="2026-07-15",
        encoder_name="statistical-v1",
        artifact_path=tmp_path / "followup.npy",
    )
    assert not np.array_equal(
        np.load(baseline_path, allow_pickle=False),
        np.load(followup_path, allow_pickle=False),
    )


def test_encoder_rejects_misaligned_mask(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    mask = nib.load(paths["baseline_mask"])
    shifted_affine = mask.affine.copy()
    shifted_affine[0, 3] += 10
    shifted_path = tmp_path / "shifted_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(mask.dataobj), shifted_affine), shifted_path)

    with pytest.raises(ValueError, match="affines do not match"):
        encode_nifti_volume(
            paths["baseline_image"],
            shifted_path,
            patient_id="P1",
            study_date="2026-01-15",
            encoder_name="statistical-v1",
            artifact_path=tmp_path / "embedding.npy",
        )


def test_m3d_preprocessing_has_fixed_contract():
    data = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    volume = VolumeInput(
        image=data,
        mask=None,
        shape=data.shape,
        spacing_mm=(1.0, 1.0, 2.0),
        warnings=(),
    )
    first, metadata, warnings = preprocess_m3d_volume(volume)
    second, _, _ = preprocess_m3d_volume(volume)
    assert first.shape == (1, 32, 256, 256)
    assert first.dtype == np.float32
    assert 0 <= float(first.min()) <= float(first.max()) <= 1
    assert np.array_equal(first, second)
    assert metadata["resize"]["target_shape_cdhw"] == [1, 32, 256, 256]
    assert any("small-lesion detail" in warning for warning in warnings)


def test_physical_preprocessing_preserves_mask_and_records_transform():
    image = np.full((12, 14, 8), -1000, dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[5:7, 6:9, 3:5] = True
    image[mask] = 100
    volume = VolumeInput(
        image=image,
        mask=mask,
        shape=image.shape,
        spacing_mm=(2.0, 1.0, 5.0),
        warnings=(),
        modality="CT",
        phase="portal_venous",
    )
    processed, processed_mask, metadata, _ = preprocess_physical_roi(volume)
    assert processed_mask is not None and processed_mask.any()
    assert processed.shape == processed_mask.shape
    assert 0 <= float(processed.min()) <= float(processed.max()) <= 1
    assert metadata["physical_resampling"]["target_spacing_xyz_mm"] == [1.5, 1.5, 2.5]
    assert metadata["intensity"]["window_hu"] == [-200.0, 300.0]
    assert metadata["roi"]["basis"] == "lesion_mask_with_physical_margin"
    assert metadata["roi"]["cropped_resampled_mask_voxels"] > 0
    assert metadata["phase"] == "portal_venous"


def test_m3d_requires_explicit_remote_code_authorization(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    with pytest.raises(EncoderUnavailableError, match="allow-m3d-remote-code"):
        encode_nifti_volume(
            paths["baseline_image"],
            paths["baseline_mask"],
            patient_id="P1",
            study_date="2026-01-15",
            encoder_name="m3d-clip",
            artifact_path=tmp_path / "embedding.npy",
        )
