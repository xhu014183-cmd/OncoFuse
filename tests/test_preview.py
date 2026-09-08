from pathlib import Path

import numpy as np
import pytest

from hcc_multimodal.preview import (
    _load_canonical_pair,
    _radiological_axial,
    extract_axial_slice_previews,
    extract_lesion_zoom,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERTED = (
    PROJECT_ROOT
    / "public-data"
    / "HCC_003"
    / "public-case"
    / "converted"
)


def test_extract_axial_slice_previews_overlays_mask():
    if not (CONVERTED / "hcc003_ct.nii.gz").exists():
        pytest.skip("converted HCC_003 NIfTI is not present")

    previews = extract_axial_slice_previews(
        CONVERTED / "hcc003_ct.nii.gz",
        CONVERTED / "hcc003_tumor_mask.nii.gz",
        count=16,
        width=360,
    )

    assert len(previews) == 16
    assert all("index" in item and "data_b64" in item for item in previews)
    assert previews[0]["data_b64"].startswith("iVBOR")
    assert all(0 <= int(item["index"]) for item in previews)


def test_extract_lesion_zoom_centers_on_mask():
    if not (CONVERTED / "hcc003_ct.nii.gz").exists():
        pytest.skip("converted HCC_003 NIfTI is not present")

    zoom = extract_lesion_zoom(
        CONVERTED / "hcc003_ct.nii.gz",
        CONVERTED / "hcc003_tumor_mask.nii.gz",
        width=360,
    )

    assert zoom["data_b64"].startswith("iVBOR")
    assert 0 <= int(zoom["slice_index"]) <= 78


def test_radiological_axial_places_right_lobe_lesion_on_left():
    inputs = PROJECT_ROOT / "test-inputs" / "HCC_018"
    if not (inputs / "HCC_018_ct.nii.gz").exists():
        pytest.skip("HCC_018 test inputs are not present")

    _, mask_data = _load_canonical_pair(
        inputs / "HCC_018_ct.nii.gz",
        inputs / "HCC_018_tumor_mask.nii.gz",
    )
    center_z = round(np.argwhere(mask_data)[:, 2].mean())
    view = _radiological_axial(mask_data[:, :, center_z])

    ys, xs = np.where(view)
    assert len(ys) > 0
    # right-lobe lesion must appear on the image-left (radiological view)
    assert xs.mean() < view.shape[1] * 0.45
