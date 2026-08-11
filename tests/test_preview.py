from pathlib import Path

import pytest

from hcc_multimodal.preview import extract_axial_slice_previews, extract_lesion_zoom

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
