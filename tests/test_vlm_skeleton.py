from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from hcc_multimodal.vlm_skeleton import (
    extract_patches,
    run_mock_vlm,
    run_skeleton_demo,
)


def test_extract_2d_patches_writes_manifest_preview_and_array(tmp_path: Path):
    image_path = tmp_path / "slice.png"
    Image.fromarray(np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)).save(image_path)

    manifest = extract_patches(image_path, tmp_path / "out", patch_size=(32, 32))

    assert manifest.mode == "image_2d"
    assert manifest.patch_count == 4
    assert (tmp_path / "out" / "patches.npz").exists()
    assert (tmp_path / "out" / "patch-grid.png").exists()
    assert (tmp_path / "out" / "patch-manifest.json").exists()


def test_extract_3d_patches_and_mock_report_are_deterministic(tmp_path: Path):
    image_path = tmp_path / "volume.nii.gz"
    data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), image_path)

    manifest = extract_patches(image_path, tmp_path / "out", patch_size=(2, 8, 8))
    report = run_mock_vlm(manifest, hpi="2026-01-01 行 TACE。", labs="AFP 85 ng/mL")

    assert manifest.mode == "volume_3d"
    assert manifest.patch_count == 16
    assert report.patch_count == 16
    assert report.mode == "mock_vlm"
    assert "AFP 85 ng/mL" in report.laboratory_summary


def test_run_skeleton_demo_writes_json_and_markdown(tmp_path: Path):
    image_path = tmp_path / "slice.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_path)

    _, report, report_path = run_skeleton_demo(
        image_path,
        tmp_path / "demo",
        hpi="",
        labs="",
        patch_size=(16, 16),
    )

    assert report_path.exists()
    assert (tmp_path / "demo" / "mock-vlm-report.md").exists()
    assert report.missing_information == ["HPI", "检验结果"]
