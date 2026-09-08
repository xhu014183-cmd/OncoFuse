import json
from pathlib import Path

import pytest

from hcc_multimodal.case_runner import run_case_vlm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COHORT_DIR = PROJECT_ROOT / "public-data" / "cohort"
HCC003_DIR = PROJECT_ROOT / "public-data" / "HCC_003"


def _fake_runner(captured: dict):
    def fake(
        labs,
        *,
        fusion_modes,
        output_dir,
        image_paths,
        imaging_metadata,
        phase,
        timepoint,
        max_tokens,
        temperature,
        json_object,
        timeout_seconds,
    ):
        captured["labs"] = labs
        captured["image_paths"] = image_paths
        captured["imaging_metadata"] = imaging_metadata
        captured["phase"] = phase
        captured["output_dir"] = Path(output_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "web_demo.json").write_text(
            json.dumps(
                {
                    "arms": {
                        "auditable": {"audit_status": "pass"},
                        "open": {"audit_status": "pass"},
                    },
                    "model": "test",
                    "open_numeric_citation_count": 2,
                    "hallucination_candidates": [],
                    "diff": {"open_only": [], "auditable_only": []},
                }
            ),
            encoding="utf-8",
        )

    return fake


def test_prepared_case_with_scenario(tmp_path: Path, monkeypatch):
    if not (COHORT_DIR / "HCC_018" / "public_imaging_evidence.json").exists():
        pytest.skip("prepared HCC_018 case is not present")
    captured: dict = {}
    monkeypatch.setattr(
        "hcc_multimodal.case_runner.run_vlm_dual_arm_demo",
        _fake_runner(captured),
    )

    web_path = run_case_vlm(
        case="HCC_018",
        scenario="dual_marker_rising",
        cohort_dir=COHORT_DIR,
        hcc003_dir=HCC003_DIR,
        output_dir=tmp_path / "out",
    )

    assert web_path.exists()
    assert captured["image_paths"] == [
        COHORT_DIR / "HCC_018" / "converted" / "HCC_018_overlay.png"
    ]
    assert "12.808" in captured["imaging_metadata"]
    assert "AFP" in captured["labs"].analytes
    assert captured["labs"].analytes["AFP"].observations[-1].observed_at == "1998-12-29"
    assert (tmp_path / "out" / "labs.txt").exists()


def test_prepared_case_requires_labs_or_scenario(tmp_path: Path, monkeypatch):
    if not (COHORT_DIR / "HCC_018" / "public_imaging_evidence.json").exists():
        pytest.skip("prepared HCC_018 case is not present")
    monkeypatch.setattr(
        "hcc_multimodal.case_runner.run_vlm_dual_arm_demo",
        _fake_runner({}),
    )

    with pytest.raises(ValueError, match="Provide --labs or --scenario"):
        run_case_vlm(
            case="HCC_018",
            cohort_dir=COHORT_DIR,
            hcc003_dir=HCC003_DIR,
            output_dir=tmp_path / "out",
        )


def test_nifti_input_requires_study_date(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "hcc_multimodal.case_runner.run_vlm_dual_arm_demo",
        _fake_runner({}),
    )

    with pytest.raises(ValueError, match="--study-date"):
        run_case_vlm(
            nifti_image="ct.nii.gz",
            nifti_mask="mask.nii.gz",
            output_dir=tmp_path / "out",
        )


def test_nifti_input_measures_and_overlays(tmp_path: Path, monkeypatch):
    converted = HCC003_DIR / "public-case" / "converted"
    if not (converted / "hcc003_ct.nii.gz").exists():
        pytest.skip("converted HCC_003 NIfTI is not present")
    captured: dict = {}
    monkeypatch.setattr(
        "hcc_multimodal.case_runner.run_vlm_dual_arm_demo",
        _fake_runner(captured),
    )

    web_path = run_case_vlm(
        nifti_image=converted / "hcc003_ct.nii.gz",
        nifti_mask=converted / "hcc003_tumor_mask.nii.gz",
        study_date="1997-09-12",
        phase="portal_venous",
        scenario="markers_normal",
        output_dir=tmp_path / "out",
    )

    assert web_path.exists()
    assert (tmp_path / "out" / "imaging_evidence.json").exists()
    assert (tmp_path / "out" / "overlay.png").exists()
    assert captured["image_paths"] == [tmp_path / "out" / "overlay.png"]
    assert "lesion_count=1" in captured["imaging_metadata"]
    assert "overlay_slices=" in captured["imaging_metadata"]
    assert captured["phase"] == "portal_venous"
