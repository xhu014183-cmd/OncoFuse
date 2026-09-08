"""One-command orchestration: case + labs -> dual-mode VLM -> web_demo.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from .case_models import ClinicalLabEvidence
from .clinical_labs import parse_laboratory_report
from .cohort_comparison import (
    CASE_LAB_SCENARIOS,
    _case_dir,
    _overlay_path,
    _scenario_text,
)
from .imaging import measure_nifti
from .preview import create_overlay_montage
from .vlm_llm import imaging_metadata_text, run_vlm_dual_arm_demo


def _overlay_slice_indices(overlay_path: str | Path) -> list[int]:
    sidecar = Path(overlay_path).with_name(Path(overlay_path).stem + "_slices.json")
    if not sidecar.exists():
        return []
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")).get(
            "slice_indices", []
        )
    except (ValueError, TypeError, OSError):
        return []


def _load_labs(path: str | Path, patient_id: str) -> ClinicalLabEvidence:
    target = Path(path)
    if target.suffix.lower() == ".txt":
        return parse_laboratory_report(target, patient_id=patient_id)
    try:
        return ClinicalLabEvidence.model_validate_json(
            target.read_text(encoding="utf-8")
        )
    except ValidationError:
        raise ValueError(
            f"{target} is not a ClinicalLabEvidence artifact; convert it first "
            "with `hcc-demo parse-labs`"
        ) from None


def run_case_vlm(
    *,
    case: str | None = None,
    scenario: str | None = None,
    labs_path: str | Path | None = None,
    image_paths: list[str | Path] | None = None,
    imaging_evidence_path: str | Path | None = None,
    nifti_image: str | Path | None = None,
    nifti_mask: str | Path | None = None,
    study_date: str | None = None,
    phase: str | None = None,
    cohort_dir: str | Path = "public-data/cohort",
    hcc003_dir: str | Path = "public-data/HCC_003",
    output_dir: str | Path,
    fusion_modes: tuple[Literal["auditable", "open"], ...] = ("auditable", "open"),
    max_tokens: int = 1024,
    temperature: float = 0.3,
    json_object: bool = False,
    timeout_seconds: float = 120.0,
) -> Path:
    """Run one case end-to-end and return the web-ready ``web_demo.json`` path.

    Imaging input resolution order:

    1. explicit ``image_paths`` + ``imaging_evidence_path``;
    2. a prepared ``case`` (overlay + imaging evidence from the cohort dir);
    3. ``nifti_image`` + ``nifti_mask`` measured on the fly.

    Laboratory input: ``labs_path`` (text report or ClinicalLabEvidence JSON),
    or a ``scenario`` whose observations are anchored to the case study date.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patient_id = case or "CASE"
    resolved_phase = phase or "unknown"

    resolved_images: list[str | Path] = [
        Path(path) for path in (image_paths or [])
    ]
    evidence_path = (
        Path(imaging_evidence_path) if imaging_evidence_path is not None else None
    )
    if case is not None:
        case_dir = _case_dir(case, cohort_dir, hcc003_dir)
        if not resolved_images:
            resolved_images = [_overlay_path(case_dir, case)]
        if evidence_path is None:
            evidence_path = case_dir / "public_imaging_evidence.json"
        if study_date is None:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            study_date = evidence.get("study_date") or "unknown"
        if phase is None:
            attribution_path = case_dir / "converted" / "ATTRIBUTION.json"
            if attribution_path.exists():
                try:
                    attribution = json.loads(
                        attribution_path.read_text(encoding="utf-8")
                    )
                    resolved_phase = str(attribution.get("phase") or "unknown")
                except (ValueError, TypeError, OSError):
                    resolved_phase = "unknown"
    elif nifti_image is not None and nifti_mask is not None:
        if study_date is None:
            raise ValueError("--study-date is required when measuring NIfTI input")
        evidence_path = output / "imaging_evidence.json"
        imaging = measure_nifti(
            nifti_image,
            nifti_mask,
            patient_id=patient_id,
            study_date=study_date,
            modality="CT",
            phase=resolved_phase,
            provider="user_supplied",
            inference_mode="supplied_seg",
            minimum_lesion_volume_ml=1.0,
        )
        imaging.write_json(evidence_path)
        overlay_path = create_overlay_montage(
            nifti_image, nifti_mask, output / "overlay.png"
        )
        resolved_images = [overlay_path]

    if not resolved_images:
        raise ValueError(
            "No visual input resolved; provide --case, or --image/--imaging-evidence, "
            "or --nifti-image/--nifti-mask"
        )
    if evidence_path is None or not evidence_path.exists():
        raise ValueError(f"Imaging evidence not found: {evidence_path}")
    imaging_metadata = imaging_metadata_text(evidence_path)
    overlay_slices = _overlay_slice_indices(resolved_images[0])
    if overlay_slices:
        imaging_metadata += f"; overlay_slices={overlay_slices}"

    if labs_path is not None:
        labs = _load_labs(labs_path, patient_id=patient_id)
    elif scenario is not None:
        if scenario not in CASE_LAB_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        if study_date is None or study_date in ("", "unknown"):
            raise ValueError(
                "Cannot anchor scenario labs without a known study date "
                "(pass --study-date or use a prepared case)"
            )
        labs_path = output / "labs.txt"
        labs_path.write_text(
            _scenario_text(CASE_LAB_SCENARIOS[scenario], study_date),
            encoding="utf-8",
        )
        labs = parse_laboratory_report(labs_path, patient_id=patient_id)
    else:
        raise ValueError("Provide --labs or --scenario")

    run_vlm_dual_arm_demo(
        labs=labs,
        fusion_modes=fusion_modes,
        output_dir=output,
        image_paths=resolved_images,
        imaging_metadata=imaging_metadata,
        phase=resolved_phase,
        timepoint="single",
        max_tokens=max_tokens,
        temperature=temperature,
        json_object=json_object,
        timeout_seconds=timeout_seconds,
    )
    web_path = output / "web_demo.json"
    if not web_path.exists():
        raise RuntimeError("run_vlm_dual_arm_demo completed without web_demo.json")
    return web_path


__all__ = ["run_case_vlm"]
