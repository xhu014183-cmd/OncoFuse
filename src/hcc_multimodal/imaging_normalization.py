"""Normalize one CT examination into a deidentified three-dimensional NIfTI volume."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pydicom

from .case_input_loader import CaseInputLoader
from .schemas import PIPELINE_VERSION, SegRole
from .tcia import convert_ct_and_mass_seg


@dataclass(frozen=True)
class NormalizedImagingInput:
    """Local-only normalized CT and optional mask used by both imaging tracks."""

    image_path: Path
    mask_path: Path | None
    study_date: str
    phase: str
    seg_role: SegRole | None
    qc: dict[str, Any]


def _dicom_date(value: object) -> str:
    text = str(value or "").strip()
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return "unknown"


def _orientation(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(getattr(dataset, "ImageOrientationPatient", []), dtype=float)
    if values.shape != (6,):
        raise ValueError("CT ImageOrientationPatient is missing or malformed")
    row_direction = values[:3]
    column_direction = values[3:]
    if not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-4):
        raise ValueError("CT row direction cosine is not unit length")
    if not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-4):
        raise ValueError("CT column direction cosine is not unit length")
    if not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-4):
        raise ValueError("CT orientation directions are not orthogonal")
    return row_direction, column_direction, np.cross(row_direction, column_direction)


def _infer_phase(dataset: Any) -> str:
    text = " ".join(
        str(getattr(dataset, name, ""))
        for name in ("SeriesDescription", "ProtocolName")
    ).casefold()
    if "portal" in text or "venous" in text:
        return "portal_venous"
    if "arter" in text:
        return "arterial"
    if "delay" in text or "equilibrium" in text:
        return "delayed"
    if "noncontrast" in text or "non-contrast" in text or "plain" in text:
        return "non_contrast"
    return "unknown"


def _load_selected_ct(directory: Path) -> tuple[list[Any], list[str]]:
    datasets: list[Any] = []
    modalities: set[str] = set()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(str(path), force=False)
        except Exception:  # noqa: BLE001, S112
            continue
        modality = str(getattr(dataset, "Modality", "")).upper()
        modalities.add(modality)
        if modality == "CT" and hasattr(dataset, "PixelData"):
            datasets.append(dataset)
    if not datasets:
        if "MR" in modalities:
            raise ValueError("LiON-inspired MVP accepts CT only; MR is not supported")
        raise ValueError(f"No pixel-bearing CT DICOM instances found in {directory}")
    series_counts = Counter(str(item.SeriesInstanceUID) for item in datasets)
    series_uid = min(series_counts, key=lambda key: (-series_counts[key], key))
    selected = [
        item for item in datasets if str(item.SeriesInstanceUID) == series_uid
    ]
    warnings: list[str] = []
    if len(series_counts) > 1:
        warnings.append(
            "Multiple CT series were present; the largest SeriesInstanceUID group was selected"
        )
    return selected, warnings


def _convert_dicom_ct(directory: Path, target: Path) -> tuple[Path, dict[str, Any]]:
    datasets, warnings = _load_selected_ct(directory)
    first_unsorted = datasets[0]
    row_direction, column_direction, normal = _orientation(first_unsorted)
    rows = int(first_unsorted.Rows)
    columns = int(first_unsorted.Columns)
    pixel_spacing = [float(value) for value in first_unsorted.PixelSpacing]
    if len(pixel_spacing) != 2 or min(pixel_spacing) <= 0:
        raise ValueError("CT PixelSpacing must contain two positive values")
    for dataset in datasets:
        candidate_row, candidate_column, _ = _orientation(dataset)
        if int(dataset.Rows) != rows or int(dataset.Columns) != columns:
            raise ValueError("CT rows/columns vary within the selected series")
        if not np.allclose(candidate_row, row_direction, atol=1e-4) or not np.allclose(
            candidate_column, column_direction, atol=1e-4
        ):
            raise ValueError("CT orientation varies within the selected series")
        if not np.allclose(
            np.asarray(dataset.PixelSpacing, dtype=float), pixel_spacing, atol=1e-4
        ):
            raise ValueError("CT PixelSpacing varies within the selected series")
        if np.asarray(getattr(dataset, "ImagePositionPatient", [])).shape != (3,):
            raise ValueError("CT ImagePositionPatient is missing or malformed")
    selected = sorted(
        (
            float(
                np.dot(
                    np.asarray(dataset.ImagePositionPatient, dtype=float), normal
                )
            ),
            dataset,
        )
        for dataset in datasets
    )
    positions = np.asarray([position for position, _ in selected], dtype=float)
    if len(np.unique(np.round(positions, 5))) != len(positions):
        raise ValueError("Duplicate slice positions found in the selected CT series")
    if len(positions) > 1:
        differences = np.diff(positions)
        slice_spacing = float(np.median(differences))
        if slice_spacing <= 0 or not np.allclose(
            differences, slice_spacing, atol=max(1e-3, abs(slice_spacing) * 0.01)
        ):
            raise ValueError("CT slice spacing is non-positive or non-uniform")
    else:
        slice_spacing = float(
            getattr(first_unsorted, "SpacingBetweenSlices", 0.0)
            or getattr(first_unsorted, "SliceThickness", 0.0)
        )
        if slice_spacing <= 0:
            raise ValueError("Single-slice CT requires a positive slice spacing")
    slices: list[np.ndarray] = []
    for _, dataset in selected:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(dataset.pixel_array.astype(np.float32) * slope + intercept)
    volume = np.stack(slices, axis=-1)
    first = selected[0][1]
    row_spacing, column_spacing = pixel_spacing
    lps_affine = np.eye(4, dtype=float)
    lps_affine[:3, 0] = column_direction * row_spacing
    lps_affine[:3, 1] = row_direction * column_spacing
    lps_affine[:3, 2] = normal * slice_spacing
    lps_affine[:3, 3] = np.asarray(first.ImagePositionPatient, dtype=float)
    ras_affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps_affine
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, ras_affine), target)
    patient_ids = sorted(
        {str(getattr(item, "PatientID", "")).strip() for item in datasets}
    )
    if not patient_ids or patient_ids == [""]:
        warnings.append("DICOM PatientID is missing")
    return target, {
        "selected_series_instance_uid": str(first.SeriesInstanceUID),
        "source_patient_id_present": bool(patient_ids and patient_ids != [""]),
        "shape": [int(value) for value in volume.shape],
        "spacing_mm": [row_spacing, column_spacing, slice_spacing],
        "study_date": _dicom_date(getattr(first, "StudyDate", "")),
        "inferred_phase": _infer_phase(first),
        "warnings": warnings,
    }


def _is_nifti(path: Path) -> bool:
    name = path.name.casefold()
    return name.endswith((".nii", ".nii.gz"))


def normalize_case_imaging(
    case: CaseInputLoader,
    *,
    output_dir: str | Path,
) -> NormalizedImagingInput:
    """Normalize a 1.0-1.2 case into one CT NIfTI and write local intake QC."""
    output = Path(output_dir)
    if case.modality != "CT":
        raise ValueError(
            "LiON-inspired run-report accepts CT only; MR must use the legacy evidence-summary path"
        )
    normalized_dir = output / "normalized-imaging"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    mask_path = case.seg_path
    seg_role = cast(SegRole | None, case.seg_role)
    details: dict[str, Any]
    if case.imaging_source_type == "nifti":
        image_path = case.nifti_image_path
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"NIfTI image not found: {image_path}")
        image = cast(nib.Nifti1Image, nib.load(str(image_path)))
        if len(image.shape) != 3:
            raise ValueError("LiON-inspired MVP accepts one three-dimensional CT volume")
        details = {
            "shape": [int(value) for value in image.shape],
            "spacing_mm": [
                round(float(value), 6)
                for value in nib.affines.voxel_sizes(image.affine)
            ],
            "warnings": [
                "NIfTI has no DICOM modality metadata; CT modality is accepted from the case contract"
            ],
        }
        if mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")
        source_type = "nifti"
    elif case.imaging_source_type == "dicom":
        directory = case.dicom_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"DICOM directory not found: {directory}")
        if mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(f"Segmentation file not found: {mask_path}")
        if mask_path is not None and not _is_nifti(mask_path):
            image_path, mask_path, _ = convert_ct_and_mass_seg(
                directory,
                mask_path,
                normalized_dir,
                patient_id=case.patient_id,
            )
            image = cast(nib.Nifti1Image, nib.load(str(image_path)))
            details = {
                "shape": [int(value) for value in image.shape],
                "spacing_mm": [
                    round(float(value), 6)
                    for value in nib.affines.voxel_sizes(image.affine)
                ],
                "warnings": [],
                "dicom_seg_converted": True,
            }
        else:
            image_path, details = _convert_dicom_ct(
                directory, normalized_dir / "ct.nii.gz"
            )
        source_type = "dicom"
    else:
        raise ValueError(f"Unsupported imaging source type: {case.imaging_source_type!r}")

    study_date = case.study_date or str(details.get("study_date") or "unknown")
    phase = case.phase or str(details.get("inferred_phase") or "unknown")
    warnings = [str(item) for item in details.get("warnings", [])]
    if phase == "unknown":
        warnings.append("CT acquisition phase is unknown")
    qc = {
        "schema_version": "1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "warning" if warnings else "pass",
        "source_type": source_type,
        "modality": "CT",
        "single_exam": True,
        "single_phase": True,
        "study_date": study_date,
        "phase": phase,
        "seg_supplied": mask_path is not None,
        "seg_role": seg_role,
        "details": {**details, "warnings": sorted(set(warnings))},
        "privacy": {
            "external_payload": "rerendered_png_only",
            "dicom_metadata_forwarded": False,
            "patient_identifier_forwarded": False,
            "filesystem_path_forwarded": False,
        },
    }
    (output / "imaging-input-qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return NormalizedImagingInput(
        image_path=image_path,
        mask_path=mask_path,
        study_date=study_date,
        phase=phase,
        seg_role=seg_role,
        qc=qc,
    )


__all__ = ["NormalizedImagingInput", "normalize_case_imaging"]
