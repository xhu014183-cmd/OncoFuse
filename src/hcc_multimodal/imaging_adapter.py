"""DICOM CT/MR intake and optional expert SEG measurement adapter."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pydicom

from .case_models import (
    DicomGeometrySummary,
    ImageToolEvidence,
    ImagingInterpretationEvidence,
    QuantitativeLesion,
    StudyIdentity,
)
from .imaging import measure_nifti
from .schemas import QualityCheck, QualityEvidence, QualityStatus, SourceReference
from .tcia import convert_ct_and_mass_seg

_PHI_TAGS = (
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "AccessionNumber",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
)


def _iso_dicom_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return "unknown"


def _phase(dataset: Any) -> str:
    text = " ".join(str(getattr(dataset, key, "")) for key in ("SeriesDescription", "ProtocolName", "SequenceName")).casefold()
    if any(term in text for term in ("portal", "venous", "pv", "门静脉")):
        return "portal_venous"
    if "arter" in text or "动脉" in text:
        return "arterial"
    if "delay" in text or "延迟" in text:
        return "delayed"
    return "unknown"


def _read_series(directory: Path) -> tuple[list[Any], list[str]]:
    datasets: list[Any] = []
    errors: list[str] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception:  # noqa: S112, BLE001
            continue
        if str(getattr(dataset, "Modality", "")) in {"CT", "MR"} and hasattr(dataset, "SOPInstanceUID"):
            datasets.append(dataset)
        for tag in _PHI_TAGS:
            value = str(getattr(dataset, tag, "") or "").strip()
            if value and value not in {"ANON", "ANONYMOUS", "REDACTED"}:
                errors.append(f"PHI field {tag} is populated in {path.name}")
    return datasets, sorted(set(errors))


def _geometry(datasets: list[Any]) -> tuple[DicomGeometrySummary, list[QualityCheck], list[str]]:
    first = datasets[0]
    checks: list[QualityCheck] = []
    warnings: list[str] = []
    orientation = np.asarray(getattr(first, "ImageOrientationPatient", []), dtype=float)
    positions: list[float] = []
    if orientation.size == 6:
        normal = np.cross(orientation[:3], orientation[3:])
        for dataset in datasets:
            position = np.asarray(getattr(dataset, "ImagePositionPatient", []), dtype=float)
            if position.size == 3:
                positions.append(float(np.dot(position, normal)))
    else:
        normal = np.array([0.0, 0.0, 1.0])
        warnings.append("ImageOrientationPatient is missing or malformed")
    positions.sort()
    if len(positions) > 1:
        diffs = np.diff(positions)
        spacing_z = float(np.median(np.abs(diffs)))
        if not np.allclose(np.abs(diffs), spacing_z, atol=max(1e-3, spacing_z * 0.01)):
            warnings.append("Slice spacing is non-uniform")
        if len({round(value, 5) for value in positions}) != len(positions):
            warnings.append("Duplicate slice positions detected")
    else:
        spacing_z = float(getattr(first, "SpacingBetweenSlices", 0.0) or getattr(first, "SliceThickness", 0.0) or 0.0)
    pixel_spacing = [float(value) for value in getattr(first, "PixelSpacing", [0.0, 0.0])]
    if len(pixel_spacing) != 2 or min(pixel_spacing) <= 0 or spacing_z <= 0:
        warnings.append("Pixel or slice spacing is unavailable/non-positive")
    checks.append(QualityCheck(check_id="ORIENTATION", status="pass" if orientation.size == 6 else "fail", message="DICOM orientation inspected"))
    checks.append(QualityCheck(check_id="SLICE_SPACING", status="warning" if any("spacing" in item.lower() for item in warnings) else "pass", message="Slice spacing inspected"))
    return DicomGeometrySummary(
        rows=int(getattr(first, "Rows", 1)),
        columns=int(getattr(first, "Columns", 1)),
        slice_count=len(datasets),
        spacing_mm=[pixel_spacing[0] if len(pixel_spacing) > 0 else 0.0, pixel_spacing[1] if len(pixel_spacing) > 1 else 0.0, spacing_z],
        orientation=[float(value) for value in orientation] if orientation.size == 6 else [0.0] * 6,
        slice_position_range_mm=[positions[0], positions[-1]] if positions else [0.0, 0.0],
    ), checks, warnings


def _tool(path: str | Path | None) -> ImageToolEvidence | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ImageToolEvidence.model_validate(payload)


def parse_imaging_study(
    dicom_dir: str | Path,
    *,
    patient_id: str,
    seg_path: str | Path | None = None,
    image_evidence_path: str | Path | None = None,
    phase: str | None = None,
    output_dir: str | Path | None = None,
) -> ImagingInterpretationEvidence:
    directory = Path(dicom_dir)
    datasets, phi_errors = _read_series(directory)
    if not datasets:
        raise ValueError(f"No CT or MR DICOM instances found in {directory}")
    series_counts = Counter(str(getattr(item, "SeriesInstanceUID", "")) for item in datasets)
    selected_series = min(series_counts, key=lambda key: (-series_counts[key], key))
    selected = [item for item in datasets if str(getattr(item, "SeriesInstanceUID", "")) == selected_series]
    first = min(selected, key=lambda item: str(getattr(item, "SOPInstanceUID", "")))
    modality = cast(Literal["CT", "MR"], str(getattr(first, "Modality", "")).upper())
    study_date = _iso_dicom_date(getattr(first, "StudyDate", ""))
    tool = _tool(image_evidence_path)
    resolved_phase = phase or (tool.phase if tool is not None and tool.phase != "unknown" else _phase(first) if modality == "CT" else "unknown")
    geometry, checks, geometry_warnings = _geometry(selected)
    identity = StudyIdentity(
        patient_id=patient_id,
        study_instance_uid=str(getattr(first, "StudyInstanceUID", "unknown")),
        series_instance_uid=selected_series,
        frame_of_reference_uid=str(getattr(first, "FrameOfReferenceUID", "")) or None,
        modality=modality,
        study_date=study_date,
        phase=resolved_phase,
        series_description=str(getattr(first, "SeriesDescription", "")) or None,
        sequence_name=str(getattr(first, "SequenceName", "")) or None,
    )
    errors = list(phi_errors)
    warnings = list(geometry_warnings)
    if len(series_counts) > 1:
        warnings.append("Multiple CT/MR series were found; the largest SeriesInstanceUID group was selected")
    if not str(getattr(first, "PatientID", "")):
        errors.append("DICOM PatientID is missing")
    elif str(first.PatientID) != patient_id:
        errors.append("DICOM PatientID does not match the research pseudonym")
    quality_status: QualityStatus = "fail" if errors else "warning" if warnings else "pass"
    geometry_qc = QualityEvidence(status=quality_status, spacing_mm=geometry.spacing_mm, checks=checks, warnings=warnings, errors=errors)
    if tool is not None:
        if tool.patient_id != patient_id:
            errors.append("Image-tool evidence patient ID does not match")
        if tool.modality != modality:
            errors.append("Image-tool evidence modality does not match DICOM")
        if tool.study_date != study_date and study_date != "unknown":
            warnings.append("Image-tool evidence study date differs from DICOM")
    qualitative = tool.findings if tool is not None else []
    quantitative: list[QuantitativeLesion] = []
    seg_warning: list[str] = []
    if seg_path is not None:
        try:
            target = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="hcc-imaging-"))
            image_path, mask_path, _ = convert_ct_and_mass_seg(directory, seg_path, target)
            measured = measure_nifti(image_path, mask_path, patient_id=patient_id, study_date=study_date, modality=modality, phase=resolved_phase, provider="dicom-seg")
            quantitative = [QuantitativeLesion(lesion_id=item.lesion_id, volume_ml=item.volume_ml, centroid_world_mm=item.centroid_world_mm, max_3d_extent_mm=item.max_3d_extent_mm, source="expert_seg") for item in measured.lesions]
            warnings.extend(measured.quality.warnings)
            if modality == "MR":
                seg_warning.append("MR SEG measurements use geometry only; no DWI, ADC, or enhancement interpretation is performed")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SEG quantitative processing failed: {exc}")
    if tool is None and not quantitative:
        warnings.append("No structured image-tool observations or SEG measurements were supplied")
    tool_limitations = list(tool.limitations) if tool else []
    if quantitative:
        stale_limitations = [
            item for item in tool_limitations
            if any(term in item.casefold() for term in ("no seg", "without seg", "missing segmentation", "cannot quantify", "\u6ca1\u6709 seg", "\u7f3a\u5c11 seg", "\u4e0d\u63d0\u4f9b\u5206\u5272\u5b9a\u91cf"))
        ]
        if stale_limitations:
            warnings.append("An image-tool segmentation limitation was superseded by the supplied SEG measurement")
            tool_limitations = [item for item in tool_limitations if item not in stale_limitations]
    reported_extents = [item.reported_max_extent_mm for item in tool.findings if item.reported_max_extent_mm is not None] if tool else []
    measured_extents = [item.max_3d_extent_mm for item in quantitative]
    extent_conflict = bool(reported_extents and measured_extents and abs(max(reported_extents) - max(measured_extents)) > max(5.0, max(measured_extents) * 0.25))
    if tool is not None and tool.reported_lesion_count is not None and quantitative and tool.reported_lesion_count != len(quantitative):
        warnings.append("Image-tool lesion count conflicts with SEG connected components")
        consistency: Literal["pass", "warning", "fail", "not_comparable", "unavailable"] = "fail"
    elif extent_conflict:
        warnings.append("Image-tool maximum extent conflicts with the program-computed SEG extent")
        consistency = "fail"
    elif tool is None:
        consistency = "unavailable"
    elif quantitative:
        consistency = "warning" if warnings else "pass"
    else:
        consistency = "not_comparable"
    final_status: QualityStatus = "fail" if errors else "warning" if warnings else "pass"
    return ImagingInterpretationEvidence(
        patient_id=patient_id,
        study_date=study_date,
        modality=modality,
        phase=resolved_phase,
        interpretation_mode="quantitative_and_qualitative" if quantitative and qualitative else "quantitative_only" if quantitative else "qualitative_only" if qualitative else "unavailable",
        study_identity=identity,
        geometry=geometry,
        geometry_qc=geometry_qc,
        quantitative_measurements=quantitative,
        qualitative_observations=qualitative,
        tool_consistency=consistency,
        limitations=sorted(set(seg_warning + tool_limitations + (["No segmentation; no physical lesion quantification"] if not quantitative else []))),
        quality=QualityEvidence(status=final_status, spacing_mm=geometry.spacing_mm, checks=checks, warnings=sorted(set(warnings)), errors=sorted(set(errors))),
        provenance={"dicom_directory": directory.name, "selected_series_uid": selected_series, "adapter": "dicom-ct-mr-v1", "seg_supplied": seg_path is not None},
        sources=[SourceReference(source_id=directory.name, source_type="dicom_series", uri=directory.name, deidentified=not bool(phi_errors))],
    )


__all__ = ["parse_imaging_study"]
