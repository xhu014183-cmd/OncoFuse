from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from .schemas import PIPELINE_VERSION, SCHEMA_VERSION

COLLECTION_ID = "hcc_tace_seg"
COLLECTION_DOI = "https://doi.org/10.7937/TCIA.5FNA-0924"
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_PATIENT_ID = "HCC_003"
STUDY_UID = "1.3.6.1.4.1.14519.5.2.1.1706.8374.128750515241701445125322964595"
CT_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.1706.8374.281650679207816520863173918688"
SEG_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.1706.8374.106355502486885782622426045632"


def _single_dicom(directory: Path) -> Path:
    files = sorted(directory.glob("*.dcm"))
    if len(files) != 1:
        raise ValueError(f"Expected one SEG DICOM in {directory}, found {len(files)}")
    return files[0]


def _io_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def download_hcc003(output_dir: str | Path) -> tuple[Path, Path]:
    """Download the fixed HCC_003 CT and DICOM SEG series from NCI IDC."""
    if os.name == "nt" and sys.flags.utf8_mode == 0:
        raise RuntimeError(
            "On Windows, set PYTHONUTF8=1 before running the public-data downloader"
        )
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError(
            "Install public-data dependencies with: pip install -e '.[public-data]'"
        ) from exc

    output = Path(output_dir)
    raw = output / "raw"
    study = raw / SOURCE_PATIENT_ID
    ct_dir = study / f"CT_{CT_SERIES_UID}"
    seg_dir = study / f"SEG_{SEG_SERIES_UID}"
    if ct_dir.exists() and seg_dir.exists():
        return ct_dir, _single_dicom(seg_dir)

    client = IDCClient()
    client.download_dicom_series(
        [CT_SERIES_UID, SEG_SERIES_UID],
        str(raw),
        quiet=True,
        show_progress_bar=True,
        dirTemplate="%PatientID/%Modality_%SeriesInstanceUID",
    )
    if not ct_dir.exists() or not seg_dir.exists():
        raise FileNotFoundError("IDC download completed without the expected CT/SEG directories")
    return ct_dir, _single_dicom(seg_dir)


def _frame_source_uids(frame: Any) -> list[str]:
    references: list[str] = []
    for derivation in getattr(frame, "DerivationImageSequence", []):
        references.extend(
            str(item.ReferencedSOPInstanceUID)
            for item in getattr(derivation, "SourceImageSequence", [])
        )
    return references


def _orientation(dataset: Any) -> np.ndarray:
    values = np.asarray(dataset.ImageOrientationPatient, dtype=float)
    if values.shape != (6,):
        raise ValueError(f"Invalid ImageOrientationPatient: {values.tolist()}")
    row_direction = values[:3]
    column_direction = values[3:]
    if not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-4):
        raise ValueError("DICOM row direction cosine is not unit length")
    if not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-4):
        raise ValueError("DICOM column direction cosine is not unit length")
    if not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-4):
        raise ValueError("DICOM row and column directions are not orthogonal")
    return values


def _acquisition_id(dataset: Any) -> str:
    for keyword in ("AcquisitionNumber", "TemporalPositionIdentifier", "AcquisitionTime"):
        value = getattr(dataset, keyword, None)
        if value not in (None, ""):
            return f"{keyword}:{value}"
    return "acquisition:unspecified"


def _frame_position(frame: Any) -> np.ndarray | None:
    sequences = list(getattr(frame, "PlanePositionSequence", []))
    sequences.extend(getattr(frame, "PlanePositionSlideSequence", []))
    for item in sequences:
        if hasattr(item, "ImagePositionPatient"):
            return np.asarray(item.ImagePositionPatient, dtype=float)
    return None


def _seg_frame_orientation(seg: Any, frame: Any) -> np.ndarray:
    for group in (frame, seg.SharedFunctionalGroupsSequence[0]):
        sequence = getattr(group, "PlaneOrientationSequence", [])
        if sequence:
            values = np.asarray(sequence[0].ImageOrientationPatient, dtype=float)
            if values.shape == (6,):
                return values
    raise ValueError("SEG frame has no ImageOrientationPatient")


def _seg_frame_spacing(seg: Any, frame: Any) -> tuple[float, float]:
    for group in (frame, seg.SharedFunctionalGroupsSequence[0]):
        sequence = getattr(group, "PixelMeasuresSequence", [])
        if sequence:
            values = tuple(float(value) for value in sequence[0].PixelSpacing)
            if len(values) == 2 and all(value > 0 for value in values):
                return values
    raise ValueError("SEG frame has no valid PixelSpacing")


def _map_seg_frame_to_ct(
    seg: Any,
    frame: Any,
    mask_frame: np.ndarray,
    ct_dataset: Any,
) -> tuple[np.ndarray, float]:
    """Map a SEG pixel plane to the CT NumPy row/column grid in patient LPS space."""
    seg_position = _frame_position(frame)
    if seg_position is None:
        raise ValueError("SEG frame has no ImagePositionPatient for planar mapping")
    seg_orientation = _seg_frame_orientation(seg, frame)
    seg_row_direction = seg_orientation[:3]
    seg_column_direction = seg_orientation[3:]
    seg_row_spacing, seg_column_spacing = _seg_frame_spacing(seg, frame)
    ct_orientation = _orientation(ct_dataset)
    ct_row_direction = ct_orientation[:3]
    ct_column_direction = ct_orientation[3:]
    ct_row_spacing, ct_column_spacing = map(float, ct_dataset.PixelSpacing)
    ct_position = np.asarray(ct_dataset.ImagePositionPatient, dtype=float)

    seg_basis = np.column_stack(
        (seg_column_direction * seg_row_spacing, seg_row_direction * seg_column_spacing)
    )
    ct_basis = np.column_stack(
        (ct_column_direction * ct_row_spacing, ct_row_direction * ct_column_spacing)
    )
    inverse_seg_basis = np.linalg.pinv(seg_basis)
    matrix = inverse_seg_basis @ ct_basis
    offset = inverse_seg_basis @ (ct_position - seg_position)
    mapped = ndimage.affine_transform(
        np.asarray(mask_frame, dtype=np.uint8),
        matrix=matrix,
        offset=offset,
        output_shape=(int(ct_dataset.Rows), int(ct_dataset.Columns)),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.uint8)

    normal = np.cross(ct_row_direction, ct_column_direction)
    plane_error = abs(float(np.dot(seg_position - ct_position, normal)))
    corners = np.asarray(
        [[0, 0], [0, int(ct_dataset.Columns) - 1], [int(ct_dataset.Rows) - 1, 0], [int(ct_dataset.Rows) - 1, int(ct_dataset.Columns) - 1]],
        dtype=float,
    )
    mapped_corners = (matrix @ corners.T).T + offset
    lower_error = np.maximum(-mapped_corners, 0.0)
    upper_bound = np.asarray(mask_frame.shape, dtype=float) - 1
    upper_error = np.maximum(mapped_corners - upper_bound, 0.0)
    in_plane_error_pixels = float(max(lower_error.max(), upper_error.max()))
    if in_plane_error_pixels > 0.51:
        raise ValueError(
            f"SEG-to-CT planar mapping exceeds the source frame by {in_plane_error_pixels:.3f} pixels"
        )
    return mapped, plane_error


def convert_ct_and_mass_seg(
    ct_dir: str | Path,
    seg_path: str | Path,
    output_dir: str | Path,
    *,
    patient_id: str | None = None,
) -> tuple[Path, Path, Path]:
    """Convert one coherent CT acquisition and Mass SEG onto a complete NIfTI grid."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "Install public-data dependencies with: pip install -e '.[public-data]'"
        ) from exc

    ct_dir = Path(ct_dir)
    seg_path = Path(seg_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ct_by_uid: dict[str, Any] = {}
    for path in ct_dir.rglob("*.dcm"):
        dataset = pydicom.dcmread(_io_path(path))
        if hasattr(dataset, "PixelData"):
            ct_by_uid[str(dataset.SOPInstanceUID)] = dataset
    if not ct_by_uid:
        raise ValueError(f"No pixel-bearing CT instances found in {ct_dir}")

    seg = pydicom.dcmread(_io_path(seg_path))
    if str(getattr(seg, "Modality", "")) != "SEG":
        raise ValueError(f"Expected DICOM SEG modality, got {getattr(seg, 'Modality', None)!r}")
    mass_numbers = [
        int(item.SegmentNumber)
        for item in seg.SegmentSequence
        if str(item.SegmentLabel).strip().lower() == "mass"
    ]
    if len(mass_numbers) != 1:
        raise ValueError(f"Expected one segment labelled 'Mass', found {mass_numbers}")
    mass_number = mass_numbers[0]
    pixel_frames = np.asarray(seg.pixel_array)
    if pixel_frames.ndim == 2:
        pixel_frames = pixel_frames[np.newaxis, ...]
    if pixel_frames.shape[0] != len(seg.PerFrameFunctionalGroupsSequence):
        raise ValueError(
            "SEG pixel frame count does not match PerFrameFunctionalGroupsSequence"
        )
    mass_frames = [
        (index, frame, pixel_frames[index])
        for index, frame in enumerate(seg.PerFrameFunctionalGroupsSequence)
        if int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == mass_number
    ]
    if not mass_frames:
        raise ValueError("The DICOM SEG contains no frames for the Mass segment")
    frames = [
        (index, frame, mask_frame)
        for index, frame, mask_frame in mass_frames
        if np.any(mask_frame)
    ]
    if not frames:
        raise ValueError("The DICOM SEG Mass segment contains no non-zero pixels")
    ignored_empty_frame_count = len(mass_frames) - len(frames)

    referenced_counts: Counter[str] = Counter()
    for _, frame, _ in frames:
        for uid in _frame_source_uids(frame):
            if uid in ct_by_uid:
                referenced_counts[_acquisition_id(ct_by_uid[uid])] += 1
    if referenced_counts:
        selected_acquisition = min(
            referenced_counts,
            key=lambda key: (-referenced_counts[key], key),
        )
    else:
        acquisition_datasets: dict[str, list[Any]] = {}
        for dataset in ct_by_uid.values():
            acquisition_datasets.setdefault(_acquisition_id(dataset), []).append(dataset)
        positional_scores: dict[str, float] = {}
        for acquisition_id, datasets in acquisition_datasets.items():
            reference = datasets[0]
            try:
                candidate_orientation = _orientation(reference)
            except ValueError:
                continue
            candidate_normal = np.cross(
                candidate_orientation[:3], candidate_orientation[3:]
            )
            candidate_positions = np.asarray(
                sorted(
                    float(
                        np.dot(
                            np.asarray(dataset.ImagePositionPatient, dtype=float),
                            candidate_normal,
                        )
                    )
                    for dataset in datasets
                ),
                dtype=float,
            )
            if len(candidate_positions) > 1:
                candidate_differences = np.diff(candidate_positions)
                candidate_spacing = float(np.median(candidate_differences))
                if candidate_spacing <= 0 or not np.allclose(
                    candidate_differences,
                    candidate_spacing,
                    atol=max(1e-3, candidate_spacing * 0.01),
                ):
                    continue
            else:
                candidate_spacing = float(
                    getattr(reference, "SpacingBetweenSlices", 0.0)
                    or getattr(reference, "SliceThickness", 0.0)
                )
                if candidate_spacing <= 0:
                    continue
            frame_errors: list[float] = []
            for _, frame, _ in frames:
                frame_position = _frame_position(frame)
                if frame_position is None:
                    frame_errors = []
                    break
                projection = float(np.dot(frame_position, candidate_normal))
                frame_errors.append(
                    float(np.min(np.abs(candidate_positions - projection)))
                )
            if frame_errors and max(frame_errors) <= candidate_spacing / 2 + 1e-3:
                positional_scores[acquisition_id] = max(frame_errors)
        if not positional_scores:
            raise ValueError(
                "Non-zero Mass SEG frames neither reference nor geometrically align with "
                "a complete supplied CT acquisition"
            )
        selected_acquisition = min(
            positional_scores,
            key=lambda key: (positional_scores[key], key),
        )
    selected_datasets = [
        dataset for dataset in ct_by_uid.values() if _acquisition_id(dataset) == selected_acquisition
    ]
    if not selected_datasets:
        raise ValueError(f"No CT instances found for selected {selected_acquisition}")

    first_unsorted = selected_datasets[0]
    dicom_patient_ids = {
        str(getattr(dataset, "PatientID", "")).strip()
        for dataset in [*selected_datasets, seg]
        if str(getattr(dataset, "PatientID", "")).strip()
    }
    if len(dicom_patient_ids) != 1:
        raise ValueError(
            "CT and SEG must contain one consistent non-empty DICOM PatientID"
        )
    dicom_patient_id = next(iter(dicom_patient_ids))
    if patient_id is not None and patient_id != dicom_patient_id:
        raise ValueError(
            "Requested patient_id does not match the CT/SEG DICOM PatientID"
        )
    resolved_patient_id = patient_id or dicom_patient_id
    orientation = _orientation(first_unsorted)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)
    rows = int(first_unsorted.Rows)
    columns = int(first_unsorted.Columns)
    row_spacing, column_spacing = [float(value) for value in first_unsorted.PixelSpacing]
    study_uids = {str(dataset.StudyInstanceUID) for dataset in selected_datasets}
    series_uids = {str(dataset.SeriesInstanceUID) for dataset in selected_datasets}
    if len(study_uids) != 1 or len(series_uids) != 1:
        raise ValueError("Selected CT acquisition contains multiple Study or Series Instance UIDs")
    for dataset in selected_datasets:
        if int(dataset.Rows) != rows or int(dataset.Columns) != columns:
            raise ValueError("CT rows/columns vary within the selected acquisition")
        if not np.allclose(_orientation(dataset), orientation, atol=1e-4):
            raise ValueError("CT ImageOrientationPatient varies within the selected acquisition")
        if not np.allclose(np.asarray(dataset.PixelSpacing, float), [row_spacing, column_spacing], atol=1e-4):
            raise ValueError("CT PixelSpacing varies within the selected acquisition")

    selected = sorted(
        (
            float(np.dot(np.asarray(dataset.ImagePositionPatient, dtype=float), normal)),
            dataset,
        )
        for dataset in selected_datasets
    )
    positions = np.asarray([position for position, _ in selected], dtype=float)
    if len(np.unique(np.round(positions, 5))) != len(positions):
        raise ValueError("Duplicate slice positions remain within the selected CT acquisition")
    if len(positions) > 1:
        differences = np.diff(positions)
        slice_spacing = float(np.median(differences))
        if slice_spacing <= 0:
            raise ValueError(f"Invalid computed slice spacing: {slice_spacing}")
        if not np.allclose(differences, slice_spacing, atol=max(1e-3, slice_spacing * 0.01)):
            raise ValueError(
                "CT has missing slices or non-uniform spacing: "
                f"spacing deltas range {differences.min():.6g}..{differences.max():.6g} mm"
            )
    else:
        slice_spacing = float(getattr(first_unsorted, "SpacingBetweenSlices", 0.0) or getattr(first_unsorted, "SliceThickness", 0.0))
        if slice_spacing <= 0:
            raise ValueError("Single-slice CT requires positive SliceThickness or SpacingBetweenSlices")

    first = selected[0][1]
    target_index_by_uid = {
        str(dataset.SOPInstanceUID): index for index, (_, dataset) in enumerate(selected)
    }
    mask_volume = np.zeros((rows, columns, len(selected)), dtype=np.uint8)
    mapped_frames = 0
    max_landmark_error = 0.0
    for frame_index, frame, mask_frame in frames:
        if np.asarray(mask_frame).shape != (rows, columns):
            raise ValueError(
                f"SEG frame {frame_index} shape {np.asarray(mask_frame).shape} does not match CT "
                f"shape {(rows, columns)}"
            )
        target_indices = {
            target_index_by_uid[uid]
            for uid in _frame_source_uids(frame)
            if uid in target_index_by_uid
        }
        frame_position = _frame_position(frame)
        if len(target_indices) == 1:
            target_index = target_indices.pop()
        elif len(target_indices) > 1:
            raise ValueError(f"SEG frame {frame_index} references multiple slices in one acquisition")
        elif frame_position is not None:
            projection = float(np.dot(frame_position, normal))
            target_index = int(np.argmin(np.abs(positions - projection)))
            error = abs(float(positions[target_index] - projection))
            if error > slice_spacing / 2 + 1e-3:
                raise ValueError(
                    f"SEG frame {frame_index} is {error:.3f} mm from the nearest CT slice"
                )
        else:
            raise ValueError(
                f"SEG frame {frame_index} has neither a selected-acquisition SOP reference nor a position"
            )
        mapped_frame, landmark_error = _map_seg_frame_to_ct(
            seg,
            frame,
            np.asarray(mask_frame),
            selected[target_index][1],
        )
        max_landmark_error = max(max_landmark_error, landmark_error)
        if landmark_error > slice_spacing / 2 + 1e-3:
            raise ValueError(
                f"SEG frame {frame_index} plane error {landmark_error:.3f} mm exceeds half a voxel"
            )
        mask_volume[:, :, target_index] |= mapped_frame
        mapped_frames += 1

    ct_slices: list[np.ndarray] = []
    for _, source in selected:
        slope = float(getattr(source, "RescaleSlope", 1.0))
        intercept = float(getattr(source, "RescaleIntercept", 0.0))
        ct_slices.append(source.pixel_array.astype(np.float32) * slope + intercept)
    ct_volume = np.stack(ct_slices, axis=-1)

    # NumPy axis 0 is DICOM row (second IOP direction); axis 1 is column (first IOP).
    row_spacing, column_spacing = [float(value) for value in first.PixelSpacing]
    lps_affine = np.eye(4, dtype=float)
    lps_affine[:3, 0] = column_direction * row_spacing
    lps_affine[:3, 1] = row_direction * column_spacing
    lps_affine[:3, 2] = normal * slice_spacing
    lps_affine[:3, 3] = np.asarray(first.ImagePositionPatient, dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_affine = lps_to_ras @ lps_affine

    output_stem = re.sub(r"[^a-z0-9]+", "", resolved_patient_id.lower())
    if not output_stem:
        raise ValueError("DICOM PatientID cannot be converted to a safe output filename")
    image_path = output / f"{output_stem}_ct.nii.gz"
    mask_path = output / f"{output_stem}_tumor_mask.nii.gz"
    attribution_path = output / "ATTRIBUTION.json"
    nib.save(nib.Nifti1Image(ct_volume, ras_affine), image_path)
    nib.save(nib.Nifti1Image(mask_volume, ras_affine), mask_path)
    ct_for = str(getattr(first, "FrameOfReferenceUID", "")) or None
    seg_for = str(getattr(seg, "FrameOfReferenceUID", "")) or None
    warnings: list[str] = []
    if ignored_empty_frame_count:
        warnings.append(
            f"Ignored {ignored_empty_frame_count} empty Mass SEG frames during geometry "
            "selection and mapping"
        )
    if not referenced_counts:
        warnings.append(
            "The non-zero Mass SEG frames had no matching per-frame SOP references; "
            "the CT acquisition was selected by verified patient-space alignment"
        )
    if len(referenced_counts) > 1:
        warnings.append(
            "The source SeriesInstanceUID contains multiple acquisitions; "
            f"selected {selected_acquisition} deterministically"
        )
    if ct_for and seg_for and ct_for != seg_for:
        warnings.append(
            "SEG and CT FrameOfReferenceUID differ; direct SOP references and per-frame "
            "landmarks were used to verify the mapping"
        )
    attribution = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "collection": "HCC-TACE-Seg",
        "collection_idc_id": COLLECTION_ID,
        "collection_doi": COLLECTION_DOI,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "source_patient_id": resolved_patient_id,
        "source_patient_id_validation": "consistent CT/SEG DICOM PatientID",
        "study_instance_uid": next(iter(study_uids)),
        "ct_series_instance_uid": next(iter(series_uids)),
        "seg_series_instance_uid": str(seg.SeriesInstanceUID),
        "ct_frame_of_reference_uid": ct_for,
        "seg_frame_of_reference_uid": seg_for,
        "selected_acquisition_id": selected_acquisition,
        "selected_segment_label": "Mass",
        "selected_segment_number": mass_number,
        "phase": _infer_phase(ct_by_uid),
        "conversion": "complete selected CT acquisition plus frame-mapped DICOM SEG; LPS converted to RAS",
        "geometry_qc": {
            "status": "warning" if warnings else "pass",
            "ct_instances_in_target": len(selected),
            "seg_frames_mapped": mapped_frames,
            "target_shape": list(ct_volume.shape),
            "spacing_mm": [row_spacing, column_spacing, slice_spacing],
            "max_seg_landmark_error_mm": max_landmark_error,
            "warnings": warnings,
        },
        "redistribution": "Generated files remain subject to source license and attribution terms",
    }
    attribution_path.write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return image_path, mask_path, attribution_path


def prepare_hcc003(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    ct_dir, seg_path = download_hcc003(output)
    return convert_ct_and_mass_seg(ct_dir, seg_path, output / "converted")


def _seg_referenced_series_uids(seg_path: str | Path) -> list[str]:
    """Return the CT series UIDs a DICOM SEG references as its source series."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "Install public-data dependencies with: pip install -e '.[public-data]'"
        ) from exc
    seg = pydicom.dcmread(_io_path(Path(seg_path)))
    uids: list[str] = []
    for series_item in getattr(seg, "ReferencedSeriesSequence", []):
        uid = getattr(series_item, "SeriesInstanceUID", None)
        if uid:
            uids.append(str(uid))
    return uids


PHASE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bportal\b|\bvenous\b", re.IGNORECASE), "portal_venous"),
    (re.compile(r"\barterial\b|\bartery\b", re.IGNORECASE), "arterial"),
    (re.compile(r"\bdelayed\b|\bequilibrium\b", re.IGNORECASE), "delayed"),
    (
        re.compile(r"\bnon[- ]?contrast\b|\bunenhanced\b|\bplain\b|\bpre\b", re.IGNORECASE),
        "non_contrast",
    ),
]


def _infer_phase(ct_by_uid: dict[str, Any]) -> str:
    """Infer the contrast phase from CT DICOM series descriptions/protocols."""
    parts: list[str] = []
    for dataset in ct_by_uid.values():
        for attr in ("SeriesDescription", "ProtocolName"):
            value = getattr(dataset, attr, None)
            if value:
                parts.append(str(value))
    text = " ".join(parts)
    for pattern, label in PHASE_PATTERNS:
        if pattern.search(text):
            return label
    return "unknown"


def prepare_public_case(
    output_dir: str | Path,
    *,
    patient_id: str,
    seg_series_uid: str,
    ct_series_uid: str | None = None,
) -> tuple[Path, Path, Path]:
    """Download one HCC-TACE-Seg patient's CT+SEG series and convert to NIfTI.

    When ``ct_series_uid`` is omitted, the SEG is downloaded first and its
    ``ReferencedSeriesSequence`` is used to discover the CT series it was
    segmented from. Existing local files are reused, so re-running is a no-op
    download. Returns ``(image_path, mask_path, attribution_path)``.
    """
    if os.name == "nt" and sys.flags.utf8_mode == 0:
        raise RuntimeError(
            "On Windows, set PYTHONUTF8=1 before running the public-data downloader"
        )
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError(
            "Install public-data dependencies with: pip install -e '.[public-data]'"
        ) from exc

    output = Path(output_dir)
    raw = output / "raw"
    study = raw / patient_id
    if ct_series_uid is None:
        seg_only = study / f"SEG_{seg_series_uid}"
        if not seg_only.exists():
            client = IDCClient()
            client.download_dicom_series(
                [seg_series_uid],
                str(raw),
                quiet=True,
                show_progress_bar=True,
                dirTemplate="%PatientID/%Modality_%SeriesInstanceUID",
            )
        if not seg_only.exists():
            raise FileNotFoundError(
                f"IDC download completed without SEG series {seg_series_uid}"
            )
        referenced = _seg_referenced_series_uids(_single_dicom(seg_only))
        if not referenced:
            raise ValueError(
                f"SEG {seg_series_uid} has no ReferencedSeriesSequence; "
                "pass --ct-series explicitly"
            )
        ct_series_uid = referenced[0]
        print(
            f"SEG {seg_series_uid} references CT series {ct_series_uid[:30]}... "
            f"({len(referenced)} referenced)"
        )

    ct_dir = study / f"CT_{ct_series_uid}"
    seg_dir = study / f"SEG_{seg_series_uid}"
    missing = [path for path in (ct_dir, seg_dir) if not path.exists()]
    if missing:
        client = IDCClient()
        client.download_dicom_series(
            [ct_series_uid, seg_series_uid],
            str(raw),
            quiet=True,
            show_progress_bar=True,
            dirTemplate="%PatientID/%Modality_%SeriesInstanceUID",
        )
    if not ct_dir.exists() or not seg_dir.exists():
        raise FileNotFoundError(
            f"IDC download completed without the expected series for {patient_id}"
        )
    return convert_ct_and_mass_seg(
        ct_dir,
        _single_dicom(seg_dir),
        output / "converted",
        patient_id=patient_id,
    )


LAB_SCENARIOS = {
    "dual_marker_rising": {
        "description": "AFP and DCP both rise from normal to above the supplied upper reference",
        "observations": [
            ("1997-06-12", "AFP", 6.0, "ng/mL", 7.0),
            ("1997-09-12", "AFP", 85.3, "ng/mL", 7.0),
            ("1997-06-12", "DCP", 25.0, "mAU/mL", 40.0),
            ("1997-09-12", "DCP", 68.0, "mAU/mL", 40.0),
        ],
    },
    "afp_negative_dcp_rising": {
        "description": "AFP remains normal while DCP rises above the supplied upper reference",
        "observations": [
            ("1997-06-12", "AFP", 5.8, "ng/mL", 7.0),
            ("1997-09-12", "AFP", 6.2, "ng/mL", 7.0),
            ("1997-06-12", "DCP", 22.0, "mAU/mL", 40.0),
            ("1997-09-12", "DCP", 85.0, "mAU/mL", 40.0),
        ],
    },
    "markers_normal": {
        "description": "AFP and DCP remain within their supplied reference ranges",
        "observations": [
            ("1997-06-12", "AFP", 5.5, "ng/mL", 7.0),
            ("1997-09-12", "AFP", 6.0, "ng/mL", 7.0),
            ("1997-06-12", "DCP", 20.0, "mAU/mL", 40.0),
            ("1997-09-12", "DCP", 22.0, "mAU/mL", 40.0),
        ],
    },
}


def write_composite_labs(
    path: str | Path,
    patient_id: str,
    scenario_id: str = "dual_marker_rising",
) -> Path:
    """Write a declared PoC marker scenario unrelated to the TCIA subject."""
    if scenario_id not in LAB_SCENARIOS:
        raise ValueError(f"Unknown lab scenario: {scenario_id}")
    scenario = LAB_SCENARIOS[scenario_id]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [
            {"source_id": f"synthetic-{scenario_id}", "source_type": "synthetic_laboratory"}
        ],
        "patient_id": patient_id,
        "data_relationship": (
            "Synthetic teaching values; not measured from or linked to TCIA subject HCC_003"
        ),
        "evidence_context": {
            "data_origin": "synthetic",
            "pairing_status": "unpaired_poc_composite",
        },
        "scenario_id": scenario_id,
        "scenario_description": scenario["description"],
        "observations": [
            {
                "date": date,
                "marker": marker,
                "value": value,
                "unit": unit,
                "upper_reference": upper,
            }
            for date, marker, value, unit, upper in scenario["observations"]
        ],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def enumerate_hcc_tace_seg_candidates(
    *,
    require_seg: bool = True,
) -> list[dict[str, Any]]:
    """Enumerate HCC-TACE-Seg patients with CT and SEG series from the IDC index.

    Metadata-only screening: no imaging is downloaded. Each record lists the
    patient's studies (timepoints), CT/SEG series, and an estimated download
    size, so a small development cohort can be selected before any transfer.
    """
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError(
            "Install public-data dependencies with: pip install -e '.[public-data]'"
        ) from exc

    client = IDCClient()
    patients = client.get_patients(COLLECTION_ID)
    records: list[dict[str, Any]] = []
    for patient in patients:
        patient_id = patient["PatientID"]
        try:
            studies = client.get_dicom_studies(patient_id)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            records.append({"patient_id": patient_id, "error": str(exc)})
            continue
        study_records: list[dict[str, Any]] = []
        total_mb = 0.0
        ct_series_count = 0
        seg_series_count = 0
        timepoints: list[str] = []
        for study in studies:
            try:
                series = client.get_dicom_series(study["StudyInstanceUID"])
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
                study_records.append(
                    {"study_uid": study["StudyInstanceUID"], "error": str(exc)}
                )
                continue
            ct = [item for item in series if item.get("Modality") == "CT"]
            seg = [item for item in series if item.get("Modality") == "SEG"]
            ct_series_count += len(ct)
            seg_series_count += len(seg)
            study_mb = sum(
                float(item.get("series_size_MB") or 0.0) for item in ct + seg
            )
            total_mb += study_mb
            if study.get("StudyDate"):
                timepoints.append(str(study["StudyDate"]))
            study_records.append(
                {
                    "study_uid": study["StudyInstanceUID"],
                    "study_date": study.get("StudyDate"),
                    "description": study.get("StudyDescription"),
                    "ct_series": [
                        {
                            "series_uid": item["SeriesInstanceUID"],
                            "description": item.get("SeriesDescription"),
                            "instances": item.get("ImageCount")
                            or item.get("instance_count"),
                            "size_mb": item.get("series_size_MB"),
                        }
                        for item in ct
                    ],
                    "seg_series": [item["SeriesInstanceUID"] for item in seg],
                    "estimated_mb": round(study_mb, 1),
                }
            )
        records.append(
            {
                "patient_id": patient_id,
                "study_count": len(studies),
                "timepoints": timepoints,
                "ct_series_count": ct_series_count,
                "seg_series_count": seg_series_count,
                "estimated_ct_seg_mb": round(total_mb, 1),
                "studies": study_records,
            }
        )
    if require_seg:
        records = [
            record
            for record in records
            if "error" not in record and record.get("seg_series_count", 0) > 0
        ]
    records.sort(key=lambda record: record["patient_id"])
    return records
