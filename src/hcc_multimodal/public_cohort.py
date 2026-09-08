"""Batch preparation of public HCC-TACE-Seg cases for the development cohort."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .imaging import measure_nifti
from .preview import create_overlay_montage
from .schemas import PIPELINE_VERSION, SCHEMA_VERSION
from .tcia import prepare_public_case


def _load_candidates(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seg_series_by_patient(
    candidates: list[dict[str, Any]],
    patient_id: str,
) -> list[dict[str, Any]]:
    """Return (seg_uid, study_date, study_description) for one patient."""
    record = next(
        (item for item in candidates if item.get("patient_id") == patient_id),
        None,
    )
    if record is None:
        raise ValueError(f"Patient {patient_id} not found in the candidate manifest")
    rows: list[dict[str, Any]] = []
    for study in record.get("studies", []):
        for seg_uid in study.get("seg_series", []):
            rows.append(
                {
                    "seg_series_uid": seg_uid,
                    "study_date": study.get("study_date"),
                    "study_description": study.get("description"),
                }
            )
    return rows


def _prepare_with_local_ct_fallback(
    patient_dir: Path,
    *,
    patient_id: str,
    seg_series_uid: str,
) -> tuple[tuple[Path, Path, Path], list[str]]:
    """Prepare a SEG with its reference CT, then strictly try local CT series.

    Some public SEG objects contain an incorrect referenced-series UID even
    though the matching CT series has already been downloaded for another SEG
    from the same patient.  The fallback never downloads an unreferenced CT and
    does not relax conversion QC: every local candidate must still pass the
    patient-space geometry checks in ``convert_ct_and_mass_seg``.
    """
    errors: list[str] = []
    try:
        prepared = prepare_public_case(
            patient_dir,
            patient_id=patient_id,
            seg_series_uid=seg_series_uid,
        )
        return prepared, errors
    except Exception as exc:  # noqa: BLE001 - retain the candidate failure audit
        errors.append(f"referenced CT: {exc}")

    raw_patient_dir = patient_dir / "raw" / patient_id
    local_ct_uids = sorted(
        path.name.removeprefix("CT_")
        for path in raw_patient_dir.glob("CT_*")
        if path.is_dir() and path.name.removeprefix("CT_")
    )
    for ct_series_uid in local_ct_uids:
        try:
            prepared = prepare_public_case(
                patient_dir,
                patient_id=patient_id,
                seg_series_uid=seg_series_uid,
                ct_series_uid=ct_series_uid,
            )
            return prepared, errors
        except Exception as exc:  # noqa: BLE001 - try the next local CT candidate
            errors.append(f"local CT {ct_series_uid[:24]}...: {exc}")
    raise RuntimeError("; ".join(errors))


def prepare_public_cohort(
    patient_ids: list[str],
    *,
    candidates_path: str | Path,
    output_dir: str | Path,
    minimum_lesion_volume_ml: float = 1.0,
) -> Path:
    """Prepare public cases: download CT+SEG, convert, measure, and overlay.

    For each patient the candidate manifest is used to try each SEG series in
    order until one converts successfully (the SEG-to-CT series is discovered
    automatically). Already-converted cases are skipped, so re-running resumes
    after interruptions. Returns the cohort manifest path.
    """
    output = Path(output_dir)
    if minimum_lesion_volume_ml <= 0:
        raise ValueError("minimum_lesion_volume_ml must be positive")
    output.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(candidates_path)
    manifest_path = output / "cohort_manifest.json"
    previous: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            previous = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("patients", [])
        except (ValueError, TypeError, OSError):
            previous = []
    previous_by_patient = {item["patient_id"]: item for item in previous}
    results: list[dict[str, Any]] = []
    for patient_id in patient_ids:
        seg_rows = _seg_series_by_patient(candidates, patient_id)
        if not seg_rows:
            results.append(
                {
                    "patient_id": patient_id,
                    "status": "error",
                    "error": "no SEG series found in the candidate manifest",
                }
            )
            continue
        patient_dir = output / patient_id
        converted_dir = patient_dir / "converted"
        evidence_path = patient_dir / "public_imaging_evidence.json"
        already_prepared = False
        if evidence_path.exists():
            try:
                existing = json.loads(evidence_path.read_text(encoding="utf-8"))
                already_prepared = int(existing.get("lesion_count") or 0) > 0
            except (ValueError, TypeError, OSError):
                already_prepared = False
        attribution_path = converted_dir / "ATTRIBUTION.json"
        if already_prepared:
            try:
                existing_attribution = json.loads(
                    attribution_path.read_text(encoding="utf-8")
                )
                already_prepared = (
                    existing_attribution.get("source_patient_id") == patient_id
                    and float(
                        previous_by_patient.get(patient_id, {}).get(
                            "minimum_lesion_volume_ml", -1.0
                        )
                    )
                    == minimum_lesion_volume_ml
                )
            except (ValueError, TypeError, OSError):
                already_prepared = False
        if already_prepared:
            existing = json.loads(evidence_path.read_text(encoding="utf-8"))
            image_files = sorted(converted_dir.glob("*_ct.nii.gz"))
            mask_files = sorted(converted_dir.glob("*_tumor_mask.nii.gz"))
            attribution: dict[str, Any] | None = None
            if attribution_path.exists():
                try:
                    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
                except (ValueError, TypeError, OSError):
                    attribution = None
            results.append(
                {
                    "patient_id": patient_id,
                    "status": "already_prepared",
                    "evidence_file": str(evidence_path),
                    "image_file": str(image_files[0]) if len(image_files) == 1 else None,
                    "mask_file": str(mask_files[0]) if len(mask_files) == 1 else None,
                    "overlay_file": str(
                        patient_dir / "converted" / f"{patient_id}_overlay.png"
                    ),
                    "lesion_count": existing.get("lesion_count"),
                    "total_tumor_volume_ml": existing.get("total_tumor_volume_ml"),
                    "max_lesion_extent_mm": existing.get("max_lesion_extent_mm"),
                    "quality_status": existing.get("quality", {}).get("status"),
                    "minimum_lesion_volume_ml": minimum_lesion_volume_ml,
                    "study_date": existing.get("study_date"),
                    "seg_series_uid": (attribution or {}).get(
                        "seg_series_instance_uid"
                    ),
                    "ct_series_uid": (attribution or {}).get("ct_series_instance_uid"),
                }
            )
            continue
        prepared: dict[str, Any] | None = None
        errors: list[str] = []
        for seg_row in seg_rows:
            try:
                (
                    (image_path, mask_path, attribution_path),
                    preparation_warnings,
                ) = _prepare_with_local_ct_fallback(
                    patient_dir,
                    patient_id=patient_id,
                    seg_series_uid=seg_row["seg_series_uid"],
                )
            except Exception as exc:  # noqa: BLE001 - one bad SEG must not abort the batch
                errors.append(
                    f"seg {seg_row['seg_series_uid'][:24]}... "
                    f"({seg_row.get('study_date')}): {exc}"
                )
                continue
            errors.extend(
                f"seg {seg_row['seg_series_uid'][:24]}... {warning}"
                for warning in preparation_warnings
            )
            attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
            try:
                imaging = measure_nifti(
                    image_path,
                    mask_path,
                    patient_id=patient_id,
                    study_date=seg_row["study_date"] or "unknown",
                    modality="CT",
                    phase=str(attribution.get("phase", "unknown")),
                    provider="TCIA-HCC-TACE-Seg-DICOM-SEG",
                    inference_mode="public_expert_annotation",
                    minimum_lesion_volume_ml=minimum_lesion_volume_ml,
                    frame_of_reference_uid=attribution.get("ct_frame_of_reference_uid"),
                    study_instance_uid=attribution.get("study_instance_uid"),
                    series_instance_uid=attribution.get("ct_series_instance_uid"),
                    segment_series_instance_uid=attribution.get(
                        "seg_series_instance_uid"
                    ),
                    segment_number=attribution.get("selected_segment_number"),
                    acquisition_id=attribution.get("selected_acquisition_id"),
                    source_quality_warnings=attribution.get("geometry_qc", {}).get(
                        "warnings", []
                    ),
                )
                imaging.write_json(evidence_path)
                overlay_path = create_overlay_montage(
                    image_path,
                    mask_path,
                    converted_dir / f"{patient_id}_overlay.png",
                )
            except Exception as exc:  # noqa: BLE001 - measurement failure per candidate
                errors.append(f"seg {seg_row['seg_series_uid'][:24]}... measured: {exc}")
                continue
            if imaging.lesion_count == 0:
                errors.append(
                    f"seg {seg_row['seg_series_uid'][:24]}... "
                    f"({seg_row.get('study_date')}): converted but measurement found "
                    "no lesion above the minimum-volume threshold"
                )
                continue
            prepared = {
                "patient_id": patient_id,
                "status": "ok",
                "seg_series_uid": seg_row["seg_series_uid"],
                "ct_series_uid": attribution.get("ct_series_instance_uid"),
                "study_date": seg_row.get("study_date"),
                "study_description": seg_row.get("study_description"),
                "image_file": str(image_path),
                "mask_file": str(mask_path),
                "overlay_file": str(overlay_path),
                "evidence_file": str(evidence_path),
                "lesion_count": imaging.lesion_count,
                "total_tumor_volume_ml": imaging.total_tumor_volume_ml,
                "max_lesion_extent_mm": imaging.max_lesion_extent_mm,
                "quality_status": imaging.quality.status,
                "minimum_lesion_volume_ml": minimum_lesion_volume_ml,
                "geometry_warnings": attribution.get("geometry_qc", {}).get(
                    "warnings", []
                ),
                "errors": errors,
            }
            break
        if prepared is None:
            results.append(
                {
                    "patient_id": patient_id,
                    "status": "error",
                    "error": "; ".join(errors) if errors else "no SEG converted",
                }
            )
        else:
            results.append(prepared)

    merged: dict[str, dict[str, Any]] = {
        item["patient_id"]: item for item in previous
    }
    for item in results:
        merged[item["patient_id"]] = item
    final_patients = [merged[key] for key in sorted(merged)]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "minimum_lesion_volume_ml": minimum_lesion_volume_ml,
        "patients": final_patients,
        "ok_count": sum(1 for item in final_patients if item.get("status") == "ok"),
        "already_prepared_count": sum(
            1 for item in final_patients if item.get("status") == "already_prepared"
        ),
        "error_count": sum(
            1 for item in final_patients if item.get("status") == "error"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["prepare_public_cohort"]
