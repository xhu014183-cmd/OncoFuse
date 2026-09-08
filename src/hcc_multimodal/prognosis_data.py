"""Public-data synchronization and cohort adapters for HCC OS research."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .prognosis_features import extract_mask_features
from .prognosis_models import (
    CohortExclusion,
    PublicPrognosisCohortArtifact,
    PublicPrognosisRecord,
    SurvivalEndpointRecord,
)
from .public_cohort import prepare_public_cohort
from .schemas import QualityCheck, QualityEvidence, SourceReference

WAW_RECORD_ID = "12741586"
WAW_DOI = "10.5281/zenodo.12741586"
WAW_VERSION = "v2_15_07_2024"
HCC_TACE_SEG_VERSION = "v2"
HCC_TACE_SEG_DOI = "10.7937/TCIA.5FNA-0924"
HCC_CLINICAL_URL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "HCC-TACE-Seg_clinical_data-V2.xlsx"
)
WAW_METADATA_FILES = {
    "clinical_data_wawtace_v2_15_07_2024.xlsx",
    "ct_hcc_metadata_v2.csv",
    "radiomics_data_wawtace_09_05_2024.xlsx",
    "supplementary_table_s1_definitions_v2.xlsx",
    "tumor_masks_wawtace_v1_08_05_2024.zip",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    target: Path,
    *,
    expected_md5: str | None = None,
) -> Path:
    """Resumable download that promotes a verified .part file atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_md5 is None or _md5_file(target) == expected_md5):
        return target
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", 200))
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_md5 is not None and _md5_file(partial) != expected_md5:
        raise ValueError(f"MD5 mismatch for {target.name}")
    partial.replace(target)
    return target


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (destination / member.filename).resolve()
            if root not in resolved.parents and resolved != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sync_waw(root: Path, tier: Literal["metadata", "pilot", "full"]) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"https://zenodo.org/api/records/{WAW_RECORD_ID}", timeout=60
    ) as response:
        record = json.loads(response.read().decode("utf-8"))
    requested = set(WAW_METADATA_FILES)
    if tier == "full":
        requested.update(
            item["key"] for item in record["files"] if item["key"].startswith("ct_scans_")
        )
    downloads = root / "downloads"
    files: list[dict[str, Any]] = []
    for item in record["files"]:
        if item["key"] not in requested:
            continue
        checksum = str(item["checksum"])
        expected_md5 = checksum.removeprefix("md5:")
        target = _download(
            item["links"]["self"],
            downloads / item["key"],
            expected_md5=expected_md5,
        )
        files.append(
            {
                "name": item["key"],
                "bytes": target.stat().st_size,
                "md5": expected_md5,
                "sha256": _sha256_file(target),
                "source_url": item["links"]["self"],
            }
        )
        if item["key"] == "tumor_masks_wawtace_v1_08_05_2024.zip":
            _safe_extract(target, root / "extracted" / "tumor_masks")
        if tier == "full" and item["key"].startswith("ct_scans_"):
            _safe_extract(target, root / "extracted" / "ct_scans")
    return {
        "dataset": "waw_tace",
        "version": WAW_VERSION,
        "doi": WAW_DOI,
        "license": "CC BY 4.0",
        "tier": tier,
        "files": files,
    }


def _bundled_candidates() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "hcc_tace_seg_candidates.json"


def _stable_patients(candidates: list[dict[str, Any]], count: int) -> list[str]:
    ranked = sorted(
        (str(item["patient_id"]) for item in candidates),
        key=lambda patient_id: _sha256_text(f"hcc-tace-seg-pilot:{patient_id}"),
    )
    return ranked[:count]


def _baseline_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict every patient to the earliest dated study that contains a SEG."""
    baseline: list[dict[str, Any]] = []
    for item in candidates:
        studies = [study for study in item.get("studies", []) if study.get("seg_series")]
        selected = min(
            studies,
            key=lambda study: (
                str(study.get("study_date") or "9999-99-99"),
                str(study.get("study_uid") or ""),
            ),
            default=None,
        )
        record = dict(item)
        record["studies"] = [selected] if selected is not None else []
        record["study_count"] = len(record["studies"])
        record["timepoints"] = (
            [selected.get("study_date")] if selected is not None else []
        )
        baseline.append(record)
    return baseline


def _sync_hcc(
    root: Path,
    tier: Literal["metadata", "pilot", "full"],
) -> dict[str, Any]:
    downloads = root / "downloads"
    clinical = _download(HCC_CLINICAL_URL, downloads / "HCC-TACE-Seg_clinical_data-V2.xlsx")
    candidates_target = root / "metadata" / "hcc_tace_seg_candidates.json"
    candidates_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_bundled_candidates(), candidates_target)
    candidates = json.loads(candidates_target.read_text(encoding="utf-8"))
    baseline_candidates = _baseline_candidates(candidates)
    baseline_target = root / "metadata" / "hcc_tace_seg_baseline_candidates.json"
    _write_json(baseline_target, baseline_candidates)
    prepared_manifest: str | None = None
    preparation_counts: dict[str, int] | None = None
    selected: list[str] = []
    if tier in {"pilot", "full"}:
        selected = (
            _stable_patients(candidates, 5)
            if tier == "pilot"
            else sorted(str(item["patient_id"]) for item in candidates)
        )
        prepared_path = prepare_public_cohort(
            selected,
            candidates_path=baseline_target,
            output_dir=root / "prepared",
            minimum_lesion_volume_ml=0.01,
        )
        prepared_manifest = str(prepared_path.resolve())
        prepared_payload = json.loads(prepared_path.read_text(encoding="utf-8"))
        preparation_counts = {
            "requested": len(selected),
            "prepared": sum(
                item.get("status") in {"ok", "already_prepared"}
                for item in prepared_payload.get("patients", [])
            ),
            "errors": sum(
                item.get("status") == "error"
                for item in prepared_payload.get("patients", [])
            ),
        }
    return {
        "dataset": "hcc_tace_seg",
        "version": HCC_TACE_SEG_VERSION,
        "doi": HCC_TACE_SEG_DOI,
        "license": "CC BY 4.0",
        "tier": tier,
        "clinical_file": {
            "name": clinical.name,
            "bytes": clinical.stat().st_size,
            "sha256": _sha256_file(clinical),
            "source_url": HCC_CLINICAL_URL,
        },
        "candidate_count": len(candidates),
        "baseline_candidate_count": sum(
            bool(item.get("studies")) for item in baseline_candidates
        ),
        "baseline_selection": "earliest dated study containing DICOM SEG",
        "minimum_connected_component_volume_ml": 0.01,
        "series_inventory": str(candidates_target),
        "baseline_candidate_manifest": str(baseline_target),
        "selected_patient_ids": selected,
        "prepared_manifest": prepared_manifest,
        "preparation_counts": preparation_counts,
    }


def sync_prognosis_data(
    dataset: Literal["waw-tace", "hcc-tace-seg"],
    *,
    tier: Literal["metadata", "pilot", "full"],
    data_root: str | Path,
    accept_license: bool,
) -> Path:
    if not accept_license:
        raise ValueError("--accept-license is required for public dataset synchronization")
    root = Path(data_root).resolve() / dataset
    root.mkdir(parents=True, exist_ok=True)
    details = _sync_waw(root, tier) if dataset == "waw-tace" else _sync_hcc(root, tier)
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "license_accepted": True,
        "data_root": str(root),
        **details,
    }
    manifest_path = _write_json(root / "dataset-manifest.json", manifest)
    counts = details.get("preparation_counts")
    if isinstance(counts, dict) and int(counts.get("errors", 0)) > 0:
        raise RuntimeError(
            "Public imaging preparation failed for one or more patients; "
            f"audit retained at {manifest_path}"
        )
    return manifest_path


def _subject_identity(cohort: str, version: str, source_id: str) -> tuple[str, str]:
    source_hash = _sha256_text(f"{cohort}:{version}:{source_id}")
    prefix = "WAW" if cohort == "waw_tace" else "TCIA"
    return f"{prefix}_{source_hash[:12].upper()}", source_hash


def _source(cohort: str, source_id: str, source_type: str) -> SourceReference:
    return SourceReference(
        source_id=source_id,
        source_type=source_type,
        data_origin="real_public",
        deidentified=True,
        details={"cohort": cohort},
    )


class WawTaceAdapter:
    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root)

    def build(
        self,
    ) -> tuple[list[PublicPrognosisRecord], list[SurvivalEndpointRecord], list[CohortExclusion]]:
        clinical_path = self.root / "downloads" / "clinical_data_wawtace_v2_15_07_2024.xlsx"
        if not clinical_path.is_file():
            raise FileNotFoundError(f"WAW clinical file not found: {clinical_path}")
        frame = pd.read_excel(clinical_path)
        records: list[PublicPrognosisRecord] = []
        endpoints: list[SurvivalEndpointRecord] = []
        exclusions: list[CohortExclusion] = []
        for _, row in frame.iterrows():
            source_id = str(int(row["PATPRI"]))
            patient_id, source_hash = _subject_identity("waw_tace", WAW_VERSION, source_id)
            try:
                masks = sorted(
                    (self.root / "extracted" / "tumor_masks").rglob(
                        f"{source_id}_*_tumor_seg.nrrd"
                    )
                )
                if not masks:
                    raise ValueError("WAW_TUMOR_MASK_MISSING")
                phases = {
                    match.group(1)
                    for path in masks
                    if (match := re.match(rf"{re.escape(source_id)}_(\d)_", path.name))
                }
                if len(phases) != 1:
                    raise ValueError("WAW_MASK_PHASE_AMBIGUOUS")
                phase_code = next(iter(phases))
                phase = {"0": "noncontrast", "1": "arterial", "2": "portal_venous", "3": "delayed"}.get(
                    phase_code, "unknown"
                )
                portal_image: Path | None = None
                if phase_code == "2":
                    matches = list(
                        (self.root / "extracted" / "ct_scans").rglob(
                            f"{source_id}_2_scan.nii*"
                        )
                    )
                    portal_image = matches[0] if len(matches) == 1 else None
                imaging = extract_mask_features(
                    masks,
                    portal_image_path=portal_image,
                )
                albumin = float(row["lab_albumin"])
                warnings = list(imaging.warnings)
                warnings.append(
                    "SOURCE_UNIT_OVERRIDE: WAW v2 lab_albumin values are interpreted as g/dL; "
                    "the source dictionary labels them g/L"
                )
                records.append(
                    PublicPrognosisRecord(
                        patient_id=patient_id,
                        cohort_id="waw_tace",
                        dataset_version=WAW_VERSION,
                        source_patient_sha256=source_hash,
                        baseline_phase=phase,
                        geometry_qc=imaging.geometry_qc,
                        age_years=float(row["age"]),
                        female=int(row["gender_woman"]),
                        afp_ng_ml=float(row["lab_afp"]),
                        lesion_count=imaging.lesion_count,
                        total_tumor_volume_ml=imaging.total_tumor_volume_ml,
                        max_lesion_extent_mm=imaging.max_lesion_extent_mm,
                        largest_lesion_sphericity=imaging.largest_lesion_sphericity,
                        albumin_g_dl=albumin,
                        bilirubin_mg_dl=float(row["lab_bilirubin"]),
                        inr=float(row["lab_inr"]),
                        alt_iu_l=float(row["lab_alt"]),
                        creatinine_mg_dl=float(row["lab_creatinine"]),
                        portal_mean_hu=imaging.portal_mean_hu,
                        portal_std_hu=imaging.portal_std_hu,
                        portal_p10_hu=imaging.portal_p10_hu,
                        portal_p90_hu=imaging.portal_p90_hu,
                        quality_warnings=warnings,
                        source_refs=[
                            _source("waw_tace", clinical_path.name, "clinical_xlsx"),
                            *[
                                _source("waw_tace", path.name, "public_tumor_seg")
                                for path in masks
                            ],
                        ],
                    )
                )
                endpoints.append(
                    SurvivalEndpointRecord(
                        patient_id=patient_id,
                        cohort_id="waw_tace",
                        duration_days=float(row["survival_time"]),
                        event=int(row["death"]),
                        source_ref=_source("waw_tace", clinical_path.name, "outcome_xlsx"),
                    )
                )
            except (ValueError, TypeError, KeyError, OSError) as exc:
                exclusions.append(
                    CohortExclusion(
                        cohort_id="waw_tace",
                        source_patient_sha256=source_hash,
                        patient_id=patient_id,
                        reason_codes=[str(exc).split(":", maxsplit=1)[0]],
                        messages=[str(exc)],
                    )
                )
        return records, endpoints, exclusions


class HccTaceSegAdapter:
    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root)

    def build(
        self,
    ) -> tuple[list[PublicPrognosisRecord], list[SurvivalEndpointRecord], list[CohortExclusion]]:
        clinical_path = self.root / "downloads" / "HCC-TACE-Seg_clinical_data-V2.xlsx"
        prepared_path = self.root / "prepared" / "cohort_manifest.json"
        if not clinical_path.is_file():
            raise FileNotFoundError(f"HCC-TACE-Seg clinical file not found: {clinical_path}")
        frame = pd.read_excel(clinical_path, sheet_name="data table")
        baseline_path = self.root / "metadata" / "hcc_tace_seg_baseline_candidates.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(
                "HCC baseline candidate manifest not found; run sync-prognosis-data first"
            )
        baseline_rows = json.loads(baseline_path.read_text(encoding="utf-8"))
        allowed_baseline_seg = {
            str(item["patient_id"]): {
                str(seg_uid)
                for study in item.get("studies", [])
                for seg_uid in study.get("seg_series", [])
            }
            for item in baseline_rows
        }
        prepared_rows: dict[str, dict[str, Any]] = {}
        if prepared_path.is_file():
            prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
            prepared_rows = {
                str(item["patient_id"]): item
                for item in prepared.get("patients", [])
                if item.get("status") in {"ok", "already_prepared"}
            }
        records: list[PublicPrognosisRecord] = []
        endpoints: list[SurvivalEndpointRecord] = []
        exclusions: list[CohortExclusion] = []
        for _, row in frame.iterrows():
            source_id = str(row["TCIA_ID"])
            patient_id, source_hash = _subject_identity(
                "hcc_tace_seg", HCC_TACE_SEG_VERSION, source_id
            )
            try:
                prepared = prepared_rows.get(source_id)
                if prepared is None:
                    raise ValueError("HCC_BASELINE_SEG_NOT_PREPARED")
                if str(prepared.get("seg_series_uid")) not in allowed_baseline_seg.get(
                    source_id, set()
                ):
                    raise ValueError("HCC_NON_BASELINE_SEG_BLOCKED")
                image_path, mask_path = self._prepared_image_and_mask(
                    source_id, prepared
                )
                evidence_path = Path(str(prepared["evidence_file"]))
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                phase = str(evidence.get("phase") or "annotation_native_phase")
                portal = image_path if phase == "portal_venous" else None
                imaging = extract_mask_features(mask_path, portal_image_path=portal)
                records.append(
                    PublicPrognosisRecord(
                        patient_id=patient_id,
                        cohort_id="hcc_tace_seg",
                        dataset_version=HCC_TACE_SEG_VERSION,
                        source_patient_sha256=source_hash,
                        baseline_phase=phase,
                        geometry_qc=imaging.geometry_qc,
                        age_years=float(row["age"]),
                        female=1 if int(row["Sex"]) == 2 else 0,
                        afp_ng_ml=float(row["AFP"]),
                        lesion_count=imaging.lesion_count,
                        total_tumor_volume_ml=imaging.total_tumor_volume_ml,
                        max_lesion_extent_mm=imaging.max_lesion_extent_mm,
                        largest_lesion_sphericity=imaging.largest_lesion_sphericity,
                        portal_mean_hu=imaging.portal_mean_hu,
                        portal_std_hu=imaging.portal_std_hu,
                        portal_p10_hu=imaging.portal_p10_hu,
                        portal_p90_hu=imaging.portal_p90_hu,
                        missing_items=[
                            "albumin",
                            "bilirubin",
                            "INR",
                            "ALT",
                            "creatinine",
                            "DCP",
                            "AFP-L3%",
                        ],
                        quality_warnings=imaging.warnings,
                        source_refs=[
                            _source("hcc_tace_seg", clinical_path.name, "clinical_xlsx"),
                            _source(
                                "hcc_tace_seg",
                                f"{source_id}/{mask_path.name}",
                                "public_dicom_seg",
                            ),
                        ],
                    )
                )
                endpoints.append(
                    SurvivalEndpointRecord(
                        patient_id=patient_id,
                        cohort_id="hcc_tace_seg",
                        duration_days=float(row["OS"]) * 7.0,
                        event=int(row["Death_1_StillAliveorLostToFU_0"]),
                        source_ref=_source("hcc_tace_seg", clinical_path.name, "outcome_xlsx"),
                    )
                )
            except (ValueError, TypeError, KeyError, OSError) as exc:
                exclusions.append(
                    CohortExclusion(
                        cohort_id="hcc_tace_seg",
                        source_patient_sha256=source_hash,
                        patient_id=patient_id,
                        reason_codes=[str(exc).split(":", maxsplit=1)[0]],
                        messages=[str(exc)],
                    )
                )
        return records, endpoints, exclusions

    def _prepared_image_and_mask(
        self, source_id: str, prepared: dict[str, Any]
    ) -> tuple[Path, Path]:
        """Resolve both fresh and resume-mode public cohort manifests."""
        if prepared.get("image_file") and prepared.get("mask_file"):
            return Path(str(prepared["image_file"])), Path(str(prepared["mask_file"]))
        converted = self.root / "prepared" / source_id / "converted"
        images = sorted(converted.glob("*_ct.nii.gz"))
        masks = sorted(converted.glob("*_tumor_mask.nii.gz"))
        if len(images) != 1 or len(masks) != 1:
            raise ValueError("HCC_PREPARED_NIFTI_NOT_UNIQUE")
        return images[0], masks[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def _cohort_subset(
    artifact: PublicPrognosisCohortArtifact,
    cohort_id: Literal["waw_tace", "hcc_tace_seg"],
) -> PublicPrognosisCohortArtifact:
    features = [
        record for record in artifact.feature_records if record.cohort_id == cohort_id
    ]
    endpoints = [
        record for record in artifact.endpoint_records if record.cohort_id == cohort_id
    ]
    exclusions = [
        record for record in artifact.exclusions if record.cohort_id == cohort_id
    ]
    return PublicPrognosisCohortArtifact(
        feature_records=features,
        endpoint_records=endpoints,
        exclusions=exclusions,
        counts={"included": len(features), "excluded": len(exclusions)},
        quality=artifact.quality,
        sources=[
            source
            for source in artifact.sources
            if source.details.get("cohort") == cohort_id
        ],
    )


def _plot_cohort_flow(
    artifact: PublicPrognosisCohortArtifact, target: Path
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional research dependency
        return None
    included = {
        cohort: sum(item.cohort_id == cohort for item in artifact.feature_records)
        for cohort in ("waw_tace", "hcc_tace_seg")
    }
    excluded = {
        cohort: sum(item.cohort_id == cohort for item in artifact.exclusions)
        for cohort in ("waw_tace", "hcc_tace_seg")
    }
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    boxes = [
        (0.05, 0.68, "WAW-TACE\nDevelopment source: 233"),
        (
            0.38,
            0.68,
            f"Eligible WAW: {included['waw_tace']}\nExcluded: {excluded['waw_tace']}",
        ),
        (0.05, 0.20, "HCC-TACE-Seg\nExternal source: 105"),
        (
            0.38,
            0.20,
            (
                "Prepared external: "
                f"{included['hcc_tace_seg']}\n"
                f"Excluded/not prepared: {excluded['hcc_tace_seg']}"
            ),
        ),
        (0.73, 0.44, "Frozen models\nOne-time external validation"),
    ]
    for x_position, y_position, label in boxes:
        axis.text(
            x_position,
            y_position,
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#eaf2f8"},
        )
    for start, end in (
        ((0.17, 0.68), (0.29, 0.68)),
        ((0.17, 0.20), (0.29, 0.20)),
        ((0.50, 0.68), (0.66, 0.49)),
        ((0.50, 0.20), (0.66, 0.39)),
    ):
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#34495e", "lw": 1.5},
        )
    axis.set_title("Prespecified public-cohort flow", fontsize=14)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(target)


def _plot_feature_distributions(
    artifact: PublicPrognosisCohortArtifact, target: Path
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional research dependency
        return None
    specifications = [
        ("age_years", "Age (years)", False),
        ("female", "Female indicator", False),
        ("afp_ng_ml", "log1p AFP", True),
        ("lesion_count", "log1p lesion count", True),
        ("total_tumor_volume_ml", "log1p total volume", True),
        ("max_lesion_extent_mm", "log1p maximum extent", True),
        ("largest_lesion_sphericity", "Largest-lesion sphericity", False),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    flattened = axes.ravel()
    cohorts = (("waw_tace", "WAW-TACE"), ("hcc_tace_seg", "HCC-TACE-Seg"))
    for axis, (field, title, use_log) in zip(flattened, specifications):
        for cohort_id, label in cohorts:
            values = np.asarray(
                [
                    float(getattr(record, field))
                    for record in artifact.feature_records
                    if record.cohort_id == cohort_id
                ],
                dtype=float,
            )
            if not values.size:
                continue
            plotted = np.log1p(values) if use_log else values
            axis.hist(plotted, bins=12, alpha=0.55, label=label)
        axis.set_title(title)
        axis.set_ylabel("Patients")
        if axis.has_data():
            axis.legend(fontsize=8)
    flattened[-1].axis("off")
    figure.suptitle("Prespecified baseline feature distributions")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(target)


def build_prognosis_cohort(
    data_root: str | Path,
    output_dir: str | Path,
) -> Path:
    data = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_records: list[PublicPrognosisRecord] = []
    endpoint_records: list[SurvivalEndpointRecord] = []
    exclusions: list[CohortExclusion] = []
    for adapter in (
        WawTaceAdapter(data / "waw-tace"),
        HccTaceSegAdapter(data / "hcc-tace-seg"),
    ):
        features, endpoints, rejected = adapter.build()
        feature_records.extend(features)
        endpoint_records.extend(endpoints)
        exclusions.extend(rejected)
    expected = {"waw_tace": 233, "hcc_tace_seg": 105}
    included = {
        cohort: sum(record.cohort_id == cohort for record in feature_records)
        for cohort in expected
    }
    excluded = {
        cohort: sum(record.cohort_id == cohort for record in exclusions)
        for cohort in expected
    }
    warnings = [
        f"{cohort}: included {included[cohort]} of expected {expected[cohort]}"
        for cohort in expected
        if included[cohort] != expected[cohort]
    ]
    checks = [
        QualityCheck(
            check_id=f"{cohort.upper()}_EXPECTED_COUNT",
            status="pass" if included[cohort] == expected[cohort] else "warning",
            message=f"included={included[cohort]} expected={expected[cohort]}",
        )
        for cohort in expected
    ]
    artifact = PublicPrognosisCohortArtifact(
        feature_records=feature_records,
        endpoint_records=endpoint_records,
        exclusions=exclusions,
        counts={
            **{f"{cohort}_included": count for cohort, count in included.items()},
            **{f"{cohort}_excluded": count for cohort, count in excluded.items()},
            "total_included": len(feature_records),
            "total_excluded": len(exclusions),
        },
        quality=QualityEvidence(
            status="warning" if warnings else "pass",
            checks=checks,
            warnings=warnings,
        ),
        sources=[
            _source("waw_tace", WAW_DOI, "public_dataset"),
            _source("hcc_tace_seg", HCC_TACE_SEG_DOI, "public_dataset"),
        ],
    )
    artifact_path = output / "prognosis-cohort.json"
    artifact.write_json(artifact_path)
    development = _cohort_subset(artifact, "waw_tace")
    external = _cohort_subset(artifact, "hcc_tace_seg")
    development.write_json(output / "development-cohort.json")
    external.write_json(output / "external-test-cohort.json")
    figures = {
        "cohort_flow": _plot_cohort_flow(
            artifact, output / "figures" / "cohort-flow.png"
        ),
        "feature_distributions": _plot_feature_distributions(
            artifact, output / "figures" / "feature-distributions.png"
        ),
    }
    _write_json(
        output / "figure-manifest.json",
        {key: value for key, value in figures.items() if value is not None},
    )
    _write_csv(
        output / "prognosis-cohort.csv",
        [record.to_dict() for record in feature_records],
    )
    _write_csv(
        output / "survival-endpoints.csv",
        [record.to_dict() for record in endpoint_records],
    )
    _write_csv(
        output / "development-features.csv",
        [record.to_dict() for record in development.feature_records],
    )
    _write_csv(
        output / "development-endpoints.csv",
        [record.to_dict() for record in development.endpoint_records],
    )
    _write_csv(
        output / "external-test-features.csv",
        [record.to_dict() for record in external.feature_records],
    )
    _write_csv(
        output / "external-test-endpoints.csv",
        [record.to_dict() for record in external.endpoint_records],
    )
    _write_json(output / "exclusion-manifest.json", [item.to_dict() for item in exclusions])
    _write_json(output / "cohort-qc.json", {"counts": artifact.counts, "quality": artifact.quality.to_dict()})
    _write_json(
        output / "source-unit-audit.json",
        {
            "waw_tace": {
                "field": "lab_albumin",
                "declared_unit": "g/L",
                "operational_unit": "g/dL",
                "dataset_version": WAW_VERSION,
                "reason_code": "SOURCE_UNIT_OVERRIDE",
                "sensitivity_analysis_required": True,
            }
        },
    )
    _write_json(
        output / "feature-dictionary.json",
        {
            "outcome_file": "survival-endpoints.csv",
            "development_artifact": "development-cohort.json",
            "locked_external_artifact": "external-test-cohort.json",
            "outcomes_forbidden_in_feature_file": [
                "event",
                "duration_days",
                "death",
                "survival_time",
                "progression",
                "tace_number",
                "treatment_response",
            ],
            "primary_features": [
                "age_years",
                "female",
                "afp_ng_ml",
                "lesion_count",
                "total_tumor_volume_ml",
                "max_lesion_extent_mm",
                "largest_lesion_sphericity",
            ],
            "minimum_connected_component_volume_ml": 0.01,
            "exploratory_portal_features": [
                "portal_mean_hu",
                "portal_std_hu",
                "portal_p10_hu",
                "portal_p90_hu",
            ],
        },
    )
    manifests = []
    for folder in (data / "waw-tace", data / "hcc-tace-seg"):
        path = folder / "dataset-manifest.json"
        if path.is_file():
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
    _write_json(output / "dataset-manifest.json", {"datasets": manifests})
    return artifact_path


__all__ = [
    "HccTaceSegAdapter",
    "WawTaceAdapter",
    "build_prognosis_cohort",
    "sync_prognosis_data",
]
