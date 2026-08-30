"""Unified case-input JSON loader for the three-line pipeline.

The loader reads a single JSON envelope and materializes inline laboratory/HPI
content into temporary files so that the existing file-based parsers do not need
to change. All file paths in the envelope may be relative to the envelope file.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


class CaseInputLoader:
    """Load a unified case-input JSON and materialize any inline content as temp files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Case input file not found: {self.path}")
        self.payload: dict[str, Any] = json.loads(
            self.path.read_text(encoding="utf-8-sig")
        )
        self._validate_minimal()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="hcc-case-input-")
        self.temp_root = Path(self._temp_dir.name)

    def _validate_minimal(self) -> None:
        required = {"schema_version", "case_id", "patient_id", "imaging", "laboratory"}
        missing = required - set(self.payload)
        if missing:
            raise ValueError(f"Case input missing required fields: {sorted(missing)}")
        version = self.payload.get("schema_version")
        if version not in {"1.0.0", "1.1.0", "1.2.0"}:
            raise ValueError("Unsupported case_input schema_version")
        if not str(self.payload.get("case_id", "")).strip():
            raise ValueError("case_id must be a non-empty case tracking identifier")
        if not str(self.payload.get("patient_id", "")).strip():
            raise ValueError("patient_id must be a non-empty research pseudonym")
        if not isinstance(self.payload.get("imaging"), dict):
            raise TypeError("imaging must be an object")
        if not isinstance(self.payload.get("laboratory"), dict):
            raise TypeError("laboratory must be an object")
        if version in {"1.1.0", "1.2.0"} and not self.payload.get("clinical_task"):
            raise ValueError(
                "clinical_task is required for case_input schema_version 1.1.0/1.2.0"
            )
        task = self.payload.get("clinical_task")
        allowed_tasks = {
            None,
            "screening",
            "diagnostic_workup",
            "treatment_baseline",
            "post_treatment_response",
            "recurrence_surveillance",
            "unspecified",
        }
        if task not in allowed_tasks:
            raise ValueError(f"Unsupported clinical_task: {task!r}")
        if self.payload.get("index_date"):
            try:
                date.fromisoformat(str(self.payload["index_date"]))
            except ValueError as exc:
                raise ValueError("index_date must use ISO-8601 YYYY-MM-DD format") from exc
        imaging = self.payload["imaging"]
        if imaging.get("modality", "CT") not in {"CT", "MR"}:
            raise ValueError("imaging.modality must be CT or MR")
        if any(
            isinstance(imaging.get(field), list) and len(imaging[field]) > 1
            for field in ("phases", "exams", "studies")
        ):
            raise ValueError("Multiple CT examinations/phases are not supported in 1.2")
        if version == "1.2.0":
            source_type = imaging.get("source_type")
            if source_type not in {"dicom", "nifti"}:
                raise ValueError("imaging.source_type must be 'dicom' or 'nifti'")
            has_dicom = bool(imaging.get("dicom_dir"))
            has_nifti = bool(imaging.get("nifti_image"))
            if source_type == "dicom" and (not has_dicom or has_nifti):
                raise ValueError(
                    "source_type=dicom requires dicom_dir and forbids nifti_image"
                )
            if source_type == "nifti" and (not has_nifti or has_dicom):
                raise ValueError(
                    "source_type=nifti requires nifti_image and forbids dicom_dir"
                )
            seg_roles = {
                "expert_reference",
                "public_reference",
                "user_supplied",
                "model_prediction",
            }
            if imaging.get("seg") and imaging.get("seg_role") not in seg_roles:
                raise ValueError("imaging.seg_role is required when seg is supplied")
            relationship = self.payload.get("data_relationship")
            if not isinstance(relationship, dict):
                raise TypeError("data_relationship is required and must be an object")
            required_relationship = {
                "imaging_origin",
                "laboratory_origin",
                "pairing_status",
                "statement",
            }
            missing_relationship = required_relationship - set(relationship)
            if missing_relationship:
                raise ValueError(
                    "data_relationship missing fields: "
                    f"{sorted(missing_relationship)}"
                )
            origins = {
                "real_clinical",
                "real_public",
                "synthetic",
                "user_supplied",
                "unknown",
            }
            pairing_statuses = {
                "same_subject",
                "unpaired_poc_composite",
                "user_supplied_unverified",
            }
            if relationship.get("imaging_origin") not in origins:
                raise ValueError("Unsupported data_relationship.imaging_origin")
            if relationship.get("laboratory_origin") not in origins:
                raise ValueError("Unsupported data_relationship.laboratory_origin")
            if relationship.get("pairing_status") not in pairing_statuses:
                raise ValueError("Unsupported data_relationship.pairing_status")
            if not str(relationship.get("statement") or "").strip():
                raise ValueError("data_relationship.statement must be non-empty")
            if (
                relationship.get("imaging_origin") == "real_public"
                and relationship.get("laboratory_origin") == "synthetic"
                and relationship.get("pairing_status") != "unpaired_poc_composite"
            ):
                raise ValueError(
                    "real_public imaging plus synthetic laboratory evidence must use "
                    "pairing_status=unpaired_poc_composite"
                )
        study_date = imaging.get("study_date")
        if study_date:
            try:
                date.fromisoformat(str(study_date))
            except ValueError as exc:
                raise ValueError("imaging.study_date must use ISO-8601 YYYY-MM-DD") from exc

    def _resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        target = Path(value)
        if target.is_absolute():
            return target
        return (self.path.parent / target).resolve()

    def _materialize_inline(self, key: str, inline: dict[str, str]) -> Path:
        ext = {"text": ".txt", "json": ".json", "csv": ".csv"}[inline["format"]]
        target = self.temp_root / f"{key}{ext}"
        target.write_text(inline["content"], encoding="utf-8")
        return target

    @property
    def case_id(self) -> str:
        return self.payload["case_id"]

    @property
    def patient_id(self) -> str:
        return self.payload["patient_id"]

    @property
    def index_date(self) -> str | None:
        return self.payload.get("index_date")

    @property
    def clinical_task(self) -> str:
        return str(self.payload.get("clinical_task") or "unspecified")

    @property
    def demographics(self) -> dict[str, Any]:
        return dict(self.payload.get("demographics") or {})

    @property
    def clinical_context(self) -> dict[str, Any]:
        return dict(self.payload.get("clinical_context") or {})

    @property
    def dicom_dir(self) -> Path:
        resolved = self._resolve(self.payload["imaging"].get("dicom_dir"))
        if resolved is None:
            raise ValueError("imaging.dicom_dir is required")
        return resolved

    @property
    def imaging_source_type(self) -> str:
        return str(self.payload["imaging"].get("source_type") or "dicom")

    @property
    def nifti_image_path(self) -> Path | None:
        return self._resolve(self.payload["imaging"].get("nifti_image"))

    @property
    def seg_path(self) -> Path | None:
        return self._resolve(self.payload["imaging"].get("seg"))

    @property
    def phase(self) -> str | None:
        return self.payload["imaging"].get("phase")

    @property
    def modality(self) -> str:
        return str(self.payload["imaging"].get("modality") or "CT")

    @property
    def study_date(self) -> str | None:
        return self.payload["imaging"].get("study_date") or self.index_date

    @property
    def seg_role(self) -> str | None:
        return self.payload["imaging"].get("seg_role")

    @property
    def data_relationship(self) -> dict[str, Any]:
        relationship = self.payload.get("data_relationship")
        if isinstance(relationship, dict):
            return dict(relationship)
        return {
            "imaging_origin": "unknown",
            "laboratory_origin": "unknown",
            "pairing_status": "user_supplied_unverified",
            "statement": (
                "Legacy case input did not declare whether imaging and laboratory "
                "evidence belong to the same subject"
            ),
        }

    @property
    def image_evidence_path(self) -> Path | None:
        value = self.payload["imaging"].get("image_evidence")
        if value is None:
            return None
        if isinstance(value, dict):
            target = self.temp_root / "image_evidence.json"
            target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            return target
        return self._resolve(value)

    @property
    def labs_path(self) -> Path:
        lab = self.payload["laboratory"]
        if lab["source_type"] == "file":
            resolved = self._resolve(lab.get("file_path"))
            if resolved is None:
                raise ValueError("laboratory.file_path is required when source_type=file")
            return resolved
        return self._materialize_inline("labs", lab["inline"])

    @property
    def hpi_path(self) -> Path | None:
        hpi = self.payload.get("hpi")
        if hpi is None:
            events = self.clinical_context.get("treatment_events") or []
            if not events:
                return None
            lines = [
                " ".join(
                    part
                    for part in (
                        str(event.get("date", "")).strip(),
                        str(event.get("type", "")).strip(),
                        str(event.get("description") or "").strip(),
                    )
                    if part
                )
                for event in events
                if isinstance(event, dict)
            ]
            if not lines:
                return None
            return self._materialize_inline(
                "clinical_context_timeline",
                {"format": "text", "content": "\n".join(lines)},
            )
        if hpi["source_type"] == "file":
            return self._resolve(hpi["file_path"])
        return self._materialize_inline("hpi", hpi["inline"])

    def write_normalized_manifest(self, output_dir: str | Path | None = None) -> Path:
        """Persist the non-binary intake contract used for this run."""
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": self.payload["schema_version"],
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "clinical_task": self.clinical_task,
            "index_date": self.index_date,
            "demographics": self.demographics,
            "clinical_context": self.clinical_context,
            "data_relationship": self.data_relationship,
            "source_inventory": {
                "imaging_source_type": self.imaging_source_type,
                "dicom_supplied": bool(self.payload["imaging"].get("dicom_dir")),
                "nifti_supplied": bool(self.payload["imaging"].get("nifti_image")),
                "seg_supplied": bool(self.payload["imaging"].get("seg")),
                "seg_role": self.seg_role,
                "structured_image_evidence_supplied": bool(
                    self.payload["imaging"].get("image_evidence")
                ),
                "laboratory_source_type": self.payload["laboratory"].get("source_type"),
                "hpi_supplied": bool(self.payload.get("hpi")),
                "treatment_events_supplied": bool(
                    self.clinical_context.get("treatment_events")
                ),
            },
        }
        target = target_dir / "case-input.normalized.json"
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @property
    def output_dir(self) -> Path:
        value = self.payload.get("output_dir", "case-output")
        resolved = self._resolve(value)
        if resolved is None:
            raise ValueError("output_dir cannot be empty")
        return resolved

    @property
    def llm_response(self) -> dict[str, Any] | None:
        value = self.payload.get("llm_response")
        if value is None:
            return None
        return json.loads(value) if isinstance(value, str) else value


__all__ = ["CaseInputLoader"]
