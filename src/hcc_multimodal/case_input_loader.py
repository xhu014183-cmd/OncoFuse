"""Unified case-input JSON loader for the three-line pipeline.

The loader reads a single JSON envelope and materializes inline laboratory/HPI
content into temporary files so that the existing file-based parsers do not need
to change. All file paths in the envelope may be relative to the envelope file.
"""

from __future__ import annotations

import json
import tempfile
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
        if self.payload.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported case_input schema_version")

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
    def dicom_dir(self) -> Path:
        resolved = self._resolve(self.payload["imaging"]["dicom_dir"])
        if resolved is None:
            raise ValueError("imaging.dicom_dir is required")
        return resolved

    @property
    def seg_path(self) -> Path | None:
        return self._resolve(self.payload["imaging"].get("seg"))

    @property
    def phase(self) -> str | None:
        return self.payload["imaging"].get("phase")

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
            resolved = self._resolve(lab["file_path"])
            if resolved is None:
                raise ValueError("laboratory.file_path is required when source_type=file")
            return resolved
        return self._materialize_inline("labs", lab["inline"])

    @property
    def hpi_path(self) -> Path | None:
        hpi = self.payload.get("hpi")
        if hpi is None:
            return None
        if hpi["source_type"] == "file":
            return self._resolve(hpi["file_path"])
        return self._materialize_inline("hpi", hpi["inline"])

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
