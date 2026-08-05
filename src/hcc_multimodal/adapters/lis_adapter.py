"""Demo-grade LIS (Laboratory Information System) adapter.

Simulates the data-flow role of a real LIS/HL7/FHIR interface at near-zero
cost:

* **File drop** -- watch (or one-shot scan) a directory for LIS export files
  (JSON/CSV/TXT) and ingest each new file as it appears.
* **REST endpoint** -- a dependency-free ``http.server`` based demo endpoint
  ``POST /api/v1/lis/ingest`` that accepts a JSON laboratory payload.

Every ingestion runs the standard pipeline -- parse via
:mod:`hcc_multimodal.clinical_labs`, then GALAD-style scoring via
:mod:`hcc_multimodal.galad` -- and returns a normalized payload ready for
downstream fusion.  Patient metadata (age, sex) may be supplied explicitly or
carried in the payload's ``patientInfo`` block.

This adapter is a research demonstration shim; it is NOT a validated HL7/FHIR
interface and must not be connected to real clinical systems.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..clinical_labs import parse_laboratory_report
from ..galad import GALADResult, calculate_galad_score

SUPPORTED_SUFFIXES = (".json", ".csv", ".txt")
MAX_BODY_BYTES = 1_000_000


class LISAdapter:
    """Ingest LIS-style exports and emit parsed labs + GALAD-style payloads."""

    def __init__(
        self,
        *,
        default_age_years: float | None = None,
        default_sex: Any = None,
        coefficients_path: str | Path | None = None,
    ) -> None:
        self.default_age_years = default_age_years
        self.default_sex = default_sex
        self.coefficients_path = coefficients_path
        self._seen: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------ core
    def ingest_file(
        self,
        path: str | Path,
        *,
        patient_id: str,
        age_years: float | None = None,
        sex: Any = None,
    ) -> dict[str, Any]:
        """Parse one LIS export file and return the normalized payload."""
        source = Path(path)
        labs = parse_laboratory_report(source, patient_id=patient_id)
        galad = calculate_galad_score(
            labs,
            age_years=age_years if age_years is not None else self.default_age_years,
            sex=sex if sex is not None else self.default_sex,
            coefficients_path=self.coefficients_path,
        )
        return self._payload(labs, galad, transport="file", source=source.name)

    def ingest_payload(
        self,
        payload: dict[str, Any],
        *,
        patient_id: str | None = None,
        age_years: float | None = None,
        sex: Any = None,
    ) -> dict[str, Any]:
        """Ingest an in-memory LIS JSON payload (e.g. from the REST endpoint).

        Recognizes an optional ``patientInfo`` block with ``patientId``,
        ``age``/``patientAge`` and ``gender``/``sex`` fields; explicit keyword
        arguments take precedence.
        """
        info = payload.get("patientInfo") if isinstance(payload, dict) else None
        if not isinstance(info, dict):
            info = {}
        resolved_id = patient_id or str(info.get("patientId") or "LIS_UNKNOWN")
        resolved_age = age_years
        if resolved_age is None:
            raw_age = info.get("age", info.get("patientAge"))
            if raw_age is not None:
                try:
                    resolved_age = float(raw_age)
                except (TypeError, ValueError):
                    resolved_age = None
        if resolved_age is None:
            resolved_age = self.default_age_years
        resolved_sex = sex if sex is not None else info.get("gender", info.get("sex"))
        if resolved_sex is None:
            resolved_sex = self.default_sex

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            temp_path = Path(handle.name)
        try:
            labs = parse_laboratory_report(temp_path, patient_id=resolved_id)
        finally:
            temp_path.unlink(missing_ok=True)
        galad = calculate_galad_score(
            labs,
            age_years=resolved_age,
            sex=resolved_sex,
            coefficients_path=self.coefficients_path,
        )
        return self._payload(labs, galad, transport="payload", source="inline")

    @staticmethod
    def _payload(labs: Any, galad: GALADResult, *, transport: str, source: str) -> dict[str, Any]:
        return {
            "patient_id": labs.patient_id,
            "transport": transport,
            "source": source,
            "lab_evidence": labs.to_dict(),
            "galad": galad.to_dict(),
            "provenance": {
                "adapter": "lis-adapter-demo-v1",
                "note": (
                    "Demo LIS shim; not a validated HL7/FHIR interface. "
                    "GALAD-style score uses illustrative uncalibrated coefficients."
                ),
            },
        }

    # ----------------------------------------------------------- file watch
    def scan_directory(
        self,
        directory: str | Path,
        *,
        patient_id: str = "LIS_UNKNOWN",
        pattern: str = "*.json",
    ) -> list[dict[str, Any]]:
        """One-shot ingest of every supported file currently in a directory."""
        root = Path(directory)
        results = []
        for path in sorted(root.glob(pattern)):
            if path.suffix.casefold() in SUPPORTED_SUFFIXES:
                results.append(self.ingest_file(path, patient_id=patient_id))
                self._seen.add((str(path), path.stat().st_mtime_ns))
        return results

    def watch_directory(
        self,
        directory: str | Path,
        *,
        patient_id: str = "LIS_UNKNOWN",
        pattern: str = "*.json",
        interval_seconds: float = 2.0,
        max_cycles: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield a payload for each newly appearing/changed file.

        ``max_cycles`` bounds the polling loop (useful for tests and demos);
        ``None`` polls forever.
        """
        root = Path(directory)
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            emitted = False
            for path in sorted(root.glob(pattern)):
                if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                    continue
                stamp = (str(path), path.stat().st_mtime_ns)
                if stamp in self._seen:
                    continue
                emitted = True
                payload = self.ingest_file(path, patient_id=patient_id)
                self._seen.add(stamp)
                yield payload
            if not emitted:
                time.sleep(interval_seconds)

    # -------------------------------------------------------------- REST demo
    def create_request_handler(self) -> type[BaseHTTPRequestHandler]:
        """Build a ``BaseHTTPRequestHandler`` subclass bound to this adapter."""
        adapter = self

        class LISRequestHandler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, body: dict[str, Any]) -> None:
                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/api/v1/lis/ingest":
                    self._send_json(404, {"error": "unknown endpoint"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length < 0 or length > MAX_BODY_BYTES:
                        self._send_json(413, {"error": "request body too large"})
                        return
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send_json(400, {"error": "request body must be valid JSON"})
                    return
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": "request body must be a JSON object"})
                    return
                try:
                    result = adapter.ingest_payload(
                        payload,
                        patient_id=payload.get("patient_id"),
                        age_years=payload.get("age_years"),
                        sex=payload.get("sex"),
                    )
                except Exception as exc:  # noqa: BLE001 -- fail closed, never leak internals
                    self._send_json(422, {"error": f"ingestion failed: {type(exc).__name__}"})
                    return
                self._send_json(200, result)

            def log_message(self, format: str, *args: Any) -> None:  # keep demo quiet
                return

        return LISRequestHandler

    def serve(self, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
        """Start the demo REST endpoint (blocking); returns the server instance.

        ``POST /api/v1/lis/ingest`` accepts a LIS JSON payload and returns the
        parsed laboratory evidence plus the GALAD-style result.
        """
        server = ThreadingHTTPServer((host, port), self.create_request_handler())
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return server


__all__ = ["SUPPORTED_SUFFIXES", "LISAdapter"]
