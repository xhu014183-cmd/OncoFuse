from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hcc_multimodal.adapters.lis_adapter import LISAdapter


def _lis_payload(patient_id: str = "LIS_001") -> dict:
    return {
        "patientInfo": {"patientId": patient_id, "age": 62, "gender": "男"},
        "testResultInfos": [
            {"inspectionDate": "2026-07-01", "systemTestItemName": "AFP", "result": "400", "unit": "ng/mL", "referenceRange": "0-7"},
            {"inspectionDate": "2026-07-01", "systemTestItemName": "AFP-L3%", "result": "15", "unit": "%", "referenceRange": "0-10"},
            {"inspectionDate": "2026-07-01", "systemTestItemName": "PIVKA-II", "result": "1000", "unit": "mAU/mL", "referenceRange": "0-40"},
        ],
    }


def _serve(handler):
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_ingest_file_produces_galad_payload(tmp_path: Path):
    path = tmp_path / "lis_export.json"
    path.write_text(json.dumps(_lis_payload()), encoding="utf-8")
    adapter = LISAdapter()
    result = adapter.ingest_file(path, patient_id="LIS_001", age_years=62, sex="男")
    assert result["transport"] == "file"
    assert result["galad"]["status"] == "calculated"
    assert result["galad"]["risk_tier"] == "HIGH"
    assert "AFP" in result["lab_evidence"]["analytes"]


def test_ingest_payload_reads_patient_info():
    adapter = LISAdapter()
    result = adapter.ingest_payload(_lis_payload("LIS_002"))
    assert result["patient_id"] == "LIS_002"
    assert result["galad"]["status"] == "calculated"
    assert result["galad"]["inputs"]["age_years"] == 62
    assert result["galad"]["inputs"]["sex"] == "male"


def test_explicit_metadata_overrides_patient_info():
    adapter = LISAdapter()
    result = adapter.ingest_payload(
        _lis_payload("LIS_003"), patient_id="OVERRIDE", age_years=45, sex="female"
    )
    assert result["patient_id"] == "OVERRIDE"
    assert result["galad"]["inputs"]["sex"] == "female"
    assert result["galad"]["inputs"]["age_years"] == 45


def test_missing_metadata_yields_incomplete_galad():
    payload = _lis_payload("LIS_004")
    payload["patientInfo"] = {"patientId": "LIS_004"}
    adapter = LISAdapter()
    result = adapter.ingest_payload(payload)
    assert result["galad"]["status"] == "incomplete"
    assert "age_years" in result["galad"]["missing_inputs"]
    assert "sex" in result["galad"]["missing_inputs"]


def test_scan_directory_ingests_supported_files(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps(_lis_payload("LIS_A")), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_lis_payload("LIS_B")), encoding="utf-8")
    (tmp_path / "ignore.md").write_text("not a lab file", encoding="utf-8")
    adapter = LISAdapter(default_age_years=60, default_sex="male")
    results = adapter.scan_directory(tmp_path)
    assert len(results) == 2
    assert all(item["galad"]["status"] == "calculated" for item in results)


def test_watch_directory_picks_up_new_file(tmp_path: Path):
    adapter = LISAdapter(default_age_years=60, default_sex="male")
    watcher = adapter.watch_directory(tmp_path, interval_seconds=0.01, max_cycles=5)
    (tmp_path / "new.json").write_text(json.dumps(_lis_payload("LIS_WATCH")), encoding="utf-8")
    first = next(watcher)
    assert first["galad"]["status"] == "calculated"
    # the same file is not re-ingested on later cycles
    assert next(watcher, None) is None


def test_rest_endpoint_round_trip():
    adapter = LISAdapter()
    server, port = _serve(adapter.create_request_handler())
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/lis/ingest",
            data=json.dumps(_lis_payload("LIS_REST")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert body["patient_id"] == "LIS_REST"
        assert body["galad"]["risk_tier"] == "HIGH"
    finally:
        server.shutdown()
        server.server_close()


def test_rest_unknown_endpoint_returns_404():
    adapter = LISAdapter()
    server, port = _serve(adapter.create_request_handler())
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/other",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_rest_non_object_body_returns_400():
    adapter = LISAdapter()
    server, port = _serve(adapter.create_request_handler())
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/lis/ingest",
            data=b"[1, 2, 3]",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
