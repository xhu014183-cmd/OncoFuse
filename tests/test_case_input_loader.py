from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcc_multimodal.case_input_loader import CaseInputLoader


def _write_case(tmp_path: Path, **overrides) -> Path:
    """Write a minimal valid case-input envelope and return its path."""
    payload = {
        "schema_version": "1.0.0",
        "case_id": "CASE_001",
        "patient_id": "DEMO_001",
        "index_date": "2026-07-20",
        "imaging": {
            "dicom_dir": "study",
            "seg": "seg.dcm",
            "phase": "portal_venous",
            "image_evidence": "image_evidence.json",
        },
        "laboratory": {
            "source_type": "file",
            "file_path": "labs.txt",
        },
        "hpi": {
            "source_type": "inline",
            "inline": {"format": "text", "content": "2026-01-15 行 TACE。"},
        },
        "output_dir": "case-output",
    }
    payload.update(overrides)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_case_and_resolves_relative_paths(tmp_path: Path):
    loader = CaseInputLoader(_write_case(tmp_path))
    assert loader.case_id == "CASE_001"
    assert loader.patient_id == "DEMO_001"
    assert loader.index_date == "2026-07-20"
    assert loader.phase == "portal_venous"
    assert loader.dicom_dir == (tmp_path / "study").resolve()
    assert loader.seg_path == (tmp_path / "seg.dcm").resolve()
    assert loader.labs_path == (tmp_path / "labs.txt").resolve()
    assert loader.image_evidence_path == (tmp_path / "image_evidence.json").resolve()
    assert loader.output_dir == (tmp_path / "case-output").resolve()


def test_inline_labs_and_hpi_are_materialized(tmp_path: Path):
    payload = {
        "schema_version": "1.0.0",
        "case_id": "CASE_002",
        "patient_id": "DEMO_002",
        "imaging": {"dicom_dir": "study"},
        "laboratory": {
            "source_type": "inline",
            "inline": {"format": "json", "content": '{"observations": []}'},
        },
        "hpi": {
            "source_type": "inline",
            "inline": {"format": "text", "content": "event"},
        },
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loader = CaseInputLoader(path)
    assert loader.labs_path.name == "labs.json"
    assert loader.labs_path.read_text(encoding="utf-8") == '{"observations": []}'
    assert loader.hpi_path is not None
    assert loader.hpi_path.name == "hpi.txt"
    assert loader.hpi_path.read_text(encoding="utf-8") == "event"


def test_missing_required_field_raises(tmp_path: Path):
    path = _write_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["case_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case_id"):
        CaseInputLoader(path)


def test_unsupported_schema_version_raises(tmp_path: Path):
    path = _write_case(tmp_path, schema_version="0.9.0")
    with pytest.raises(ValueError, match="schema_version"):
        CaseInputLoader(path)


def test_v11_requires_clinical_task(tmp_path: Path):
    path = _write_case(tmp_path, schema_version="1.1.0")
    with pytest.raises(ValueError, match="clinical_task"):
        CaseInputLoader(path)


def test_v11_exposes_task_and_context(tmp_path: Path):
    path = _write_case(
        tmp_path,
        schema_version="1.1.0",
        clinical_task="recurrence_surveillance",
        demographics={"age_years": 58, "sex_at_birth": "male"},
        clinical_context={"risk_factors": ["chronic_HBV"]},
    )
    loader = CaseInputLoader(path)
    assert loader.clinical_task == "recurrence_surveillance"
    assert loader.demographics["age_years"] == 58
    assert loader.clinical_context["risk_factors"] == ["chronic_HBV"]


def test_invalid_index_date_raises(tmp_path: Path):
    path = _write_case(tmp_path, index_date="2026/07/20")
    with pytest.raises(ValueError, match="index_date"):
        CaseInputLoader(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CaseInputLoader(tmp_path / "nope.json")


def test_image_evidence_dict_is_materialized(tmp_path: Path):
    path = _write_case(
        tmp_path,
        imaging={"dicom_dir": "study", "image_evidence": {"lesion_count": 2}},
    )
    loader = CaseInputLoader(path)
    assert loader.image_evidence_path is not None
    assert loader.image_evidence_path.name == "image_evidence.json"
    data = json.loads(loader.image_evidence_path.read_text(encoding="utf-8"))
    assert data["lesion_count"] == 2


def test_labs_file_without_path_raises(tmp_path: Path):
    path = _write_case(tmp_path, laboratory={"source_type": "file"})
    loader = CaseInputLoader(path)
    with pytest.raises(ValueError, match="file_path"):
        _ = loader.labs_path


def test_dicom_dir_required_raises(tmp_path: Path):
    path = _write_case(tmp_path, imaging={})
    loader = CaseInputLoader(path)
    with pytest.raises(ValueError, match="dicom_dir"):
        _ = loader.dicom_dir


def test_llm_response_string_is_parsed(tmp_path: Path):
    path = _write_case(tmp_path, llm_response='{"state": "ok"}')
    assert CaseInputLoader(path).llm_response == {"state": "ok"}


def test_llm_response_dict_passthrough(tmp_path: Path):
    path = _write_case(tmp_path, llm_response={"state": "ok"})
    assert CaseInputLoader(path).llm_response == {"state": "ok"}


def test_llm_response_defaults_to_none(tmp_path: Path):
    path = _write_case(tmp_path)
    assert CaseInputLoader(path).llm_response is None


def test_output_dir_defaults(tmp_path: Path):
    path = _write_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["output_dir"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert CaseInputLoader(path).output_dir == (tmp_path / "case-output").resolve()


def test_missing_hpi_returns_none(tmp_path: Path):
    path = _write_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["hpi"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert CaseInputLoader(path).hpi_path is None


def test_treatment_events_become_timeline_when_hpi_is_absent(tmp_path: Path):
    path = _write_case(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["hpi"]
    payload["clinical_context"] = {
        "treatment_events": [
            {"date": "2026-01-15", "type": "TACE", "description": "index treatment"}
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loader = CaseInputLoader(path)
    timeline_path = loader.hpi_path
    assert timeline_path is not None
    assert "2026-01-15 TACE index treatment" in timeline_path.read_text(encoding="utf-8")


def test_writes_normalized_manifest_without_binary_paths(tmp_path: Path):
    loader = CaseInputLoader(_write_case(tmp_path))
    target = loader.write_normalized_manifest(tmp_path / "out")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    assert manifest["case_id"] == "CASE_001"
    assert manifest["source_inventory"]["dicom_supplied"] is True
    assert "dicom_dir" not in manifest["source_inventory"]


def test_missing_seg_returns_none(tmp_path: Path):
    path = _write_case(tmp_path, imaging={"dicom_dir": "study"})
    assert CaseInputLoader(path).seg_path is None


def _v12_payload() -> dict:
    return {
        "schema_version": "1.2.0",
        "case_id": "CASE_120",
        "patient_id": "RESEARCH_120",
        "clinical_task": "recurrence_surveillance",
        "index_date": "2026-07-20",
        "data_relationship": {
            "imaging_origin": "real_public",
            "laboratory_origin": "synthetic",
            "pairing_status": "unpaired_poc_composite",
            "statement": "public image and synthetic labs for demonstration",
        },
        "imaging": {
            "source_type": "nifti",
            "nifti_image": "ct.nii.gz",
            "dicom_dir": None,
            "seg": "mask.nii.gz",
            "seg_role": "public_reference",
            "modality": "CT",
            "study_date": "2026-07-20",
            "phase": "portal_venous",
        },
        "laboratory": {
            "source_type": "inline",
            "inline": {"format": "text", "content": "2026-07-20 AFP 8 ng/mL 0-7"},
        },
    }


def test_v12_nifti_contract_and_relationship(tmp_path: Path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(_v12_payload()), encoding="utf-8")
    loader = CaseInputLoader(path)
    assert loader.imaging_source_type == "nifti"
    assert loader.nifti_image_path == (tmp_path / "ct.nii.gz").resolve()
    assert loader.seg_role == "public_reference"
    assert loader.data_relationship["pairing_status"] == "unpaired_poc_composite"


@pytest.mark.parametrize(
    "imaging",
    [
        {
            "source_type": "dicom",
            "dicom_dir": None,
            "nifti_image": "ct.nii.gz",
        },
        {
            "source_type": "nifti",
            "dicom_dir": "study",
            "nifti_image": "ct.nii.gz",
        },
    ],
)
def test_v12_rejects_invalid_dicom_nifti_selection(tmp_path: Path, imaging: dict):
    payload = _v12_payload()
    payload["imaging"] = imaging
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        CaseInputLoader(path)


def test_v12_rejects_seg_without_valid_role(tmp_path: Path):
    payload = _v12_payload()
    payload["imaging"]["seg_role"] = "radiologist"
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seg_role"):
        CaseInputLoader(path)


def test_v12_public_image_synthetic_labs_must_be_unpaired(tmp_path: Path):
    payload = _v12_payload()
    payload["data_relationship"]["pairing_status"] = "same_subject"
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unpaired_poc_composite"):
        CaseInputLoader(path)


def test_legacy_relationship_is_unverified(tmp_path: Path):
    relationship = CaseInputLoader(_write_case(tmp_path)).data_relationship
    assert relationship["pairing_status"] == "user_supplied_unverified"
