from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcc_multimodal.contracts import (
    LAB_FEATURE_NAMES,
    build_lab_feature_vector,
    build_multimodal_case_evidence,
    evidence_age_days,
)
from hcc_multimodal.imaging import measure_nifti
from hcc_multimodal.labs import load_lab_evidence
from hcc_multimodal.schemas import LabEvidence
from hcc_multimodal.synthetic import generate_synthetic_case
from hcc_multimodal.tcia import write_composite_labs


def _evidence(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    imaging = measure_nifti(
        paths["followup_image"],
        paths["followup_mask"],
        patient_id="DEMO_HCC_001",
        study_date="2026-07-15",
    )
    labs = load_lab_evidence(paths["labs"], index_time="2026-07-15")
    return paths, imaging, labs


def test_lab_feature_vector_has_stable_order_and_missing_mask(tmp_path: Path):
    _, _, labs = _evidence(tmp_path)
    vector = build_lab_feature_vector(labs, index_time="2026-07-15")
    assert tuple(vector.feature_names) == LAB_FEATURE_NAMES
    assert len(vector.values) == len(vector.availability_mask) == 14
    assert np.isfinite(vector.values).all()
    assert vector.values[vector.feature_names.index("AFP.latest_uln_ratio")] == pytest.approx(
        85.3 / 7.0
    )
    assert vector.values[vector.feature_names.index("DCP.missing")] == 0
    assert vector.marker_availability == {"AFP": True, "DCP": True}


def test_missing_marker_is_not_encoded_as_normal(tmp_path: Path):
    payload = {
        "patient_id": "P1",
        "observations": [
            {
                "date": "2026-07-15",
                "marker": "AFP",
                "value": 5,
                "unit": "ng/mL",
                "upper_reference": 7,
            }
        ],
    }
    path = tmp_path / "labs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    vector = build_lab_feature_vector(
        load_lab_evidence(path, index_time="2026-07-15"),
        index_time="2026-07-15",
    )
    missing_index = vector.feature_names.index("DCP.missing")
    ratio_index = vector.feature_names.index("DCP.latest_uln_ratio")
    assert vector.values[missing_index] == 1
    assert vector.availability_mask[missing_index] is True
    assert vector.values[ratio_index] == 0
    assert vector.availability_mask[ratio_index] is False


def test_temporal_context_rejects_future_evidence():
    assert evidence_age_days("2026-07-15", "2026-07-15") == 0
    assert evidence_age_days("2026-07-01", "2026-07-15") == 14
    with pytest.raises(ValueError, match="occurs after index time"):
        evidence_age_days("2026-07-16", "2026-07-15")
    with pytest.raises(ValueError, match="ISO date"):
        evidence_age_days("07/15/2026", "2026-07-15")


def test_lab_loader_excludes_future_observations(tmp_path: Path):
    payload = {
        "patient_id": "P1",
        "observations": [
            {"date": "2026-07-15", "marker": "AFP", "value": 5, "unit": "ng/mL"},
            {"date": "2026-08-15", "marker": "AFP", "value": 500, "unit": "ng/mL"},
        ],
    }
    path = tmp_path / "labs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    labs = load_lab_evidence(path, index_time="2026-07-15")
    assert labs.markers["AFP"].latest_value == 5
    assert any("Excluded future AFP" in warning for warning in labs.warnings)


def test_multimodal_case_preserves_unpaired_synthetic_truth(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    patient_id = "COMPOSITE_PUBLIC_HCC_003"
    imaging = measure_nifti(
        paths["followup_image"],
        paths["followup_mask"],
        patient_id=patient_id,
        study_date="1997-09-12",
    )
    labs_path = write_composite_labs(tmp_path / "labs.json", patient_id)
    labs = load_lab_evidence(labs_path, index_time="1997-09-12")
    case = build_multimodal_case_evidence(
        imaging=imaging,
        labs=labs,
        index_time="1997-09-12",
        pairing_status="unpaired_poc_composite",
        data_relationship="real public image plus unrelated synthetic labs",
        imaging_origin="real_public",
        lab_origin="synthetic",
    )
    assert case.modalities["imaging"].data_origin == "real_public"
    assert case.modalities["laboratory"].data_origin == "synthetic"
    assert case.pairing_status == "unpaired_poc_composite"
    assert any("not patient-level" in warning for warning in case.warnings)


def test_multimodal_case_rejects_patient_mismatch(tmp_path: Path):
    paths, imaging, _ = _evidence(tmp_path)
    payload = json.loads(paths["labs"].read_text(encoding="utf-8"))
    payload["patient_id"] = "OTHER"
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps(payload), encoding="utf-8")
    labs = load_lab_evidence(mismatch, index_time="2026-07-15")
    with pytest.raises(ValueError, match="does not match"):
        build_multimodal_case_evidence(
            imaging=imaging,
            labs=labs,
            index_time="2026-07-15",
            pairing_status="same_subject",
            data_relationship="synthetic same-subject case",
            imaging_origin="synthetic",
            lab_origin="synthetic",
        )


def test_public_contract_requires_exact_schema_version_and_timezone(tmp_path: Path):
    _, _, labs = _evidence(tmp_path)
    payload = labs.to_dict()
    payload["schema_version"] = "0.1.0"
    with pytest.raises(ValueError):
        LabEvidence.model_validate(payload)
    payload = labs.to_dict()
    payload["generated_at"] = "2026-07-18T10:00:00"
    with pytest.raises(ValueError, match="timezone"):
        LabEvidence.model_validate(payload)
