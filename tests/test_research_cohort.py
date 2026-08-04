from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pydicom
import pytest
from pydantic import ValidationError
from test_dicom_geometry import _write_phantom

from hcc_multimodal.research_cohort import (
    _case_protocol_findings,
    _resolve_contained,
    build_research_cohort,
    load_research_manifest,
    load_research_protocol,
    validate_research_cohort,
)
from hcc_multimodal.research_models import (
    IndexTreatment,
    ResearchCase,
    ResearchCohortManifest,
    ResearchProtocol,
    ResearchRunManifest,
    StudyReference,
)


def _write_study(
    data_root: Path,
    patient_id: str,
    study_id: str,
    study_date: date,
    *,
    registration_status: str,
) -> tuple[StudyReference, Path]:
    study_root = data_root / patient_id / study_id
    ct_dir, seg_path = _write_phantom(study_root)
    for path in [*ct_dir.glob("*.dcm"), seg_path]:
        dataset = pydicom.dcmread(path)
        dataset.PatientID = patient_id
        pydicom.dcmwrite(path, dataset, enforce_file_format=True)
    ct_header = pydicom.dcmread(next(ct_dir.glob("*.dcm")), stop_before_pixels=True)
    seg_header = pydicom.dcmread(seg_path, stop_before_pixels=True)
    return (
        StudyReference(
            study_id=study_id,
            study_date=study_date,
            phase="portal_venous",
            ct_dir=ct_dir.relative_to(data_root).as_posix(),
            seg_file=seg_path.relative_to(data_root).as_posix(),
            study_instance_uid=str(ct_header.StudyInstanceUID),
            series_instance_uid=str(ct_header.SeriesInstanceUID),
            seg_series_instance_uid=str(seg_header.SeriesInstanceUID),
            frame_of_reference_uid=str(ct_header.FrameOfReferenceUID),
            scanner_group="thin_slice",
            registration_status=registration_status,
        ),
        next(ct_dir.glob("*.dcm")),
    )


def _write_cohort(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    data_root = tmp_path / "data"
    cases: list[ResearchCase] = []
    first_ct_by_patient: dict[str, Path] = {}
    for patient_id, center_id in (("DEV_001", "CENTER_DEV"), ("EXT_001", "CENTER_EXT")):
        baseline, baseline_ct = _write_study(
            data_root,
            patient_id,
            "BASELINE",
            date(2026, 1, 1),
            registration_status="not_applicable",
        )
        followup, _ = _write_study(
            data_root,
            patient_id,
            "FOLLOWUP_167D",
            date(2026, 7, 1),
            registration_status="verified",
        )
        labs_path = data_root / patient_id / "labs.json"
        labs_path.write_text(
            json.dumps(
                {
                    "patient_id": patient_id,
                    "observations": [
                        {
                            "date": "2026-01-01",
                            "marker": "AFP",
                            "value": 5,
                            "unit": "ng/mL",
                            "upper_reference": 7,
                        },
                        {
                            "date": "2026-07-01",
                            "marker": "AFP",
                            "value": 10,
                            "unit": "ng/mL",
                            "upper_reference": 7,
                        },
                        {
                            "date": "2026-01-01",
                            "marker": "DCP",
                            "value": 20,
                            "unit": "mAU/mL",
                            "upper_reference": 40,
                        },
                        {
                            "date": "2026-07-01",
                            "marker": "DCP",
                            "value": 60,
                            "unit": "mAU/mL",
                            "upper_reference": 40,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        cases.append(
            ResearchCase(
                patient_id=patient_id,
                center_id=center_id,
                treatment=IndexTreatment(
                    treatment_date=date(2026, 1, 15),
                    treatment_type="TACE",
                ),
                baseline=baseline,
                followups=[followup],
                labs_file=labs_path.relative_to(data_root).as_posix(),
            )
        )
        first_ct_by_patient[patient_id] = baseline_ct

    protocol = ResearchProtocol(
        cohort_id="SYNTHETIC_MULTICENTER",
        external_test_center_ids=["CENTER_EXT"],
        split_salt="fixed-synthetic-salt",
    )
    manifest = ResearchCohortManifest(
        cohort_id=protocol.cohort_id,
        data_root="data",
        cases=cases,
    )
    protocol_path = tmp_path / "protocol.yaml"
    manifest_path = tmp_path / "manifest.json"
    protocol.write_json(protocol_path)
    manifest.write_json(manifest_path)
    return protocol_path, manifest_path, first_ct_by_patient


def test_real_format_multicenter_cohort_build_is_label_free(tmp_path: Path):
    protocol_path, manifest_path, _ = _write_cohort(tmp_path)
    validation_path = validate_research_cohort(
        protocol_path,
        manifest_path,
        tmp_path / "validation.json",
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["counts"] == {
        "excluded": 0,
        "failed": 0,
        "included": 2,
        "indeterminate": 0,
    }

    run_path = build_research_cohort(protocol_path, manifest_path, tmp_path / "features")
    run = ResearchRunManifest.model_validate_json(run_path.read_text(encoding="utf-8"))
    assert run.label_data_loaded is False
    assert {case.patient_id: case.split for case in run.cases} == {
        "DEV_001": "development",
        "EXT_001": "external_test",
    }
    assert all(case.disposition == "included" for case in run.cases)
    assert all(case.followups[0].rule_fusion_score == 0.5 for case in run.cases)
    lock = json.loads((tmp_path / "features" / "research_run_lock.json").read_text())
    assert lock["label_data_loaded"] is False
    assert not list((tmp_path / "features").rglob("*adjudicat*"))


def test_phi_header_causes_explicit_failed_disposition(tmp_path: Path):
    protocol_path, manifest_path, ct_paths = _write_cohort(tmp_path)
    path = ct_paths["DEV_001"]
    dataset = pydicom.dcmread(path)
    dataset.PatientName = "REAL^NAME"
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)

    result_path = validate_research_cohort(
        protocol_path,
        manifest_path,
        tmp_path / "validation.json",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    by_patient = {case["patient_id"]: case for case in result["cases"]}
    assert by_patient["DEV_001"]["disposition"] == "failed"
    assert "PHI_SCAN_FAILED" in by_patient["DEV_001"]["reason_codes"]
    assert any("PatientName" in item for item in by_patient["DEV_001"]["phi_findings"])
    assert by_patient["EXT_001"]["disposition"] == "included"
    assert sum(result["counts"].values()) == 2


def test_missing_seg_and_unsupported_lab_unit_are_audited(tmp_path: Path):
    protocol_path, manifest_path, _ = _write_cohort(tmp_path)
    missing_seg = tmp_path / "data" / "EXT_001" / "FOLLOWUP_167D" / "mass-seg.dcm"
    missing_seg.unlink()
    labs_path = tmp_path / "data" / "DEV_001" / "labs.json"
    labs = json.loads(labs_path.read_text(encoding="utf-8"))
    labs["observations"].append(
        {
            "date": "2026-06-30",
            "marker": "AFP",
            "value": 999,
            "unit": "unsupported-unit",
        }
    )
    labs_path.write_text(json.dumps(labs), encoding="utf-8")

    result_path = validate_research_cohort(
        protocol_path,
        manifest_path,
        tmp_path / "validation.json",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    by_patient = {case["patient_id"]: case for case in result["cases"]}
    assert "SEG_FILE_MISSING:FOLLOWUP_167D" in by_patient["EXT_001"]["reason_codes"]
    assert by_patient["EXT_001"]["disposition"] == "failed"
    assert by_patient["DEV_001"]["disposition"] == "included"
    assert any(message.startswith("LAB_WARNING:") for message in by_patient["DEV_001"]["messages"])


def test_protocol_and_manifest_contracts_reject_drift(tmp_path: Path):
    protocol_path, manifest_path, _ = _write_cohort(tmp_path)
    protocol = load_research_protocol(protocol_path)
    manifest = load_research_manifest(manifest_path)
    case = manifest.cases[0]

    with pytest.raises(ValidationError, match="Duplicate patient IDs"):
        ResearchCohortManifest(
            cohort_id=manifest.cohort_id,
            data_root=manifest.data_root,
            cases=[case, case],
        )
    with pytest.raises(ValidationError, match="baseline study must occur before"):
        ResearchCase(
            patient_id="P_BAD_DATE",
            center_id="CENTER_DEV",
            treatment=IndexTreatment(
                treatment_date=date(2026, 1, 15),
                treatment_type="TACE",
            ),
            baseline=case.baseline.model_copy(update={"study_date": date(2026, 1, 15)}),
            followups=case.followups,
            labs_file="labs.json",
        )
    with pytest.raises(ValidationError, match="preregistered at 0.5"):
        ResearchProtocol(
            cohort_id="C",
            external_test_center_ids=["EXT"],
            split_salt="fixed-salt",
            fixed_score_threshold=0.6,
        )

    altered = case.model_copy(
        update={
            "baseline": case.baseline.model_copy(
                update={"study_date": date(2025, 11, 1), "phase": "arterial"}
            ),
            "followups": [
                case.followups[0].model_copy(
                    update={"phase": "arterial", "registration_status": "unavailable"}
                )
            ],
        }
    )
    exclusions, warnings = _case_protocol_findings(protocol, altered)
    assert set(exclusions) == {
        "BASELINE_OUTSIDE_42_DAY_WINDOW",
        "BASELINE_PHASE_NOT_PORTAL_VENOUS",
        "NO_ELIGIBLE_REGISTERED_PORTAL_VENOUS_FOLLOWUP",
    }
    assert warnings == ["FOLLOWUP_PHASE_EXCLUDED:FOLLOWUP_167D"]
    with pytest.raises(ValueError, match="escapes the cohort data root"):
        _resolve_contained(tmp_path / "data", "../labels.json", "labs_file")


def test_schema_1_accepts_historical_pipeline_version(tmp_path: Path):
    protocol_path, _, _ = _write_cohort(tmp_path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["pipeline_version"] = "0.2.0"
    assert ResearchProtocol.model_validate(payload).pipeline_version == "0.2.0"
    payload["pipeline_version"] = "development"
    with pytest.raises(ValidationError, match="MAJOR.MINOR.PATCH"):
        ResearchProtocol.model_validate(payload)
