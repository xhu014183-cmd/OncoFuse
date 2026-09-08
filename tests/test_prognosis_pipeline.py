from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from hcc_multimodal.prognosis_data import (
    HccTaceSegAdapter,
    WawTaceAdapter,
    build_prognosis_cohort,
)
from hcc_multimodal.prognosis_features import MaskFeatureResult, extract_mask_features
from hcc_multimodal.prognosis_models import (
    CoxModelBundle,
    PublicPrognosisCohortArtifact,
    PublicPrognosisRecord,
    SurvivalEndpointRecord,
)
from hcc_multimodal.prognosis_report import load_prognosis_case, run_prognosis_report
from hcc_multimodal.prognosis_survival import (
    _fit_coefficients,
    concordance_index,
    evaluate_external_prognosis,
    score_feature_mapping,
    train_prognosis_models,
)
from hcc_multimodal.schemas import QualityEvidence, SourceReference


def _nifti_pair(tmp_path: Path) -> tuple[Path, Path]:
    image = np.zeros((20, 20, 20), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    mask[12:16, 12:16, 12:16] = 1
    image[mask > 0] = np.arange(int(mask.sum()), dtype=np.float32)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    image_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    return image_path, mask_path


def _bundle(*, status: str = "evaluated") -> CoxModelBundle:
    names = [
        "age_years",
        "female",
        "afp_ng_ml",
        "lesion_count",
        "total_tumor_volume_ml",
        "max_lesion_extent_mm",
        "largest_lesion_sphericity",
    ]
    return CoxModelBundle(
        model_id="test-fused-v1",
        model_name="fused_core",
        feature_names=names,
        transformations={
            "age_years": "identity",
            "female": "identity",
            "afp_ng_ml": "log1p",
            "lesion_count": "log1p",
            "total_tumor_volume_ml": "log1p",
            "max_lesion_extent_mm": "log1p",
            "largest_lesion_sphericity": "identity",
        },
        means=[60, 0.5, 4, 1, 2, 3, 0.7],
        scales=[10, 0.5, 2, 1, 2, 1, 0.1],
        coefficients=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, -0.1],
        penalizer=0.1,
        baseline_event_times_days=[100, 365, 730],
        baseline_cumulative_hazard=[0.1, 0.3, 0.6],
        development_reference_risks=[-2, -1, 0, 1, 2],
        median_risk_threshold=0,
        training_n=233,
        training_event_n=168,
        cross_validation_seed=1729,
        model_hash="a" * 64,
        external_validation_status=status,
        limitations=["retrospective research model"],
    )


def test_mask_features_use_26_connectivity_and_portal_statistics(tmp_path: Path) -> None:
    image_path, mask_path = _nifti_pair(tmp_path)
    result = extract_mask_features(mask_path, portal_image_path=image_path)
    assert result.lesion_count == 2
    assert result.total_tumor_volume_ml == pytest.approx(0.256)
    assert result.max_lesion_extent_mm == pytest.approx(8.0)
    assert 0 < result.largest_lesion_sphericity <= 1
    assert result.portal_mean_hu is not None
    assert result.portal_p90_hu is not None


def test_empty_mask_and_geometry_mismatch_fail_closed(tmp_path: Path) -> None:
    image_path, mask_path = _nifti_pair(tmp_path)
    empty_path = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20)), np.eye(4)), empty_path)
    with pytest.raises(ValueError, match="empty"):
        extract_mask_features(empty_path)
    result = extract_mask_features(mask_path, portal_image_path=empty_path)
    assert result.lesion_count == 2
    assert result.portal_mean_hu is None
    assert any(
        warning.startswith("PORTAL_INTENSITY_UNAVAILABLE")
        for warning in result.warnings
    )
    assert image_path.is_file()


def test_separate_lesion_grids_are_measured_without_resampling(tmp_path: Path) -> None:
    first = np.zeros((12, 12, 12), dtype=np.uint8)
    second = np.zeros((8, 8, 8), dtype=np.uint8)
    first[1:4, 1:4, 1:4] = 1
    second[2:5, 2:5, 2:5] = 1
    first_path = tmp_path / "lesion_1.nii.gz"
    second_path = tmp_path / "lesion_2.nii.gz"
    nib.save(nib.Nifti1Image(first, np.eye(4)), first_path)
    nib.save(nib.Nifti1Image(second, np.diag([2.0, 2.0, 2.0, 1.0])), second_path)

    result = extract_mask_features([first_path, second_path])

    assert result.lesion_count == 2
    assert result.total_tumor_volume_ml == pytest.approx(0.243)
    assert result.max_lesion_extent_mm == pytest.approx(6.0)
    assert any("SOURCE_MASK_GRIDS_DIFFER" in item for item in result.warnings)


def test_case_input_rejects_any_outcome_field(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "case_id": "CASE",
        "patient_id": "PATIENT",
        "clinical_task": "tace_overall_survival_prognosis",
        "imaging": {
            "source_type": "nifti",
            "nifti_image": "ct.nii.gz",
            "seg": "mask.nii.gz",
            "seg_role": "public_reference",
        },
        "clinical": {"age_years": 60, "sex": "male", "afp_ng_ml": 100},
        "data_relationship": {
            "imaging_origin": "real_public",
            "laboratory_origin": "real_public",
            "pairing_status": "same_subject",
            "statement": "same public subject",
        },
        "event": 1,
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Outcome fields"):
        load_prognosis_case(path)


def test_cox_coefficients_and_c_index_have_expected_direction() -> None:
    matrix = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    durations = np.asarray([500.0, 400.0, 300.0, 200.0, 100.0])
    events = np.ones(5, dtype=int)
    coefficient = _fit_coefficients(matrix, durations, events, 0.1)
    risks = matrix @ coefficient
    assert coefficient[0] > 0
    assert concordance_index(durations, events, risks) == pytest.approx(1.0)


def test_model_scoring_is_deterministic_and_complete() -> None:
    values = {
        "age_years": 65,
        "female": 0,
        "afp_ng_ml": 100,
        "lesion_count": 2,
        "total_tumor_volume_ml": 10,
        "max_lesion_extent_mm": 30,
        "largest_lesion_sphericity": 0.8,
    }
    first = score_feature_mapping(_bundle(), values)
    second = score_feature_mapping(_bundle(), values)
    assert first == second
    with pytest.raises(ValueError, match="Missing model feature"):
        score_feature_mapping(_bundle(), {"age_years": 65})


def test_deterministic_prognosis_report_keeps_outcomes_out_of_prompt(tmp_path: Path) -> None:
    image_path, mask_path = _nifti_pair(tmp_path)
    model_path = tmp_path / "model.json"
    _bundle().write_json(model_path)
    case = {
        "schema_version": "1.0.0",
        "case_id": "CASE_001",
        "patient_id": "RESEARCH_001",
        "clinical_task": "tace_overall_survival_prognosis",
        "imaging": {
            "source_type": "nifti",
            "nifti_image": str(image_path),
            "seg": str(mask_path),
            "seg_role": "public_reference",
            "phase": "portal_venous",
            "study_date": "2026-01-01",
        },
        "clinical": {"age_years": 65, "sex": "male", "afp_ng_ml": 100},
        "data_relationship": {
            "imaging_origin": "real_public",
            "laboratory_origin": "real_public",
            "pairing_status": "same_subject",
            "statement": "公开影像与同一受试者临床表",
        },
        "output_dir": str(tmp_path / "unused"),
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "report"
    result = run_prognosis_report(case_path, model_path, output_dir=output)
    assert result.evidence.status == "pass"
    assert result.report.review_required is True
    prompt = (output / "deepseek-prompt.json").read_text(encoding="utf-8")
    assert "duration_days" not in prompt
    assert '"event"' not in prompt
    assert "survival_time" not in prompt
    glm_audit = json.loads((output / "provider-audit" / "glm.json").read_text())
    assert glm_audit["error_code"] == "GLM_DISABLED"
    deepseek_audit = json.loads(
        (output / "provider-audit" / "deepseek.json").read_text()
    )
    assert deepseek_audit["error_code"] == "DEEPSEEK_DISABLED"


def test_public_records_do_not_accept_outcome_fields() -> None:
    source = SourceReference(source_id="x", source_type="test")
    record = PublicPrognosisRecord(
        patient_id="WAW_12345678",
        cohort_id="waw_tace",
        dataset_version="v2",
        source_patient_sha256="b" * 64,
        geometry_qc="pass",
        age_years=60,
        female=0,
        afp_ng_ml=10,
        lesion_count=1,
        total_tumor_volume_ml=1,
        max_lesion_extent_mm=10,
        largest_lesion_sphericity=0.8,
        source_refs=[source],
    )
    endpoint = SurvivalEndpointRecord(
        patient_id=record.patient_id,
        cohort_id="waw_tace",
        duration_days=100,
        event=1,
        source_ref=source,
    )
    assert "event" not in record.to_dict()
    assert endpoint.event == 1


def _cohort_record(index: int, cohort: str) -> PublicPrognosisRecord:
    prefix = "WAW" if cohort == "waw_tace" else "TCIA"
    return PublicPrognosisRecord(
        patient_id=f"{prefix}_{index:08d}",
        cohort_id=cohort,
        dataset_version="test-v1",
        source_patient_sha256=f"{index + (0 if cohort == 'waw_tace' else 1000):064x}",
        geometry_qc="pass",
        age_years=40 + index,
        female=index % 2,
        afp_ng_ml=float(2 + index**2),
        lesion_count=1 + index % 4,
        total_tumor_volume_ml=0.5 + index * 0.7,
        max_lesion_extent_mm=5 + index * 0.8,
        largest_lesion_sphericity=0.45 + (index % 8) * 0.05,
        albumin_g_dl=3.0 + (index % 10) * 0.15,
        bilirubin_mg_dl=0.4 + index * 0.03,
        inr=0.9 + (index % 9) * 0.04,
        alt_iu_l=15 + index * 1.7,
        creatinine_mg_dl=0.5 + (index % 11) * 0.08,
    )


def test_cohort_builder_writes_separate_development_and_external_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    waw = _cohort_record(1, "waw_tace")
    hcc = _cohort_record(2, "hcc_tace_seg")
    source = SourceReference(source_id="synthetic", source_type="test")
    waw_endpoint = SurvivalEndpointRecord(
        patient_id=waw.patient_id,
        cohort_id="waw_tace",
        duration_days=500,
        event=1,
        source_ref=source,
    )
    hcc_endpoint = SurvivalEndpointRecord(
        patient_id=hcc.patient_id,
        cohort_id="hcc_tace_seg",
        duration_days=600,
        event=0,
        source_ref=source,
    )
    monkeypatch.setattr(
        WawTaceAdapter,
        "build",
        lambda _self: ([waw], [waw_endpoint], []),
    )
    monkeypatch.setattr(
        HccTaceSegAdapter,
        "build",
        lambda _self: ([hcc], [hcc_endpoint], []),
    )

    output = tmp_path / "cohort-output"
    build_prognosis_cohort(tmp_path / "data", output)

    development = PublicPrognosisCohortArtifact.model_validate_json(
        (output / "development-cohort.json").read_text()
    )
    external = PublicPrognosisCohortArtifact.model_validate_json(
        (output / "external-test-cohort.json").read_text()
    )
    assert {item.cohort_id for item in development.feature_records} == {"waw_tace"}
    assert {item.cohort_id for item in external.feature_records} == {"hcc_tace_seg"}
    assert "event" not in (output / "development-features.csv").read_text().splitlines()[0]
    assert (output / "figure-manifest.json").is_file()


def test_training_and_one_time_external_validation(tmp_path: Path) -> None:
    features = [
        *[_cohort_record(index, "waw_tace") for index in range(30)],
        *[_cohort_record(index, "hcc_tace_seg") for index in range(20)],
    ]
    source = SourceReference(source_id="synthetic", source_type="test")
    endpoints = [
        SurvivalEndpointRecord(
            patient_id=record.patient_id,
            cohort_id=record.cohort_id,
            duration_days=float(1000 - index * 10),
            event=0 if index % 7 == 0 else 1,
            source_ref=source,
        )
        for index, record in enumerate(features)
    ]
    combined = PublicPrognosisCohortArtifact(
        feature_records=features,
        endpoint_records=endpoints,
        exclusions=[],
        counts={"total_included": len(features)},
        quality=QualityEvidence(status="pass"),
    )
    combined_path = tmp_path / "combined.json"
    combined.write_json(combined_path)
    development = PublicPrognosisCohortArtifact(
        feature_records=features[:30],
        endpoint_records=endpoints[:30],
        exclusions=[],
        counts={"total_included": 30},
        quality=QualityEvidence(status="pass"),
    )
    external_test = PublicPrognosisCohortArtifact(
        feature_records=features[30:],
        endpoint_records=endpoints[30:],
        exclusions=[],
        counts={"total_included": 20},
        quality=QualityEvidence(status="pass"),
    )
    development_path = tmp_path / "development.json"
    external_path = tmp_path / "external-test.json"
    development.write_json(development_path)
    external_test.write_json(external_path)
    models = tmp_path / "models"
    internal = train_prognosis_models(
        development_path, models, bootstrap_iterations=5
    )
    assert internal.is_file()
    assert (models / "model-bundle-fused.json").is_file()
    assert (models / "model-bundle-fused-extended-no-albumin.json").is_file()
    external_dir = tmp_path / "external"
    external = evaluate_external_prognosis(
        external_path,
        models,
        external_dir,
        unlock_external=True,
        bootstrap_iterations=5,
    )
    external_payload = json.loads(external.read_text())
    assert external_payload["external_data_used_for_tuning"] is False
    assert set(external_payload["results"]) == {
        "clinical_core",
        "imaging_core",
        "fused_core",
    }
    evaluated = CoxModelBundle.model_validate_json(
        (models / "model-bundle-fused.json").read_text()
    )
    assert evaluated.external_validation_status == "evaluated"
    with pytest.raises(RuntimeError, match="already been opened"):
        evaluate_external_prognosis(
            external_path,
            models,
            external_dir,
            unlock_external=True,
            bootstrap_iterations=5,
        )
    with pytest.raises(ValueError, match="development-only"):
        train_prognosis_models(combined_path, tmp_path / "leaky-models")


def _fake_mask_result() -> MaskFeatureResult:
    return MaskFeatureResult(
        lesion_count=2,
        total_tumor_volume_ml=12.0,
        max_lesion_extent_mm=30.0,
        largest_lesion_sphericity=0.75,
        portal_mean_hu=None,
        portal_std_hu=None,
        portal_p10_hu=None,
        portal_p90_hu=None,
        geometry_qc="pass",
        warnings=[],
    )


def test_waw_adapter_maps_units_and_keeps_outcomes_separate(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "waw-tace"
    downloads = root / "downloads"
    mask_dir = root / "extracted" / "tumor_masks" / "bundle" / "7"
    downloads.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (mask_dir / "7_1_0_tumor_seg.nrrd").write_bytes(b"fixture")
    pd.DataFrame(
        [
            {
                "PATPRI": 7,
                "age": 66,
                "gender_woman": 1,
                "death": 1,
                "survival_time": 800,
                "lab_albumin": 3.8,
                "lab_creatinine": 0.9,
                "lab_bilirubin": 1.0,
                "lab_afp": 50,
                "lab_inr": 1.1,
                "lab_alt": 40,
            }
        ]
    ).to_excel(downloads / "clinical_data_wawtace_v2_15_07_2024.xlsx", index=False)
    monkeypatch.setattr(
        "hcc_multimodal.prognosis_data.extract_mask_features",
        lambda *_args, **_kwargs: _fake_mask_result(),
    )
    records, endpoints, exclusions = WawTaceAdapter(root).build()
    assert not exclusions
    assert records[0].albumin_g_dl == 3.8
    assert any("SOURCE_UNIT_OVERRIDE" in item for item in records[0].quality_warnings)
    assert "event" not in records[0].to_dict()
    assert endpoints[0].duration_days == 800
    assert endpoints[0].event == 1


def test_hcc_adapter_converts_os_weeks_and_sex(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "hcc-tace-seg"
    downloads = root / "downloads"
    prepared = root / "prepared" / "HCC_003"
    metadata = root / "metadata"
    downloads.mkdir(parents=True)
    prepared.mkdir(parents=True)
    metadata.mkdir(parents=True)
    baseline_seg_uid = "1.2.3.4.5"
    (metadata / "hcc_tace_seg_baseline_candidates.json").write_text(
        json.dumps(
            [
                {
                    "patient_id": "HCC_003",
                    "studies": [{"seg_series": [baseline_seg_uid]}],
                }
            ]
        ),
        encoding="utf-8",
    )
    clinical = pd.DataFrame(
        [
            {
                "TCIA_ID": "HCC_003",
                "OS": 21.428571,
                "Death_1_StillAliveorLostToFU_0": 1,
                "age": 53,
                "Sex": 2,
                "AFP": 1555.2,
            }
        ]
    )
    with pd.ExcelWriter(downloads / "HCC-TACE-Seg_clinical_data-V2.xlsx") as writer:
        clinical.to_excel(writer, sheet_name="data table", index=False)
    image = prepared / "ct.nii.gz"
    mask = prepared / "mask.nii.gz"
    evidence = prepared / "evidence.json"
    image.write_bytes(b"fixture")
    mask.write_bytes(b"fixture")
    evidence.write_text(json.dumps({"phase": "portal_venous"}), encoding="utf-8")
    cohort_manifest = root / "prepared" / "cohort_manifest.json"
    prepared_patient = {
        "patient_id": "HCC_003",
        "status": "ok",
        "seg_series_uid": baseline_seg_uid,
        "image_file": str(image),
        "mask_file": str(mask),
        "evidence_file": str(evidence),
    }
    cohort_manifest.write_text(
        json.dumps({"patients": [prepared_patient]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hcc_multimodal.prognosis_data.extract_mask_features",
        lambda *_args, **_kwargs: _fake_mask_result(),
    )
    records, endpoints, exclusions = HccTaceSegAdapter(root).build()
    assert not exclusions
    assert records[0].female == 1
    assert records[0].afp_ng_ml == 1555.2
    assert records[0].source_refs[1].source_id == "HCC_003/mask.nii.gz"
    assert endpoints[0].duration_days == pytest.approx(150.0)
    assert endpoints[0].event == 1

    prepared_patient["seg_series_uid"] = "follow-up-seg"
    cohort_manifest.write_text(
        json.dumps({"patients": [prepared_patient]}), encoding="utf-8"
    )
    blocked_records, blocked_endpoints, blocked = HccTaceSegAdapter(root).build()
    assert not blocked_records
    assert not blocked_endpoints
    assert blocked[0].reason_codes == ["HCC_NON_BASELINE_SEG_BLOCKED"]
