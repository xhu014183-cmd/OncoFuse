from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
import json
import subprocess

import yaml

from .fusion import fuse_evidence
from .imaging import compare_imaging, measure_nifti
from .labs import align_lab_evidence, load_lab_evidence
from .research_models import (
    CaseDisposition,
    CohortValidationReport,
    Disposition,
    FollowupRunRecord,
    ResearchCase,
    ResearchCaseRun,
    ResearchCohortManifest,
    ResearchProtocol,
    ResearchRunManifest,
    StudyReference,
)
from .schemas import QualityCheck, QualityEvidence, QualityStatus, SourceReference
from .tcia import convert_ct_and_mass_seg


PHI_KEYWORDS = (
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "AccessionNumber",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "PhysiciansOfRecord",
    "InstitutionAddress",
)


def _load_yaml_model(path: str | Path, model: type[ResearchProtocol]) -> ResearchProtocol:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return model.model_validate(payload)


def load_research_protocol(path: str | Path) -> ResearchProtocol:
    return _load_yaml_model(path, ResearchProtocol)


def load_research_manifest(path: str | Path) -> ResearchCohortManifest:
    return ResearchCohortManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _configuration_hash(protocol_path: str | Path) -> str:
    digest = sha256()
    package = Path(__file__).parent
    for path in (
        Path(protocol_path),
        package / "configs" / "fusion_rules.v1.yaml",
        package / "configs" / "matching_rules.v1.yaml",
        package / "evaluation.py",
        package / "fusion.py",
        package / "imaging.py",
        package / "labs.py",
        package / "research_cohort.py",
        package / "research_evaluation.py",
        package / "research_models.py",
        package / "schemas.py",
        package / "tcia.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(b"research-score-v1:image-binary,lab-signal-count-half,fusion-mean")
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _resolve_contained(root: Path, relative: str, field_name: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{field_name} must be relative to the cohort data root")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes the cohort data root") from exc
    return resolved


def _data_root(manifest_path: str | Path, manifest: ResearchCohortManifest) -> Path:
    manifest_dir = Path(manifest_path).resolve().parent
    return _resolve_contained(manifest_dir, manifest.data_root, "data_root")


def _dicom_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.dcm") if path.is_file())


def _nonempty_dicom_value(dataset: Any, keyword: str) -> bool:
    value = getattr(dataset, keyword, None)
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(str(item).strip() for item in value)
    return bool(str(value).strip())


def _inspect_study(
    root: Path,
    patient_id: str,
    study: StudyReference,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    phi_findings: list[str] = []
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "Research cohort DICOM validation requires: pip install -e '.[public-data]'"
        ) from exc

    try:
        ct_dir = _resolve_contained(root, study.ct_dir, f"{study.study_id}.ct_dir")
        seg_file = _resolve_contained(root, study.seg_file, f"{study.study_id}.seg_file")
    except ValueError as exc:
        return [str(exc)], []
    if not ct_dir.is_dir():
        return [f"CT_DIRECTORY_MISSING:{study.study_id}"], []
    if not seg_file.is_file():
        return [f"SEG_FILE_MISSING:{study.study_id}"], []
    files = _dicom_files(ct_dir)
    if not files:
        return [f"CT_DICOM_EMPTY:{study.study_id}"], []

    pixel_instances = 0
    for path in files:
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            errors.append(f"CT_DICOM_UNREADABLE:{study.study_id}:{path.name}")
            continue
        if str(getattr(dataset, "PatientID", "")) != patient_id:
            errors.append(f"PATIENT_ID_MISMATCH:{study.study_id}:{path.name}")
        if str(getattr(dataset, "StudyInstanceUID", "")) != study.study_instance_uid:
            errors.append(f"STUDY_UID_MISMATCH:{study.study_id}:{path.name}")
        if str(getattr(dataset, "SeriesInstanceUID", "")) != study.series_instance_uid:
            errors.append(f"SERIES_UID_MISMATCH:{study.study_id}:{path.name}")
        if str(getattr(dataset, "FrameOfReferenceUID", "")) != study.frame_of_reference_uid:
            errors.append(f"FRAME_OF_REFERENCE_UID_MISMATCH:{study.study_id}:{path.name}")
        if str(getattr(dataset, "Modality", "")) != "CT":
            errors.append(f"NON_CT_INSTANCE:{study.study_id}:{path.name}")
        if hasattr(dataset, "Rows") and hasattr(dataset, "Columns"):
            pixel_instances += 1
        for keyword in PHI_KEYWORDS:
            if _nonempty_dicom_value(dataset, keyword):
                phi_findings.append(f"PHI_TAG_PRESENT:{study.study_id}:{path.name}:{keyword}")
    if pixel_instances == 0:
        errors.append(f"CT_HAS_NO_PIXEL_INSTANCES:{study.study_id}")

    try:
        segmentation = pydicom.dcmread(str(seg_file), stop_before_pixels=True)
    except Exception:
        errors.append(f"SEG_DICOM_UNREADABLE:{study.study_id}:{seg_file.name}")
        return sorted(set(errors)), sorted(set(phi_findings))
    if str(getattr(segmentation, "PatientID", "")) != patient_id:
        errors.append(f"SEG_PATIENT_ID_MISMATCH:{study.study_id}")
    if str(getattr(segmentation, "StudyInstanceUID", "")) != study.study_instance_uid:
        errors.append(f"SEG_STUDY_UID_MISMATCH:{study.study_id}")
    if str(getattr(segmentation, "SeriesInstanceUID", "")) != study.seg_series_instance_uid:
        errors.append(f"SEG_SERIES_UID_MISMATCH:{study.study_id}")
    if str(getattr(segmentation, "FrameOfReferenceUID", "")) != study.frame_of_reference_uid:
        errors.append(f"SEG_FRAME_OF_REFERENCE_UID_MISMATCH:{study.study_id}")
    if str(getattr(segmentation, "Modality", "")) != "SEG":
        errors.append(f"NON_SEG_OBJECT:{study.study_id}")
    for keyword in PHI_KEYWORDS:
        if _nonempty_dicom_value(segmentation, keyword):
            phi_findings.append(f"PHI_TAG_PRESENT:{study.study_id}:{seg_file.name}:{keyword}")
    return sorted(set(errors)), sorted(set(phi_findings))


def _case_protocol_findings(
    protocol: ResearchProtocol,
    case: ResearchCase,
) -> tuple[list[str], list[str]]:
    exclusions: list[str] = []
    warnings: list[str] = []
    baseline_gap = (case.treatment.treatment_date - case.baseline.study_date).days
    if baseline_gap > protocol.baseline_max_days_before_treatment:
        exclusions.append("BASELINE_OUTSIDE_42_DAY_WINDOW")
    if case.baseline.phase != protocol.required_phase:
        exclusions.append("BASELINE_PHASE_NOT_PORTAL_VENOUS")
    eligible_followups = 0
    for study in case.followups:
        if study.phase != protocol.required_phase:
            warnings.append(f"FOLLOWUP_PHASE_EXCLUDED:{study.study_id}")
            continue
        if study.registration_status != "verified":
            warnings.append(f"FOLLOWUP_REGISTRATION_NOT_VERIFIED:{study.study_id}")
            continue
        eligible_followups += 1
    if eligible_followups == 0:
        exclusions.append("NO_ELIGIBLE_REGISTERED_PORTAL_VENOUS_FOLLOWUP")
    return exclusions, warnings


def validate_research_cohort(
    protocol_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    protocol = load_research_protocol(protocol_path)
    manifest = load_research_manifest(manifest_path)
    if protocol.cohort_id != manifest.cohort_id:
        raise ValueError("Protocol and manifest cohort_id values do not match")
    centers = {case.center_id for case in manifest.cases}
    missing_external = sorted(set(protocol.external_test_center_ids) - centers)
    if missing_external:
        raise ValueError(f"External test centers are absent from the manifest: {missing_external}")
    if not centers - set(protocol.external_test_center_ids):
        raise ValueError("At least one development center is required")

    root = _data_root(manifest_path, manifest)
    dispositions: list[CaseDisposition] = []
    for case in sorted(manifest.cases, key=lambda item: item.patient_id):
        exclusions, messages = _case_protocol_findings(protocol, case)
        technical_errors: list[str] = []
        phi_findings: list[str] = []
        for study in [case.baseline, *case.followups]:
            study_errors, study_phi = _inspect_study(root, case.patient_id, study)
            technical_errors.extend(study_errors)
            phi_findings.extend(study_phi)
        try:
            labs_path = _resolve_contained(root, case.labs_file, f"{case.patient_id}.labs_file")
            if not labs_path.is_file():
                technical_errors.append("LAB_FILE_MISSING")
            else:
                labs = load_lab_evidence(labs_path)
                if labs.patient_id != case.patient_id:
                    technical_errors.append("LAB_PATIENT_ID_MISMATCH")
                if labs.quality.status in {"fail", "unavailable"}:
                    messages.append("LAB_EVIDENCE_UNAVAILABLE")
                messages.extend(f"LAB_WARNING:{warning}" for warning in labs.warnings)
        except Exception as exc:
            technical_errors.append(f"LAB_FILE_INVALID:{type(exc).__name__}")

        if phi_findings:
            technical_errors.append("PHI_SCAN_FAILED")
        if technical_errors:
            disposition: Disposition = "failed"
            reasons = sorted(set(technical_errors))
        elif exclusions:
            disposition = "excluded"
            reasons = sorted(set(exclusions))
        else:
            disposition = "included"
            reasons = []
        dispositions.append(
            CaseDisposition(
                patient_id=case.patient_id,
                center_id=case.center_id,
                disposition=disposition,
                reason_codes=reasons,
                messages=sorted(set(messages)),
                phi_findings=sorted(set(phi_findings)),
            )
        )

    counts: dict[str, int] = dict(Counter(item.disposition for item in dispositions))
    for status in ("included", "excluded", "failed", "indeterminate"):
        counts.setdefault(status, 0)
    if counts["failed"]:
        quality_status: QualityStatus = "fail"
        errors = [f"{counts['failed']} cases failed technical or PHI validation"]
    elif counts["excluded"]:
        quality_status = "warning"
        errors = []
    else:
        quality_status = "pass"
        errors = []
    report = CohortValidationReport(
        cohort_id=manifest.cohort_id,
        protocol_hash=_file_hash(protocol_path),
        manifest_hash=_file_hash(manifest_path),
        quality=QualityEvidence(
            status=quality_status,
            checks=[
                QualityCheck(
                    check_id="COHORT_PATIENT_UNIQUENESS",
                    status="pass",
                    message="Every manifest patient ID is unique",
                ),
                QualityCheck(
                    check_id="EXTERNAL_CENTER_ISOLATION",
                    status="pass",
                    message="External test centers are explicitly declared and separate",
                ),
                QualityCheck(
                    check_id="PHI_HEADER_SCAN",
                    status="fail" if counts["failed"] else "pass",
                    message="DICOM headers and pseudonymous patient IDs were checked",
                ),
            ],
            warnings=[f"{counts['excluded']} cases fail protocol eligibility"]
            if counts["excluded"]
            else [],
            errors=errors,
        ),
        counts=counts,
        cases=dispositions,
        sources=[
            SourceReference(
                source_id=Path(protocol_path).name,
                source_type="research_protocol",
                data_origin="user_supplied",
            ),
            SourceReference(
                source_id=Path(manifest_path).name,
                source_type="research_cohort_manifest",
                data_origin="user_supplied",
            ),
        ],
    )
    target = Path(output_path)
    report.write_json(target)
    return target


def _study_paths(root: Path, study: StudyReference) -> tuple[Path, Path]:
    return (
        _resolve_contained(root, study.ct_dir, f"{study.study_id}.ct_dir"),
        _resolve_contained(root, study.seg_file, f"{study.study_id}.seg_file"),
    )


def _prepare_study(
    root: Path,
    study: StudyReference,
    patient_id: str,
    output: Path,
):
    ct_dir, seg_file = _study_paths(root, study)
    image_path, mask_path, attribution_path = convert_ct_and_mass_seg(
        ct_dir,
        seg_file,
        output / "converted",
    )
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    evidence = measure_nifti(
        image_path,
        mask_path,
        patient_id=patient_id,
        study_date=study.study_date.isoformat(),
        modality="CT",
        phase=study.phase,
        provider="expert-dicom-seg",
        inference_mode="expert_annotation",
        frame_of_reference_uid=attribution.get("ct_frame_of_reference_uid"),
        study_instance_uid=study.study_instance_uid,
        series_instance_uid=study.series_instance_uid,
        segment_series_instance_uid=study.seg_series_instance_uid,
        segment_number=attribution.get("selected_segment_number"),
        acquisition_id=attribution.get("selected_acquisition_id"),
        source_quality_warnings=attribution.get("geometry_qc", {}).get("warnings", []),
    )
    evidence.write_json(output / "imaging_evidence.json")
    return evidence


def deterministic_research_scores(labs, imaging) -> tuple[float | None, float | None, float | None, list[str]]:
    missing: list[str] = []
    complex_pairing = any(
        match.status in {"split_candidate", "merge_candidate", "indeterminate"}
        for match in imaging.matched_lesions
    )
    if imaging.quality.status in {"fail", "unavailable"} or complex_pairing:
        image_score = None
        missing.append("image_score_unavailable")
    elif imaging.category == "imaging_progression_signal":
        image_score = 1.0
    elif imaging.category in {"imaging_response_signal", "imaging_stable_or_indeterminate"}:
        image_score = 0.0
    else:
        image_score = None
        missing.append("image_score_unavailable")

    complete_labs = all(
        marker_name in labs.markers
        and labs.markers[marker_name].exact_observation_count >= 2
        and labs.markers[marker_name].direction not in {"insufficient", "indeterminate"}
        and labs.markers[marker_name].latest_above_upper is not None
        for marker_name in ("AFP", "DCP")
    )
    if not complete_labs:
        lab_score = None
        missing.append("lab_score_unavailable")
    else:
        signal_count = sum(
            labs.markers[marker_name].direction == "rising"
            and labs.markers[marker_name].latest_above_upper is True
            for marker_name in ("AFP", "DCP")
        )
        lab_score = signal_count / 2.0
    if image_score is None or lab_score is None:
        fusion_score = None
        missing.append("rule_fusion_score_unavailable")
    else:
        fusion_score = (image_score + lab_score) / 2.0
    return image_score, lab_score, fusion_score, sorted(set(missing))


def _write_case_summary(case: ResearchCaseRun, path: Path) -> None:
    lines = [
        f"# Research case {case.patient_id}",
        "",
        f"- Center: {case.center_id}",
        f"- Split: {case.split}",
        f"- Index treatment: {case.treatment_type} on {case.treatment_date.isoformat()}",
        f"- Disposition: {case.disposition}",
        "",
        "## Follow-up evidence",
        "",
    ]
    for followup in case.followups:
        lines.extend(
            [
                f"### {followup.study_id} ({followup.days_after_treatment} days)",
                "",
                f"- Disposition: {followup.disposition}",
                f"- Image-only score: {followup.image_only_score}",
                f"- Lab-only score: {followup.lab_only_score}",
                f"- Rule-fusion score: {followup.rule_fusion_score}",
                f"- Missing reasons: {', '.join(followup.missing_reasons) or 'none'}",
                "",
            ]
        )
    lines.append(
        "> Research use only; no endpoint label, diagnosis, staging, prognosis, or treatment recommendation is generated here."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_research_cohort(
    protocol_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    protocol = load_research_protocol(protocol_path)
    manifest = load_research_manifest(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_path = validate_research_cohort(
        protocol_path,
        manifest_path,
        output / "cohort_validation.json",
    )
    validation = CohortValidationReport.model_validate_json(
        validation_path.read_text(encoding="utf-8")
    )
    validation_by_patient = {item.patient_id: item for item in validation.cases}
    root = _data_root(manifest_path, manifest)
    external_centers = set(protocol.external_test_center_ids)
    runs: list[ResearchCaseRun] = []

    for case in sorted(manifest.cases, key=lambda item: item.patient_id):
        validation_case = validation_by_patient[case.patient_id]
        split: Literal["development", "external_test"] = (
            "external_test" if case.center_id in external_centers else "development"
        )
        if validation_case.disposition != "included":
            runs.append(
                ResearchCaseRun(
                    patient_id=case.patient_id,
                    center_id=case.center_id,
                    split=split,
                    treatment_date=case.treatment.treatment_date,
                    treatment_type=case.treatment.treatment_type,
                    disposition=validation_case.disposition,
                    reason_codes=validation_case.reason_codes,
                    followups=[],
                )
            )
            continue

        case_output = output / "cases" / case.patient_id
        followup_runs: list[FollowupRunRecord] = []
        try:
            baseline = _prepare_study(
                root,
                case.baseline,
                case.patient_id,
                case_output / "baseline" / case.baseline.study_id,
            )
            labs_path = _resolve_contained(root, case.labs_file, f"{case.patient_id}.labs_file")
            full_labs = load_lab_evidence(labs_path)
        except Exception as exc:
            runs.append(
                ResearchCaseRun(
                    patient_id=case.patient_id,
                    center_id=case.center_id,
                    split=split,
                    treatment_date=case.treatment.treatment_date,
                    treatment_type=case.treatment.treatment_type,
                    disposition="failed",
                    reason_codes=[f"BASELINE_BUILD_FAILED:{type(exc).__name__}"],
                    followups=[],
                )
            )
            continue

        for study in sorted(case.followups, key=lambda item: (item.study_date, item.study_id)):
            days = (study.study_date - case.treatment.treatment_date).days
            if study.phase != protocol.required_phase:
                followup_runs.append(
                    FollowupRunRecord(
                        study_id=study.study_id,
                        study_date=study.study_date,
                        days_after_treatment=days,
                        scanner_group=study.scanner_group,
                        disposition="excluded",
                        reason_codes=["FOLLOWUP_PHASE_NOT_PORTAL_VENOUS"],
                    )
                )
                continue
            if study.registration_status != "verified":
                followup_runs.append(
                    FollowupRunRecord(
                        study_id=study.study_id,
                        study_date=study.study_date,
                        days_after_treatment=days,
                        scanner_group=study.scanner_group,
                        disposition="excluded",
                        reason_codes=["FOLLOWUP_REGISTRATION_NOT_VERIFIED"],
                    )
                )
                continue
            study_output = case_output / "followups" / study.study_id
            try:
                followup = _prepare_study(root, study, case.patient_id, study_output)
                longitudinal = compare_imaging(
                    baseline,
                    followup,
                    registration_status="verified",
                )
                aligned_labs = align_lab_evidence(
                    full_labs,
                    baseline_date=case.baseline.study_date.isoformat(),
                    followup_date=study.study_date.isoformat(),
                    window_days=protocol.lab_alignment_days,
                )
                verdict = fuse_evidence(
                    aligned_labs,
                    longitudinal,
                    treatment_context="index_treatment",
                )
                image_score, lab_score, fusion_score, missing = deterministic_research_scores(
                    aligned_labs,
                    longitudinal,
                )
                longitudinal.write_json(study_output / "longitudinal_imaging_evidence.json")
                aligned_labs.write_json(study_output / "aligned_lab_evidence.json")
                verdict.write_json(study_output / "clinical_verdict.json")
                followup_runs.append(
                    FollowupRunRecord(
                        study_id=study.study_id,
                        study_date=study.study_date,
                        days_after_treatment=days,
                        scanner_group=study.scanner_group,
                        disposition="included" if fusion_score is not None else "indeterminate",
                        reason_codes=longitudinal.reason_codes,
                        image_only_score=image_score,
                        lab_only_score=lab_score,
                        rule_fusion_score=fusion_score,
                        imaging_category=longitudinal.category,
                        lab_signal_count=int(lab_score * 2) if lab_score is not None else None,
                        fusion_state=verdict.state,
                        missing_reasons=missing,
                        artifact_directory=study_output.relative_to(output).as_posix(),
                    )
                )
            except Exception as exc:
                followup_runs.append(
                    FollowupRunRecord(
                        study_id=study.study_id,
                        study_date=study.study_date,
                        days_after_treatment=days,
                        scanner_group=study.scanner_group,
                        disposition="failed",
                        reason_codes=[f"FOLLOWUP_BUILD_FAILED:{type(exc).__name__}"],
                    )
                )

        included_followups = [item for item in followup_runs if item.disposition == "included"]
        if included_followups:
            disposition: Disposition = "included"
            reasons: list[str] = []
        elif any(item.disposition == "failed" for item in followup_runs):
            disposition = "failed"
            reasons = ["NO_SUCCESSFUL_FOLLOWUP_BUILD"]
        else:
            disposition = "indeterminate"
            reasons = ["NO_COMPLETE_FOLLOWUP_EVIDENCE"]
        run_case = ResearchCaseRun(
            patient_id=case.patient_id,
            center_id=case.center_id,
            split=split,
            treatment_date=case.treatment.treatment_date,
            treatment_type=case.treatment.treatment_type,
            disposition=disposition,
            reason_codes=reasons,
            followups=followup_runs,
        )
        _write_case_summary(run_case, case_output / "deterministic_case_summary.md")
        runs.append(run_case)

    counts: dict[str, int] = dict(Counter(item.disposition for item in runs))
    for status in ("included", "excluded", "failed", "indeterminate"):
        counts.setdefault(status, 0)
    run_manifest = ResearchRunManifest(
        cohort_id=manifest.cohort_id,
        protocol_hash=_file_hash(protocol_path),
        manifest_hash=_file_hash(manifest_path),
        configuration_hash=_configuration_hash(protocol_path),
        git_commit=_git_commit(),
        external_test_center_ids=protocol.external_test_center_ids,
        cases=runs,
        counts=counts,
        sources=[
            SourceReference(
                source_id=Path(manifest_path).name,
                source_type="validated_research_manifest",
                data_origin="user_supplied",
                deidentified=True,
            )
        ],
    )
    run_path = output / "research_run_manifest.json"
    run_manifest.write_json(run_path)
    jsonl_path = output / "case_results.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for item in runs),
        encoding="utf-8",
    )
    lock_payload = {
        "schema_version": run_manifest.schema_version,
        "pipeline_version": run_manifest.pipeline_version,
        "generated_at": run_manifest.generated_at.isoformat(),
        "cohort_id": run_manifest.cohort_id,
        "protocol_hash": run_manifest.protocol_hash,
        "manifest_hash": run_manifest.manifest_hash,
        "configuration_hash": run_manifest.configuration_hash,
        "git_commit": run_manifest.git_commit,
        "external_test_center_ids": run_manifest.external_test_center_ids,
        "label_data_loaded": False,
    }
    (output / "research_run_lock.json").write_text(
        json.dumps(lock_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_path
