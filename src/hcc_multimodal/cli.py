from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from .case_llm import render_with_optional_llm
from .case_models import (
    CaseResearchSummary,
    ClinicalLabEvidence,
    HpiTimelineEvidence,
    ImagingInterpretationEvidence,
)
from .case_runner import run_case_vlm
from .case_summary import summarize_case
from .clinical_labs import parse_laboratory_report
from .cohort_comparison import run_cohort_comparison
from .contracts import DATA_ORIGINS, PAIRING_STATUSES, build_multimodal_case_evidence
from .deepseek import run_deepseek_audit
from .evaluation import EvaluationCohort, evaluate_cohort_file
from .fusion import fuse_cross_sectional_evidence, fuse_evidence
from .hpi import parse_hpi_timeline
from .imaging import compare_imaging, measure_nifti
from .imaging_adapter import parse_imaging_study
from .labs import load_lab_evidence
from .preview import create_overlay_montage
from .prognosis_data import build_prognosis_cohort, sync_prognosis_data
from .prognosis_models import (
    ControlledPrognosisReport,
    CoxModelBundle,
    PrognosticEvidence,
    PublicPrognosisCohortArtifact,
)
from .prognosis_report import run_prognosis_report
from .prognosis_survival import (
    evaluate_external_prognosis,
    train_prognosis_models,
)
from .prompting import build_report_prompt, write_prompt_bundle
from .public_cohort import prepare_public_cohort
from .registration import register_volumes
from .report_pipeline import run_report_pipeline
from .research_cohort import build_research_cohort, validate_research_cohort
from .research_evaluation import evaluate_research_cohort, validate_adjudications
from .research_models import (
    AdjudicationSet,
    CohortValidationReport,
    ResearchCohortManifest,
    ResearchEvaluationArtifact,
    ResearchProtocol,
    ResearchRunManifest,
)
from .schemas import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    ClinicalVerdict,
    ControlledReport,
    GlmImagingEvidence,
    ImageEmbeddingEvidence,
    ImagingCrosscheckEvidence,
    ImagingEvidence,
    JsonModel,
    LabEvidence,
    LionInspiredImagingEvidence,
    LongitudinalImagingEvidence,
    MultimodalCaseEvidence,
    RegistrationEvidence,
    VlmDemoReport,
    json_schema_for,
)
from .synthetic import generate_synthetic_case
from .tcia import (
    LAB_SCENARIOS,
    convert_ct_and_mass_seg,
    enumerate_hcc_tace_seg_candidates,
    prepare_hcc003,
    prepare_public_case,
    write_composite_labs,
)
from .vlm_llm import imaging_metadata_text, run_vlm_dual_arm_demo
from .vlm_prompting import VlmTaskPrompt, build_vlm_task_prompt, write_vlm_prompt_bundle
from .vlm_skeleton import run_skeleton_demo
from .volume_encoders import available_volume_encoders, encode_nifti_volume


def _schema_models() -> dict[str, type[JsonModel]]:
    return {
        "clinical-verdict": ClinicalVerdict,
        "controlled-report": ControlledReport,
        "controlled-prognosis-report": ControlledPrognosisReport,
        "cox-model-bundle": CoxModelBundle,
        "glm-imaging-evidence": GlmImagingEvidence,
        "evaluation-cohort": EvaluationCohort,
        "image-embedding-evidence": ImageEmbeddingEvidence,
        "imaging-evidence": ImagingEvidence,
        "imaging-crosscheck-evidence": ImagingCrosscheckEvidence,
        "lab-evidence": LabEvidence,
        "longitudinal-imaging-evidence": LongitudinalImagingEvidence,
        "lion-inspired-imaging-evidence": LionInspiredImagingEvidence,
        "multimodal-case-evidence": MultimodalCaseEvidence,
        "prognostic-evidence": PrognosticEvidence,
        "public-prognosis-cohort": PublicPrognosisCohortArtifact,
        "research-protocol": ResearchProtocol,
        "research-cohort-manifest": ResearchCohortManifest,
        "cohort-validation-report": CohortValidationReport,
        "research-run-manifest": ResearchRunManifest,
        "adjudication-set": AdjudicationSet,
        "research-evaluation": ResearchEvaluationArtifact,
        "imaging-interpretation-evidence": ImagingInterpretationEvidence,
        "clinical-lab-evidence": ClinicalLabEvidence,
        "hpi-timeline-evidence": HpiTimelineEvidence,
        "case-research-summary": CaseResearchSummary,
        "vlm-task-prompt": VlmTaskPrompt,
        "vlm-demo-report": VlmDemoReport,
    }


def _encoder_options(
    *,
    device: str,
    allow_m3d_remote_code: bool,
    allow_model_download: bool,
) -> dict[str, object]:
    return {
        "device": device,
        "allow_remote_code": allow_m3d_remote_code,
        "allow_model_download": allow_model_download,
    }


def _encode_optional(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    patient_id: str,
    study_date: str,
    encoder_name: str | None,
    output: Path,
    stem: str,
    encoder_device: str,
    allow_m3d_remote_code: bool,
    allow_model_download: bool,
    modality: str = "CT",
    phase: str = "unknown",
):
    if not encoder_name or encoder_name == "none":
        return None
    evidence, _ = encode_nifti_volume(
        image_path,
        mask_path,
        patient_id=patient_id,
        study_date=study_date,
        encoder_name=encoder_name,
        artifact_path=output / f"{stem}.npy",
        encoder_options=_encoder_options(
            device=encoder_device,
            allow_m3d_remote_code=allow_m3d_remote_code,
            allow_model_download=allow_model_download,
        ),
        modality=modality,
        phase=phase,
    )
    evidence.write_json(output / f"{stem}_evidence.json")
    return evidence


def run_case(
    *,
    baseline_image: str | Path,
    baseline_mask: str | Path,
    followup_image: str | Path,
    followup_mask: str | Path,
    labs_path: str | Path,
    patient_id: str,
    baseline_date: str,
    followup_date: str,
    output_dir: str | Path,
    modality: str = "CT",
    baseline_phase: str = "unknown",
    followup_phase: str = "unknown",
    image_encoder: str | None = None,
    encoder_device: str = "auto",
    allow_m3d_remote_code: bool = False,
    allow_model_download: bool = False,
    pairing_status: str = "user_supplied_unverified",
    imaging_origin: str = "user_supplied",
    lab_origin: str = "user_supplied",
    data_relationship: str = (
        "User-supplied imaging and laboratory inputs; subject pairing has not been verified by the demo"
    ),
    registration_status: Literal[
        "verified", "assumed_same_grid", "failed", "unavailable"
    ] | None = None,
    registration_mode: Literal["auto", "off"] = "auto",
    treatment_events: list[dict[str, object]] | None = None,
) -> Path:
    output = Path(output_dir)
    registration_evidence: RegistrationEvidence | None = None
    if registration_mode == "auto" and registration_status is None:
        outcome = register_volumes(
            baseline_image,
            baseline_mask,
            followup_image,
            followup_mask,
            output / "registration",
        )
        registration_evidence = outcome.evidence
        if outcome.evidence.status == "verified" and outcome.image_path and outcome.mask_path:
            followup_image = outcome.image_path
            followup_mask = outcome.mask_path
            registration_status = "verified"
        elif outcome.evidence.status == "failed":
            registration_status = "failed"
        # "unavailable" keeps registration_status=None so legacy geometry inference applies
    baseline = measure_nifti(
        baseline_image, baseline_mask,
        patient_id=patient_id,
        study_date=baseline_date,
        modality=modality,
        phase=baseline_phase,
    )
    followup = measure_nifti(
        followup_image, followup_mask,
        patient_id=patient_id,
        study_date=followup_date,
        modality=modality,
        phase=followup_phase,
    )
    longitudinal = compare_imaging(
        baseline,
        followup,
        registration_status=registration_status,
        registration=registration_evidence,
    )
    labs = load_lab_evidence(labs_path, index_time=followup_date)
    verdict = fuse_evidence(labs, longitudinal, treatment_events=treatment_events)

    _encode_optional(
        baseline_image,
        baseline_mask,
        patient_id=patient_id,
        study_date=baseline_date,
        encoder_name=image_encoder,
        output=output,
        stem="baseline_image_embedding",
        encoder_device=encoder_device,
        allow_m3d_remote_code=allow_m3d_remote_code,
        allow_model_download=allow_model_download,
        modality=modality,
        phase=baseline_phase,
    )
    followup_embedding = _encode_optional(
        followup_image,
        followup_mask,
        patient_id=patient_id,
        study_date=followup_date,
        encoder_name=image_encoder,
        output=output,
        stem="followup_image_embedding",
        encoder_device=encoder_device,
        allow_m3d_remote_code=allow_m3d_remote_code,
        allow_model_download=allow_model_download,
        modality=modality,
        phase=followup_phase,
    )

    baseline.write_json(output / "baseline_imaging_evidence.json")
    followup.write_json(output / "followup_imaging_evidence.json")
    longitudinal.write_json(output / "longitudinal_imaging_evidence.json")
    labs.write_json(output / "lab_evidence.json")
    verdict_path = output / "clinical_verdict.json"
    verdict.write_json(verdict_path)
    case_evidence = build_multimodal_case_evidence(
        imaging=followup,
        labs=labs,
        index_time=followup_date,
        pairing_status=pairing_status,
        data_relationship=data_relationship,
        imaging_origin=imaging_origin,
        lab_origin=lab_origin,
        image_embedding=followup_embedding,
    )
    case_evidence.write_json(output / "multimodal_case_evidence.json")
    prompt = build_report_prompt(
        imaging_evidence={
            "baseline": baseline.to_dict(),
            "followup": followup.to_dict(),
            "longitudinal_comparison": longitudinal.to_dict(),
        },
        lab_evidence=labs,
        verdict=verdict,
        scenario_id="synthetic_longitudinal_progression",
        report_type="longitudinal_synthetic_demo",
        data_relationship=data_relationship,
        imaging_source_type=(
            "synthetic CT volumes with generated segmentation masks"
            if imaging_origin == "synthetic"
            else "user-supplied longitudinal 3D imaging with aligned segmentation masks"
        ),
        lab_source_type=(
            "synthetic AFP/DCP observations from the same generated case"
            if lab_origin == "synthetic"
            else "user-supplied AFP/DCP observations"
        ),
    )
    write_prompt_bundle(
        prompt,
        json_path=output / "llm_prompt.json",
        text_path=output / "llm_prompt.txt",
    )
    return verdict_path


def run_demo(
    output_dir: str | Path,
    *,
    image_encoder: str | None = "statistical-v1",
    encoder_device: str = "auto",
    allow_m3d_remote_code: bool = False,
    allow_model_download: bool = False,
) -> Path:
    output = Path(output_dir)
    paths = generate_synthetic_case(output / "input")
    return run_case(
        baseline_image=paths["baseline_image"],
        baseline_mask=paths["baseline_mask"],
        followup_image=paths["followup_image"],
        followup_mask=paths["followup_mask"],
        labs_path=paths["labs"],
        patient_id="DEMO_HCC_001",
        baseline_date="2026-01-15",
        followup_date="2026-07-15",
        output_dir=output,
        image_encoder=image_encoder,
        encoder_device=encoder_device,
        allow_m3d_remote_code=allow_m3d_remote_code,
        allow_model_download=allow_model_download,
        pairing_status="same_subject",
        imaging_origin="synthetic",
        lab_origin="synthetic",
        data_relationship=(
            "All imaging and laboratory inputs are synthetic and belong to one generated demo case"
        ),
        registration_status="assumed_same_grid",
        baseline_phase="synthetic_single_phase",
        followup_phase="synthetic_single_phase",
    )


def run_public_demo(
    output_dir: str | Path,
    *,
    image_encoder: str | None = "statistical-v1",
    encoder_device: str = "auto",
    allow_m3d_remote_code: bool = False,
    allow_model_download: bool = False,
) -> Path:
    output = Path(output_dir)
    patient_id = "COMPOSITE_PUBLIC_HCC_003"
    image_path, mask_path, attribution_path = prepare_hcc003(output / "public-case")
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    create_overlay_montage(
        image_path,
        mask_path,
        output / "public-case" / "converted" / "hcc003_overlay.png",
    )
    imaging = measure_nifti(
        image_path,
        mask_path,
        patient_id=patient_id,
        study_date="1997-09-12",
        modality="CT",
        phase=str(attribution.get("phase", "unknown")),
        provider="TCIA-HCC-TACE-Seg-DICOM-SEG",
        inference_mode="public_expert_annotation",
        minimum_lesion_volume_ml=1.0,
        frame_of_reference_uid=attribution.get("ct_frame_of_reference_uid"),
        study_instance_uid=attribution.get("study_instance_uid"),
        series_instance_uid=attribution.get("ct_series_instance_uid"),
        segment_series_instance_uid=attribution.get("seg_series_instance_uid"),
        segment_number=attribution.get("selected_segment_number"),
        acquisition_id=attribution.get("selected_acquisition_id"),
        source_quality_warnings=attribution.get("geometry_qc", {}).get("warnings", []),
    )
    imaging.write_json(output / "public_imaging_evidence.json")
    image_embedding = _encode_optional(
        image_path,
        mask_path,
        patient_id=patient_id,
        study_date="1997-09-12",
        encoder_name=image_encoder,
        output=output,
        stem="public_image_embedding",
        encoder_device=encoder_device,
        allow_m3d_remote_code=allow_m3d_remote_code,
        allow_model_download=allow_model_download,
        modality="CT",
        phase=str(attribution.get("phase", "unknown")),
    )
    scenario_results = []
    verdict_path = output / "public_composite_verdict.json"
    for scenario_id, scenario in LAB_SCENARIOS.items():
        scenario_dir = output / "scenarios" / scenario_id
        lab_path = write_composite_labs(
            scenario_dir / "synthetic_labs.json",
            patient_id,
            scenario_id,
        )
        labs = load_lab_evidence(lab_path, index_time="1997-09-12")
        verdict = fuse_cross_sectional_evidence(labs, imaging)
        labs.write_json(scenario_dir / "lab_evidence.json")
        verdict.write_json(scenario_dir / "verdict.json")
        data_relationship = (
            "CT 与专家 Mass mask 来自真实公开影像病例；AFP/DCP 是为验证流程而构造的"
            "合成 PoC 场景，不是该公开影像病例的真实检验结果。"
        )
        case_evidence = build_multimodal_case_evidence(
            imaging=imaging,
            labs=labs,
            index_time="1997-09-12",
            pairing_status="unpaired_poc_composite",
            data_relationship=data_relationship,
            imaging_origin="real_public",
            lab_origin="synthetic",
            image_embedding=image_embedding,
        )
        case_evidence.write_json(scenario_dir / "multimodal_case_evidence.json")
        prompt = build_report_prompt(
            imaging_evidence=imaging.to_dict(),
            lab_evidence=labs,
            verdict=verdict,
            scenario_id=scenario_id,
            report_type="cross_sectional_public_imaging_poc",
            data_relationship=data_relationship,
            imaging_source_type="real public CT with expert DICOM Mass segmentation",
            lab_source_type="declared synthetic PoC AFP/DCP values",
        )
        write_prompt_bundle(
            prompt,
            json_path=scenario_dir / "llm_prompt.json",
            text_path=scenario_dir / "llm_prompt.txt",
        )
        scenario_results.append(
            {
                "scenario_id": scenario_id,
                "description": scenario["description"],
                "afp_latest": labs.markers["AFP"].latest_value,
                "afp_direction": labs.markers["AFP"].direction,
                "dcp_latest": labs.markers["DCP"].latest_value,
                "dcp_direction": labs.markers["DCP"].direction,
                "fusion_state": verdict.state,
                "modality_concordance": verdict.modality_concordance,
                "reason_codes": verdict.reason_codes,
            }
        )
        if scenario_id == "dual_marker_rising":
            labs.write_json(output / "public_composite_lab_evidence.json")
            verdict.write_json(verdict_path)
            case_evidence.write_json(output / "public_multimodal_case_evidence.json")
            write_prompt_bundle(
                prompt,
                json_path=output / "public_llm_prompt.json",
                text_path=output / "public_llm_prompt.txt",
            )
    scenario_matrix = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [{"source_id": "HCC_003", "source_type": "public_composite_scenarios"}],
        "scenarios": scenario_results,
    }
    (output / "scenario_matrix.json").write_text(
        json.dumps(scenario_matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return verdict_path


def _add_encoder_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: str,
) -> None:
    parser.add_argument(
        "--image-encoder",
        choices=("none", *available_volume_encoders()),
        default=default,
        help="Optional 3D representation provider; M3D is never downloaded implicitly",
    )
    parser.add_argument(
        "--encoder-device",
        default="auto",
        help="Encoder device, for example auto, cpu, or cuda",
    )
    parser.add_argument(
        "--allow-m3d-remote-code",
        action="store_true",
        help="Explicitly allow the pinned M3D Hugging Face custom model code",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow downloading the pinned optional model when it is not cached",
    )


def _write_three_line_outputs(
    *,
    imaging: ImagingInterpretationEvidence,
    labs: ClinicalLabEvidence,
    timeline: HpiTimelineEvidence | None,
    output: str | Path,
    llm_response: str | Path | None = None,
) -> tuple[Path, Path]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    imaging_path = target / "imaging.json"
    labs_path = target / "labs.json"
    imaging.write_json(imaging_path)
    labs.write_json(labs_path)
    if timeline is not None:
        timeline.write_json(target / "timeline.json")
    summary = summarize_case(imaging, labs, timeline)
    summary_path = target / "case-summary.json"
    summary.write_json(summary_path)
    raw_response = Path(llm_response).read_text(encoding="utf-8") if llm_response else None
    markdown, audit = render_with_optional_llm(summary, raw_response)
    if llm_response:
        audit["response_source"] = Path(llm_response).name
        audit["raw_response"] = raw_response
    (target / "case-summary.md").write_text(markdown, encoding="utf-8")
    (target / "report-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path, target / "case-summary.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="HCC multimodal evidence demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate a synthetic NIfTI/mask/lab case")
    generate.add_argument("--output", default="demo-output/input")
    run = subparsers.add_parser("run-demo", help="Generate and run the complete synthetic case")
    run.add_argument("--output", default="demo-output")
    _add_encoder_arguments(run, default="statistical-v1")
    run_report = subparsers.add_parser(
        "run-report",
        help="Run the unified LiON-inspired/GLM/laboratory/DeepSeek report pipeline",
    )
    run_report.add_argument("--case-input", required=True)
    run_report.add_argument(
        "--output",
        default=None,
        help="Override the output_dir declared in the case input",
    )
    run_report.add_argument("--glm-mode", choices=("off", "live"), default="off")
    run_report.add_argument(
        "--report-mode",
        choices=("deterministic", "live"),
        default="deterministic",
    )
    run_report.add_argument("--require-live-models", action="store_true")
    run_report.add_argument("--timeout", type=float, default=180.0)
    sync_prognosis = subparsers.add_parser(
        "sync-prognosis-data",
        help="Synchronize licensed WAW-TACE or HCC-TACE-Seg prognosis research data",
    )
    sync_prognosis.add_argument(
        "--dataset", choices=("waw-tace", "hcc-tace-seg"), required=True
    )
    sync_prognosis.add_argument(
        "--tier", choices=("metadata", "pilot", "full"), required=True
    )
    sync_prognosis.add_argument("--data-root", required=True)
    sync_prognosis.add_argument("--accept-license", action="store_true")
    build_prognosis = subparsers.add_parser(
        "build-prognosis-cohort",
        help="Build physically separated baseline features and OS endpoints",
    )
    build_prognosis.add_argument("--data-root", required=True)
    build_prognosis.add_argument("--output", required=True)
    train_prognosis = subparsers.add_parser(
        "train-prognosis-model",
        help="Train frozen WAW-TACE penalized Cox model bundles",
    )
    train_prognosis.add_argument("--cohort", required=True)
    train_prognosis.add_argument("--output", required=True)
    train_prognosis.add_argument("--seed", type=int, default=1729)
    train_prognosis.add_argument("--bootstrap-iterations", type=int, default=1000)
    external_prognosis = subparsers.add_parser(
        "evaluate-prognosis-model",
        help="Open the locked HCC-TACE-Seg external test exactly once per output",
    )
    external_prognosis.add_argument("--cohort", required=True)
    external_prognosis.add_argument("--models", required=True)
    external_prognosis.add_argument("--output", required=True)
    external_prognosis.add_argument("--unlock-external", action="store_true")
    external_prognosis.add_argument("--seed", type=int, default=1729)
    external_prognosis.add_argument("--bootstrap-iterations", type=int, default=1000)
    run_prognosis = subparsers.add_parser(
        "run-prognosis-report",
        help="Run LiON-inspired/GLM/frozen-Cox/DeepSeek controlled prognosis reporting",
    )
    run_prognosis.add_argument("--case-input", required=True)
    run_prognosis.add_argument("--model", required=True)
    run_prognosis.add_argument("--output", default=None)
    run_prognosis.add_argument("--glm-mode", choices=("off", "live"), default="off")
    run_prognosis.add_argument(
        "--report-mode", choices=("deterministic", "live"), default="deterministic"
    )
    run_prognosis.add_argument("--timeout", type=float, default=180.0)
    public = subparsers.add_parser(
        "run-public-demo",
        help="Download TCIA HCC_003, convert its Mass SEG to NIfTI, and run a composite demo",
    )
    public.add_argument("--output", default="public-data/HCC_003")
    _add_encoder_arguments(public, default="statistical-v1")
    analyze = subparsers.add_parser("analyze", help="Analyze supplied NIfTI images, masks, and labs")
    analyze.add_argument("--baseline-image", required=True)
    analyze.add_argument("--baseline-mask", required=True)
    analyze.add_argument("--followup-image", required=True)
    analyze.add_argument("--followup-mask", required=True)
    analyze.add_argument("--labs", required=True)
    analyze.add_argument("--patient-id", required=True)
    analyze.add_argument("--baseline-date", required=True, help="ISO date, for example 2026-01-15")
    analyze.add_argument("--followup-date", required=True, help="ISO date, for example 2026-07-15")
    analyze.add_argument("--modality", choices=("CT", "MR"), default="CT")
    analyze.add_argument("--baseline-phase", default="unknown")
    analyze.add_argument("--followup-phase", default="unknown")
    analyze.add_argument(
        "--registration-status",
        choices=("verified", "assumed_same_grid", "failed", "unavailable"),
        default=None,
        help="Declared registration QC; omitted values are inferred conservatively from geometry",
    )
    analyze.add_argument(
        "--registration-mode",
        choices=("auto", "off"),
        default="auto",
        help=(
            "auto rigidly registers follow-up onto baseline via SimpleITK when "
            "--registration-status is not declared (degrades to legacy inference "
            "when SimpleITK is missing); off skips registration"
        ),
    )
    analyze.add_argument(
        "--treatment-events",
        default=None,
        help="Optional JSON file containing an array of intervening treatment events",
    )
    analyze.add_argument("--output", default="demo-output")
    analyze.add_argument("--imaging-origin", choices=DATA_ORIGINS, default="user_supplied")
    analyze.add_argument("--lab-origin", choices=DATA_ORIGINS, default="user_supplied")
    analyze.add_argument(
        "--pairing-status",
        choices=PAIRING_STATUSES,
        default="user_supplied_unverified",
    )
    analyze.add_argument(
        "--data-relationship",
        default=(
            "User-supplied imaging and laboratory inputs; subject pairing has not been "
            "verified by the demo"
        ),
    )
    _add_encoder_arguments(analyze, default="none")
    encode = subparsers.add_parser(
        "encode-volume",
        help="Encode one NIfTI volume into a separate, auditable representation artifact",
    )
    encode.add_argument("--image", required=True)
    encode.add_argument("--mask", default=None)
    encode.add_argument("--patient-id", required=True)
    encode.add_argument("--study-date", required=True)
    encode.add_argument("--modality", choices=("CT", "MR"), default="CT")
    encode.add_argument("--phase", default="unknown")
    encode.add_argument("--output", default="embedding-output")
    _add_encoder_arguments(encode, default="statistical-v1")
    deepseek = subparsers.add_parser(
        "run-deepseek",
        help="Send one generated scenario prompt to DeepSeek and validate the JSON report",
    )
    deepseek.add_argument(
        "--scenario",
        default="dual_marker_rising",
        choices=tuple(LAB_SCENARIOS),
    )
    deepseek.add_argument("--public-dir", default="public-data/HCC_003")
    deepseek.add_argument("--output", default=None)
    convert = subparsers.add_parser(
        "convert-dicom-seg",
        help="Convert one CT acquisition and a DICOM SEG Mass segment to aligned NIfTI",
    )
    convert.add_argument("--ct-dir", required=True)
    convert.add_argument("--seg", required=True)
    convert.add_argument("--output", required=True)
    evaluate = subparsers.add_parser(
        "evaluate-cohort",
        help="Evaluate deterministic baselines on a patient-level research cohort contract",
    )
    evaluate.add_argument("--cohort", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--bootstrap-iterations", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=1729)
    export = subparsers.add_parser(
        "export-schemas",
        help="Export versioned JSON Schemas for all public artifacts",
    )
    export.add_argument("--output", default="schemas")
    validate = subparsers.add_parser(
        "validate-json",
        help="Validate one JSON artifact against a public Pydantic contract",
    )
    validate.add_argument("--type", required=True, choices=tuple(_schema_models()))
    validate.add_argument("--input", required=True)
    validate_cohort = subparsers.add_parser(
        "validate-research-cohort",
        help="Validate a deidentified multicenter research protocol and cohort manifest",
    )
    validate_cohort.add_argument("--protocol", required=True)
    validate_cohort.add_argument("--manifest", required=True)
    validate_cohort.add_argument("--output", required=True)
    build_cohort = subparsers.add_parser(
        "build-research-cohort",
        help="Build label-free deterministic evidence for a validated research cohort",
    )
    build_cohort.add_argument("--protocol", required=True)
    build_cohort.add_argument("--manifest", required=True)
    build_cohort.add_argument("--output", required=True)
    validate_labels = subparsers.add_parser(
        "validate-adjudications",
        help="Validate blinded review records against a locked research run",
    )
    validate_labels.add_argument("--run", required=True)
    validate_labels.add_argument("--adjudications", required=True)
    validate_labels.add_argument("--output", required=True)
    validate_labels.add_argument("--external-unlock-audit", default=None)
    evaluate_research = subparsers.add_parser(
        "evaluate-research-cohort",
        help="Evaluate locked deterministic baselines using separate adjudication labels",
    )
    evaluate_research.add_argument("--protocol", required=True)
    evaluate_research.add_argument("--manifest", required=True)
    evaluate_research.add_argument("--run", required=True)
    evaluate_research.add_argument("--adjudications", required=True)
    evaluate_research.add_argument("--output", required=True)
    evaluate_research.add_argument(
        "--scope",
        choices=("development", "external_test"),
        required=True,
    )
    evaluate_research.add_argument(
        "--unlock-external",
        action="store_true",
        help="Explicitly authorize one external-test label access in this output directory",
    )
    list_candidates = subparsers.add_parser(
        "list-public-candidates",
        help=(
            "Enumerate HCC-TACE-Seg patients with CT+SEG series from the IDC index "
            "(metadata only, no downloads)"
        ),
    )
    list_candidates.add_argument("--output", required=True)
    prepare_case = subparsers.add_parser(
        "prepare-public-case",
        help=(
            "Download one HCC-TACE-Seg patient's CT+SEG series and convert to "
            "aligned NIfTI (CT series auto-discovered from the SEG when omitted)"
        ),
    )
    prepare_case.add_argument("--patient", required=True)
    prepare_case.add_argument("--seg-series", required=True)
    prepare_case.add_argument("--ct-series", default=None)
    prepare_case.add_argument("--output", required=True)
    prepare_batch = subparsers.add_parser(
        "prepare-public-batch",
        help=(
            "Prepare multiple HCC-TACE-Seg cases (download, convert, measure, "
            "overlay) using the candidate manifest; resumable across runs"
        ),
    )
    prepare_batch.add_argument(
        "--patients",
        required=True,
        help="Comma-separated HCC patient IDs, e.g. HCC_004,HCC_014",
    )
    prepare_batch.add_argument(
        "--candidates",
        required=True,
        help="Path to docs/hcc_tace_seg_candidates.json",
    )
    prepare_batch.add_argument("--output", required=True)
    run_comparison = subparsers.add_parser(
        "run-cohort-comparison",
        help=(
            "Run the dual-mode VLM comparison over synthetic laboratory "
            "scenarios for public cases; resumable across runs"
        ),
    )
    run_comparison.add_argument(
        "--cases",
        required=True,
        help="Comma-separated patient IDs, e.g. HCC_003,HCC_004",
    )
    run_comparison.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario IDs (default: all three)",
    )
    run_comparison.add_argument("--cohort-dir", default="public-data/cohort")
    run_comparison.add_argument("--hcc003-dir", default="public-data/HCC_003")
    run_comparison.add_argument("--max-tokens", type=int, default=1024)
    run_comparison.add_argument("--temperature", type=float, default=0.3)
    run_comparison.add_argument("--timeout", type=float, default=120.0)
    run_comparison.add_argument("--no-json-object", action="store_true")
    run_comparison.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run already-completed case x scenario comparisons",
    )
    run_case_vlm_parser = subparsers.add_parser(
        "run-case-vlm",
        help=(
            "One-command end-to-end: case or NIfTI + labs/scenario -> dual-mode "
            "VLM -> web_demo.json"
        ),
    )
    run_case_vlm_parser.add_argument(
        "--case",
        default=None,
        help="Prepared case ID (cohort dir or HCC_003)",
    )
    run_case_vlm_parser.add_argument(
        "--scenario",
        default=None,
        help="Synthetic lab scenario anchored to the case study date",
    )
    run_case_vlm_parser.add_argument(
        "--labs",
        default=None,
        help="Lab report text or ClinicalLabEvidence JSON",
    )
    run_case_vlm_parser.add_argument("--image", action="append", default=[])
    run_case_vlm_parser.add_argument("--imaging-evidence", default=None)
    run_case_vlm_parser.add_argument("--nifti-image", default=None)
    run_case_vlm_parser.add_argument("--nifti-mask", default=None)
    run_case_vlm_parser.add_argument("--study-date", default=None)
    run_case_vlm_parser.add_argument("--cohort-dir", default="public-data/cohort")
    run_case_vlm_parser.add_argument("--hcc003-dir", default="public-data/HCC_003")
    run_case_vlm_parser.add_argument("--output", required=True)
    run_case_vlm_parser.add_argument(
        "--fusion-mode",
        choices=("auditable", "open", "both"),
        default="both",
    )
    run_case_vlm_parser.add_argument("--visual-token-count", type=int, default=32)
    run_case_vlm_parser.add_argument("--max-tokens", type=int, default=1024)
    run_case_vlm_parser.add_argument("--temperature", type=float, default=0.3)
    run_case_vlm_parser.add_argument("--timeout", type=float, default=120.0)
    run_case_vlm_parser.add_argument("--no-json-object", action="store_true")
    vlm_web_cmd = subparsers.add_parser(
        "vlm-web",
        help=(
            "Launch the local visualization service: upload imaging + lab report "
            "and get an auditable interpretation"
        ),
    )
    vlm_web_cmd.add_argument("--port", type=int, default=7861)
    vlm_web_cmd.add_argument(
        "--index",
        default=None,
        help="Path to index.html (default: docs/demo/index.html)",
    )
    parse_imaging = subparsers.add_parser("parse-imaging", help="Validate a CT/MR DICOM series and optional SEG")
    parse_imaging.add_argument("--dicom-dir", required=True)
    parse_imaging.add_argument("--seg", default=None)
    parse_imaging.add_argument("--image-evidence", default=None)
    parse_imaging.add_argument("--patient-id", required=True)
    parse_imaging.add_argument("--phase", default=None)
    parse_imaging.add_argument("--output", required=True)
    parse_labs = subparsers.add_parser("parse-labs", help="Parse TXT, JSON, or CSV laboratory evidence")
    parse_labs.add_argument("--input", required=True)
    parse_labs.add_argument("--patient-id", required=True)
    parse_labs.add_argument("--output", required=True)
    parse_hpi = subparsers.add_parser("parse-hpi", help="Extract a deterministic HPI timeline")
    parse_hpi.add_argument("--input", required=True)
    parse_hpi.add_argument("--patient-id", required=True)
    parse_hpi.add_argument("--labs", default=None)
    parse_hpi.add_argument("--imaging", default=None)
    parse_hpi.add_argument("--index-date", default=None, help="Optional ISO cutoff; later events are excluded")
    parse_hpi.add_argument("--output", required=True)
    summarize = subparsers.add_parser("summarize-case", help="Fuse validated imaging, labs, and HPI artifacts")
    summarize.add_argument("--imaging", required=True)
    summarize.add_argument("--labs", required=True)
    summarize.add_argument("--timeline", default=None)
    summarize.add_argument("--llm-response", default=None, help="Optional offline JSON rewrite to validate; external LLM calls remain disabled")
    summarize.add_argument("--output", required=True)
    analyze_case = subparsers.add_parser("analyze-case", help="Run all three CPU-friendly evidence lines")
    analyze_case.add_argument("--case-input", default=None, help="Unified patient-level case JSON; replaces the individual input arguments")
    analyze_case.add_argument("--dicom-dir", default=None)
    analyze_case.add_argument("--seg", default=None)
    analyze_case.add_argument("--image-evidence", default=None)
    analyze_case.add_argument("--labs", default=None)
    analyze_case.add_argument("--hpi", default=None)
    analyze_case.add_argument("--patient-id", default=None)
    analyze_case.add_argument("--phase", default=None)
    analyze_case.add_argument("--llm-response", default=None, help="Optional offline JSON rewrite to validate; external LLM calls remain disabled")
    analyze_case.add_argument("--output", default=None)
    vlm_skeleton = subparsers.add_parser(
        "vlm-skeleton",
        help="Run a CPU-only patch extraction and mock VLM demonstration",
    )
    vlm_skeleton.add_argument("--image", required=True, help="JPG/PNG image or 3D NIfTI volume")
    vlm_skeleton.add_argument("--hpi", default="", help="Optional HPI text")
    vlm_skeleton.add_argument("--labs", default="", help="Optional laboratory report text")
    vlm_skeleton.add_argument("--output", required=True)
    vlm_skeleton.add_argument(
        "--patch-size",
        nargs="+",
        type=int,
        default=None,
        help="Optional patch dimensions: H W for images, or D H W for NIfTI",
    )
    vlm_web = subparsers.add_parser(
        "vlm-skeleton-web",
        help="Launch the local static browser UI for the patch-based mock VLM demo",
    )
    vlm_web.add_argument("--port", type=int, default=7860)
    vlm_web.add_argument("--share", action="store_true")
    vlm_prompt = subparsers.add_parser(
        "vlm-prompt",
        help=(
            "Build auditable/open VLM task prompts for one case; 'open' injects "
            "laboratory observations as UNVERIFIED_CONTEXT"
        ),
    )
    vlm_prompt.add_argument(
        "--labs",
        default=None,
        help="Optional ClinicalLabEvidence JSON (analyze-case or parse-labs output)",
    )
    vlm_prompt.add_argument(
        "--timeline",
        default=None,
        help="Optional HpiTimelineEvidence JSON (analyze-case or parse-hpi output)",
    )
    vlm_prompt.add_argument("--phase", default="unknown")
    vlm_prompt.add_argument(
        "--timepoint",
        choices=("baseline", "followup", "single"),
        default="single",
    )
    vlm_prompt.add_argument("--visual-token-count", type=int, default=32)
    vlm_prompt.add_argument(
        "--fusion-mode",
        choices=("auditable", "open", "both"),
        default="both",
        help=(
            "auditable withholds labs from the VLM prompt; open injects labs as "
            "UNVERIFIED_CONTEXT; both writes both bundles"
        ),
    )
    vlm_prompt.add_argument("--output", required=True)
    vlm_live = subparsers.add_parser(
        "vlm-live",
        help=(
            "Run dual-mode VLM arms against a real OpenAI-compatible LLM with "
            "fail-closed audit and per-number citation extraction"
        ),
    )
    vlm_live.add_argument(
        "--labs",
        required=True,
        help="ClinicalLabEvidence JSON (analyze-case or parse-labs output)",
    )
    vlm_live.add_argument(
        "--timeline",
        default=None,
        help="Optional HpiTimelineEvidence JSON (analyze-case or parse-hpi output)",
    )
    vlm_live.add_argument(
        "--image",
        action="append",
        default=[],
        help=(
            "Real image(s) to attach as vision input (PNG/JPG/WEBP); "
            "replaces <im_patch> placeholders"
        ),
    )
    vlm_live.add_argument(
        "--imaging-evidence",
        default=None,
        help=(
            "Optional ImagingEvidence / LongitudinalImagingEvidence JSON; its "
            "deterministic measurements are injected as prompt metadata"
        ),
    )
    vlm_live.add_argument("--phase", default="unknown")
    vlm_live.add_argument(
        "--timepoint",
        choices=("baseline", "followup", "single"),
        default="single",
    )
    vlm_live.add_argument(
        "--fusion-mode",
        choices=("auditable", "open", "both"),
        default="both",
        help=(
            "auditable withholds labs from the VLM prompt; open injects labs as "
            "UNVERIFIED_CONTEXT; both runs and audits both arms"
        ),
    )
    vlm_live.add_argument("--visual-token-count", type=int, default=32)
    vlm_live.add_argument("--max-tokens", type=int, default=1500)
    vlm_live.add_argument("--timeout", type=float, default=300.0)
    vlm_live.add_argument("--temperature", type=float, default=0.0)
    vlm_live.add_argument(
        "--no-json-object",
        action="store_true",
        help=(
            "Omit response_format=json_object; required for reasoning-class models "
            "(e.g. deepseek-r1-distill) that return empty content in JSON mode"
        ),
    )
    vlm_live.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "generate":
        paths = generate_synthetic_case(args.output)
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "run-report":
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        result = run_report_pipeline(
            args.case_input,
            output_dir=args.output,
            glm_mode=cast(Literal["off", "live"], args.glm_mode),
            report_mode=cast(Literal["deterministic", "live"], args.report_mode),
            require_live_models=args.require_live_models,
            timeout_seconds=args.timeout,
        )
        print(f"Wrote {result.output_dir / 'controlled-report.json'}")
        print(f"Wrote {result.output_dir / 'controlled-report.md'}")
        print(f"Wrote {result.output_dir / 'pipeline-audit.json'}")
        if not result.strict_success:
            parser.exit(
                status=2,
                message=(
                    "Strict live-model acceptance failed; audit artifacts were retained in "
                    f"{result.output_dir}\n"
                ),
            )
    elif args.command == "sync-prognosis-data":
        if (
            os.name == "nt"
            and sys.flags.utf8_mode == 0
            and args.dataset == "hcc-tace-seg"
            and args.tier in {"pilot", "full"}
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "hcc_multimodal.cli",
                    *sys.argv[1:],
                ],
                check=False,
            )
            raise SystemExit(completed.returncode)
        manifest_path = sync_prognosis_data(
            args.dataset,
            tier=args.tier,
            data_root=args.data_root,
            accept_license=args.accept_license,
        )
        print(f"Wrote {manifest_path}")
    elif args.command == "build-prognosis-cohort":
        cohort_path = build_prognosis_cohort(args.data_root, args.output)
        print(f"Wrote {cohort_path}")
    elif args.command == "train-prognosis-model":
        if args.bootstrap_iterations <= 0:
            parser.error("--bootstrap-iterations must be positive")
        validation_path = train_prognosis_models(
            args.cohort,
            args.output,
            seed=args.seed,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(f"Wrote {validation_path}")
    elif args.command == "evaluate-prognosis-model":
        if args.bootstrap_iterations <= 0:
            parser.error("--bootstrap-iterations must be positive")
        external_path = evaluate_external_prognosis(
            args.cohort,
            args.models,
            args.output,
            unlock_external=args.unlock_external,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        print(f"Wrote {external_path}")
    elif args.command == "run-prognosis-report":
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        prognosis_result = run_prognosis_report(
            args.case_input,
            args.model,
            output_dir=args.output,
            glm_mode=cast(Literal["off", "live"], args.glm_mode),
            report_mode=cast(Literal["deterministic", "live"], args.report_mode),
            timeout_seconds=args.timeout,
        )
        print(f"Wrote {prognosis_result.output_dir / 'controlled-prognosis-report.json'}")
        print(f"Wrote {prognosis_result.output_dir / 'controlled-prognosis-report.md'}")
    elif args.command == "run-demo":
        verdict_path = run_demo(
            args.output,
            image_encoder=args.image_encoder,
            encoder_device=args.encoder_device,
            allow_m3d_remote_code=args.allow_m3d_remote_code,
            allow_model_download=args.allow_model_download,
        )
        print(f"Wrote {verdict_path}")
        print(verdict_path.read_text(encoding="utf-8"))
    elif args.command == "run-public-demo":
        verdict_path = run_public_demo(
            args.output,
            image_encoder=args.image_encoder,
            encoder_device=args.encoder_device,
            allow_m3d_remote_code=args.allow_m3d_remote_code,
            allow_model_download=args.allow_model_download,
        )
        print(f"Wrote {verdict_path}")
        print(verdict_path.read_text(encoding="utf-8"))
    elif args.command == "analyze":
        treatment_events = None
        if args.treatment_events:
            treatment_events = json.loads(Path(args.treatment_events).read_text(encoding="utf-8"))
            if not isinstance(treatment_events, list):
                parser.error("--treatment-events JSON must contain an array")
        verdict_path = run_case(
            baseline_image=args.baseline_image,
            baseline_mask=args.baseline_mask,
            followup_image=args.followup_image,
            followup_mask=args.followup_mask,
            labs_path=args.labs,
            patient_id=args.patient_id,
            baseline_date=args.baseline_date,
            followup_date=args.followup_date,
            modality=args.modality,
            baseline_phase=args.baseline_phase,
            followup_phase=args.followup_phase,
            output_dir=args.output,
            image_encoder=args.image_encoder,
            encoder_device=args.encoder_device,
            allow_m3d_remote_code=args.allow_m3d_remote_code,
            allow_model_download=args.allow_model_download,
            pairing_status=args.pairing_status,
            imaging_origin=args.imaging_origin,
            lab_origin=args.lab_origin,
            data_relationship=args.data_relationship,
            registration_status=args.registration_status,
            registration_mode=args.registration_mode,
            treatment_events=treatment_events,
        )
        print(f"Wrote {verdict_path}")
        print(verdict_path.read_text(encoding="utf-8"))
    elif args.command == "encode-volume":
        if args.image_encoder == "none":
            parser.error("encode-volume requires an image encoder other than 'none'")
        output = Path(args.output)
        evidence, artifact_path = encode_nifti_volume(
            args.image,
            args.mask,
            patient_id=args.patient_id,
            study_date=args.study_date,
            encoder_name=args.image_encoder,
            artifact_path=output / "image_embedding.npy",
            encoder_options=_encoder_options(
                device=args.encoder_device,
                allow_m3d_remote_code=args.allow_m3d_remote_code,
                allow_model_download=args.allow_model_download,
            ),
            modality=args.modality,
            phase=args.phase,
        )
        evidence_path = output / "image_embedding_evidence.json"
        evidence.write_json(evidence_path)
        print(f"Wrote {artifact_path}")
        print(f"Wrote {evidence_path}")
        print(evidence_path.read_text(encoding="utf-8"))
    elif args.command == "run-deepseek":
        public_dir = Path(args.public_dir)
        scenario_dir = public_dir / "scenarios" / args.scenario
        output_path = (
            Path(args.output)
            if args.output
            else scenario_dir / "deepseek_result.json"
        )
        result_path = run_deepseek_audit(
            prompt_path=scenario_dir / "llm_prompt.json",
            verdict_path=scenario_dir / "verdict.json",
            output_path=output_path,
            scenario_id=args.scenario,
        )
        print(f"Wrote {result_path}")
        print(result_path.read_text(encoding="utf-8"))
    elif args.command == "convert-dicom-seg":
        image_path, mask_path, attribution_path = convert_ct_and_mass_seg(
            args.ct_dir,
            args.seg,
            args.output,
        )
        print(f"Wrote {image_path}")
        print(f"Wrote {mask_path}")
        print(f"Wrote {attribution_path}")
    elif args.command == "evaluate-cohort":
        if args.bootstrap_iterations <= 0:
            parser.error("--bootstrap-iterations must be positive")
        result_path = evaluate_cohort_file(
            args.cohort,
            args.output,
            split=args.split,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        print(f"Wrote {result_path}")
        print(result_path.read_text(encoding="utf-8"))
    elif args.command == "export-schemas":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for name, model in _schema_models().items():
            target = output / f"{name}.schema.json"
            target.write_text(
                json.dumps(json_schema_for(model), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {target}")
    elif args.command == "validate-json":
        model = _schema_models()[args.type]
        artifact = model.model_validate_json(Path(args.input).read_text(encoding="utf-8"))
        print(f"Valid {args.type} schema_version={getattr(artifact, 'schema_version', 'nested')}")
    elif args.command == "validate-research-cohort":
        result_path = validate_research_cohort(
            args.protocol,
            args.manifest,
            args.output,
        )
        print(f"Wrote {result_path}")
        print(result_path.read_text(encoding="utf-8"))
    elif args.command == "build-research-cohort":
        result_path = build_research_cohort(
            args.protocol,
            args.manifest,
            args.output,
        )
        print(f"Wrote {result_path}")
        print(result_path.read_text(encoding="utf-8"))
    elif args.command == "validate-adjudications":
        result_path = validate_adjudications(
            args.run,
            args.adjudications,
            args.output,
            external_unlock_audit=args.external_unlock_audit,
        )
        print(f"Wrote {result_path}")
        print(result_path.read_text(encoding="utf-8"))
    elif args.command == "evaluate-research-cohort":
        result_path = evaluate_research_cohort(
            args.protocol,
            args.manifest,
            args.run,
            args.adjudications,
            args.output,
            scope=args.scope,
            unlock_external=args.unlock_external,
        )
        print(f"Wrote {result_path}")
        print(result_path.read_text(encoding="utf-8"))
    elif args.command == "list-public-candidates":
        records = enumerate_hcc_tace_seg_candidates(require_seg=True)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "hcc_tace_seg_candidates.json"
        manifest_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        md_lines = [
            "| Patient | 时间点 | CT 系列 | SEG 系列 | 预计 CT+SEG (MB) |",
            "|---|---|---|---|---|",
        ]
        for record in records:
            md_lines.append(
                f"| {record['patient_id']} | {len(record['timepoints'])} "
                f"({', '.join(record['timepoints'][:3])}) | "
                f"{record['ct_series_count']} | {record['seg_series_count']} | "
                f"{record['estimated_ct_seg_mb']} |"
            )
        (output / "hcc_tace_seg_candidates.md").write_text(
            "\n".join(md_lines) + "\n",
            encoding="utf-8",
        )
        total_mb = sum(record["estimated_ct_seg_mb"] for record in records)
        print(f"Wrote {manifest_path}")
        print(f"Wrote {output / 'hcc_tace_seg_candidates.md'}")
        print(
            f"{len(records)} patients with CT+SEG; total estimated "
            f"{total_mb:.0f} MB across all candidates"
        )
    elif args.command == "prepare-public-case":
        image_path, mask_path, attribution_path = prepare_public_case(
            args.output,
            patient_id=args.patient,
            seg_series_uid=args.seg_series,
            ct_series_uid=args.ct_series,
        )
        print(f"Wrote {image_path}")
        print(f"Wrote {mask_path}")
        print(f"Wrote {attribution_path}")
    elif args.command == "prepare-public-batch":
        patients = [item.strip() for item in args.patients.split(",") if item.strip()]
        manifest_path = prepare_public_cohort(
            patients,
            candidates_path=args.candidates,
            output_dir=args.output,
        )
        print(f"Wrote {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"ok={manifest['ok_count']} already={manifest['already_prepared_count']} "
            f"errors={manifest['error_count']}"
        )
        for item in manifest["patients"]:
            if item.get("status") == "ok":
                print(
                    f"  {item['patient_id']}: ok, "
                    f"{item['lesion_count']} lesion(s), "
                    f"{item['total_tumor_volume_ml']} mL"
                )
            else:
                print(f"  {item['patient_id']}: {item.get('status')}")
    elif args.command == "run-cohort-comparison":
        cases = [item.strip() for item in args.cases.split(",") if item.strip()]
        scenarios = (
            [item.strip() for item in args.scenarios.split(",") if item.strip()]
            if args.scenarios
            else None
        )
        summary_path = run_cohort_comparison(
            cases,
            scenarios=scenarios,
            cohort_dir=args.cohort_dir,
            hcc003_dir=args.hcc003_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            json_object=not args.no_json_object,
            timeout_seconds=args.timeout,
            rerun=args.rerun,
        )
        print(f"Wrote {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"runs={summary['run_count']} both_pass={summary['both_pass_count']} "
            f"auditable_pass={summary['auditable_pass_count']} "
            f"open_pass={summary['open_pass_count']} "
            f"hallucinations={summary['total_hallucination_candidates']} "
            f"open_citations={summary['total_open_numeric_citations']}"
        )
    elif args.command == "run-case-vlm":
        if not 8 <= args.visual_token_count <= 64:
            parser.error("--visual-token-count must be between 8 and 64")
        if args.fusion_mode == "both":
            case_modes: tuple[Literal["auditable", "open"], ...] = ("auditable", "open")
        else:
            case_modes = (cast(Literal["auditable", "open"], args.fusion_mode),)
        web_path = run_case_vlm(
            case=args.case,
            scenario=args.scenario,
            labs_path=args.labs,
            image_paths=args.image or None,
            imaging_evidence_path=args.imaging_evidence,
            nifti_image=args.nifti_image,
            nifti_mask=args.nifti_mask,
            study_date=args.study_date,
            cohort_dir=args.cohort_dir,
            hcc003_dir=args.hcc003_dir,
            output_dir=args.output,
            fusion_modes=case_modes,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            json_object=not args.no_json_object,
            timeout_seconds=args.timeout,
        )
        print(f"Wrote {web_path}")
        web = json.loads(web_path.read_text(encoding="utf-8"))
        print(
            f"audit: "
            f"{web['arms'].get('auditable', {}).get('audit_status')} / "
            f"{web['arms'].get('open', {}).get('audit_status')} "
            f"· model={web.get('model')} "
            f"· open_citations={web.get('open_numeric_citation_count')} "
            f"· hallucinations={len(web.get('hallucination_candidates') or [])}"
        )
    elif args.command == "vlm-web":
        from .vlm_web import launch_vlm_web

        project_root = Path(__file__).resolve().parent.parent.parent
        index_path = (
            Path(args.index)
            if args.index
            else project_root / "docs" / "demo" / "index.html"
        )
        launch_vlm_web(index_path=index_path, port=args.port)
    elif args.command == "parse-imaging":
        imaging_result = parse_imaging_study(args.dicom_dir, patient_id=args.patient_id, seg_path=args.seg, image_evidence_path=args.image_evidence, phase=args.phase, output_dir=Path(args.output).parent / ".imaging-work")
        imaging_result.write_json(args.output)
        print(f"Wrote {args.output}")
    elif args.command == "parse-labs":
        labs_result = parse_laboratory_report(args.input, patient_id=args.patient_id)
        labs_result.write_json(args.output)
        print(f"Wrote {args.output}")
    elif args.command == "parse-hpi":
        hpi_labs = ClinicalLabEvidence.model_validate_json(Path(args.labs).read_text(encoding="utf-8")) if args.labs else None
        hpi_imaging = ImagingInterpretationEvidence.model_validate_json(Path(args.imaging).read_text(encoding="utf-8")) if args.imaging else None
        hpi_result = parse_hpi_timeline(args.input, patient_id=args.patient_id, labs=hpi_labs, imaging=hpi_imaging, index_date=args.index_date)
        hpi_result.write_json(args.output)
        print(f"Wrote {args.output}")
    elif args.command == "summarize-case":
        summary_imaging = ImagingInterpretationEvidence.model_validate_json(Path(args.imaging).read_text(encoding="utf-8"))
        summary_labs = ClinicalLabEvidence.model_validate_json(Path(args.labs).read_text(encoding="utf-8"))
        summary_timeline = HpiTimelineEvidence.model_validate_json(Path(args.timeline).read_text(encoding="utf-8")) if args.timeline else None
        summary_path, markdown_path = _write_three_line_outputs(imaging=summary_imaging, labs=summary_labs, timeline=summary_timeline, output=args.output, llm_response=args.llm_response)
        print(f"Wrote {summary_path}")
        print(f"Wrote {markdown_path}")
    elif args.command == "analyze-case":
        case_loader = None
        index_date = None
        if args.case_input:
            from .case_input_loader import CaseInputLoader

            case_loader = CaseInputLoader(args.case_input)
            if case_loader.imaging_source_type != "dicom":
                parser.error(
                    "analyze-case is the legacy DICOM-only entry point; use run-report "
                    "for case_input 1.2 NIfTI inputs"
                )
            args.dicom_dir = str(case_loader.dicom_dir)
            args.seg = str(case_loader.seg_path) if case_loader.seg_path else None
            args.image_evidence = (
                str(case_loader.image_evidence_path)
                if case_loader.image_evidence_path
                else None
            )
            args.labs = str(case_loader.labs_path)
            args.hpi = str(case_loader.hpi_path) if case_loader.hpi_path else None
            args.patient_id = case_loader.patient_id
            args.phase = case_loader.phase or args.phase
            args.output = str(case_loader.output_dir)
            index_date = case_loader.index_date
        missing_args = [
            name
            for name, value in (
                ("--dicom-dir", args.dicom_dir),
                ("--labs", args.labs),
                ("--patient-id", args.patient_id),
                ("--output", args.output),
            )
            if not value
        ]
        if missing_args:
            parser.error(
                "analyze-case requires --case-input or all of " + ", ".join(missing_args)
            )
        if case_loader is not None:
            case_loader.write_normalized_manifest(args.output)
        imaging = parse_imaging_study(args.dicom_dir, patient_id=args.patient_id, seg_path=args.seg, image_evidence_path=args.image_evidence, phase=args.phase, output_dir=args.output)
        labs = parse_laboratory_report(args.labs, patient_id=args.patient_id)
        timeline_cutoff = index_date or (imaging.study_date if imaging.study_date != "unknown" else None)
        timeline = parse_hpi_timeline(args.hpi, patient_id=args.patient_id, labs=labs, imaging=imaging, index_date=timeline_cutoff) if args.hpi else None
        summary_path, markdown_path = _write_three_line_outputs(imaging=imaging, labs=labs, timeline=timeline, output=args.output, llm_response=args.llm_response)
        print(f"Wrote {summary_path}")
        print(f"Wrote {markdown_path}")
    elif args.command == "vlm-skeleton":
        patch_size = tuple(args.patch_size) if args.patch_size else None
        _manifest, report, report_path = run_skeleton_demo(
            args.image,
            args.output,
            hpi=args.hpi,
            labs=args.labs,
            patch_size=patch_size,
        )
        print(f"Wrote {Path(args.output) / 'patch-manifest.json'}")
        print(f"Wrote {Path(args.output) / 'patch-grid.png'}")
        print(f"Wrote {report_path}")
        print(report.to_json())
    elif args.command == "vlm-skeleton-web":
        from .vlm_skeleton_web import launch_demo

        launch_demo(port=args.port, share=args.share)
    elif args.command == "vlm-prompt":
        if not 8 <= args.visual_token_count <= 64:
            parser.error("--visual-token-count must be between 8 and 64")
        prompt_labs = (
            ClinicalLabEvidence.model_validate_json(
                Path(args.labs).read_text(encoding="utf-8")
            )
            if args.labs
            else None
        )
        prompt_timeline = (
            HpiTimelineEvidence.model_validate_json(
                Path(args.timeline).read_text(encoding="utf-8")
            )
            if args.timeline
            else None
        )
        output = Path(args.output)
        if args.fusion_mode == "both":
            modes: tuple[Literal["auditable", "open"], ...] = ("auditable", "open")
        else:
            modes = (cast(Literal["auditable", "open"], args.fusion_mode),)
        for mode in modes:
            prompt = build_vlm_task_prompt(
                labs=prompt_labs,
                timeline=prompt_timeline,
                fusion_mode=mode,
                phase=args.phase,
                timepoint=args.timepoint,
                visual_token_count=args.visual_token_count,
            )
            json_path, text_path = write_vlm_prompt_bundle(
                prompt,
                json_path=output / f"vlm_prompt_{mode}.json",
                text_path=output / f"vlm_prompt_{mode}.txt",
            )
            print(f"[{mode}] Wrote {json_path}")
            print(f"[{mode}] Wrote {text_path}")
    elif args.command == "vlm-live":
        if not 8 <= args.visual_token_count <= 64:
            parser.error("--visual-token-count must be between 8 and 64")
        live_labs = ClinicalLabEvidence.model_validate_json(
            Path(args.labs).read_text(encoding="utf-8")
        )
        live_timeline = (
            HpiTimelineEvidence.model_validate_json(
                Path(args.timeline).read_text(encoding="utf-8")
            )
            if args.timeline
            else None
        )
        if args.fusion_mode == "both":
            live_modes: tuple[Literal["auditable", "open"], ...] = ("auditable", "open")
        else:
            live_modes = (cast(Literal["auditable", "open"], args.fusion_mode),)
        live_imaging_metadata = (
            imaging_metadata_text(args.imaging_evidence)
            if args.imaging_evidence
            else None
        )
        comparison_path = run_vlm_dual_arm_demo(
            labs=live_labs,
            timeline=live_timeline,
            phase=args.phase,
            timepoint=args.timepoint,
            fusion_modes=live_modes,
            output_dir=args.output,
            visual_token_count=args.visual_token_count,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
            temperature=args.temperature,
            json_object=not args.no_json_object,
            image_paths=args.image or None,
            imaging_metadata=live_imaging_metadata,
        )
        print(f"Wrote {comparison_path}")
        print(comparison_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
