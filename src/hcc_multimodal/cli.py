from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import json

from .contracts import DATA_ORIGINS, PAIRING_STATUSES, build_multimodal_case_evidence
from .fusion import fuse_cross_sectional_evidence, fuse_evidence
from .deepseek import run_deepseek_audit
from .evaluation import EvaluationCohort, evaluate_cohort_file
from .imaging import compare_imaging, measure_nifti
from .labs import load_lab_evidence
from .preview import create_overlay_montage
from .prompting import build_report_prompt, write_prompt_bundle
from .synthetic import generate_synthetic_case
from .schemas import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    ClinicalVerdict,
    ControlledReport,
    ImageEmbeddingEvidence,
    ImagingEvidence,
    JsonModel,
    LabEvidence,
    LongitudinalImagingEvidence,
    MultimodalCaseEvidence,
    json_schema_for,
)
from .tcia import (
    LAB_SCENARIOS,
    convert_ct_and_mass_seg,
    prepare_hcc003,
    write_composite_labs,
)
from .volume_encoders import available_volume_encoders, encode_nifti_volume


def _schema_models() -> dict[str, type[JsonModel]]:
    return {
        "clinical-verdict": ClinicalVerdict,
        "controlled-report": ControlledReport,
        "evaluation-cohort": EvaluationCohort,
        "image-embedding-evidence": ImageEmbeddingEvidence,
        "imaging-evidence": ImagingEvidence,
        "lab-evidence": LabEvidence,
        "longitudinal-imaging-evidence": LongitudinalImagingEvidence,
        "multimodal-case-evidence": MultimodalCaseEvidence,
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
    treatment_events: list[dict[str, object]] | None = None,
) -> Path:
    output = Path(output_dir)
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="HCC multimodal evidence demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate a synthetic NIfTI/mask/lab case")
    generate.add_argument("--output", default="demo-output/input")
    run = subparsers.add_parser("run-demo", help="Generate and run the complete synthetic case")
    run.add_argument("--output", default="demo-output")
    _add_encoder_arguments(run, default="statistical-v1")
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
    args = parser.parse_args()

    if args.command == "generate":
        paths = generate_synthetic_case(args.output)
        for name, path in paths.items():
            print(f"{name}: {path}")
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


if __name__ == "__main__":
    main()
