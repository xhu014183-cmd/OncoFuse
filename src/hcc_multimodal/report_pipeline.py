"""Unified LiON-inspired HCC report orchestration shared by CLI and Web."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from .case_input_loader import CaseInputLoader
from .clinical_labs import parse_laboratory_report
from .deepseek import (
    build_deepseek_narrative_prompt,
    build_template_report,
    call_deepseek,
    call_deepseek_narrative,
    validate_deepseek_narrative,
    validate_report,
)
from .glm_vision import run_glm_observer
from .hpi import parse_hpi_timeline
from .imaging_crosscheck import crosscheck_imaging_evidence
from .imaging_normalization import normalize_case_imaging
from .lion_inspired import PrecomputedMaskLionBackend, report_imaging_payload
from .multimodal_fusion import (
    clinical_labs_to_report_labs,
    fuse_lion_multimodal_evidence,
)
from .prompting import build_report_prompt
from .reporting import write_human_markdown
from .schemas import (
    PIPELINE_VERSION,
    ControlledReport,
    DataRelationship,
    DeepseekNarrativeEvidence,
    SourceReference,
)

GlmMode = Literal["off", "live"]
ReportMode = Literal["deterministic", "live"]


@dataclass(frozen=True)
class RunReportResult:
    output_dir: Path
    artifact_index: dict[str, str]
    strict_success: bool
    glm_live_success: bool
    deepseek_live_success: bool
    degraded: bool
    report: ControlledReport


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _safe_provider_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop unvalidated model text and any accidentally supplied credential fields."""
    forbidden = {
        "api_key",
        "authorization",
        "raw_content",
        "request",
        "response",
        "report",
        "prompt",
    }
    return {key: value for key, value in payload.items() if key.casefold() not in forbidden}


def _artifact_index(output: Path) -> dict[str, str]:
    names = (
        "case-input.normalized.json",
        "imaging-input-qc.json",
        "lion-inspired-evidence.json",
        "glm-imaging-evidence.json",
        "imaging-crosscheck.json",
        "clinical-lab-evidence.json",
        "clinical-verdict.json",
        "deepseek-prompt.json",
        "deepseek-narrative-prompt.json",
        "deepseek-narrative.json",
        "controlled-report.json",
        "controlled-report.md",
        "pipeline-audit.json",
        "web_demo.json",
        "provider-audit/glm.json",
        "provider-audit/deepseek.json",
    )
    return {name: str((output / name).resolve()) for name in names}


def run_report_pipeline(
    case_input: str | Path,
    *,
    output_dir: str | Path | None = None,
    glm_mode: GlmMode = "off",
    report_mode: ReportMode = "deterministic",
    require_live_models: bool = False,
    timeout_seconds: float = 180.0,
) -> RunReportResult:
    """Run the complete auditable pipeline and retain artifacts on model failure."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    case = CaseInputLoader(case_input)
    output = Path(output_dir) if output_dir is not None else case.output_dir
    output.mkdir(parents=True, exist_ok=True)
    provider_dir = output / "provider-audit"
    provider_dir.mkdir(parents=True, exist_ok=True)
    case.write_normalized_manifest(output)

    normalized = normalize_case_imaging(case, output_dir=output)
    lion = PrecomputedMaskLionBackend().infer(
        image_path=normalized.image_path,
        mask_path=normalized.mask_path,
        mask_role=normalized.seg_role,
        patient_id=case.patient_id,
        study_date=normalized.study_date,
        phase=normalized.phase,
    )
    lion.write_json(output / "lion-inspired-evidence.json")

    glm_started = perf_counter()
    glm, glm_audit = run_glm_observer(
        lion=lion,
        image_path=normalized.image_path,
        mask_path=normalized.mask_path,
        output_dir=output,
        mode=glm_mode,
        timeout_seconds=timeout_seconds,
    )
    glm_latency_ms = round((perf_counter() - glm_started) * 1000, 3)
    glm.write_json(output / "glm-imaging-evidence.json")
    glm_audit = _safe_provider_audit(
        {
            **glm_audit,
            "latency_ms": glm_latency_ms,
            "response_status": glm.status,
            "live_requested": glm_mode == "live",
        }
    )
    _write_json(provider_dir / "glm.json", glm_audit)

    crosscheck = crosscheck_imaging_evidence(lion, glm)
    crosscheck.write_json(output / "imaging-crosscheck.json")
    clinical_labs = parse_laboratory_report(case.labs_path, patient_id=case.patient_id)
    clinical_labs.write_json(output / "clinical-lab-evidence.json")
    report_labs = clinical_labs_to_report_labs(clinical_labs)

    treatment_dates: list[str] = []
    if case.hpi_path is not None:
        timeline = parse_hpi_timeline(
            case.hpi_path,
            patient_id=case.patient_id,
            labs=clinical_labs,
            index_date=case.index_date,
        )
        timeline.write_json(output / "hpi-timeline-evidence.json")
        treatment_dates = timeline.treatment_dates
    relationship = DataRelationship.model_validate(case.data_relationship)
    verdict = fuse_lion_multimodal_evidence(
        lion=lion,
        glm=glm,
        crosscheck=crosscheck,
        labs=report_labs,
        relationship=relationship,
        treatment_dates=treatment_dates,
    )
    verdict.write_json(output / "clinical-verdict.json")

    glm_observations = [
        f"{item.finding_id}: {item.location} — {item.observation}"
        for item in glm.observations
    ]
    prompt = build_report_prompt(
        imaging_evidence=report_imaging_payload(
            lion, glm_observations=glm_observations
        ),
        lab_evidence=report_labs,
        verdict=verdict,
        scenario_id=case.case_id,
        report_type="lion_inspired_hcc_multimodal_report",
        data_relationship=relationship.statement,
        imaging_source_type="LiON-inspired/precomputed-mask CT evidence",
        lab_source_type="standardized supplied laboratory observations",
    )
    _write_json(output / "deepseek-prompt.json", prompt)

    template_report = build_template_report(prompt, verdict)
    template_validation = validate_report(
        template_report,
        verdict,
        case.case_id,
        prompt_bundle=prompt,
        expected_report_type="lion_inspired_hcc_multimodal_report",
    )
    if not template_validation["valid"]:
        raise RuntimeError(
            "Deterministic report template failed validation: "
            + json.dumps(template_validation["errors"], ensure_ascii=False)
        )

    deepseek_started = perf_counter()
    configured_model = (
        os.environ.get("DEEPSEEK_MODEL")
        or os.environ.get("LLM_MODEL_NAME")
        or "unconfigured"
    )
    use_narrative_renderer = configured_model.casefold().startswith("deepseek-r1")
    provider_audit: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": "disabled",
        "model": "deterministic-template",
        "live_requested": report_mode == "live",
        "prompt_version": "controlled-report-v1",
    }
    narrative_prompt_payload: dict[str, Any] = {
        "prompt_version": "deepseek-hcc-narrative-v1",
        "status": "not_requested",
    }
    _write_json(output / "deepseek-narrative-prompt.json", narrative_prompt_payload)
    narrative_evidence = DeepseekNarrativeEvidence(
        case_id=case.case_id,
        status="unavailable",
        provider="disabled",
        model=configured_model,
        limitations=["Optional DeepSeek language assistance was not requested"],
        sources=[
            SourceReference(
                source_id="controlled-report.json",
                source_type="locked_controlled_report",
                deidentified=True,
            )
        ],
    )
    external_report: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    deepseek_live_success = False
    deepseek_renderer = "deterministic_template"
    if report_mode == "live":
        try:
            if use_narrative_renderer:
                narrative_prompt_payload = build_deepseek_narrative_prompt(
                    template_report
                )
                _write_json(
                    output / "deepseek-narrative-prompt.json",
                    narrative_prompt_payload,
                )
                deepseek_result = call_deepseek_narrative(
                    narrative_prompt_payload,
                    timeout_seconds=timeout_seconds,
                )
                candidate_narrative = deepseek_result["raw_content"]
                validation = validate_deepseek_narrative(
                    candidate_narrative,
                    data_scope=template_report["data_scope"],
                )
                if not validation["valid"]:
                    first_result = deepseek_result
                    retry_prompt = dict(narrative_prompt_payload)
                    retry_codes = sorted(
                        {item["code"] for item in validation["errors"]}
                    )
                    retry_prompt["user_message"] += (
                        "\n\n上一次正文未通过安全校验。错误码："
                        + json.dumps(retry_codes, ensure_ascii=False)
                        + "。请重新生成全新的最终正文，并严格满足全部限制。"
                    )
                    narrative_prompt_payload = retry_prompt
                    _write_json(
                        output / "deepseek-narrative-prompt.json",
                        narrative_prompt_payload,
                    )
                    deepseek_result = call_deepseek_narrative(
                        narrative_prompt_payload,
                        timeout_seconds=timeout_seconds,
                    )
                    deepseek_result["attempt_count"] = int(
                        first_result.get("attempt_count", 1)
                    ) + int(deepseek_result.get("attempt_count", 1))
                    deepseek_result["validation_retry_count"] = 1
                    candidate_narrative = deepseek_result["raw_content"]
                    validation = validate_deepseek_narrative(
                        candidate_narrative,
                        data_scope=template_report["data_scope"],
                    )
                if validation["valid"]:
                    narrative_evidence = DeepseekNarrativeEvidence(
                        case_id=case.case_id,
                        status="pass",
                        provider="deepseek",
                        model=deepseek_result["model"],
                        narrative=candidate_narrative,
                        limitations=[
                            "Narrative is optional language assistance over locked evidence"
                        ],
                        sources=[
                            SourceReference(
                                source_id="controlled-report.json",
                                source_type="locked_controlled_report",
                                deidentified=True,
                            )
                        ],
                    )
                    deepseek_live_success = True
                    deepseek_renderer = "external_narrative"
                else:
                    narrative_evidence = DeepseekNarrativeEvidence(
                        case_id=case.case_id,
                        status="blocked",
                        provider="deepseek",
                        model=deepseek_result["model"],
                        validation_errors=validation["errors"],
                        limitations=[
                            "Unvalidated model prose was blocked and not added to the report"
                        ],
                        sources=[
                            SourceReference(
                                source_id="controlled-report.json",
                                source_type="locked_controlled_report",
                                deidentified=True,
                            )
                        ],
                    )
                provider_audit = _safe_provider_audit(
                    {
                        **deepseek_result,
                        "provider": "deepseek",
                        "live_requested": True,
                        "validation": validation,
                        "audit_status": "pass" if deepseek_live_success else "fail",
                        "error_code": (
                            None
                            if deepseek_live_success
                            else "DEEPSEEK_NARRATIVE_BLOCKED"
                        ),
                    }
                )
            else:
                deepseek_result = call_deepseek(
                    prompt,
                    timeout_seconds=timeout_seconds,
                )
                candidate = deepseek_result["report"]
                validation = validate_report(
                    candidate,
                    verdict,
                    case.case_id,
                    prompt_bundle=prompt,
                    expected_report_type="lion_inspired_hcc_multimodal_report",
                )
                if validation["valid"]:
                    external_report = candidate
                    deepseek_live_success = True
                    deepseek_renderer = "external_llm"
                provider_audit = _safe_provider_audit(
                    {
                        **deepseek_result,
                        "report": None,
                        "provider": "deepseek",
                        "live_requested": True,
                        "prompt_version": "controlled-report-v1",
                        "validation": validation,
                        "audit_status": "pass" if external_report is not None else "fail",
                        "error_code": (
                            None
                            if external_report is not None
                            else "DEEPSEEK_OUTPUT_BLOCKED"
                        ),
                    }
                )
        except RuntimeError as exc:
            narrative_evidence = DeepseekNarrativeEvidence(
                case_id=case.case_id,
                status="unavailable",
                provider="deepseek",
                model=configured_model,
                limitations=[str(exc)],
                sources=[
                    SourceReference(
                        source_id="controlled-report.json",
                        source_type="locked_controlled_report",
                        deidentified=True,
                    )
                ],
            )
            provider_audit = {
                "generated_at": datetime.now(UTC).isoformat(),
                "provider": "deepseek",
                "model": "configured",
                "live_requested": True,
                "prompt_version": "controlled-report-v1",
                "audit_status": "fail",
                "error_code": "DEEPSEEK_UNAVAILABLE",
                "error": str(exc),
            }
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            narrative_evidence = DeepseekNarrativeEvidence(
                case_id=case.case_id,
                status="blocked",
                provider="deepseek",
                model=configured_model,
                validation_errors=[
                    {"code": "DEEPSEEK_OUTPUT_BLOCKED", "message": str(exc)}
                ],
                limitations=[
                    "Unvalidated model content was blocked and not added to the report"
                ],
                sources=[
                    SourceReference(
                        source_id="controlled-report.json",
                        source_type="locked_controlled_report",
                        deidentified=True,
                    )
                ],
            )
            provider_audit = {
                "generated_at": datetime.now(UTC).isoformat(),
                "provider": "deepseek",
                "model": "configured",
                "live_requested": True,
                "prompt_version": "controlled-report-v1",
                "audit_status": "fail",
                "error_code": "DEEPSEEK_OUTPUT_BLOCKED",
                "error": str(exc),
            }
    report_payload = external_report or template_report
    final_validation = validate_report(
        report_payload,
        verdict,
        case.case_id,
        prompt_bundle=prompt,
        expected_report_type="lion_inspired_hcc_multimodal_report",
    )
    if not final_validation["valid"]:
        raise RuntimeError(
            "Controlled report failed deterministic validation: "
            + json.dumps(final_validation["errors"], ensure_ascii=False)
        )
    report = ControlledReport.model_validate(report_payload)
    report.write_json(output / "controlled-report.json")
    narrative_evidence.write_json(output / "deepseek-narrative.json")
    write_human_markdown(
        report.to_dict(),
        output / "controlled-report.md",
        ai_narrative=narrative_evidence.narrative,
    )
    provider_audit.update(
        {
            "latency_ms": round((perf_counter() - deepseek_started) * 1000, 3),
            "renderer": deepseek_renderer,
            "response_status": (
                "pass"
                if deepseek_live_success or report_mode == "deterministic"
                else "degraded"
            ),
            "final_validation": final_validation,
        }
    )
    _write_json(provider_dir / "deepseek.json", _safe_provider_audit(provider_audit))

    glm_live_success = glm_mode == "live" and glm.status in {"pass", "warning"}
    degraded = (glm_mode == "live" and not glm_live_success) or (
        report_mode == "live" and not deepseek_live_success
    )
    strict_success = (
        (not require_live_models)
        or (
            glm_mode == "live"
            and report_mode == "live"
            and glm_live_success
            and deepseek_live_success
        )
    )
    index = _artifact_index(output)
    audit = {
        "schema_version": "1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "case_id": case.case_id,
        "patient_id": case.patient_id,
        "status": "degraded" if degraded else "pass",
        "strict_success": strict_success,
        "modes": {"glm": glm_mode, "report": report_mode},
        "live_status": {
            "glm": glm_live_success,
            "deepseek": deepseek_live_success,
        },
        "data_relationship": relationship.to_dict(),
        "duration_ms": round((perf_counter() - started_clock) * 1000, 3),
        "artifact_index": index,
        "secrets_recorded": False,
    }
    _write_json(output / "pipeline-audit.json", audit)
    web_demo = {
        "schema_version": "1.0.0",
        "case_id": case.case_id,
        "data_relationship": relationship.to_dict(),
        "degraded": degraded,
        "lion_inspired": lion.to_dict(),
        "glm_imaging": glm.to_dict(),
        "imaging_crosscheck": crosscheck.to_dict(),
        "laboratory": clinical_labs.to_dict(),
        "clinical_verdict": verdict.to_dict(),
        "controlled_report": report.to_dict(),
        "deepseek_narrative": narrative_evidence.to_dict(),
        "artifact_index": index,
    }
    _write_json(output / "web_demo.json", web_demo)
    return RunReportResult(
        output_dir=output.resolve(),
        artifact_index=index,
        strict_success=strict_success,
        glm_live_success=glm_live_success,
        deepseek_live_success=deepseek_live_success,
        degraded=degraded,
        report=report,
    )


__all__ = ["RunReportResult", "run_report_pipeline"]
