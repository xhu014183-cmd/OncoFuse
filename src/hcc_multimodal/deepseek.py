from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import math
import os
import re

from pydantic import ValidationError

from .reporting import write_human_markdown
from .schemas import PIPELINE_VERSION, SCHEMA_VERSION, ControlledReport


NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
BOUNDARY_PATTERNS = {
    "DIAGNOSTIC_ASSERTION": [
        re.compile(r"\b(?:diagnos(?:e|ed|is)|confirm(?:s|ed)?|definitive)\b.{0,40}\b(?:HCC|cancer|carcinoma)\b", re.I),
        re.compile(r"(?:诊断为|确诊|证实为).{0,20}(?:HCC|肝癌|肝细胞癌)"),
    ],
    "STAGING_OR_RESPONSE_ASSERTION": [
        re.compile(r"\b(?:BCLC|LI-RADS|LR-[1-5M]|m?RECIST)\b", re.I),
        re.compile(r"(?:分期为|属于.{0,8}期)"),
    ],
    "TREATMENT_RECOMMENDATION": [
        re.compile(r"\b(?:recommend|should|advise|initiate|start)\b.{0,50}\b(?:treat|therapy|surgery|ablation|drug|dose)\w*\b", re.I),
        re.compile(r"(?:建议|推荐|应当|需要).{0,30}(?:治疗|用药|手术|消融|剂量)"),
    ],
    "PROMPT_INJECTION_ECHO": [
        re.compile(r"ignore (?:all |the )?(?:previous|prior|system) instructions", re.I),
        re.compile(r"忽略.{0,12}(?:之前|系统|上述).{0,8}(?:指令|提示)"),
    ],
}


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _verdict_dict(value: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError("expected_verdict must be a dictionary or JSON model")


def _narrative_text(report: dict[str, Any]) -> str:
    assessment = report.get("multimodal_assessment") or {}
    values = [
        report.get("data_scope"),
        *(report.get("imaging_summary") or []),
        *(report.get("laboratory_summary") or []),
        assessment.get("summary"),
        *(assessment.get("supporting_evidence") or []),
        *(assessment.get("conflicting_evidence") or []),
        *(assessment.get("missing_evidence") or []),
        *(report.get("uncertainty") or []),
        *(report.get("quality_and_limits") or []),
    ]
    return "\n".join(str(value) for value in values if value is not None)


def _negated(line: str, match_start: int) -> bool:
    prefix = line[max(0, match_start - 60) : match_start].casefold()
    stripped = prefix.strip()
    if stripped in ("no", "not") or stripped.startswith(("no ", "not ")):
        return True
    return any(
        token in prefix for token in (" not ", " no ", "without ", "不得", "不用于", "未进行")
    )


def _collect_number_tokens(value: Any) -> set[float]:
    tokens: set[float] = set()
    if isinstance(value, bool) or value is None:
        return tokens
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            tokens.add(float(value))
        return tokens
    if isinstance(value, str):
        for match in NUMBER_RE.finditer(value):
            number = float(match.group())
            if math.isfinite(number):
                tokens.add(number)
        return tokens
    if isinstance(value, dict):
        for item in value.values():
            tokens.update(_collect_number_tokens(item))
    elif isinstance(value, list):
        for item in value:
            tokens.update(_collect_number_tokens(item))
    return tokens


def validate_report(
    report: dict[str, Any],
    expected_verdict: dict[str, Any] | Any,
    expected_scenario_id: str,
    *,
    prompt_bundle: dict[str, Any] | None = None,
    expected_report_type: str | None = None,
) -> dict[str, Any]:
    """Validate structure, locked facts, evidence preservation, and safety boundaries."""
    errors: list[dict[str, str]] = []
    verdict = _verdict_dict(expected_verdict)
    try:
        ControlledReport.model_validate(report)
    except ValidationError as exc:
        for item in exc.errors(include_url=False):
            location = ".".join(str(part) for part in item["loc"])
            errors.append(
                {
                    "code": "SCHEMA_INVALID",
                    "message": f"{location}: {item['msg']}",
                }
            )

    assessment = report.get("multimodal_assessment")
    if not isinstance(assessment, dict):
        assessment = {}
    locked = {
        "case_id": (report.get("case_id"), verdict.get("patient_id")),
        "scenario_id": (report.get("scenario_id"), expected_scenario_id),
        "state": (assessment.get("state"), verdict.get("state")),
        "concordance": (assessment.get("concordance"), verdict.get("modality_concordance")),
        "review_required": (report.get("review_required"), verdict.get("requires_clinician_review")),
        "data_quality_status": (report.get("data_quality_status"), verdict.get("quality_status")),
        "intended_use": (report.get("intended_use"), verdict.get("intended_use")),
        "disclaimer": (
            report.get("disclaimer"),
            "Research use only; not for diagnosis, staging, prognosis, or treatment decisions.",
        ),
    }
    if expected_report_type is not None:
        locked["report_type"] = (report.get("report_type"), expected_report_type)
    for name, (actual, expected) in locked.items():
        if actual != expected:
            errors.append(
                {
                    "code": "LOCKED_FIELD_MISMATCH",
                    "message": f"{name}: expected {expected!r}, got {actual!r}",
                }
            )

    for field in ("supporting_evidence", "conflicting_evidence", "missing_evidence"):
        actual = assessment.get(field)
        expected = verdict.get(field, [])
        if actual != expected:
            errors.append(
                {
                    "code": "EVIDENCE_OMITTED_OR_CHANGED",
                    "message": f"multimodal_assessment.{field} must copy the validated verdict exactly",
                }
            )

    narrative = _narrative_text(report)
    for code, patterns in BOUNDARY_PATTERNS.items():
        violation = None
        for line in narrative.splitlines():
            for pattern in patterns:
                match = pattern.search(line)
                if match and not _negated(line, match.start()):
                    violation = pattern
                    break
            if violation:
                break
        if violation:
            errors.append(
                {"code": code, "message": f"Boundary pattern matched: {violation.pattern}"}
            )

    if prompt_bundle is not None:
        required_quality = prompt_bundle.get("required_quality_items") or []
        actual_quality = report.get("quality_and_limits") or []
        omitted = [item for item in required_quality if item not in actual_quality]
        if omitted:
            errors.append(
                {
                    "code": "KEY_QC_OMITTED",
                    "message": f"Missing required quality items: {omitted}",
                }
            )
        allowed_numbers = _collect_number_tokens(prompt_bundle.get("evidence_payload"))
        allowed_numbers.update(_collect_number_tokens(prompt_bundle.get("locked_fields")))
        report_numbers = _collect_number_tokens(narrative)
        invented = sorted(number for number in report_numbers if number not in allowed_numbers)
        if invented:
            errors.append(
                {
                    "code": "EVIDENCE_VALUE_TAMPERED",
                    "message": f"Report contains numeric values absent from validated evidence: {invented}",
                }
            )
    return {
        "status": "pass" if not errors else "fail",
        "valid": not errors,
        "errors": errors,
    }


def lock_report_fields(
    report: dict[str, Any],
    expected_verdict: dict[str, Any] | Any,
    expected_scenario_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compatibility shim: locked fields are now rejected rather than repaired."""
    validation = validate_report(report, expected_verdict, expected_scenario_id)
    mismatches = [
        item for item in validation["errors"] if item["code"] == "LOCKED_FIELD_MISMATCH"
    ]
    if mismatches:
        raise ValueError(f"Locked report fields do not match: {mismatches}")
    return report, []


def build_template_report(
    prompt_bundle: dict[str, Any],
    expected_verdict: dict[str, Any] | Any,
) -> dict[str, Any]:
    """Render a deterministic report when the external model is unavailable."""
    verdict = _verdict_dict(expected_verdict)
    evidence = prompt_bundle["evidence_payload"]
    imaging = evidence["imaging_evidence"]
    markers = evidence["lab_evidence"]["markers"]
    if "longitudinal_comparison" in imaging:
        comparison = imaging["longitudinal_comparison"]
        imaging_summary = [
            f"Longitudinal imaging category: {comparison['category']}; "
            f"volume change: {comparison['volume_change_pct']}%."
        ]
    else:
        imaging_summary = [
            f"Supplied mask contains {imaging['mass_region_count']} retained region(s), "
            f"total volume {imaging['annotated_mass_volume_ml']} mL."
        ]
    laboratory_summary = []
    for name in ("AFP", "DCP"):
        marker = markers.get(name)
        if marker is None:
            laboratory_summary.append(f"{name} evidence is unavailable.")
        else:
            laboratory_summary.append(
                f"{name}: latest {marker['latest_comparator']} {marker['latest_value']} "
                f"{marker['unit']}; trend {marker['direction']}."
            )
    return {
        "case_id": verdict["patient_id"],
        "scenario_id": evidence["scenario_id"],
        "report_type": evidence["report_type"],
        "data_scope": evidence["data_relationship"],
        "imaging_summary": imaging_summary,
        "laboratory_summary": laboratory_summary,
        "multimodal_assessment": {
            "state": verdict["state"],
            "concordance": verdict["modality_concordance"],
            "summary": "The validated evidence was rendered without changing the machine-generated state.",
            "supporting_evidence": verdict["supporting_evidence"],
            "conflicting_evidence": verdict["conflicting_evidence"],
            "missing_evidence": verdict["missing_evidence"],
        },
        "uncertainty": verdict["missing_evidence"],
        "data_quality_status": verdict["quality_status"],
        "quality_and_limits": prompt_bundle["required_quality_items"],
        "review_required": verdict["requires_clinician_review"],
        "intended_use": verdict["intended_use"],
        "disclaimer": (
            "Research use only; not for diagnosis, staging, prognosis, or treatment decisions."
        ),
    }


def call_deepseek(
    prompt_bundle: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 300.0,
    max_tokens: int = 3000,
) -> dict[str, Any]:
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    endpoint_base = base_url or os.environ.get("LLM_BASE_URL")
    model_name = model or os.environ.get("LLM_MODEL_NAME")
    if not key or not endpoint_base or not model_name:
        raise RuntimeError("External LLM configuration is unavailable")
    endpoint = endpoint_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_bundle["system_message"]},
            {"role": "user", "content": prompt_bundle["user_message"]},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM connection failed: {exc}") from exc

    message = payload["choices"][0]["message"]
    content = message.get("content") or ""
    report = _extract_json(content)
    return {
        "provider": "openai-compatible",
        "model": model_name,
        "usage": payload.get("usage", {}),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "report": report,
        "raw_content": content,
        "reasoning_content_present": bool(message.get("reasoning_content")),
    }


def run_deepseek_audit(
    *,
    prompt_path: str | Path,
    verdict_path: str | Path,
    output_path: str | Path,
    scenario_id: str,
) -> Path:
    prompt = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    verdict = json.loads(Path(verdict_path).read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {"source_id": Path(prompt_path).name, "source_type": "validated_prompt_bundle"},
            {"source_id": Path(verdict_path).name, "source_type": "clinical_verdict"},
        ],
        "scenario_id": scenario_id,
    }
    try:
        llm_result = call_deepseek(prompt)
    except RuntimeError as exc:
        report = build_template_report(prompt, verdict)
        result.update(
            {
                "renderer": "deterministic_template",
                "fallback_reason": str(exc),
                "raw_content": None,
                "report": report,
            }
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result.update(
            {
                "renderer": "external_llm",
                "audit_status": "fail",
                "error_code": "LLM_JSON_INVALID",
                "error": str(exc),
            }
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
    else:
        report = llm_result.pop("report")
        result.update(llm_result)
        result.update({"renderer": "external_llm", "report": report})

    validation = validate_report(
        result["report"],
        verdict,
        scenario_id,
        prompt_bundle=prompt,
        expected_report_type=prompt["locked_fields"]["report_type"],
    )
    result["validation"] = validation
    result["audit_status"] = validation["status"]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if validation["valid"]:
        markdown_target = target.with_name("controlled_report.md")
        write_human_markdown(result["report"], markdown_target)
        result["human_report_file"] = markdown_target.name
    else:
        result["error_code"] = "LLM_OUTPUT_BLOCKED"
        result["human_report_file"] = None
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
