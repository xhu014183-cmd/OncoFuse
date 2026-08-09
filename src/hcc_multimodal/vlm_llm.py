"""Fail-closed audit pipeline for the dual-mode (auditable / open) VLM demo.

The auditable arm sends a prompt with no laboratory context; the open arm
injects ``[UNVERIFIED_CONTEXT]`` laboratory observations. Both arms share one
extraction target (``VlmDemoReport``) and the same deterministic audit:
structure, locked disclaimer, safety boundaries, and per-number attribution.
Every numeric token in the report must trace back to the supplied context;
otherwise the report is blocked (``EVIDENCE_VALUE_TAMPERED``). When the
external model is unavailable, a deterministic template renders the supplied
context and must pass the same validator.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .case_models import ClinicalLabEvidence, HpiTimelineEvidence
from .deepseek import (
    BOUNDARY_PATTERNS,
    NUMBER_RE,
    _collect_number_tokens,
    _extract_json,
    _negated,
)
from .schemas import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    VlmDemoReport,
    VlmNumericCitation,
)
from .vlm_prompting import (
    RESEARCH_DISCLAIMER,
    ClinicalContextItem,
    VlmTaskPrompt,
    build_vlm_task_prompt,
)

ISO_DATE_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{1,2}-\d{1,2}")

SYSTEM_MESSAGE = (
    "You are a constrained structured-extraction model for an HCC research demo.\n"
    "Follow the task instructions exactly. Image tokens may be placeholders: never "
    "claim to see a scan and never invent enhancement, lesion counts, volumes, or "
    "imaging measurements. Copy any supplied laboratory numbers verbatim with their "
    "evidence IDs; never re-derive, round, extrapolate, or invent values. Do not "
    "diagnose, stage, predict prognosis, or recommend treatment. Return exactly one "
    "JSON object with the fields listed in the task, with no Markdown or commentary."
)


def _system_message(has_images: bool) -> str:
    if not has_images:
        return SYSTEM_MESSAGE
    return (
        "You are a constrained structured-extraction model for an HCC research demo.\n"
        "The task includes real medical images. Describe only what the images and the "
        "supplied structured context support; never invent enhancement, lesion counts, "
        "volumes, or measurements that are not visible or supplied. Copy any supplied "
        "laboratory numbers verbatim with their evidence IDs; never re-derive, round, "
        "extrapolate, or invent values. Do not diagnose, stage, predict prognosis, or "
        "recommend treatment. Return exactly one JSON object with the fields listed in "
        "the task, with no Markdown or commentary."
    )


def _image_url_part(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(target.suffix.lower())
    if mime is None:
        raise ValueError(
            f"Unsupported image type {target.suffix!r}; use PNG, JPG, JPEG, or WEBP"
        )
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def imaging_metadata_text(path: str | Path) -> str:
    """Compress deterministic imaging evidence into one prompt-safe metadata line."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    comparison = data.get("longitudinal_comparison")
    if comparison:
        return (
            "Imaging measurements (deterministic): "
            f"baseline_volume_ml={comparison.get('baseline_total_volume_ml')}; "
            f"followup_volume_ml={comparison.get('followup_total_volume_ml')}; "
            f"volume_change_pct={comparison.get('volume_change_pct')}; "
            f"lesion_count {comparison.get('baseline_lesion_count')}->"
            f"{comparison.get('followup_lesion_count')}; "
            f"new_lesion_signal={comparison.get('new_lesion_signal')}; "
            f"category={comparison.get('category')}"
        )
    return (
        "Imaging measurements (deterministic): "
        f"lesion_count={data.get('lesion_count')}; "
        f"total_volume_ml={data.get('total_tumor_volume_ml')}; "
        f"max_extent_mm={data.get('max_lesion_extent_mm')}; "
        f"quality={data.get('quality', {}).get('status')}"
    )


def call_vlm_llm(
    prompt: VlmTaskPrompt,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 300.0,
    max_tokens: int = 1500,
    temperature: float = 0.0,
    json_object: bool = True,
    image_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Send one dual-mode VLM task prompt to an OpenAI-compatible endpoint."""
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    endpoint_base = base_url or os.environ.get("LLM_BASE_URL")
    model_name = model or os.environ.get("LLM_MODEL_NAME")
    if not key or not endpoint_base or not model_name:
        raise RuntimeError("External LLM configuration is unavailable")
    endpoint = endpoint_base.rstrip("/") + "/chat/completions"
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt.prompt_text}]
    if image_paths:
        user_parts.extend(_image_url_part(path) for path in image_paths)
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _system_message(bool(image_paths))},
            {"role": "user", "content": user_parts if image_paths else prompt.prompt_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}
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


def _lab_observations_from_context(
    items: list[ClinicalContextItem],
) -> list[tuple[float, ClinicalContextItem]]:
    observations: list[tuple[float, ClinicalContextItem]] = []
    for item in items:
        if item.category != "laboratory":
            continue
        for match in NUMBER_RE.finditer(item.statement):
            number = float(match.group())
            if math.isfinite(number):
                observations.append((number, item))
    return observations


def _analyte_from_statement(statement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_%+\-]+)", statement)
    return match.group(1) if match else "unknown"


def _unit_from_statement(statement: str) -> str:
    match = re.search(
        r"(?:eq|lt|le|gt|ge)\s*[-+]?\d+(?:\.\d+)?\s*([A-Za-z0-9/%]+)",
        statement,
    )
    return match.group(1) if match else "unknown"


def _strip_evidence_ids(text: str) -> str:
    """Remove ``[LAB_001]``-style IDs so digits inside them are not treated as values."""
    return re.sub(r"\[(?:LAB|HPI)_\d+\]", "", text)


def _mask_iso_dates(text: str) -> str:
    """Replace ISO dates so hyphen separators are not parsed as negative numbers."""
    return ISO_DATE_RE.sub(" DATE ", text)


def _date_components(text: str) -> set[float]:
    """Return year/month/day as positive numbers for every ISO date in the text."""
    components: set[float] = set()
    for match in ISO_DATE_RE.finditer(text):
        year, month, day = match.group().split("-")
        components.update(float(part) for part in (year, month, day))
    return components


def _as_text_list(value: Any) -> list[str]:
    """Normalize a narrative field to a list of strings.

    Models occasionally return a single string where a list is required; unpacking
    the raw value with ``*`` would then iterate characters. Wrap scalars instead.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_numeric_citations(
    report: dict[str, Any],
    prompt: VlmTaskPrompt,
) -> tuple[list[VlmNumericCitation], list[float]]:
    """Resolve every narrative numeric token to a lab observation.

    Returns ``(citations, unattributed)`` where unattributed numbers are
    candidates for ``EVIDENCE_VALUE_TAMPERED`` unless they appear in the prompt
    itself (dates, phase, reference limits, etc.).
    """
    lab_numbers = _lab_observations_from_context(prompt.clinical_context)
    narrative_fields = [
        *_as_text_list(report.get("imaging_observations")),
        *_as_text_list(report.get("clinical_context_summary")),
        *_as_text_list(report.get("uncertainties")),
        *_as_text_list(report.get("missing_information")),
        str(report.get("evidence_concordance", "")),
        str(report.get("image_conditioning_statement", "")),
    ]
    citations: list[VlmNumericCitation] = []
    unattributed: list[float] = []
    seen: set[tuple[float, str]] = set()
    for field_index, text in enumerate(narrative_fields):
        location = f"field[{field_index}]"
        for match in NUMBER_RE.finditer(_strip_evidence_ids(str(text))):
            number = float(match.group())
            if not math.isfinite(number):
                continue
            key = (number, location)
            if key in seen:
                continue
            seen.add(key)
            matched = next(
                (
                    obs_item
                    for value, obs_item in lab_numbers
                    if math.isclose(value, number, rel_tol=1e-6, abs_tol=1e-9)
                ),
                None,
            )
            if matched is not None:
                citations.append(
                    VlmNumericCitation(
                        evidence_id=matched.evidence_id,
                        observed_at=matched.observed_at,
                        analyte=_analyte_from_statement(matched.statement),
                        value=number,
                        unit=_unit_from_statement(matched.statement),
                        used_in=location,
                    )
                )
            else:
                unattributed.append(number)
    return citations, unattributed


def _vlm_narrative(report: dict[str, Any]) -> str:
    values = [
        *_as_text_list(report.get("imaging_observations")),
        *_as_text_list(report.get("clinical_context_summary")),
        *_as_text_list(report.get("uncertainties")),
        *_as_text_list(report.get("missing_information")),
        report.get("evidence_concordance", ""),
        report.get("image_conditioning_statement", ""),
    ]
    return "\n".join(str(value) for value in values if value is not None)


def validate_vlm_report(
    report: dict[str, Any],
    *,
    fusion_mode: Literal["auditable", "open"],
    prompt: VlmTaskPrompt,
) -> dict[str, Any]:
    """Validate structure, locked disclaimer, boundaries, and numeric fidelity."""
    errors: list[dict[str, str]] = []
    reported_mode = report.get("fusion_mode")
    if reported_mode is not None and reported_mode != fusion_mode:
        errors.append(
            {
                "code": "LOCKED_FIELD_MISMATCH",
                "message": (
                    f"fusion_mode: expected {fusion_mode!r}, "
                    f"got {reported_mode!r}"
                ),
            }
        )
    report["fusion_mode"] = fusion_mode
    try:
        VlmDemoReport.model_validate(report)
    except ValidationError as exc:
        for item in exc.errors(include_url=False):
            location = ".".join(str(part) for part in item["loc"])
            errors.append(
                {
                    "code": "SCHEMA_INVALID",
                    "message": f"{location}: {item['msg']}",
                }
            )

    if report.get("research_disclaimer") != RESEARCH_DISCLAIMER:
        errors.append(
            {
                "code": "LOCKED_FIELD_MISMATCH",
                "message": "research_disclaimer must equal the fixed research disclaimer",
            }
        )

    narrative = _vlm_narrative(report)
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

    citations, unattributed = build_numeric_citations(report, prompt)
    allowed = _collect_number_tokens(_mask_iso_dates(prompt.prompt_text))
    allowed.update(_date_components(prompt.prompt_text))
    for number in unattributed:
        if number not in allowed:
            errors.append(
                {
                    "code": "EVIDENCE_VALUE_TAMPERED",
                    "message": (
                        f"Report contains numeric value {number} unsupported by the "
                        f"supplied context (fusion_mode={fusion_mode})"
                    ),
                }
            )
    if fusion_mode == "auditable" and prompt.clinical_context:
        errors.append(
            {
                "code": "EVIDENCE_OMITTED_OR_CHANGED",
                "message": "auditable arm must not receive laboratory clinical context",
            }
        )
    return {
        "status": "pass" if not errors else "fail",
        "valid": not errors,
        "errors": errors,
        "numeric_citations": [citation.to_dict() for citation in citations],
        "unattributed_numbers": sorted(set(unattributed)),
    }


def build_vlm_template_report(prompt: VlmTaskPrompt) -> dict[str, Any]:
    """Deterministic fallback that renders only the supplied context."""
    if prompt.fusion_mode == "open":
        lab_items = [
            item for item in prompt.clinical_context if item.category == "laboratory"
        ]
        clinical_context_summary = [
            f"{item.statement} [{item.evidence_id}]" for item in lab_items
        ]
    else:
        clinical_context_summary = []
    return {
        "fusion_mode": prompt.fusion_mode,
        "imaging_observations": [
            (
                "Image tokens were supplied as placeholders; this deterministic "
                "fallback does not produce visual observations."
            )
        ],
        "clinical_context_summary": clinical_context_summary,
        "evidence_concordance": "insufficient_evidence",
        "uncertainties": [
            "External model unavailable; deterministic fallback rendered the supplied context."
        ],
        "missing_information": [
            "Real visual-token inference is not available in this text-only harness."
        ],
        "image_conditioning_statement": (
            "This output was produced by a deterministic fallback; image tokens were "
            "not interpreted."
        ),
        "research_disclaimer": RESEARCH_DISCLAIMER,
    }


def _run_single_arm(
    prompt: VlmTaskPrompt,
    *,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float,
    json_object: bool,
    image_paths: list[str | Path] | None,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fusion_mode": prompt.fusion_mode,
        "renderer": None,
        "raw_content": None,
        "report": None,
        "validation": None,
        "audit_status": None,
    }
    try:
        llm_result = call_vlm_llm(
            prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            json_object=json_object,
            image_paths=image_paths,
        )
    except RuntimeError as exc:
        report = build_vlm_template_report(prompt)
        validation = validate_vlm_report(report, fusion_mode=prompt.fusion_mode, prompt=prompt)
        report["numeric_citations"] = validation["numeric_citations"]
        result.update(
            {
                "renderer": "deterministic_template",
                "fallback_reason": str(exc),
                "raw_content": None,
                "report": report,
                "validation": validation,
                "audit_status": validation["status"],
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
    else:
        report = llm_result.pop("report")
        validation = validate_vlm_report(report, fusion_mode=prompt.fusion_mode, prompt=prompt)
        report["numeric_citations"] = validation["numeric_citations"]
        result.update(llm_result)
        result.update(
            {
                "renderer": "external_llm",
                "report": report,
                "validation": validation,
                "audit_status": validation["status"],
            }
        )
    return result


def _norm_statement(value: str) -> str:
    normalized = NUMBER_RE.sub("N", value)
    return re.sub(r"\s+", " ", normalized).strip()


def _comparison_diff(
    auditable_report: dict[str, Any] | None,
    open_report: dict[str, Any] | None,
) -> dict[str, list[str]]:
    def statements(report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []
        return [
            *_as_text_list(report.get("imaging_observations")),
            *_as_text_list(report.get("clinical_context_summary")),
            *_as_text_list(report.get("uncertainties")),
            *_as_text_list(report.get("missing_information")),
            str(report.get("evidence_concordance", "")),
            str(report.get("image_conditioning_statement", "")),
        ]

    auditable_lines = statements(auditable_report)
    open_lines = statements(open_report)
    auditable_keys = {_norm_statement(line) for line in auditable_lines}
    open_keys = {_norm_statement(line) for line in open_lines}
    return {
        "auditable_only": [line for line in auditable_lines if _norm_statement(line) not in open_keys],
        "open_only": [line for line in open_lines if _norm_statement(line) not in auditable_keys],
    }


def run_vlm_dual_arm_demo(
    *,
    labs: ClinicalLabEvidence | None = None,
    timeline: HpiTimelineEvidence | None = None,
    phase: str = "unknown",
    timepoint: Literal["baseline", "followup", "single"] = "single",
    fusion_modes: tuple[Literal["auditable", "open"], ...] = ("auditable", "open"),
    output_dir: str | Path,
    visual_token_count: int = 32,
    max_tokens: int = 1500,
    timeout_seconds: float = 300.0,
    temperature: float = 0.0,
    json_object: bool = True,
    image_paths: list[str | Path] | None = None,
    imaging_metadata: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> Path:
    """Run the requested dual-mode arms, audit each fail-closed, and write artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = list(image_paths or [])
    arms: dict[str, dict[str, Any]] = {}
    for mode in fusion_modes:
        prompt = build_vlm_task_prompt(
            labs=labs,
            timeline=timeline,
            fusion_mode=mode,
            phase=phase,
            timepoint=timepoint,
            visual_token_count=visual_token_count,
            image_count=len(images),
            imaging_metadata=imaging_metadata,
        )
        prompt_path = output / f"vlm_prompt_{mode}.json"
        prompt.write_json(prompt_path)
        arm = _run_single_arm(
            prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            json_object=json_object,
            image_paths=images or None,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        arm["prompt_file"] = prompt_path.name
        (output / f"vlm_arm_{mode}.json").write_text(
            json.dumps(arm, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        arms[mode] = arm

    auditable_report = arms.get("auditable", {}).get("report")
    open_report = arms.get("open", {}).get("report")
    open_citations = (open_report or {}).get("numeric_citations") or []
    auditable_citations = (auditable_report or {}).get("numeric_citations") or []
    open_validation = arms.get("open", {}).get("validation") or {}
    hallucination_candidates = [
        item["message"]
        for item in open_validation.get("errors", [])
        if item["code"] == "EVIDENCE_VALUE_TAMPERED"
    ]
    comparison: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "arms": {
            mode: {
                "renderer": arms[mode].get("renderer"),
                "audit_status": arms[mode].get("audit_status"),
                "error_code": arms[mode].get("error_code"),
                "fallback_reason": arms[mode].get("fallback_reason"),
                "prompt_file": arms[mode].get("prompt_file"),
                "report_file": f"vlm_arm_{mode}.json",
            }
            for mode in fusion_modes
        },
        "diff": _comparison_diff(auditable_report, open_report),
        "open_numeric_citations": open_citations,
        "open_numeric_citation_count": len(open_citations),
        "auditable_numeric_citation_count": len(auditable_citations),
        "hallucination_candidates": hallucination_candidates,
    }
    comparison_path = output / "dual_arm_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    web_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": arms.get("open", {}).get("model")
        or arms.get("auditable", {}).get("model"),
        "arms": {
            mode: {
                "fusion_mode": arms[mode].get("fusion_mode"),
                "renderer": arms[mode].get("renderer"),
                "audit_status": arms[mode].get("audit_status"),
                "error_code": arms[mode].get("error_code"),
                "fallback_reason": arms[mode].get("fallback_reason"),
                "report": arms[mode].get("report"),
            }
            for mode in fusion_modes
        },
        "diff": _comparison_diff(auditable_report, open_report),
        "hallucination_candidates": hallucination_candidates,
        "open_numeric_citation_count": len(open_citations),
        "auditable_numeric_citation_count": len(auditable_citations),
    }
    web_path = output / "web_demo.json"
    web_path.write_text(
        json.dumps(web_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison_path


__all__ = [
    "SYSTEM_MESSAGE",
    "build_numeric_citations",
    "build_vlm_template_report",
    "call_vlm_llm",
    "imaging_metadata_text",
    "run_vlm_dual_arm_demo",
    "validate_vlm_report",
]
