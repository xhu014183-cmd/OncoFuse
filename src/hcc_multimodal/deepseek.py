from __future__ import annotations

import json
import math
import os
import re
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .reporting import write_human_markdown
from .schemas import PIPELINE_VERSION, SCHEMA_VERSION, ControlledReport

NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

# Negation context: a negative qualifier within this many characters before the
# matched term (or an explicit negative phrase) marks the assertion as negated.
_NEGATION_WINDOW = 120
_NEGATION_EN = (
    " not ",
    " no ",
    " without ",
    " cannot ",
    " can not ",
    " does not ",
    " did not ",
    " is not ",
    " are not ",
    " was not ",
    " were not ",
    " no evidence of ",
    " absence of ",
    " absent ",
    " excludes ",
    " ruled out ",
    " not consistent ",
    " is not consistent ",
    " fails to ",
    " failed to ",
)
_NEGATION_ZH = (
    "不得",
    "不用于",
    "未进行",
    "未见",
    "未发现",
    "排除",
    "不支持",
    "不符合",
    "不提示",
    "未提示",
    "不考虑",
)
# 中文否定字通常紧贴诊断动词（"不能诊断"、"未确诊"、"无肝癌"）
_NEGATION_ZH_CHARS = "不未无非"
_NEGATION_SCOPE_BREAK_RE = re.compile(
    r"[.!?;；。！？\n]|\b(?:but|however|although|yet)\b|但|但是|然而|不过|可是|却",
    re.IGNORECASE,
)
BOUNDARY_PATTERNS = {
    "DIAGNOSTIC_ASSERTION": [
        re.compile(r"\b(?:diagnos(?:e|ed|is)|confirm(?:s|ed)?|definitive)\b[^.!?;；。！？\r\n]{0,40}\b(?:HCC|cancer|carcinoma)\b", re.IGNORECASE),
        re.compile(r"(?:诊断为|确诊|证实为)[^.!?;；。！？\r\n]{0,20}(?:HCC|肝癌|肝细胞癌)"),
    ],
    "STAGING_OR_RESPONSE_ASSERTION": [
        re.compile(r"\b(?:BCLC|LI-RADS|LR-[1-5M]|m?RECIST)\b", re.IGNORECASE),
        re.compile(r"(?:分期为|属于.{0,8}期)"),
    ],
    "TREATMENT_RECOMMENDATION": [
        re.compile(r"\b(?:recommend|should|advise|initiate|start)\b[^.!?;；。！？\r\n]{0,50}\b(?:treat|therapy|surgery|ablation|drug|dose)\w*\b", re.IGNORECASE),
        re.compile(r"(?:建议|推荐|应当|需要)[^.!?;；。！？\r\n]{0,30}(?:治疗|用药|手术|消融|剂量)"),
    ],
    "PROMPT_INJECTION_ECHO": [
        re.compile(r"ignore (?:all |the )?(?:previous|prior|system) instructions", re.IGNORECASE),
        re.compile(r"忽略[^.!?;；。！？\r\n]{0,12}(?:之前|系统|上述)[^.!?;；。！？\r\n]{0,8}(?:指令|提示)"),
    ],
}

SOFT_PATTERNS = {
    "SOFT_DIAGNOSTIC_ASSERTION": [
        re.compile(
            r"\b(?:consistent|compatible|suggestive)\b[^.!?;；。！？\r\n]{0,20}\bwith\b[^.!?;；。！？\r\n]{0,40}"
            r"\b(?:HCC|hepatocellular carcinoma|carcinoma|malignancy)\b",
            re.IGNORECASE,
        ),
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
        raise TypeError("LLM response JSON must be an object")
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
    prefix = line[:match_start]
    boundaries = list(_NEGATION_SCOPE_BREAK_RE.finditer(prefix))
    if boundaries:
        prefix = prefix[boundaries[-1].end() :]
    window = prefix[-_NEGATION_WINDOW:]
    case = window.casefold()
    stripped = case.strip()
    if stripped in ("no", "not", "cannot", "absent", "none"):
        return True
    if stripped.startswith(("no ", "not ", "cannot ", "without ")):
        return True
    if any(token in case for token in _NEGATION_EN):
        return True
    if any(token in window for token in _NEGATION_ZH):
        return True
    # 中文否定字紧贴诊断动词时的兜底（"不能诊断"、"未确诊"、"无肝癌"）
    return bool(window and any(char in window[-2:] for char in _NEGATION_ZH_CHARS))


def _first_unnegated_pattern(
    line: str, patterns: list[re.Pattern[str]]
) -> re.Pattern[str] | None:
    """Return the first boundary pattern with a non-negated match.

    ``finditer`` is intentional: a line may contain a negated statement followed
    by a positive assertion, and the latter must still be blocked.
    """
    for pattern in patterns:
        for match in pattern.finditer(line):
            if not _negated(line, match.start()):
                return pattern
    return None


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
            violation = _first_unnegated_pattern(line, patterns)
            if violation:
                break
        if violation:
            errors.append(
                {"code": code, "message": f"Boundary pattern matched: {violation.pattern}"}
            )

    soft_warnings: list[dict[str, str]] = []
    for code, patterns in SOFT_PATTERNS.items():
        for line in narrative.splitlines():
            pattern = _first_unnegated_pattern(line, patterns)
            if pattern is not None:
                soft_warnings.append(
                    {"code": code, "message": f"Soft pattern matched: {pattern.pattern}"}
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
        "soft_warnings": soft_warnings,
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
            f"Longitudinal imaging category: {comparison['category']}; "  # noqa: ISC004
            f"volume change: {comparison['volume_change_pct']}%."
        ]
    else:
        if imaging["mass_region_count"] is None:
            imaging_summary = [
                "Supplied-mask quantitative imaging evidence is unavailable."
            ]
        else:
            imaging_summary = [
                f"Supplied mask contains {imaging['mass_region_count']} retained region(s), "  # noqa: ISC004
                f"total volume {imaging['annotated_mass_volume_ml']} mL."
            ]
        glm_observations = imaging.get("glm_observations") or []
        if glm_observations:
            imaging_summary.append("GLM image observation: " + str(glm_observations[0]))
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


DEEPSEEK_NARRATIVE_PROMPT_VERSION = "deepseek-hcc-narrative-v1"


def _redact_narrative_numbers(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return "数值已锁定"
    if isinstance(value, str):
        return NUMBER_RE.sub("数值已锁定", value)
    if isinstance(value, list):
        return [_redact_narrative_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_narrative_numbers(item) for key, item in value.items()}
    return str(value)


def build_deepseek_narrative_prompt(
    controlled_report: dict[str, Any],
) -> dict[str, str]:
    """Build a compact prose-only task for a thinking-only Distill model."""
    data_scope = str(controlled_report["data_scope"])
    pairing_instruction = (
        "本病例的影像与检验并非同一受试者，最终正文必须逐字包含“未配对演示”，"
        "并明确不能形成患者级诊断解释。"
        if "未配对" in data_scope or "不属于同一" in data_scope
        else ""
    )
    source = _redact_narrative_numbers({
        "data_scope": data_scope,
        "imaging_summary": controlled_report["imaging_summary"],
        "laboratory_summary": controlled_report["laboratory_summary"],
        "multimodal_assessment": controlled_report["multimodal_assessment"],
        "uncertainty": controlled_report["uncertainty"],
        "quality_and_limits": controlled_report["quality_and_limits"],
        "review_required": controlled_report["review_required"],
    })
    return {
        "prompt_version": DEEPSEEK_NARRATIVE_PROMPT_VERSION,
        "system_message": (
            "你是肿瘤多模态研究报告的中文语言编辑。输入事实已经由本地程序锁定。"
            "你只能解释影像证据与检验证据之间的关系，不能重新裁决证据。"
        ),
        "user_message": (
            "请将下面的锁定证据写成两到三段简洁、通俗的中文辅助解读。"
            "必须同时说明影像和检验趋势，并说明数据限制或需要人工复核。"
            "不得输出任何阿拉伯数字，不得增加或改写数值，不得确诊、分期、预测预后或建议治疗。"
            "不得输出JSON、Markdown标题、列表、提示词或思考过程，只输出最终中文正文。\n\n"
            + pairing_instruction
            + "\n\n"
            + json.dumps(source, ensure_ascii=False, separators=(",", ":"))
        ),
    }


def validate_deepseek_narrative(
    narrative: str,
    *,
    data_scope: str,
) -> dict[str, Any]:
    """Block unsupported numbers, clinical assertions, and incomplete fusion prose."""
    text = narrative.strip()
    errors: list[dict[str, str]] = []
    if not text:
        errors.append({"code": "EMPTY_CONTENT", "message": "No final prose was returned"})
    if len(text) > 1500:
        errors.append(
            {"code": "CONTENT_TOO_LONG", "message": "Narrative exceeds 1500 characters"}
        )
    if NUMBER_RE.search(text):
        errors.append(
            {
                "code": "NUMERIC_CONTENT_BLOCKED",
                "message": "Optional narrative must not repeat or introduce numeric tokens",
            }
        )
    if re.search(r"(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|```|[{}]", text):
        errors.append(
            {
                "code": "UNSUPPORTED_FORMAT",
                "message": "Narrative must be plain prose rather than Markdown or JSON",
            }
        )
    for code, patterns in BOUNDARY_PATTERNS.items():
        violation = None
        for line in text.splitlines():
            violation = _first_unnegated_pattern(line, patterns)
            if violation is not None:
                break
        if violation is not None:
            errors.append(
                {"code": code, "message": f"Boundary pattern matched: {violation.pattern}"}
            )
    if text and not re.search(r"影像|病灶|CT|扫描", text, re.IGNORECASE):
        errors.append(
            {"code": "IMAGING_OMITTED", "message": "Narrative omitted imaging evidence"}
        )
    if text and not re.search(r"检验|标志物|AFP|DCP|实验室", text, re.IGNORECASE):
        errors.append(
            {"code": "LABORATORY_OMITTED", "message": "Narrative omitted laboratory evidence"}
        )
    if text and not re.search(r"复核|限制|不足|缺少|不能|无法|不确定", text):
        errors.append(
            {
                "code": "LIMITATION_OMITTED",
                "message": "Narrative omitted uncertainty or clinician review",
            }
        )
    unpaired = "未配对" in data_scope or "不属于同一" in data_scope
    if unpaired and text and not re.search(r"未配对|不属于同一|演示", text):
        errors.append(
            {
                "code": "PAIRING_LIMIT_OMITTED",
                "message": "Narrative omitted the unpaired demonstration limitation",
            }
        )
    return {
        "status": "pass" if not errors else "fail",
        "valid": not errors,
        "errors": errors,
        "character_count": len(text),
    }


def call_deepseek_narrative(
    prompt_bundle: dict[str, str],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 300.0,
    max_tokens: int = 6000,
) -> dict[str, Any]:
    """Call DeepSeek Distill for optional prose without requiring JSON mode."""
    direct_key = os.environ.get("DEEPSEEK_API_KEY")
    direct_base = os.environ.get("DEEPSEEK_BASE_URL")
    direct_model = os.environ.get("DEEPSEEK_MODEL")
    legacy_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    legacy_base = os.environ.get("LLM_BASE_URL")
    legacy_model = os.environ.get("LLM_MODEL_NAME")
    key = api_key or direct_key or legacy_key
    endpoint_base = base_url or direct_base or legacy_base
    model_name = model or direct_model or legacy_model
    using_legacy = not direct_key and bool(legacy_key or legacy_base or legacy_model)
    if not key or not endpoint_base or not model_name:
        raise RuntimeError("DeepSeek narrative configuration is unavailable")

    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_bundle["system_message"]},
            {"role": "user", "content": prompt_bundle["user_message"]},
        ],
        "max_tokens": max_tokens,
    }
    request = Request(
        endpoint_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek narrative HTTP {exc.code}: {detail[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"DeepSeek narrative connection failed: {exc}") from exc

    choice = payload["choices"][0]
    message = choice["message"]
    return {
        "provider": "openai-compatible",
        "model": model_name,
        "usage": payload.get("usage", {}),
        "finish_reason": choice.get("finish_reason"),
        "raw_content": (message.get("content") or "").strip(),
        "reasoning_content_present": bool(message.get("reasoning_content")),
        "thinking_mode": "required" if "deepseek-r1" in model_name.casefold() else "provider_default",
        "provider_dialect": (
            "dashscope" if "aliyuncs.com" in endpoint_base.casefold() else "openai-compatible"
        ),
        "json_object_requested": False,
        "configuration_source": "legacy_llm" if using_legacy else "deepseek",
        "prompt_version": prompt_bundle["prompt_version"],
        "attempt_count": 1,
        "validation_retry_count": 0,
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
    direct_key = os.environ.get("DEEPSEEK_API_KEY")
    direct_base = os.environ.get("DEEPSEEK_BASE_URL")
    direct_model = os.environ.get("DEEPSEEK_MODEL")
    legacy_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    legacy_base = os.environ.get("LLM_BASE_URL")
    legacy_model = os.environ.get("LLM_MODEL_NAME")
    direct_requested = bool(api_key or direct_key)
    key = api_key or direct_key or legacy_key
    endpoint_base = (
        base_url
        or direct_base
        or ("https://api.deepseek.com" if direct_requested else None)
        or legacy_base
    )
    model_name = (
        model
        or direct_model
        or ("deepseek-v4-flash" if direct_requested else None)
        or legacy_model
    )
    using_legacy = (
        api_key is None
        and base_url is None
        and model is None
        and not direct_key
        and bool(legacy_key or legacy_base or legacy_model)
    )
    if using_legacy:
        warnings.warn(
            "LLM_* configuration is deprecated for DeepSeek; use DEEPSEEK_* variables",
            DeprecationWarning,
            stacklevel=2,
        )
    if not key or not endpoint_base or not model_name:
        raise RuntimeError("External LLM configuration is unavailable")
    endpoint = endpoint_base.rstrip("/") + "/chat/completions"
    endpoint_host = endpoint_base.casefold()
    is_dashscope = (
        "dashscope.aliyuncs.com" in endpoint_host
        or ".maas.aliyuncs.com" in endpoint_host
    )
    normalized_model = model_name.casefold()
    dashscope_hybrid_model = normalized_model.startswith(
        ("deepseek-v4", "deepseek-v3.1", "deepseek-v3.2")
    )
    dashscope_thinking_only_model = normalized_model.startswith("deepseek-r1")
    if is_dashscope and dashscope_thinking_only_model:
        raise RuntimeError(
            f"DashScope model {model_name} is thinking-only and cannot be used for "
            "the controlled JSON report; choose deepseek-v4 with enable_thinking=false "
            "or the non-thinking deepseek-v3 model"
        )
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_bundle["system_message"]},
            {"role": "user", "content": prompt_bundle["user_message"]},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    json_object_requested = not is_dashscope or normalized_model.startswith(
        "deepseek-v4"
    )
    if json_object_requested:
        body["response_format"] = {"type": "json_object"}
    if is_dashscope:
        if dashscope_hybrid_model:
            body["enable_thinking"] = False
    elif not using_legacy:
        body["thinking"] = {"type": "disabled"}
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
        "thinking_mode": (
            "disabled"
            if (is_dashscope and not dashscope_thinking_only_model) or not using_legacy
            else "provider_default"
        ),
        "provider_dialect": "dashscope" if is_dashscope else "deepseek",
        "json_object_requested": json_object_requested,
        "configuration_source": "legacy_llm" if using_legacy else "deepseek",
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
        "generated_at": datetime.now(UTC).isoformat(),
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
