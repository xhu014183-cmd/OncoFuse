"""Fail-closed adapter for optional LLM rewriting of a locked case summary."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field

from .case_models import CaseResearchSummary, EvidenceConcordance
from .case_summary import render_case_markdown
from .schemas import JsonModel


class CaseNarrativeRewrite(JsonModel):
    patient_id: str
    imaging_summary: list[str] = Field(min_length=1)
    laboratory_summary: list[str] = Field(min_length=1)
    timeline_summary: list[str] = Field(min_length=1)
    evidence_concordance: EvidenceConcordance
    data_gaps: list[str]
    uncertainty: list[str]
    disclaimer: str


_FORBIDDEN = re.compile(r"\b(diagnos|bclc|li-?rads|stage|staging|prognos|recommend(?:ed|ation)? treatment|therapy recommendation)\b|" + "\u8bca\u65ad|\u5206\u671f|\u9884\u540e|\u6cbb\u7597\u5efa\u8bae", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _locked_numbers(summary: CaseResearchSummary) -> set[str]:
    fields = [
        *summary.imaging_summary.findings,
        *summary.laboratory_summary.findings,
        *summary.timeline_summary.findings,
    ]
    return {match.group(0) for text in fields for match in _NUMBER.finditer(text)}


def validate_case_rewrite(
    summary: CaseResearchSummary,
    payload: str | dict[str, Any],
) -> tuple[CaseNarrativeRewrite | None, list[str]]:
    errors: list[str] = []
    try:
        raw = json.loads(payload) if isinstance(payload, str) else payload
        rewrite = CaseNarrativeRewrite.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, [f"LLM response schema validation failed: {exc}"]
    if rewrite.patient_id != summary.patient_id:
        errors.append("LLM changed the locked patient pseudonym")
    if rewrite.evidence_concordance != summary.evidence_concordance:
        errors.append("LLM changed the locked evidence-concordance state")
    if rewrite.disclaimer != summary.research_disclaimer:
        errors.append("LLM changed the fixed research disclaimer")
    if rewrite.data_gaps != summary.data_gaps or rewrite.uncertainty != summary.uncertainty:
        errors.append("LLM changed or omitted locked QC/gap fields")
    text = "\n".join([*rewrite.imaging_summary, *rewrite.laboratory_summary, *rewrite.timeline_summary])
    if _FORBIDDEN.search(text):
        errors.append("LLM output contains a diagnostic, staging, prognostic, or treatment assertion")
    missing_numbers = sorted(_locked_numbers(summary) - set(_NUMBER.findall(text)))
    if missing_numbers:
        errors.append(f"LLM omitted or changed locked numeric evidence: {', '.join(missing_numbers)}")
    return (None if errors else rewrite), errors


def render_with_optional_llm(
    summary: CaseResearchSummary,
    response: str | dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if response is None:
        return render_case_markdown(summary), {"mode": "deterministic", "valid": True, "errors": []}
    rewrite, errors = validate_case_rewrite(summary, response)
    if rewrite is None:
        return render_case_markdown(summary), {"mode": "deterministic_fallback", "valid": False, "errors": errors}
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"
    markdown = "\n".join([
        "# Case research evidence summary", "", "## Current imaging", "", bullets(rewrite.imaging_summary), "",
        "## Laboratory results", "", bullets(rewrite.laboratory_summary), "", "## Clinical timeline", "", bullets(rewrite.timeline_summary), "",
        "## Evidence concordance", "", rewrite.evidence_concordance, "", "## Data gaps", "", bullets(rewrite.data_gaps), "",
        "## Uncertainty", "", bullets(rewrite.uncertainty), "", "## Research-use statement", "", rewrite.disclaimer, "",
    ])
    return markdown, {"mode": "validated_llm_rewrite", "valid": True, "errors": []}


__all__ = ["CaseNarrativeRewrite", "render_with_optional_llm", "validate_case_rewrite"]
