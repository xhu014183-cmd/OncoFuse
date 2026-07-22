"""Structured prompts for image-conditioned HCC research summaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .case_models import ClinicalLabEvidence, HpiTimelineEvidence
from .schemas import JsonModel


IMAGE_PATCH_TOKEN = "<im_patch>"
VISUAL_TOKEN_COUNT = 32
RESEARCH_DISCLAIMER = (
    "Research evidence summary only; not for diagnosis, staging, prognosis, "
    "or treatment decisions."
)


class ClinicalContextItem(JsonModel):
    evidence_id: str
    observed_at: str | None
    category: Literal["laboratory", "treatment", "hpi", "quality"]
    statement: str


class VlmTaskPrompt(JsonModel):
    task_version: str
    task_name: Literal["image_conditioned_structured_research_summary"]
    visual_token_count: int = Field(ge=8, le=64)
    prompt_text: str
    clinical_context: list[ClinicalContextItem]
    output_fields: list[str]
    locked_constraints: list[str]
    metadata: dict[str, Any]


def _clinical_items(
    labs: ClinicalLabEvidence | None,
    timeline: HpiTimelineEvidence | None,
) -> list[ClinicalContextItem]:
    items: list[ClinicalContextItem] = []
    if labs is not None:
        counter = 1
        for name, evidence in sorted(labs.analytes.items()):
            latest = next(
                (item for item in reversed(evidence.observations) if item.value is not None),
                None,
            )
            if latest is None:
                continue
            items.append(
                ClinicalContextItem(
                    evidence_id=f"LAB_{counter:03d}",
                    observed_at=latest.observed_at,
                    category="laboratory",
                    statement=(
                        f"{name} {latest.comparator} {latest.value:g} {latest.unit}; "
                        f"trajectory={evidence.trajectory_state}; "
                        f"above_reference={latest.above_reference}"
                    ),
                )
            )
            counter += 1
        for warning in labs.quality.warnings:
            items.append(
                ClinicalContextItem(
                    evidence_id=f"LAB_{counter:03d}",
                    observed_at=None,
                    category="quality",
                    statement=warning,
                )
            )
            counter += 1
    if timeline is not None:
        counter = 1
        for event in timeline.events:
            if event.event_type not in {"treatment", "surgery", "procedure", "imaging"}:
                continue
            items.append(
                ClinicalContextItem(
                    evidence_id=f"HPI_{counter:03d}",
                    observed_at=event.event_date,
                    category="treatment" if event.event_type in {"treatment", "surgery"} else "hpi",
                    statement=f"{event.normalized_concept}: {event.text}",
                )
            )
            counter += 1
        items.append(
            ClinicalContextItem(
                evidence_id=f"HPI_{counter:03d}",
                observed_at=timeline.index_date,
                category="quality",
                statement=f"timeline_overall_trend={timeline.overall_trend}",
            )
        )
    return items


def build_vlm_task_prompt(
    *,
    labs: ClinicalLabEvidence | None = None,
    timeline: HpiTimelineEvidence | None = None,
    phase: str = "unknown",
    timepoint: Literal["baseline", "followup", "single"] = "single",
    visual_token_count: int = VISUAL_TOKEN_COUNT,
) -> VlmTaskPrompt:
    items = _clinical_items(labs, timeline)
    context = "\n".join(
        f"[{item.evidence_id}] date={item.observed_at or 'unknown'} {item.statement}"
        for item in items
    ) or "No structured clinical context was supplied."
    visual_prefix = " ".join([IMAGE_PATCH_TOKEN] * visual_token_count)
    prompt = f"""{visual_prefix}
Image metadata: phase={phase}; timepoint={timepoint}.
Structured clinical context:
{context}

Task: Generate an image-conditioned structured HCC research evidence summary.
Return one JSON object with exactly these fields:
imaging_observations, clinical_context_summary, evidence_concordance,
uncertainties, missing_information, image_conditioning_statement,
research_disclaimer.

Rules:
- State only observations supported by the 3D visual tokens or cited clinical evidence IDs.
- Do not diagnose, stage, predict prognosis, or recommend treatment.
- Do not invent enhancement phases, measurements, laboratory values, or events.
- Explicitly state how the image affected the output in image_conditioning_statement.
- research_disclaimer must equal: {RESEARCH_DISCLAIMER}
"""
    return VlmTaskPrompt(
        task_version="hcc-vlm-summary-v1",
        task_name="image_conditioned_structured_research_summary",
        visual_token_count=visual_token_count,
        prompt_text=prompt,
        clinical_context=items,
        output_fields=[
            "imaging_observations",
            "clinical_context_summary",
            "evidence_concordance",
            "uncertainties",
            "missing_information",
            "image_conditioning_statement",
            "research_disclaimer",
        ],
        locked_constraints=[
            "no diagnosis",
            "no staging",
            "no prognosis",
            "no treatment recommendation",
            "no invented numeric evidence",
        ],
        metadata={"phase": phase, "timepoint": timepoint},
    )


__all__ = [
    "IMAGE_PATCH_TOKEN",
    "RESEARCH_DISCLAIMER",
    "VISUAL_TOKEN_COUNT",
    "ClinicalContextItem",
    "VlmTaskPrompt",
    "build_vlm_task_prompt",
]
