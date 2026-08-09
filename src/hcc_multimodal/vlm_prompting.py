"""Structured prompts for image-conditioned HCC research summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .case_models import ClinicalLabEvidence, HpiTimelineEvidence
from .schemas import JsonModel

IMAGE_PATCH_TOKEN = "<im_patch>"
VISUAL_TOKEN_COUNT = 32
FusionMode = Literal["auditable", "open"]
RESEARCH_DISCLAIMER = (
    "Research evidence summary only; not for diagnosis, staging, prognosis, "
    "or treatment decisions."
)
UNVERIFIED_CONTEXT_MARKER = "[UNVERIFIED_CONTEXT]"


class ClinicalContextItem(JsonModel):
    evidence_id: str
    observed_at: str | None
    category: Literal["laboratory", "treatment", "hpi", "quality"]
    statement: str
    source_ref: str | None = None


class VlmTaskPrompt(JsonModel):
    task_version: str
    task_name: Literal["image_conditioned_structured_research_summary"]
    fusion_mode: FusionMode = "auditable"
    lab_context_injected: bool = False
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
                    source_ref=f"clinical_labs:{latest.observed_at}:{name}",
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
                    source_ref=f"hpi:{event.event_date or 'unknown'}",
                )
            )
            counter += 1
        items.append(
            ClinicalContextItem(
                evidence_id=f"HPI_{counter:03d}",
                observed_at=timeline.index_date,
                category="quality",
                statement=f"timeline_overall_trend={timeline.overall_trend}",
                source_ref="hpi:timeline",
            )
        )
    return items


def build_vlm_task_prompt(
    *,
    labs: ClinicalLabEvidence | None = None,
    timeline: HpiTimelineEvidence | None = None,
    fusion_mode: FusionMode = "auditable",
    phase: str = "unknown",
    timepoint: Literal["baseline", "followup", "single"] = "single",
    visual_token_count: int = VISUAL_TOKEN_COUNT,
    image_count: int = 0,
    imaging_metadata: str | None = None,
) -> VlmTaskPrompt:
    """Build a VLM task prompt for one fusion mode.

    ``auditable`` (default) withholds laboratory and clinical-history context
    from the prompt by design: the VLM sees image tokens and imaging metadata
    only, while laboratory trends are computed deterministically and fused at
    the rules layer. ``open`` injects laboratory observations (and, when
    supplied, the HPI timeline) as explicit ``[UNVERIFIED_CONTEXT]`` items so
    the model is forced to copy numbers verbatim and cite each evidence ID.

    When ``image_count > 0`` the visual prefix is replaced by a note that real
    images are attached as ``image_url`` parts (for vision models such as
    GLM-4V); otherwise the ``<im_patch>`` placeholder tokens are emitted for the
    M3D-LaMed path. ``imaging_metadata`` optionally appends deterministic
    measurement text (e.g. volumes, lesion counts) that both arms may cite.
    """
    inject = fusion_mode == "open"
    items = _clinical_items(labs if inject else None, timeline if inject else None)
    marker = f"{UNVERIFIED_CONTEXT_MARKER} " if inject else ""
    context = "\n".join(
        f"{marker}[{item.evidence_id}] date={item.observed_at or 'unknown'} {item.statement}"
        + (f"; source={item.source_ref}" if item.source_ref else "")
        for item in items
    )
    if inject:
        context = (
            "The clinical context below is UNVERIFIED_CONTEXT: values come from "
            "a deterministic parser, never from visual tokens. Copy every "
            "laboratory number verbatim and cite its evidence ID.\n" + context
        ) if context else "No structured clinical context was supplied."
    else:
        context = (
            "None. Laboratory and clinical-history context was withheld before "
            "prompt construction by design (auditable fusion mode). Do not "
            "reference any laboratory value or clinical-history event."
        )
    if image_count > 0:
        visual_prefix = (
            f"Visual input: {image_count} attached image(s) as image_url parts "
            "(placeholder tokens disabled)."
        )
    else:
        visual_prefix = " ".join([IMAGE_PATCH_TOKEN] * visual_token_count)
    imaging_metadata_line = f"{imaging_metadata}\n" if imaging_metadata else ""
    prompt = f"""{visual_prefix}
Image metadata: phase={phase}; timepoint={timepoint}; fusion_mode={fusion_mode}.
{imaging_metadata_line}Structured clinical context:
{context}

Task: Generate an image-conditioned structured HCC research evidence summary.
Return exactly one JSON object (not an array, not Markdown) with exactly these fields:
- fusion_mode: string
- imaging_observations: array of strings
- clinical_context_summary: array of strings
- evidence_concordance: string
- uncertainties: array of strings
- missing_information: array of strings
- image_conditioning_statement: string
- research_disclaimer: string

Rules:
- State only observations supported by the 3D visual tokens or cited clinical evidence IDs.
- Do not diagnose, stage, predict prognosis, or recommend treatment.
- Do not invent enhancement phases, measurements, laboratory values, or events.
"""
    if inject:
        prompt += (
            "- Laboratory numbers are UNVERIFIED_CONTEXT: copy each value "
            "verbatim with its unit and cite the [LAB_###] evidence ID; never "
            "re-derive, round, extrapolate, or invent values.\n"
        )
    else:
        prompt += (
            "- No laboratory values or clinical-history events were supplied; "
            "do not reference any.\n"
        )
    prompt += (
        "- Explicitly state how the image affected the output in image_conditioning_statement.\n"
        f"- research_disclaimer must equal: {RESEARCH_DISCLAIMER}\n"
    )
    locked_constraints = [
        "no diagnosis",
        "no staging",
        "no prognosis",
        "no treatment recommendation",
        "no invented numeric evidence",
    ]
    if inject:
        locked_constraints.append(
            "laboratory numbers must be copied verbatim and cited from [LAB_###] UNVERIFIED_CONTEXT items"
        )
    else:
        locked_constraints.append("no laboratory values were supplied to the model")
    metadata: dict[str, Any] = {
        "phase": phase,
        "timepoint": timepoint,
        "clinical_context_injected": inject,
        "unverified_context": inject,
        "labs_present_but_withheld": labs is not None and not inject,
        "timeline_present_but_withheld": timeline is not None and not inject,
        "visual_input": "attached_images" if image_count > 0 else "im_patch_placeholders",
        "imaging_metadata_present": imaging_metadata is not None,
        "lab_trend_provider": (
            "deterministic labs.py trend features (rules layer)"
            if not inject
            else "none (raw observations injected)"
        ),
    }
    return VlmTaskPrompt(
        task_version="hcc-vlm-summary-v2",
        task_name="image_conditioned_structured_research_summary",
        fusion_mode=fusion_mode,
        lab_context_injected=inject,
        visual_token_count=visual_token_count,
        prompt_text=prompt,
        clinical_context=items,
        output_fields=[
            "fusion_mode",
            "imaging_observations",
            "clinical_context_summary",
            "evidence_concordance",
            "uncertainties",
            "missing_information",
            "image_conditioning_statement",
            "research_disclaimer",
        ],
        locked_constraints=locked_constraints,
        metadata=metadata,
    )


def render_vlm_prompt_text(prompt: VlmTaskPrompt) -> str:
    """Render a VLM task prompt as a human-readable text bundle."""
    context_lines = "\n".join(
        f"[{item.evidence_id}] date={item.observed_at or 'unknown'} {item.statement}"
        + (f"  (source={item.source_ref})" if item.source_ref else "")
        for item in prompt.clinical_context
    ) or "(none)"
    return (
        "=== TASK ===\n"
        f"{prompt.task_name}  v{prompt.task_version}  fusion_mode={prompt.fusion_mode}\n"
        f"visual_token_count={prompt.visual_token_count}  "
        f"lab_context_injected={prompt.lab_context_injected}\n"
        "=== PROMPT ===\n"
        f"{prompt.prompt_text}\n"
        "=== CLINICAL CONTEXT ===\n"
        f"{context_lines}\n"
        "=== OUTPUT FIELDS ===\n"
        + "\n".join(f"- {field}" for field in prompt.output_fields)
        + "\n=== LOCKED CONSTRAINTS ===\n"
        + "\n".join(f"- {constraint}" for constraint in prompt.locked_constraints)
        + "\n"
    )


def write_vlm_prompt_bundle(
    prompt: VlmTaskPrompt,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> tuple[Path, Path]:
    """Write a VLM task prompt as JSON (validated contract) and readable text.

    The JSON payload is the ``VlmTaskPrompt`` model itself so it validates
    directly with ``hcc-demo validate-json --type vlm-task-prompt``. No
    timestamp is embedded, keeping repeated runs byte-identical.
    """
    json_target = Path(json_path)
    text_target = Path(text_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    text_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(prompt.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text_target.write_text(render_vlm_prompt_text(prompt), encoding="utf-8")
    return json_target, text_target


__all__ = [
    "IMAGE_PATCH_TOKEN",
    "RESEARCH_DISCLAIMER",
    "UNVERIFIED_CONTEXT_MARKER",
    "VISUAL_TOKEN_COUNT",
    "ClinicalContextItem",
    "FusionMode",
    "VlmTaskPrompt",
    "build_vlm_task_prompt",
    "render_vlm_prompt_text",
    "write_vlm_prompt_bundle",
]
