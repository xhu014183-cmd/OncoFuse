"""Frozen-LLM runtime that replaces image placeholders with projected 3D tokens."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field

from .multimodal_connector import ConnectorConfig, build_multimodal_connector, freeze_module
from .schemas import JsonModel
from .vlm_prompting import IMAGE_PATCH_TOKEN, RESEARCH_DISCLAIMER


class FrozenVlmConfig(JsonModel):
    llm_model_name: str
    llm_revision: str
    output_token_count: int = Field(default=32, ge=8, le=64)
    attention_heads: int = Field(default=8, gt=0)
    resampler_layers: int = Field(default=2, ge=1, le=8)
    trust_remote_code: bool = False
    local_files_only: bool = True
    device: str = "auto"
    dtype: str = "float32"


class VlmStructuredSummary(JsonModel):
    imaging_observations: list[str]
    clinical_context_summary: list[str]
    evidence_concordance: str
    uncertainties: list[str]
    missing_information: list[str]
    image_conditioning_statement: str = Field(min_length=1)
    research_disclaimer: str


_FORBIDDEN = re.compile(
    r"\b(?:bclc|li-?rads|diagnos\w*|stage|staging|prognos\w*)\b|"
    + "\u8bca\u65ad|\u5206\u671f|\u9884\u540e|\u6cbb\u7597\u5efa\u8bae",
    re.IGNORECASE,
)


def build_frozen_vlm(config: FrozenVlmConfig) -> Any:
    """Load a frozen causal LLM and attach the only trainable connector."""
    try:
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Frozen VLM training requires optional Torch/Transformers dependencies; "
            "install .[vlm-train]"
        ) from exc

    dtype = getattr(torch, config.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported Torch dtype: {config.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_name,
        revision=config.llm_revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_model_name,
        revision=config.llm_revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_PATCH_TOKEN]})
    llm.resize_token_embeddings(len(tokenizer))
    image_token_id = int(tokenizer.convert_tokens_to_ids(IMAGE_PATCH_TOKEN))
    hidden_size = int(llm.config.hidden_size)
    connector = build_multimodal_connector(
        ConnectorConfig(
            vision_hidden_size=768,
            llm_hidden_size=hidden_size,
            output_token_count=config.output_token_count,
            attention_heads=config.attention_heads,
            resampler_layers=config.resampler_layers,
        )
    )
    resolved_device = config.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_module(llm)
    llm.eval().to(device=resolved_device, dtype=dtype)
    connector.to(device=resolved_device, dtype=dtype)

    class FrozenM3DVlm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.llm = llm
            self.connector = connector
            self.tokenizer = tokenizer
            self.image_token_id = image_token_id
            self.device_name = resolved_device
            self.dtype_value = dtype
            self.contract = config

        def _embeddings(
            self,
            input_ids: Any,
            visual_tokens: Any,
            visual_attention_mask: Any,
            spatial_positions: Any,
            phase_ids: Any,
            timepoint_ids: Any,
        ) -> Any:
            projected = self.connector(
                visual_tokens,
                visual_attention_mask,
                spatial_positions,
                phase_ids,
                timepoint_ids,
            )
            embeddings = self.llm.get_input_embeddings()(input_ids).clone()
            for batch_index in range(input_ids.shape[0]):
                positions = torch.where(input_ids[batch_index] == self.image_token_id)[0]
                if positions.numel() != projected.shape[1]:
                    raise ValueError(
                        "Prompt image placeholder count does not match connector output: "
                        f"{positions.numel()} != {projected.shape[1]}"
                    )
                embeddings[batch_index, positions] = projected[batch_index]
            return embeddings

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            labels: Any,
            visual_tokens: Any,
            visual_attention_mask: Any,
            spatial_positions: Any,
            phase_ids: Any,
            timepoint_ids: Any,
        ) -> Any:
            inputs_embeds = self._embeddings(
                input_ids,
                visual_tokens,
                visual_attention_mask,
                spatial_positions,
                phase_ids,
                timepoint_ids,
            )
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

        def generate_summary(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            visual_tokens: Any,
            visual_attention_mask: Any,
            spatial_positions: Any,
            phase_ids: Any,
            timepoint_ids: Any,
            max_new_tokens: int = 512,
        ) -> str:
            inputs_embeds = self._embeddings(
                input_ids,
                visual_tokens,
                visual_attention_mask,
                spatial_positions,
                phase_ids,
                timepoint_ids,
            )
            with torch.inference_mode():
                generated = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    return FrozenM3DVlm()


def validate_vlm_summary(payload: str | dict[str, Any]) -> VlmStructuredSummary:
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        raw = json.loads(text)
    else:
        raw = payload
    summary = VlmStructuredSummary.model_validate(raw)
    if summary.research_disclaimer != RESEARCH_DISCLAIMER:
        raise ValueError("VLM changed the fixed research disclaimer")
    combined = "\n".join(
        [
            *summary.imaging_observations,
            *summary.clinical_context_summary,
            summary.evidence_concordance,
            summary.image_conditioning_statement,
        ]
    )
    if _FORBIDDEN.search(combined):
        raise ValueError("VLM output crossed the diagnosis/staging/prognosis safety boundary")
    return summary


__all__ = [
    "FrozenVlmConfig",
    "VlmStructuredSummary",
    "build_frozen_vlm",
    "validate_vlm_summary",
]
