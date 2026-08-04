"""Trainable resampler/projector between frozen M3D tokens and a frozen LLM."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .schemas import JsonModel

PHASE_IDS = {
    "unknown": 0,
    "noncontrast": 1,
    "arterial": 2,
    "portal_venous": 3,
    "delayed": 4,
}
TIMEPOINT_IDS = {"single": 0, "baseline": 1, "followup": 2}


class ConnectorConfig(JsonModel):
    vision_hidden_size: int = Field(default=768, gt=0)
    llm_hidden_size: int = Field(gt=0)
    output_token_count: int = Field(default=32, ge=8, le=64)
    attention_heads: int = Field(default=8, gt=0)
    resampler_layers: int = Field(default=2, ge=1, le=8)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    phase_vocab_size: int = Field(default=len(PHASE_IDS), gt=0)
    timepoint_vocab_size: int = Field(default=len(TIMEPOINT_IDS), gt=0)


def build_multimodal_connector(config: ConnectorConfig) -> Any:
    """Build the optional Torch module without importing Torch in the CPU core."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "The VLM connector requires optional Torch dependencies; install .[vlm-train]"
        ) from exc

    if config.vision_hidden_size % config.attention_heads:
        raise ValueError("vision_hidden_size must be divisible by attention_heads")

    class ResamplerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_norm = nn.LayerNorm(config.vision_hidden_size)
            self.context_norm = nn.LayerNorm(config.vision_hidden_size)
            self.attention = nn.MultiheadAttention(
                config.vision_hidden_size,
                config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.feed_forward = nn.Sequential(
                nn.LayerNorm(config.vision_hidden_size),
                nn.Linear(config.vision_hidden_size, config.vision_hidden_size * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.vision_hidden_size * 4, config.vision_hidden_size),
            )

        def forward(self, queries: Any, context: Any, padding_mask: Any) -> Any:
            attended, _ = self.attention(
                self.query_norm(queries),
                self.context_norm(context),
                self.context_norm(context),
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            queries = queries + attended
            return queries + self.feed_forward(queries)

    class M3DToLlmConnector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config_contract = config
            self.queries = nn.Parameter(
                torch.randn(1, config.output_token_count, config.vision_hidden_size) * 0.02
            )
            self.position_projection = nn.Linear(3, config.vision_hidden_size)
            self.phase_embedding = nn.Embedding(
                config.phase_vocab_size,
                config.vision_hidden_size,
            )
            self.timepoint_embedding = nn.Embedding(
                config.timepoint_vocab_size,
                config.vision_hidden_size,
            )
            self.blocks = nn.ModuleList(
                ResamplerBlock() for _ in range(config.resampler_layers)
            )
            self.projector = nn.Sequential(
                nn.Linear(config.vision_hidden_size, config.llm_hidden_size),
                nn.GELU(),
                nn.Linear(config.llm_hidden_size, config.llm_hidden_size),
                nn.LayerNorm(config.llm_hidden_size),
            )

        def forward(
            self,
            visual_tokens: Any,
            attention_mask: Any,
            spatial_positions: Any,
            phase_ids: Any,
            timepoint_ids: Any,
        ) -> Any:
            if visual_tokens.ndim != 3:
                raise ValueError("visual_tokens must have shape [B,N,D]")
            if attention_mask.shape != visual_tokens.shape[:2]:
                raise ValueError("attention_mask must have shape [B,N]")
            if spatial_positions.shape != (*visual_tokens.shape[:2], 3):
                raise ValueError("spatial_positions must have shape [B,N,3]")
            context = visual_tokens + self.position_projection(spatial_positions)
            context = context + self.phase_embedding(phase_ids)[:, None, :]
            context = context + self.timepoint_embedding(timepoint_ids)[:, None, :]
            queries = self.queries.expand(visual_tokens.shape[0], -1, -1)
            padding_mask = ~attention_mask.bool()
            for block in self.blocks:
                queries = block(queries, context, padding_mask)
            return self.projector(queries)

    return M3DToLlmConnector()


def freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def trainable_parameter_count(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "PHASE_IDS",
    "TIMEPOINT_IDS",
    "ConnectorConfig",
    "build_multimodal_connector",
    "freeze_module",
    "trainable_parameter_count",
]
