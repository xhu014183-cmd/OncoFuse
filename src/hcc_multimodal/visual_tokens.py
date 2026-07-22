"""Auditable full-token artifacts for 3D vision-language models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .schemas import ArtifactModel, QualityEvidence, SourceReference


@dataclass(frozen=True)
class VisualTokenOutput:
    """In-memory token sequence; never serialized into public JSON."""

    tokens: np.ndarray
    attention_mask: np.ndarray
    global_embedding: np.ndarray
    spatial_positions: np.ndarray
    spatial_mapping: Literal["model_declared", "inferred", "unavailable"]
    grid_shape: tuple[int, int, int] | None
    preprocessing: dict[str, Any]
    warnings: tuple[str, ...]


class VisualTokenManifest(ArtifactModel):
    source_evidence_id: str
    patient_id: str
    study_date: str
    phase: str
    timepoint: Literal["baseline", "followup", "single"]
    encoder_name: str
    model_name: str
    model_revision: str
    token_count: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    has_cls_token: bool
    grid_shape_dhw: list[int] | None
    spatial_mapping: Literal["model_declared", "inferred", "unavailable"]
    artifact_file: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_arrays: list[str]
    preprocessing: dict[str, Any]
    available: bool = True
    quality: QualityEvidence
    warnings: list[str]
    intended_use: str


def save_visual_tokens(
    output: VisualTokenOutput,
    path: str | Path,
) -> tuple[Path, str]:
    target = Path(path)
    if target.suffix.casefold() != ".npz":
        raise ValueError("Visual token artifact path must end in .npz")
    tokens = np.asarray(output.tokens, dtype=np.float32)
    mask = np.asarray(output.attention_mask, dtype=np.uint8)
    global_embedding = np.asarray(output.global_embedding, dtype=np.float32)
    positions = np.asarray(output.spatial_positions, dtype=np.float32)
    if tokens.ndim != 2 or not tokens.size or not np.isfinite(tokens).all():
        raise ValueError("Visual tokens must be a non-empty finite [N,D] array")
    if mask.shape != (tokens.shape[0],):
        raise ValueError("Visual token attention mask must have shape [N]")
    if global_embedding.shape != (tokens.shape[1],):
        raise ValueError("Global embedding must have shape [D]")
    if positions.shape != (tokens.shape[0], 3):
        raise ValueError("Spatial positions must have shape [N,3]")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        tokens=tokens,
        attention_mask=mask,
        global_embedding=global_embedding,
        spatial_positions=positions,
    )
    return target, sha256(target.read_bytes()).hexdigest()


def load_visual_tokens(path: str | Path) -> VisualTokenOutput:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        mask = np.asarray(payload["attention_mask"], dtype=np.uint8)
        global_embedding = np.asarray(payload["global_embedding"], dtype=np.float32)
        positions = np.asarray(payload["spatial_positions"], dtype=np.float32)
    output = VisualTokenOutput(
        tokens=tokens,
        attention_mask=mask,
        global_embedding=global_embedding,
        spatial_positions=positions,
        spatial_mapping="unavailable",
        grid_shape=None,
        preprocessing={},
        warnings=(),
    )
    # Reuse the writer validation without modifying the source artifact.
    if output.tokens.ndim != 2 or output.attention_mask.shape != (output.tokens.shape[0],):
        raise ValueError("Invalid visual token artifact shapes")
    if output.global_embedding.shape != (output.tokens.shape[1],):
        raise ValueError("Invalid global embedding shape")
    if output.spatial_positions.shape != (output.tokens.shape[0], 3):
        raise ValueError("Invalid spatial position shape")
    if not all(np.isfinite(value).all() for value in (tokens, global_embedding, positions)):
        raise ValueError("Visual token artifact contains non-finite values")
    return output


def manifest_sources(image_path: str | Path) -> list[SourceReference]:
    image = Path(image_path)
    return [
        SourceReference(
            source_id=image.name,
            source_type="nifti_image",
            uri=image.name,
        )
    ]


__all__ = [
    "VisualTokenManifest",
    "VisualTokenOutput",
    "load_visual_tokens",
    "manifest_sources",
    "save_visual_tokens",
]
