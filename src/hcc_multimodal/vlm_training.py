"""Stage-one connector training with frozen visual encoder outputs and frozen LLM."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from .multimodal_connector import PHASE_IDS, TIMEPOINT_IDS, trainable_parameter_count
from .schemas import ArtifactModel, JsonModel, QualityEvidence, SourceReference
from .visual_tokens import load_visual_tokens
from .vlm_model import FrozenVlmConfig, build_frozen_vlm


class VlmTrainingExample(JsonModel):
    example_id: str
    patient_id: str
    split: Literal["train", "validation", "test"]
    token_artifact: str
    phase: str = "unknown"
    timepoint: Literal["baseline", "followup", "single"] = "single"
    prompt_text: str
    target: dict[str, Any]


class VlmTrainingManifest(ArtifactModel):
    dataset_id: str
    data_root: str = "."
    examples: list[VlmTrainingExample] = Field(min_length=1)

    @model_validator(mode="after")
    def patient_splits_do_not_leak(self) -> VlmTrainingManifest:
        patient_splits: dict[str, set[str]] = {}
        for example in self.examples:
            patient_splits.setdefault(example.patient_id, set()).add(example.split)
        leaked = sorted(patient for patient, splits in patient_splits.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"Patients occur in multiple splits: {leaked}")
        return self


class ConnectorTrainingResult(ArtifactModel):
    dataset_id: str
    llm_model_name: str
    llm_revision: str
    connector_file: str
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    train_example_count: int = Field(gt=0)
    trainable_parameter_count: int = Field(gt=0)
    frozen_llm_parameter_count: int = Field(gt=0)
    epoch_losses: list[float]
    started_at: datetime
    finished_at: datetime
    quality: QualityEvidence


def train_connector(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    model_config: FrozenVlmConfig,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 1729,
) -> tuple[ConnectorTrainingResult, Path]:
    """Train only the resampler/projector over precomputed M3D token artifacts."""
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Connector training requires optional .[vlm-train] dependencies") from exc
    manifest_source = Path(manifest_path)
    manifest = VlmTrainingManifest.model_validate_json(
        manifest_source.read_text(encoding="utf-8")
    )
    train_examples = [example for example in manifest.examples if example.split == "train"]
    if not train_examples:
        raise ValueError("Training manifest has no train examples")
    root = (manifest_source.parent / manifest.data_root).resolve()
    torch.manual_seed(seed)
    runtime = build_frozen_vlm(model_config)
    connector_parameters = [
        parameter for parameter in runtime.connector.parameters() if parameter.requires_grad
    ]
    if not connector_parameters:
        raise RuntimeError("Connector exposes no trainable parameters")
    if any(parameter.requires_grad for parameter in runtime.llm.parameters()):
        raise RuntimeError("LLM parameters must remain frozen during stage-one alignment")
    optimizer = torch.optim.AdamW(connector_parameters, lr=learning_rate)
    tokenizer = runtime.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = runtime.device_name
    started = datetime.now(UTC)
    epoch_losses: list[float] = []
    runtime.train()
    runtime.llm.eval()
    for _ in range(epochs):
        losses: list[float] = []
        for example in train_examples:
            token_output = load_visual_tokens(root / example.token_artifact)
            target_text = json.dumps(example.target, ensure_ascii=False, separators=(",", ":"))
            prompt_ids = tokenizer(example.prompt_text, add_special_tokens=True).input_ids
            full = tokenizer(
                example.prompt_text + "\n" + target_text,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = full.input_ids.to(device)
            attention_mask = full.attention_mask.to(device)
            labels = input_ids.clone()
            labels[:, : len(prompt_ids)] = -100
            visual_tokens = torch.from_numpy(token_output.tokens).unsqueeze(0).to(device)
            visual_mask = torch.from_numpy(token_output.attention_mask).unsqueeze(0).to(device)
            positions = torch.from_numpy(token_output.spatial_positions).unsqueeze(0).to(device)
            phase_id = PHASE_IDS.get(example.phase, PHASE_IDS["unknown"])
            timepoint_id = TIMEPOINT_IDS[example.timepoint]
            optimizer.zero_grad(set_to_none=True)
            result = runtime(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                visual_tokens=visual_tokens,
                visual_attention_mask=visual_mask,
                spatial_positions=positions,
                phase_ids=torch.tensor([phase_id], device=device),
                timepoint_ids=torch.tensor([timepoint_id], device=device),
            )
            loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Connector training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    connector_path = output / "m3d_llm_connector.pt"
    torch.save(runtime.connector.state_dict(), connector_path)
    finished = datetime.now(UTC)
    trainable = trainable_parameter_count(runtime.connector)
    frozen = sum(parameter.numel() for parameter in runtime.llm.parameters())
    result_artifact = ConnectorTrainingResult(
        dataset_id=manifest.dataset_id,
        llm_model_name=model_config.llm_model_name,
        llm_revision=model_config.llm_revision,
        connector_file=connector_path.name,
        epochs=epochs,
        learning_rate=learning_rate,
        train_example_count=len(train_examples),
        trainable_parameter_count=trainable,
        frozen_llm_parameter_count=frozen,
        epoch_losses=epoch_losses,
        started_at=started,
        finished_at=finished,
        quality=QualityEvidence(
            status="warning",
            warnings=[
                "Stage-one alignment trains only the connector; clinical validity is not established"
            ],
        ),
        sources=[
            SourceReference(
                source_id=manifest_source.name,
                source_type="vlm_training_manifest",
                uri=manifest_source.name,
                deidentified=True,
            )
        ],
    )
    result_artifact.write_json(output / "connector_training_result.json")
    return result_artifact, connector_path


__all__ = [
    "ConnectorTrainingResult",
    "VlmTrainingExample",
    "VlmTrainingManifest",
    "train_connector",
]
