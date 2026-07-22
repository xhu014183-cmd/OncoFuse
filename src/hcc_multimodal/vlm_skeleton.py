"""CPU-friendly patch extraction and mock VLM flow for demonstrations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from .schemas import JsonModel


PatchMode = Literal["image_2d", "volume_3d"]


class PatchRecord(JsonModel):
    patch_id: str
    index: int
    coordinates: dict[str, int]
    shape: list[int]
    source_ref: str


class PatchManifest(JsonModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    pipeline_version: str = "0.4.0"
    generated_at: datetime
    source_file: str
    source_kind: Literal["image", "nifti"]
    mode: PatchMode
    input_shape: list[int]
    normalized_range: list[float] = Field(min_length=2, max_length=2)
    patch_size: list[int]
    grid_shape: list[int]
    patch_count: int = Field(gt=0)
    patches: list[PatchRecord]
    artifact_file: str
    preview_file: str
    demo_warning: str


class MockVlmReport(JsonModel):
    mode: Literal["mock_vlm"] = "mock_vlm"
    patch_count: int = Field(ge=0)
    visual_token_count: int = Field(ge=0)
    imaging_observations: list[str]
    hpi_summary: list[str]
    laboratory_summary: list[str]
    integrated_interpretation: list[str]
    uncertainties: list[str]
    missing_information: list[str]
    image_conditioning_statement: str
    research_disclaimer: str


def _normalize(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Input image contains no finite pixels or voxels")
    low, high = np.percentile(values[finite], [1, 99])
    if high <= low:
        high = low + 1.0
    return np.clip((np.nan_to_num(values, nan=float(low)) - low) / (high - low), 0.0, 1.0)


def _grid_and_pad(shape: tuple[int, ...], patch_size: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    grid = tuple((size + patch - 1) // patch for size, patch in zip(shape, patch_size, strict=True))
    padded = tuple(grid_value * patch for grid_value, patch in zip(grid, patch_size, strict=True))
    return grid, padded


def _save_2d_preview(data: np.ndarray, records: list[PatchRecord], path: Path) -> None:
    image = Image.fromarray(np.round(data * 255).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for record in records:
        x = record.coordinates["x"]
        y = record.coordinates["y"]
        width = record.shape[1]
        height = record.shape[0]
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(255, 80, 40), width=1)
        draw.text((x + 3, y + 3), record.patch_id, fill=(255, 255, 0))
    image.save(path)


def _save_3d_preview(data: np.ndarray, grid: tuple[int, int, int], path: Path) -> None:
    middle = data.shape[0] // 2
    image = Image.fromarray(np.round(data[middle] * 255).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    _, height, width = data.shape
    patch_height = max(height // grid[1], 1)
    patch_width = max(width // grid[2], 1)
    for y in range(0, height, patch_height):
        draw.line((0, y, width, y), fill=(255, 80, 40), width=1)
    for x in range(0, width, patch_width):
        draw.line((x, 0, x, height), fill=(255, 80, 40), width=1)
    draw.text((6, 6), f"axial z={middle}; 3D grid={grid}", fill=(255, 255, 0))
    image.save(path)


def extract_patches(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    patch_size: tuple[int, ...] | None = None,
    max_patches: int = 256,
) -> PatchManifest:
    """Extract non-overlapping 2D image or 3D NIfTI patches and an audit preview."""
    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = source.name.casefold()
    is_nifti = suffix.endswith(".nii") or suffix.endswith(".nii.gz")
    if is_nifti:
        image = nib.as_closest_canonical(nib.load(str(source)))
        if len(image.shape) != 3:
            raise ValueError(f"NIfTI input must be 3D, got shape {image.shape}")
        raw = np.asarray(image.dataobj, dtype=np.float32)
        data = _normalize(np.transpose(raw, (2, 1, 0)))
        size = patch_size or (8, 64, 64)
        if len(size) != 3 or any(value <= 0 for value in size):
            raise ValueError("3D patch_size must contain three positive integers")
        mode: PatchMode = "volume_3d"
        source_kind: Literal["image", "nifti"] = "nifti"
    else:
        image = Image.open(source).convert("L")
        data = _normalize(np.asarray(image, dtype=np.float32))
        size = patch_size or (32, 32)
        if len(size) != 2 or any(value <= 0 for value in size):
            raise ValueError("2D patch_size must contain two positive integers")
        mode = "image_2d"
        source_kind = "image"

    grid, padded_shape = _grid_and_pad(tuple(int(value) for value in data.shape), tuple(size))
    patch_count = int(np.prod(grid))
    if patch_count > max_patches:
        raise ValueError(
            f"Patch grid would contain {patch_count} patches, above max_patches={max_patches}; "
            "increase patch size or max_patches"
        )
    padded = np.pad(
        data,
        tuple((0, target - actual) for actual, target in zip(data.shape, padded_shape, strict=True)),
        mode="edge",
    )
    records: list[PatchRecord] = []
    patches: list[np.ndarray] = []
    for index in range(patch_count):
        grid_index = np.unravel_index(index, grid)
        starts = tuple(grid_index[axis] * size[axis] for axis in range(len(size)))
        slices = tuple(slice(start, start + size[axis]) for axis, start in enumerate(starts))
        patches.append(padded[slices].astype(np.float32, copy=False))
        if mode == "image_2d":
            coords = {"y": int(starts[0]), "x": int(starts[1])}
        else:
            coords = {"z": int(starts[0]), "y": int(starts[1]), "x": int(starts[2])}
        records.append(
            PatchRecord(
                patch_id=f"P_{index + 1:03d}",
                index=index,
                coordinates=coords,
                shape=list(size),
                source_ref=source.name,
            )
        )
    artifact = output / "patches.npz"
    np.savez_compressed(artifact, patches=np.stack(patches), normalized_image=data)
    preview = output / "patch-grid.png"
    if mode == "image_2d":
        _save_2d_preview(data, records, preview)
    else:
        _save_3d_preview(data, (grid[0], grid[1], grid[2]), preview)
    manifest = PatchManifest(
        generated_at=datetime.now(timezone.utc),
        source_file=source.name,
        source_kind=source_kind,
        mode=mode,
        input_shape=list(map(int, data.shape)),
        normalized_range=[0.0, 1.0],
        patch_size=list(size),
        grid_shape=list(grid),
        patch_count=patch_count,
        patches=records,
        artifact_file=artifact.name,
        preview_file=preview.name,
        demo_warning=(
            "Patch extraction is real, but the encoder and report are simulated. "
            "This artifact is not a clinical interpretation."
        ),
    )
    manifest.write_json(output / "patch-manifest.json")
    return manifest


def _text_lines(text: str | None, limit: int = 4) -> list[str]:
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()][:limit]


def run_mock_vlm(
    manifest: PatchManifest,
    *,
    hpi: str = "",
    labs: str = "",
    visual_token_count: int = 32,
) -> MockVlmReport:
    """Create a deterministic image-conditioned report for UI and pipeline demos."""
    lab_lines = _text_lines(labs)
    marker_lines = [
        line for line in lab_lines if re.search(r"\b(AFP|DCP|PIVKA|AFP[- ]?L3)\b", line, re.I)
    ]
    hpi_lines = _text_lines(hpi)
    return MockVlmReport(
        patch_count=manifest.patch_count,
        visual_token_count=visual_token_count,
        imaging_observations=[
            f"输入{manifest.mode}已切分为 {manifest.patch_count} 个 patch。",
            "当前 Demo 已将 patch 汇总为图像上下文，但未运行医学视觉模型。",
        ],
        hpi_summary=hpi_lines or ["未提供 HPI 文本。"],
        laboratory_summary=marker_lines or (lab_lines[:2] if lab_lines else ["未提供检验文本。"]),
        integrated_interpretation=[
            "这是一个可复现的 VLM 处理流向演示：patch → visual tokens → 文字上下文 → 结构化输出。",
        ],
        uncertainties=[
            "Mock VLM 不判断病灶性质、分期、预后或治疗反应。",
            "需要真实视觉编码器和人工复核才能形成研究影像证据。",
        ],
        missing_information=[
            item for item, present in (("HPI", bool(hpi_lines)), ("检验结果", bool(lab_lines))) if not present
        ],
        image_conditioning_statement=(
            f"本次输出使用了 {manifest.patch_count} 个图像 patch 的处理元数据；"
            "未调用真实 VLM 权重。"
        ),
        research_disclaimer="仅用于研究流程演示，不用于诊断、分期或治疗决策。",
    )


def render_mock_report(report: MockVlmReport) -> str:
    def section(title: str, lines: list[str]) -> str:
        return f"## {title}\n" + "\n".join(f"- {line}" for line in lines) + "\n"

    return (
        "# HCC VLM 骨架演示报告\n\n"
        f"运行模式：`{report.mode}`；patch 数：`{report.patch_count}`；"
        f"模拟 visual tokens：`{report.visual_token_count}`\n\n"
        + section("影像处理", report.imaging_observations)
        + section("HPI", report.hpi_summary)
        + section("检验", report.laboratory_summary)
        + section("综合输出", report.integrated_interpretation)
        + section("不确定性", report.uncertainties)
        + section("缺失信息", report.missing_information or ["无"])
        + f"## 图像条件化说明\n{report.image_conditioning_statement}\n\n"
        + f"> {report.research_disclaimer}\n"
    )


def run_skeleton_demo(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    hpi: str = "",
    labs: str = "",
    patch_size: tuple[int, ...] | None = None,
) -> tuple[PatchManifest, MockVlmReport, Path]:
    manifest = extract_patches(source_path, output_dir, patch_size=patch_size)
    report = run_mock_vlm(manifest, hpi=hpi, labs=labs)
    report_path = Path(output_dir) / "mock-vlm-report.json"
    report.write_json(report_path)
    (Path(output_dir) / "mock-vlm-report.md").write_text(
        render_mock_report(report), encoding="utf-8"
    )
    return manifest, report, report_path


__all__ = [
    "MockVlmReport",
    "PatchManifest",
    "PatchRecord",
    "extract_patches",
    "render_mock_report",
    "run_mock_vlm",
    "run_skeleton_demo",
]
