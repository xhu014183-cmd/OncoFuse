from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def create_overlay_montage(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    *,
    window_low: float = -100.0,
    window_high: float = 300.0,
) -> Path:
    """Create a three-slice CT montage with a red mask overlay."""
    image = cast(nib.Nifti1Image, nib.load(str(image_path)))
    mask = cast(nib.Nifti1Image, nib.load(str(mask_path)))
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError("Image and mask must have matching shape and affine")
    image_data = np.asarray(image.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask.dataobj) > 0
    occupied = np.flatnonzero(mask_data.any(axis=(0, 1)))
    if occupied.size == 0:
        raise ValueError("Cannot create a tumor overlay because the mask is empty")
    fractions = (0.25, 0.5, 0.75)
    indices = [
        int(occupied[round((occupied.size - 1) * fraction)])
        for fraction in fractions
    ]

    panels: list[Image.Image] = []
    for index in indices:
        pixels = np.clip(
            (image_data[:, :, index] - window_low) / (window_high - window_low),
            0,
            1,
        )
        gray = np.rot90((pixels * 255).astype(np.uint8))
        region = np.rot90(mask_data[:, :, index])
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[region] = (
            0.35 * rgb[region] + 0.65 * np.asarray([235, 55, 45])
        ).astype(np.uint8)
        panel = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((8, 8, 104, 30), fill=(0, 0, 0))
        draw.text((14, 12), f"slice {index}", fill=(255, 255, 255))
        panels.append(panel)

    width = sum(panel.width for panel in panels)
    montage = Image.new("RGB", (width, panels[0].height), color=(0, 0, 0))
    x = 0
    for panel in panels:
        montage.paste(panel, (x, 0))
        x += panel.width
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    montage.save(target)
    sidecar = target.with_name(target.stem + "_slices.json")
    sidecar.write_text(
        json.dumps({"slice_indices": indices}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def extract_axial_slice_previews(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    count: int = 16,
    width: int = 360,
    window_low: float = -200.0,
    window_high: float = 300.0,
) -> list[dict[str, object]]:
    """Sample axial slices through the lesion as base64 PNG previews.

    Returns a list of ``{"index": int, "data_b64": str}`` where the PNG shows
    the CT window with the tumor mask overlaid in red. Slice indices are
    concentrated on the mask-occupied range so the viewer lands on the lesion.
    """
    image = cast(nib.Nifti1Image, nib.load(str(image_path)))
    mask = cast(nib.Nifti1Image, nib.load(str(mask_path)))
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError("Image and mask must have matching shape and affine")
    image_data = np.asarray(image.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask.dataobj) > 0
    occupied = np.flatnonzero(mask_data.any(axis=(0, 1)))
    depth = image_data.shape[2]
    if occupied.size:
        lo, hi = int(occupied.min()), int(occupied.max())
    else:
        lo, hi = 0, max(depth - 1, 0)
    margin = 8
    lo = max(0, lo - margin)
    hi = min(depth - 1, hi + margin)
    span = max(hi - lo, 1)
    if count <= 1:
        indices = [lo]
    else:
        indices = [
            min(depth - 1, max(0, round(lo + span * t / (count - 1))))
            for t in range(count)
        ]
        seen: set[int] = set()
        unique_indices: list[int] = []
        for index in indices:
            if index not in seen:
                seen.add(index)
                unique_indices.append(index)
        indices = unique_indices
    previews: list[dict[str, object]] = []
    for index in indices:
        pixels = np.clip(
            (image_data[:, :, index] - window_low) / (window_high - window_low),
            0,
            1,
        )
        gray = np.rot90((pixels * 255).astype(np.uint8))
        region = np.rot90(mask_data[:, :, index])
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[region] = (
            0.35 * rgb[region] + 0.65 * np.asarray([235, 55, 45])
        ).astype(np.uint8)
        panel = Image.fromarray(rgb, mode="RGB")
        if panel.width > width:
            height = max(round(panel.height * width / panel.width), 1)
            panel = panel.resize((width, height))
        buffer = io.BytesIO()
        panel.save(buffer, format="PNG")
        previews.append(
            {
                "index": index,
                "data_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        )
    return previews


def extract_lesion_zoom(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    width: int = 360,
    crop: int = 150,
    window_low: float = -200.0,
    window_high: float = 300.0,
) -> dict[str, object]:
    """Crop and magnify the lesion on its centroid axial slice."""
    image = cast(nib.Nifti1Image, nib.load(str(image_path)))
    mask = cast(nib.Nifti1Image, nib.load(str(mask_path)))
    image_data = np.asarray(image.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask.dataobj) > 0
    coords = np.argwhere(mask_data)
    if coords.size == 0:
        raise ValueError("Cannot zoom into an empty tumor mask")
    center_x, center_y, center_z = coords.mean(axis=0)
    center_z = round(center_z)
    center_y = round(center_y)
    center_x = round(center_x)
    n_x, n_y, _ = mask_data.shape
    half = crop // 2
    y0 = max(0, center_y - half)
    x0 = max(0, center_x - half)
    y1 = min(n_y, y0 + crop)
    x1 = min(n_x, x0 + crop)
    pixels = np.clip(
        (image_data[y0:y1, x0:x1, center_z] - window_low) / (window_high - window_low),
        0,
        1,
    )
    gray = np.rot90((pixels * 255).astype(np.uint8))
    region = np.rot90(mask_data[y0:y1, x0:x1, center_z])
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[region] = (
        0.35 * rgb[region] + 0.65 * np.asarray([235, 55, 45])
    ).astype(np.uint8)
    panel = Image.fromarray(rgb, mode="RGB")
    if panel.width > width:
        panel = panel.resize((width, max(round(panel.height * width / panel.width), 1)))
    buffer = io.BytesIO()
    panel.save(buffer, format="PNG")
    return {
        "slice_index": center_z,
        "data_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
