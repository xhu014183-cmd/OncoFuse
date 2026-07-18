from __future__ import annotations

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
    return target
