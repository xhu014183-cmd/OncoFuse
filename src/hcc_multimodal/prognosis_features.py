"""Dataset-independent SEG morphology and portal-venous intensity features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import nibabel as nib
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MaskFeatureResult:
    lesion_count: int
    total_tumor_volume_ml: float
    max_lesion_extent_mm: float
    largest_lesion_sphericity: float
    portal_mean_hu: float | None
    portal_std_hu: float | None
    portal_p10_hu: float | None
    portal_p90_hu: float | None
    geometry_qc: Literal["pass", "warning"]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesion_count": self.lesion_count,
            "total_tumor_volume_ml": self.total_tumor_volume_ml,
            "max_lesion_extent_mm": self.max_lesion_extent_mm,
            "largest_lesion_sphericity": self.largest_lesion_sphericity,
            "portal_mean_hu": self.portal_mean_hu,
            "portal_std_hu": self.portal_std_hu,
            "portal_p10_hu": self.portal_p10_hu,
            "portal_p90_hu": self.portal_p90_hu,
            "geometry_qc": self.geometry_qc,
            "warnings": self.warnings,
        }


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _load_nrrd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nrrd
    except ImportError as exc:  # pragma: no cover - exercised with public-data extra
        raise RuntimeError(
            "Reading WAW-TACE NRRD masks requires the research extra: "
            "pip install -e '.[research]'"
        ) from exc
    data, header = nrrd.read(str(path), index_order="F")
    directions = np.asarray(header.get("space directions"), dtype=float)
    if directions.shape != (3, 3) or not np.isfinite(directions).all():
        spacings = np.asarray(header.get("spacings"), dtype=float)
        if spacings.shape != (3,) or not np.isfinite(spacings).all():
            raise ValueError(f"NRRD has no usable 3D geometry: {path}")
        directions = np.diag(spacings)
    origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError(f"NRRD has invalid space origin: {path}")
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = directions.T
    affine[:3, 3] = origin
    return np.asarray(data), affine


def _load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    name = source.name.casefold()
    if name.endswith((".nii", ".nii.gz")):
        image = cast(nib.Nifti1Image, nib.load(str(source)))
        if len(image.shape) != 3:
            raise ValueError(f"Only 3D volumes are supported: {source}")
        return np.asarray(image.dataobj), np.asarray(image.affine, dtype=float)
    if name.endswith((".nrrd", ".nhdr")):
        return _load_nrrd(source)
    raise ValueError(f"Unsupported volume format: {source}")


def _surface_area(mask: np.ndarray, spacing: np.ndarray) -> float:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    area = 0.0
    for axis in range(3):
        transitions = np.diff(padded.astype(np.int8), axis=axis) != 0
        face_area = float(np.prod(np.delete(spacing, axis)))
        area += float(np.count_nonzero(transitions)) * face_area
    return area


def _sphericity(mask: np.ndarray, spacing: np.ndarray) -> float:
    volume_mm3 = float(np.count_nonzero(mask) * np.prod(spacing))
    area_mm2 = _surface_area(mask, spacing)
    if volume_mm3 <= 0 or area_mm2 <= 0:
        raise ValueError("Cannot calculate sphericity for an empty component")
    value = math.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0) / area_mm2
    return min(float(value), 1.0)


def _measure_grid(
    data: np.ndarray,
    affine: np.ndarray,
    *,
    minimum_lesion_volume_ml: float,
    source_index: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    mask = np.asarray(data) > 0
    if mask.ndim != 3:
        raise ValueError("Segmentation mask must be three-dimensional")
    if not mask.any():
        raise ValueError("Tumor segmentation is empty")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8:
        raise ValueError("Segmentation affine is singular")
    spacing = np.asarray(nib.affines.voxel_sizes(affine), dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Segmentation spacing is invalid")
    source_shape = mask.shape
    nonzero = np.argwhere(mask)
    lower = nonzero.min(axis=0)
    upper = nonzero.max(axis=0) + 1
    crop = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    cropped = mask[crop]
    labels, count = ndimage.label(
        cropped,
        structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
    )
    voxel_volume_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    components: list[dict[str, Any]] = []
    warnings: list[str] = []
    for label in range(1, count + 1):
        component = labels == label
        voxel_count = int(np.count_nonzero(component))
        volume_ml = voxel_count * voxel_volume_ml
        if volume_ml < minimum_lesion_volume_ml:
            warnings.append(
                f"Discarded source {source_index} component {label} below "
                f"{minimum_lesion_volume_ml:.4f} mL"
            )
            continue
        coordinates = np.argwhere(component)
        extent_mm = float(
            ((coordinates.max(axis=0) - coordinates.min(axis=0) + 1) * spacing).max()
        )
        components.append(
            {
                "mask": component,
                "crop": crop,
                "source_shape": source_shape,
                "affine": affine,
                "spacing": spacing,
                "volume_ml": volume_ml,
                "extent_mm": extent_mm,
                "sphericity": _sphericity(component, spacing),
            }
        )
    if np.any(spacing > 5.0):
        warnings.append(f"Source {source_index} has a voxel dimension above 5 mm")
    return components, warnings


def extract_mask_features(
    mask_paths: str | Path | Sequence[str | Path],
    *,
    portal_image_path: str | Path | None = None,
    minimum_lesion_volume_ml: float = 0.01,
) -> MaskFeatureResult:
    """Extract a fixed, low-dimensional feature set from one or more SEG files.

    Masks sharing one grid are merged before relabelling. WAW occasionally stores
    separate lesions on different source grids; those masks are validated and
    relabelled independently, then aggregated without unverified resampling.
    """
    paths = [mask_paths] if isinstance(mask_paths, (str, Path)) else list(mask_paths)
    if not paths:
        raise ValueError("At least one segmentation mask is required")
    loaded = [_load_volume(path) for path in paths]
    first_data, first_affine = loaded[0]
    common_grid = all(
        data.shape == first_data.shape and np.allclose(affine, first_affine, atol=1e-3)
        for data, affine in loaded[1:]
    )
    grids: list[tuple[np.ndarray, np.ndarray]]
    warnings: list[str] = []
    if common_grid:
        union = np.zeros(first_data.shape, dtype=bool)
        for data, _ in loaded:
            union |= np.asarray(data) > 0
        grids = [(union, first_affine)]
    else:
        grids = loaded
        warnings.append(
            "SOURCE_MASK_GRIDS_DIFFER: separate public lesion masks were measured "
            "on their native grids without resampling"
        )
    components: list[dict[str, Any]] = []
    for source_index, (data, affine) in enumerate(grids, 1):
        measured, source_warnings = _measure_grid(
            data,
            affine,
            minimum_lesion_volume_ml=minimum_lesion_volume_ml,
            source_index=source_index,
        )
        components.extend(measured)
        warnings.extend(source_warnings)
    if not components:
        raise ValueError("No tumor component passed the minimum-volume threshold")
    components.sort(key=lambda item: float(item["volume_ml"]), reverse=True)
    portal_values: np.ndarray | None = None
    if portal_image_path is not None:
        image_data, image_affine = _load_volume(portal_image_path)
        largest = components[0]
        if image_data.shape != largest["source_shape"] or not np.allclose(
            image_affine, largest["affine"], atol=1e-3
        ):
            warnings.append(
                "PORTAL_INTENSITY_UNAVAILABLE: portal image and largest-lesion mask "
                "geometry do not match"
            )
        else:
            portal_values = np.asarray(image_data, dtype=float)[largest["crop"]][
                largest["mask"]
            ]
            portal_values = portal_values[np.isfinite(portal_values)]
            if not portal_values.size:
                warnings.append(
                    "PORTAL_INTENSITY_UNAVAILABLE: largest lesion has no finite CT values"
                )
                portal_values = None
    geometry_qc: Literal["pass", "warning"] = (
        "warning" if warnings else "pass"
    )
    return MaskFeatureResult(
        lesion_count=len(components),
        total_tumor_volume_ml=_round(sum(float(item["volume_ml"]) for item in components)),
        max_lesion_extent_mm=_round(max(float(item["extent_mm"]) for item in components)),
        largest_lesion_sphericity=_round(float(components[0]["sphericity"])),
        portal_mean_hu=_round(float(np.mean(portal_values))) if portal_values is not None else None,
        portal_std_hu=_round(float(np.std(portal_values))) if portal_values is not None else None,
        portal_p10_hu=_round(float(np.percentile(portal_values, 10))) if portal_values is not None else None,
        portal_p90_hu=_round(float(np.percentile(portal_values, 90))) if portal_values is not None else None,
        geometry_qc=geometry_qc,
        warnings=warnings,
    )


__all__ = ["MaskFeatureResult", "extract_mask_features"]
