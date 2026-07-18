from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol, cast
import math

import nibabel as nib
import numpy as np
from scipy import ndimage

from .schemas import (
    ImageEmbeddingEvidence,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
)


M3D_MODEL_ID = "GoodBaiBai88/M3D-CLIP"
M3D_MODEL_REVISION = "ae091d89a0ef38b533ecc4ed21426f7658853963"
M3D_TARGET_SHAPE = (32, 256, 256)
TARGET_SPACING_XYZ_MM = (1.5, 1.5, 2.5)
CT_WINDOW_HU = (-200.0, 300.0)
ROI_MARGIN_MM = 16.0


class EncoderUnavailableError(RuntimeError):
    """Raised when an explicitly requested optional encoder cannot be used."""


@dataclass(frozen=True)
class VolumeInput:
    image: np.ndarray
    mask: np.ndarray | None
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    warnings: tuple[str, ...]
    modality: str = "CT"
    phase: str = "unknown"


@dataclass(frozen=True)
class EncodingResult:
    vector: np.ndarray
    normalized: bool
    preprocessing: dict[str, Any]
    mask_usage: str
    warnings: tuple[str, ...]


class VolumeEncoder(Protocol):
    name: str
    model_name: str
    model_revision: str
    intended_use: str

    def encode(self, volume: VolumeInput) -> EncodingResult:
        ...


def _load_volume(
    image_path: str | Path,
    mask_path: str | Path | None,
    *,
    modality: str = "CT",
    phase: str = "unknown",
) -> VolumeInput:
    image = cast(nib.Nifti1Image, nib.load(str(image_path)))
    if len(image.shape) != 3:
        raise ValueError(f"Only 3D NIfTI volumes are supported, got shape {image.shape}")

    mask = None
    if mask_path is not None:
        mask_image = cast(nib.Nifti1Image, nib.load(str(mask_path)))
        if image.shape != mask_image.shape:
            raise ValueError(
                f"Image shape {image.shape} does not match mask shape {mask_image.shape}"
            )
        if not np.allclose(image.affine, mask_image.affine, atol=1e-3):
            raise ValueError("Image and mask affines do not match")
        mask_image = cast(nib.Nifti1Image, nib.as_closest_canonical(mask_image))
        mask = np.asarray(mask_image.dataobj) > 0

    image = cast(nib.Nifti1Image, nib.as_closest_canonical(image))
    data = np.asarray(image.dataobj, dtype=np.float32)
    warnings: list[str] = []
    if not np.isfinite(data).all():
        warnings.append("Non-finite image values were replaced before encoding")
    shape = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    voxel_sizes = nib.affines.voxel_sizes(image.affine)
    spacing = (float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2]))
    return VolumeInput(
        image=data,
        mask=mask,
        shape=shape,
        spacing_mm=spacing,
        warnings=tuple(warnings),
        modality=modality.upper(),
        phase=phase,
    )


def _finite_image(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Image contains no finite values")
    if finite.all():
        return data.astype(np.float32, copy=False)
    replacement = float(np.median(data[finite]))
    return np.where(finite, data, replacement).astype(np.float32, copy=False)


class StatisticalVolumeEncoder:
    """Deterministic, non-learned provider used to exercise the adapter contract."""

    name = "statistical-v1"
    model_name = "non-learned-volume-summary"
    model_revision = "1"
    intended_use = (
        "deterministic integration baseline only; not a learned representation and not for "
        "clinical prediction"
    )
    _quantiles = (1, 10, 25, 50, 75, 90, 99)

    def encode(self, volume: VolumeInput) -> EncodingResult:
        data, mask, transform, preprocessing_warnings = preprocess_physical_roi(volume)
        global_values = data.reshape(-1)
        features = [*np.percentile(global_values, self._quantiles)]
        features.extend([float(global_values.mean()), float(global_values.std())])
        warnings = list(preprocessing_warnings)

        if mask is not None and mask.any():
            masked = data[mask]
            features.extend(np.percentile(masked, self._quantiles))
            features.extend([float(masked.mean()), float(masked.std())])
            mask_fraction = float(mask.mean())
            mask_usage = "physically resampled lesion ROI and mask-derived summary features"
        else:
            features.extend([0.0] * (len(self._quantiles) + 2))
            mask_fraction = 0.0
            mask_usage = "mask unavailable or empty; masked features are zero placeholders"
            warnings.append("No non-empty mask was available to the statistical encoder")

        features.append(mask_fraction)
        features.extend(TARGET_SPACING_XYZ_MM)
        features.extend(data.shape)
        feature_names = [
            *(f"global_p{value}" for value in self._quantiles),
            "global_mean",
            "global_std",
            *(f"masked_p{value}" for value in self._quantiles),
            "masked_mean",
            "masked_std",
            "mask_fraction",
            "spacing_x_mm",
            "spacing_y_mm",
            "spacing_z_mm",
            "shape_x",
            "shape_y",
            "shape_z",
        ]
        vector = np.asarray(features, dtype=np.float32)
        return EncodingResult(
            vector=vector,
            normalized=False,
            preprocessing={
                "version": "statistical-volume-v2",
                "orientation": "closest canonical orientation",
                **transform,
                "feature_names": feature_names,
            },
            mask_usage=mask_usage,
            warnings=tuple(sorted(set(warnings))),
        )


def _resize_exact(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    order: int = 1,
) -> np.ndarray:
    factors = [target / current for target, current in zip(target_shape, volume.shape)]
    resized = ndimage.zoom(volume, zoom=factors, order=order, mode="nearest", prefilter=False)
    output = np.zeros(target_shape, dtype=np.float32)
    source_slices = tuple(slice(0, min(source, target)) for source, target in zip(resized.shape, target_shape))
    target_slices = tuple(slice(0, min(source, target)) for source, target in zip(resized.shape, target_shape))
    output[target_slices] = resized[source_slices]
    return output


def preprocess_physical_roi(
    volume: VolumeInput,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], tuple[str, ...]]:
    """Apply physical resampling, fixed CT windowing, and synchronized lesion ROI cropping."""
    data = _finite_image(volume.image)
    warnings = list(volume.warnings)
    if volume.modality != "CT":
        warnings.append(
            f"Fixed CT intensity window was not applied to modality {volume.modality}; "
            "finite min-max normalization was used"
        )
        minimum = float(data.min())
        maximum = float(data.max())
        normalized = (data - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(data)
        intensity = {
            "method": "finite_min_max",
            "input_min": minimum,
            "input_max": maximum,
        }
    else:
        window_lower, window_upper = CT_WINDOW_HU
        normalized = (np.clip(data, window_lower, window_upper) - window_lower) / (
            window_upper - window_lower
        )
        intensity = {
            "method": "fixed_ct_window",
            "window_hu": [window_lower, window_upper],
            "output_range": [0, 1],
        }

    zoom_factors = tuple(
        input_spacing / target_spacing
        for input_spacing, target_spacing in zip(volume.spacing_mm, TARGET_SPACING_XYZ_MM, strict=True)
    )
    resampled = ndimage.zoom(
        normalized.astype(np.float32),
        zoom=zoom_factors,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)
    resampled_mask = None
    original_mask_voxels = 0
    if volume.mask is not None:
        original_mask_voxels = int(volume.mask.sum())
        resampled_mask = ndimage.zoom(
            volume.mask.astype(np.uint8),
            zoom=zoom_factors,
            order=0,
            mode="nearest",
            prefilter=False,
        ) > 0
        if original_mask_voxels and not resampled_mask.any():
            raise ValueError("Physical resampling removed all lesion-mask voxels")

    crop_bounds = [[0, int(size)] for size in resampled.shape]
    if resampled_mask is not None and resampled_mask.any():
        coordinates = np.argwhere(resampled_mask)
        roi_lower = coordinates.min(axis=0)
        roi_upper = coordinates.max(axis=0) + 1
        margin = np.ceil(
            np.asarray([ROI_MARGIN_MM] * 3) / np.asarray(TARGET_SPACING_XYZ_MM)
        ).astype(int)
        roi_lower = np.maximum(roi_lower - margin, 0)
        roi_upper = np.minimum(roi_upper + margin, resampled.shape)
        crop_bounds = [[int(roi_lower[axis]), int(roi_upper[axis])] for axis in range(3)]
        slices = tuple(slice(int(roi_lower[axis]), int(roi_upper[axis])) for axis in range(3))
        resampled = resampled[slices]
        resampled_mask = resampled_mask[slices]
        if not resampled_mask.any():
            raise ValueError("ROI crop removed all lesion-mask voxels")
    else:
        warnings.append("No non-empty lesion mask was available; the complete resampled volume was used")

    transform = {
        "physical_resampling": {
            "input_shape_xyz": list(volume.shape),
            "input_spacing_xyz_mm": list(volume.spacing_mm),
            "target_spacing_xyz_mm": list(TARGET_SPACING_XYZ_MM),
            "zoom_factors_xyz": [round(value, 8) for value in zoom_factors],
            "resampled_shape_xyz": [int(round(volume.shape[i] * zoom_factors[i])) for i in range(3)],
            "image_interpolation": "linear",
            "mask_interpolation": "nearest",
        },
        "intensity": intensity,
        "roi": {
            "basis": "lesion_mask_with_physical_margin" if original_mask_voxels else "complete_volume",
            "margin_mm": ROI_MARGIN_MM if original_mask_voxels else None,
            "crop_bounds_xyz_start_stop": crop_bounds,
            "cropped_shape_xyz": list(resampled.shape),
            "input_mask_voxels": original_mask_voxels,
            "cropped_resampled_mask_voxels": int(resampled_mask.sum()) if resampled_mask is not None else 0,
        },
        "phase": volume.phase,
        "modality": volume.modality,
    }
    return resampled, resampled_mask, transform, tuple(sorted(set(warnings)))


def preprocess_m3d_volume(volume: VolumeInput) -> tuple[np.ndarray, dict[str, Any], tuple[str, ...]]:
    """Convert canonical NIfTI data to M3D's [C,D,H,W] model input contract."""
    normalized, resampled_mask, transform, preprocessing_warnings = preprocess_physical_roi(volume)
    warnings = list(preprocessing_warnings)
    depth_first = np.transpose(normalized, (2, 1, 0))
    resized = _resize_exact(depth_first.astype(np.float32), M3D_TARGET_SHAPE)
    resized_mask_voxels = None
    if resampled_mask is not None:
        resized_mask = _resize_exact(
            np.transpose(resampled_mask.astype(np.float32), (2, 1, 0)),
            M3D_TARGET_SHAPE,
            order=0,
        ) > 0
        resized_mask_voxels = int(resized_mask.sum())
        if resampled_mask.any() and not resized_mask.any():
            raise ValueError("M3D fixed-shape resize removed all lesion-mask voxels")
    model_input = resized[np.newaxis, ...]
    preprocessing = {
        "version": "m3d-clip-nifti-v2",
        "orientation": "closest canonical orientation",
        "axis_order": "NIfTI X,Y,Z transposed to channel,D,H,W",
        **transform,
        "resize": {
            "target_shape_cdhw": [1, *M3D_TARGET_SHAPE],
            "method": "trilinear scipy.ndimage.zoom",
            "mask_method": "nearest scipy.ndimage.zoom",
            "resized_mask_voxels": resized_mask_voxels,
        },
    }
    warnings.append(
        "M3D-CLIP resizes the physical ROI to 32 slices; small-lesion detail may be lost"
    )
    warnings.append(
        "The CT window and resampling adapter are research preprocessing, not a clinically validated protocol"
    )
    return model_input, preprocessing, tuple(sorted(set(warnings)))


class M3DClipVolumeEncoder:
    name = "m3d-clip"
    model_name = M3D_MODEL_ID
    model_revision = M3D_MODEL_REVISION
    intended_use = (
        "research and non-commercial evaluation only; training-data and derived-weight rights "
        "require separate review before commercial use"
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        allow_remote_code: bool = False,
        allow_model_download: bool = False,
    ) -> None:
        self.device = device
        self.allow_remote_code = allow_remote_code
        self.allow_model_download = allow_model_download
        self._model = None
        self._torch = None

    def _load_model(self):
        if not self.allow_remote_code:
            raise EncoderUnavailableError(
                "M3D-CLIP uses pinned Hugging Face remote model code. Re-run with explicit "
                "--allow-m3d-remote-code after reviewing the model implementation."
            )
        try:
            import torch
            from transformers import AutoModel
            import monai  # noqa: F401
        except ImportError as exc:
            raise EncoderUnavailableError(
                "M3D-CLIP optional dependencies are missing. Install them with: "
                "pip install -e '.[m3d]'"
            ) from exc

        if self._model is None:
            try:
                model = AutoModel.from_pretrained(
                    self.model_name,
                    revision=self.model_revision,
                    trust_remote_code=True,
                    local_files_only=not self.allow_model_download,
                )
            except OSError as exc:
                raise EncoderUnavailableError(
                    "The pinned M3D-CLIP model is not available in the local cache. Re-run with "
                    "--allow-model-download to download the research weights."
                ) from exc
            resolved_device = self.device
            if resolved_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = resolved_device
            self._model = model.eval().to(device=resolved_device)
            self._torch = torch
        return self._model, self._torch

    def encode(self, volume: VolumeInput) -> EncodingResult:
        model_input, preprocessing, warnings = preprocess_m3d_volume(volume)
        model, torch = self._load_model()
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            tokens = model.encode_image(tensor)
        if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[-1] != 768:
            raise RuntimeError(f"Unexpected M3D-CLIP output shape: {tuple(tokens.shape)}")
        vector = tokens[0, 0].detach().float().cpu().numpy().astype(np.float32)
        preprocessing = {
            **preprocessing,
            "model_input_shape_bcdhw": [1, 1, *M3D_TARGET_SHAPE],
            "pooling": "CLS token from encode_image output",
            "device": self.device,
            "dtype": "float32",
        }
        return EncodingResult(
            vector=vector,
            normalized=True,
            preprocessing=preprocessing,
            mask_usage="mask is not consumed by M3D-CLIP; mask measurements remain a separate branch",
            warnings=warnings,
        )


EncoderFactory = Callable[..., VolumeEncoder]
_ENCODER_FACTORIES: dict[str, EncoderFactory] = {
    StatisticalVolumeEncoder.name: lambda **_: StatisticalVolumeEncoder(),
    M3DClipVolumeEncoder.name: lambda **options: M3DClipVolumeEncoder(**options),
}


def available_volume_encoders() -> tuple[str, ...]:
    return tuple(sorted(_ENCODER_FACTORIES))


def register_volume_encoder(name: str, factory: EncoderFactory, *, replace: bool = False) -> None:
    if not name or name.strip() != name:
        raise ValueError("Encoder name must be a non-empty, trimmed string")
    if name in _ENCODER_FACTORIES and not replace:
        raise ValueError(f"Volume encoder {name!r} is already registered")
    _ENCODER_FACTORIES[name] = factory


def create_volume_encoder(name: str, **options: Any) -> VolumeEncoder:
    try:
        factory = _ENCODER_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown volume encoder {name!r}; available encoders: "
            f"{', '.join(available_volume_encoders())}"
        ) from exc
    return factory(**options)


def encode_nifti_volume(
    image_path: str | Path,
    mask_path: str | Path | None,
    *,
    patient_id: str,
    study_date: str,
    encoder_name: str,
    artifact_path: str | Path,
    encoder_options: dict[str, Any] | None = None,
    modality: str = "CT",
    phase: str = "unknown",
) -> tuple[ImageEmbeddingEvidence, Path]:
    """Run one provider and write its vector separately from the JSON manifest."""
    encoder = create_volume_encoder(encoder_name, **(encoder_options or {}))
    volume = _load_volume(image_path, mask_path, modality=modality, phase=phase)
    result = encoder.encode(volume)
    vector = np.asarray(result.vector, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"Encoder output must be a non-empty 1D vector, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Encoder output contains non-finite values")
    l2_norm = float(np.linalg.norm(vector))
    if not math.isfinite(l2_norm) or l2_norm <= 0:
        raise ValueError("Encoder output has an invalid L2 norm")
    if result.normalized:
        vector = vector / l2_norm
        l2_norm = float(np.linalg.norm(vector))

    target = Path(artifact_path)
    if target.suffix.lower() != ".npy":
        raise ValueError("Image embedding artifact path must end in .npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, vector, allow_pickle=False)
    digest = sha256(target.read_bytes()).hexdigest()
    source_evidence_id = f"{patient_id}:imaging:{study_date}"
    quality_status: QualityStatus = "warning" if result.warnings else "pass"
    evidence = ImageEmbeddingEvidence(
        source_evidence_id=source_evidence_id,
        patient_id=patient_id,
        study_date=study_date,
        phase=phase,
        encoder_name=encoder.name,
        model_name=encoder.model_name,
        model_revision=encoder.model_revision,
        embedding_dimension=int(vector.size),
        embedding_dtype=str(vector.dtype),
        normalized=result.normalized,
        l2_norm=round(l2_norm, 8),
        artifact_file=target.name,
        artifact_sha256=digest,
        input_shape=list(volume.shape),
        input_spacing_mm=[round(value, 6) for value in volume.spacing_mm],
        preprocessing=result.preprocessing,
        mask_usage=result.mask_usage,
        available=True,
        quality=QualityEvidence(
            status=quality_status,
            checks=[
                QualityCheck(
                    check_id="EMBEDDING_FINITE",
                    status="pass",
                    message="Embedding is one-dimensional, finite, and non-empty",
                ),
                QualityCheck(
                    check_id="EMBEDDING_L2_NORM",
                    status="pass",
                    message=f"Embedding L2 norm is {l2_norm:.8g}",
                ),
            ],
            warnings=list(result.warnings),
        ),
        warnings=list(result.warnings),
        intended_use=encoder.intended_use,
        sources=[
            SourceReference(
                source_id=Path(image_path).name,
                source_type="nifti_image",
                uri=Path(image_path).name,
            )
        ],
    )
    return evidence, target
