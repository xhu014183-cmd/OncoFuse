from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .schemas import RegistrationEvidence

REGISTRATION_METHOD = "simpleitk-euler3d-mattes-mi-v1"
MAX_SANE_TRANSLATION_MM = 50.0


@dataclass
class RegistrationOutcome:
    """Result of a registration attempt; resampled paths exist only when verified."""

    evidence: RegistrationEvidence
    image_path: Path | None = None
    mask_path: Path | None = None


def simpleitk_available() -> bool:
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        return False
    return True


def _mask_centroid_world(path: Path) -> np.ndarray | None:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj) > 0  # type: ignore[attr-defined]
    if not data.any():
        return None
    centroid_voxel = np.argwhere(data).mean(axis=0)
    return np.asarray(nib.affines.apply_affine(image.affine, centroid_voxel), dtype=float)  # type: ignore[attr-defined]


def register_volumes(
    baseline_image: str | Path,
    baseline_mask: str | Path,
    followup_image: str | Path,
    followup_mask: str | Path,
    output_dir: str | Path,
) -> RegistrationOutcome:
    """Rigidly register the follow-up study onto the baseline grid and resample.

    Fail-closed contract:
    - "unavailable": SimpleITK missing; callers should degrade to legacy status inference.
    - "failed": registration or resampling raised; callers must block longitudinal analysis.
    - "verified": resampled follow-up image/mask are written to output_dir and returned.

    Coordinate frame note: SimpleITK reads NIfTI (RAS) into its internal LPS
    frame, so reported translation/rotation/matrix values are in ITK LPS physical
    coordinates (x and y negated relative to nibabel RAS). The resampled NIfTI
    outputs are converted back to RAS on write and remain nibabel-compatible.
    """
    warnings: list[str] = []
    if not simpleitk_available():
        warnings.append("SimpleITK is not installed; rigid registration is unavailable")
        return RegistrationOutcome(
            evidence=RegistrationEvidence(
                status="unavailable",
                method=REGISTRATION_METHOD,
                warnings=warnings,
            )
        )

    import SimpleITK as sitk

    baseline_image_path = Path(baseline_image)
    baseline_mask_path = Path(baseline_mask)
    followup_image_path = Path(followup_image)
    followup_mask_path = Path(followup_mask)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    try:
        fixed = sitk.ReadImage(str(baseline_image_path), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(followup_image_path), sitk.sitkFloat32)

        initial = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.2, seed=42)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-3,
            numberOfIterations=200,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=1e-8,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInitialTransform(initial, inPlace=False)
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        transform = registration.Execute(fixed, moving)
        metric_value = float(registration.GetMetricValue())
        if isinstance(transform, sitk.CompositeTransform):
            transform.FlattenTransform()
            euler = transform.GetNthTransform(0).Downcast()
        else:
            euler = transform.Downcast()

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024.0)
        moved_image = resampler.Execute(moving)
        moving_mask = sitk.ReadImage(str(followup_mask_path), sitk.sitkUInt8)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        moved_mask = resampler.Execute(moving_mask)

        image_path = output / "followup_to_baseline_image.nii.gz"
        mask_path = output / "followup_to_baseline_tumor_mask.nii.gz"
        sitk.WriteImage(moved_image, str(image_path))
        sitk.WriteImage(moved_mask, str(mask_path))
    except Exception as exc:  # noqa: BLE001  # fail-closed boundary
        warnings.append(f"Rigid registration failed: {exc}")
        return RegistrationOutcome(
            evidence=RegistrationEvidence(
                status="failed",
                method=REGISTRATION_METHOD,
                warnings=warnings,
            )
        )

    rotation_deg = [round(float(np.degrees(value)), 3) for value in euler.GetParameters()[:3]]
    translation = [round(float(value), 3) for value in euler.GetTranslation()]
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = np.asarray(euler.GetMatrix(), dtype=float).reshape(3, 3)
    transform_4x4[:3, 3] = np.asarray(euler.GetTranslation(), dtype=float)

    magnitude = float(np.linalg.norm(np.asarray(translation, dtype=float)))
    if magnitude > MAX_SANE_TRANSLATION_MM:
        warnings.append(
            f"Registration translation magnitude {magnitude:.1f} mm exceeds the "
            f"{MAX_SANE_TRANSLATION_MM:.0f} mm sanity bound; inspect before trusting"
        )

    baseline_centroid = _mask_centroid_world(baseline_mask_path)
    moved_centroid = _mask_centroid_world(mask_path)
    residual: float | None = None
    if baseline_centroid is None or moved_centroid is None:
        warnings.append("Could not compute mask centroids for the registration residual")
    else:
        residual = round(float(np.linalg.norm(moved_centroid - baseline_centroid)), 3)

    evidence = RegistrationEvidence(
        status="verified",
        method=REGISTRATION_METHOD,
        transform_matrix=[[round(float(value), 6) for value in row] for row in transform_4x4.tolist()],
        translation_mm=translation,
        rotation_deg=rotation_deg,
        metric_value=round(metric_value, 6),
        centroid_residual_mm=residual,
        warnings=warnings,
    )
    return RegistrationOutcome(evidence=evidence, image_path=image_path, mask_path=mask_path)
