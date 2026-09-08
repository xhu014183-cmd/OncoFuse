"""LiON-inspired pixel/lesion/patient evidence adapter.

This module deliberately does not claim to run the clinical LiON model.  The
MVP backend converts a supplied, aligned segmentation mask into a hierarchical
and auditable evidence artifact.  A future PLAN/LiON implementation can replace
the backend without changing downstream fusion or reporting contracts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib

from .imaging import measure_nifti
from .schemas import (
    LionInspiredImagingEvidence,
    LionLesionEvidence,
    LionPatientEvidence,
    LionPixelEvidence,
    QualityCheck,
    QualityEvidence,
    SegRole,
    SourceReference,
)

LION_INTENDED_USE = (
    "LiON-inspired research evidence hierarchy using a supplied segmentation; "
    "not the clinical LiON model and not for diagnosis"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LionBackend(Protocol):
    """Replaceable interface for pixel/lesion/patient imaging evidence."""

    def infer(
        self,
        *,
        image_path: str | Path,
        mask_path: str | Path | None,
        mask_role: SegRole | None,
        patient_id: str,
        study_date: str,
        phase: str,
    ) -> LionInspiredImagingEvidence: ...


class PrecomputedMaskLionBackend:
    """Build hierarchical evidence from an existing mask without classification."""

    name = "precomputed_mask"

    def infer(
        self,
        *,
        image_path: str | Path,
        mask_path: str | Path | None,
        mask_role: SegRole | None,
        patient_id: str,
        study_date: str,
        phase: str,
    ) -> LionInspiredImagingEvidence:
        image = Path(image_path)
        if not image.is_file():
            raise FileNotFoundError(f"NIfTI image not found: {image}")
        volume = cast(nib.Nifti1Image, nib.load(str(image)))
        if len(volume.shape) != 3:
            raise ValueError("LiON-inspired MVP accepts one three-dimensional CT volume")

        common_limitations = [
            "Single-phase CT cannot establish a complete dynamic enhancement pattern",
            "No malignancy probability, HCC class probability, LI-RADS, or staging is produced",
            "This is a LiON-inspired evidence adapter, not the clinical LiON model",
        ]
        if mask_path is None:
            message = "No segmentation was supplied; pixel and lesion quantification are unavailable"
            quality = QualityEvidence(
                status="unavailable",
                checks=[
                    QualityCheck(
                        check_id="LION_MASK_AVAILABLE",
                        status="unavailable",
                        message=message,
                    )
                ],
                warnings=[message],
            )
            return LionInspiredImagingEvidence(
                patient_id=patient_id,
                study_date=study_date,
                modality="CT",
                phase=phase,
                status="unavailable",
                pixel_evidence=LionPixelEvidence(
                    status="unavailable",
                    mask_available=False,
                    geometry_shape=[int(value) for value in volume.shape],
                    spacing_mm=[
                        round(float(value), 6)
                        for value in nib.affines.voxel_sizes(volume.affine)
                    ],
                ),
                lesion_evidence=[],
                patient_evidence=LionPatientEvidence(),
                quality=quality,
                limitations=[
                    *common_limitations,
                    "Missing segmentation must not be interpreted as absence of a lesion",
                ],
                provenance={
                    "adapter": "lion-inspired-precomputed-mask-v1",
                    "image_file": image.name,
                    "mask_supplied": False,
                },
                intended_use=LION_INTENDED_USE,
                sources=[
                    SourceReference(
                        source_id=image.name,
                        source_type="nifti_image",
                        uri=image.name,
                        deidentified=True,
                    )
                ],
            )

        mask = Path(mask_path)
        if not mask.is_file():
            raise FileNotFoundError(f"Segmentation mask not found: {mask}")
        if mask_role is None:
            raise ValueError("mask_role is required when a segmentation is supplied")
        measured = measure_nifti(
            image,
            mask,
            patient_id=patient_id,
            study_date=study_date,
            modality="CT",
            phase=phase,
            provider="lion-inspired-precomputed-mask",
            inference_mode="precomputed_mask",
        )
        retained = measured.quality.status in {"pass", "warning"} and bool(
            measured.lesions
        )
        lesions = [
            LionLesionEvidence(
                lesion_id=f"LESION_{index:03d}",
                voxel_count=item.voxel_count,
                volume_ml=item.volume_ml,
                max_3d_extent_mm=item.max_3d_extent_mm,
                centroid_world_mm=item.centroid_world_mm,
                bbox_voxel_ijk=item.bbox_voxel_ijk,
                source_measurement_id=item.lesion_id,
            )
            for index, item in enumerate(measured.lesions, 1)
        ]
        status = measured.quality.status if retained else "unavailable"
        limitations = [*common_limitations, *measured.interpretation_limits]
        if not retained:
            limitations.append(
                "No component passed validation; this is unavailable evidence, not a negative finding"
            )
        return LionInspiredImagingEvidence(
            patient_id=patient_id,
            study_date=study_date,
            modality="CT",
            phase=phase,
            status=status,
            pixel_evidence=LionPixelEvidence(
                status=status,
                mask_available=True,
                mask_role=mask_role,
                mask_sha256=_sha256(mask),
                image_mask_aligned=measured.quality.image_mask_aligned,
                geometry_shape=measured.geometry.shape,
                spacing_mm=measured.geometry.spacing_mm,
            ),
            lesion_evidence=lesions,
            patient_evidence=LionPatientEvidence(
                lesion_count=len(lesions) if retained else None,
                total_tumor_volume_ml=(
                    measured.total_tumor_volume_ml if retained else None
                ),
                max_lesion_extent_mm=(
                    measured.max_lesion_extent_mm if retained else None
                ),
            ),
            quality=measured.quality,
            limitations=sorted(set(limitations)),
            provenance={
                "adapter": "lion-inspired-precomputed-mask-v1",
                "image_file": image.name,
                "mask_file": mask.name,
                "mask_role": mask_role,
                "measurement_provider": measured.provider,
            },
            intended_use=LION_INTENDED_USE,
            sources=[
                SourceReference(
                    source_id=image.name,
                    source_type="nifti_image",
                    uri=image.name,
                    deidentified=True,
                ),
                SourceReference(
                    source_id=mask.name,
                    source_type="nifti_segmentation",
                    uri=mask.name,
                    deidentified=True,
                ),
            ],
        )


class PlanLionBackend:
    """Reserved integration point for a future public PLAN or licensed LiON backend."""

    def infer(self, **_: object) -> LionInspiredImagingEvidence:
        raise NotImplementedError(
            "PLAN/LiON inference is intentionally out of scope for the MVP; "
            "use PrecomputedMaskLionBackend"
        )


def report_imaging_payload(
    lion: LionInspiredImagingEvidence,
    *,
    glm_observations: list[str] | None = None,
) -> dict[str, object]:
    """Return the whitelisted shape accepted by the controlled report prompt."""
    patient = lion.patient_evidence
    return {
        "study_date": lion.study_date,
        "modality": lion.modality,
        "phase": lion.phase,
        "quality": lion.quality.to_dict(),
        "geometry": {
            "spacing_mm": lion.pixel_evidence.spacing_mm,
            "shape": lion.pixel_evidence.geometry_shape,
        },
        "lesion_count": patient.lesion_count,
        "total_tumor_volume_ml": patient.total_tumor_volume_ml,
        "max_lesion_extent_mm": patient.max_lesion_extent_mm,
        "interpretation_limits": lion.limitations,
        "glm_observations": list(glm_observations or []),
    }


__all__ = [
    "LION_INTENDED_USE",
    "LionBackend",
    "PlanLionBackend",
    "PrecomputedMaskLionBackend",
    "report_imaging_payload",
]
