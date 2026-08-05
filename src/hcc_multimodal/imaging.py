from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Literal, cast

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .schemas import (
    ImageGeometry,
    ImagingEvidence,
    LesionEvidence,
    LesionMatchEvidence,
    LongitudinalImagingEvidence,
    MatchConfidence,
    MatchStatus,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    RegistrationEvidence,
    SourceReference,
)

DEFAULT_MATCHING_RULES_PATH = Path(__file__).with_name("configs") / "matching_rules.v1.yaml"


def load_matching_rules(path: str | Path | None = None) -> tuple[str, dict[str, float]]:
    source = Path(path) if path is not None else DEFAULT_MATCHING_RULES_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    required = {
        "version",
        "max_centroid_distance_mm",
        "accept_cost",
        "split_merge_distance_mm",
        "distance_weight",
        "overlap_weight",
        "volume_weight",
        "embedding_weight",
        "volume_increase_signal_pct",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Invalid matching rule configuration: {source}")
    version = str(payload["version"])
    values = {key: float(payload[key]) for key in required if key != "version"}
    return version, values


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _validate_pair(
    image: nib.Nifti1Image,
    mask: nib.Nifti1Image,
) -> tuple[list[QualityCheck], list[str]]:
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")
    if len(image.shape) != 3:
        raise ValueError(f"Only 3D NIfTI volumes are supported, got shape {image.shape}")
    if not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise ValueError("Image and mask affines do not match")
    if abs(float(np.linalg.det(image.affine[:3, :3]))) < 1e-8:
        raise ValueError("NIfTI affine is singular")

    spacing = np.asarray(nib.affines.voxel_sizes(image.affine), dtype=float)
    warnings: list[str] = []
    checks = [
        QualityCheck(
            check_id="IMAGE_MASK_GRID_MATCH",
            status="pass",
            message="Image and mask shape and affine match",
        ),
        QualityCheck(
            check_id="AFFINE_INVERTIBLE",
            status="pass",
            message="NIfTI affine is invertible",
        ),
    ]
    if np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ValueError(f"Invalid NIfTI voxel spacing: {spacing.tolist()}")
    if any(size > 5.0 for size in spacing):
        message = "At least one voxel dimension exceeds 5 mm; small lesions may be unstable"
        warnings.append(message)
        checks.append(QualityCheck(check_id="SPACING_RESOLUTION", status="warning", message=message))
    else:
        checks.append(
            QualityCheck(
                check_id="SPACING_RESOLUTION",
                status="pass",
                message="Voxel dimensions are at most 5 mm",
            )
        )
    return checks, warnings


def _bbox_world(
    affine: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> list[list[float]]:
    corners = np.asarray(
        list(product(*[(minimum[axis], maximum[axis]) for axis in range(3)])),
        dtype=float,
    )
    world = nib.affines.apply_affine(affine, corners)
    return [
        [_round(value) for value in world.min(axis=0)],
        [_round(value) for value in world.max(axis=0)],
    ]


def measure_nifti(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    patient_id: str,
    study_date: str,
    modality: str = "CT",
    phase: str = "unknown",
    provider: str = "precomputed-mask-adapter",
    inference_mode: str = "precomputed_mask",
    minimum_lesion_volume_ml: float = 0.01,
    frame_of_reference_uid: str | None = None,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    segment_series_instance_uid: str | None = None,
    segment_number: int | None = None,
    acquisition_id: str | None = None,
    source_quality_warnings: list[str] | None = None,
) -> ImagingEvidence:
    """Measure an aligned 3D image/mask pair in physical RAS coordinates."""
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    image = cast(nib.Nifti1Image, nib.load(str(image_path)))
    mask = cast(nib.Nifti1Image, nib.load(str(mask_path)))
    checks, warnings = _validate_pair(image, mask)
    for message in source_quality_warnings or []:
        warnings.append(message)
        checks.append(
            QualityCheck(check_id="SOURCE_GEOMETRY_QC", status="warning", message=message)
        )

    image_data = np.asarray(image.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask.dataobj) > 0
    if not np.isfinite(image_data).all():
        message = "Image contains non-finite values; lesion intensity ignores them"
        warnings.append(message)
        checks.append(QualityCheck(check_id="FINITE_IMAGE_VALUES", status="warning", message=message))
    else:
        checks.append(
            QualityCheck(check_id="FINITE_IMAGE_VALUES", status="pass", message="Image values are finite")
        )

    spacing = np.asarray(nib.affines.voxel_sizes(image.affine), dtype=np.float64)
    voxel_volume_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    labels, count = ndimage.label(mask_data, structure=structure)

    lesions: list[LesionEvidence] = []
    for component in range(1, count + 1):
        coordinates = np.argwhere(labels == component)
        voxel_count = int(coordinates.shape[0])
        volume_ml = voxel_count * voxel_volume_ml
        if volume_ml < minimum_lesion_volume_ml:
            warnings.append(
                f"Discarded component {component} below minimum volume "
                f"({volume_ml:.4f} mL < {minimum_lesion_volume_ml:.4f} mL)"
            )
            continue

        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        physical_extents = (maximum - minimum + 1) * spacing
        centroid_voxel = coordinates.mean(axis=0)
        centroid_world = nib.affines.apply_affine(image.affine, centroid_voxel)
        values = image_data[labels == component]
        finite_values = values[np.isfinite(values)]
        mean_intensity = float(finite_values.mean()) if finite_values.size else None
        lesions.append(
            LesionEvidence(
                lesion_id=f"component-{component}",
                voxel_count=voxel_count,
                volume_ml=_round(volume_ml),
                max_3d_extent_mm=_round(float(physical_extents.max())),
                centroid_world_mm=[_round(value) for value in centroid_world],
                bbox_voxel_ijk=[minimum.astype(int).tolist(), maximum.astype(int).tolist()],
                bbox_world_mm=_bbox_world(image.affine, minimum, maximum),
                mean_image_intensity=_round(mean_intensity) if mean_intensity is not None else None,
            )
        )

    lesions.sort(key=lambda lesion: lesion.volume_ml, reverse=True)
    lesions = [lesion.model_copy(update={"lesion_id": f"L{index}"}) for index, lesion in enumerate(lesions, 1)]
    if not lesions:
        message = "No tumor component passed the minimum-volume threshold"
        warnings.append(message)
        checks.append(
            QualityCheck(
                check_id="NONEMPTY_TUMOR_MASK",
                status="unavailable" if not mask_data.any() else "warning",
                message=message,
            )
        )
    else:
        checks.append(
            QualityCheck(
                check_id="NONEMPTY_TUMOR_MASK",
                status="pass",
                message=f"{len(lesions)} lesion components passed the volume threshold",
            )
        )

    geometry = ImageGeometry(
        coordinate_system="RAS",
        shape=[int(value) for value in image.shape],
        spacing_mm=[_round(value, 6) for value in spacing],
        affine=[[_round(value, 8) for value in row] for row in image.affine],
        orientation_codes=list(nib.aff2axcodes(image.affine)),
        frame_of_reference_uid=frame_of_reference_uid,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        segment_series_instance_uid=segment_series_instance_uid,
        segment_number=segment_number,
        acquisition_id=acquisition_id,
        phase=phase,
    )
    status: QualityStatus = (
        "unavailable" if not mask_data.any() else "warning" if warnings else "pass"
    )
    return ImagingEvidence(
        patient_id=patient_id,
        study_date=study_date,
        modality=modality.upper(),
        body_region="liver",
        phase=phase,
        provider=provider,
        inference_mode=inference_mode,
        geometry=geometry,
        quality=QualityEvidence(
            status=status,
            image_mask_aligned=True,
            spacing_mm=geometry.spacing_mm,
            checks=checks,
            warnings=warnings,
        ),
        lesion_count=len(lesions),
        total_tumor_volume_ml=_round(sum(lesion.volume_ml for lesion in lesions)),
        max_lesion_extent_mm=max((lesion.max_3d_extent_mm for lesion in lesions), default=None),
        lesions=lesions,
        provenance={"image_file": image_path.name, "tumor_mask_file": mask_path.name},
        sources=[
            SourceReference(source_id=image_path.name, source_type="nifti_image", uri=image_path.name),
            SourceReference(source_id=mask_path.name, source_type="nifti_segmentation", uri=mask_path.name),
        ],
        interpretation_limits=[
            "Measurements depend on the supplied segmentation mask",
            "Maximum extent is a 3D bounding-box metric, not RECIST or mRECIST diameter",
            "No LI-RADS, enhancement-pattern, pathology, diagnosis, or treatment inference is performed",
        ],
    )


def _bbox_iou(first: LesionEvidence, second: LesionEvidence) -> float:
    first_box = np.asarray(first.bbox_world_mm, dtype=float)
    second_box = np.asarray(second.bbox_world_mm, dtype=float)
    lower = np.maximum(first_box[0], second_box[0])
    upper = np.minimum(first_box[1], second_box[1])
    intersection = float(np.prod(np.maximum(upper - lower, 0.0)))
    first_volume = float(np.prod(np.maximum(first_box[1] - first_box[0], 0.0)))
    second_volume = float(np.prod(np.maximum(second_box[1] - second_box[0], 0.0)))
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0 else 0.0


def _cosine(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else None


def _registration_status(
    baseline: ImagingEvidence,
    followup: ImagingEvidence,
    declared: Literal["verified", "assumed_same_grid", "failed", "unavailable"] | None,
) -> Literal["verified", "assumed_same_grid", "failed", "unavailable"]:
    if declared is not None:
        return declared
    baseline_for = baseline.geometry.frame_of_reference_uid
    followup_for = followup.geometry.frame_of_reference_uid
    if baseline_for and followup_for:
        return "verified" if baseline_for == followup_for else "failed"
    if baseline.geometry.shape == followup.geometry.shape and np.allclose(
        baseline.geometry.affine,
        followup.geometry.affine,
        atol=1e-3,
    ):
        return "assumed_same_grid"
    return "unavailable"


def compare_imaging(
    baseline: ImagingEvidence,
    followup: ImagingEvidence,
    *,
    registration_status: Literal["verified", "assumed_same_grid", "failed", "unavailable"] | None = None,
    thresholds: dict[str, float] | None = None,
    rules_path: str | Path | None = None,
    baseline_embeddings: dict[str, np.ndarray] | None = None,
    followup_embeddings: dict[str, np.ndarray] | None = None,
    registration: RegistrationEvidence | None = None,
) -> LongitudinalImagingEvidence:
    """Globally match lesions after checking identity, time, geometry, and registration."""
    if baseline.patient_id != followup.patient_id:
        raise ValueError("Baseline and follow-up patient IDs do not match")
    if baseline.study_date >= followup.study_date:
        raise ValueError("Follow-up study date must occur after baseline study date")

    threshold_version, configured = load_matching_rules(rules_path)
    config = {**configured, **(thresholds or {})}
    resolved_registration = _registration_status(baseline, followup, registration_status)
    phase_compatible = (
        baseline.phase == "unknown"
        or followup.phase == "unknown"
        or baseline.phase == followup.phase
    )
    critical_failure = (
        baseline.quality.status in {"fail", "unavailable"}
        or followup.quality.status in {"fail", "unavailable"}
        or resolved_registration in {"failed", "unavailable"}
        or not phase_compatible
    )
    checks = [
        QualityCheck(
            check_id="SAME_PATIENT",
            status="pass",
            message="Baseline and follow-up patient identifiers match",
        ),
        QualityCheck(
            check_id="TEMPORAL_ORDER",
            status="pass",
            message="Follow-up occurs after baseline",
        ),
        QualityCheck(
            check_id="REGISTRATION",
            status=(
                "pass" if resolved_registration == "verified" else
                "warning" if resolved_registration == "assumed_same_grid" else "fail"
            ),
            message=f"Registration status: {resolved_registration}",
        ),
        QualityCheck(
            check_id="PHASE_COMPATIBILITY",
            status=(
                "pass"
                if baseline.phase == followup.phase and baseline.phase != "unknown"
                else "warning"
                if phase_compatible
                else "fail"
            ),
            message=f"Baseline phase={baseline.phase}; follow-up phase={followup.phase}",
        ),
    ]
    if critical_failure:
        error = "Quantitative longitudinal analysis blocked by imaging QC or registration status"
        return LongitudinalImagingEvidence(
            patient_id=baseline.patient_id,
            baseline_date=baseline.study_date,
            followup_date=followup.study_date,
            baseline_total_volume_ml=baseline.total_tumor_volume_ml,
            followup_total_volume_ml=followup.total_tumor_volume_ml,
            volume_change_pct=None,
            baseline_lesion_count=baseline.lesion_count,
            followup_lesion_count=followup.lesion_count,
            new_lesion_signal=None,
            matched_lesions=[],
            category="imaging_unavailable",
            quality=QualityEvidence(status="fail", checks=checks, errors=[error]),
            registration_status=resolved_registration,
            reason_codes=["LONGITUDINAL_QC_BLOCKED"],
            method="hungarian-physical-geometry",
            threshold_version=threshold_version,
            warnings=[error],
            registration=registration,
        )

    change_pct = None
    if baseline.total_tumor_volume_ml > 0:
        change_pct = _round(
            100.0
            * (followup.total_tumor_volume_ml - baseline.total_tumor_volume_ml)
            / baseline.total_tumor_volume_ml,
            1,
        )

    baseline_lesions = baseline.lesions
    followup_lesions = followup.lesions
    costs = np.full((len(baseline_lesions), len(followup_lesions)), np.inf, dtype=float)
    components: dict[tuple[int, int], tuple[float, float, float, float | None]] = {}
    max_distance = config["max_centroid_distance_mm"]
    for baseline_index, prior in enumerate(baseline_lesions):
        for followup_index, current in enumerate(followup_lesions):
            distance = float(
                np.linalg.norm(
                    np.asarray(current.centroid_world_mm) - np.asarray(prior.centroid_world_mm)
                )
            )
            overlap = _bbox_iou(prior, current)
            volume_cost = min(abs(math.log(max(current.volume_ml, 1e-8) / max(prior.volume_ml, 1e-8))), 3.0) / 3.0
            similarity = _cosine(
                (baseline_embeddings or {}).get(prior.lesion_id),
                (followup_embeddings or {}).get(current.lesion_id),
            )
            embedding_cost = 1.0 - similarity if similarity is not None else 0.0
            weight_sum = (
                config["distance_weight"]
                + config["overlap_weight"]
                + config["volume_weight"]
                + (config["embedding_weight"] if similarity is not None else 0.0)
            )
            total = (
                config["distance_weight"] * min(distance / max_distance, 1.5)
                + config["overlap_weight"] * (1.0 - overlap)
                + config["volume_weight"] * volume_cost
                + (config["embedding_weight"] * embedding_cost if similarity is not None else 0.0)
            ) / weight_sum
            costs[baseline_index, followup_index] = total
            components[(baseline_index, followup_index)] = (distance, overlap, volume_cost, similarity)

    accepted: dict[int, int] = {}
    if costs.size:
        row_indices, column_indices = linear_sum_assignment(costs)
        for baseline_index, followup_index in zip(row_indices, column_indices, strict=True):
            distance = components[(baseline_index, followup_index)][0]
            if distance <= max_distance and costs[baseline_index, followup_index] <= config["accept_cost"]:
                accepted[baseline_index] = followup_index

    matches: list[LesionMatchEvidence] = []
    matched_followup = set(accepted.values())
    for baseline_index, followup_index in accepted.items():
        prior = baseline_lesions[baseline_index]
        current = followup_lesions[followup_index]
        distance, overlap, _, similarity = components[(baseline_index, followup_index)]
        lesion_change = 100 * (current.volume_ml - prior.volume_ml) / prior.volume_ml
        total_cost = float(costs[baseline_index, followup_index])
        confidence: MatchConfidence = (
            "high" if total_cost <= 0.4 else "moderate" if total_cost <= 0.6 else "low"
        )
        matches.append(
            LesionMatchEvidence(
                status="matched",
                baseline_lesion_ids=[prior.lesion_id],
                followup_lesion_ids=[current.lesion_id],
                centroid_distance_mm=_round(distance),
                bbox_iou=_round(overlap, 4),
                volume_change_pct=_round(lesion_change, 1),
                embedding_cosine_similarity=_round(similarity, 4) if similarity is not None else None,
                total_cost=_round(total_cost, 4),
                confidence=confidence,
                reason_codes=["HUNGARIAN_GLOBAL_ASSIGNMENT"],
            )
        )

    split_merge_distance = config["split_merge_distance_mm"]
    for followup_index, current in enumerate(followup_lesions):
        if followup_index in matched_followup:
            continue
        nearby_baseline_indices = [
            index
            for index, lesion in enumerate(baseline_lesions)
            if np.linalg.norm(
                np.asarray(current.centroid_world_mm) - np.asarray(lesion.centroid_world_mm)
            ) <= split_merge_distance
        ]
        nearby_baseline = [baseline_lesions[index].lesion_id for index in nearby_baseline_indices]
        if any(index in accepted for index in nearby_baseline_indices):
            status: MatchStatus = "split_candidate"
            reason = "ONE_TO_MANY_PROXIMITY"
        elif nearby_baseline:
            status = "indeterminate"
            reason = "NEARBY_PAIR_REJECTED_BY_GLOBAL_COST"
        else:
            status = "new"
            reason = "NO_ACCEPTED_BASELINE_MATCH"
        matches.append(
            LesionMatchEvidence(
                status=status,
                baseline_lesion_ids=nearby_baseline,
                followup_lesion_ids=[current.lesion_id],
                confidence="low" if nearby_baseline else "moderate",
                reason_codes=[reason],
            )
        )
    for baseline_index, prior in enumerate(baseline_lesions):
        if baseline_index in accepted:
            continue
        nearby_followup_indices = [
            index
            for index, lesion in enumerate(followup_lesions)
            if np.linalg.norm(
                np.asarray(prior.centroid_world_mm) - np.asarray(lesion.centroid_world_mm)
            ) <= split_merge_distance
        ]
        nearby_followup = [followup_lesions[index].lesion_id for index in nearby_followup_indices]
        if any(index in matched_followup for index in nearby_followup_indices):
            status = "merge_candidate"
            reason = "MANY_TO_ONE_PROXIMITY"
        elif nearby_followup:
            status = "indeterminate"
            reason = "NEARBY_PAIR_REJECTED_BY_GLOBAL_COST"
        else:
            status = "disappeared"
            reason = "NO_ACCEPTED_FOLLOWUP_MATCH"
        matches.append(
            LesionMatchEvidence(
                status=status,
                baseline_lesion_ids=[prior.lesion_id],
                followup_lesion_ids=nearby_followup,
                confidence="low" if nearby_followup else "moderate",
                reason_codes=[reason],
            )
        )

    new_lesion_signal = any(item.status == "new" for item in matches)
    reasons: list[str] = []
    if new_lesion_signal:
        reasons.append("NEW_LESION_GEOMETRIC_SIGNAL")
    if any(item.status in {"split_candidate", "merge_candidate"} for item in matches):
        reasons.append("COMPLEX_PAIRING_CANDIDATE")
    if any(item.status == "indeterminate" for item in matches):
        reasons.append("PAIRING_INDETERMINATE")
    volume_threshold = config["volume_increase_signal_pct"]
    if change_pct is not None and change_pct >= volume_threshold:
        reasons.append("TOTAL_TUMOR_VOLUME_INCREASE")
    if change_pct is not None and change_pct <= -volume_threshold:
        reasons.append("TOTAL_TUMOR_VOLUME_DECREASE")

    if new_lesion_signal or (change_pct is not None and change_pct >= volume_threshold):
        category = "imaging_progression_signal"
    elif change_pct is not None and change_pct <= -volume_threshold:
        category = "imaging_response_signal"
    else:
        category = "imaging_stable_or_indeterminate"
    warnings = ["The category is a research heuristic and is not RECIST, mRECIST, or LI-RADS"]
    if resolved_registration == "assumed_same_grid":
        warnings.append("Registration was inferred from an identical NIfTI grid, not independently verified")
    quality_status: QualityStatus = "warning" if warnings else "pass"
    return LongitudinalImagingEvidence(
        patient_id=baseline.patient_id,
        baseline_date=baseline.study_date,
        followup_date=followup.study_date,
        baseline_total_volume_ml=baseline.total_tumor_volume_ml,
        followup_total_volume_ml=followup.total_tumor_volume_ml,
        volume_change_pct=change_pct,
        baseline_lesion_count=baseline.lesion_count,
        followup_lesion_count=followup.lesion_count,
        new_lesion_signal=new_lesion_signal,
        matched_lesions=matches,
        category=category,
        quality=QualityEvidence(status=quality_status, checks=checks, warnings=warnings),
        registration_status=resolved_registration,
        reason_codes=reasons,
        method="hungarian-centroid-bbox-volume" + ("-embedding" if config["embedding_weight"] else ""),
        threshold_version=threshold_version,
        warnings=warnings,
        registration=registration,
    )
