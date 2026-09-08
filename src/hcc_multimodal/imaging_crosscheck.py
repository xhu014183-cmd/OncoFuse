"""Deterministic attribution cross-check for LiON-inspired and GLM evidence."""

from __future__ import annotations

from typing import Literal

from .schemas import (
    GlmImagingEvidence,
    ImagingCrosscheckEvidence,
    LionInspiredImagingEvidence,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
)


def crosscheck_imaging_evidence(
    lion: LionInspiredImagingEvidence,
    glm: GlmImagingEvidence,
) -> ImagingCrosscheckEvidence:
    if lion.patient_id != glm.patient_id or lion.study_date != glm.study_date:
        raise ValueError("LiON-inspired and GLM evidence identity/date must match")
    lion_ids = [item.lesion_id for item in lion.lesion_evidence]
    referenced = sorted(
        {
            item.lesion_id
            for item in glm.observations
            if item.lesion_id is not None
        }
    )
    lion_set = set(lion_ids)
    referenced_set = set(referenced)
    covered = sorted(lion_set & referenced_set)
    uncovered = sorted(lion_set - referenced_set)
    unknown = sorted(referenced_set - lion_set)
    supporting: list[str] = []
    conflicting: list[str] = []
    missing: list[str] = []

    lion_available = lion.status in {"pass", "warning"}
    glm_available = glm.status in {"pass", "warning"}
    if covered:
        supporting.append(
            f"GLM supplied image-attributable descriptions for {len(covered)} supplied-mask lesion region(s)"
        )
    if unknown:
        conflicting.append(
            "GLM referenced lesion identifiers absent from LiON-inspired evidence: "
            + ", ".join(unknown)
        )
    if uncovered:
        missing.append(
            "LiON-inspired lesion regions without a GLM description: "
            + ", ".join(uncovered)
        )
    if not lion_available:
        missing.append("LiON-inspired quantitative evidence is unavailable")
    if not glm_available:
        missing.append("Validated GLM qualitative image evidence is unavailable")

    if not lion_available and not glm_available:
        status: Literal["pass", "warning", "fail", "not_comparable", "unavailable"] = (
            "unavailable"
        )
        quality_status: QualityStatus = "unavailable"
    elif not lion_available or not glm_available:
        status = "not_comparable"
        quality_status = "warning"
    elif unknown:
        status = "fail"
        quality_status = "fail"
    elif uncovered or not glm.observations:
        status = "warning"
        quality_status = "warning"
    else:
        status = "pass"
        quality_status = "pass"
    errors = conflicting if quality_status == "fail" else []
    warnings = sorted(set(missing + ([] if quality_status == "fail" else conflicting)))
    return ImagingCrosscheckEvidence(
        patient_id=lion.patient_id,
        study_date=lion.study_date,
        status=status,
        lion_lesion_ids=lion_ids,
        glm_referenced_lesion_ids=referenced,
        covered_lesion_ids=covered,
        uncovered_lesion_ids=uncovered,
        unknown_lesion_ids=unknown,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        missing_evidence=missing,
        quality=QualityEvidence(
            status=quality_status,
            checks=[
                QualityCheck(
                    check_id="IMAGING_ATTRIBUTION_CROSSCHECK",
                    status=quality_status,
                    message=f"LiON-inspired/GLM attribution cross-check status: {status}",
                )
            ],
            warnings=warnings,
            errors=errors,
        ),
        limitations=[
            "LiON-inspired measurements and GLM observations derive from the same CT and are not independent modalities",
            "A pass status means attributable coverage only, not medical correctness or increased diagnostic confidence",
        ],
        intended_use="research attribution and quality control only",
        sources=[
            SourceReference(
                source_id="lion-inspired-evidence.json",
                source_type="lion_inspired_imaging_evidence",
            ),
            SourceReference(
                source_id="glm-imaging-evidence.json",
                source_type="glm_imaging_evidence",
            ),
        ],
    )


__all__ = ["crosscheck_imaging_evidence"]
