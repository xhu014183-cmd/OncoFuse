"""Bridges the clinical lab parser into the controlled LiON-inspired fusion path."""

from __future__ import annotations

from typing import cast

from .case_models import ClinicalLabEvidence
from .schemas import (
    ClinicalVerdict,
    Comparator,
    Concordance,
    DataRelationship,
    GlmImagingEvidence,
    ImagingCrosscheckEvidence,
    LabEvidence,
    LionInspiredImagingEvidence,
    MarkerName,
    MarkerObservation,
    MarkerTrend,
    ParseStatus,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
    TrendDirection,
)


def _trend_direction(value: str) -> TrendDirection:
    if value in {"persistent_rising", "rebound_after_nadir"}:
        return "rising"
    if value == "falling_after_treatment":
        return "falling"
    if value in {"stable_normal", "stable_abnormal", "plateau_after_treatment"}:
        return "stable"
    if value == "discordant_markers":
        return "indeterminate"
    return "insufficient"


def clinical_labs_to_report_labs(labs: ClinicalLabEvidence) -> LabEvidence:
    """Convert the richer clinical parser artifact into the legacy report contract."""
    markers: dict[str, MarkerTrend] = {}
    warnings = list(labs.quality.warnings)
    rejected: list[MarkerObservation] = []
    for name_value in ("AFP", "DCP"):
        name = cast(MarkerName, name_value)
        analyte = labs.analytes.get(name)
        if analyte is None:
            warnings.append(f"{name} is missing or has no usable observations")
            continue
        observations: list[MarkerObservation] = []
        for item in analyte.observations:
            mapped_status: ParseStatus = cast(
                ParseStatus,
                item.parse_status
                if item.parse_status in {"exact", "censored", "missing", "invalid", "unsupported_unit"}
                else "invalid",
            )
            observation = MarkerObservation(
                date=item.observed_at,
                marker=name,
                value=item.value if mapped_status in {"exact", "censored"} else None,
                comparator=cast(Comparator, item.comparator),
                unit=item.unit,
                original_unit=item.original_unit,
                reference_low=item.reference_low,
                reference_high=item.reference_high,
                upper_reference=item.reference_high,
                parse_status=mapped_status,
                source_text=item.source_text,
                quality_status=item.quality_status,
            )
            if mapped_status in {"exact", "censored"}:
                observations.append(observation)
            else:
                rejected.append(observation)
        observations.sort(key=lambda value: value.date)
        latest = observations[-1] if observations else None
        baseline = next(
            (value for value in observations if value.parse_status == "exact"),
            None,
        )
        exact_count = sum(value.parse_status == "exact" for value in observations)
        censored_count = sum(value.parse_status == "censored" for value in observations)
        change_pct = None
        if (
            baseline is not None
            and latest is not None
            and baseline.value is not None
            and baseline.value != 0
            and latest.value is not None
        ):
            change_pct = round(
                100.0 * (latest.value - baseline.value) / abs(baseline.value), 1
            )
        direction = _trend_direction(analyte.trajectory_state)
        latest_above = analyte.latest_above_reference if latest is not None else None
        marker_status: QualityStatus = (
            "unavailable"
            if latest is None
            else "warning"
            if direction in {"insufficient", "indeterminate"} or analyte.uncertainty
            else "pass"
        )
        markers[name] = MarkerTrend(
            marker=name,
            observations=observations,
            latest_value=latest.value if latest else None,
            latest_comparator=latest.comparator if latest else None,
            unit=latest.unit if latest else None,
            upper_reference=latest.reference_high if latest else None,
            latest_above_upper=bool(latest_above) if latest_above is not None else None,
            direction=direction,
            change_pct=change_pct,
            interval_days=None,
            exact_observation_count=exact_count,
            censored_observation_count=censored_count,
            quality_status=marker_status,
            uncertainty=analyte.uncertainty,
        )
        warnings.extend(analyte.uncertainty)
    status: QualityStatus = (
        "fail"
        if labs.quality.status == "fail"
        else "warning"
        if warnings or len(markers) < 2
        else "pass"
    )
    return LabEvidence(
        patient_id=labs.patient_id,
        markers=markers,
        rejected_observations=rejected,
        quality=QualityEvidence(
            status=status,
            checks=[
                QualityCheck(
                    check_id="CLINICAL_TO_REPORT_LABS",
                    status=status,
                    message="Clinical laboratory evidence mapped to AFP/DCP report contract",
                )
            ],
            warnings=sorted(set(warnings)),
            errors=list(labs.quality.errors),
        ),
        warnings=sorted(set(warnings)),
        provenance={
            **labs.provenance,
            "adapter": "clinical-labs-to-report-labs-v1",
        },
        sources=labs.sources,
    )


def fuse_lion_multimodal_evidence(
    *,
    lion: LionInspiredImagingEvidence,
    glm: GlmImagingEvidence,
    crosscheck: ImagingCrosscheckEvidence,
    labs: LabEvidence,
    relationship: DataRelationship,
    treatment_dates: list[str] | None = None,
) -> ClinicalVerdict:
    """Create a conservative, deterministic verdict for the controlled renderer."""
    if len({lion.patient_id, glm.patient_id, crosscheck.patient_id, labs.patient_id}) != 1:
        raise ValueError("All evidence artifacts must share one patient pseudonym")
    supporting = list(crosscheck.supporting_evidence)
    conflicting = list(crosscheck.conflicting_evidence)
    missing = list(crosscheck.missing_evidence)
    reasons: list[str] = ["SINGLE_PHASE_LIMIT"]
    if treatment_dates:
        supporting.append(
            "Supplied HPI timeline contains treatment anchor date(s): "
            + ", ".join(sorted(set(treatment_dates)))
        )
        reasons.append("HPI_TREATMENT_ANCHOR_AVAILABLE")
    else:
        missing.append("No dated treatment anchor is available from the supplied HPI")

    lesion_count = lion.patient_evidence.lesion_count
    has_regions = lesion_count is not None and lesion_count > 0
    if lion.status in {"pass", "warning"} and has_regions:
        total_volume = lion.patient_evidence.total_tumor_volume_ml
        supporting.append(
            f"Supplied mask contains {lesion_count} retained region(s), total segmented volume "
            + (f"{total_volume:g} mL" if total_volume is not None else "unavailable")
        )
        reasons.append("LION_QUANT_AVAILABLE")
    else:
        missing.append("LiON-inspired supplied-mask quantification is unavailable")
        reasons.append("LION_MASK_UNAVAILABLE")

    if glm.status in {"pass", "warning"}:
        reasons.append("GLM_QUAL_AVAILABLE")
        supporting.extend(
            f"GLM {item.finding_id}: {item.location} — {item.observation}"
            for item in glm.observations
        )
        missing.extend(glm.uncertainties)
        missing.extend(glm.missing_information)
    else:
        reasons.append("GLM_BLOCKED")
        missing.extend(glm.missing_information)

    marker_signals: list[str] = []
    for name in ("AFP", "DCP"):
        marker = labs.markers.get(name)
        if marker is None:
            missing.append(f"{name} evidence is unavailable")
            continue
        if marker.direction == "rising" and marker.latest_above_upper is True:
            marker_signals.append(name)
            supporting.append(
                f"{name} is rising and above its supplied upper reference "
                f"({marker.latest_value:g} {marker.unit})"
            )
        elif marker.direction in {"insufficient", "indeterminate"}:
            missing.append(f"{name} numeric trend is {marker.direction}")

    missing.extend(
        [
            "Single-phase CT cannot establish a complete dynamic enhancement pattern",
            "Formal radiologist interpretation and pathology are not supplied by this pipeline",
        ]
    )
    if lion.phase == "unknown":
        missing.append("CT acquisition phase is unknown")
    if crosscheck.status in {"warning", "fail", "not_comparable"}:
        reasons.append("IMAGING_CROSSCHECK_WARNING")

    if relationship.pairing_status == "unpaired_poc_composite":
        state = "unpaired_poc_composite"
        concordance: Concordance = "illustrative_only"
        reasons.append("UNPAIRED_POC_COMPOSITE")
        missing.extend(
            [
                "Imaging and laboratory evidence are not measurements from the same subject",
                "No patient-level multimodal interpretation is permitted",
            ]
        )
    elif relationship.pairing_status == "user_supplied_unverified":
        state = "unverified_evidence_pairing"
        concordance = "insufficient"
        reasons.append("PAIRING_UNVERIFIED")
        missing.append("Same-subject pairing has not been verified")
    elif has_regions and marker_signals:
        state = "concordant_research_signal"
        concordance = "high"
    elif has_regions:
        state = "segmented_region_without_rising_marker_signal"
        concordance = "mixed"
        conflicting.append(
            "Supplied-mask lesion regions are not accompanied by a rising abnormal AFP/DCP signal"
        )
    elif marker_signals:
        state = "marker_signal_with_unavailable_quantitative_imaging"
        concordance = "insufficient"
        missing.append("Quantitative lesion evidence is unavailable for marker comparison")
    else:
        state = "insufficient_evidence"
        concordance = "insufficient"

    quality_status: QualityStatus = (
        "fail"
        if crosscheck.status == "fail" or labs.quality.status == "fail"
        else "warning"
    )
    return ClinicalVerdict(
        patient_id=lion.patient_id,
        state=state,
        modality_concordance=concordance,
        supporting_evidence=sorted(set(supporting)),
        conflicting_evidence=sorted(set(conflicting)),
        missing_evidence=sorted(set(missing)),
        reason_codes=sorted(set(reasons)),
        rule_traces=[],
        threshold_version="lion-inspired-fusion-v1",
        quality_status=quality_status,
        requires_clinician_review=True,
        intended_use=(
            relationship.statement
            + "; research evidence summary only; not for diagnosis, staging, prognosis, or treatment decisions"
        ),
        sources=[
            SourceReference(
                source_id="lion-inspired-evidence.json",
                source_type="lion_inspired_imaging_evidence",
                data_origin=relationship.imaging_origin,
            ),
            SourceReference(
                source_id="glm-imaging-evidence.json",
                source_type="glm_imaging_evidence",
                data_origin=relationship.imaging_origin,
            ),
            SourceReference(
                source_id="clinical-lab-evidence.json",
                source_type="clinical_laboratory_evidence",
                data_origin=relationship.laboratory_origin,
            ),
        ],
    )


__all__ = ["clinical_labs_to_report_labs", "fuse_lion_multimodal_evidence"]
