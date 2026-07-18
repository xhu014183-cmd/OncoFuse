from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    ClinicalVerdict,
    Concordance,
    ImagingEvidence,
    LabEvidence,
    LongitudinalImagingEvidence,
    QualityStatus,
    RuleTrace,
    RuleTraceStatus,
    SourceReference,
)


DEFAULT_RULES_PATH = Path(__file__).with_name("configs") / "fusion_rules.v1.yaml"


def load_fusion_rules(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_RULES_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("version"), str):
        raise ValueError(f"Invalid fusion rule configuration: {source}")
    return payload


def _latest_lab_date(labs: LabEvidence) -> date | None:
    values = [
        date.fromisoformat(observation.date)
        for marker in labs.markers.values()
        for observation in marker.observations
        if observation.parse_status in {"exact", "censored"}
    ]
    return max(values, default=None)


def _marker_progression_signals(labs: LabEvidence) -> list[str]:
    signals: list[str] = []
    for name in ("AFP", "DCP"):
        marker = labs.markers.get(name)
        if (
            marker
            and marker.quality_status in {"pass", "warning"}
            and marker.direction == "rising"
            and marker.latest_above_upper is True
        ):
            signals.append(name)
    return signals


def _trace(
    rule_id: str,
    status: RuleTraceStatus,
    evidence_refs: list[str],
    explanation: str,
    version: str,
) -> RuleTrace:
    return RuleTrace(
        rule_id=rule_id,
        status=status,
        evidence_refs=evidence_refs,
        explanation=explanation,
        threshold_version=version,
    )


def fuse_evidence(
    labs: LabEvidence,
    imaging: LongitudinalImagingEvidence,
    *,
    treatment_events: list[dict[str, Any]] | None = None,
    rules_path: str | Path | None = None,
) -> ClinicalVerdict:
    """Fuse validated evidence through versioned, fully traceable research rules."""
    if labs.patient_id != imaging.patient_id:
        raise ValueError(
            f"Lab patient ID {labs.patient_id!r} does not match imaging patient ID "
            f"{imaging.patient_id!r}"
        )
    rules = load_fusion_rules(rules_path)
    version = rules["version"]
    lab_ref = f"{labs.patient_id}:labs"
    imaging_ref = f"{imaging.patient_id}:imaging:{imaging.baseline_date}:{imaging.followup_date}"
    traces: list[RuleTrace] = []
    supporting: list[str] = []
    conflicting: list[str] = []
    missing: list[str] = []
    reasons = list(imaging.reason_codes)

    imaging_blocked = imaging.quality.status in set(rules["quality"]["blocking_statuses"])
    labs_blocked = labs.quality.status in set(rules["quality"]["blocking_statuses"])
    traces.append(
        _trace(
            "FUSION.QC.IMAGING",
            "blocked" if imaging_blocked else "not_fired",
            [imaging_ref],
            f"Imaging quality status is {imaging.quality.status}",
            version,
        )
    )
    traces.append(
        _trace(
            "FUSION.QC.LAB",
            "blocked" if labs_blocked else "not_fired",
            [lab_ref],
            f"Laboratory quality status is {labs.quality.status}",
            version,
        )
    )

    latest_lab = _latest_lab_date(labs)
    followup_date = date.fromisoformat(imaging.followup_date)
    max_age = int(rules["laboratory"]["max_evidence_age_days"])
    lab_age = (followup_date - latest_lab).days if latest_lab is not None else None
    stale_labs = lab_age is None or lab_age < 0 or lab_age > max_age
    traces.append(
        _trace(
            "FUSION.LAB.FRESHNESS",
            "blocked" if stale_labs else "not_fired",
            [lab_ref, imaging_ref],
            f"Latest laboratory evidence age is {lab_age!r} days; maximum is {max_age}",
            version,
        )
    )
    if stale_labs:
        missing.append("Laboratory evidence is unavailable or outside the configured freshness window")

    intervening_treatment = bool(treatment_events) and bool(
        rules["treatment"]["intervening_event_blocks_concordance"]
    )
    traces.append(
        _trace(
            "FUSION.TREATMENT.INTERVENING_EVENT",
            "blocked" if intervening_treatment else "not_fired",
            [imaging_ref],
            "An intervening treatment event prevents attribution of longitudinal change"
            if intervening_treatment
            else "No intervening treatment event was supplied",
            version,
        )
    )

    marker_signals = [] if labs_blocked or stale_labs else _marker_progression_signals(labs)
    for name in ("AFP", "DCP"):
        marker = labs.markers.get(name)
        fired = name in marker_signals
        traces.append(
            _trace(
                f"FUSION.LAB.{name}.RISING_ABOVE_REFERENCE",
                "fired" if fired else "not_fired",
                [f"{lab_ref}:{name}"],
                (
                    f"{name} direction={marker.direction}, above_reference={marker.latest_above_upper}"
                    if marker
                    else f"{name} is unavailable"
                ),
                version,
            )
        )
        if fired and marker:
            supporting.append(
                f"{name} is rising and above its supplied upper reference "
                f"({marker.latest_value:g} {marker.unit})"
            )
            reasons.append(f"{name}_RISING_ABOVE_REFERENCE")
        elif marker is None:
            missing.append(f"{name} evidence is unavailable")
        elif marker.direction in {"insufficient", "indeterminate"}:
            missing.append(f"{name} numeric trend is {marker.direction}")

    imaging_progression = not imaging_blocked and imaging.category == rules["imaging"]["progression_category"]
    imaging_response = not imaging_blocked and imaging.category == rules["imaging"]["response_category"]
    traces.append(
        _trace(
            "FUSION.IMAGING.PROGRESSION",
            "fired" if imaging_progression else "not_fired",
            [imaging_ref],
            f"Longitudinal imaging category is {imaging.category}",
            version,
        )
    )
    if imaging.new_lesion_signal:
        supporting.append("Imaging geometry produced a new-lesion signal")
    if imaging.volume_change_pct is not None:
        supporting.append(f"Total segmented tumor volume changed {imaging.volume_change_pct:+.1f}%")

    if imaging_blocked:
        state = "insufficient_evidence"
        concordance: Concordance = "insufficient"
        missing.append("Longitudinal imaging failed critical quality or registration checks")
        reasons.append("IMAGING_QC_BLOCKED")
    elif intervening_treatment:
        state = "indeterminate_after_intervening_treatment"
        concordance = "insufficient"
        missing.append("Treatment context prevents direct attribution of longitudinal evidence")
        reasons.append("INTERVENING_TREATMENT")
    elif imaging_progression and marker_signals:
        state = "concordant_progression_signal"
        concordance = "high"
    elif imaging_progression and not marker_signals and not labs_blocked and not stale_labs:
        state = "discordant_imaging_progression"
        concordance = "discordant"
        conflicting.append(
            "Imaging progression signal is not accompanied by a rising abnormal AFP/DCP signal"
        )
        reasons.append("IMAGING_LAB_DISCORDANCE")
    elif marker_signals and imaging_response:
        state = "discordant_marker_rise_imaging_response"
        concordance = "discordant"
        conflicting.append("AFP/DCP progression signal conflicts with the imaging response signal")
        reasons.append("IMAGING_LAB_DISCORDANCE")
    elif marker_signals:
        state = "discordant_marker_rise_without_imaging_progression"
        concordance = "discordant"
        conflicting.append("Rising abnormal AFP/DCP is not accompanied by imaging progression")
        reasons.append("IMAGING_LAB_DISCORDANCE")
    else:
        state = "insufficient_evidence"
        concordance = "insufficient"
        missing.append("No concordant positive signal is present and the evidence does not establish absence")
        reasons.append("INSUFFICIENT_CONCORDANT_EVIDENCE")

    missing.extend(
        [
            "Radiologist interpretation and enhancement-pattern assessment are not provided",
            "Clinical history, liver function, pathology, and complete treatment context are outside this rule",
        ]
    )
    quality_status: QualityStatus = "fail" if imaging_blocked else "warning"
    return ClinicalVerdict(
        patient_id=labs.patient_id,
        state=state,
        modality_concordance=concordance,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        missing_evidence=sorted(set(missing)),
        reason_codes=sorted(set(reasons)),
        rule_traces=traces,
        threshold_version=version,
        quality_status=quality_status,
        requires_clinician_review=True,
        intended_use="research prototype only; not for diagnosis, staging, prognosis, or treatment decisions",
        sources=[
            SourceReference(source_id=imaging_ref, source_type="longitudinal_imaging_evidence"),
            SourceReference(source_id=lab_ref, source_type="laboratory_evidence"),
        ],
    )


def fuse_cross_sectional_evidence(
    labs: LabEvidence,
    imaging: ImagingEvidence,
    *,
    rules_path: str | Path | None = None,
) -> ClinicalVerdict:
    """Fuse an explicitly unpaired public-image/synthetic-lab PoC without clinical claims."""
    if labs.patient_id != imaging.patient_id:
        raise ValueError(
            f"Lab patient ID {labs.patient_id!r} does not match imaging patient ID "
            f"{imaging.patient_id!r}"
        )
    rules = load_fusion_rules(rules_path)
    version = rules["version"]
    marker_signals = [] if labs.quality.status == "fail" else _marker_progression_signals(labs)
    has_segmented_mass = imaging.quality.status != "fail" and imaging.lesion_count > 0
    supporting: list[str] = []
    missing: list[str] = []
    reasons: list[str] = []
    traces: list[RuleTrace] = []

    if has_segmented_mass:
        supporting.append(
            f"Expert or supplied mask contains {imaging.lesion_count} retained region(s), "
            f"total segmented volume {imaging.total_tumor_volume_ml:g} mL"
        )
        reasons.append("SEGMENTED_MASS_PRESENT")
    traces.append(
        _trace(
            "FUSION.CROSS_SECTIONAL.SEGMENTED_REGION",
            "fired" if has_segmented_mass else "not_fired",
            [f"{imaging.patient_id}:imaging:{imaging.study_date}"],
            f"Retained lesion count is {imaging.lesion_count}",
            version,
        )
    )
    for name in marker_signals:
        marker = labs.markers[name]
        supporting.append(
            f"Declared synthetic {name} is rising above its supplied reference "
            f"({marker.latest_value:g} {marker.unit})"
        )
        reasons.append(f"SYNTHETIC_{name}_RISING_ABOVE_REFERENCE")
    for name in ("AFP", "DCP"):
        if name not in labs.markers:
            missing.append(f"{name} evidence is unavailable")

    if has_segmented_mass and marker_signals:
        state = "cross_sectional_lesion_marker_signal"
    elif has_segmented_mass:
        state = "segmented_lesion_without_marker_signal"
    elif marker_signals:
        state = "marker_signal_without_segmented_lesion"
    else:
        state = "insufficient_cross_sectional_evidence"
    missing.extend(
        [
            "Synthetic AFP/DCP values are not measurements from the public imaging subject",
            "No patient-level longitudinal pairing is available",
            "Formal radiologist interpretation and enhancement-pattern assessment are unavailable",
        ]
    )
    return ClinicalVerdict(
        patient_id=labs.patient_id,
        state=state,
        modality_concordance="illustrative_only",
        supporting_evidence=supporting,
        conflicting_evidence=[],
        missing_evidence=missing,
        reason_codes=sorted(set(reasons)),
        rule_traces=traces,
        threshold_version=version,
        quality_status="warning",
        requires_clinician_review=True,
        intended_use=(
            "research demonstration using public imaging plus unrelated synthetic labs; "
            "not patient-level multimodal evidence"
        ),
        sources=[
            SourceReference(
                source_id=f"{imaging.patient_id}:imaging:{imaging.study_date}",
                source_type="cross_sectional_imaging_evidence",
                data_origin="real_public",
            ),
            SourceReference(
                source_id=f"{labs.patient_id}:synthetic_labs",
                source_type="laboratory_evidence",
                data_origin="synthetic",
            ),
        ],
    )
