from __future__ import annotations

from datetime import date
import math

from .schemas import (
    EvidenceMetadata,
    ImageEmbeddingEvidence,
    ImagingEvidence,
    LabEvidence,
    LabFeatureVector,
    MultimodalCaseEvidence,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
)


DATA_ORIGINS = ("real_clinical", "real_public", "synthetic", "user_supplied", "unknown")
PAIRING_STATUSES = ("same_subject", "unpaired_poc_composite", "user_supplied_unverified")
MARKERS = ("AFP", "DCP")
FEATURE_SUFFIXES = (
    "latest_uln_ratio",
    "change_uln_ratio",
    "slope_uln_per_30d",
    "observation_count",
    "latest_age_days",
    "trend_rising",
    "missing",
)
LAB_FEATURE_NAMES = tuple(
    f"{marker}.{suffix}" for marker in MARKERS for suffix in FEATURE_SUFFIXES
)


def _iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def evidence_age_days(observed_at: str | None, index_time: str) -> int | None:
    index = _iso_date(index_time, "index_time")
    if observed_at is None:
        return None
    observed = _iso_date(observed_at, "observed_at")
    age = (index - observed).days
    if age < 0:
        raise ValueError(
            f"Evidence observed at {observed_at} occurs after index time {index_time}"
        )
    return age


def _append_feature(
    names: list[str],
    values: list[float],
    availability: list[bool],
    name: str,
    value: float | int | None,
) -> None:
    names.append(name)
    numeric = float(value) if value is not None else None
    available = numeric is not None and math.isfinite(numeric)
    values.append(numeric if available and numeric is not None else 0.0)
    availability.append(available)


def build_lab_feature_vector(labs: LabEvidence, *, index_time: str) -> LabFeatureVector:
    """Build fixed-order AFP/DCP features without treating missing values as normal."""
    index = _iso_date(index_time, "index_time")
    names: list[str] = []
    values: list[float] = []
    availability: list[bool] = []
    marker_availability: dict[str, bool] = {}
    warnings = list(labs.warnings)

    for name in MARKERS:
        marker = labs.markers.get(name)
        marker_availability[name] = marker is not None
        if marker is None:
            for suffix in FEATURE_SUFFIXES[:-1]:
                _append_feature(names, values, availability, f"{name}.{suffix}", None)
            _append_feature(names, values, availability, f"{name}.missing", 1.0)
            continue

        dated = []
        for observation in marker.observations:
            if observation.parse_status != "exact":
                continue
            observed = _iso_date(observation.date, f"{name} observation date")
            if observed > index:
                raise ValueError(
                    f"{name} observation at {observation.date} occurs after index time {index_time}"
                )
            dated.append((observed, observation))
        dated.sort(key=lambda item: item[0])
        if not dated:
            for suffix in FEATURE_SUFFIXES[:-1]:
                _append_feature(names, values, availability, f"{name}.{suffix}", None)
            _append_feature(names, values, availability, f"{name}.missing", 1.0)
            warnings.append(f"{name} has no exact values eligible for numeric features")
            continue
        first_date, first = dated[0]
        latest_date, latest = dated[-1]

        latest_ratio = None
        if latest.upper_reference is not None and latest.upper_reference > 0:
            assert latest.value is not None
            latest_ratio = latest.value / latest.upper_reference
        first_ratio = None
        if first.upper_reference is not None and first.upper_reference > 0:
            assert first.value is not None
            first_ratio = first.value / first.upper_reference
        change_ratio = (
            latest_ratio - first_ratio
            if latest_ratio is not None and first_ratio is not None and len(dated) >= 2
            else None
        )
        interval_days = (latest_date - first_date).days
        slope = (
            change_ratio * 30.0 / interval_days
            if change_ratio is not None and interval_days > 0
            else None
        )
        latest_age = (index - latest_date).days
        trend_rising = 1.0 if marker.direction == "rising" else (
            0.0 if marker.direction in {"stable", "falling"} else None
        )

        feature_values = (
            latest_ratio,
            change_ratio,
            slope,
            len(dated),
            latest_age,
            trend_rising,
            0.0,
        )
        for suffix, value in zip(FEATURE_SUFFIXES, feature_values, strict=True):
            _append_feature(names, values, availability, f"{name}.{suffix}", value)

        if latest_ratio is None:
            warnings.append(f"{name} upper reference is unavailable; ULN-normalized features are masked")
        if len(dated) < 2 or interval_days <= 0:
            warnings.append(f"{name} requires two distinct dates for change and slope features")

    if tuple(names) != LAB_FEATURE_NAMES:
        raise RuntimeError("Internal lab feature ordering changed unexpectedly")
    return LabFeatureVector(
        source_evidence_id=f"{labs.patient_id}:labs:{index_time}",
        patient_id=labs.patient_id,
        index_time=index_time,
        feature_names=names,
        values=values,
        availability_mask=availability,
        marker_availability=marker_availability,
        normalization=(
            "marker values normalized to the supplied upper reference; slope is ULN ratio "
            "change per 30 days; unavailable values are zero placeholders with mask=false"
        ),
        quality=QualityEvidence(
            status="warning" if warnings else "pass",
            checks=[
                QualityCheck(
                    check_id="LAB_FEATURE_FINITE",
                    status="pass",
                    message="All feature slots are finite and missing values are explicitly masked",
                )
            ],
            warnings=sorted(set(warnings)),
        ),
        warnings=sorted(set(warnings)),
        sources=[
            SourceReference(
                source_id=f"{labs.patient_id}:labs:{index_time}",
                source_type="lab_evidence",
                data_origin=str(labs.provenance.get("data_origin", "unknown")),
            )
        ],
    )


def _latest_lab_date(labs: LabEvidence) -> str | None:
    dates = [
        observation.date
        for marker in labs.markers.values()
        for observation in marker.observations
        if observation.parse_status in {"exact", "censored"}
    ]
    return max(dates, default=None)


def _metadata(
    *,
    evidence_id: str,
    observed_at: str | None,
    index_time: str,
    available: bool,
    quality_status: QualityStatus,
    data_origin: str,
    source_system: str,
) -> EvidenceMetadata:
    if data_origin not in DATA_ORIGINS:
        raise ValueError(f"Unknown data_origin {data_origin!r}; expected one of {DATA_ORIGINS}")
    return EvidenceMetadata(
        evidence_id=evidence_id,
        observed_at=observed_at,
        index_time=index_time,
        evidence_age_days=evidence_age_days(observed_at, index_time),
        available=available,
        quality_status=quality_status,
        data_origin=data_origin,
        source_system=source_system,
    )


def build_multimodal_case_evidence(
    *,
    imaging: ImagingEvidence,
    labs: LabEvidence,
    index_time: str,
    pairing_status: str,
    data_relationship: str,
    imaging_origin: str,
    lab_origin: str,
    image_embedding: ImageEmbeddingEvidence | None = None,
) -> MultimodalCaseEvidence:
    """Align modality envelopes without changing the deterministic fusion verdict."""
    _iso_date(index_time, "index_time")
    if imaging.patient_id != labs.patient_id:
        raise ValueError(
            f"Imaging patient ID {imaging.patient_id!r} does not match lab patient ID "
            f"{labs.patient_id!r}"
        )
    if pairing_status not in PAIRING_STATUSES:
        raise ValueError(
            f"Unknown pairing_status {pairing_status!r}; expected one of {PAIRING_STATUSES}"
        )
    declared_origin = labs.provenance.get("data_origin")
    if declared_origin and declared_origin != lab_origin:
        raise ValueError(
            f"Lab origin {lab_origin!r} conflicts with declared origin {declared_origin!r}"
        )
    declared_pairing = labs.provenance.get("pairing_status")
    if declared_pairing and declared_pairing != pairing_status:
        raise ValueError(
            f"Pairing status {pairing_status!r} conflicts with declared status "
            f"{declared_pairing!r}"
        )

    imaging_evidence_id = f"{imaging.patient_id}:imaging:{imaging.study_date}"
    if image_embedding is not None:
        if image_embedding.patient_id != imaging.patient_id:
            raise ValueError("Image embedding patient ID does not match imaging evidence")
        if image_embedding.study_date != imaging.study_date:
            raise ValueError("Image embedding study date does not match imaging evidence")
        if image_embedding.source_evidence_id != imaging_evidence_id:
            raise ValueError("Image embedding source evidence ID does not match imaging evidence")

    lab_features = build_lab_feature_vector(labs, index_time=index_time)
    lab_observed_at = _latest_lab_date(labs)
    modalities = {
        "imaging": _metadata(
            evidence_id=imaging_evidence_id,
            observed_at=imaging.study_date,
            index_time=index_time,
            available=imaging.quality.status not in {"fail", "unavailable"},
            quality_status=imaging.quality.status,
            data_origin=imaging_origin,
            source_system=imaging.provider,
        ),
        "laboratory": _metadata(
            evidence_id=lab_features.source_evidence_id,
            observed_at=lab_observed_at,
            index_time=index_time,
            available=bool(labs.markers),
            quality_status=labs.quality.status,
            data_origin=lab_origin,
            source_system=labs.provenance.get("adapter", "unknown"),
        ),
    }
    warnings = list(lab_features.warnings)
    if pairing_status == "unpaired_poc_composite":
        warnings.append(
            "The modalities are an unpaired PoC composite and are not patient-level multimodal evidence"
        )
    if image_embedding is None:
        warnings.append("No learned image embedding was requested; deterministic evidence remains available")

    case_status: QualityStatus = (
        "fail"
        if any(item.quality_status == "fail" for item in modalities.values())
        else "unavailable"
        if any(item.quality_status == "unavailable" for item in modalities.values())
        else "warning"
        if warnings or any(item.quality_status == "warning" for item in modalities.values())
        else "pass"
    )
    return MultimodalCaseEvidence(
        patient_id=imaging.patient_id,
        index_time=index_time,
        pairing_status=pairing_status,
        data_relationship=data_relationship,
        modalities=modalities,
        lab_features=lab_features,
        image_embedding=image_embedding,
        quality=QualityEvidence(
            status=case_status,
            checks=[
                QualityCheck(
                    check_id="MODALITY_IDENTITY_ALIGNMENT",
                    status="pass",
                    message="Imaging and laboratory patient identifiers match",
                ),
                QualityCheck(
                    check_id="PAIRING_STATUS",
                    status="pass" if pairing_status == "same_subject" else "warning",
                    message=f"Declared pairing status: {pairing_status}",
                ),
            ],
            warnings=sorted(set(warnings)),
            errors=(
                ["At least one required modality failed or is unavailable after quality validation"]
                if case_status in {"fail", "unavailable"}
                else []
            ),
        ),
        warnings=sorted(set(warnings)),
        sources=[
            SourceReference(
                source_id=imaging_evidence_id,
                source_type="imaging_evidence",
                data_origin=imaging_origin,
            ),
            SourceReference(
                source_id=lab_features.source_evidence_id,
                source_type="laboratory_evidence",
                data_origin=lab_origin,
            ),
        ],
        intended_use=(
            "research integration contract only; not a trained fusion prediction and not for "
            "diagnosis or treatment decisions"
        ),
    )
