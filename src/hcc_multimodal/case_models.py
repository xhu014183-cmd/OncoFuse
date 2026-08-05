from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import (
    ArtifactModel,
    Comparator,
    JsonModel,
    QualityEvidence,
    QualityStatus,
)

ClinicalParseStatus = Literal[
    "exact",
    "censored",
    "missing",
    "invalid",
    "unsupported_unit",
    "ambiguous",
]
ClinicalLabGroup = Literal[
    "tumor_marker",
    "liver_reserve",
    "liver_injury",
    "viral_etiology",
]
TrajectoryState = Literal[
    "insufficient",
    "stable_normal",
    "stable_abnormal",
    "falling_after_treatment",
    "plateau_after_treatment",
    "rebound_after_nadir",
    "persistent_rising",
    "discordant_markers",
]
HpiEventType = Literal[
    "symptom",
    "imaging",
    "laboratory",
    "procedure",
    "treatment",
    "surgery",
    "hospitalization",
    "pathology_text",
    "followup",
    "other",
]
EvidenceConcordance = Literal[
    "concordant_response",
    "concordant_progression",
    "discordant_evidence",
    "stable_with_uncertainty",
    "insufficient_evidence",
]


class SourceSpan(JsonModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class StudyIdentity(JsonModel):
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str | None
    modality: Literal["CT", "MR"]
    study_date: str
    phase: str = "unknown"
    series_description: str | None = None
    sequence_name: str | None = None


class DicomGeometrySummary(JsonModel):
    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    slice_count: int = Field(gt=0)
    spacing_mm: list[float] = Field(min_length=3, max_length=3)
    orientation: list[float] = Field(min_length=6, max_length=6)
    slice_position_range_mm: list[float] = Field(min_length=2, max_length=2)


class ImageToolFinding(JsonModel):
    finding_id: str
    location: str
    observation: str
    confidence: Literal["high", "moderate", "low", "unavailable"] = "unavailable"
    source_ref: str
    reported_max_extent_mm: float | None = Field(default=None, gt=0)


class ImageToolEvidence(JsonModel):
    patient_id: str
    study_date: str
    modality: Literal["CT", "MR"]
    phase: str = "unknown"
    source_tool: str
    source_version: str
    findings: list[ImageToolFinding]
    reported_lesion_count: int | None = Field(default=None, ge=0)
    limitations: list[str] = Field(default_factory=list)


class QuantitativeLesion(JsonModel):
    lesion_id: str
    volume_ml: float = Field(gt=0)
    centroid_world_mm: list[float] = Field(min_length=3, max_length=3)
    max_3d_extent_mm: float = Field(gt=0)
    source: Literal["expert_seg", "supplied_seg"]


class ImagingInterpretationEvidence(ArtifactModel):
    patient_id: str
    study_date: str
    modality: Literal["CT", "MR"]
    phase: str
    interpretation_mode: Literal["quantitative_and_qualitative", "quantitative_only", "qualitative_only", "unavailable"]
    study_identity: StudyIdentity
    geometry: DicomGeometrySummary
    geometry_qc: QualityEvidence
    quantitative_measurements: list[QuantitativeLesion]
    qualitative_observations: list[ImageToolFinding]
    tool_consistency: Literal["pass", "warning", "fail", "not_comparable", "unavailable"]
    limitations: list[str]
    quality: QualityEvidence
    provenance: dict[str, Any]

    @model_validator(mode="after")
    def mode_matches_evidence(self) -> ImagingInterpretationEvidence:
        has_quantitative = bool(self.quantitative_measurements)
        has_qualitative = bool(self.qualitative_observations)
        expected = (
            "quantitative_and_qualitative"
            if has_quantitative and has_qualitative
            else "quantitative_only"
            if has_quantitative
            else "qualitative_only"
            if has_qualitative
            else "unavailable"
        )
        if self.interpretation_mode != expected:
            raise ValueError(f"interpretation_mode must be {expected!r} for the supplied evidence")
        return self


class ClinicalLabObservation(JsonModel):
    observed_at: str
    analyte: str
    group: ClinicalLabGroup
    value: float | None
    comparator: Comparator = "eq"
    unit: str
    original_unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    parse_status: ClinicalParseStatus
    source_text: str
    source_span: SourceSpan | None
    quality_status: QualityStatus
    above_reference: bool | None = None

    @model_validator(mode="after")
    def parsed_value_consistent(self) -> ClinicalLabObservation:
        if self.parse_status in {"exact", "censored"} and self.value is None:
            raise ValueError("usable observations require a numeric value")
        if self.parse_status not in {"exact", "censored"} and self.value is not None:
            raise ValueError("unusable observations must not expose a numeric value")
        if self.comparator != "eq" and self.parse_status != "censored":
            raise ValueError("non-equality comparators require censored status")
        return self


class ClinicalAnalyteEvidence(JsonModel):
    analyte: str
    group: ClinicalLabGroup
    observations: list[ClinicalLabObservation]
    latest_value: float | None
    latest_unit: str | None
    latest_above_reference: bool | None
    baseline_value: float | None
    nadir_value: float | None
    trajectory_state: TrajectoryState
    log_slope_per_day: float | None
    nadir_ratio: float | None
    rebound_ratio: float | None
    uncertainty: list[str] = Field(default_factory=list)


class LiverReserveEvidence(JsonModel):
    albi_score: float | None = None
    albi_grade: int | None = Field(default=None, ge=1, le=3)
    child_pugh_status: Literal["full", "partial", "unavailable"] = "unavailable"
    child_pugh_score: int | None = Field(default=None, ge=5, le=15)
    child_pugh_grade: Literal["A", "B", "C"] | None = None
    missing_components: list[str] = Field(default_factory=list)


class ClinicalLabEvidence(ArtifactModel):
    patient_id: str
    source_format: Literal["text", "json", "csv"]
    analytes: dict[str, ClinicalAnalyteEvidence]
    rejected_observations: list[ClinicalLabObservation]
    tumor_marker_evidence: list[str]
    liver_reserve_evidence: LiverReserveEvidence
    liver_injury_evidence: list[str]
    viral_etiology_evidence: list[str]
    confounders: list[str]
    missing_items: list[str]
    quality: QualityEvidence
    provenance: dict[str, Any]


class HpiEvent(JsonModel):
    event_date: str | None
    available_at: str | None
    event_type: HpiEventType
    text: str
    normalized_concept: str
    date_source: Literal["explicit", "relative_resolved", "relative_unresolved", "missing"]
    source_span: SourceSpan | None
    confidence: Literal["high", "moderate", "low"]
    source: str


class MarkerTrajectorySummary(JsonModel):
    analyte: str
    state: TrajectoryState
    baseline_value: float | None
    baseline_date: str | None
    nadir_value: float | None
    nadir_date: str | None
    latest_value: float | None
    latest_date: str | None
    log_slope_per_day: float | None
    nadir_ratio: float | None
    rebound_ratio: float | None


class HpiTimelineEvidence(ArtifactModel):
    patient_id: str
    index_date: str | None
    events: list[HpiEvent]
    treatment_dates: list[str]
    marker_trajectories: dict[str, MarkerTrajectorySummary]
    imaging_state: Literal[
        "single_timepoint",
        "stable",
        "response_signal",
        "progression_signal",
        "new_lesion_signal",
        "indeterminate",
        "unavailable",
    ]
    liver_function_state: Literal["stable", "improving", "worsening", "mixed", "insufficient"]
    overall_trend: EvidenceConcordance
    quality: QualityEvidence
    limitations: list[str]


class EvidenceLineSummary(JsonModel):
    status: Literal["available", "partial", "blocked", "unavailable"]
    headline: str
    findings: list[str]
    limitations: list[str]
    quality_status: QualityStatus


class CaseResearchSummary(ArtifactModel):
    patient_id: str
    case_status: Literal["complete", "partial", "blocked"]
    imaging_summary: EvidenceLineSummary
    laboratory_summary: EvidenceLineSummary
    timeline_summary: EvidenceLineSummary
    evidence_concordance: EvidenceConcordance
    key_findings: list[str]
    data_gaps: list[str]
    uncertainty: list[str]
    requires_human_review: bool
    quality: QualityEvidence
    research_disclaimer: Literal[
        "Research evidence summary only; not for diagnosis, staging, prognosis, or treatment decisions."
    ] = "Research evidence summary only; not for diagnosis, staging, prognosis, or treatment decisions."
