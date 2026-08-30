from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "0.4.0"

QualityStatus = Literal["pass", "warning", "fail", "unavailable"]
Comparator = Literal["eq", "lt", "le", "gt", "ge"]
ParseStatus = Literal["exact", "censored", "missing", "invalid", "unsupported_unit"]
MarkerName = Literal["AFP", "DCP"]
TrendDirection = Literal["rising", "falling", "stable", "insufficient", "indeterminate"]
MatchStatus = Literal[
    "matched", "new", "disappeared", "split_candidate", "merge_candidate", "indeterminate"
]
MatchConfidence = Literal["high", "moderate", "low", "unavailable"]
RuleTraceStatus = Literal["fired", "not_fired", "blocked"]
Concordance = Literal[
    "high", "mixed", "low", "discordant", "insufficient", "unavailable", "illustrative_only"
]
DataOrigin = Literal[
    "real_clinical",
    "real_public",
    "synthetic",
    "user_supplied",
    "unknown",
]
PairingStatus = Literal[
    "same_subject",
    "unpaired_poc_composite",
    "user_supplied_unverified",
]
SegRole = Literal[
    "expert_reference",
    "public_reference",
    "user_supplied",
    "model_prediction",
]


def _generated_at() -> datetime:
    return datetime.now(UTC)


class JsonModel(BaseModel):
    """Strict JSON model used at every public pipeline boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")


class SourceReference(JsonModel):
    source_id: str
    source_type: str
    uri: str | None = None
    data_origin: str = "unknown"
    deidentified: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactModel(JsonModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    pipeline_version: str = PIPELINE_VERSION
    generated_at: datetime = Field(default_factory=_generated_at)
    sources: list[SourceReference] = Field(default_factory=list)

    @field_validator("generated_at")
    @classmethod
    def generated_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @field_validator("pipeline_version")
    @classmethod
    def pipeline_version_must_be_semver(cls, value: str) -> str:
        parts = value.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("pipeline_version must use numeric MAJOR.MINOR.PATCH syntax")
        return value


class QualityCheck(JsonModel):
    check_id: str
    status: QualityStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class QualityEvidence(JsonModel):
    status: QualityStatus
    image_mask_aligned: bool | None = None
    spacing_mm: list[float] = Field(default_factory=list)
    checks: list[QualityCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_messages(self) -> QualityEvidence:
        if self.status == "fail" and not self.errors:
            raise ValueError("Failed quality evidence must include at least one error")
        return self


class ImageGeometry(JsonModel):
    coordinate_system: Literal["RAS", "LPS", "unknown"] = "unknown"
    voxel_axis_order: str = "i,j,k"
    shape: list[int]
    spacing_mm: list[float]
    affine: list[list[float]]
    orientation_codes: list[str] = Field(default_factory=list)
    frame_of_reference_uid: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    segment_series_instance_uid: str | None = None
    segment_number: int | None = None
    acquisition_id: str | None = None
    phase: str = "unknown"

    @field_validator("shape")
    @classmethod
    def three_dimensional_shape(cls, value: list[int]) -> list[int]:
        if len(value) != 3 or any(item <= 0 for item in value):
            raise ValueError("Image geometry shape must contain three positive dimensions")
        return value

    @field_validator("affine")
    @classmethod
    def four_by_four_affine(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("Image geometry affine must be 4x4")
        return value


class EvidenceMetadata(JsonModel):
    evidence_id: str
    observed_at: str | None
    index_time: str
    evidence_age_days: int | None
    available: bool
    quality_status: QualityStatus
    data_origin: str
    source_system: str


class LesionEvidence(JsonModel):
    lesion_id: str
    voxel_count: int = Field(ge=1)
    volume_ml: float = Field(gt=0)
    max_3d_extent_mm: float = Field(gt=0)
    centroid_world_mm: list[float]
    bbox_voxel_ijk: list[list[int]]
    bbox_world_mm: list[list[float]]
    mean_image_intensity: float | None


class ImagingEvidence(ArtifactModel):
    patient_id: str
    study_date: str
    modality: str
    body_region: str
    phase: str = "unknown"
    provider: str
    inference_mode: str
    geometry: ImageGeometry
    quality: QualityEvidence
    lesion_count: int = Field(ge=0)
    total_tumor_volume_ml: float = Field(ge=0)
    max_lesion_extent_mm: float | None
    lesions: list[LesionEvidence]
    provenance: dict[str, Any]
    interpretation_limits: list[str]

    @model_validator(mode="after")
    def lesion_summary_consistent(self) -> ImagingEvidence:
        if self.lesion_count != len(self.lesions):
            raise ValueError("lesion_count must equal the number of lesion records")
        return self


class DataRelationship(JsonModel):
    imaging_origin: DataOrigin
    laboratory_origin: DataOrigin
    pairing_status: PairingStatus
    statement: str


class LionPixelEvidence(JsonModel):
    status: QualityStatus
    mask_available: bool
    mask_role: SegRole | None = None
    mask_sha256: str | None = None
    image_mask_aligned: bool | None = None
    geometry_shape: list[int] = Field(default_factory=list)
    spacing_mm: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def mask_contract_is_consistent(self) -> LionPixelEvidence:
        if self.mask_available and self.mask_role is None:
            raise ValueError("Available pixel evidence requires mask_role")
        if self.mask_sha256 is not None and len(self.mask_sha256) != 64:
            raise ValueError("mask_sha256 must be a SHA-256 hex digest")
        return self


class LionLesionEvidence(JsonModel):
    lesion_id: str
    voxel_count: int = Field(ge=1)
    volume_ml: float = Field(gt=0)
    max_3d_extent_mm: float = Field(gt=0)
    centroid_world_mm: list[float] = Field(min_length=3, max_length=3)
    bbox_voxel_ijk: list[list[int]]
    source_measurement_id: str


class LionPatientEvidence(JsonModel):
    lesion_count: int | None = Field(default=None, ge=0)
    total_tumor_volume_ml: float | None = Field(default=None, ge=0)
    max_lesion_extent_mm: float | None = Field(default=None, gt=0)
    diagnostic_capability: Literal["not_available"] = "not_available"


class LionInspiredImagingEvidence(ArtifactModel):
    patient_id: str
    study_date: str
    modality: Literal["CT"]
    phase: str = "unknown"
    backend: Literal["precomputed_mask"] = "precomputed_mask"
    status: QualityStatus
    pixel_evidence: LionPixelEvidence
    lesion_evidence: list[LionLesionEvidence]
    patient_evidence: LionPatientEvidence
    quality: QualityEvidence
    limitations: list[str]
    provenance: dict[str, Any]
    intended_use: str

    @model_validator(mode="after")
    def hierarchy_is_consistent(self) -> LionInspiredImagingEvidence:
        count = self.patient_evidence.lesion_count
        if count is not None and count != len(self.lesion_evidence):
            raise ValueError("patient lesion_count must equal lesion_evidence length")
        if not self.pixel_evidence.mask_available and count is not None:
            raise ValueError("Unavailable mask evidence cannot expose a lesion count")
        return self


class GlmImagingFinding(JsonModel):
    finding_id: str
    lesion_id: str | None = None
    location: str
    observation: str
    confidence: Literal["high", "moderate", "low", "unavailable"]
    source_refs: list[str] = Field(min_length=1)


class GlmImagingEvidence(ArtifactModel):
    patient_id: str
    study_date: str
    phase: str = "unknown"
    status: QualityStatus
    provider: Literal["zhipu", "disabled", "unavailable"]
    model: str
    prompt_version: str
    observations: list[GlmImagingFinding]
    uncertainties: list[str]
    missing_information: list[str]
    image_conditioning_statement: str
    quality: QualityEvidence
    limitations: list[str]
    provenance: dict[str, Any]


class ImagingCrosscheckEvidence(ArtifactModel):
    patient_id: str
    study_date: str
    status: Literal["pass", "warning", "fail", "not_comparable", "unavailable"]
    lion_lesion_ids: list[str]
    glm_referenced_lesion_ids: list[str]
    covered_lesion_ids: list[str]
    uncovered_lesion_ids: list[str]
    unknown_lesion_ids: list[str]
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    missing_evidence: list[str]
    quality: QualityEvidence
    limitations: list[str]
    intended_use: str


class LesionMatchEvidence(JsonModel):
    status: MatchStatus
    baseline_lesion_ids: list[str] = Field(default_factory=list)
    followup_lesion_ids: list[str] = Field(default_factory=list)
    centroid_distance_mm: float | None = None
    bbox_iou: float | None = None
    volume_change_pct: float | None = None
    embedding_cosine_similarity: float | None = None
    total_cost: float | None = None
    confidence: MatchConfidence = "unavailable"
    reason_codes: list[str] = Field(default_factory=list)


class RegistrationEvidence(JsonModel):
    """Auditable record of a rigid baseline/follow-up registration attempt."""

    status: Literal["verified", "failed", "unavailable"]
    method: str
    transform_matrix: list[list[float]] | None = None
    translation_mm: list[float] | None = None
    rotation_deg: list[float] | None = None
    metric_value: float | None = None
    centroid_residual_mm: float | None = None
    coordinate_frame: str = "itk_lps"
    warnings: list[str] = Field(default_factory=list)


class LongitudinalImagingEvidence(ArtifactModel):
    patient_id: str
    baseline_date: str
    followup_date: str
    baseline_total_volume_ml: float
    followup_total_volume_ml: float
    volume_change_pct: float | None
    baseline_lesion_count: int
    followup_lesion_count: int
    new_lesion_signal: bool | None
    matched_lesions: list[LesionMatchEvidence]
    category: str
    quality: QualityEvidence
    registration_status: Literal["verified", "assumed_same_grid", "failed", "unavailable"]
    reason_codes: list[str]
    method: str
    threshold_version: str
    warnings: list[str]
    registration: RegistrationEvidence | None = None


class MarkerObservation(JsonModel):
    date: str
    marker: MarkerName
    value: float | None
    comparator: Comparator = "eq"
    unit: str
    original_unit: str = ""
    reference_low: float | None = None
    reference_high: float | None = None
    upper_reference: float | None = None
    parse_status: ParseStatus
    source_text: str
    quality_status: QualityStatus

    @model_validator(mode="after")
    def parsed_value_consistent(self) -> MarkerObservation:
        if self.parse_status in {"exact", "censored"} and self.value is None:
            raise ValueError("Parsed observations require a numeric value")
        if self.parse_status in {"missing", "invalid", "unsupported_unit"} and self.value is not None:
            raise ValueError("Unusable observations must not expose a numeric value")
        if self.comparator != "eq" and self.parse_status != "censored":
            raise ValueError("Non-equality comparators require parse_status='censored'")
        return self


class MarkerTrend(JsonModel):
    marker: MarkerName
    observations: list[MarkerObservation]
    latest_value: float | None
    latest_comparator: Comparator | None
    unit: str | None
    upper_reference: float | None
    latest_above_upper: bool | None
    direction: TrendDirection
    change_pct: float | None
    interval_days: int | None
    exact_observation_count: int
    censored_observation_count: int
    quality_status: QualityStatus
    uncertainty: list[str] = Field(default_factory=list)


class LabEvidence(ArtifactModel):
    patient_id: str
    markers: dict[str, MarkerTrend]
    rejected_observations: list[MarkerObservation] = Field(default_factory=list)
    quality: QualityEvidence
    warnings: list[str]
    provenance: dict[str, Any]


class RuleTrace(JsonModel):
    rule_id: str
    status: RuleTraceStatus
    evidence_refs: list[str]
    explanation: str
    threshold_version: str


class ClinicalVerdict(ArtifactModel):
    patient_id: str
    state: str
    modality_concordance: Concordance
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    missing_evidence: list[str]
    reason_codes: list[str]
    rule_traces: list[RuleTrace] = Field(default_factory=list)
    threshold_version: str = "fusion-v1"
    quality_status: QualityStatus = "warning"
    requires_clinician_review: bool
    intended_use: str


class ControlledAssessment(JsonModel):
    state: str
    concordance: str
    summary: str
    supporting_evidence: list[str]
    conflicting_evidence: list[str]
    missing_evidence: list[str]


class ControlledReport(JsonModel):
    case_id: str
    scenario_id: str
    report_type: str
    data_scope: str
    imaging_summary: list[str] = Field(max_length=2)
    laboratory_summary: list[str]
    multimodal_assessment: ControlledAssessment
    uncertainty: list[str]
    data_quality_status: QualityStatus
    quality_and_limits: list[str]
    review_required: bool
    intended_use: str
    disclaimer: Literal[
        "Research use only; not for diagnosis, staging, prognosis, or treatment decisions."
    ]


class DeepseekNarrativeEvidence(ArtifactModel):
    """Optional validated prose generated after the controlled report is locked."""

    case_id: str
    status: Literal["pass", "blocked", "unavailable"]
    provider: str
    model: str
    prompt_version: str = "deepseek-hcc-narrative-v1"
    narrative: str | None = None
    validation_errors: list[dict[str, str]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    intended_use: str = (
        "optional language assistance over locked research evidence; not a diagnostic conclusion"
    )

    @model_validator(mode="after")
    def narrative_matches_status(self) -> DeepseekNarrativeEvidence:
        if self.status == "pass" and not (self.narrative or "").strip():
            raise ValueError("A passing DeepSeek narrative must contain validated prose")
        if self.status != "pass" and self.narrative is not None:
            raise ValueError("Blocked or unavailable narrative evidence must not retain prose")
        return self


class VlmNumericCitation(JsonModel):
    """Deterministic per-number attribution attached at the extraction layer.

    The VLM writes evidence IDs (``[LAB_001]``) into its free text; the audit
    pipeline resolves every numeric token back to a supplied laboratory
    observation and records the anchor here. This is the extraction-layer
    source reference that makes diff highlighting attributable.
    """

    evidence_id: str
    observed_at: str | None = None
    analyte: str
    value: float
    comparator: Comparator = "eq"
    unit: str
    used_in: str


class VlmDemoReport(JsonModel):
    """Structured extraction target shared by both dual-mode VLM arms.

    ``numeric_citations`` is populated by the deterministic audit after model
    validation; the model itself only fills the free-text fields.
    """

    fusion_mode: Literal["auditable", "open"]
    imaging_observations: list[str]
    clinical_context_summary: list[str]
    evidence_concordance: str
    uncertainties: list[str]
    missing_information: list[str]
    image_conditioning_statement: str
    research_disclaimer: Literal[
        "Research evidence summary only; not for diagnosis, staging, prognosis, or treatment decisions."
    ]
    numeric_citations: list[VlmNumericCitation] = Field(default_factory=list)


class ImageEmbeddingEvidence(ArtifactModel):
    """Manifest for a separately stored image representation artifact."""

    source_evidence_id: str
    patient_id: str
    study_date: str
    phase: str = "unknown"
    encoder_name: str
    model_name: str
    model_revision: str
    embedding_dimension: int
    embedding_dtype: str
    normalized: bool
    l2_norm: float | None = None
    artifact_file: str
    artifact_sha256: str
    input_shape: list[int]
    input_spacing_mm: list[float]
    preprocessing: dict[str, Any]
    mask_usage: str
    available: bool
    quality: QualityEvidence
    warnings: list[str]
    intended_use: str

    @model_validator(mode="after")
    def validate_artifact(self) -> ImageEmbeddingEvidence:
        if self.available and self.embedding_dimension <= 0:
            raise ValueError("Available image embeddings must have a positive dimension")
        if not self.available and self.embedding_dimension != 0:
            raise ValueError("Unavailable image embeddings must have dimension zero")
        if self.available and not self.artifact_file:
            raise ValueError("Available image embeddings require an artifact file")
        if self.available and len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
        if self.l2_norm is not None and not math.isfinite(self.l2_norm):
            raise ValueError("Embedding L2 norm must be finite")
        return self


class LabFeatureVector(ArtifactModel):
    source_evidence_id: str
    patient_id: str
    index_time: str
    feature_names: list[str]
    values: list[float]
    availability_mask: list[bool]
    marker_availability: dict[str, bool]
    normalization: str
    quality: QualityEvidence
    warnings: list[str]

    @model_validator(mode="after")
    def validate_vector(self) -> LabFeatureVector:
        size = len(self.feature_names)
        if len(self.values) != size or len(self.availability_mask) != size:
            raise ValueError("Lab feature names, values, and availability mask must have equal lengths")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("Lab feature values must all be finite")
        return self


class MultimodalCaseEvidence(ArtifactModel):
    patient_id: str
    index_time: str
    pairing_status: str
    data_relationship: str
    modalities: dict[str, EvidenceMetadata]
    lab_features: LabFeatureVector
    image_embedding: ImageEmbeddingEvidence | None
    quality: QualityEvidence
    warnings: list[str]
    intended_use: str


def json_schema_for(model: type[JsonModel]) -> dict[str, Any]:
    """Return a stable JSON Schema representation for CLI export and fixtures."""
    return json.loads(json.dumps(model.model_json_schema(), sort_keys=True))
