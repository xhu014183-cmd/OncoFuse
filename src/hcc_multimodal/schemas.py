from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import json
import math

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


def _generated_at() -> datetime:
    return datetime.now(timezone.utc)


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
    def status_matches_messages(self) -> "QualityEvidence":
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
    def lesion_summary_consistent(self) -> "ImagingEvidence":
        if self.lesion_count != len(self.lesions):
            raise ValueError("lesion_count must equal the number of lesion records")
        return self


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
    def parsed_value_consistent(self) -> "MarkerObservation":
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
    def validate_artifact(self) -> "ImageEmbeddingEvidence":
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
    def validate_vector(self) -> "LabFeatureVector":
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
