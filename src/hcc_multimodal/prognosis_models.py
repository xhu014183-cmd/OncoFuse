"""Strict contracts for the public HCC overall-survival research track."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .schemas import ArtifactModel, JsonModel, QualityEvidence, SourceReference

DatasetId = Literal["waw_tace", "hcc_tace_seg"]
ModelName = Literal["clinical_core", "imaging_core", "fused_core", "fused_extended"]


class PublicPrognosisRecord(JsonModel):
    """Baseline-only feature record. Outcome fields are deliberately absent."""

    patient_id: str = Field(min_length=8)
    cohort_id: DatasetId
    dataset_version: str = Field(min_length=1)
    source_patient_sha256: str = Field(min_length=64, max_length=64)
    cancer_type: Literal["HCC"] = "HCC"
    index_treatment: Literal["TACE"] = "TACE"
    baseline_phase: str = "annotation_native_phase"
    seg_role: Literal["public_reference"] = "public_reference"
    geometry_qc: Literal["pass", "warning"]
    age_years: float = Field(gt=0, le=120)
    female: int = Field(ge=0, le=1)
    afp_ng_ml: float = Field(ge=0)
    lesion_count: int = Field(ge=1)
    total_tumor_volume_ml: float = Field(gt=0)
    max_lesion_extent_mm: float = Field(gt=0)
    largest_lesion_sphericity: float = Field(gt=0, le=1.05)
    albumin_g_dl: float | None = Field(default=None, gt=0)
    bilirubin_mg_dl: float | None = Field(default=None, ge=0)
    inr: float | None = Field(default=None, gt=0)
    alt_iu_l: float | None = Field(default=None, ge=0)
    creatinine_mg_dl: float | None = Field(default=None, ge=0)
    portal_mean_hu: float | None = None
    portal_std_hu: float | None = Field(default=None, ge=0)
    portal_p10_hu: float | None = None
    portal_p90_hu: float | None = None
    missing_items: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)

    @field_validator("source_patient_sha256")
    @classmethod
    def source_hash_is_hex(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError("source_patient_sha256 must be lowercase SHA-256 hex")
        return value.lower()

    @model_validator(mode="after")
    def portal_statistics_are_atomic(self) -> PublicPrognosisRecord:
        values = (
            self.portal_mean_hu,
            self.portal_std_hu,
            self.portal_p10_hu,
            self.portal_p90_hu,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("Portal intensity statistics must be supplied together")
        return self


class SurvivalEndpointRecord(JsonModel):
    """Outcome-only record stored separately from baseline features."""

    patient_id: str = Field(min_length=8)
    cohort_id: DatasetId
    duration_days: float = Field(gt=0)
    event: int = Field(ge=0, le=1)
    source_ref: SourceReference


class CohortExclusion(JsonModel):
    cohort_id: DatasetId
    source_patient_sha256: str = Field(min_length=64, max_length=64)
    patient_id: str | None = None
    reason_codes: list[str] = Field(min_length=1)
    messages: list[str] = Field(min_length=1)


class PublicPrognosisCohortArtifact(ArtifactModel):
    feature_records: list[PublicPrognosisRecord]
    endpoint_records: list[SurvivalEndpointRecord]
    exclusions: list[CohortExclusion]
    counts: dict[str, int]
    quality: QualityEvidence

    @model_validator(mode="after")
    def features_and_endpoints_match(self) -> PublicPrognosisCohortArtifact:
        feature_ids = [record.patient_id for record in self.feature_records]
        endpoint_ids = [record.patient_id for record in self.endpoint_records]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("Feature patient IDs must be unique")
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("Endpoint patient IDs must be unique")
        if set(feature_ids) != set(endpoint_ids):
            raise ValueError("Feature and endpoint files must contain identical patient IDs")
        return self


class CoxModelBundle(ArtifactModel):
    model_id: str = Field(min_length=1)
    model_name: ModelName
    endpoint: Literal["overall_survival_after_first_tace"] = (
        "overall_survival_after_first_tace"
    )
    development_cohort: Literal["waw_tace"] = "waw_tace"
    external_validation_cohort: Literal["hcc_tace_seg"] | None = "hcc_tace_seg"
    feature_names: list[str] = Field(min_length=1)
    transformations: dict[str, Literal["identity", "log1p"]]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    penalizer: float = Field(ge=0)
    baseline_event_times_days: list[float]
    baseline_cumulative_hazard: list[float]
    development_reference_risks: list[float]
    median_risk_threshold: float
    training_n: int = Field(gt=0)
    training_event_n: int = Field(gt=0)
    cross_validation_seed: int
    model_hash: str = Field(min_length=64, max_length=64)
    external_validation_status: Literal["not_opened", "evaluated"] = "not_opened"
    limitations: list[str]

    @model_validator(mode="after")
    def vector_shapes_match(self) -> CoxModelBundle:
        size = len(self.feature_names)
        if not all(len(values) == size for values in (self.means, self.scales, self.coefficients)):
            raise ValueError("Model vectors must match feature_names")
        if set(self.transformations) != set(self.feature_names):
            raise ValueError("Every feature requires exactly one transformation")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("Model scales must be positive")
        if len(self.baseline_event_times_days) != len(self.baseline_cumulative_hazard):
            raise ValueError("Baseline hazard arrays must have equal length")
        if any(
            later < earlier
            for earlier, later in zip(
                self.baseline_cumulative_hazard,
                self.baseline_cumulative_hazard[1:],
            )
        ):
            raise ValueError("Baseline cumulative hazard must be monotonic")
        return self


class PrognosticEvidence(ArtifactModel):
    case_id: str
    patient_id: str
    status: Literal["pass", "warning", "unavailable"]
    model_id: str
    model_hash: str = Field(min_length=64, max_length=64)
    endpoint: Literal["overall_survival_after_first_tace"]
    applicability: Literal["applicable", "limited", "not_applicable"]
    risk_index: float | None = None
    development_percentile: float | None = Field(default=None, ge=0, le=100)
    research_risk_group: Literal["lower_than_development_median", "at_or_above_development_median"] | None = None
    external_validation_status: Literal["not_opened", "evaluated"]
    feature_values: dict[str, float]
    supporting_evidence: list[str]
    missing_evidence: list[str]
    limitations: list[str]
    quality: QualityEvidence
    intended_use: str

    @model_validator(mode="after")
    def scores_match_status(self) -> PrognosticEvidence:
        values = (self.risk_index, self.development_percentile, self.research_risk_group)
        if self.status == "unavailable" and any(value is not None for value in values):
            raise ValueError("Unavailable prognostic evidence cannot expose a risk result")
        if self.status != "unavailable" and any(value is None for value in values):
            raise ValueError("Available prognostic evidence requires all risk results")
        return self


class ControlledPrognosisReport(JsonModel):
    case_id: str
    report_type: Literal["hcc_tace_os_research_prognosis"] = (
        "hcc_tace_os_research_prognosis"
    )
    data_scope: str
    imaging_summary: list[str]
    clinical_summary: list[str]
    prognostic_assessment: dict[str, Any]
    uncertainty: list[str]
    data_quality_status: Literal["pass", "warning", "unavailable"]
    review_required: Literal[True] = True
    intended_use: str
    disclaimer: Literal[
        "Research use only; not for clinical prognosis or treatment decisions."
    ] = "Research use only; not for clinical prognosis or treatment decisions."


__all__ = [
    "CohortExclusion",
    "ControlledPrognosisReport",
    "CoxModelBundle",
    "DatasetId",
    "ModelName",
    "PrognosticEvidence",
    "PublicPrognosisCohortArtifact",
    "PublicPrognosisRecord",
    "SurvivalEndpointRecord",
]
