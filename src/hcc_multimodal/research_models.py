from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .schemas import ArtifactModel, JsonModel, QualityEvidence


Disposition = Literal["included", "excluded", "failed", "indeterminate"]
ReviewLabel = Literal["progression", "no_progression", "indeterminate"]
EvaluationScope = Literal["development", "external_test"]


class ResearchProtocol(ArtifactModel):
    cohort_id: str = Field(min_length=1)
    external_test_center_ids: list[str] = Field(min_length=1)
    required_phase: Literal["portal_venous"] = "portal_venous"
    endpoint_name: Literal["radiographic_progression_within_180_days"] = (
        "radiographic_progression_within_180_days"
    )
    endpoint_window_days: Literal[180] = 180
    negative_followup_min_days: Literal[150] = 150
    negative_followup_max_days: Literal[210] = 210
    baseline_max_days_before_treatment: Literal[42] = 42
    lab_alignment_days: Literal[14] = 14
    primary_metric: Literal["auprc"] = "auprc"
    fixed_score_threshold: float = Field(default=0.5, ge=0, le=1)
    bootstrap_iterations: Literal[2000] = 2000
    bootstrap_seed: int = 1729
    minimum_inferential_stratum_size: Literal[20] = 20
    split_salt: str = Field(min_length=8)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)

    @field_validator("external_test_center_ids")
    @classmethod
    def unique_external_centers(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("external_test_center_ids must be unique")
        if any(not center.strip() for center in value):
            raise ValueError("external test center IDs must be non-empty")
        return value

    @field_validator("fixed_score_threshold")
    @classmethod
    def fixed_threshold_is_preregistered(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("fixed_score_threshold is preregistered at 0.5")
        return value


class StudyReference(JsonModel):
    study_id: str = Field(min_length=1)
    study_date: date
    phase: str
    ct_dir: str = Field(min_length=1)
    seg_file: str = Field(min_length=1)
    study_instance_uid: str = Field(min_length=1)
    series_instance_uid: str = Field(min_length=1)
    seg_series_instance_uid: str = Field(min_length=1)
    frame_of_reference_uid: str = Field(min_length=1)
    scanner_group: str = "unknown"
    registration_status: Literal[
        "not_applicable", "verified", "failed", "unavailable"
    ] = "not_applicable"


class IndexTreatment(JsonModel):
    treatment_date: date
    treatment_type: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class ResearchCase(JsonModel):
    patient_id: str = Field(min_length=1)
    center_id: str = Field(min_length=1)
    treatment: IndexTreatment
    baseline: StudyReference
    followups: list[StudyReference] = Field(min_length=1)
    labs_file: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_studies_and_temporal_order(self) -> "ResearchCase":
        studies = [self.baseline, *self.followups]
        ids = [study.study_id for study in studies]
        if len(ids) != len(set(ids)):
            raise ValueError("study_id values must be unique within a patient")
        if self.baseline.study_date >= self.treatment.treatment_date:
            raise ValueError("baseline study must occur before index treatment")
        if self.baseline.registration_status != "not_applicable":
            raise ValueError("baseline registration_status must be not_applicable")
        if any(study.study_date <= self.treatment.treatment_date for study in self.followups):
            raise ValueError("all follow-up studies must occur after index treatment")
        return self


class ResearchCohortManifest(ArtifactModel):
    cohort_id: str = Field(min_length=1)
    data_root: str = "."
    deidentified: Literal[True] = True
    legal_research_use_confirmed: Literal[True] = True
    cases: list[ResearchCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_patients(self) -> "ResearchCohortManifest":
        patient_ids = [case.patient_id for case in self.cases]
        duplicates = sorted(
            patient_id for patient_id, count in Counter(patient_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate patient IDs in manifest: {duplicates}")
        return self


class CaseDisposition(JsonModel):
    patient_id: str
    center_id: str
    disposition: Disposition
    reason_codes: list[str]
    messages: list[str]
    phi_findings: list[str] = Field(default_factory=list)


class CohortValidationReport(ArtifactModel):
    cohort_id: str
    protocol_hash: str
    manifest_hash: str
    quality: QualityEvidence
    counts: dict[str, int]
    cases: list[CaseDisposition]

    @model_validator(mode="after")
    def counts_cover_all_cases(self) -> "CohortValidationReport":
        if sum(self.counts.values()) != len(self.cases):
            raise ValueError("Disposition counts must cover every manifest case")
        return self


class FollowupRunRecord(JsonModel):
    study_id: str
    study_date: date
    days_after_treatment: int
    scanner_group: str
    disposition: Disposition
    reason_codes: list[str]
    image_only_score: float | None = Field(default=None, ge=0, le=1)
    lab_only_score: float | None = Field(default=None, ge=0, le=1)
    rule_fusion_score: float | None = Field(default=None, ge=0, le=1)
    imaging_category: str | None = None
    lab_signal_count: int | None = Field(default=None, ge=0, le=2)
    fusion_state: str | None = None
    missing_reasons: list[str] = Field(default_factory=list)
    artifact_directory: str | None = None


class ResearchCaseRun(JsonModel):
    patient_id: str
    center_id: str
    split: Literal["development", "external_test"]
    treatment_date: date
    treatment_type: str
    disposition: Disposition
    reason_codes: list[str]
    followups: list[FollowupRunRecord]


class ResearchRunManifest(ArtifactModel):
    cohort_id: str
    protocol_hash: str
    manifest_hash: str
    configuration_hash: str
    git_commit: str
    external_test_center_ids: list[str]
    label_data_loaded: Literal[False] = False
    cases: list[ResearchCaseRun]
    counts: dict[str, int]

    @model_validator(mode="after")
    def run_counts_cover_all_cases(self) -> "ResearchRunManifest":
        if sum(self.counts.values()) != len(self.cases):
            raise ValueError("Run disposition counts must cover every manifest case")
        return self


class AdjudicationReview(JsonModel):
    reviewer_id_hash: str = Field(min_length=8)
    label: ReviewLabel
    reason_codes: list[str] = Field(min_length=1)
    reviewed_at: datetime


class AdjudicationRecord(JsonModel):
    patient_id: str
    study_id: str
    reviewers: list[AdjudicationReview] = Field(min_length=2, max_length=2)
    adjudicator: AdjudicationReview | None = None
    final_label: ReviewLabel

    @model_validator(mode="after")
    def final_label_matches_review_process(self) -> "AdjudicationRecord":
        reviewer_ids = [review.reviewer_id_hash for review in self.reviewers]
        if len(set(reviewer_ids)) != 2:
            raise ValueError("Two distinct blinded reviewers are required")
        labels = {review.label for review in self.reviewers}
        if len(labels) == 1:
            agreed = self.reviewers[0].label
            if self.final_label != agreed:
                raise ValueError("final_label must equal the two-reviewer consensus")
            if self.adjudicator is not None:
                raise ValueError("adjudicator must be absent when blinded reviewers agree")
        else:
            if self.adjudicator is None:
                raise ValueError("Reviewer disagreement requires a third adjudicator")
            if self.adjudicator.reviewer_id_hash in reviewer_ids:
                raise ValueError("Adjudicator must be distinct from both blinded reviewers")
            if self.final_label != self.adjudicator.label:
                raise ValueError("final_label must equal the adjudicator label")
        return self


class AdjudicationSet(ArtifactModel):
    cohort_id: str
    scope: EvaluationScope
    records: list[AdjudicationRecord]

    @model_validator(mode="after")
    def unique_patient_studies(self) -> "AdjudicationSet":
        keys = [(record.patient_id, record.study_id) for record in self.records]
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate adjudication records: {duplicates}")
        return self


class AdjudicationValidationReport(ArtifactModel):
    cohort_id: str
    scope: EvaluationScope
    valid: bool
    record_count: int
    raw_agreement: float | None
    cohen_kappa: float | None
    adjudication_rate: float | None
    errors: list[str]
    warnings: list[str]


class PatientEndpointResult(JsonModel):
    patient_id: str
    center_id: str
    split: Literal["development", "external_test"]
    label: Literal[0, 1] | None
    endpoint_status: Literal["positive", "negative", "indeterminate"]
    selected_study_id: str | None
    selected_days_after_treatment: int | None
    reason_codes: list[str]
    image_only_score: float | None
    lab_only_score: float | None
    rule_fusion_score: float | None
    scanner_group: str
    treatment_type: str
    missing_pattern: str


class ExternalTestUnlockAudit(ArtifactModel):
    cohort_id: str
    protocol_hash: str
    manifest_hash: str
    configuration_hash: str
    git_commit: str
    external_test_center_ids: list[str]
    unlocked_at: datetime
    explicit_authorization: Literal[True] = True


class ResearchEvaluationArtifact(ArtifactModel):
    cohort_id: str
    scope: EvaluationScope
    primary_metric: Literal["auprc"] = "auprc"
    score_threshold: float = Field(default=0.5, ge=0, le=1)
    bootstrap_iterations: int = Field(default=2000, gt=0)
    protocol_hash: str
    manifest_hash: str
    configuration_hash: str
    git_commit: str
    quality: QualityEvidence
    adjudication: dict[str, Any]
    endpoint_counts: dict[str, int]
    case_dispositions: list[PatientEndpointResult]
    baselines: dict[str, Any]
    paired_differences: dict[str, Any]
    stratified_results: dict[str, Any]
    sensitivity_analyses: dict[str, Any]
    missingness: dict[str, Any]
    exploratory_only: bool
    intended_use: str

    @field_validator("score_threshold")
    @classmethod
    def score_threshold_is_preregistered(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("score_threshold is preregistered at 0.5")
        return value
