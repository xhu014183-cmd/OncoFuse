from __future__ import annotations

from hcc_multimodal.imaging_crosscheck import crosscheck_imaging_evidence
from hcc_multimodal.schemas import (
    GlmImagingEvidence,
    GlmImagingFinding,
    LionInspiredImagingEvidence,
    LionLesionEvidence,
    LionPatientEvidence,
    LionPixelEvidence,
    QualityCheck,
    QualityEvidence,
)


def _quality(status="pass"):
    return QualityEvidence(
        status=status,
        checks=[QualityCheck(check_id="TEST", status=status, message="test")],
    )


def _lion() -> LionInspiredImagingEvidence:
    lesion = LionLesionEvidence(
        lesion_id="LESION_001",
        voxel_count=100,
        volume_ml=0.1,
        max_3d_extent_mm=5.0,
        centroid_world_mm=[1.0, 2.0, 3.0],
        bbox_voxel_ijk=[[1, 1, 1], [5, 5, 5]],
        source_measurement_id="L1",
    )
    return LionInspiredImagingEvidence(
        patient_id="P1",
        study_date="2026-07-20",
        modality="CT",
        status="pass",
        pixel_evidence=LionPixelEvidence(
            status="pass",
            mask_available=True,
            mask_role="user_supplied",
            mask_sha256="a" * 64,
        ),
        lesion_evidence=[lesion],
        patient_evidence=LionPatientEvidence(
            lesion_count=1,
            total_tumor_volume_ml=0.1,
            max_lesion_extent_mm=5.0,
        ),
        quality=_quality(),
        limitations=[],
        provenance={},
        intended_use="research",
    )


def _glm(lesion_id: str | None) -> GlmImagingEvidence:
    return GlmImagingEvidence(
        patient_id="P1",
        study_date="2026-07-20",
        status="pass",
        provider="zhipu",
        model="test",
        prompt_version="test",
        observations=[
            GlmImagingFinding(
                finding_id="FINDING_001",
                lesion_id=lesion_id,
                location="right liver",
                observation="visible heterogeneous region",
                confidence="moderate",
                source_refs=["IMAGE_MONTAGE"],
            )
        ],
        uncertainties=[],
        missing_information=[],
        image_conditioning_statement="conditioned on supplied render",
        quality=_quality(),
        limitations=[],
        provenance={},
    )


def test_crosscheck_pass_means_attribution_not_independent_confidence():
    evidence = crosscheck_imaging_evidence(_lion(), _glm("LESION_001"))
    assert evidence.status == "pass"
    assert evidence.covered_lesion_ids == ["LESION_001"]
    assert "not independent modalities" in " ".join(evidence.limitations)


def test_crosscheck_unknown_lesion_fails():
    evidence = crosscheck_imaging_evidence(_lion(), _glm("LESION_999"))
    assert evidence.status == "fail"
    assert evidence.unknown_lesion_ids == ["LESION_999"]
