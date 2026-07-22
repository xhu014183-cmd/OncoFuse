"""HCC multimodal evidence demo."""

from .contracts import build_lab_feature_vector, build_multimodal_case_evidence
from .case_summary import summarize_case
from .case_llm import render_with_optional_llm, validate_case_rewrite
from .clinical_labs import parse_laboratory_report
from .fusion import fuse_evidence
from .hpi import parse_hpi_timeline
from .imaging import compare_imaging, measure_nifti
from .imaging_adapter import parse_imaging_study
from .labs import load_lab_evidence
from .volume_encoders import available_volume_encoders, encode_nifti_volume

__all__ = [
    "available_volume_encoders",
    "build_lab_feature_vector",
    "build_multimodal_case_evidence",
    "compare_imaging",
    "encode_nifti_volume",
    "fuse_evidence",
    "load_lab_evidence",
    "measure_nifti",
    "render_with_optional_llm",
    "parse_hpi_timeline",
    "parse_imaging_study",
    "parse_laboratory_report",
    "summarize_case",
    "validate_case_rewrite",
]
