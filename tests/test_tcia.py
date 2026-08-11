from pathlib import Path
from types import SimpleNamespace

import pytest

from hcc_multimodal.tcia import _infer_phase, _seg_referenced_series_uids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SEG_DIR = (
    PROJECT_ROOT
    / "public-data"
    / "HCC_003"
    / "public-case"
    / "raw"
    / "HCC_003"
    / "SEG_1.3.6.1.4.1.14519.5.2.1.1706.8374.106355502486885782622426045632"
)
EXPECTED_CT_SERIES = (
    "1.3.6.1.4.1.14519.5.2.1.1706.8374.281650679207816520863173918688"
)


def test_seg_references_its_source_ct_series():
    if not PUBLIC_SEG_DIR.exists():
        pytest.skip("public HCC_003 SEG data is not present in the workspace")
    seg_files = list(PUBLIC_SEG_DIR.glob("*.dcm"))
    assert seg_files, "SEG directory exists but contains no DICOM files"

    uids = _seg_referenced_series_uids(seg_files[0])

    assert EXPECTED_CT_SERIES in uids


def test_infer_phase_portal_venous():
    datasets = {
        "a": SimpleNamespace(
            SeriesDescription="LIVER PORTAL VENOUS PHASE", ProtocolName=None
        )
    }
    assert _infer_phase(datasets) == "portal_venous"


def test_infer_phase_non_contrast_from_protocol():
    datasets = {
        "a": SimpleNamespace(SeriesDescription=None, ProtocolName="PRE LIVER")
    }
    assert _infer_phase(datasets) == "non_contrast"


def test_infer_phase_unknown_for_ambiguous_description():
    datasets = {
        "a": SimpleNamespace(
            SeriesDescription="Recon 2: LIVER 3 PHASE (AP)", ProtocolName=None
        )
    }
    assert _infer_phase(datasets) == "unknown"
