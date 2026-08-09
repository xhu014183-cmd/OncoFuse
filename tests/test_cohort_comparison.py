from pathlib import Path

import pytest

from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.cohort_comparison import (
    CASE_LAB_SCENARIOS,
    _case_dir,
    _scenario_text,
)


def test_scenario_text_anchors_dates_to_study_date():
    text = _scenario_text(CASE_LAB_SCENARIOS["dual_marker_rising"], "1998-12-29")

    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("1998-09-30 AFP 6 ng/mL 0-7")
    assert lines[1].startswith("1998-12-29 AFP 85.3 ng/mL 0-7")


def test_rebound_discordant_has_three_timepoints():
    text = _scenario_text(CASE_LAB_SCENARIOS["rebound_discordant"], "1997-09-12")

    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 6
    assert any(line.startswith("1997-06-14 AFP 320 ng/mL") for line in lines)
    assert any(line.startswith("1997-07-29 AFP 6.5 ng/mL") for line in lines)
    assert any(line.startswith("1997-09-12 AFP 96 ng/mL") for line in lines)
    assert any(line.startswith("1997-09-12 DCP 30 mAU/mL") for line in lines)


def test_case_dir_resolution(tmp_path: Path):
    cohort = tmp_path / "cohort"
    hcc003 = tmp_path / "hcc003"
    (cohort / "HCC_004").mkdir(parents=True)
    (hcc003).mkdir(parents=True)
    (cohort / "HCC_004" / "public_imaging_evidence.json").write_text(
        "{}", encoding="utf-8"
    )
    (hcc003 / "public_imaging_evidence.json").write_text("{}", encoding="utf-8")

    assert _case_dir("HCC_004", cohort, hcc003) == cohort / "HCC_004"
    assert _case_dir("HCC_003", cohort, hcc003) == hcc003


def test_case_dir_raises_when_missing():
    with pytest.raises(ValueError, match="missing public_imaging_evidence"):
        _case_dir("HCC_999", Path("cohort"), Path("hcc003"))


def test_1990s_lab_dates_are_parsed(tmp_path: Path):
    text = _scenario_text(CASE_LAB_SCENARIOS["dual_marker_rising"], "1997-09-12")
    path = tmp_path / "labs.txt"
    path.write_text(text, encoding="utf-8")

    labs = parse_laboratory_report(path, patient_id="HCC_003")

    afp = labs.analytes["AFP"]
    assert [obs.observed_at for obs in afp.observations] == [
        "1997-06-14",
        "1997-09-12",
    ]
