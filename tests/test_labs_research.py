from __future__ import annotations

from pathlib import Path
import json

import pytest

from hcc_multimodal.labs import load_lab_evidence


def _load(tmp_path: Path, observations: list[dict], *, index_time: str = "2026-07-15"):
    path = tmp_path / "labs.json"
    path.write_text(
        json.dumps({"patient_id": "P1", "observations": observations}),
        encoding="utf-8",
    )
    return load_lab_evidence(path, index_time=index_time)


def test_scientific_notation_reference_range_and_unit_normalization(tmp_path: Path):
    labs = _load(
        tmp_path,
        [
            {
                "date": "2026-01-01",
                "marker": "AFP",
                "value": "5.2E+04",
                "unit": "ug/L",
                "referenceRange": "0-7",
            },
            {
                "date": "2026-07-15",
                "marker": "AFP",
                "value": "+6.0e4",
                "unit": "ng/mL",
                "referenceRange": "0-7",
            },
        ],
    )
    marker = labs.markers["AFP"]
    assert marker.observations[0].value == 52000
    assert marker.observations[0].reference_low == 0
    assert marker.observations[0].reference_high == 7
    assert marker.unit == "ng/mL"
    assert marker.direction == "stable"


def test_censored_negative_missing_and_unsupported_units_are_not_fabricated(tmp_path: Path):
    labs = _load(
        tmp_path,
        [
            {"date": "2026-01-01", "marker": "AFP", "value": "<15", "unit": "ng/mL"},
            {"date": "2026-02-01", "marker": "AFP", "value": "-0.25", "unit": "ng/mL"},
            {"date": "2026-03-01", "marker": "AFP", "value": "未检测", "unit": "ng/mL"},
            {"date": "2026-04-01", "marker": "DCP", "value": "20", "unit": "ng/mL"},
        ],
    )
    afp = labs.markers["AFP"]
    assert afp.observations[0].comparator == "lt"
    assert afp.observations[0].parse_status == "censored"
    assert afp.observations[1].value == pytest.approx(-0.25)
    statuses = {item.parse_status for item in labs.rejected_observations}
    assert statuses == {"missing", "unsupported_unit"}
    assert all(item.value is None for item in labs.rejected_observations)
    assert "DCP" not in labs.markers


def test_duplicate_timestamp_makes_trend_indeterminate(tmp_path: Path):
    labs = _load(
        tmp_path,
        [
            {"date": "2026-01-01", "marker": "DCP", "value": "20", "unit": "AU/L"},
            {"date": "2026-01-01", "marker": "DCP", "value": "22", "unit": "mAU/mL"},
            {"date": "2026-07-01", "marker": "DCP", "value": "60", "unit": "mAU/mL"},
        ],
    )
    marker = labs.markers["DCP"]
    assert marker.direction == "indeterminate"
    assert marker.unit == "mAU/mL"
    assert any("Duplicate exact observations" in item for item in marker.uncertainty)
    assert labs.quality.status == "warning"


def test_all_unusable_values_fail_lab_quality(tmp_path: Path):
    labs = _load(
        tmp_path,
        [{"date": "2026-01-01", "marker": "AFP", "value": "N/A", "unit": "ng/mL"}],
    )
    assert labs.quality.status == "fail"
    assert labs.quality.errors
    assert not labs.markers
