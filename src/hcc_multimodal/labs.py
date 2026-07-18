from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any
import json
import re

from .schemas import (
    Comparator,
    LabEvidence,
    MarkerName,
    MarkerObservation,
    MarkerTrend,
    ParseStatus,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
    TrendDirection,
)


MARKER_ALIASES: dict[MarkerName, tuple[str, ...]] = {
    "AFP": ("AFP", "甲胎蛋白", "ALPHA-FETOPROTEIN"),
    "DCP": ("DCP", "PIVKA", "PIVKA-II", "异常凝血酶原"),
}
CANONICAL_UNITS = {"AFP": "ng/mL", "DCP": "mAU/mL"}
UNIT_ALIASES = {
    "AFP": {
        "ng/ml": 1.0,
        "ng/milliliter": 1.0,
        "ug/l": 1.0,
        "μg/l": 1.0,
        "mcg/l": 1.0,
    },
    "DCP": {
        "mau/ml": 1.0,
        "mau/milliliter": 1.0,
        "au/l": 1.0,
    },
}
MISSING_EXPRESSIONS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not available",
    "未检",
    "未检测",
    "缺失",
}
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VALUE_RE = re.compile(rf"^\s*(?P<comparator><=|>=|<|>)?\s*(?P<number>{NUMBER_PATTERN})\s*$")
RANGE_RE = re.compile(rf"^\s*(?P<low>{NUMBER_PATTERN})\s*[-~～至]\s*(?P<high>{NUMBER_PATTERN})\s*$")


def _canonical_marker(value: object) -> MarkerName | None:
    normalized = str(value or "").strip().upper()
    for canonical, aliases in MARKER_ALIASES.items():
        if any(alias.upper() in normalized for alias in aliases):
            return canonical
    return None


def _normalize_unit(marker: str, value: object) -> tuple[str, float | None]:
    original = str(value or "").strip()
    normalized = original.casefold().replace(" ", "").replace("µ", "μ")
    factor = UNIT_ALIASES[marker].get(normalized)
    return CANONICAL_UNITS[marker], factor


def _parse_numeric(value: object) -> tuple[float | None, Comparator, ParseStatus]:
    source = str(value if value is not None else "").strip()
    if source.casefold() in MISSING_EXPRESSIONS:
        return None, "eq", "missing"
    match = VALUE_RE.fullmatch(source.replace(",", ""))
    if not match:
        return None, "eq", "invalid"
    comparators: dict[str | None, Comparator] = {
        None: "eq",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
    }
    comparator = comparators[match.group("comparator")]
    number = float(match.group("number"))
    return number, comparator, "exact" if comparator == "eq" else "censored"


def _parse_reference(value: object) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    source = str(value).strip().replace(",", "")
    if not source:
        return None, None
    range_match = RANGE_RE.fullmatch(source)
    if range_match:
        low = float(range_match.group("low"))
        high = float(range_match.group("high"))
        return (low, high) if low <= high else (high, low)
    value_match = VALUE_RE.fullmatch(source)
    if value_match:
        number = float(value_match.group("number"))
        comparator = value_match.group("comparator")
        if comparator in {"<", "<="}:
            return None, number
        if comparator in {">", ">="}:
            return number, None
        return None, number
    numbers = re.findall(NUMBER_PATTERN, source)
    if len(numbers) == 2:
        low, high = map(float, numbers)
        return (low, high) if low <= high else (high, low)
    return None, None


def _row_observation(row: dict[str, Any]) -> MarkerObservation | None:
    marker = _canonical_marker(
        row.get("marker") or row.get("systemTestItemName") or row.get("itemName")
    )
    if marker is None:
        return None
    raw_value = row.get("value", row.get("result"))
    value, comparator, parse_status = _parse_numeric(raw_value)
    original_unit = str(row.get("unit") or "").strip()
    unit, factor = _normalize_unit(marker, original_unit)
    reference_low, reference_high = _parse_reference(row.get("referenceRange"))
    if row.get("reference_low") is not None:
        parsed, _, status = _parse_numeric(row["reference_low"])
        reference_low = parsed if status == "exact" else None
    upper = row.get("reference_high", row.get("upper_reference"))
    if upper is not None:
        parsed, _, status = _parse_numeric(upper)
        reference_high = parsed if status == "exact" else None

    if parse_status in {"exact", "censored"} and factor is None:
        value = None
        parse_status = "unsupported_unit"
    elif value is not None and factor is not None:
        value *= factor
        reference_low = reference_low * factor if reference_low is not None else None
        reference_high = reference_high * factor if reference_high is not None else None

    quality_status: QualityStatus = "pass" if parse_status == "exact" else (
        "warning" if parse_status == "censored" else "fail"
    )
    observed_at = str(row.get("observed_at") or row.get("date") or row.get("inspectionDate") or "")[:10]
    return MarkerObservation(
        date=observed_at,
        marker=marker,
        value=value,
        comparator=comparator,
        unit=unit,
        original_unit=original_unit,
        reference_low=reference_low,
        reference_high=reference_high,
        upper_reference=reference_high,
        parse_status=parse_status,
        source_text=str(raw_value if raw_value is not None else ""),
        quality_status=quality_status,
    )


def _latest_above_upper(observation: MarkerObservation) -> bool | None:
    if observation.value is None or observation.reference_high is None:
        return None
    if observation.comparator == "eq":
        return observation.value > observation.reference_high
    if observation.comparator == "gt" and observation.value >= observation.reference_high:
        return True
    if observation.comparator == "ge" and observation.value > observation.reference_high:
        return True
    if observation.comparator in {"lt", "le"} and observation.value <= observation.reference_high:
        return False
    return None


def _trend(marker: MarkerName, observations: list[MarkerObservation]) -> MarkerTrend:
    ordered = sorted(observations, key=lambda item: item.date)
    usable = [item for item in ordered if item.parse_status in {"exact", "censored"}]
    exact = [item for item in usable if item.parse_status == "exact"]
    censored = [item for item in usable if item.parse_status == "censored"]
    latest = usable[-1] if usable else None
    uncertainty: list[str] = []
    direction: TrendDirection = "insufficient"
    change_pct = None
    interval_days = None

    duplicate_dates = sorted({item.date for item in exact if sum(o.date == item.date for o in exact) > 1})
    if duplicate_dates:
        direction = "indeterminate"
        uncertainty.append(f"Duplicate exact observations occur at: {', '.join(duplicate_dates)}")
    elif len(exact) >= 2:
        first = exact[0]
        last = exact[-1]
        interval_days = (date.fromisoformat(last.date) - date.fromisoformat(first.date)).days
        if interval_days <= 0:
            direction = "indeterminate"
            uncertainty.append("Trend requires observations on distinct dates")
        else:
            assert first.value is not None and last.value is not None
            first_value = float(first.value)
            latest_value = float(last.value)
            if first_value != 0:
                change_pct = round(100 * (latest_value - first_value) / abs(first_value), 1)
            reference = last.reference_high or first.reference_high or 1.0
            material_delta = max(reference * 0.1, 1.0)
            if latest_value >= first_value * 1.2 and latest_value - first_value >= material_delta:
                direction = "rising"
            elif latest_value <= first_value * 0.8 and first_value - latest_value >= material_delta:
                direction = "falling"
            else:
                direction = "stable"
            if interval_days > 365:
                uncertainty.append(f"Trend spans {interval_days} days; evidence may be stale")
    if censored:
        uncertainty.append("Censored values are retained but excluded from numeric trend estimation")
    if any(item.parse_status == "unsupported_unit" for item in ordered):
        uncertainty.append("At least one observation used an unsupported unit")

    if not usable:
        status: QualityStatus = "unavailable"
    elif direction in {"insufficient", "indeterminate"} or uncertainty:
        status = "warning"
    else:
        status = "pass"
    return MarkerTrend(
        marker=marker,
        observations=ordered,
        latest_value=latest.value if latest else None,
        latest_comparator=latest.comparator if latest else None,
        unit=latest.unit if latest else None,
        upper_reference=latest.reference_high if latest else None,
        latest_above_upper=_latest_above_upper(latest) if latest else None,
        direction=direction,
        change_pct=change_pct,
        interval_days=interval_days,
        exact_observation_count=len(exact),
        censored_observation_count=len(censored),
        quality_status=status,
        uncertainty=uncertainty,
    )


def _filter_to_index_time(
    observations: list[MarkerObservation],
    index_time: str | None,
) -> tuple[list[MarkerObservation], list[MarkerObservation], list[str]]:
    cutoff = None
    if index_time is not None:
        try:
            cutoff = date.fromisoformat(index_time)
        except ValueError as exc:
            raise ValueError(f"index_time must be an ISO date, got {index_time!r}") from exc
    kept: list[MarkerObservation] = []
    rejected: list[MarkerObservation] = []
    warnings: list[str] = []
    for observation in observations:
        try:
            observed = date.fromisoformat(observation.date)
        except ValueError:
            rejected.append(observation)
            warnings.append(
                f"Excluded {observation.marker} observation with invalid date {observation.date!r}"
            )
            continue
        if cutoff is not None and observed > cutoff:
            rejected.append(observation)
            warnings.append(
                f"Excluded future {observation.marker} observation at {observation.date} "
                f"after index time {index_time}"
            )
            continue
        if observation.parse_status in {"missing", "invalid", "unsupported_unit"}:
            rejected.append(observation)
            warnings.append(
                f"Excluded {observation.marker} value {observation.source_text!r}: "
                f"{observation.parse_status}"
            )
            continue
        kept.append(observation)
    return kept, rejected, warnings


def load_lab_evidence(path: str | Path, *, index_time: str | None = None) -> LabEvidence:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Laboratory input must be a JSON object")
    rows = payload.get("observations")
    if rows is None:
        rows = payload.get("testResultInfos", [])
    if not isinstance(rows, list):
        raise ValueError("Laboratory observations must be an array")
    parsed = [item for row in rows if isinstance(row, dict) and (item := _row_observation(row))]
    observations, rejected, warnings = _filter_to_index_time(parsed, index_time)

    grouped: dict[MarkerName, list[MarkerObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.marker].append(observation)
    markers: dict[str, MarkerTrend] = {
        name: _trend(name, values) for name, values in sorted(grouped.items())
    }
    for required in ("AFP", "DCP"):
        if required not in markers:
            warnings.append(f"{required} is missing or has no usable observations")
        elif markers[required].exact_observation_count < 2:
            warnings.append(f"{required} has fewer than two exact observations; trend is unavailable")
        if required in markers:
            warnings.extend(markers[required].uncertainty)

    patient_info = payload.get("patientInfo") or {}
    patient_id = str(payload.get("patient_id") or patient_info.get("patientId") or "UNKNOWN")
    provenance: dict[str, Any] = {"source_file": path.name, "adapter": "afp-dcp-json-v1"}
    context = payload.get("evidence_context") or {}
    for key in ("data_origin", "pairing_status"):
        if context.get(key):
            provenance[key] = str(context[key])
    if payload.get("data_relationship"):
        provenance["data_relationship"] = str(payload["data_relationship"])
    if index_time is not None:
        provenance["index_time"] = index_time

    usable_count = len(observations)
    if usable_count == 0:
        status: QualityStatus = "fail"
        errors = ["No usable AFP or DCP observations remain after validation"]
    elif warnings or rejected:
        status = "warning"
        errors = []
    else:
        status = "pass"
        errors = []
    checks = [
        QualityCheck(
            check_id="LAB_PARSE_USABLE",
            status="pass" if usable_count else "fail",
            message=f"{usable_count} usable and {len(rejected)} rejected observations",
        ),
        QualityCheck(
            check_id="LAB_REQUIRED_MARKERS",
            status="pass" if all(name in markers for name in ("AFP", "DCP")) else "warning",
            message="AFP and DCP availability checked",
        ),
    ]
    return LabEvidence(
        patient_id=patient_id,
        markers=markers,
        rejected_observations=rejected,
        quality=QualityEvidence(
            status=status,
            checks=checks,
            warnings=sorted(set(warnings)),
            errors=errors,
        ),
        warnings=sorted(set(warnings)),
        provenance=provenance,
        sources=[
            SourceReference(
                source_id=path.name,
                source_type="laboratory_json",
                uri=path.name,
                data_origin=str(context.get("data_origin") or "unknown"),
                deidentified=None,
            )
        ],
    )


def align_lab_evidence(
    labs: LabEvidence,
    *,
    baseline_date: str,
    followup_date: str,
    window_days: int = 14,
) -> LabEvidence:
    """Select one observation per marker around each image without future-data preference."""
    if window_days < 0:
        raise ValueError("window_days must be non-negative")
    targets = (date.fromisoformat(baseline_date), date.fromisoformat(followup_date))
    if targets[0] >= targets[1]:
        raise ValueError("followup_date must occur after baseline_date")

    aligned_markers: dict[str, MarkerTrend] = {}
    warnings = list(labs.warnings)
    selected_dates: dict[str, list[str]] = {}
    for marker_name in ("AFP", "DCP"):
        marker = labs.markers.get(marker_name)
        if marker is None:
            warnings.append(f"{marker_name} is unavailable for image-aligned analysis")
            continue
        selected: list[MarkerObservation] = []
        for target in targets:
            candidates: list[tuple[int, int, int, MarkerObservation]] = []
            for observation in marker.observations:
                observed = date.fromisoformat(observation.date)
                distance = abs((observed - target).days)
                if distance <= window_days:
                    after_image = 1 if observed > target else 0
                    candidates.append(
                        (distance, after_image, -observed.toordinal(), observation)
                    )
            if candidates:
                selected.append(min(candidates, key=lambda item: item[:3])[3])
        unique = {(item.date, item.source_text, item.comparator): item for item in selected}
        ordered = sorted(unique.values(), key=lambda item: item.date)
        if len(ordered) < 2:
            warnings.append(
                f"{marker_name} lacks distinct baseline/follow-up observations within "
                f"+/-{window_days} days"
            )
        if ordered:
            marker_key: MarkerName = "AFP" if marker_name == "AFP" else "DCP"
            aligned_markers[marker_name] = _trend(marker_key, ordered)
            selected_dates[marker_name] = [item.date for item in ordered]

    if not aligned_markers:
        status: QualityStatus = "fail"
        errors = ["No AFP or DCP observations align with the image dates"]
    elif warnings or any(
        marker.exact_observation_count < 2 for marker in aligned_markers.values()
    ):
        status = "warning"
        errors = []
    else:
        status = "pass"
        errors = []
    provenance = {
        **labs.provenance,
        "adapter": "image-aligned-afp-dcp-v1",
        "baseline_date": baseline_date,
        "followup_date": followup_date,
        "alignment_window_days": window_days,
        "selected_dates": selected_dates,
    }
    return LabEvidence(
        patient_id=labs.patient_id,
        markers=aligned_markers,
        rejected_observations=labs.rejected_observations,
        quality=QualityEvidence(
            status=status,
            checks=[
                QualityCheck(
                    check_id="LAB_IMAGE_ALIGNMENT",
                    status="pass" if status == "pass" else "warning" if aligned_markers else "fail",
                    message=(
                        f"Selected observations within +/-{window_days} days of baseline "
                        "and follow-up imaging"
                    ),
                    details={"selected_dates": selected_dates},
                )
            ],
            warnings=sorted(set(warnings)),
            errors=errors,
        ),
        warnings=sorted(set(warnings)),
        provenance=provenance,
        sources=labs.sources,
    )
