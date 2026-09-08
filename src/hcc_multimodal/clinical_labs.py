"""CPU-only laboratory report parsing for the three-line case pipeline.

The parser deliberately favors traceability over aggressive extraction.  Every
observation keeps its source text and (for plain text reports) character span;
values which cannot be normalized are retained as rejected observations.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, cast

from .case_models import (
    ClinicalAnalyteEvidence,
    ClinicalLabEvidence,
    ClinicalLabGroup,
    ClinicalLabObservation,
    ClinicalParseStatus,
    LiverReserveEvidence,
    SourceSpan,
    TrajectoryState,
)
from .schemas import (
    Comparator,
    QualityCheck,
    QualityEvidence,
    QualityStatus,
    SourceReference,
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VALUE_RE = re.compile(rf"(?P<cmp><=|>=|<|>)?\s*(?P<number>{NUMBER})")
DATE_RE = re.compile(
    r"(?P<date>(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:19|20)\d{2}" + "\u5e74" + r"\d{1,2}" + "\u6708" + r"\d{1,2}" + "\u65e5)"
)
RANGE_RE = re.compile(rf"(?P<low>{NUMBER})\s*(?:-|~|" + "\u81f3" + "|" + "\u5230" + rf")\s*(?P<high>{NUMBER})")

ANALYTE_DEFINITIONS: dict[str, tuple[ClinicalLabGroup, tuple[str, ...], str]] = {
    "AFP": ("tumor_marker", ("AFP", "甲胎蛋白", "ALPHA-FETOPROTEIN"), "ng/mL"),
    "DCP": ("tumor_marker", ("DCP", "PIVKA-II", "PIVKA II", "PIVKA"), "mAU/mL"),
    "AFP-L3%": ("tumor_marker", ("AFP-L3%", "AFP-L3", "甲胎蛋白异质体"), "%"),
    "total_bilirubin": ("liver_reserve", ("TOTAL BILIRUBIN", "TBIL", "总胆红素", "胆红素"), "umol/L"),
    "albumin": ("liver_reserve", ("ALBUMIN", "ALB", "白蛋白"), "g/L"),
    "PT": ("liver_reserve", ("PROTHROMBIN TIME", "PT", "凝血酶原时间"), "s"),
    "INR": ("liver_reserve", ("INR", "国际标准化比值"), "ratio"),
    "ALT": ("liver_injury", ("ALT", "谷丙转氨酶", "丙氨酸氨基转移酶"), "U/L"),
    "AST": ("liver_injury", ("AST", "谷草转氨酶", "天门冬氨酸氨基转移酶"), "U/L"),
    "ALP": ("liver_injury", ("ALP", "碱性磷酸酶"), "U/L"),
    "GGT": ("liver_injury", ("GGT", "γ-GT", "GGT"), "U/L"),
    "platelet": ("liver_injury", ("PLT", "PLATELET", "血小板"), "10^9/L"),
    "creatinine": ("liver_injury", ("CREATININE", "CREA", "肌酐"), "umol/L"),
    "HBsAg": ("viral_etiology", ("HBSAG", "乙肝表面抗原"), "qualitative"),
    "HBeAg": ("viral_etiology", ("HBEAG", "乙肝E抗原"), "qualitative"),
    "HBV-DNA": ("viral_etiology", ("HBV-DNA", "HBV DNA", "乙肝病毒DNA"), "IU/mL"),
    "anti-HCV": ("viral_etiology", ("ANTI-HCV", "抗-HCV", "丙肝抗体"), "qualitative"),
    "HCV-RNA": ("viral_etiology", ("HCV-RNA", "HCV RNA", "丙肝病毒RNA"), "IU/mL"),
}

_UNICODE_ALIASES: dict[str, tuple[str, ...]] = {
    "AFP": ("\u7532\u80ce\u86cb\u767d",),
    "AFP-L3%": ("\u7532\u80ce\u86cb\u767d\u5f02\u8d28\u4f53",),
    "total_bilirubin": ("\u603b\u80c6\u7ea2\u7d20", "\u80c6\u7ea2\u7d20"),
    "albumin": ("\u767d\u86ce\u767d",),
    "PT": ("\u51dd\u8840\u9176\u539f\u65f6\u95f4",),
    "INR": ("\u56fd\u9645\u6807\u51c6\u5316\u6bd4\u503c",),
    "ALT": ("\u4e19\u6c28\u9178\u6c28\u57fa\u8f6c\u79fb\u9176",),
    "AST": ("\u5929\u95e8\u51ac\u6c28\u9178\u6c28\u57fa\u8f6c\u79fb\u9176",),
    "ALP": ("\u78b1\u6027\u78f7\u9178\u9176",),
    "platelet": ("\u8840\u5c0f\u677f",),
    "creatinine": ("\u808c\u9150",),
    "HBsAg": ("\u4e59\u809d\u8868\u9762\u6297\u539f",),
    "HBeAg": ("\u4e59\u809dE\u6297\u539f",),
    "anti-HCV": ("\u4e19\u809d\u6297\u4f53",),
}

_UNIT_FACTORS: dict[str, dict[str, tuple[str, float]]] = {
    "AFP": {"ng/ml": ("ng/mL", 1.0), "ug/l": ("ng/mL", 1.0), "µg/l": ("ng/mL", 1.0), "mcg/l": ("ng/mL", 1.0)},
    "DCP": {"mau/ml": ("mAU/mL", 1.0), "au/l": ("mAU/mL", 1.0), "mau/l": ("mAU/mL", 0.001)},
    "AFP-L3%": {"%": ("%", 1.0), "percent": ("%", 1.0)},
    "total_bilirubin": {"umol/l": ("umol/L", 1.0), "μmol/l": ("umol/L", 1.0), "mg/dl": ("umol/L", 17.104)},
    "albumin": {"g/l": ("g/L", 1.0), "g/dl": ("g/L", 10.0)},
    "PT": {"s": ("s", 1.0), "sec": ("s", 1.0)},
    "INR": {"": ("ratio", 1.0), "ratio": ("ratio", 1.0)},
    "ALT": {"u/l": ("U/L", 1.0), "iu/l": ("U/L", 1.0)},
    "AST": {"u/l": ("U/L", 1.0), "iu/l": ("U/L", 1.0)},
    "ALP": {"u/l": ("U/L", 1.0), "iu/l": ("U/L", 1.0)},
    "GGT": {"u/l": ("U/L", 1.0), "iu/l": ("U/L", 1.0)},
    "platelet": {"10^9/l": ("10^9/L", 1.0), "10⁹/l": ("10^9/L", 1.0), "k/ul": ("10^9/L", 1.0)},
    "creatinine": {"umol/l": ("umol/L", 1.0), "μmol/l": ("umol/L", 1.0), "mg/dl": ("umol/L", 88.4)},
    "HBV-DNA": {"iu/ml": ("IU/mL", 1.0), "copies/ml": ("IU/mL", 1.0)},
    "HCV-RNA": {"iu/ml": ("IU/mL", 1.0), "copies/ml": ("IU/mL", 1.0)},
}

_MISSING = {"", "-", "--", "na", "n/a", "nan", "none", "null", "未检", "未检测", "缺失"}


def _date(value: Any) -> str:
    text = str(value or "").strip()
    match = DATE_RE.search(text)
    if not match:
        return "unknown"
    raw = match.group("date").replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-")
    parts = raw.split("-")
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def _canonical_analyte(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    matches: list[tuple[int, str]] = []
    for canonical, (_, aliases, _) in ANALYTE_DEFINITIONS.items():
        aliases = (*aliases, *_UNICODE_ALIASES.get(canonical, ()))
        if canonical.casefold() == text:
            matches.append((len(canonical), canonical))
        for alias in aliases:
            if alias.casefold() == text or alias.casefold() in text:
                matches.append((len(alias), canonical))
    return max(matches, default=(0, None))[1]


def _clean_unit(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().casefold().replace("μ", "µ"))


def _parse_value(value: Any) -> tuple[float | None, str, ClinicalParseStatus]:
    text = str(value if value is not None else "").strip().replace(",", "")
    if text.casefold() in _MISSING:
        return None, "eq", "missing"
    match = VALUE_RE.fullmatch(text)
    if not match:
        return None, "eq", "invalid"
    comparator = {None: "eq", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[match.group("cmp")]
    return float(match.group("number")), comparator, "exact" if comparator == "eq" else "censored"


def _reference(value: Any) -> tuple[float | None, float | None]:
    text = str(value or "").strip().replace(",", "")
    match = RANGE_RE.search(text)
    if match:
        low, high = float(match.group("low")), float(match.group("high"))
        return min(low, high), max(low, high)
    number = VALUE_RE.fullmatch(text)
    if number:
        value = float(number.group("number"))
        return (None, value) if number.group("cmp") in {"<", "<="} else (value, None)
    return None, None


def _normalize(analyte: str, value: float | None, unit: str) -> tuple[float | None, str, ClinicalParseStatus]:
    canonical = ANALYTE_DEFINITIONS[analyte][2]
    if analyte in {"HBsAg", "HBeAg", "anti-HCV"}:
        return value, canonical, "exact" if value is not None else "missing"
    key = _clean_unit(unit)
    factor = _UNIT_FACTORS.get(analyte, {}).get(key)
    if factor is None:
        return None, canonical, "unsupported_unit" if value is not None else "missing"
    return (value * factor[1] if value is not None else None), factor[0], "exact"


def _above(value: float | None, comparator: str, high: float | None) -> bool | None:
    if value is None or high is None:
        return None
    if comparator == "eq":
        return value > high
    if comparator in {"gt", "ge"}:
        return value >= high if comparator == "gt" else value > high
    return value <= high


def _observation(row: dict[str, Any], *, source_text: str, span: tuple[int, int] | None = None) -> ClinicalLabObservation | None:
    analyte = _canonical_analyte(row.get("analyte") or row.get("marker") or row.get("name") or row.get("item") or row.get("systemTestItemName"))
    if analyte is None:
        return None
    group, _, default_unit = ANALYTE_DEFINITIONS[analyte]
    raw_value = row.get("value", row.get("result"))
    qualitative = str(raw_value or "").strip().casefold()
    raw_numeric: float | None
    comparator: str
    status: ClinicalParseStatus
    if analyte in {"HBsAg", "HBeAg", "anti-HCV"} and qualitative in {"positive", "reactive", "\u9633\u6027", "\u9634\u6027", "negative", "nonreactive"}:
        raw_numeric, comparator, status = (1.0 if qualitative in {"positive", "reactive", "\u9633\u6027"} else 0.0), "eq", "exact"
    else:
        raw_numeric, comparator, status = _parse_value(raw_value)
    original_unit = str(row.get("unit") or row.get("units") or default_unit)
    value, unit, normalized_status = _normalize(analyte, raw_numeric, original_unit)
    if status in {"missing", "invalid"}:
        normalized_status = status
        value = None
    elif status == "censored" and normalized_status == "exact":
        normalized_status = "censored"
    low, high = _reference(row.get("reference_range") or row.get("referenceRange") or row.get("range"))
    if row.get("reference_low") is not None:
        low = float(row["reference_low"])
    if row.get("reference_high") is not None:
        high = float(row["reference_high"])
    if normalized_status in {"exact", "censored"}:
        reference_value, _, reference_status = _normalize(analyte, low, original_unit)
        low = reference_value if reference_status == "exact" else None
        reference_value, _, reference_status = _normalize(analyte, high, original_unit)
        high = reference_value if reference_status == "exact" else None
    quality: QualityStatus = "pass" if normalized_status == "exact" else "warning" if normalized_status == "censored" else "fail"
    return ClinicalLabObservation(
        observed_at=_date(row.get("observed_at") or row.get("date") or row.get("inspectionDate")),
        analyte=analyte,
        group=group,
        value=value,
        comparator=cast(Comparator, comparator),
        unit=unit,
        original_unit=original_unit,
        reference_low=low,
        reference_high=high,
        parse_status=normalized_status,
        source_text=source_text,
        source_span=SourceSpan(start=span[0], end=span[1]) if span else None,
        quality_status=quality,
        above_reference=_above(value, comparator, high),
    )


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise TypeError("JSON laboratory input must be an object or array")
    rows = payload.get("observations") or payload.get("results") or payload.get("testResultInfos")
    if rows is None:
        rows = [payload]
    if not isinstance(rows, list):
        raise TypeError("Laboratory observations must be an array")
    return [row for row in rows if isinstance(row, dict)]


def _rows_from_text(text: str) -> list[tuple[dict[str, Any], tuple[int, int]]]:
    rows: list[tuple[dict[str, Any], tuple[int, int]]] = []
    for match in re.finditer(r"[^\r\n]+", text):
        line = match.group(0)
        analyte = _canonical_analyte(line)
        if analyte is None:
            continue
        definition = ANALYTE_DEFINITIONS[analyte]
        alias_match = next((re.search(re.escape(alias), line, re.IGNORECASE) for alias in definition[1] if re.search(re.escape(alias), line, re.IGNORECASE)), None)
        start = alias_match.start() if alias_match else 0
        value_match = VALUE_RE.search(line[alias_match.end() if alias_match else 0:])
        tail_start = alias_match.end() if alias_match else 0
        if value_match:
            value_start = (alias_match.end() if alias_match else 0) + value_match.start()
            value_text = value_match.group(0)
            remainder = line[value_start + len(value_text):]
        elif analyte in {"HBsAg", "HBeAg", "anti-HCV"}:
            qualitative_match = re.search(r"positive|negative|reactive|nonreactive|\u9633\u6027|\u9634\u6027", line[tail_start:], re.IGNORECASE)
            value_text = qualitative_match.group(0) if qualitative_match else ""
            remainder = line[tail_start + (qualitative_match.end() if qualitative_match else 0):]
        else:
            value_text = ""
            remainder = line
        unit_match = re.search(r"([%µμa-zA-Z0-9/^]+(?:\s*/\s*[a-zA-Z0-9]+)?)", remainder)
        unit = unit_match.group(1) if unit_match else definition[2]
        range_match = RANGE_RE.search(remainder[unit_match.end():] if unit_match else remainder)
        rows.append(({
            "analyte": analyte,
            "value": value_text,
            "unit": unit,
            "reference_range": range_match.group(0) if range_match else "",
            "observed_at": _date(line),
        }, (match.start() + start, match.end())))
    return rows


def _parse_file(path: Path) -> tuple[str, list[ClinicalLabObservation]]:
    suffix = path.suffix.casefold()
    raw = path.read_text(encoding="utf-8-sig")
    observations: list[ClinicalLabObservation] = []
    if suffix == ".json":
        payload = json.loads(raw)
        for row in _rows_from_payload(payload):
            item = _observation(row, source_text=json.dumps(row, ensure_ascii=False))
            if item is not None:
                observations.append(item)
        return "json", observations
    if suffix == ".csv":
        for row in csv.DictReader(io.StringIO(raw)):
            item = _observation(dict(row), source_text=",".join(str(value) for value in row.values()))
            if item is not None:
                observations.append(item)
        return "csv", observations
    for row, span in _rows_from_text(raw):
        item = _observation(row, source_text=raw[span[0]:span[1]], span=span)
        if item is not None:
            observations.append(item)
    return "text", observations


def _trajectory(analyte: str, observations: list[ClinicalLabObservation]) -> ClinicalAnalyteEvidence:
    ordered = sorted(observations, key=lambda item: item.observed_at)
    usable = [item for item in ordered if item.value is not None and item.parse_status in {"exact", "censored"}]
    exact = [item for item in usable if item.parse_status == "exact"]
    latest = usable[-1] if usable else None
    baseline = exact[0] if exact else (usable[0] if usable else None)
    nadir = min(exact, key=lambda item: item.value or math.inf) if exact else None
    slope = None
    if len(exact) >= 2:
        try:
            days = (date.fromisoformat(exact[-1].observed_at) - date.fromisoformat(exact[0].observed_at)).days
            if days > 0 and exact[0].value is not None and exact[-1].value is not None:
                slope = math.log(max(exact[-1].value, 1e-9) / max(exact[0].value, 1e-9)) / days
        except ValueError:
            pass
    uncertainty: list[str] = []
    state: TrajectoryState = "insufficient"
    if len(exact) == 1:
        state = "stable_abnormal" if exact[0].above_reference else "stable_normal"
        uncertainty.append("Only one exact observation is available")
    elif len(exact) >= 2:
        first, last = exact[0].value, exact[-1].value
        assert first is not None and last is not None
        relative = (last - first) / max(abs(first), 1e-9)
        if nadir is not None and nadir is not exact[0] and nadir is not exact[-1] and last > (nadir.value or 0) * 1.5:
            state = "rebound_after_nadir"
        elif relative >= 0.2:
            state = "persistent_rising"
        elif relative <= -0.2:
            state = "falling_after_treatment"
        else:
            high = latest.reference_high if latest and latest.reference_high is not None else math.inf
            state = "stable_abnormal" if last > high else "stable_normal"
    if any(item.parse_status == "censored" for item in usable):
        uncertainty.append("Censored values are retained but limit numeric trend certainty")
    if latest is not None and latest.reference_high is None:
        uncertainty.append("The latest observation has no usable upper reference limit")
    return ClinicalAnalyteEvidence(
        analyte=analyte,
        group=ANALYTE_DEFINITIONS[analyte][0],
        observations=ordered,
        latest_value=latest.value if latest else None,
        latest_unit=latest.unit if latest else None,
        latest_above_reference=latest.above_reference if latest else None,
        baseline_value=baseline.value if baseline else None,
        nadir_value=nadir.value if nadir else None,
        trajectory_state=state,
        log_slope_per_day=slope,
        nadir_ratio=(nadir.value / baseline.value if nadir and baseline and nadir.value is not None and baseline.value is not None else None),
        rebound_ratio=(latest.value / nadir.value if latest and nadir and latest.value is not None and nadir.value is not None else None),
        uncertainty=uncertainty,
    )


def _liver_reserve(analytes: dict[str, ClinicalAnalyteEvidence]) -> LiverReserveEvidence:
    bilirubin = analytes.get("total_bilirubin")
    albumin = analytes.get("albumin")
    score = None
    grade = None
    if bilirubin and albumin and bilirubin.latest_value and albumin.latest_value and bilirubin.latest_value > 0:
        score = 0.66 * math.log10(bilirubin.latest_value) - 0.085 * albumin.latest_value
        grade = 1 if score <= -2.60 else 2 if score <= -1.39 else 3
    missing = [name for name in ("total_bilirubin", "albumin", "PT/INR") if (name == "PT/INR" and "PT" not in analytes and "INR" not in analytes) or (name != "PT/INR" and name not in analytes)]
    missing.extend(["ascites", "hepatic_encephalopathy"])
    return LiverReserveEvidence(albi_score=score, albi_grade=grade, missing_components=missing)


def parse_laboratory_report(path: str | Path, *, patient_id: str) -> ClinicalLabEvidence:
    """Parse TXT, JSON, or CSV laboratory input into the public v0.4 contract."""
    source = Path(path)
    source_format, observations = _parse_file(source)
    grouped: dict[str, list[ClinicalLabObservation]] = defaultdict(list)
    rejected: list[ClinicalLabObservation] = []
    for item in observations:
        if item.parse_status in {"exact", "censored"}:
            grouped[item.analyte].append(item)
        else:
            rejected.append(item)
    analytes = {name: _trajectory(name, values) for name, values in sorted(grouped.items())}
    tumor = [name for name in analytes if ANALYTE_DEFINITIONS[name][0] == "tumor_marker"]
    injury = [name for name in analytes if ANALYTE_DEFINITIONS[name][0] == "liver_injury"]
    viral = [name for name in analytes if ANALYTE_DEFINITIONS[name][0] == "viral_etiology"]
    missing = [name for name in ("AFP", "DCP", "AFP-L3%", "total_bilirubin", "albumin") if name not in analytes]
    warnings = [f"{item.analyte}: {item.parse_status}" for item in rejected]
    warnings.extend(message for evidence in analytes.values() for message in evidence.uncertainty)
    if not analytes:
        status: QualityStatus = "fail"
        errors = ["No recognized laboratory observations were parsed"]
    elif rejected or missing:
        status, errors = "warning", []
    else:
        status, errors = "pass", []
    confounders: list[str] = []
    if any(name in analytes for name in ("HBsAg", "HBV-DNA", "HCV-RNA")):
        confounders.append("Viral activity may influence AFP and liver injury markers")
    return ClinicalLabEvidence(
        patient_id=patient_id,
        source_format=cast(Any, source_format),
        analytes=analytes,
        rejected_observations=rejected,
        tumor_marker_evidence=tumor,
        liver_reserve_evidence=_liver_reserve(analytes),
        liver_injury_evidence=injury,
        viral_etiology_evidence=viral,
        confounders=confounders,
        missing_items=missing,
        quality=QualityEvidence(
            status=status,
            checks=[QualityCheck(check_id="LAB_RECOGNITION", status="pass" if analytes else "fail", message=f"Recognized {len(observations)} observations")],
            warnings=sorted(set(warnings)),
            errors=errors,
        ),
        provenance={"source_file": source.name, "adapter": "clinical-labs-v1"},
        sources=[SourceReference(source_id=source.name, source_type="laboratory_report", uri=source.name, deidentified=True)],
    )


__all__ = ["parse_laboratory_report"]
