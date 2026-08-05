"""Deterministic HPI event extraction and longitudinal trend state machine."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from .case_models import (
    ClinicalLabEvidence,
    EvidenceConcordance,
    HpiEvent,
    HpiEventType,
    HpiTimelineEvidence,
    ImagingInterpretationEvidence,
    MarkerTrajectorySummary,
    SourceSpan,
)
from .clinical_labs import _date
from .schemas import QualityCheck, QualityEvidence, QualityStatus, SourceReference

DATE_RE = re.compile(r"(?P<date>20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{2}" + "\u5e74" + r"\d{1,2}" + "\u6708" + r"\d{1,2}" + "\u65e5)" )
RELATIVE_RE = re.compile(r"(?:post[- ]?op|post[- ]?treatment|" + "\u672f\u540e" + "|" + "\u6cbb\u7597\u540e" + r")\s*(?P<number>\d+|[\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\s*(?P<unit>day|days|week|weeks|month|months|" + "\u5929" + "|" + "\u5468" + "|" + "\u4e2a?\u6708" + ")", re.IGNORECASE)


Confidence = Literal["high", "moderate", "low"]
ImagingState = Literal["single_timepoint", "stable", "response_signal", "progression_signal", "new_lesion_signal", "indeterminate", "unavailable"]


def _event_type(text: str) -> tuple[HpiEventType, str, Confidence]:
    lowered = text.casefold()
    terms: tuple[tuple[HpiEventType, tuple[str, ...], str], ...] = (
        ("treatment", ("tace", "tae", "radiotherapy", "chemotherapy", "immunotherapy", "\u6cbb\u7597", "\u4ecb\u5165"), "treatment"),
        ("surgery", ("surgery", "resection", "operation", "\u624b\u672f", "\u5207\u9664"), "surgery"),
        ("laboratory", ("afp", "dcp", "pivka", "bilirubin", "albumin", "\u68c0\u9a8c", "\u7532\u80ce\u86cb\u767d"), "laboratory"),
        ("imaging", ("ct", "mri", "mr", "ultrasound", "\u5f71\u50cf", "\u589e\u5f3a"), "imaging"),
        ("pathology_text", ("pathology", "biopsy", "histology", "\u75c5\u7406", "\u6d3b\u68c0"), "pathology"),
        ("hospitalization", ("admitted", "hospitalized", "inpatient", "\u4f4f\u9662"), "hospitalization"),
        ("followup", ("follow-up", "followup", "review", "\u590d\u67e5", "\u968f\u8bbf"), "followup"),
        ("symptom", ("pain", "fever", "jaundice", "symptom", "\u75c7\u72b6", "\u8179\u75db", "\u9ec4\u75b8"), "symptom"),
    )
    for kind, keywords, concept in terms:
        if any(keyword.casefold() in lowered for keyword in keywords):
            return kind, concept, "high"
    return "other", "other", "low"


def _event_from_row(row: dict[str, Any], *, source: str, span: tuple[int, int] | None = None) -> HpiEvent:
    text = str(row.get("text") or row.get("event") or row.get("description") or "").strip()
    kind, concept, confidence = _event_type(text)
    event_date = _date(row.get("event_date") or row.get("date"))
    if event_date == "unknown":
        event_date_value = None
        date_source: Literal["explicit", "relative_unresolved", "missing"] = "relative_unresolved" if RELATIVE_RE.search(text) else "missing"
    else:
        event_date_value = event_date
        date_source = "explicit"
    return HpiEvent(
        event_date=event_date_value,
        available_at=_date(row.get("available_at")) if row.get("available_at") else event_date_value,
        event_type=kind,
        text=text,
        normalized_concept=str(row.get("normalized_concept") or concept),
        date_source=date_source,
        source_span=SourceSpan(start=span[0], end=span[1]) if span else None,
        confidence=cast(Confidence, str(row.get("confidence") or confidence)),
        source=source,
    )


def _parse_input(path: Path) -> list[HpiEvent]:
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.casefold() == ".json":
        payload = json.loads(raw)
        rows = payload if isinstance(payload, list) else payload.get("events", []) if isinstance(payload, dict) else []
        return [_event_from_row(row, source=path.name) for row in rows if isinstance(row, dict)]
    events: list[HpiEvent] = []
    for match in re.finditer(r"[^\r\n]+", raw):
        text = match.group(0).strip()
        if not text:
            continue
        date_match = DATE_RE.search(text)
        row = {"date": date_match.group("date") if date_match else None, "text": text}
        events.append(_event_from_row(row, source=path.name, span=(match.start(), match.end())))
    return events


def _resolve_relative(events: list[HpiEvent]) -> list[HpiEvent]:
    anchors = sorted(item.event_date for item in events if item.event_date and item.event_type in {"treatment", "surgery"})
    if not anchors:
        return events
    resolved: list[HpiEvent] = []
    for item in events:
        if item.event_date or not (match := RELATIVE_RE.search(item.text)):
            resolved.append(item)
            continue
        anchor = anchors[-1]
        raw_number = match.group("number")
        number = int(raw_number) if raw_number.isdigit() else {"\u4e00": 1, "\u4e8c": 2, "\u4e24": 2, "\u4e09": 3, "\u56db": 4, "\u4e94": 5, "\u516d": 6, "\u4e03": 7, "\u516b": 8, "\u4e5d": 9, "\u5341": 10}.get(raw_number, 1)
        unit = match.group("unit").casefold()
        days = number * (30 if "month" in unit or "\u6708" in unit else 7 if "week" in unit or "\u5468" in unit else 1)
        resolved_date = date.fromisoformat(anchor) + timedelta(days=days)
        resolved.append(item.model_copy(update={"event_date": resolved_date.isoformat(), "available_at": resolved_date.isoformat(), "date_source": "relative_resolved"}))
    return resolved


def _marker_summary(labs: ClinicalLabEvidence, treatment_dates: list[str]) -> dict[str, MarkerTrajectorySummary]:
    result: dict[str, MarkerTrajectorySummary] = {}
    for name, evidence in labs.analytes.items():
        if evidence.group != "tumor_marker":
            continue
        usable = [item for item in evidence.observations if item.value is not None]
        baseline = usable[0] if usable else None
        latest = usable[-1] if usable else None
        nadir = min(usable, key=lambda item: item.value or float("inf")) if usable else None
        state = evidence.trajectory_state
        if state == "falling_after_treatment" and not treatment_dates:
            state = "insufficient"
        result[name] = MarkerTrajectorySummary(
            analyte=name,
            state=state,
            baseline_value=evidence.baseline_value,
            baseline_date=baseline.observed_at if baseline else None,
            nadir_value=evidence.nadir_value,
            nadir_date=nadir.observed_at if nadir else None,
            latest_value=evidence.latest_value,
            latest_date=latest.observed_at if latest else None,
            log_slope_per_day=evidence.log_slope_per_day,
            nadir_ratio=evidence.nadir_ratio,
            rebound_ratio=evidence.rebound_ratio,
        )
    return result


def _imaging_state(imaging: ImagingInterpretationEvidence | None) -> ImagingState:
    if imaging is None or imaging.interpretation_mode == "unavailable":
        return "unavailable"
    text = " ".join(item.observation.casefold() for item in imaging.qualitative_observations)
    if any(term in text for term in ("progression", "new lesion", "increase", "worsen", "\u8fdb\u5c55", "\u65b0\u53d1", "\u589e\u5927")):
        return "progression_signal"
    if any(term in text for term in ("response", "decrease", "stable", "\u7f29\u5c0f", "\u7a33\u5b9a")):
        return "response_signal" if any(term in text for term in ("response", "decrease", "\u7f29\u5c0f")) else "stable"
    return "single_timepoint"


def parse_hpi_timeline(
    path: str | Path,
    *,
    patient_id: str,
    labs: ClinicalLabEvidence | None = None,
    imaging: ImagingInterpretationEvidence | None = None,
    index_date: str | None = None,
) -> HpiTimelineEvidence:
    source = Path(path)
    events = _resolve_relative(_parse_input(source))
    if labs is not None:
        for evidence in labs.analytes.values():
            for observation in evidence.observations:
                events.append(HpiEvent(
                    event_date=None if observation.observed_at == "unknown" else observation.observed_at,
                    available_at=None if observation.observed_at == "unknown" else observation.observed_at,
                    event_type="laboratory",
                    text=observation.source_text,
                    normalized_concept=observation.analyte,
                    date_source="explicit" if observation.observed_at != "unknown" else "missing",
                    source_span=observation.source_span,
                    confidence="high" if observation.parse_status == "exact" else "moderate",
                    source="laboratory",
                ))
    if imaging is not None:
        events.append(HpiEvent(
            event_date=imaging.study_date,
            available_at=imaging.study_date,
            event_type="imaging",
            text=f"{imaging.modality} study {imaging.study_date}",
            normalized_concept="imaging_study",
            date_source="explicit",
            source_span=None,
            confidence="high",
            source="imaging",
        ))
    future_events = 0
    if index_date is not None:
        kept_events: list[HpiEvent] = []
        for event in events:
            if event.event_date is not None and event.event_date > index_date:
                future_events += 1
            else:
                kept_events.append(event)
        events = kept_events
    events.sort(key=lambda item: (item.event_date is None, item.event_date or "", item.source, item.text))
    treatment_dates = sorted({item.event_date for item in events if item.event_date and item.event_type in {"treatment", "surgery"}})
    index_dates = [item.event_date for item in events if item.event_date]
    marker_trajectories = _marker_summary(labs, treatment_dates) if labs else {}
    states = [item.state for item in marker_trajectories.values()]
    imaging_state = _imaging_state(imaging)
    rising = any(state in {"persistent_rising", "rebound_after_nadir"} for state in states)
    falling = any(state == "falling_after_treatment" for state in states)
    if imaging_state in {"progression_signal", "new_lesion_signal"} and rising:
        overall: EvidenceConcordance = "concordant_progression"
    elif imaging_state == "response_signal" and falling:
        overall = "concordant_response"
    elif imaging_state != "unavailable" and (rising or falling) and imaging_state == "single_timepoint":
        overall = "stable_with_uncertainty"
    elif imaging_state == "unavailable" and not states:
        overall = "insufficient_evidence"
    elif imaging_state in {"progression_signal", "response_signal"} and states:
        overall = "discordant_evidence"
    else:
        overall = "stable_with_uncertainty" if states or imaging_state != "unavailable" else "insufficient_evidence"
    limitations: list[str] = []
    if not events:
        limitations.append("No dated HPI, laboratory, or imaging events were available")
    if imaging_state == "single_timepoint":
        limitations.append("Only one qualitative imaging time point is available")
    if not labs:
        limitations.append("Laboratory observations were not supplied to the timeline")
    if future_events:
        limitations.append(f"Excluded {future_events} event(s) after the declared index date")
    liver_state: Literal["stable", "improving", "worsening", "mixed", "insufficient"] = "insufficient"
    if labs:
        reserve_names = ("total_bilirubin", "ALT", "AST")
        directions: list[int] = []
        for name in reserve_names:
            reserve_evidence = labs.analytes.get(name)
            if reserve_evidence and reserve_evidence.baseline_value is not None and reserve_evidence.latest_value is not None and reserve_evidence.baseline_value != 0:
                delta = (reserve_evidence.latest_value - reserve_evidence.baseline_value) / abs(reserve_evidence.baseline_value)
                directions.append(1 if delta >= 0.2 else -1 if delta <= -0.2 else 0)
        if directions and all(value <= 0 for value in directions) and any(value < 0 for value in directions):
            liver_state = "improving"
        elif directions and all(value >= 0 for value in directions) and any(value > 0 for value in directions):
            liver_state = "worsening"
        elif directions:
            liver_state = "mixed" if any(value != 0 for value in directions) else "stable"
        else:
            liver_state = "stable"
    quality_status: QualityStatus = "fail" if not events else "warning" if limitations else "pass"
    return HpiTimelineEvidence(
        patient_id=patient_id,
        index_date=index_date or (max(index_dates) if index_dates else None),
        events=events,
        treatment_dates=treatment_dates,
        marker_trajectories=marker_trajectories,
        imaging_state=imaging_state,
        liver_function_state=liver_state,
        overall_trend=overall,
        quality=QualityEvidence(
            status=quality_status,
            checks=[QualityCheck(check_id="HPI_EVENTS", status="pass" if events else "fail", message=f"Collected {len(events)} timeline events")],
            warnings=limitations if quality_status == "warning" else [],
            errors=["No timeline evidence available"] if quality_status == "fail" else [],
        ),
        limitations=limitations,
        sources=[SourceReference(source_id=source.name, source_type="hpi", uri=source.name, deidentified=True)],
    )


__all__ = ["parse_hpi_timeline"]
