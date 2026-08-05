"""Deterministic GALAD serological risk scoring for the research demo.

The published GALAD model (Johnson et al., Cancer Epidemiol Biomarkers Prev
2014) combines patient age, sex and the serum markers AFP, AFP-L3% and
PIVKA-II (DCP) in a logistic model::

    logit(P) = b0 + b1 * age + b2 * sex_male
                  + b3 * log10(AFP) + b4 * AFP-L3% + b5 * log10(DCP)
    score    = 1 / (1 + exp(-logit(P)))

All coefficients are published and live in the versioned configuration
``configs/galad_coefficients.v1.yaml`` (Johnson et al. 2014 parameters);
an optional Chinese-cohort re-fit (``configs/galad_coefficients.c_galad.v1.yaml``,
C-GALAD, HBV-predominant population) can be swapped in via
``coefficients_path``.  This module follows the same fail-closed philosophy
as the rest of the pipeline: if any required input is missing, out of range,
or in an unsupported state, the result is returned as ``status="incomplete"``
with the offending fields listed -- no silent imputation is ever performed.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from .case_models import ClinicalLabEvidence
from .schemas import ArtifactModel, JsonModel

DEFAULT_COEFFICIENTS_PATH = (
    Path(__file__).with_name("configs") / "galad_coefficients.v1.yaml"
)

GALADStatus = Literal["calculated", "incomplete"]
RiskTier = Literal["LOW", "INTERMEDIATE", "HIGH"]
NormalizedSex = Literal["male", "female"]

_SEX_MALE = {"male", "m", "男", "男性", "1"}
_SEX_FEMALE = {"female", "f", "女", "女性", "0"}

REQUIRED_ANALYTES: tuple[str, ...] = ("AFP", "AFP-L3%", "DCP")

INTENDED_USE = (
    "GALAD hepatocellular carcinoma risk score (Johnson et al. 2014); "
    "for multimodal risk-stratification demonstration only; not for "
    "diagnosis, staging, prognosis, or treatment decisions."
)


class GALADInputs(JsonModel):
    """Normalized inputs that actually entered the logit computation."""

    age_years: float = Field(gt=0, lt=150)
    sex: NormalizedSex
    afp_ng_ml: float = Field(gt=0)
    afp_l3_pct: float = Field(ge=0, le=100)
    dcp_mau_ml: float = Field(gt=0)


class GALADResult(ArtifactModel):
    """Auditable GALAD-style scoring output (fail-closed)."""

    patient_id: str
    status: GALADStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    logit: float | None = None
    risk_tier: RiskTier | None = None
    inputs: GALADInputs | None = None
    term_contributions: dict[str, float] = Field(default_factory=dict)
    contributing_factors: dict[str, float] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    coefficient_version: str
    evidence_refs: list[str] = Field(default_factory=list)
    intended_use: str = INTENDED_USE


def load_galad_coefficients(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the versioned GALAD coefficient configuration."""
    source = Path(path) if path is not None else DEFAULT_COEFFICIENTS_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("version"), str):
        raise ValueError(f"Invalid GALAD coefficient configuration: {source}")  # noqa: TRY004
    terms = payload.get("terms")
    required = {
        "age_years",
        "sex_male",
        "log10_afp_ng_ml",
        "afp_l3_pct",
        "log10_dcp_mau_ml",
    }
    if not isinstance(payload.get("intercept"), (int, float)):
        raise ValueError(  # noqa: TRY004
            f"GALAD configuration lacks a numeric intercept: {source}"
        )
    if not isinstance(terms, dict) or not required.issubset(terms):
        raise ValueError(f"GALAD configuration lacks required terms {required}: {source}")
    tiers = payload.get("risk_tiers") or {}
    low, intermediate = tiers.get("low_below"), tiers.get("intermediate_below")
    if not (isinstance(low, (int, float)) and isinstance(intermediate, (int, float))):
        raise ValueError(  # noqa: TRY004
            f"GALAD configuration lacks numeric risk tier cutoffs: {source}"
        )
    if not 0.0 < low < intermediate < 1.0:
        raise ValueError(f"GALAD risk tier cutoffs must satisfy 0 < low < intermediate < 1: {source}")
    return payload


def normalize_sex(value: Any) -> NormalizedSex | None:
    """Map free-form sex/gender labels onto the binary GALAD encoding.

    Accepted encodings: ``male``/``m``/``男``/``男性``/``1`` map to male and
    ``female``/``f``/``女``/``女性``/``0`` map to female.  Any other value
    (including e.g. ``2`` used by some LIS systems) returns ``None`` and the
    caller treats sex as missing (fail-closed).
    """
    text = str(value or "").strip().casefold()
    if text in _SEX_MALE:
        return "male"
    if text in _SEX_FEMALE:
        return "female"
    return None


def _risk_tier(score: float, config: dict[str, Any]) -> RiskTier:
    tiers = config["risk_tiers"]
    if score < float(tiers["low_below"]):
        return "LOW"
    if score < float(tiers["intermediate_below"]):
        return "INTERMEDIATE"
    return "HIGH"


def _as_float(value: Any) -> float | None:
    """Safely coerce a value to float; ``None`` when it is not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_analyte_value_and_date(
    labs: ClinicalLabEvidence,
    analyte: str,
) -> tuple[float | None, str | None]:
    """Return ``(latest usable value, observation date)`` for an analyte.

    ``observed_at`` is ``None`` when the latest usable observation carries no
    usable date (e.g. ``"unknown"``).
    """
    evidence = labs.analytes.get(analyte)
    if evidence is None:
        return None, None
    usable = [
        obs
        for obs in evidence.observations
        if obs.value is not None and obs.parse_status in {"exact", "censored"}
    ]
    if not usable:
        return None, None
    latest = max(usable, key=lambda obs: obs.observed_at)
    observed_at = latest.observed_at if latest.observed_at != "unknown" else None
    return latest.value, observed_at


def calculate_galad_from_values(
    afp_ng_ml: float | None,
    afp_l3_pct: float | None,
    dcp_mau_ml: float | None,
    *,
    age_years: float | None,
    sex: Any,
    patient_id: str,
    coefficients_path: str | Path | None = None,
    evidence_refs: list[str] | None = None,
) -> GALADResult:
    """Score from explicitly supplied, already unit-normalized values.

    Fail-closed: any missing or out-of-range input yields
    ``status="incomplete"`` and no score is emitted.  Non-numeric values are
    treated as missing rather than raising.
    """
    config = load_galad_coefficients(coefficients_path)
    version = config["version"]

    missing: list[str] = []
    normalized_sex = normalize_sex(sex)
    if normalized_sex is None:
        missing.append("sex")
    age_value = _as_float(age_years)
    if age_value is None or not (0 < age_value < 150):
        missing.append("age_years")
    afp_value = _as_float(afp_ng_ml)
    if afp_value is None or afp_value <= 0:
        missing.append("AFP")
    l3_value = _as_float(afp_l3_pct)
    if l3_value is None or not (0 <= l3_value <= 100):
        missing.append("AFP-L3%")
    dcp_value = _as_float(dcp_mau_ml)
    if dcp_value is None or dcp_value <= 0:
        missing.append("DCP")

    if missing:
        return GALADResult(
            patient_id=patient_id,
            status="incomplete",
            missing_inputs=missing,
            coefficient_version=version,
            evidence_refs=list(evidence_refs or []),
        )

    # Type narrowing only: the checks above guarantee all values are valid.
    assert normalized_sex is not None
    assert age_value is not None and afp_value is not None
    assert l3_value is not None and dcp_value is not None
    inputs = GALADInputs(
        age_years=age_value,
        sex=normalized_sex,
        afp_ng_ml=afp_value,
        afp_l3_pct=l3_value,
        dcp_mau_ml=dcp_value,
    )

    terms = config["terms"]
    contributions = {
        "age_years": float(terms["age_years"]) * inputs.age_years,
        "sex_male": float(terms["sex_male"]) * (1.0 if inputs.sex == "male" else 0.0),
        "log10_afp_ng_ml": float(terms["log10_afp_ng_ml"]) * math.log10(inputs.afp_ng_ml),
        "afp_l3_pct": float(terms["afp_l3_pct"]) * inputs.afp_l3_pct,
        "log10_dcp_mau_ml": float(terms["log10_dcp_mau_ml"]) * math.log10(inputs.dcp_mau_ml),
    }
    logit = float(config["intercept"]) + sum(contributions.values())
    score = 1.0 / (1.0 + math.exp(-logit))

    total = sum(abs(value) for value in contributions.values())
    shares = (
        {name: abs(value) / total for name, value in contributions.items()}
        if total > 0
        else {name: 0.0 for name in contributions}
    )

    return GALADResult(
        patient_id=patient_id,
        status="calculated",
        score=score,
        logit=logit,
        risk_tier=_risk_tier(score, config),
        inputs=inputs,
        term_contributions=contributions,
        contributing_factors=shares,
        coefficient_version=version,
        evidence_refs=list(evidence_refs or []),
    )


def calculate_galad_score(
    labs: ClinicalLabEvidence,
    *,
    age_years: float | None,
    sex: Any,
    coefficients_path: str | Path | None = None,
    max_analyte_date_span_days: float | None = 90,
) -> GALADResult:
    """Score a parsed :class:`ClinicalLabEvidence` (GALAD-style, fail-closed).

    Marker values are taken from the latest usable observation per analyte;
    unit normalization has already been performed by
    :func:`hcc_multimodal.clinical_labs.parse_laboratory_report` (AFP in
    ng/mL, DCP/PIVKA-II in mAU/mL, AFP-L3% in percent).

    By default the three marker observations must be drawn within
    ``max_analyte_date_span_days`` of one another; otherwise the result is
    ``status="incomplete"`` with ``contemporaneous_window`` listed, because a
    score built from non-contemporaneous measurements is not a valid GALAD
    estimate.  Pass ``None`` to disable the window check.
    """
    values = {
        name: _latest_analyte_value_and_date(labs, name)
        for name in REQUIRED_ANALYTES
    }
    evidence_refs: list[str] = []
    observed_dates: dict[str, str] = {}
    for name in REQUIRED_ANALYTES:
        value, observed_at = values[name]
        if value is None:
            continue
        ref = f"{labs.patient_id}:labs:{name}"
        if observed_at is not None:
            ref = f"{ref}@{observed_at}"
            observed_dates[name] = observed_at
        evidence_refs.append(ref)

    result = calculate_galad_from_values(
        values["AFP"][0],
        values["AFP-L3%"][0],
        values["DCP"][0],
        age_years=age_years,
        sex=sex,
        patient_id=labs.patient_id,
        coefficients_path=coefficients_path,
        evidence_refs=evidence_refs,
    )
    if result.status != "calculated" or len(observed_dates) < 2:
        return result

    parsed_dates = []
    for raw in observed_dates.values():
        try:
            parsed_dates.append(date.fromisoformat(raw))
        except ValueError:
            pass
    if len(parsed_dates) < 2:
        return result
    span_days = (max(parsed_dates) - min(parsed_dates)).days
    if max_analyte_date_span_days is not None and span_days > max_analyte_date_span_days:
        return GALADResult(
            patient_id=labs.patient_id,
            status="incomplete",
            missing_inputs=[*result.missing_inputs, "contemporaneous_window"],
            coefficient_version=result.coefficient_version,
            evidence_refs=result.evidence_refs,
        )
    return result


__all__ = [
    "DEFAULT_COEFFICIENTS_PATH",
    "INTENDED_USE",
    "REQUIRED_ANALYTES",
    "GALADInputs",
    "GALADResult",
    "GALADStatus",
    "RiskTier",
    "calculate_galad_from_values",
    "calculate_galad_score",
    "load_galad_coefficients",
    "normalize_sex",
]
