"""Single-case controlled prognosis reporting over a frozen Cox model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .deepseek import call_deepseek_narrative
from .glm_vision import run_glm_observer
from .lion_inspired import PrecomputedMaskLionBackend
from .prognosis_features import extract_mask_features
from .prognosis_models import (
    ControlledPrognosisReport,
    CoxModelBundle,
    PrognosticEvidence,
)
from .prognosis_survival import score_feature_mapping
from .schemas import QualityCheck, QualityEvidence, SourceReference

FORBIDDEN_OUTCOME_KEYS = {
    "death",
    "event",
    "os",
    "overall_survival",
    "survival_time",
    "duration_days",
    "progression",
    "ttp",
    "tace_number",
    "treatment_response",
}


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrognosisImagingInput(_StrictInput):
    source_type: Literal["nifti"] = "nifti"
    nifti_image: str
    seg: str
    seg_role: Literal[
        "expert_reference", "public_reference", "user_supplied", "model_prediction"
    ]
    phase: str = "unknown"
    study_date: str = "unknown"


class PrognosisClinicalInput(_StrictInput):
    age_years: float = Field(gt=0, le=120)
    sex: Literal["male", "female"]
    afp_ng_ml: float = Field(ge=0)
    albumin_g_dl: float | None = Field(default=None, gt=0)
    bilirubin_mg_dl: float | None = Field(default=None, ge=0)
    inr: float | None = Field(default=None, gt=0)
    alt_iu_l: float | None = Field(default=None, ge=0)
    creatinine_mg_dl: float | None = Field(default=None, ge=0)


class PrognosisRelationshipInput(_StrictInput):
    imaging_origin: Literal[
        "real_clinical", "real_public", "synthetic", "user_supplied"
    ]
    laboratory_origin: Literal[
        "real_clinical", "real_public", "synthetic", "user_supplied"
    ]
    pairing_status: Literal["same_subject"]
    statement: str = Field(min_length=1)


class PrognosisCaseInput(_StrictInput):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    clinical_task: Literal["tace_overall_survival_prognosis"]
    imaging: PrognosisImagingInput
    clinical: PrognosisClinicalInput
    data_relationship: PrognosisRelationshipInput
    output_dir: str = "prognosis-output"

    @model_validator(mode="after")
    def public_and_synthetic_cannot_claim_pairing(self) -> PrognosisCaseInput:
        if {
            self.data_relationship.imaging_origin,
            self.data_relationship.laboratory_origin,
        } == {"real_public", "synthetic"}:
            raise ValueError(
                "Unpaired public imaging and synthetic clinical values cannot be used for prognosis"
            )
        return self


@dataclass(frozen=True)
class PrognosisRunResult:
    output_dir: Path
    report: ControlledPrognosisReport
    evidence: PrognosticEvidence
    glm_live_success: bool
    deepseek_live_success: bool
    degraded: bool


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).casefold())
            keys.update(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_prognosis_case(path: str | Path) -> tuple[PrognosisCaseInput, Path]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    leaked = sorted(_walk_keys(payload) & FORBIDDEN_OUTCOME_KEYS)
    if leaked:
        raise ValueError(f"Outcome fields are forbidden in prognosis case input: {leaked}")
    return PrognosisCaseInput.model_validate(payload), source.parent


def _resolve(root: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (root / value).resolve()


def _feature_values(
    case: PrognosisCaseInput,
    imaging: dict[str, Any],
) -> dict[str, float]:
    clinical = case.clinical
    values: dict[str, float] = {
        "age_years": clinical.age_years,
        "female": 1.0 if clinical.sex == "female" else 0.0,
        "afp_ng_ml": clinical.afp_ng_ml,
        "lesion_count": float(imaging["lesion_count"]),
        "total_tumor_volume_ml": float(imaging["total_tumor_volume_ml"]),
        "max_lesion_extent_mm": float(imaging["max_lesion_extent_mm"]),
        "largest_lesion_sphericity": float(imaging["largest_lesion_sphericity"]),
    }
    for field in (
        "albumin_g_dl",
        "bilirubin_mg_dl",
        "inr",
        "alt_iu_l",
        "creatinine_mg_dl",
    ):
        value = getattr(clinical, field)
        if value is not None:
            values[field] = float(value)
    return values


def _safe_audit(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"api_key", "authorization", "raw_content", "prompt", "response", "request"}
    return {key: value for key, value in payload.items() if key.casefold() not in blocked}


def _controlled_report(
    case: PrognosisCaseInput,
    evidence: PrognosticEvidence,
    lion: Any,
    glm: Any,
) -> ControlledPrognosisReport:
    patient = lion.patient_evidence
    imaging_summary = [
        (
            "LiON-inspired/precomputed-mask定量："
            f"{patient.lesion_count}个连通病灶，总体积{patient.total_tumor_volume_ml} mL，"
            f"最大三维包围盒范围{patient.max_lesion_extent_mm} mm。"
        )
    ]
    if glm.status in {"pass", "warning"}:
        imaging_summary.extend(
            f"GLM可见征象：{item.location}—{item.observation}"
            for item in glm.observations[:2]
        )
    else:
        imaging_summary.append("GLM定性影像观察不可用；这不代表影像阴性。")
    clinical_summary = [
        (
            f"基线年龄{case.clinical.age_years:g}岁，性别{case.clinical.sex}，"
            f"AFP {case.clinical.afp_ng_ml:g} ng/mL。"
        )
    ]
    group_text = (
        "达到或高于WAW-TACE开发队列中位风险"
        if evidence.research_risk_group == "at_or_above_development_median"
        else "低于WAW-TACE开发队列中位风险"
    )
    return ControlledPrognosisReport(
        case_id=case.case_id,
        data_scope=case.data_relationship.statement,
        imaging_summary=imaging_summary,
        clinical_summary=clinical_summary,
        prognostic_assessment={
            "status": evidence.status,
            "endpoint": evidence.endpoint,
            "model_id": evidence.model_id,
            "model_hash": evidence.model_hash,
            "risk_index": evidence.risk_index,
            "development_percentile": evidence.development_percentile,
            "research_risk_group": evidence.research_risk_group,
            "summary": group_text,
            "external_validation_status": evidence.external_validation_status,
        },
        uncertainty=evidence.limitations,
        data_quality_status=evidence.status,
        intended_use=evidence.intended_use,
    )


def _markdown(
    report: ControlledPrognosisReport,
    narrative: str | None,
) -> str:
    assessment = report.prognostic_assessment
    lines = [
        "# HCC TACE 总生存研究报告",
        "",
        f"病例：{report.case_id}",
        "",
        "## 影像证据",
        "",
        *[f"- {item}" for item in report.imaging_summary],
        "",
        "## 基线临床证据",
        "",
        *[f"- {item}" for item in report.clinical_summary],
        "",
        "## 研究级预后分层",
        "",
        f"- {assessment['summary']}。",
        f"- 风险指数：{assessment['risk_index']:.6f}。",
        f"- 开发队列百分位：{assessment['development_percentile']:.1f}%。",
        f"- 模型：{assessment['model_id']}（{assessment['model_hash']}）。",
        "",
    ]
    if narrative:
        lines.extend(["## DeepSeek辅助解读", "", narrative, ""])
    lines.extend(
        [
            "## 限制",
            "",
            *[f"- {item}" for item in report.uncertainty],
            "",
            f"> {report.disclaimer}",
            "",
        ]
    )
    return "\n".join(lines)


def _prognosis_narrative_prompt(report: ControlledPrognosisReport) -> dict[str, str]:
    redacted = report.to_dict()

    def scrub(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return "数值由本地报告锁定"
        if isinstance(value, str):
            return re.sub(r"\d+(?:\.\d+)?", "数值由本地报告锁定", value)
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return str(value)

    return {
        "prompt_version": "deepseek-hcc-prognosis-narrative-v1",
        "system_message": (
            "你是研究报告语言编辑。所有模型结果由本地程序锁定；你不能重新计算、"
            "修改风险、预测生存月数、诊断、分期或建议治疗。"
        ),
        "user_message": (
            "请用两段通俗中文解释影像肿瘤负荷、AFP和研究级相对风险之间的关系。"
            "必须说明这是回顾性研究模型、需要人工复核且不能用于临床预后。"
            "不得输出阿拉伯数字、JSON、Markdown、列表或思考过程。只输出正文。\n\n"
            + json.dumps(scrub(redacted), ensure_ascii=False, separators=(",", ":"))
        ),
    }


def _validate_narrative(text: str) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not text.strip():
        errors.append({"code": "EMPTY_CONTENT", "message": "No prose returned"})
    if re.search(r"\d", text):
        errors.append({"code": "NUMERIC_CONTENT_BLOCKED", "message": "Narrative contains digits"})
    for pattern, code in (
        (r"确诊|诊断为|分期为", "DIAGNOSTIC_ASSERTION"),
        (r"建议(?:手术|化疗|用药|治疗)|应当治疗", "TREATMENT_RECOMMENDATION"),
        (r"生存期为|还能活|预计存活", "INDIVIDUAL_SURVIVAL_CLAIM"),
    ):
        if re.search(pattern, text):
            errors.append({"code": code, "message": f"Blocked pattern: {pattern}"})
    for pattern, code in (
        (r"影像|病灶|CT", "IMAGING_OMITTED"),
        (r"AFP|检验|标志物", "LABORATORY_OMITTED"),
        (r"风险|预后", "RISK_OMITTED"),
        (r"研究|回顾性|限制|复核|不能", "LIMITATION_OMITTED"),
    ):
        if not re.search(pattern, text, re.IGNORECASE):
            errors.append({"code": code, "message": f"Required concept missing: {pattern}"})
    return errors


def run_prognosis_report(
    case_input: str | Path,
    model_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    glm_mode: Literal["off", "live"] = "off",
    report_mode: Literal["deterministic", "live"] = "deterministic",
    timeout_seconds: float = 180.0,
) -> PrognosisRunResult:
    started = perf_counter()
    case, root = load_prognosis_case(case_input)
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _resolve(root, case.output_dir)
    )
    output.mkdir(parents=True, exist_ok=True)
    provider = output / "provider-audit"
    provider.mkdir(parents=True, exist_ok=True)
    image_path = _resolve(root, case.imaging.nifti_image)
    mask_path = _resolve(root, case.imaging.seg)
    bundle = CoxModelBundle.model_validate_json(
        Path(model_path).read_text(encoding="utf-8")
    )
    _write_json(output / "case-input.normalized.json", case.model_dump(mode="json"))
    lion = PrecomputedMaskLionBackend().infer(
        image_path=image_path,
        mask_path=mask_path,
        mask_role=case.imaging.seg_role,
        patient_id=case.patient_id,
        study_date=case.imaging.study_date,
        phase=case.imaging.phase,
    )
    lion.write_json(output / "lion-inspired-evidence.json")
    glm, glm_audit = run_glm_observer(
        lion=lion,
        image_path=image_path,
        mask_path=mask_path,
        output_dir=output,
        mode=glm_mode,
        timeout_seconds=timeout_seconds,
    )
    glm.write_json(output / "glm-imaging-evidence.json")
    _write_json(provider / "glm.json", _safe_audit(glm_audit))
    measured = extract_mask_features(
        mask_path,
        portal_image_path=image_path if case.imaging.phase == "portal_venous" else None,
    )
    values = _feature_values(case, measured.to_dict())
    missing = [name for name in bundle.feature_names if name not in values]
    limitations = list(bundle.limitations)
    if case.imaging.phase != "portal_venous":
        limitations.append(
            "Input is not declared portal-venous phase; GLM interpretation and transportability are limited"
        )
    if case.imaging.seg_role == "model_prediction":
        limitations.append(
            "Model-predicted segmentation was not used in the public reference-SEG validation"
        )
    if missing:
        evidence = PrognosticEvidence(
            case_id=case.case_id,
            patient_id=case.patient_id,
            status="unavailable",
            model_id=bundle.model_id,
            model_hash=bundle.model_hash,
            endpoint=bundle.endpoint,
            applicability="not_applicable",
            external_validation_status=bundle.external_validation_status,
            feature_values=values,
            supporting_evidence=[],
            missing_evidence=missing,
            limitations=[*limitations, "Required model features are missing"],
            quality=QualityEvidence(
                status="unavailable",
                checks=[
                    QualityCheck(
                        check_id="PROGNOSIS_FEATURE_COMPLETENESS",
                        status="unavailable",
                        message=f"Missing features: {missing}",
                    )
                ],
                warnings=["No risk result was produced"],
            ),
            intended_use="research risk-stratification prototype only",
            sources=[
                SourceReference(
                    source_id=Path(model_path).name,
                    source_type="frozen_cox_model",
                    data_origin="derived",
                )
            ],
        )
        evidence.write_json(output / "prognostic-evidence.json")
        raise ValueError(f"Required prognosis features are missing: {missing}")
    risk, percentile, group = score_feature_mapping(bundle, values)
    warning = (
        bundle.external_validation_status != "evaluated"
        or case.imaging.phase != "portal_venous"
        or measured.geometry_qc == "warning"
    )
    evidence = PrognosticEvidence(
        case_id=case.case_id,
        patient_id=case.patient_id,
        status="warning" if warning else "pass",
        model_id=bundle.model_id,
        model_hash=bundle.model_hash,
        endpoint=bundle.endpoint,
        applicability="limited" if warning else "applicable",
        risk_index=round(risk, 8),
        development_percentile=round(percentile, 3),
        research_risk_group=cast(Any, group),
        external_validation_status=bundle.external_validation_status,
        feature_values={name: values[name] for name in bundle.feature_names},
        supporting_evidence=[
            "Quantitative tumor burden was derived from the supplied SEG",
            "Age, sex, and AFP were supplied as same-subject baseline evidence",
            "Risk was computed locally from a frozen WAW-TACE Cox model",
        ],
        missing_evidence=(
            ["External validation has not yet been opened for this model bundle"]
            if bundle.external_validation_status != "evaluated"
            else []
        ),
        limitations=sorted(set(limitations)),
        quality=QualityEvidence(
            status="warning" if warning else "pass",
            checks=[
                QualityCheck(
                    check_id="PROGNOSIS_FEATURE_COMPLETENESS",
                    status="pass",
                    message="All frozen-model features are available",
                ),
                QualityCheck(
                    check_id="OUTCOME_LEAKAGE_GUARD",
                    status="pass",
                    message="No outcome field was accepted by the case contract",
                ),
            ],
            warnings=sorted(set(measured.warnings + evidence_warning(bundle, case))),
        ),
        intended_use="research risk-stratification prototype only; not clinical prognosis",
        sources=[
            SourceReference(
                source_id=Path(model_path).name,
                source_type="frozen_cox_model",
                data_origin="derived",
                details={"model_hash": bundle.model_hash},
            ),
            SourceReference(
                source_id=image_path.name,
                source_type="nifti_image",
                data_origin=case.data_relationship.imaging_origin,
                deidentified=(
                    True
                    if case.data_relationship.imaging_origin == "real_public"
                    else None
                ),
            ),
            SourceReference(
                source_id=mask_path.name,
                source_type="nifti_segmentation",
                data_origin=case.data_relationship.imaging_origin,
                deidentified=(
                    True
                    if case.data_relationship.imaging_origin == "real_public"
                    else None
                ),
                details={"seg_role": case.imaging.seg_role},
            ),
        ],
    )
    evidence.write_json(output / "prognostic-evidence.json")
    report = _controlled_report(case, evidence, lion, glm)
    report.write_json(output / "controlled-prognosis-report.json")
    prompt = _prognosis_narrative_prompt(report)
    _write_json(output / "deepseek-prompt.json", prompt)
    narrative: str | None = None
    deepseek_success = False
    deepseek_audit: dict[str, Any] = {
        "provider": "disabled",
        "model": "deterministic-template",
        "live_requested": report_mode == "live",
        "audit_status": "unavailable",
        "error_code": "DEEPSEEK_DISABLED",
    }
    if report_mode == "live":
        try:
            result = call_deepseek_narrative(
                prompt,
                timeout_seconds=timeout_seconds,
            )
            candidate = str(result.get("raw_content") or "")
            errors = _validate_narrative(candidate)
            deepseek_audit = _safe_audit(
                {
                    **result,
                    "audit_status": "pass" if not errors else "fail",
                    "validation_errors": errors,
                }
            )
            if not errors:
                narrative = candidate
                deepseek_success = True
        except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            deepseek_audit = {
                "provider": "deepseek",
                "live_requested": True,
                "audit_status": "fail",
                "error_code": "DEEPSEEK_UNAVAILABLE",
                "error": str(exc),
            }
    _write_json(provider / "deepseek.json", deepseek_audit)
    _write_json(
        output / "deepseek-narrative.json",
        {
            "status": "pass" if deepseek_success else "unavailable",
            "narrative": narrative,
            "model": deepseek_audit.get("model", "unconfigured"),
        },
    )
    (output / "controlled-prognosis-report.md").write_text(
        _markdown(report, narrative), encoding="utf-8"
    )
    _write_json(
        output / "pipeline-audit.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "pipeline": "hcc-public-os-prognosis-v1",
            "model_id": bundle.model_id,
            "model_hash": bundle.model_hash,
            "glm_mode": glm_mode,
            "report_mode": report_mode,
            "glm_live_success": glm.status in {"pass", "warning"} and glm_mode == "live",
            "deepseek_live_success": deepseek_success,
            "degraded": (glm_mode == "live" and glm.status not in {"pass", "warning"})
            or (report_mode == "live" and not deepseek_success),
            "outcome_fields_received": False,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        },
    )
    return PrognosisRunResult(
        output_dir=output,
        report=report,
        evidence=evidence,
        glm_live_success=glm.status in {"pass", "warning"} and glm_mode == "live",
        deepseek_live_success=deepseek_success,
        degraded=(glm_mode == "live" and glm.status not in {"pass", "warning"})
        or (report_mode == "live" and not deepseek_success),
    )


def evidence_warning(
    bundle: CoxModelBundle,
    case: PrognosisCaseInput,
) -> list[str]:
    warnings: list[str] = []
    if bundle.external_validation_status != "evaluated":
        warnings.append("Frozen model bundle is not marked externally evaluated")
    if case.imaging.phase != "portal_venous":
        warnings.append("CT phase is not declared portal_venous")
    return warnings


__all__ = [
    "PrognosisCaseInput",
    "PrognosisRunResult",
    "load_prognosis_case",
    "run_prognosis_report",
]
