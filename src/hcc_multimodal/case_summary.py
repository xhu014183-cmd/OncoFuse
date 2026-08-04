"""Deterministic fusion and Markdown rendering for a research evidence summary."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, cast

from .case_models import (
    CaseResearchSummary,
    ClinicalLabEvidence,
    EvidenceConcordance,
    EvidenceLineSummary,
    HpiTimelineEvidence,
    ImagingInterpretationEvidence,
)
from .schemas import QualityEvidence, QualityStatus, SourceReference

_FORBIDDEN = re.compile(r"\b(diagnos|stage|staging|prognos|treatment recommendation|therapy recommendation|li-?rads|bclc)\b|" + "\u8bca\u65ad|\u5206\u671f|\u6cbb\u7597\u5efa\u8bae|\u9884\u540e", re.IGNORECASE)


def _safe(text: str) -> str | None:
    cleaned = " ".join(str(text).split())
    return None if _FORBIDDEN.search(cleaned) else cleaned


def _status(quality: QualityStatus, *, available: bool) -> str:
    if not available:
        return "unavailable"
    return "blocked" if quality == "fail" else "partial" if quality == "warning" else "available"


LineStatus = Literal["available", "partial", "blocked", "unavailable"]


def summarize_case(
    imaging: ImagingInterpretationEvidence,
    labs: ClinicalLabEvidence,
    timeline: HpiTimelineEvidence | None = None,
) -> CaseResearchSummary:
    imaging_findings: list[str] = []
    if imaging.quantitative_measurements:
        total = sum(item.volume_ml for item in imaging.quantitative_measurements)
        imaging_findings.append(f"{len(imaging.quantitative_measurements)} SEG 病灶，程序计算总量 {total:.2f} mL")
    for finding in imaging.qualitative_observations:
        safe = _safe(finding.observation)
        if safe:
            imaging_findings.append(f"{finding.location}: {safe}")
    if not imaging_findings:
        imaging_findings.append("当前影像没有可用的结构化观察或分割定量")
    lab_findings: list[str] = []
    for name in sorted(labs.analytes):
        evidence = labs.analytes[name]
        if evidence.latest_value is not None:
            value = f"{evidence.latest_value:g} {evidence.latest_unit or ''}".strip()
            lab_findings.append(f"{name}: 最新 {value}，趋势 {evidence.trajectory_state}")
    if not lab_findings:
        lab_findings.append("没有可用的数值检验证据")
    timeline_findings: list[str] = []
    if timeline is not None:
        if timeline.treatment_dates:
            timeline_findings.append(f"治疗/手术锚点: {', '.join(timeline.treatment_dates)}")
        timeline_findings.append(f"时间线事件 {len(timeline.events)} 条，总体状态 {timeline.overall_trend}")
        for name, trajectory in sorted(timeline.marker_trajectories.items()):
            if trajectory.baseline_value is not None and trajectory.latest_value is not None:
                timeline_findings.append(f"{name}: 基线 {trajectory.baseline_value:g}，最新 {trajectory.latest_value:g}，{trajectory.state}")
    else:
        timeline_findings.append("未提供 HPI 时间线；趋势分析不完整")
    concordance: EvidenceConcordance = timeline.overall_trend if timeline is not None else "insufficient_evidence"
    key_findings = imaging_findings[:3] + lab_findings[:4]
    if timeline_findings:
        key_findings.append(timeline_findings[-1])
    gaps = sorted(set(imaging.limitations + labs.missing_items + labs.quality.warnings + (timeline.limitations if timeline else ["缺少 HPI 时间线"])))
    uncertainty = sorted(set(imaging.quality.warnings + labs.quality.warnings + (timeline.limitations if timeline else [])))
    line_quality = [imaging.quality.status, labs.quality.status, timeline.quality.status if timeline else "unavailable"]
    quality_status: QualityStatus = "fail" if "fail" in line_quality else "warning" if "warning" in line_quality or "unavailable" in line_quality else "pass"
    errors = ["At least one required evidence line failed validation"] if quality_status == "fail" else []
    return CaseResearchSummary(
        patient_id=imaging.patient_id,
        case_status="blocked" if quality_status == "fail" else "partial" if quality_status != "pass" else "complete",
        imaging_summary=EvidenceLineSummary(status=cast(LineStatus, _status(imaging.quality.status, available=imaging.interpretation_mode != "unavailable")), headline=f"{imaging.modality} {imaging.study_date} ({imaging.interpretation_mode})", findings=imaging_findings, limitations=imaging.limitations, quality_status=imaging.quality.status),
        laboratory_summary=EvidenceLineSummary(status=cast(LineStatus, _status(labs.quality.status, available=bool(labs.analytes))), headline=f"{len(labs.analytes)} 个检验项目进入证据层", findings=lab_findings, limitations=labs.missing_items, quality_status=labs.quality.status),
        timeline_summary=EvidenceLineSummary(status=cast(LineStatus, _status(timeline.quality.status, available=bool(timeline and timeline.events)) if timeline else "unavailable"), headline=f"{len(timeline.events) if timeline else 0} 条时间线事件", findings=timeline_findings, limitations=timeline.limitations if timeline else ["缺少 HPI 时间线"], quality_status=timeline.quality.status if timeline else "unavailable"),
        evidence_concordance=concordance,
        key_findings=key_findings,
        data_gaps=gaps,
        uncertainty=uncertainty,
        requires_human_review=True,
        quality=QualityEvidence(status=quality_status, warnings=uncertainty, errors=errors),
        sources=[SourceReference(source_id="case-summary", source_type="deterministic-fusion", uri=None, deidentified=True)],
    )


def render_case_markdown(summary: CaseResearchSummary) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- 无"

    sections = [
        "# 病例研究证据摘要",
        "",
        f"- 研究假名: `{summary.patient_id}`",
        f"- 病例状态: `{summary.case_status}`",
        f"- 证据一致性: `{summary.evidence_concordance}`",
        "",
        "## 当前影像",
        "",
        f"{summary.imaging_summary.headline}",
        bullets(summary.imaging_summary.findings),
        "",
        "## 检验科结果",
        "",
        summary.laboratory_summary.headline,
        bullets(summary.laboratory_summary.findings),
        "",
        "## 病程时间线",
        "",
        summary.timeline_summary.headline,
        bullets(summary.timeline_summary.findings),
        "",
        "## 数据缺口与不确定性",
        "",
        bullets(summary.data_gaps + summary.uncertainty),
        "",
        "## 研究用途声明",
        "",
        summary.research_disclaimer,
        "",
    ]
    return "\n".join(sections)


def write_case_markdown(summary: CaseResearchSummary, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_case_markdown(summary), encoding="utf-8")
    return target


__all__ = ["render_case_markdown", "summarize_case", "write_case_markdown"]
