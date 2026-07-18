from __future__ import annotations

from pathlib import Path
from typing import Any


STATE_LABELS = {
    "cross_sectional_lesion_marker_signal": "横断面影像标注与合成标志物共同形成信号",
    "segmented_lesion_without_marker_signal": "存在影像标注，标志物未形成支持信号",
    "marker_signal_without_segmented_lesion": "存在标志物信号，未见有效分割区域",
    "insufficient_cross_sectional_evidence": "横断面证据不足",
    "concordant_progression_signal": "纵向影像与标志物共同形成进展信号",
    "discordant_imaging_progression": "影像与标志物证据不一致",
    "discordant_marker_rise_imaging_response": "标志物与影像反应信号不一致",
    "discordant_marker_rise_without_imaging_progression": "标志物与影像证据不一致",
    "insufficient_evidence": "当前证据不足",
    "indeterminate_after_intervening_treatment": "治疗事件后证据归因不确定",
}


def _bullets(items: list[str], empty_text: str) -> str:
    if not items:
        return f"- {empty_text}"
    return "\n".join(f"- {item}" for item in items)


def render_human_markdown(report: dict[str, Any]) -> str:
    """Render only a report that has already passed the blocking validator."""
    assessment = report["multimodal_assessment"]
    state = assessment["state"]
    state_label = STATE_LABELS.get(state, "结构化多模态证据结果")
    review = "是" if report.get("review_required") else "否"
    return (
        "# HCC 多模态研究证据摘要\n\n"
        f"> 数据范围：{report['data_scope']}\n\n"
        "## 综合结果\n\n"
        f"**{state_label}。** {assessment['summary']}\n\n"
        f"- 人工复核：**{review}**\n"
        f"- 数据质量：**{report['data_quality_status']}**\n"
        f"- 场景：{report['scenario_id']}\n\n"
        "## 影像证据\n\n"
        + _bullets(report.get("imaging_summary") or [], "未提供有效影像摘要")
        + "\n\n## 检验证据\n\n"
        + _bullets(report.get("laboratory_summary") or [], "未提供有效检验摘要")
        + "\n\n## 支持证据\n\n"
        + _bullets(assessment.get("supporting_evidence") or [], "暂无支持证据")
        + "\n\n## 冲突证据\n\n"
        + _bullets(assessment.get("conflicting_evidence") or [], "未记录显式冲突")
        + "\n\n## 缺失与不确定性\n\n"
        + _bullets(assessment.get("missing_evidence") or [], "未记录缺失证据")
        + "\n\n"
        + _bullets(report.get("uncertainty") or [], "未记录额外不确定性")
        + "\n\n## 质量与使用边界\n\n"
        + _bullets(report.get("quality_and_limits") or [], "未记录额外限制")
        + f"\n\n> {report['disclaimer']}\n\n"
        "<details>\n<summary>机器审计字段</summary>\n\n"
        f"- state: {state}\n"
        f"- concordance: {assessment['concordance']}\n"
        f"- report_type: {report['report_type']}\n"
        f"- case_id: {report['case_id']}\n"
        "</details>\n"
    )


def write_human_markdown(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_human_markdown(report), encoding="utf-8")
    return target
