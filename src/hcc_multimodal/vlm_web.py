"""Local visualization service: upload imaging + labs -> auditable interpretation."""

from __future__ import annotations

import base64
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .case_runner import run_case_vlm
from .clinical_labs import parse_laboratory_report
from .preview import extract_axial_slice_previews, extract_lesion_zoom
from .vlm_llm import imaging_metadata_text
from .vlm_prompting import RESEARCH_DISCLAIMER

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_BODY_BYTES = 512 * 1024 * 1024
SERVICE_VERSION = "0.4.0-web"
_VERSION_PLACEHOLDER = "vlm-web（文件模式）"


def _lab_summary(labs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, analyte in sorted(labs.analytes.items()):
        rows.append(
            {
                "marker": name,
                "latest_value": analyte.latest_value,
                "latest_unit": analyte.latest_unit,
                "latest_above_reference": analyte.latest_above_reference,
                "trajectory": analyte.trajectory_state,
            }
        )
    return rows


def _lab_series(labs: Any) -> dict[str, dict[str, Any]]:
    """Time series of usable observations per analyte for trend charts."""
    series: dict[str, dict[str, Any]] = {}
    for name, analyte in sorted(labs.analytes.items()):
        points = [
            {"date": obs.observed_at, "value": obs.value}
            for obs in analyte.observations
            if obs.value is not None and obs.observed_at not in (None, "unknown")
        ]
        if not points:
            continue
        latest = next(
            (obs for obs in reversed(analyte.observations) if obs.value is not None),
            None,
        )
        series[name] = {
            "points": points,
            "upper_reference": latest.reference_high if latest else None,
        }
    return series


def _risk_tier(
    imaging_metadata: str, lab_summary: list[dict[str, Any]]
) -> dict[str, str]:
    """Research tier derived from the guideline heuristic (not a diagnosis)."""
    lesion_match = re.search(r"lesion_count=(\d+)", imaging_metadata)
    extent_match = re.search(r"max_extent_mm=([\d.]+)", imaging_metadata)
    lesion_count = int(lesion_match.group(1)) if lesion_match else 0
    max_extent_mm = float(extent_match.group(1)) if extent_match else None
    afp = next((row for row in lab_summary if row["marker"] == "AFP"), None)
    dcp = next((row for row in lab_summary if row["marker"] == "DCP"), None)
    afp_value = afp.get("latest_value") if afp else None
    dcp_value = dcp.get("latest_value") if dcp else None
    large = max_extent_mm is not None and max_extent_mm >= 20.0
    afp_high = afp_value is not None and float(afp_value) > 200.0
    dcp_high = dcp_value is not None and float(dcp_value) > 40.0
    if lesion_count <= 0:
        return {"level": "low", "label": "无达标病灶，倾向性不适用"}
    if large and (afp_high or dcp_high):
        return {
            "level": "high",
            "label": "高风险特征组合（病灶 ≥2 cm + AFP>200 或 DCP>40）",
        }
    if large or afp_high or dcp_high:
        return {
            "level": "medium",
            "label": "单一高风险特征（大病灶或标志物显著升高）",
        }
    return {
        "level": "low",
        "label": "低倾向（病灶 <2 cm 且标志物未超显著阈值）",
    }


def _summary_text(
    imaging_metadata: str,
    lab_summary: list[dict[str, Any]],
    tier: dict[str, str],
) -> str:
    lesion_match = re.search(r"lesion_count=(\d+)", imaging_metadata)
    extent_match = re.search(r"max_extent_mm=([\d.]+)", imaging_metadata)
    lesion_count = int(lesion_match.group(1)) if lesion_match else 0
    max_extent_mm = float(extent_match.group(1)) if extent_match else None
    imaging_part = f"影像见 {lesion_count} 个病灶"
    if max_extent_mm is not None:
        imaging_part += f"（最大径 {max_extent_mm:.1f} mm）"
    markers = "、".join(
        f"{row['marker']} {row['latest_value']} {row['latest_unit'] or ''}"
        for row in lab_summary
    ) or "无可用检验"
    return f"{imaging_part}；检验：{markers}；倾向：{tier['label']}（非诊断）"


def _next_steps(tier_level: str) -> list[str]:
    steps = {
        "high": [
            "结合多期增强 MRI / 病理活检确认（研究口径）",
            "评估肝功能与基础肝病背景",
            "与专科医生讨论下一步管理",
        ],
        "medium": [
            "完善多期动态增强影像",
            "复查肿瘤标志物确认趋势",
            "结合临床背景综合评估",
        ],
        "low": ["按临床路径随访", "定期复查影像与标志物"],
    }
    return steps.get(tier_level, steps["low"])


def _timeline(study_date: str, lab_series: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = [{"date": study_date, "event": "CT 影像"}]
    for name, series in lab_series.items():
        for point in series.get("points", []):
            events.append({"date": str(point["date"]), "event": f"{name} 检验"})
    unique: dict[tuple[str, str], None] = {}
    for event in events:
        unique[(event["date"], event["event"])] = None
    ordered = [{"date": key[0], "event": key[1]} for key in sorted(unique)]
    return ordered


def _lab_change(
    lab_series: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    change: dict[str, dict[str, Any] | None] = {}
    for name, series in lab_series.items():
        points = series.get("points", [])
        if len(points) >= 2 and points[0].get("value"):
            start = float(points[0]["value"])
            end = float(points[-1]["value"])
            change[name] = {
                "start": start,
                "end": end,
                "change_pct": round((end - start) / start * 100, 1),
            }
        else:
            change[name] = None
    return change


def _clinical_report(
    imaging_metadata: str,
    lab_summary: list[dict[str, Any]],
    lab_series: dict[str, dict[str, Any]],
    vis_lines: list[str],
    tier_level: str,
) -> dict[str, str]:
    """Doctor-facing clinical prose: no rule-engine jargon, no diagnosis."""
    lesion_match = re.search(r"lesion_count=(\d+)", imaging_metadata)
    extent_match = re.search(r"max_extent_mm=([\d.]+)", imaging_metadata)
    volume_match = re.search(r"total_volume_ml=([\d.]+)", imaging_metadata)
    lesion_count = int(lesion_match.group(1)) if lesion_match else 0
    max_extent_mm = float(extent_match.group(1)) if extent_match else None
    volume_ml = float(volume_match.group(1)) if volume_match else None

    imaging_parts = [f"肝脏内可见 {lesion_count} 个占位性病灶"]
    if max_extent_mm is not None:
        imaging_parts.append(f"最大径约 {max_extent_mm:.1f} mm")
    if volume_ml is not None:
        imaging_parts.append(f"体积约 {volume_ml:.2f} mL")
    imaging_text = "，".join(imaging_parts) + "。"
    if vis_lines:
        imaging_text += "影像显示病灶边界清楚、内部密度不均（AI 辅助观察）。"

    lab_parts: list[str] = []
    for row in lab_summary:
        name = row["marker"]
        value = row.get("latest_value")
        unit = row.get("latest_unit") or ""
        upper = (lab_series.get(name) or {}).get("upper_reference")
        status = "高于参考上限" if row.get("latest_above_reference") else "在参考范围内"
        trend = str(row.get("trajectory") or "未知")
        lab_parts.append(
            f"{name} {value} {unit}（参考 {upper} 以下），{status}，趋势为{trend}"
        )
    lab_text = "；".join(lab_parts) or "暂无可用检验数据"

    elevated = [row for row in lab_summary if row.get("latest_above_reference")]
    rising = [
        row for row in lab_summary if "rising" in str(row.get("trajectory") or "")
    ]
    if lesion_count > 0 and elevated and rising:
        names = "、".join(row["marker"] for row in elevated)
        assessment = (
            f"影像所见病灶与 {names} 升高且呈持续上升趋势相符，提示病变进展信号一致。"
            "结合病灶较大且标志物明显升高，属于需要优先评估的情形。"
        )
    elif lesion_count > 0 and elevated:
        names = "、".join(row["marker"] for row in elevated)
        assessment = (
            f"影像见病灶，{names} 高于参考上限，但趋势尚不明确，"
            "建议结合临床背景与复查结果综合判断。"
        )
    elif lesion_count > 0:
        assessment = (
            "影像见病灶，但肿瘤标志物在参考范围内，影像与检验表现不一致，"
            "需结合临床背景进一步评估。"
        )
    else:
        assessment = "影像未检出达到阈值的病灶，建议结合临床随访。"

    if tier_level == "high":
        suggestion = (
            "建议完善多期增强 MRI 评估病灶血供特点，必要时行病理活检明确性质；"
            "同时评估肝功能与基础肝病背景，并由专科医师综合判断下一步管理。"
        )
    elif tier_level == "medium":
        suggestion = (
            "建议完善多期动态增强影像，复查肿瘤标志物确认趋势，"
            "并结合临床背景进行综合评估。"
        )
    else:
        suggestion = "建议按临床路径定期随访，复查影像与肿瘤标志物。"

    return {
        "imaging": imaging_text,
        "laboratory": lab_text,
        "assessment": assessment,
        "suggestion": suggestion,
    }


def _deterministic_impression(
    imaging_metadata: str, lab_summary: list[dict[str, Any]]
) -> str:
    """Rules-level clinical impression: imaging findings x lab trends (not diagnosis)."""
    lesion_match = re.search(r"lesion_count=(\d+)", imaging_metadata)
    lesion_count = int(lesion_match.group(1)) if lesion_match else 0
    elevated = [row for row in lab_summary if row.get("latest_above_reference")]
    rising = [
        row for row in lab_summary if "rising" in str(row.get("trajectory") or "")
    ]
    if lesion_count <= 0:
        return "确定性提示：影像未检出达到阈值的病灶，检验结果需结合临床判断（非诊断）。"
    names = "、".join(row["marker"] for row in elevated)
    if elevated and rising:
        return (
            f"确定性提示：影像见 {lesion_count} 个病灶，且 {names} 高于参考上限并呈上升趋势"
            "——影像-检验进展信号一致（规则引擎口径，非诊断）。"
        )
    if elevated:
        return (
            f"确定性提示：影像见 {lesion_count} 个病灶，{names} 高于参考上限"
            "——需结合趋势与临床判断（非诊断）。"
        )
    return (
        f"确定性提示：影像见 {lesion_count} 个病灶，检验标志物在参考范围内"
        "——未见检验-影像一致信号，需结合临床判断（非诊断）。"
    )


def _guideline_hint(
    imaging_metadata: str, lab_summary: list[dict[str, Any]]
) -> str:
    """CSCO/EASL-inspired research heuristic: lesion size x marker thresholds."""
    lesion_match = re.search(r"lesion_count=(\d+)", imaging_metadata)
    extent_match = re.search(r"max_extent_mm=([\d.]+)", imaging_metadata)
    lesion_count = int(lesion_match.group(1)) if lesion_match else 0
    max_extent_mm = float(extent_match.group(1)) if extent_match else None
    afp = next((row for row in lab_summary if row["marker"] == "AFP"), None)
    dcp = next((row for row in lab_summary if row["marker"] == "DCP"), None)
    afp_value = afp.get("latest_value") if afp else None
    dcp_value = dcp.get("latest_value") if dcp else None
    large = max_extent_mm is not None and max_extent_mm >= 20.0
    afp_high = afp_value is not None and float(afp_value) > 200.0
    dcp_high = dcp_value is not None and float(dcp_value) > 40.0
    prefix = "指南启发式（CSCO/EASL 参考，非诊断）"
    if lesion_count <= 0:
        return f"{prefix}：影像未检出达标病灶，无法应用大小-标志物组合规则。"
    if large and (afp_high or dcp_high):
        return (
            f"{prefix}：病灶最大径 ≥2 cm 且 AFP>200 ng/mL 或 DCP>40 mAU/mL——"
            "符合高风险特征组合（研究口径），建议结合增强模式、病理与临床评估。"
        )
    if large or afp_high or dcp_high:
        return (
            f"{prefix}：检出单一高风险特征（大病灶或标志物显著升高）——"
            "倾向性证据有限，建议完整多期增强与病理评估。"
        )
    return (
        f"{prefix}：未见高风险组合（病灶 <2 cm 且标志物未超显著阈值）——"
        "倾向性提示较低，仍建议按临床路径随访。"
    )


def _joint_interpretation(
    imaging_metadata: str,
    lab_summary: list[dict[str, Any]],
    impression: str,
    guideline: str,
) -> str:
    """Controlled multi-modal joint interpretation (rules layer, not diagnosis)."""
    lesion_match = re.search(r"lesion_count=(\d+)", imaging_metadata)
    extent_match = re.search(r"max_extent_mm=([\d.]+)", imaging_metadata)
    lesion_count = int(lesion_match.group(1)) if lesion_match else 0
    max_extent_mm = float(extent_match.group(1)) if extent_match else None
    imaging_part = f"影像侧：{lesion_count} 个病灶"
    if max_extent_mm is not None:
        imaging_part += f"，最大径 {max_extent_mm:.1f} mm"
    lab_part = "；".join(
        f"{row['marker']} {row['latest_value']} {row['latest_unit'] or ''}"
        f"（{row['trajectory'] or '—'}）"
        for row in lab_summary
    )
    return (
        f"多模态联合解读（受控规则融合 · 研究口径，非诊断）：{imaging_part}；"
        f"检验侧：{lab_part or '无可用观测'}。"
        f"{impression} {guideline}"
    )


class _Handler(BaseHTTPRequestHandler):
    server_version = "OncoFuseVlmWeb/0.1"

    def _index(self) -> Path:
        return Path(self.server.index_path)  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "vlm-web",
                    "version": SERVICE_VERSION,
                    "started_at": getattr(self.server, "generated_at", ""),
                },
                status=200,
            )
            return
        if path != "/":
            self.send_error(404)
            return
        index = self._index()
        if not index.exists():
            self.send_error(500, f"index.html not found: {index}")
            return
        page = index.read_text(encoding="utf-8")
        started = getattr(self.server, "generated_at", "")
        version_text = f"vlm-web {SERVICE_VERSION} · 服务启动 {started}"
        page = page.replace(_VERSION_PLACEHOLDER, version_text)
        payload = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/interpret":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("Request body is empty or exceeds the size limit")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self._interpret(request)
        except Exception as exc:  # noqa: BLE001 - any upload/run failure maps to a JSON 400
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._send_json(result, status=200)

    def _interpret(self, request: dict[str, Any]) -> dict[str, Any]:
        files = request.get("files") or {}
        options = request.get("options") or {}
        ct = files.get("ct")
        mask = files.get("mask")
        labs = files.get("labs")
        if not ct or not mask or not labs:
            raise ValueError("ct, mask, and labs files are required")
        if "auditable" not in (options.get("fusion_modes") or ["auditable"]):
            raise ValueError("the auditable arm is the required main path")

        workdir = Path(tempfile.mkdtemp(prefix="oncofuse-vlm-web-"))
        try:
            ct_path = workdir / "ct.nii.gz"
            mask_path = workdir / "mask.nii.gz"
            ct_path.write_bytes(base64.b64decode(ct["data_b64"]))
            mask_path.write_bytes(base64.b64decode(mask["data_b64"]))
            if ct_path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ValueError("CT upload exceeds the size limit")
            if mask_path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ValueError("Mask upload exceeds the size limit")
            labs_path = workdir / "labs.txt"
            labs_path.write_text(str(labs.get("content") or ""), encoding="utf-8")
            labs_evidence = parse_laboratory_report(
                labs_path, patient_id=str(options.get("patient_label") or "CASE")
            )
            study_date = str(options.get("study_date") or "").strip()
            if not study_date:
                lab_dates = [
                    obs.observed_at
                    for analyte in labs_evidence.analytes.values()
                    for obs in analyte.observations
                    if obs.observed_at and obs.observed_at != "unknown"
                ]
                if lab_dates:
                    study_date = max(lab_dates)
            if not study_date:
                raise ValueError(
                    "Study date unknown; provide a study date or lab observations with dates"
                )
            phase = str(options.get("phase") or "").strip() or None
            output = workdir / "out"
            run_case_vlm(
                nifti_image=ct_path,
                nifti_mask=mask_path,
                labs_path=labs_path,
                study_date=study_date,
                phase=phase,
                output_dir=output,
                fusion_modes=tuple(options.get("fusion_modes") or ["auditable"]),
                max_tokens=int(options.get("max_tokens") or 1024),
                temperature=float(options.get("temperature") or 0.3),
                json_object=bool(options.get("json_object") or False),
                timeout_seconds=float(options.get("timeout_seconds") or 120.0),
            )
            web = json.loads((output / "web_demo.json").read_text(encoding="utf-8"))
            auditable = web.get("arms", {}).get("auditable", {})
            imaging_metadata = imaging_metadata_text(output / "imaging_evidence.json")
            lab_summary = _lab_summary(labs_evidence)
            impression = _deterministic_impression(imaging_metadata, lab_summary)
            guideline = _guideline_hint(imaging_metadata, lab_summary)
            slices = extract_axial_slice_previews(
                ct_path, mask_path, count=16, width=360
            )
            lesion_zoom = extract_lesion_zoom(ct_path, mask_path, width=360)
            tier = _risk_tier(imaging_metadata, lab_summary)
            lab_series = _lab_series(labs_evidence)
            auditable_report = auditable.get("report") or {}
            vis_lines = [
                line
                for line in (auditable_report.get("imaging_observations") or [])
                if "consistent with the provided" not in str(line).lower()
            ]
            clinical_report = _clinical_report(
                imaging_metadata,
                lab_summary,
                lab_series,
                vis_lines,
                tier["level"],
            )
            interpretation: dict[str, Any] = {
                "patient_label": str(options.get("patient_label") or "CASE"),
                "phase": phase or "unknown",
                "study_date": study_date,
                "clinical_history": str(options.get("history") or "").strip(),
                "imaging_measurements": imaging_metadata,
                "lab_summary": lab_summary,
                "lab_series": lab_series,
                "lab_change": _lab_change(lab_series),
                "timeline": _timeline(study_date, lab_series),
                "risk_tier": tier,
                "summary": _summary_text(imaging_metadata, lab_summary, tier),
                "next_steps": _next_steps(tier["level"]),
                "slices": slices,
                "lesion_zoom": lesion_zoom,
                "clinical_impression": impression,
                "guideline_hint": guideline,
                "clinical_report": clinical_report,
                "joint_interpretation": _joint_interpretation(
                    imaging_metadata, lab_summary, impression, guideline
                ),
                "auditable_report": auditable.get("report"),
                "audit_status": auditable.get("audit_status"),
                "audit_errors": [
                    item.get("message")
                    for item in (auditable.get("validation") or {}).get("errors", [])
                ],
                "fusion_modes": list(options.get("fusion_modes") or ["auditable"]),
                "disclaimer": RESEARCH_DISCLAIMER,
            }
            return {"ok": True, "interpretation": interpretation, "web_demo": web}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _send_json(self, payload: dict[str, Any], *, status: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class VisualizationServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        index_path: str | Path,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.index_path = str(index_path)
        self.generated_at = datetime.now(UTC).isoformat(timespec="seconds")


def launch_vlm_web(*, index_path: str | Path, port: int = 7861) -> None:
    """Serve the interpretable visualization page and its /api/interpret endpoint."""
    server = VisualizationServer(("127.0.0.1", port), _Handler, index_path)
    print(f"OncoFuse visualization service: http://127.0.0.1:{port}/")
    print("Upload CT + mask + lab report to get an auditable interpretation.")
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["launch_vlm_web"]
