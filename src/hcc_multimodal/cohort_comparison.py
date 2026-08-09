"""Five-case dual-mode VLM comparison over synthetic laboratory scenarios."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .clinical_labs import parse_laboratory_report
from .schemas import PIPELINE_VERSION, SCHEMA_VERSION
from .vlm_llm import imaging_metadata_text, run_vlm_dual_arm_demo

CASE_LAB_SCENARIOS: dict[str, dict[str, Any]] = {
    "dual_marker_rising": {
        "description": "AFP and DCP both rise from normal to above the supplied upper reference",
        "observations": [
            (-90, "AFP", 6.0, "ng/mL", 7.0),
            (0, "AFP", 85.3, "ng/mL", 7.0),
            (-90, "DCP", 25.0, "mAU/mL", 40.0),
            (0, "DCP", 68.0, "mAU/mL", 40.0),
        ],
    },
    "markers_normal": {
        "description": "AFP and DCP remain within their supplied reference ranges",
        "observations": [
            (-90, "AFP", 5.5, "ng/mL", 7.0),
            (0, "AFP", 6.0, "ng/mL", 7.0),
            (-90, "DCP", 20.0, "mAU/mL", 40.0),
            (0, "DCP", 22.0, "mAU/mL", 40.0),
        ],
    },
    "rebound_discordant": {
        "description": (
            "AFP rebounded to high after a treatment-era nadir while DCP stays "
            "low, creating discordant markers and imaging-laboratory tension"
        ),
        "observations": [
            (-90, "AFP", 320.0, "ng/mL", 7.0),
            (-45, "AFP", 6.5, "ng/mL", 7.0),
            (0, "AFP", 96.0, "ng/mL", 7.0),
            (-90, "DCP", 380.0, "mAU/mL", 40.0),
            (-45, "DCP", 25.0, "mAU/mL", 40.0),
            (0, "DCP", 30.0, "mAU/mL", 40.0),
        ],
    },
}


def _scenario_text(scenario: dict[str, Any], study_date: str) -> str:
    base = date.fromisoformat(study_date)
    lines: list[str] = []
    for offset_days, marker, value, unit, upper in scenario["observations"]:
        observed = (base + timedelta(days=offset_days)).isoformat()
        lines.append(f"{observed} {marker} {value:g} {unit} 0-{upper:g}")
    return "\n".join(lines) + "\n"


def _case_dir(case: str, cohort_dir: str | Path, hcc003_dir: str | Path) -> Path:
    directory = (
        Path(hcc003_dir)
        if case == "HCC_003"
        else Path(cohort_dir) / case
    )
    if not (directory / "public_imaging_evidence.json").exists():
        raise ValueError(
            f"Case {case} is missing public_imaging_evidence.json in {directory}"
        )
    return directory


def _overlay_path(case_dir: Path, case: str) -> Path:
    candidates = [
        case_dir / "converted" / f"{case}_overlay.png",
        case_dir / "converted" / "hcc003_overlay.png",
        case_dir / "public-case" / "converted" / "hcc003_overlay.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No overlay PNG found under {case_dir / 'converted'}")


def run_cohort_comparison(
    cases: list[str],
    *,
    scenarios: list[str] | None = None,
    cohort_dir: str | Path,
    hcc003_dir: str | Path,
    fusion_modes: tuple[Literal["auditable", "open"], ...] = ("auditable", "open"),
    max_tokens: int = 1024,
    temperature: float = 0.3,
    json_object: bool = False,
    timeout_seconds: float = 120.0,
    rerun: bool = False,
) -> Path:
    """Run the dual-mode VLM comparison for cases x scenarios.

    Each case x scenario runs both arms through the real LLM and writes its
    artifacts under ``<case>/scenarios/<scenario>/vlm-live/``. Completed runs
    are reused unless ``rerun`` is set, so the function resumes after
    interruptions. Returns the path to the cohort-level comparison summary.
    """
    scenario_ids = list(scenarios or CASE_LAB_SCENARIOS)
    unknown = [item for item in scenario_ids if item not in CASE_LAB_SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")

    records: list[dict[str, Any]] = []
    for case in cases:
        case_dir = _case_dir(case, cohort_dir, hcc003_dir)
        evidence = json.loads(
            (case_dir / "public_imaging_evidence.json").read_text(encoding="utf-8")
        )
        study_date = evidence.get("study_date") or "unknown"
        overlay = _overlay_path(case_dir, case)
        imaging_metadata = imaging_metadata_text(
            case_dir / "public_imaging_evidence.json"
        )
        for scenario in scenario_ids:
            scenario_dir = case_dir / "scenarios" / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            live_dir = scenario_dir / "vlm-live"
            comparison_path = live_dir / "dual_arm_comparison.json"
            if not comparison_path.exists() or rerun:
                labs_path = scenario_dir / "labs.txt"
                labs_path.write_text(
                    _scenario_text(CASE_LAB_SCENARIOS[scenario], study_date),
                    encoding="utf-8",
                )
                labs = parse_laboratory_report(labs_path, patient_id=case)
                run_vlm_dual_arm_demo(
                    labs=labs,
                    fusion_modes=fusion_modes,
                    output_dir=live_dir,
                    image_paths=[overlay],
                    imaging_metadata=imaging_metadata,
                    phase="unknown",
                    timepoint="single",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_object=json_object,
                    timeout_seconds=timeout_seconds,
                )
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            auditable = comparison.get("arms", {}).get("auditable", {})
            open_arm = comparison.get("arms", {}).get("open", {})
            records.append(
                {
                    "patient_id": case,
                    "study_date": study_date,
                    "scenario": scenario,
                    "scenario_description": CASE_LAB_SCENARIOS[scenario][
                        "description"
                    ],
                    "auditable_audit_status": auditable.get("audit_status"),
                    "auditable_renderer": auditable.get("renderer"),
                    "open_audit_status": open_arm.get("audit_status"),
                    "open_renderer": open_arm.get("renderer"),
                    "open_numeric_citations": comparison.get(
                        "open_numeric_citation_count"
                    ),
                    "auditable_numeric_citations": comparison.get(
                        "auditable_numeric_citation_count"
                    ),
                    "hallucination_candidates": comparison.get(
                        "hallucination_candidates", []
                    ),
                    "diff_open_only": len(
                        comparison.get("diff", {}).get("open_only", [])
                    ),
                    "diff_auditable_only": len(
                        comparison.get("diff", {}).get("auditable_only", [])
                    ),
                    "live_dir": str(live_dir),
                }
            )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": None,
        "cases": cases,
        "scenarios": scenario_ids,
        "runs": records,
        "run_count": len(records),
        "both_pass_count": sum(
            1
            for item in records
            if item["auditable_audit_status"] == "pass"
            and item["open_audit_status"] == "pass"
        ),
        "auditable_pass_count": sum(
            1
            for item in records
            if item["auditable_audit_status"] == "pass"
        ),
        "open_pass_count": sum(
            1 for item in records if item["open_audit_status"] == "pass"
        ),
        "total_hallucination_candidates": sum(
            len(item["hallucination_candidates"]) for item in records
        ),
        "total_open_numeric_citations": sum(
            int(item["open_numeric_citations"] or 0) for item in records
        ),
    }
    first_model = next(
        (
            json.loads(
                (Path(item["live_dir"]) / "vlm_arm_open.json").read_text(
                    encoding="utf-8"
                )
            ).get("model")
            for item in records
            if (Path(item["live_dir"]) / "vlm_arm_open.json").exists()
        ),
        None,
    )
    summary["model"] = first_model
    summary_path = Path(cohort_dir) / "cohort_comparison.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


__all__ = ["CASE_LAB_SCENARIOS", "run_cohort_comparison"]
