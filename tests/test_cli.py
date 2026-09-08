from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from hcc_multimodal.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hcc-demo", *argv])
    main()


def test_generate_command(tmp_path: Path, monkeypatch, capsys):
    out = tmp_path / "generated"
    _run(["generate", "--output", str(out)], monkeypatch)
    captured = capsys.readouterr().out
    assert (out / "baseline_ct.nii.gz").exists()
    assert (out / "labs.json").exists()
    assert "baseline_image" in captured


def test_run_demo_command(tmp_path: Path, monkeypatch):
    out = tmp_path / "demo"
    _run(["run-demo", "--output", str(out)], monkeypatch)
    assert (out / "clinical_verdict.json").exists()
    assert (out / "multimodal_case_evidence.json").exists()


def test_analyze_command(tmp_path: Path, monkeypatch):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "input")
    out = tmp_path / "analysis"
    _run(
        [
            "analyze",
            "--baseline-image", str(paths["baseline_image"]),
            "--baseline-mask", str(paths["baseline_mask"]),
            "--followup-image", str(paths["followup_image"]),
            "--followup-mask", str(paths["followup_mask"]),
            "--labs", str(paths["labs"]),
            "--patient-id", "DEMO_HCC_001",
            "--baseline-date", "2026-01-15",
            "--followup-date", "2026-07-15",
            "--imaging-origin", "synthetic",
            "--lab-origin", "synthetic",
            "--pairing-status", "same_subject",
            "--output", str(out),
        ],
        monkeypatch,
    )
    assert (out / "clinical_verdict.json").exists()
    assert (out / "lab_evidence.json").exists()


def test_parse_labs_command(tmp_path: Path, monkeypatch):
    out = tmp_path / "labs.json"
    _run(
        [
            "parse-labs",
            "--input", str(EXAMPLES / "lab_report.synthetic.txt"),
            "--patient-id", "CLI_001",
            "--output", str(out),
        ],
        monkeypatch,
    )
    assert out.exists()


def test_parse_hpi_command(tmp_path: Path, monkeypatch):
    out = tmp_path / "timeline.json"
    _run(
        [
            "parse-hpi",
            "--input", str(EXAMPLES / "hpi.synthetic.txt"),
            "--patient-id", "CLI_001",
            "--output", str(out),
        ],
        monkeypatch,
    )
    assert out.exists()


def test_analyze_case_accepts_unified_case_input(tmp_path: Path, monkeypatch):
    fixture = runpy.run_path(str(EXAMPLES / "generate_dicom_seg_fixture.py"))
    study_dir, seg_path = fixture["generate"](tmp_path / "fixture")
    case_path = tmp_path / "case.json"
    output = tmp_path / "case-output"
    case_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1.0",
                "case_id": "CLI_CASE_001",
                "patient_id": "RESEARCH_001",
                "clinical_task": "diagnostic_workup",
                "index_date": "2026-07-20",
                "imaging": {
                    "dicom_dir": str(study_dir),
                    "seg": str(seg_path),
                    "phase": "portal_venous",
                },
                "laboratory": {
                    "source_type": "inline",
                    "inline": {
                        "format": "text",
                        "content": "2026-07-20 AFP 96 ng/mL 0-7\n2026-07-20 DCP 80 mAU/mL 0-40",
                    },
                },
                "output_dir": str(output),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _run(["analyze-case", "--case-input", str(case_path)], monkeypatch)
    assert (output / "case-input.normalized.json").exists()
    assert (output / "case-summary.json").exists()
    assert (output / "case-summary.md").exists()


@pytest.mark.parametrize(
    ("schema_type", "template"),
    [
        ("research-protocol", "research_protocol.template.json"),
        ("research-cohort-manifest", "research_manifest.template.json"),
        ("adjudication-set", "adjudications.template.json"),
    ],
)
def test_validate_json_command(tmp_path: Path, monkeypatch, schema_type: str, template: str):
    _run(
        [
            "validate-json",
            "--type", schema_type,
            "--input", str(EXAMPLES / template),
        ],
        monkeypatch,
    )


def test_export_schemas_command(tmp_path: Path, monkeypatch):
    out = tmp_path / "schemas"
    _run(["export-schemas", "--output", str(out)], monkeypatch)
    assert out.exists()
    assert any(out.glob("*.schema.json"))


def test_run_report_command_and_strict_exit(tmp_path: Path, monkeypatch):
    from hcc_multimodal.synthetic import generate_synthetic_case

    paths = generate_synthetic_case(tmp_path / "input")
    output = tmp_path / "report"
    case_path = tmp_path / "case-1.2.json"
    case_path.write_text(
        json.dumps(
            {
                "schema_version": "1.2.0",
                "case_id": "CLI_REPORT_001",
                "patient_id": "DEMO_HCC_001",
                "clinical_task": "recurrence_surveillance",
                "index_date": "2026-07-15",
                "data_relationship": {
                    "imaging_origin": "synthetic",
                    "laboratory_origin": "synthetic",
                    "pairing_status": "same_subject",
                    "statement": "synthetic CLI fixture",
                },
                "imaging": {
                    "source_type": "nifti",
                    "nifti_image": str(paths["followup_image"]),
                    "dicom_dir": None,
                    "seg": str(paths["followup_mask"]),
                    "seg_role": "user_supplied",
                    "study_date": "2026-07-15",
                    "phase": "portal_venous",
                },
                "laboratory": {
                    "source_type": "file",
                    "file_path": str(paths["labs"]),
                },
            }
        ),
        encoding="utf-8",
    )
    _run(
        [
            "run-report",
            "--case-input",
            str(case_path),
            "--output",
            str(output),
        ],
        monkeypatch,
    )
    assert (output / "controlled-report.json").exists()
    assert (output / "pipeline-audit.json").exists()

    strict_output = tmp_path / "strict-report"
    with pytest.raises(SystemExit) as exc:
        _run(
            [
                "run-report",
                "--case-input",
                str(case_path),
                "--output",
                str(strict_output),
                "--require-live-models",
            ],
            monkeypatch,
        )
    assert exc.value.code == 2
    assert (strict_output / "pipeline-audit.json").exists()
