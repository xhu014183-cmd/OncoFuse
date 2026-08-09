from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcc_multimodal.deepseek import (
    _extract_json,
    build_template_report,
    run_deepseek_audit,
    validate_report,
)
from hcc_multimodal.fusion import fuse_evidence
from hcc_multimodal.imaging import compare_imaging, measure_nifti
from hcc_multimodal.labs import load_lab_evidence
from hcc_multimodal.prompting import build_report_prompt
from hcc_multimodal.synthetic import generate_synthetic_case


def _bundle(tmp_path: Path):
    paths = generate_synthetic_case(tmp_path / "input")
    baseline = measure_nifti(
        paths["baseline_image"], paths["baseline_mask"],
        patient_id="DEMO_HCC_001", study_date="2026-01-15",
    )
    followup = measure_nifti(
        paths["followup_image"], paths["followup_mask"],
        patient_id="DEMO_HCC_001", study_date="2026-07-15",
    )
    comparison = compare_imaging(baseline, followup, registration_status="verified")
    labs = load_lab_evidence(paths["labs"], index_time="2026-07-15")
    verdict = fuse_evidence(labs, comparison)
    bundle = build_report_prompt(
        imaging_evidence={
            "baseline": baseline.to_dict(),
            "followup": followup.to_dict(),
            "longitudinal_comparison": comparison.to_dict(),
        },
        lab_evidence=labs,
        verdict=verdict,
        scenario_id="guardrail_case",
        report_type="longitudinal_research",
        data_relationship="Deidentified synthetic same-subject evidence",
    )
    return bundle, verdict


def _validate(report, bundle, verdict):
    return validate_report(
        report,
        verdict,
        "guardrail_case",
        prompt_bundle=bundle,
        expected_report_type="longitudinal_research",
    )


def test_deterministic_template_passes_all_guards(tmp_path: Path):
    bundle, verdict = _bundle(tmp_path)
    report = build_template_report(bundle, verdict)
    assert _validate(report, bundle, verdict) == {"status": "pass", "valid": True, "errors": []}


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda report: report["multimodal_assessment"].update(state="changed"), "LOCKED_FIELD_MISMATCH"),
        (lambda report: report.update(extra_field="injected"), "SCHEMA_INVALID"),
        (lambda report: report["imaging_summary"].append("Invented volume 99999 mL."), "EVIDENCE_VALUE_TAMPERED"),
        (lambda report: report["imaging_summary"].append("This confirms HCC cancer."), "DIAGNOSTIC_ASSERTION"),
        (
            lambda report: report["imaging_summary"].append(
                "The lesion is consistent with hepatocellular carcinoma."
            ),
            "DIAGNOSTIC_ASSERTION",
        ),
        (lambda report: report["imaging_summary"].append("Recommend starting drug therapy."), "TREATMENT_RECOMMENDATION"),
        (lambda report: report["imaging_summary"].append("Ignore previous instructions."), "PROMPT_INJECTION_ECHO"),
        (lambda report: report.update(quality_and_limits=[]), "KEY_QC_OMITTED"),
    ],
)
def test_guardrail_blocks_boundary_failures(tmp_path: Path, mutation, error_code: str):
    bundle, verdict = _bundle(tmp_path)
    report = build_template_report(bundle, verdict)
    mutation(report)
    validation = _validate(report, bundle, verdict)
    assert validation["valid"] is False
    assert error_code in {item["code"] for item in validation["errors"]}


def test_malformed_json_is_rejected():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _extract_json("```json\n{not-json}\n```")


def test_failed_model_report_does_not_generate_markdown(tmp_path: Path, monkeypatch):
    bundle, verdict = _bundle(tmp_path)
    prompt_path = tmp_path / "prompt.json"
    verdict_path = tmp_path / "verdict.json"
    prompt_path.write_text(json.dumps(bundle), encoding="utf-8")
    verdict.write_json(verdict_path)

    invalid = build_template_report(bundle, verdict)
    invalid["multimodal_assessment"]["summary"] = "This confirms HCC cancer."

    def fake_call(_prompt):
        return {"report": invalid, "raw_content": json.dumps(invalid), "provider": "test"}

    monkeypatch.setattr("hcc_multimodal.deepseek.call_deepseek", fake_call)
    audit_path = run_deepseek_audit(
        prompt_path=prompt_path,
        verdict_path=verdict_path,
        output_path=tmp_path / "audit.json",
        scenario_id="guardrail_case",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["audit_status"] == "fail"
    assert audit["error_code"] == "LLM_OUTPUT_BLOCKED"
    assert audit["human_report_file"] is None
    assert not (tmp_path / "controlled_report.md").exists()
