from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from hcc_multimodal.report_pipeline import run_report_pipeline


def _case(tmp_path: Path, *, with_mask: bool = True) -> Path:
    image = np.zeros((24, 24, 24), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[5:12, 5:12, 5:12] = 1
    image_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    payload = {
        "schema_version": "1.2.0",
        "case_id": "PIPELINE_001",
        "patient_id": "PIPELINE_001",
        "clinical_task": "recurrence_surveillance",
        "index_date": "2026-07-20",
        "data_relationship": {
            "imaging_origin": "synthetic",
            "laboratory_origin": "synthetic",
            "pairing_status": "same_subject",
            "statement": "fully synthetic same-case software fixture",
        },
        "imaging": {
            "source_type": "nifti",
            "nifti_image": image_path.name,
            "dicom_dir": None,
            "seg": mask_path.name if with_mask else None,
            "seg_role": "user_supplied" if with_mask else None,
            "study_date": "2026-07-20",
            "phase": "portal_venous",
        },
        "laboratory": {
            "source_type": "inline",
            "inline": {
                "format": "text",
                "content": (
                    "2026-01-20 AFP 6 ng/mL 0-7\n"
                    "2026-07-20 AFP 80 ng/mL 0-7\n"
                    "2026-01-20 DCP 20 mAU/mL 0-40\n"
                    "2026-07-20 DCP 60 mAU/mL 0-40"
                ),
            },
        },
        "output_dir": "case-output",
    }
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    return case_path


def test_deterministic_pipeline_writes_fixed_artifacts(tmp_path: Path):
    output = tmp_path / "out"
    result = run_report_pipeline(_case(tmp_path), output_dir=output)
    expected = {
        "case-input.normalized.json",
        "imaging-input-qc.json",
        "lion-inspired-evidence.json",
        "glm-imaging-evidence.json",
        "imaging-crosscheck.json",
        "clinical-lab-evidence.json",
        "clinical-verdict.json",
        "deepseek-prompt.json",
        "deepseek-narrative-prompt.json",
        "deepseek-narrative.json",
        "controlled-report.json",
        "controlled-report.md",
        "pipeline-audit.json",
        "web_demo.json",
        "provider-audit/glm.json",
        "provider-audit/deepseek.json",
    }
    assert all((output / name).is_file() for name in expected)
    assert result.strict_success is True
    prompt_text = (output / "deepseek-prompt.json").read_text(encoding="utf-8")
    assert str((tmp_path / "ct.nii.gz").resolve()) not in prompt_text
    assert "Single-phase CT" in prompt_text


def test_pipeline_without_seg_does_not_emit_zero_lesions(tmp_path: Path):
    output = tmp_path / "out"
    run_report_pipeline(_case(tmp_path, with_mask=False), output_dir=output)
    lion = json.loads((output / "lion-inspired-evidence.json").read_text())
    assert lion["patient_evidence"]["lesion_count"] is None
    assert lion["status"] == "unavailable"


def test_live_provider_failure_degrades_and_strict_status_fails(monkeypatch, tmp_path: Path):
    def fail_deepseek(*_, **__):
        raise RuntimeError("offline")

    monkeypatch.setenv("ZHIPU_API_KEY", "")
    monkeypatch.setattr(
        "hcc_multimodal.report_pipeline.call_deepseek",
        fail_deepseek,
    )
    result = run_report_pipeline(
        _case(tmp_path),
        output_dir=tmp_path / "out",
        glm_mode="live",
        report_mode="live",
        require_live_models=True,
        timeout_seconds=0.1,
    )
    assert result.degraded is True
    assert result.strict_success is False
    audit = json.loads((result.output_dir / "pipeline-audit.json").read_text())
    assert audit["status"] == "degraded"
    assert (result.output_dir / "controlled-report.json").exists()


def test_distill_pipeline_adds_only_validated_optional_narrative(
    monkeypatch, tmp_path: Path
):
    narrative = (
        "影像证据显示存在分割病灶负荷，检验标志物趋势形成另一侧研究信号。"
        "由于公开影像与合成检验属于未配对演示，二者不能解释为同一患者结论，"
        "现有信息仍有明显限制并需要人工复核。"
    )
    calls = 0

    def fake_narrative(*_, **__):
        nonlocal calls
        calls += 1
        content = (
            "影像与检验包含20个未经允许的数字，仍需复核。"
            if calls == 1
            else narrative
        )
        return {
            "provider": "openai-compatible",
            "model": "deepseek-r1-distill-qwen-32b",
            "usage": {},
            "finish_reason": "stop",
            "raw_content": content,
            "reasoning_content_present": True,
            "thinking_mode": "required",
            "provider_dialect": "dashscope",
            "json_object_requested": False,
            "configuration_source": "legacy_llm",
            "prompt_version": "deepseek-hcc-narrative-v1",
            "attempt_count": 1,
            "validation_retry_count": 0,
        }

    def fail_structured(*_, **__):
        raise AssertionError("R1 Distill must not enter the controlled JSON renderer")

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-r1-distill-qwen-32b")
    monkeypatch.setattr(
        "hcc_multimodal.report_pipeline.call_deepseek_narrative", fake_narrative
    )
    monkeypatch.setattr(
        "hcc_multimodal.report_pipeline.call_deepseek", fail_structured
    )
    result = run_report_pipeline(
        _case(tmp_path),
        output_dir=tmp_path / "out",
        report_mode="live",
    )
    evidence = json.loads(
        (result.output_dir / "deepseek-narrative.json").read_text(encoding="utf-8")
    )
    markdown = (result.output_dir / "controlled-report.md").read_text(
        encoding="utf-8"
    )
    assert result.deepseek_live_success is True
    assert calls == 2
    assert evidence["status"] == "pass"
    assert evidence["narrative"] == narrative
    assert "## AI辅助解读" in markdown
    assert narrative in markdown
    provider_audit = json.loads(
        (result.output_dir / "provider-audit" / "deepseek.json").read_text(
            encoding="utf-8"
        )
    )
    assert provider_audit["attempt_count"] == 2
    assert provider_audit["validation_retry_count"] == 1
