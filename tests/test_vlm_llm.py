from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.schemas import VlmDemoReport
from hcc_multimodal.vlm_llm import (
    _dedupe_report_fields,
    build_numeric_citations,
    build_vlm_template_report,
    call_vlm_llm,
    run_vlm_dual_arm_demo,
    validate_vlm_report,
)
from hcc_multimodal.vlm_prompting import RESEARCH_DISCLAIMER, build_vlm_task_prompt

LAB_TEXT = """2026-01-15 AFP 6.0 ng/mL 0-7
2026-04-15 AFP 18.0 ng/mL 0-7
2026-07-15 AFP 85.3 ng/mL 0-7
2026-01-15 DCP 25.0 mAU/mL 0-40
2026-07-15 DCP 68.0 mAU/mL 0-40
"""


def _labs(tmp_path: Path):
    path = tmp_path / "labs.txt"
    path.write_text(LAB_TEXT, encoding="utf-8")
    return parse_laboratory_report(path, patient_id="DEMO_001")


def test_template_open_passes_and_attaches_citations(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)
    report["numeric_citations"] = validation["numeric_citations"]

    assert validation["valid"] is True
    cited_ids = {item["evidence_id"] for item in report["numeric_citations"]}
    assert {"LAB_001", "LAB_002"} <= cited_ids
    assert any(item["value"] == 85.3 for item in report["numeric_citations"])
    assert VlmDemoReport.model_validate(report).fusion_mode == "open"


def test_template_auditable_passes_without_numbers(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="auditable")
    report = build_vlm_template_report(prompt)

    validation = validate_vlm_report(report, fusion_mode="auditable", prompt=prompt)

    assert validation["valid"] is True
    assert validation["numeric_citations"] == []
    assert validation["unattributed_numbers"] == []


def test_invented_number_is_blocked(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["clinical_context_summary"].append(
        "AFP projected to exceed 170 ng/mL within 30 days"
    )

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    assert validation["valid"] is False
    assert "EVIDENCE_VALUE_TAMPERED" in {item["code"] for item in validation["errors"]}


def test_auditable_arm_rejects_lab_echo(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="auditable")
    report = build_vlm_template_report(prompt)
    report["clinical_context_summary"].append("AFP 85.3 ng/mL")

    validation = validate_vlm_report(report, fusion_mode="auditable", prompt=prompt)

    assert validation["valid"] is False
    assert "EVIDENCE_VALUE_TAMPERED" in {item["code"] for item in validation["errors"]}


def test_reformatted_date_is_not_false_tampering(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["clinical_context_summary"] = [
        "[LAB_001] AFP 85.3 ng/mL on July 15, 2026 (persistent rising)"
    ]

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    codes = {item["code"] for item in validation["errors"]}
    assert "EVIDENCE_VALUE_TAMPERED" not in codes
    assert validation["valid"] is True


def test_iso_date_echo_in_report_is_not_false_tampering(tmp_path: Path):
    labs_path = tmp_path / "labs.txt"
    labs_path.write_text(
        "1998-09-30 AFP 6 ng/mL 0-7\n1998-12-29 AFP 85.3 ng/mL 0-7\n",
        encoding="utf-8",
    )
    labs = parse_laboratory_report(labs_path, patient_id="CASE")
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["clinical_context_summary"] = [
        "[LAB_001] AFP level at 85.3 ng/mL on 1998-12-29 (persistent rising)"
    ]

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    codes = {item["code"] for item in validation["errors"]}
    assert "EVIDENCE_VALUE_TAMPERED" not in codes
    assert validation["valid"] is True


def test_dedupe_report_fields_removes_repeats_order_preserving():
    report = {
        "imaging_observations": ["A finding", "a finding", "B finding", "B finding"],
        "uncertainties": ["phase unknown", "phase unknown", "other"],
        "missing_information": ["only once"],
    }

    _dedupe_report_fields(report)

    assert report["imaging_observations"] == ["A finding", "B finding"]
    assert report["uncertainties"] == ["phase unknown", "other"]
    assert report["missing_information"] == ["only once"]


def test_diagnostic_assertion_is_blocked(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["evidence_concordance"] = "This confirms HCC cancer."

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    assert validation["valid"] is False
    assert "DIAGNOSTIC_ASSERTION" in {item["code"] for item in validation["errors"]}


def test_soft_diagnostic_phrase_is_warning_not_blocked(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["clinical_context_summary"].append(
        "The lesion is consistent with hepatocellular carcinoma."
    )

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    assert validation["valid"] is True
    assert any(
        item["code"] == "SOFT_DIAGNOSTIC_ASSERTION"
        for item in validation["soft_warnings"]
    )


def test_disclaimer_mismatch_is_blocked(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["research_disclaimer"] = "not the fixed disclaimer"

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    assert validation["valid"] is False
    assert "LOCKED_FIELD_MISMATCH" in {item["code"] for item in validation["errors"]}


def test_missing_fusion_mode_is_machine_set(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    del report["fusion_mode"]

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    assert validation["valid"] is True
    assert report["fusion_mode"] == "open"


def test_wrong_fusion_mode_echo_is_blocked(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["fusion_mode"] = "auditable"

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    assert validation["valid"] is False
    assert "LOCKED_FIELD_MISMATCH" in {item["code"] for item in validation["errors"]}


def test_build_numeric_citations_matches_lab_items(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = {
        "imaging_observations": [],
        "clinical_context_summary": ["AFP eq 85.3 ng/mL [LAB_001]"],
        "uncertainties": [],
        "missing_information": [],
        "evidence_concordance": "insufficient_evidence",
        "image_conditioning_statement": "no statement",
    }

    citations, unattributed = build_numeric_citations(report, prompt)

    assert len(citations) == 1
    assert citations[0].evidence_id == "LAB_001"
    assert citations[0].analyte == "AFP"
    assert citations[0].unit == "ng/mL"
    assert unattributed == []


def _tiny_png(path: Path) -> Path:
    Image.new("RGB", (8, 8), color=(40, 40, 40)).save(path)
    return path


def test_string_fields_do_not_produce_character_noise(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = {
        "imaging_observations": [],
        "clinical_context_summary": "[LAB_001] AFP 85.3 ng/mL; unknown dates for [LAB_007]",
        "uncertainties": "no dates for [LAB_007]",
        "missing_information": "",
        "evidence_concordance": "insufficient_evidence",
        "image_conditioning_statement": "no statement",
    }

    citations, unattributed = build_numeric_citations(report, prompt)

    assert [citation.value for citation in citations] == [85.3]
    assert unattributed == []


def test_string_fields_block_structure_without_fake_tampering(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    report = build_vlm_template_report(prompt)
    report["clinical_context_summary"] = "[LAB_001] AFP 85.3 ng/mL; [LAB_007] unknown"

    validation = validate_vlm_report(report, fusion_mode="open", prompt=prompt)

    codes = {item["code"] for item in validation["errors"]}
    assert "SCHEMA_INVALID" in codes
    assert "EVIDENCE_VALUE_TAMPERED" not in codes


def test_run_dual_arm_demo_falls_back_and_writes_artifacts(tmp_path: Path, monkeypatch):
    labs = _labs(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("External LLM configuration is unavailable")

    monkeypatch.setattr("hcc_multimodal.vlm_llm.call_vlm_llm", unavailable)
    comparison_path = run_vlm_dual_arm_demo(
        labs=labs,
        output_dir=tmp_path / "out",
        fusion_modes=("auditable", "open"),
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["arms"]["auditable"]["renderer"] == "deterministic_template"
    assert comparison["arms"]["open"]["renderer"] == "deterministic_template"
    assert comparison["arms"]["auditable"]["audit_status"] == "pass"
    assert comparison["arms"]["open"]["audit_status"] == "pass"
    assert comparison["open_numeric_citation_count"] >= 2
    assert (tmp_path / "out" / "vlm_arm_auditable.json").exists()
    assert (tmp_path / "out" / "vlm_arm_open.json").exists()
    assert (tmp_path / "out" / "dual_arm_comparison.json").exists()


def test_run_dual_arm_demo_external_llm_passes(tmp_path: Path, monkeypatch):
    labs = _labs(tmp_path)

    def fake_call(prompt, **_kwargs):
        report = build_vlm_template_report(prompt)
        return {
            "report": report,
            "raw_content": json.dumps(report),
            "provider": "test",
            "model": "test-model",
        }

    monkeypatch.setattr("hcc_multimodal.vlm_llm.call_vlm_llm", fake_call)
    comparison_path = run_vlm_dual_arm_demo(
        labs=labs,
        output_dir=tmp_path / "out",
        fusion_modes=("auditable", "open"),
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["arms"]["auditable"]["renderer"] == "external_llm"
    assert comparison["arms"]["open"]["renderer"] == "external_llm"
    assert comparison["arms"]["auditable"]["audit_status"] == "pass"
    assert comparison["arms"]["open"]["audit_status"] == "pass"
    open_report = json.loads(
        (tmp_path / "out" / "vlm_arm_open.json").read_text(encoding="utf-8")
    )["report"]
    assert RESEARCH_DISCLAIMER in open_report["research_disclaimer"]
    assert len(open_report["numeric_citations"]) >= 2


class _ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.last_body = json.loads(body.decode("utf-8"))
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(self.server.report),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_call_vlm_llm_http_round_trip(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")
    png = _tiny_png(tmp_path / "slice.png")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    server.report = build_vlm_template_report(prompt)
    server.last_body = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = call_vlm_llm(
            prompt,
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_address[1]}",
            model="test-model",
            temperature=0.3,
            json_object=False,
            image_paths=[png],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["provider"] == "openai-compatible"
    assert result["model"] == "test-model"
    assert result["report"]["fusion_mode"] == "open"
    assert result["usage"]["total_tokens"] == 2
    assert server.last_body["temperature"] == 0.3
    assert "response_format" not in server.last_body
    content = server.last_body["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == prompt.prompt_text
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_run_dual_arm_demo_with_images_and_imaging_metadata(
    tmp_path: Path, monkeypatch
):
    labs = _labs(tmp_path)
    png = _tiny_png(tmp_path / "slice.png")
    captured: dict = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        captured["image_paths"] = _kwargs.get("image_paths")
        report = build_vlm_template_report(prompt)
        return {
            "report": report,
            "raw_content": json.dumps(report),
            "provider": "test",
            "model": "test-model",
        }

    monkeypatch.setattr("hcc_multimodal.vlm_llm.call_vlm_llm", fake_call)
    run_vlm_dual_arm_demo(
        labs=labs,
        output_dir=tmp_path / "out",
        fusion_modes=("open",),
        image_paths=[png],
        imaging_metadata=(
            "Imaging measurements (deterministic): lesion_count=2; "
            "total_volume_ml=1.632"
        ),
    )

    assert captured["image_paths"] == [png]
    prompt = captured["prompt"]
    assert prompt.metadata["visual_input"] == "attached_images"
    assert "<im_patch>" not in prompt.prompt_text
    assert "1.632" in prompt.prompt_text
