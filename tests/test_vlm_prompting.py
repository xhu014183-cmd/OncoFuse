import json
from pathlib import Path

from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.vlm_prompting import (
    UNVERIFIED_CONTEXT_MARKER,
    VlmTaskPrompt,
    build_vlm_task_prompt,
    render_vlm_prompt_text,
    write_vlm_prompt_bundle,
)

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


def test_auditable_mode_withholds_labs(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="auditable")

    assert prompt.fusion_mode == "auditable"
    assert prompt.lab_context_injected is False
    assert prompt.clinical_context == []
    assert "85.3" not in prompt.prompt_text
    assert "AFP" not in prompt.prompt_text
    assert prompt.metadata["labs_present_but_withheld"] is True
    assert "no laboratory values were supplied to the model" in prompt.locked_constraints


def test_open_mode_injects_unverified_context_with_citations(tmp_path: Path):
    labs = _labs(tmp_path)
    prompt = build_vlm_task_prompt(labs=labs, fusion_mode="open")

    assert prompt.fusion_mode == "open"
    assert prompt.lab_context_injected is True
    assert prompt.metadata["unverified_context"] is True
    assert UNVERIFIED_CONTEXT_MARKER in prompt.prompt_text
    assert "85.3" in prompt.prompt_text
    assert "copy each value verbatim" in prompt.prompt_text
    lab_items = [item for item in prompt.clinical_context if item.category == "laboratory"]
    assert {item.evidence_id for item in lab_items} == {"LAB_001", "LAB_002"}
    assert all(
        item.source_ref and item.source_ref.startswith("clinical_labs:")
        for item in lab_items
    )
    assert any(
        "verbatim and cited from [LAB_###]" in constraint
        for constraint in prompt.locked_constraints
    )


def test_prompt_bundle_roundtrip_validates(tmp_path: Path):
    prompt = build_vlm_task_prompt(fusion_mode="open")
    json_path, text_path = write_vlm_prompt_bundle(
        prompt,
        json_path=tmp_path / "prompt.json",
        text_path=tmp_path / "prompt.txt",
    )

    assert json_path.exists()
    assert text_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["fusion_mode"] == "open"
    loaded = VlmTaskPrompt.model_validate(data)
    assert loaded.prompt_text == prompt.prompt_text
    assert "=== PROMPT ===" in render_vlm_prompt_text(prompt)


def test_legacy_prompt_json_gets_mode_defaults():
    legacy = {
        "task_version": "hcc-vlm-summary-v1",
        "task_name": "image_conditioned_structured_research_summary",
        "visual_token_count": 32,
        "prompt_text": "legacy body",
        "clinical_context": [],
        "output_fields": ["imaging_observations"],
        "locked_constraints": ["no diagnosis"],
        "metadata": {},
    }

    prompt = VlmTaskPrompt.model_validate(legacy)

    assert prompt.fusion_mode == "auditable"
    assert prompt.lab_context_injected is False


def test_attached_images_replace_placeholder_tokens():
    prompt = build_vlm_task_prompt(fusion_mode="open", image_count=2)

    assert "<im_patch>" not in prompt.prompt_text
    assert "Visual input: 2 attached image(s)" in prompt.prompt_text
    assert prompt.metadata["visual_input"] == "attached_images"


def test_imaging_metadata_injected_into_prompt():
    prompt = build_vlm_task_prompt(
        fusion_mode="auditable",
        imaging_metadata=(
            "Imaging measurements (deterministic): lesion_count=2; "
            "total_volume_ml=1.632"
        ),
    )

    assert "lesion_count=2" in prompt.prompt_text
    assert "1.632" in prompt.prompt_text
    assert prompt.metadata["imaging_metadata_present"] is True


def test_auditable_prompt_has_anti_repetition_and_visual_rules():
    prompt = build_vlm_task_prompt(
        fusion_mode="auditable",
        imaging_metadata=(
            "Imaging measurements (deterministic): lesion_count=1; "
            "total_volume_ml=12.808"
        ),
    )

    assert "Do not restate the deterministic imaging measurements" in prompt.prompt_text
    assert "do not state that they are missing" in prompt.prompt_text
    assert "Never repeat the same statement across fields" in prompt.prompt_text
    assert "mention it once in uncertainties" in prompt.prompt_text
