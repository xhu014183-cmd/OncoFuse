from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import nibabel as nib
import numpy as np
import pytest

from hcc_multimodal.glm_vision import (
    GlmResponseFormatError,
    build_glm_prompt,
    call_zhipu_glm,
    create_glm_preview_images,
    run_glm_observer,
    validate_glm_response,
)
from hcc_multimodal.lion_inspired import PrecomputedMaskLionBackend


def _image_mask(tmp_path: Path) -> tuple[Path, Path]:
    image = np.linspace(-300, 300, 20**3, dtype=np.float32).reshape((20, 20, 20))
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[6:13, 6:13, 6:13] = 1
    image_path = tmp_path / "patient_CASE_SECRET_ct.nii.gz"
    mask_path = tmp_path / "patient_CASE_SECRET_mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    return image_path, mask_path


def test_preview_output_is_png_without_source_filename(tmp_path: Path):
    image, mask = _image_mask(tmp_path)
    paths, refs = create_glm_preview_images(image, mask, tmp_path / "preview")
    assert paths
    assert all(path.suffix == ".png" for path in paths)
    assert all("CASE_SECRET" not in path.name for path in paths)
    assert refs[0] == "IMAGE_MONTAGE"


def test_zhipu_request_contains_only_png_data_urls_and_prompt(monkeypatch, tmp_path: Path):
    image, mask = _image_mask(tmp_path)
    paths, _ = create_glm_preview_images(image, mask, tmp_path / "preview")
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "observations": [],
                                        "uncertainties": [],
                                        "missing_information": [],
                                        "image_conditioning_statement": "images reviewed",
                                    }
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request: Request, timeout: float):
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("hcc_multimodal.glm_vision.urlopen", fake_urlopen)
    call_zhipu_glm(
        prompt="image-only prompt",
        image_paths=paths,
        api_key="test-key",
        base_url="https://example.invalid/v4",
        model="glm-test",
    )
    content = captured["body"]["messages"][0]["content"]
    image_parts = [item for item in content if item["type"] == "image_url"]
    assert image_parts
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,")
        for item in image_parts
    )
    serialized = json.dumps(captured["body"])
    assert "CASE_SECRET" not in serialized
    assert ".nii" not in serialized


def test_glm_validator_blocks_unknown_id_number_and_diagnosis():
    response = {
        "observations": [
            {
                "finding_id": "FINDING_001",
                "lesion_id": "LESION_999",
                "location": "segment 6",
                "observation": "This is HCC measuring 20 mm",
                "confidence": "high",
                "source_refs": ["IMAGE_MONTAGE"],
            }
        ],
        "uncertainties": [],
        "missing_information": [],
        "image_conditioning_statement": "image reviewed",
    }
    parsed, errors = validate_glm_response(
        response,
        allowed_lesion_ids={"LESION_001"},
        allowed_source_refs={"IMAGE_MONTAGE"},
    )
    assert parsed is None
    codes = {item["code"] for item in errors}
    assert "UNKNOWN_LESION_ID" in codes
    assert "UNSUPPORTED_NUMERIC_OBSERVATION" in codes


def test_zhipu_multiple_json_objects_are_blocked(monkeypatch, tmp_path: Path):
    image, mask = _image_mask(tmp_path)
    paths, _ = create_glm_preview_images(image, mask, tmp_path / "preview")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"observations": []}\n{"echo": true}'},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(
        "hcc_multimodal.glm_vision.urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(GlmResponseFormatError):
        call_zhipu_glm(
            prompt="image-only",
            image_paths=paths,
            api_key="test-key",
            base_url="https://example.invalid/v4",
            model="glm-test",
        )


def test_zhipu_retries_transient_429_then_succeeds(monkeypatch, tmp_path: Path):
    image, mask = _image_mask(tmp_path)
    paths, _ = create_glm_preview_images(image, mask, tmp_path / "preview")
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "observations": [],
                                        "uncertainties": [],
                                        "missing_information": [],
                                        "image_conditioning_statement": "images reviewed",
                                    }
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request: Request, timeout: float):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                429,
                "busy",
                hdrs=None,
                fp=BytesIO(b'{"code":"1305","message":"busy"}'),
            )
        return Response()

    monkeypatch.setattr("hcc_multimodal.glm_vision.urlopen", fake_urlopen)
    result = call_zhipu_glm(
        prompt="image-only",
        image_paths=paths,
        api_key="test-key",
        base_url="https://example.invalid/v4",
        model="glm-test",
        retry_backoff_seconds=0,
    )
    assert calls == 2
    assert result["attempt_count"] == 2
    assert result["transient_retry_count"] == 1


def test_glm_prompt_pins_identifier_and_confidence_types():
    prompt = build_glm_prompt(
        phase="portal_venous",
        lesion_ids=["LESION_001"],
        source_refs=["IMAGE_MONTAGE", "LESION_001"],
    )
    assert '"finding_id": "FINDING_001"' in prompt
    assert '"confidence": "moderate"' in prompt
    assert '"high", "moderate", "low"' in prompt


def test_glm_observer_retries_one_schema_invalid_response(monkeypatch, tmp_path: Path):
    image, mask = _image_mask(tmp_path)
    lion = PrecomputedMaskLionBackend().infer(
        image_path=image,
        mask_path=mask,
        mask_role="user_supplied",
        patient_id="RESEARCH_001",
        study_date="2026-08-29",
        phase="portal_venous",
    )
    calls: list[str] = []

    def fake_call_zhipu_glm(**kwargs):
        calls.append(kwargs["prompt"])
        if len(calls) == 1:
            response = {
                "observations": [
                    {
                        "finding_id": 1,
                        "lesion_id": "LESION_001",
                        "location": "right hepatic region",
                        "observation": "heterogeneous attenuation",
                        "confidence": 0.8,
                        "source_refs": ["LESION_001"],
                    }
                ],
                "uncertainties": [],
                "missing_information": [],
                "image_conditioning_statement": "images reviewed",
            }
        else:
            response = {
                "observations": [
                    {
                        "finding_id": "FINDING_001",
                        "lesion_id": "LESION_001",
                        "location": "right hepatic region",
                        "observation": "heterogeneous attenuation",
                        "confidence": "moderate",
                        "source_refs": ["LESION_001"],
                    }
                ],
                "uncertainties": [],
                "missing_information": [],
                "image_conditioning_statement": "images reviewed",
            }
        return {
            "provider": "zhipu",
            "model": "glm-test",
            "usage": {},
            "finish_reason": "stop",
            "raw_content": json.dumps(response),
            "response": response,
            "attempt_count": 1,
            "transient_retry_count": 0,
            "format_retry_count": 0,
        }

    monkeypatch.setattr(
        "hcc_multimodal.glm_vision.call_zhipu_glm", fake_call_zhipu_glm
    )
    evidence, audit = run_glm_observer(
        lion=lion,
        image_path=image,
        mask_path=mask,
        output_dir=tmp_path / "observer",
        mode="live",
    )
    assert len(calls) == 2
    assert "SCHEMA CORRECTION RETRY" in calls[1]
    assert evidence.status == "pass"
    assert audit["schema_retry_count"] == 1
    assert audit["attempt_count"] == 2
