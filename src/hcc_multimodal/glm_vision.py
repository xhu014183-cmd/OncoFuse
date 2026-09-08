"""Zhipu GLM image-only observer for the LiON-inspired pipeline."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field, ValidationError
from scipy import ndimage

from .deepseek import (
    BOUNDARY_PATTERNS,
    NUMBER_RE,
    _extract_json,
    _first_unnegated_pattern,
)
from .schemas import (
    GlmImagingEvidence,
    GlmImagingFinding,
    JsonModel,
    LionInspiredImagingEvidence,
    QualityCheck,
    QualityEvidence,
    SourceReference,
)

GLM_PROMPT_VERSION = "glm-imaging-observer-v1"


class GlmResponseFormatError(ValueError):
    """The provider returned content that is not one strict JSON object."""


class _GlmFindingResponse(JsonModel):
    finding_id: str
    lesion_id: str | None = None
    location: str
    observation: str
    confidence: Literal["high", "moderate", "low", "unavailable"]
    source_refs: list[str] = Field(min_length=1)


class _GlmResponse(JsonModel):
    observations: list[_GlmFindingResponse]
    uncertainties: list[str]
    missing_information: list[str]
    image_conditioning_statement: str


def _radiological_axial(value: np.ndarray) -> np.ndarray:
    return np.fliplr(np.rot90(value, k=1))


def _window(value: np.ndarray) -> np.ndarray:
    return np.clip((value.astype(np.float32) + 200.0) / 500.0, 0, 1)


def _panel(
    image_slice: np.ndarray,
    mask_slice: np.ndarray | None,
    *,
    label: str,
    width: int = 256,
) -> Image.Image:
    gray = _radiological_axial((_window(image_slice) * 255).astype(np.uint8))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if mask_slice is not None:
        region = _radiological_axial(mask_slice.astype(bool))
        rgb[region] = (
            0.35 * rgb[region] + 0.65 * np.asarray([235, 55, 45])
        ).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    if image.width != width:
        height = max(round(image.height * width / image.width), 1)
        image = image.resize((width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 132, 27), fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 255))
    return image


def create_glm_preview_images(
    image_path: str | Path,
    mask_path: str | Path | None,
    output_dir: str | Path,
    *,
    max_lesion_zooms: int = 3,
) -> tuple[list[Path], list[str]]:
    """Render deidentified PNGs; never copy DICOM metadata or patient labels."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    image = cast(nib.Nifti1Image, nib.as_closest_canonical(nib.load(str(image_path))))
    image_data = np.asarray(image.dataobj, dtype=np.float32)
    if image_data.ndim != 3:
        raise ValueError("GLM preview generation requires a 3D volume")
    mask_data: np.ndarray | None = None
    if mask_path is not None:
        mask = cast(nib.Nifti1Image, nib.as_closest_canonical(nib.load(str(mask_path))))
        if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-3):
            raise ValueError("Image and mask must align before GLM preview generation")
        mask_data = np.asarray(mask.dataobj) > 0

    depth = image_data.shape[2]
    occupied = (
        np.flatnonzero(mask_data.any(axis=(0, 1)))
        if mask_data is not None and mask_data.any()
        else np.asarray([], dtype=int)
    )
    if occupied.size:
        lo = max(0, int(occupied.min()) - 8)
        hi = min(depth - 1, int(occupied.max()) + 8)
    else:
        lo, hi = 0, max(depth - 1, 0)
    indices = sorted({round(value) for value in np.linspace(lo, hi, 16)})
    panels = [
        _panel(
            image_data[:, :, index],
            mask_data[:, :, index] if mask_data is not None else None,
            label=f"VIEW_{position:03d}",
        )
        for position, index in enumerate(indices, 1)
    ]
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    montage = Image.new("RGB", (cell_width * 4, cell_height * 4), color=(0, 0, 0))
    for position, panel in enumerate(panels):
        montage.paste(panel, ((position % 4) * cell_width, (position // 4) * cell_height))
    montage_path = target / "image_montage.png"
    montage.save(montage_path)
    image_paths = [montage_path]
    source_refs = ["IMAGE_MONTAGE"]

    if mask_data is not None and mask_data.any():
        labels, count = ndimage.label(
            mask_data,
            structure=ndimage.generate_binary_structure(rank=3, connectivity=3),
        )
        components = sorted(
            range(1, count + 1),
            key=lambda value: int(np.count_nonzero(labels == value)),
            reverse=True,
        )[:max_lesion_zooms]
        for lesion_index, component in enumerate(components, 1):
            coords = np.argwhere(labels == component)
            if not coords.size:
                continue
            center_x, center_y, center_z = np.rint(coords.mean(axis=0)).astype(int)
            half = 75
            x0, x1 = max(0, center_x - half), min(image_data.shape[0], center_x + half)
            y0, y1 = max(0, center_y - half), min(image_data.shape[1], center_y + half)
            crop_image = image_data[x0:x1, y0:y1, center_z]
            crop_mask = labels[x0:x1, y0:y1, center_z] == component
            lesion_id = f"LESION_{lesion_index:03d}"
            zoom = _panel(crop_image, crop_mask, label=lesion_id, width=360)
            zoom_path = target / f"lesion_{lesion_index:03d}.png"
            zoom.save(zoom_path)
            image_paths.append(zoom_path)
            source_refs.append(lesion_id)
    return image_paths, source_refs


def _image_part(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def build_glm_prompt(*, phase: str, lesion_ids: list[str], source_refs: list[str]) -> str:
    example_lesion_id = lesion_ids[0] if lesion_ids else None
    example_source_ref = (
        example_lesion_id
        if example_lesion_id in source_refs
        else source_refs[0]
    )
    example = {
        "observations": [
            {
                "finding_id": "FINDING_001",
                "lesion_id": example_lesion_id,
                "location": "right hepatic region",
                "observation": "localized region with heterogeneous attenuation",
                "confidence": "moderate",
                "source_refs": [example_source_ref],
            }
        ],
        "uncertainties": [
            "single-phase rendering limits enhancement-pattern assessment"
        ],
        "missing_information": ["complete multiphase CT is unavailable"],
        "image_conditioning_statement": (
            "the montage and localization crop conditioned the observations"
        ),
    }
    return f"""You are an image-only observer in a research HCC evidence pipeline.
The attached images are deidentified CT renderings for one single phase ({phase}).
Red overlays are supplied segmentation regions, not automatically detected lesions.
Allowed lesion IDs: {json.dumps(lesion_ids)}.
Allowed image source references: {json.dumps(source_refs)}.

Return exactly one JSON object with these fields and no others:
- observations: array of objects with finding_id, lesion_id (or null), location,
  observation, confidence, source_refs
- uncertainties: array of strings
- missing_information: array of strings
- image_conditioning_statement: string

Required types and allowed values:
- finding_id must be a quoted string such as "FINDING_001", never a number.
- lesion_id must be one quoted allowed lesion ID or null.
- confidence must be exactly one quoted English value: "high", "moderate", "low",
  or "unavailable". Do not use a score, percentage, translation, or other label.
- source_refs must be an array containing only quoted allowed source references.

Valid shape example (copy the types and keys, not the clinical wording):
{json.dumps(example, ensure_ascii=False, indent=2)}

Rules:
- Describe only visible location, margin, internal texture, and appearance.
- Do not use laboratory values, history, malignancy probabilities, or classifications.
- Do not diagnose HCC/cancer, assign LI-RADS/BCLC/RECIST, predict prognosis, or recommend treatment.
- Do not state or estimate any number, measurement, count, percentage, or slice index.
- Use only the allowed lesion IDs and source references.
- A red overlay is localization supplied by a mask; never claim the model detected it.
- Single-phase CT cannot establish a complete dynamic enhancement pattern.
- Explain briefly how the attached images affected the observations.
"""


def call_zhipu_glm(
    *,
    prompt: str,
    image_paths: list[Path],
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 180.0,
    max_transient_retries: int = 2,
    max_format_retries: int = 1,
    retry_backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    key = api_key or os.environ.get("ZHIPU_API_KEY")
    endpoint_base = base_url or os.environ.get("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
    model_name = model or os.environ.get("ZHIPU_VISION_MODEL") or "glm-4.6v-flash"
    if not key:
        raise RuntimeError("ZHIPU_API_KEY is unavailable")
    if not image_paths:
        raise RuntimeError("GLM image call requires at least one rendered image")
    if max_transient_retries < 0 or max_format_retries < 0:
        raise ValueError("GLM retry counts must be non-negative")
    if retry_backoff_seconds < 0:
        raise ValueError("GLM retry backoff must be non-negative")

    transient_codes = {408, 409, 425, 429, 500, 502, 503, 504}
    transient_retry_count = 0
    format_retry_count = 0
    attempt_count = 0
    retry_instruction = ""
    while True:
        attempt_count += 1
        body = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *(_image_part(path) for path in image_paths),
                        {"type": "text", "text": prompt + retry_instruction},
                    ],
                }
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            endpoint_base.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            can_retry = (
                exc.code in transient_codes
                and transient_retry_count < max_transient_retries
            )
            if not can_retry:
                raise RuntimeError(
                    f"Zhipu HTTP {exc.code} after {attempt_count} attempt(s): "
                    f"{detail[:1000]}"
                ) from exc
            delay = retry_backoff_seconds * (2**transient_retry_count)
            transient_retry_count += 1
            time.sleep(delay)
            continue
        except URLError as exc:
            if transient_retry_count >= max_transient_retries:
                raise RuntimeError(
                    f"Zhipu connection failed after {attempt_count} attempt(s): {exc}"
                ) from exc
            delay = retry_backoff_seconds * (2**transient_retry_count)
            transient_retry_count += 1
            time.sleep(delay)
            continue

        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        try:
            parsed_response = _extract_json(content)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            if format_retry_count >= max_format_retries:
                raise GlmResponseFormatError(
                    f"{exc}; blocked after {attempt_count} provider attempt(s)"
                ) from exc
            format_retry_count += 1
            retry_instruction = (
                "\n\nYour previous response was rejected because it was not exactly one "
                "valid JSON object. Return one JSON object only, with every required "
                "field, and no prose, Markdown fence, second object, or trailing text."
            )
            continue
        break
    return {
        "provider": "zhipu",
        "model": model_name,
        "usage": payload.get("usage", {}),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "raw_content": content,
        "response": parsed_response,
        "attempt_count": attempt_count,
        "transient_retry_count": transient_retry_count,
        "format_retry_count": format_retry_count,
    }


def _narrative(payload: _GlmResponse) -> str:
    return "\n".join(
        [
            *(item.location for item in payload.observations),
            *(item.observation for item in payload.observations),
            *payload.uncertainties,
            *payload.missing_information,
            payload.image_conditioning_statement,
        ]
    )


def validate_glm_response(
    response: dict[str, Any],
    *,
    allowed_lesion_ids: set[str],
    allowed_source_refs: set[str],
) -> tuple[_GlmResponse | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    try:
        parsed = _GlmResponse.model_validate(response)
    except ValidationError as exc:
        return None, [
            {
                "code": "SCHEMA_INVALID",
                "message": ".".join(str(value) for value in item["loc"])
                + ": "
                + item["msg"],
            }
            for item in exc.errors(include_url=False)
        ]
    for finding in parsed.observations:
        if finding.lesion_id is not None and finding.lesion_id not in allowed_lesion_ids:
            errors.append(
                {
                    "code": "UNKNOWN_LESION_ID",
                    "message": f"Unknown lesion ID: {finding.lesion_id}",
                }
            )
        unknown_refs = sorted(set(finding.source_refs) - allowed_source_refs)
        if unknown_refs:
            errors.append(
                {
                    "code": "UNKNOWN_IMAGE_REFERENCE",
                    "message": f"Unknown image references: {unknown_refs}",
                }
            )
    narrative = _narrative(parsed)
    for code, patterns in BOUNDARY_PATTERNS.items():
        for line in narrative.splitlines():
            pattern = _first_unnegated_pattern(line, patterns)
            if pattern is not None:
                errors.append(
                    {"code": code, "message": f"Boundary pattern matched: {pattern.pattern}"}
                )
                break
    scrubbed = re.sub(r"(?:LESION|VIEW|FINDING)_\d+", "ID", narrative, flags=re.IGNORECASE)
    numbers = sorted({match.group(0) for match in NUMBER_RE.finditer(scrubbed)})
    if numbers:
        errors.append(
            {
                "code": "UNSUPPORTED_NUMERIC_OBSERVATION",
                "message": f"GLM narrative contains unsupported numbers: {numbers}",
            }
        )
    return (None if errors else parsed), errors


def _unavailable_evidence(
    *,
    lion: LionInspiredImagingEvidence,
    provider: Literal["disabled", "unavailable"],
    model: str,
    message: str,
) -> GlmImagingEvidence:
    return GlmImagingEvidence(
        patient_id=lion.patient_id,
        study_date=lion.study_date,
        phase=lion.phase,
        status="unavailable",
        provider=provider,
        model=model,
        prompt_version=GLM_PROMPT_VERSION,
        observations=[],
        uncertainties=[],
        missing_information=[message],
        image_conditioning_statement="No validated GLM image observation is available.",
        quality=QualityEvidence(
            status="unavailable",
            checks=[
                QualityCheck(
                    check_id="GLM_IMAGE_OBSERVER",
                    status="unavailable",
                    message=message,
                )
            ],
            warnings=[message],
        ),
        limitations=[
            "Unavailable GLM evidence must not be interpreted as a negative image finding"
        ],
        provenance={"prompt_version": GLM_PROMPT_VERSION},
        sources=[],
    )


def run_glm_observer(
    *,
    lion: LionInspiredImagingEvidence,
    image_path: str | Path,
    mask_path: str | Path | None,
    output_dir: str | Path,
    mode: Literal["off", "live"] = "off",
    timeout_seconds: float = 180.0,
) -> tuple[GlmImagingEvidence, dict[str, Any]]:
    """Run the image-only GLM observer and always return an auditable artifact."""
    generated_at = datetime.now(UTC).isoformat()
    configured_model = os.environ.get("ZHIPU_VISION_MODEL") or "glm-4.6v-flash"
    if mode == "off":
        evidence = _unavailable_evidence(
            lion=lion,
            provider="disabled",
            model=configured_model,
            message="GLM image observer was disabled by configuration",
        )
        return evidence, {
            "generated_at": generated_at,
            "provider": "disabled",
            "model": configured_model,
            "audit_status": "unavailable",
            "error_code": "GLM_DISABLED",
        }

    preview_dir = Path(output_dir) / "glm-previews"
    image_paths, source_refs = create_glm_preview_images(
        image_path, mask_path, preview_dir
    )
    lesion_ids = [item.lesion_id for item in lion.lesion_evidence]
    prompt = build_glm_prompt(
        phase=lion.phase,
        lesion_ids=lesion_ids,
        source_refs=source_refs,
    )
    try:
        result = call_zhipu_glm(
            prompt=prompt,
            image_paths=image_paths,
            timeout_seconds=timeout_seconds,
        )
        parsed, errors = validate_glm_response(
            result["response"],
            allowed_lesion_ids=set(lesion_ids),
            allowed_source_refs=set(source_refs),
        )
        if parsed is None:
            first_result = result
            correction_codes = sorted({item["code"] for item in errors})
            correction_prompt = (
                prompt
                + "\n\nSCHEMA CORRECTION RETRY: The previous response was rejected by "
                "deterministic validation. Error codes: "
                + json.dumps(correction_codes)
                + ". Return a newly generated response that follows the exact example "
                "types and allowed values. Return one JSON object only."
            )
            result = call_zhipu_glm(
                prompt=correction_prompt,
                image_paths=image_paths,
                timeout_seconds=timeout_seconds,
                max_format_retries=0,
            )
            result["attempt_count"] = int(first_result.get("attempt_count", 1)) + int(
                result.get("attempt_count", 1)
            )
            result["transient_retry_count"] = int(
                first_result.get("transient_retry_count", 0)
            ) + int(result.get("transient_retry_count", 0))
            result["format_retry_count"] = int(
                first_result.get("format_retry_count", 0)
            ) + int(result.get("format_retry_count", 0))
            result["schema_retry_count"] = 1
            parsed, errors = validate_glm_response(
                result["response"],
                allowed_lesion_ids=set(lesion_ids),
                allowed_source_refs=set(source_refs),
            )
    except GlmResponseFormatError as exc:
        evidence = _unavailable_evidence(
            lion=lion,
            provider="unavailable",
            model=configured_model,
            message="GLM response was blocked because it was not one strict JSON object",
        )
        return evidence, {
            "generated_at": generated_at,
            "provider": "zhipu",
            "model": configured_model,
            "audit_status": "fail",
            "error_code": "GLM_OUTPUT_BLOCKED",
            "validation_errors": [
                {"code": "SCHEMA_INVALID", "message": str(exc)}
            ],
            "image_count": len(image_paths),
            "prompt_version": GLM_PROMPT_VERSION,
        }
    except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        evidence = _unavailable_evidence(
            lion=lion,
            provider="unavailable",
            model=configured_model,
            message=f"GLM image observer failed: {exc}",
        )
        return evidence, {
            "generated_at": generated_at,
            "provider": "zhipu",
            "model": configured_model,
            "audit_status": "fail",
            "error_code": "GLM_UNAVAILABLE",
            "error": str(exc),
            "image_count": len(image_paths),
            "prompt_version": GLM_PROMPT_VERSION,
        }
    if parsed is None:
        evidence = _unavailable_evidence(
            lion=lion,
            provider="unavailable",
            model=result["model"],
            message="GLM response was blocked by deterministic validation",
        )
        audit = {
            key: value for key, value in result.items() if key != "response"
        }
        audit.update(
            {
                "generated_at": generated_at,
                "audit_status": "fail",
                "error_code": "GLM_OUTPUT_BLOCKED",
                "validation_errors": errors,
                "image_count": len(image_paths),
                "prompt_version": GLM_PROMPT_VERSION,
            }
        )
        return evidence, audit

    status: Literal["pass", "warning"] = "pass" if parsed.observations else "warning"
    warnings = [] if parsed.observations else ["GLM returned no attributable observations"]
    evidence = GlmImagingEvidence(
        patient_id=lion.patient_id,
        study_date=lion.study_date,
        phase=lion.phase,
        status=status,
        provider="zhipu",
        model=result["model"],
        prompt_version=GLM_PROMPT_VERSION,
        observations=[GlmImagingFinding.model_validate(item.to_dict()) for item in parsed.observations],
        uncertainties=parsed.uncertainties,
        missing_information=parsed.missing_information,
        image_conditioning_statement=parsed.image_conditioning_statement,
        quality=QualityEvidence(
            status=status,
            checks=[
                QualityCheck(
                    check_id="GLM_IMAGE_OBSERVER",
                    status=status,
                    message="GLM response passed strict structure and safety validation",
                )
            ],
            warnings=warnings,
        ),
        limitations=[
            "GLM sees rendered key images rather than the complete native CT volume",
            "GLM observations are explanatory and are not calibrated diagnostic predictions",
        ],
        provenance={
            "prompt_version": GLM_PROMPT_VERSION,
            "image_source_refs": source_refs,
            "image_count": len(image_paths),
        },
        sources=[
            SourceReference(
                source_id=value,
                source_type="deidentified_png_render",
                uri=None,
                deidentified=True,
            )
            for value in source_refs
        ],
    )
    audit = {key: value for key, value in result.items() if key != "response"}
    audit.update(
        {
            "generated_at": generated_at,
            "audit_status": "pass",
            "validation_errors": [],
            "image_count": len(image_paths),
            "prompt_version": GLM_PROMPT_VERSION,
        }
    )
    return evidence, audit


__all__ = [
    "GLM_PROMPT_VERSION",
    "GlmResponseFormatError",
    "build_glm_prompt",
    "call_zhipu_glm",
    "create_glm_preview_images",
    "run_glm_observer",
    "validate_glm_response",
]
