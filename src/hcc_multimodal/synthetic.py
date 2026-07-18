from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np

from .schemas import PIPELINE_VERSION, SCHEMA_VERSION


def _sphere(shape: tuple[int, int, int], center: tuple[int, int, int], radius: int) -> np.ndarray:
    grid = np.indices(shape)
    distance_squared = sum((grid[axis] - center[axis]) ** 2 for axis in range(3))
    return distance_squared <= radius**2


def generate_synthetic_case(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shape = (40, 40, 24)
    affine = np.diag([1.5, 1.5, 2.5, 1.0])
    rng = np.random.default_rng(42)

    baseline_mask = _sphere(shape, (14, 20, 12), 3)
    followup_mask = _sphere(shape, (14, 20, 12), 4)
    followup_mask |= _sphere(shape, (30, 10, 12), 2)

    baseline_image = rng.normal(55, 8, size=shape).astype(np.float32)
    followup_image = rng.normal(55, 8, size=shape).astype(np.float32)
    baseline_image[baseline_mask] += 25
    followup_image[followup_mask] += 25

    paths = {
        "baseline_image": output / "baseline_ct.nii.gz",
        "baseline_mask": output / "baseline_tumor_mask.nii.gz",
        "followup_image": output / "followup_ct.nii.gz",
        "followup_mask": output / "followup_tumor_mask.nii.gz",
        "labs": output / "labs.json",
    }
    nib.save(nib.Nifti1Image(baseline_image, affine), paths["baseline_image"])
    nib.save(nib.Nifti1Image(baseline_mask.astype(np.uint8), affine), paths["baseline_mask"])
    nib.save(nib.Nifti1Image(followup_image, affine), paths["followup_image"])
    nib.save(nib.Nifti1Image(followup_mask.astype(np.uint8), affine), paths["followup_mask"])

    lab_payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [{"source_id": "synthetic-generator-v1", "source_type": "synthetic"}],
        "patient_id": "DEMO_HCC_001",
        "data_relationship": "Synthetic imaging and laboratory values belong to one generated case",
        "evidence_context": {
            "data_origin": "synthetic",
            "pairing_status": "same_subject",
        },
        "observations": [
            {"date": "2026-01-15", "marker": "AFP", "value": 6.0, "unit": "ng/mL", "upper_reference": 7.0},
            {"date": "2026-04-15", "marker": "AFP", "value": 18.0, "unit": "ng/mL", "upper_reference": 7.0},
            {"date": "2026-07-15", "marker": "AFP", "value": 85.3, "unit": "ng/mL", "upper_reference": 7.0},
            {"date": "2026-01-15", "marker": "DCP", "value": 25.0, "unit": "mAU/mL", "upper_reference": 40.0},
            {"date": "2026-07-15", "marker": "DCP", "value": 68.0, "unit": "mAU/mL", "upper_reference": 40.0}
        ]
    }
    paths["labs"].write_text(json.dumps(lab_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths
