# OncoFuse: Fail-Closed Evidence Fusion for Liver Imaging & AFP/DCP

[中文说明](README.zh-CN.md) | English

[![CI](https://github.com/xhu014183-cmd/OncoFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/xhu014183-cmd/OncoFuse/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Schema 1.0](https://img.shields.io/badge/schema-1.0.0-green)](schemas/)
[![License](https://img.shields.io/badge/license-see%20LICENSE-orange)](LICENSE)
[![Tests: 79 passing](https://img.shields.io/badge/tests-79%20passing-brightgreen)](tests/)

**An auditable research prototype that fuses segmented 3D liver imaging with
longitudinal AFP/DCP laboratory evidence — and refuses to answer when
geometry, units, pairing, registration, or report safety cannot be verified.**

It does **not** diagnose HCC, assign LI-RADS/BCLC/RECIST/mRECIST categories,
predict prognosis, or recommend treatment. Every output is a validated JSON
artifact with provenance; every claim traces to a rule ID.

> **Live demo (synthetic data):** see [`docs/demo/`](docs/demo/) — open
> `docs/demo/index.html` locally, or enable GitHub Pages on `docs/` to share
> it. All numbers are synthetic and labeled as such.

---

## Why this exists

Most medical-AI demos answer *"how accurate is the model?"*. This repository
answers a different question:

> **Can a multimodal pipeline prove, artifact by artifact, why it said what it
> said — and stay silent when it cannot?**

- Segmentation masks are **measured deterministically** (voxel counts,
  volumes, centroids) — never "interpreted" by a language model.
- Longitudinal lesions are paired by **Hungarian global assignment** with
  explicit matched / new / disappeared / split / merge / indeterminate states.
- AFP/DCP trends keep comparators (`<`, `>=`), parse status, reference
  intervals, and original units. Unusable values stay null — never silently
  coerced.
- A versioned YAML **rule engine** fuses imaging and lab evidence into a
  verdict with reason codes, evidence references, and threshold versions.
- The LLM, if enabled at all, only **renders** compact deidentified evidence
  into a report. Changed locked fields, invented numbers, omitted QC,
  diagnostic assertions, staging, treatment advice, or prompt-injection
  echoes **block the report**. A deterministic renderer is always available
  as fallback.

## 30-second start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\hcc-demo run-demo --output demo-output
```

Expected verdict: `concordant_progression_signal` (a contract test on fully
synthetic data — not clinical performance evidence). The output directory
contains the whole evidence chain:

```text
baseline_imaging_evidence.json      lesion count, volumes, QC
followup_imaging_evidence.json
longitudinal_imaging_evidence.json  lesion pairing, +135.8% volume change
lab_evidence.json                   AFP/DCP trends with reference limits
clinical_verdict.json               fused verdict + reason codes
multimodal_case_evidence.json       audit envelope (origins, pairing status)
llm_prompt.json                     compact deidentified prompt bundle
```

Run the checks:

```powershell
.\.venv\Scripts\python -m pytest -q     # 79 tests
.\.venv\Scripts\ruff check src tests
.\.venv\Scripts\mypy src\hcc_multimodal
```

## What it can do today

| Capability | Command | Status |
|---|---|---|
| Synthetic end-to-end demo | `hcc-demo run-demo` | ✅ stable |
| Longitudinal NIfTI + mask analysis | `hcc-demo analyze` | ✅ stable |
| DICOM CT + SEG conversion with geometry QC | `hcc-demo convert-dicom-seg` | ✅ stable |
| Public demo on TCIA HCC-TACE-Seg HCC_003 | `hcc-demo run-public-demo` | ✅ stable |
| Three-line case pipeline (CT/MR + labs + HPI) | `hcc-demo analyze-case` | ✅ 0.4.0 |
| Controlled LLM report rendering + audit | `hcc-demo run-deepseek` | ✅ fail-closed |
| Cohort evaluation with bootstrap intervals | `hcc-demo evaluate-cohort` | ✅ research |
| Locked multicenter validation workflow | `hcc-demo validate-research-cohort` … | ✅ research |
| CPU patch-extraction browser demo | `hcc-demo vlm-skeleton-web` | ⚠️ **mock**, no model |
| VLM task prompts (auditable / open) | `hcc-demo vlm-prompt` | ✅ dual-mode |
| Dual-mode VLM live demo (real LLM + fail-closed audit) | `hcc-demo vlm-live` | ✅ fail-closed |
| Local visualization service (upload imaging + labs → interpretation) | `hcc-demo vlm-web` | ✅ local |
| Real 3D VLM inference (M3D-LaMed) | — | 🗺️ roadmap, see below |

## Architecture

```text
NIfTI/DICOM + aligned mask ──► deterministic measurement ──► ImagingEvidence
        optional 3D encoder ──► embedding (audit only, never overrides)

AFP/DCP observations ──► trend evidence ──► LabEvidence
HPI text ──► dated events ──► HpiTimelineEvidence

ImagingEvidence + LabEvidence (+ HPI)
        ──► versioned rule fusion ──► ClinicalVerdict
        ──► compact deidentified prompt ──► controlled LLM renderer
        ──► safety audit (schema, locked fields, numbers, boundaries)
```

Full contract: [MULTIMODAL_ARCHITECTURE.md](MULTIMODAL_ARCHITECTURE.md) ·
three-line pipeline: [docs/THREE_LINE_PIPELINE.md](docs/THREE_LINE_PIPELINE.md)

## Dual-mode VLM prompting

The future VLM entry point (`vlm_prompting.build_vlm_task_prompt`) supports two
fusion modes:

- **auditable** (default): the prompt contains image tokens + imaging metadata
  only. Laboratory and HPI context are withheld by design; trends are computed
  deterministically (`labs.py`) and fused at the rules layer. Prompt metadata
  records exactly what was withheld (`labs_present_but_withheld`).
- **open**: laboratory observations are injected as `[UNVERIFIED_CONTEXT]`
  `[LAB_###]` items with date, value, unit, reference status, trajectory, and a
  source reference; the prompt requires verbatim, cited copying and forbids
  re-derivation or extrapolation. This arm exists for side-by-side
  demonstration of hallucination / prior bias, not for the trusted path.

```powershell
hcc-demo parse-labs --input examples\lab_report.synthetic.txt --patient-id DEMO --output labs.json
hcc-demo vlm-prompt --labs labs.json --fusion-mode both --phase portal_venous --timepoint followup --output prompt-out
```

This writes `vlm_prompt_auditable.{json,txt}` and
`vlm_prompt_open.{json,txt}`. The browser demo
([`docs/demo/index.html`](docs/demo/index.html)) shows both outputs side by
side with difference highlighting and an explicit "不可审计" tag on the open
arm.

### Real LLM comparison (`vlm-live`)

`hcc-demo vlm-live` sends both arm prompts to the configured OpenAI-compatible
endpoint (`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME`), validates each
response against `VlmDemoReport`, attaches per-number `numeric_citations`
resolved from the injected `[LAB_###]` items, and blocks any report whose
numbers do not trace back to the supplied context:

```powershell
hcc-demo vlm-live --labs labs.json --fusion-mode both --output live-out --temperature 0.3 --no-json-object
```

Notes:

- Pass `--image <png>` to attach real vision input (GLM-4V etc.); pass
  `--imaging-evidence <json>` to inject deterministic imaging measurements
  (volumes, lesion counts) into the prompt.
- Reasoning-class models (e.g. `deepseek-r1-distill-qwen-32b`) can return
  empty content with `response_format=json_object`; pass `--no-json-object
  --temperature 0.3` for them. Non-reasoning extraction models are preferred
  per [docs/VLM_WEB_DEMO_PLAN.md](docs/VLM_WEB_DEMO_PLAN.md).
- Outputs: `vlm_prompt_{mode}.json`, `vlm_arm_{mode}.json` (raw response +
  audit), `dual_arm_comparison.json` (statement diff, numeric citations,
  hallucination candidates), and `web_demo.json` — a self-contained payload
  that `docs/demo/index.html` can load directly ("载入真实结果") to replace
  its static mock panes with the real dual-arm output.
- A missing, malformed, or blocked LLM call never produces a report: the
  deterministic template renders the supplied context and must pass the same
  validator.

### Local visualization service (`vlm-web`)

`hcc-demo vlm-web` serves `docs/demo/index.html` with a real input path: upload
a CT volume (`.nii.gz`) + tumor mask (`.nii.gz`) + a lab report (`.txt` /
`ClinicalLabEvidence` JSON) and the page returns an auditable interpretation.
The server measures the imaging deterministically, runs the auditable arm
(image-only VLM), computes lab trends, and fail-closes on any audit error.
The page renders a doctor-facing clinical report, not engineering jargon:

```powershell
hcc-demo vlm-web --port 7861
```

After upload the page shows:

- **Clinical report (for clinicians, not a diagnosis)** — imaging findings,
  laboratory findings, combined assessment, and next steps in plain clinical
  language; rule-engine jargon and the audit trail are collapsed into a
  "technical appendix".
- **Scrollable axial slice viewer** (16 slices, slider + prev/next) with the
  tumor mask overlaid, plus a lesion close-up. Slices are reoriented to
  canonical RAS axes and rendered in the radiological orientation (anterior on
  top, the patient's right on the image left).
- **Lab trend chart** with ULN reference lines, per-marker change percentages,
  and a timeline aligning the CT date with each lab date.
- **Risk-tier badge** (high / medium / low) derived from a guideline-style
  heuristic (lesion size × marker thresholds) — explicitly non-diagnostic.
- **Graded audit**: definite diagnoses ("confirmed HCC") still block the
  report; softer phrasing ("consistent with …") passes with a visible
  "requires clinician confirmation" warning.
- **Service health check** (`GET /api/health`) and a version badge in the page
  header (`vlm-web 0.4.0-web · started …`) so it is obvious when the server or
  page is stale; opening the file directly shows a "start the service" hint
  instead of a bare fetch error.

Open `http://127.0.0.1:7861/` in a browser. The service binds to localhost and
keeps the LLM key in the server environment; it is a local tool, not a hosted
GitHub Pages app. Check the "同时跑 open 对照臂" box to also render the
dual-mode comparison from the same upload.

## Safety invariants

- Every public artifact is validated by a strict Pydantic contract and carries
  `schema_version`, `pipeline_version`, `generated_at`, sources, and quality.
- Quality uses only `pass`, `warning`, `fail`, or `unavailable`. Critical
  imaging or registration failure blocks quantitative longitudinal analysis.
- Laboratory parsing retains comparator, source text, parse status, reference
  interval, original unit, and normalized unit.
- DICOM SEG frames are mapped into a complete, coherent CT acquisition in
  patient coordinates. SEG-referenced slices do not define the CT extent.
- Fusion rules and matching thresholds are versioned YAML. Each verdict
  returns rule IDs, evidence references, threshold version, and explanation.
- The browser demo (`vlm-skeleton-web`) is an explicit **mock**: patch
  extraction is real, the "VLM" is a deterministic template. It is labeled as
  mock in the UI and never presented as model inference.

## Roadmap

**Real 3D VLM inference** is planned and documented in
[docs/VLM_WEB_DEMO_PLAN.md](docs/VLM_WEB_DEMO_PLAN.md):

- Local DICOM de-identification + preprocessing to `[1,32,256,256]`
- Inference-only use of `M3D-LaMed-Phi-3-4B` on a rented GPU (no training)
- Two-pass output: image-only observation (English free text) → constrained
  structured extraction; clinical context never contaminates image findings
- Negative-control experiments (zero tensor) proving image conditioning —
  or honestly reporting its absence
- Mock demo and real inference remain strictly separated; failure never
  falls back to mock output

Experimental connector/training modules (`vlm_model`, `vlm_training`,
`multimodal_connector`, `visual_tokens`) exist in the tree but are **not**
part of the trusted demo path.

## Detailed usage

### Analyze aligned NIfTI data

```powershell
.\.venv\Scripts\hcc-demo analyze `
  --baseline-image baseline_ct.nii.gz --baseline-mask baseline_mass.nii.gz `
  --followup-image followup_ct.nii.gz --followup-mask followup_mass.nii.gz `
  --labs labs.json --patient-id RESEARCH_001 `
  --baseline-date 2026-01-15 --followup-date 2026-07-15 `
  --baseline-phase portal_venous --followup-phase portal_venous `
  --registration-status verified --pairing-status same_subject `
  --output research-output
```

Do not declare `verified` registration without an external QC basis. Without
the flag, identical NIfTI grids are recorded as `assumed_same_grid`; other
unverified geometry is blocked. An optional `--treatment-events` JSON array
blocks a simple concordance claim when treatment intervenes.

### Laboratory input

```json
{
  "patient_id": "RESEARCH_001",
  "observations": [
    {"date": "2026-07-15", "marker": "AFP", "value": "<15",
     "unit": "ng/mL", "referenceRange": "0-7"}
  ]
}
```

Signs, decimals, scientific notation, and `<`, `<=`, `>`, `>=` comparators are
supported. AFP `ng/mL`/`ug/L` and DCP `mAU/mL`/`AU/L` spellings are
normalized. Unknown or non-convertible units never enter trends.

### DICOM CT/SEG conversion

```powershell
.\.venv\Scripts\hcc-demo convert-dicom-seg `
  --ct-dir path\to\ct-series --seg path\to\seg.dcm --output converted
```

Checks orientation, spacing, slice order, duplicates, UIDs, per-frame SOP
references, segment identity, and SEG-to-CT landmark error; maps SEG plane
orientation and origin to the CT pixel grid including flips.

### Public demo (TCIA HCC-TACE-Seg HCC_003)

```powershell
.\.venv\Scripts\hcc-demo run-public-demo --output public-data\HCC_003
```

The public image and expert SEG are real; AFP/DCP scenarios are unrelated
synthetic values, always labeled `unpaired_poc_composite`.

### Controlled report layer

```powershell
$env:LLM_API_KEY = "..."
$env:LLM_BASE_URL = "https://example.invalid/v1"
$env:LLM_MODEL_NAME = "configured-model"
.\.venv\Scripts\hcc-demo run-deepseek `
  --scenario dual_marker_rising --public-dir public-data\HCC_003
```

If the external service is unavailable, a deterministic validated template is
produced. If model output violates a safety rule, the audit JSON is retained
and no formal Markdown is generated. Reasoning-style models are **not**
recommended for this layer (measured: repetition loops, truncated JSON); use
a non-reasoning instruct model.

### Patient-level evaluation and locked multicenter validation

See [docs/RESEARCH_EVALUATION.md](docs/RESEARCH_EVALUATION.md) and
[docs/REAL_COHORT_VALIDATION.md](docs/REAL_COHORT_VALIDATION.md) for the
cohort contract, directory boundaries, endpoint rules, and the explicit
`--unlock-external` one-time flag.

### CPU three-line case pipeline (0.4.0)

```powershell
python examples\generate_dicom_seg_fixture.py --output synthetic-case
.\.venv\Scripts\hcc-demo analyze-case `
  --dicom-dir synthetic-case\study --seg synthetic-case\seg.dcm `
  --image-evidence examples\image_evidence.synthetic.json `
  --labs examples\lab_report.synthetic.txt --hpi examples\hpi.synthetic.txt `
  --patient-id RESEARCH_001 --output case-output
```

Outputs `imaging.json`, `labs.json`, optional `timeline.json`,
`case-summary.json`, and `case-summary.md`. The HPI line derives deterministic
states such as `falling_after_treatment`, `rebound_after_nadir`, and
`persistent_rising`. External LLM rewriting is disabled by default.

### Public schemas

```powershell
.\.venv\Scripts\hcc-demo export-schemas --output schemas
.\.venv\Scripts\hcc-demo validate-json --type clinical-verdict --input demo-output\clinical_verdict.json
```

Schema 1.0 remains current; see [MIGRATION.md](MIGRATION.md).

### Optional 3D encoders

`statistical-v1` is a deterministic integration baseline. `m3d-clip` is an
explicitly authorized optional research path. Both record physical resampling,
fixed CT window, lesion ROI crop, synchronized mask transforms, phase,
finite-value checks, and L2 norm. Missing encoder support never blocks the
deterministic rule baseline.

## Data and model governance

Raw DICOM, patient identifiers, model weights, generated embeddings, and
clinical documents must not be committed. See
[DISCLAIMER.md](DISCLAIMER.md) · [DATA_SOURCES.md](DATA_SOURCES.md) ·
[MODEL_SOURCES.md](MODEL_SOURCES.md).

**Intended use:** research demonstration only. Not for diagnosis, staging,
prognosis, or treatment decisions.
