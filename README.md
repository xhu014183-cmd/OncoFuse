# HCC Multimodal Research Prototype

[中文说明](README.zh-CN.md) | English

This repository is an auditable research prototype for combining segmented 3D
liver imaging with longitudinal AFP/DCP evidence. It is designed to fail
closed when geometry, units, pairing, registration, or report safety cannot be
verified.

It does not diagnose HCC, assign LI-RADS/BCLC/RECIST/mRECIST categories,
predict prognosis, or recommend treatment.

## Safety invariants

- Every public artifact is validated by a strict Pydantic contract and carries
  `schema_version`, `pipeline_version`, `generated_at`, sources, and quality.
- Quality uses only `pass`, `warning`, `fail`, or `unavailable`. Critical
  imaging or registration failure blocks quantitative longitudinal analysis.
- Laboratory parsing retains comparator, source text, parse status, reference
  interval, original unit, and normalized unit. Unusable values remain null.
- DICOM SEG frames are mapped into a complete, coherent CT acquisition in
  patient coordinates. SEG-referenced slices do not define the CT extent.
- Lesion pairing uses Hungarian global assignment and reports matched, new,
  disappeared, split candidate, merge candidate, or indeterminate states.
- Fusion rules and matching thresholds are versioned YAML. Each verdict
  returns rule IDs, evidence references, threshold version, and explanation.
- The LLM sees only compact deidentified evidence. Schema errors, changed
  locked fields, invented numbers, omitted QC, diagnostic assertions, staging,
  treatment advice, and prompt-injection echoes block the formal report.
- A deterministic report renderer is available when the external LLM is not.
- Image embeddings are optional and never override the deterministic baseline.

## Install and verify

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,public-data]"
.\.venv\Scripts\python -m pytest --cov=hcc_multimodal -q
.\.venv\Scripts\ruff check src tests
.\.venv\Scripts\mypy src\hcc_multimodal
```

## Synthetic end-to-end run

```powershell
.\.venv\Scripts\hcc-demo run-demo --output demo-output
```

The run writes validated imaging, lab, longitudinal, embedding, multimodal,
fusion, and compact prompt artifacts. The expected synthetic verdict is
`concordant_progression_signal`; this is a contract test, not clinical
performance evidence.

## Analyze aligned NIfTI data

```powershell
.\.venv\Scripts\hcc-demo analyze `
  --baseline-image baseline_ct.nii.gz `
  --baseline-mask baseline_mass.nii.gz `
  --followup-image followup_ct.nii.gz `
  --followup-mask followup_mass.nii.gz `
  --labs labs.json `
  --patient-id RESEARCH_001 `
  --baseline-date 2026-01-15 `
  --followup-date 2026-07-15 `
  --baseline-phase portal_venous `
  --followup-phase portal_venous `
  --registration-status verified `
  --pairing-status same_subject `
  --output research-output
```

Do not declare `verified` registration unless it has an external QC basis. If
the flag is omitted, identical NIfTI grids are recorded as
`assumed_same_grid`; other unverified geometry is blocked.

An optional treatment event file is a JSON array and can be supplied with
`--treatment-events`. Intervening treatment blocks a simple concordance claim.

## Laboratory input

```json
{
  "patient_id": "RESEARCH_001",
  "observations": [
    {
      "date": "2026-07-15",
      "marker": "AFP",
      "value": "<15",
      "unit": "ng/mL",
      "referenceRange": "0-7"
    }
  ]
}
```

Supported numeric syntax includes signs, decimals, scientific notation, and
`<`, `<=`, `>`, `>=`. Common AFP `ng/mL`/`ug/L` and DCP
`mAU/mL`/`AU/L` spellings are normalized. Unknown or non-convertible units do
not enter trends or upper-limit-normalized features.

## DICOM CT/SEG conversion

```powershell
.\.venv\Scripts\hcc-demo convert-dicom-seg `
  --ct-dir path\to\ct-series `
  --seg path\to\seg.dcm `
  --output converted
```

The converter checks orientation, rows/columns, pixel spacing, slice order,
duplicate positions, nonuniform spacing, Study/Series/FoR UIDs, per-frame SOP
references, segment identity, and SEG-to-CT landmark error. It maps SEG plane
orientation and origin to the CT pixel grid, including row/column flips.

The optional public demo uses immutable HCC-TACE-Seg HCC_003 identifiers:

```powershell
.\.venv\Scripts\hcc-demo run-public-demo --output public-data\HCC_003
```

The public image and expert SEG are real; its AFP/DCP scenarios are unrelated
synthetic values and are always labeled `unpaired_poc_composite`.

## Controlled report layer

Generate the public scenario first, then run the external renderer:

```powershell
$env:LLM_API_KEY = "..."
$env:LLM_BASE_URL = "https://example.invalid/v1"
$env:LLM_MODEL_NAME = "configured-model"
.\.venv\Scripts\hcc-demo run-deepseek `
  --scenario dual_marker_rising `
  --public-dir public-data\HCC_003
```

If the external service is unavailable, the command produces a deterministic
validated template. If model output violates a safety rule, the audit JSON is
retained and no formal Markdown is generated.

## Patient-level evaluation

```powershell
.\.venv\Scripts\hcc-demo evaluate-cohort `
  --cohort cohort.json `
  --split test `
  --bootstrap-iterations 1000 `
  --output evaluation.json
```

The cohort contract fixes the endpoint as radiographic progression within a
declared follow-up window. It rejects patient leakage across splits and dates
outside the window. Output includes image-only, lab-only, and rule-fusion
baselines; AUROC/AUPRC, sensitivity, specificity, Brier score, calibration
error, patient bootstrap intervals, and center/scanner strata. The interface
always marks clinical performance claims as not permitted.

See [docs/RESEARCH_EVALUATION.md](docs/RESEARCH_EVALUATION.md) for the data
checklist and pre-registration fields.

## Locked multicenter research validation

Pipeline `0.3.0` adds a label-separated workflow for a local deidentified,
paired cohort. The protocol is fixed to pretreatment portal-venous CT within
42 days, AFP/DCP within 14 days of each image, and blinded radiographic
progression within 180 days. An entire explicitly named center is held out as
the external test set.

```powershell
.\.venv\Scripts\hcc-demo validate-research-cohort `
  --protocol examples\research_protocol.template.json `
  --manifest path\to\research_manifest.json `
  --output cohort-validation.json

.\.venv\Scripts\hcc-demo build-research-cohort `
  --protocol examples\research_protocol.template.json `
  --manifest path\to\research_manifest.json `
  --output locked-features

.\.venv\Scripts\hcc-demo validate-adjudications `
  --run locked-features\research_run_manifest.json `
  --adjudications path\outside\locked-features\development-labels.json `
  --output development-adjudication-validation.json

.\.venv\Scripts\hcc-demo evaluate-research-cohort `
  --protocol examples\research_protocol.template.json `
  --manifest path\to\research_manifest.json `
  --run locked-features\research_run_manifest.json `
  --adjudications path\outside\locked-features\development-labels.json `
  --scope development `
  --output development-evaluation
```

The feature build has no adjudication argument and records
`label_data_loaded=false`. External-center evaluation additionally requires
the explicit one-time `--unlock-external` flag and writes an unlock audit.
Formal evaluation uses 2000 patient-level paired bootstrap samples and never
calls an LLM. See [docs/REAL_COHORT_VALIDATION.md](docs/REAL_COHORT_VALIDATION.md)
for the directory boundary, endpoint rules, and expected outputs.

## CPU 三线病例分析 (pipeline 0.4.0)

新入口把一次 CT/MRI 检查、可选 SEG、现有图像转文字结果、检验报告和病程文本整理成同一份研究证据摘要。它只计算可追溯的观察、检验异常和时间趋势，不输出诊断、分期、预后或治疗建议；不安装模型权重也可以运行。

```powershell
python examples\generate_dicom_seg_fixture.py --output synthetic-case
.\.venv\Scripts\hcc-demo analyze-case `
  --dicom-dir synthetic-case\study `
  --seg synthetic-case\seg.dcm `
  --image-evidence examples\image_evidence.synthetic.json `
  --labs examples\lab_report.synthetic.txt `
  --hpi examples\hpi.synthetic.txt `
  --patient-id RESEARCH_001 `
  --output case-output
```

输出目录包含 `imaging.json`、`labs.json`、可选的 `timeline.json`、`case-summary.json` 和 `case-summary.md`。三个调试入口也可以分别运行：

```powershell
.\.venv\Scripts\hcc-demo parse-imaging --dicom-dir study --patient-id RESEARCH_001 --output imaging.json
.\.venv\Scripts\hcc-demo parse-labs --input examples\lab_report.synthetic.txt --patient-id RESEARCH_001 --output labs.json
.\.venv\Scripts\hcc-demo parse-hpi --input examples\hpi.synthetic.txt --patient-id RESEARCH_001 --labs labs.json --output timeline.json
.\.venv\Scripts\hcc-demo summarize-case --imaging imaging.json --labs labs.json --timeline timeline.json --output case-output
```

影像线优先使用程序完成 DICOM 几何和 SEG 物理测量；没有 SEG 时只保留图像工具提供的定性观察。检验线支持 TXT/JSON/CSV、科学计数法、比较符、单位转换、参考范围和 AFP/DCP/AFP-L3、肝储备、肝损伤、病毒学项目。HPI 线按日期合并治疗、影像和检验事件，输出 `falling_after_treatment`、`rebound_after_nadir`、`persistent_rising` 等确定性状态。外部 LLM 默认关闭，即使将来启用也只能改写已锁定 JSON。

## Public schemas

```powershell
.\.venv\Scripts\hcc-demo export-schemas --output schemas
.\.venv\Scripts\hcc-demo validate-json `
  --type clinical-verdict `
  --input demo-output\clinical_verdict.json
```

Schema 1.0 remains current. Pipeline 0.4 adds the three-line case artifacts
without invalidating valid 0.2/0.3 schema-1.0 artifacts. See
[MIGRATION.md](MIGRATION.md).

## Optional 3D encoders

`statistical-v1` is a deterministic integration baseline. `m3d-clip` remains
an explicitly authorized optional research path. Both now record physical
resampling, fixed CT window, lesion ROI crop, synchronized mask transforms,
phase, finite-value checks, and L2 norm. Missing encoder support never blocks
the deterministic rule baseline.

Raw DICOM, patient identifiers, model weights, generated embeddings, and
clinical documents must not be committed. See [DISCLAIMER.md](DISCLAIMER.md),
[DATA_SOURCES.md](DATA_SOURCES.md), and [MODEL_SOURCES.md](MODEL_SOURCES.md).
