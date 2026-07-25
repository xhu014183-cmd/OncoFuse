# OncoFuse v0.4.0 — Fail-Closed Evidence Fusion (trusted baseline)

> 中文摘要：OncoFuse 是一个可审计的 HCC 多模态研究原型，融合分割后的 3D 肝脏影像
> 与纵向 AFP/DCP 检验证据，在几何 / 单位 / 配对 / 配准 / 报告安全性无法验证时**拒绝给出结论**。
> 本版本为可信基线：79 项测试全过、确定性测量链、受控 LLM 渲染层、双语文档与合成数据在线演示。

## What is OncoFuse

An auditable research prototype that fuses segmented 3D liver imaging with
longitudinal AFP/DCP laboratory evidence — and **refuses to answer when
geometry, units, pairing, registration, or report safety cannot be verified**.

It does **not** diagnose HCC, assign LI-RADS/BCLC/RECIST/mRECIST categories,
predict prognosis, or recommend treatment. Every output is a validated JSON
artifact with provenance; every claim traces to a rule ID.

## Highlights in v0.4.0

- **Deterministic imaging line** — voxel-count × spacing = volume; 3D connected
  components = lesion count; Hungarian global assignment pairs longitudinal
  lesions (matched / new / disappeared / split / merge / indeterminate).
- **Controlled LLM rendering layer** — the LLM (if enabled) only *renders*
  compact de-identified evidence into a report. Changed locked fields, invented
  numbers, omitted QC, diagnostic assertions, staging, treatment advice, or
  prompt-injection echoes **block the report**. A deterministic renderer is
  always the fallback.
- **Real-call audit case** — a live DeepSeek-r1 (DashScope) run was correctly
  **rejected** by guardrails (negation bug), then **passed** after the fix.
  Shipped as documented evidence of fail-closed behavior.
- **79 passing tests**, `ruff` + `mypy` clean, 18 JSON schemas (schema 1.0.0),
  bilingual README, GitHub Pages demo (synthetic data), public TCIA HCC_003 demo.

## Try it

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\hcc-demo run-demo --output demo-output
```

Open `docs/demo/index.html` for the live demo (synthetic data, clearly labeled).

## Caveats

- Research prototype only. **Not a medical device. Not for clinical use.**
- Masks are currently supplied externally; automated segmentation is on the
  roadmap (see Roadmap).
- All demo numbers are synthetic.

## Roadmap

- **M1–M4**: real VLM inference (rented GPU, M3D-LaMed) on 5 HCC-TACE-Seg cases
  with expert review.
- **Automated tumor segmentation** (TotalSegmentator) as a fail-closed
  pre-screening trigger.
- Multicenter cohort validation expansion.

## License

See `LICENSE`.
