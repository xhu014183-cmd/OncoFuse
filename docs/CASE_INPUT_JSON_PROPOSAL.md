# 病例输入 JSON 设计方案

## 目标

- 用**一个 JSON 文件**统一描述 HCC 多模态三线管线的输入。
- **不改现有 parser**：`parse_imaging_study`、`parse_laboratory_report`、`parse_hpi_timeline` 保持原样。
- **不改现有 CLI 参数**：新增 `--case-input` 参数，现有 `--dicom-dir`、`--labs` 等继续可用。
- 支持**文件路径**和**内联文本**两种实验室/病程输入方式，方便 API 集成。

---

## JSON Schema

见 `schemas/case_input.schema.json`。核心结构：

```json
{
  "schema_version": "1.0.0",
  "case_id": "CASE_001",
  "patient_id": "DEMO_HCC_001",
  "index_date": "2026-07-20",
  "imaging": {
    "dicom_dir": "path/to/dicom",
    "seg": "path/to/seg.dcm",
    "phase": "portal_venous",
    "image_evidence": "path/to/image_evidence.json"
  },
  "laboratory": {
    "source_type": "file",
    "file_path": "examples/lab_report.synthetic.txt"
  },
  "hpi": {
    "source_type": "inline",
    "inline": {
      "format": "text",
      "content": "2026-01-15 行 TACE。\n2026-07-20 AFP 再次升至 96。"
    }
  },
  "llm_response": null,
  "output_dir": "case-output"
}
```

---

## 为什么这样设计

| 设计选择 | 理由 |
|---------|------|
| `imaging` 只接受文件路径 | DICOM/SEG 是二进制大文件，不适合内联；保持现有 `parse_imaging_study` 接口不变。 |
| `laboratory` / `hpi` 支持 `file` 或 `inline` | 文本类数据常从 API/数据库直接传入，内联可避免落盘；文件路径保留对本地测试友好。 |
| `image_evidence` 支持路径或内联对象 | 定性观察 JSON 体积小，可直接内联；也可保持现有文件引用方式。 |
| `index_date` 可选 | 默认使用影像学 `study_date`，与当前 `analyze-case` 行为一致。 |
| `case_id` 与 `patient_id` 分开 | `patient_id` 用于 DICOM 身份校验和 artifact；`case_id` 用于用户侧追踪（可等于 patient_id）。 |
| `llm_response` / `output_dir` 放入 JSON | 这样 JSON 就是完整运行契约，等价于一次 `analyze-case` 调用。 |

---

## 最小代码改动方案

### 1. 新增 loader（约 60 行）

在 `src/hcc_multimodal/case_input_loader.py` 中新增：

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .case_models import ClinicalLabEvidence, HpiTimelineEvidence, ImagingInterpretationEvidence


class CaseInputLoader:
    """Load a unified case-input JSON and materialize any inline content as temp files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self._temp_dir.name)

    def _resolve(self, value: str) -> Path:
        target = Path(value)
        if target.is_absolute():
            return target
        return self.path.parent / target

    def _materialize_inline(self, key: str, inline: dict[str, str]) -> Path:
        ext = {"text": ".txt", "json": ".json", "csv": ".csv"}[inline["format"]]
        target = self.temp_root / f"{key}{ext}"
        target.write_text(inline["content"], encoding="utf-8")
        return target

    @property
    def case_id(self) -> str:
        return self.payload["case_id"]

    @property
    def patient_id(self) -> str:
        return self.payload["patient_id"]

    @property
    def index_date(self) -> str | None:
        return self.payload.get("index_date")

    @property
    def dicom_dir(self) -> Path:
        return self._resolve(self.payload["imaging"]["dicom_dir"])

    @property
    def seg_path(self) -> Path | None:
        value = self.payload["imaging"].get("seg")
        return self._resolve(value) if value else None

    @property
    def phase(self) -> str | None:
        return self.payload["imaging"].get("phase")

    @property
    def image_evidence_path(self) -> Path | None:
        value = self.payload["imaging"].get("image_evidence")
        if value is None:
            return None
        if isinstance(value, dict):
            target = self.temp_root / "image_evidence.json"
            target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            return target
        return self._resolve(value)

    @property
    def labs_path(self) -> Path:
        lab = self.payload["laboratory"]
        if lab["source_type"] == "file":
            return self._resolve(lab["file_path"])
        return self._materialize_inline("labs", lab["inline"])

    @property
    def hpi_path(self) -> Path | None:
        hpi = self.payload.get("hpi")
        if hpi is None:
            return None
        if hpi["source_type"] == "file":
            return self._resolve(hpi["file_path"])
        return self._materialize_inline("hpi", hpi["inline"])

    @property
    def output_dir(self) -> Path:
        value = self.payload.get("output_dir", "case-output")
        return self._resolve(value)

    @property
    def llm_response(self) -> dict[str, Any] | None:
        value = self.payload.get("llm_response")
        if value is None:
            return None
        return json.loads(value) if isinstance(value, str) else value
```

### 2. CLI 改动（约 15 行）

在 `cli.py` 的 `analyze_case` 参数组中新增：

```python
analyze_case.add_argument("--case-input", default=None, help="Path to unified case-input JSON envelope")
```

在 `elif args.command == "analyze-case":` 分支开头增加：

```python
if args.case_input:
    from .case_input_loader import CaseInputLoader
    loader = CaseInputLoader(args.case_input)
    # 用 loader 的属性覆盖 CLI 参数
    args.dicom_dir = str(loader.dicom_dir)
    args.seg = str(loader.seg_path) if loader.seg_path else None
    args.image_evidence = str(loader.image_evidence_path) if loader.image_evidence_path else None
    args.labs = str(loader.labs_path)
    args.hpi = str(loader.hpi_path) if loader.hpi_path else None
    args.patient_id = loader.patient_id
    args.phase = loader.phase or args.phase
    args.output = str(loader.output_dir)
    if loader.llm_response:
        # 写入 output_dir 供后续读取，或扩展 CLI 直接传入 dict
        llm_response_path = loader.output_dir / ".llm_response.json"
        llm_response_path.write_text(json.dumps(loader.llm_response), encoding="utf-8")
        args.llm_response = str(llm_response_path)
```

之后的调用逻辑完全不变。

### 3. 可选：schema 校验

loader 初始化时可增加：

```python
from jsonschema import validate, ValidationError

schema = json.loads(Path(__file__).with_name("schemas") / "case_input.schema.json").read_text())
validate(self.payload, schema)
```

`jsonschema` 可作为可选依赖（`[json]` extra）。

---

## 示例：最小内联输入

```json
{
  "schema_version": "1.0.0",
  "case_id": "CASE_INLINE_001",
  "patient_id": "DEMO_HCC_001",
  "imaging": {
    "dicom_dir": "synthetic-case/study"
  },
  "laboratory": {
    "source_type": "inline",
    "inline": {
      "format": "text",
      "content": "2026-07-20 AFP 96 ng/mL 0-7\n2026-07-20 DCP 80 mAU/mL 0-40"
    }
  }
}
```

对应调用：

```powershell
hcc-demo analyze-case --case-input case_inline.json
```

---

## 迁移路径

1. 现有 `--dicom-dir`、`--labs` 等命令继续工作，无需迁移。
2. 新增 JSON 路径优先适配 `analyze-case`；后续可同样支持 `analyze`（NIfTI 路径）和 `run-deepseek`。
3. 如未来需要批量处理，JSON 天然适合数组封装：`[case_input_1, case_input_2, ...]`。

---

## 待确认问题

1. 是否需要支持 `laboratory.inline.content` 为 JSON 对象而不仅是字符串？（当前统一为字符串，保持 parser 的文本入口不变。）
2. `imaging.seg` 是否应支持 `inline_base64`？（建议先不支持，避免大 mask 内联。）
3. `output_dir` 放在 JSON 中是否合适，还是应继续由 CLI `--output` 控制？（本方案允许 JSON 指定，CLI `--output` 可覆盖。）
