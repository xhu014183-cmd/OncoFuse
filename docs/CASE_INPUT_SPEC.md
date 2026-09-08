# HCC 多模态病例输入规范

## 1. 输入目标

系统的外部输入是一个**去标识化患者病例包**，不是内部模型特征，也不是预先生成的 `imaging.json` 或 `labs.json`。

用户应回答四件事：

1. 这是哪个研究病例；
2. 本次希望回答什么临床问题；
3. 哪一次 CT/MR 和哪些检验结果属于该患者；
4. 影像和检验之前发生过哪些关键治疗或手术。

系统负责解析、质量检查、指出缺口，并根据实际可用证据限制报告结论。

## 2. 当前可直接运行的边界

推荐入口 `hcc-demo run-report --case-input <json>` 当前处理：

- 一次单期 3D CT，可来自 DICOM 或 NIfTI；
- 可选 DICOM SEG 或 NIfTI 分割；
- LiON-inspired 像素/病灶/患者三级定量证据；
- 可选 GLM 去标识化影像观察；
- 一份可包含多个日期的检验 TXT、CSV 或 JSON；
- 可选病程/治疗时间线。

当前命令**不会**仅凭原始影像自动产生放射学诊断，也不会对两次检查自动完成疗效比较。没有 SEG 时 LiON-inspired 定量标为 `unavailable`，病灶数保持 `null`，不得解释为无病灶。MR 继续使用旧 `analyze-case` 证据摘要入口，不标记为 LiON-inspired。

`case_input 1.2.0` 新增 `imaging.source_type`、`nifti_image`、`seg_role` 和强制的 `data_relationship`。公开影像与合成检验必须声明 `unpaired_poc_composite`，报告永久显示其演示级未配对状态。

## 3. 输入分级

### A. 最低可运行

- `patient_id`：去标识化研究编号，且必须与 DICOM `PatientID` 一致；
- `clinical_task`：本次分析目的；
- `imaging.dicom_dir`：同一次检查的 CT/MR DICOM 目录；
- `laboratory`：至少一条可识别、有日期、有单位的检验结果。

这一级只能保证管线运行，不能保证形成有内容的影像解读。

### B. 推荐的自助报告输入

在最低输入基础上增加：

- 结构化影像观察 `imaging.image_evidence`，至少包含部位、观察、置信度和来源切片；
- AFP、DCP、AFP-L3%、白蛋白、总胆红素，尽量同时提供参考范围；
- 至少两个时间点的 AFP/DCP（需要趋势分析时）；
- `index_date`，用于排除该日期之后的数据；
- 关键治疗/手术日期和临床问题；
- 年龄和出生性别（仅在使用 GALAD 类研究评分时需要）。

### C. 定量研究增强

- 与 DICOM 同一患者、同一检查、同一空间参考的 SEG；
- 明确的增强期相；
- 用于纵向比较的基线与随访影像、配准结果及治疗锚点。

这一级适合定量研究。SEG 证明的是“被分割区域的几何量”，不能代替强化模式或放射科诊断。

## 4. 临床任务枚举

| `clinical_task` | 含义 | 当前单次病例管线可提供 |
|---|---|---|
| `screening` | 高危人群筛查 | 当前影像/标志物证据摘要与缺口 |
| `diagnostic_workup` | 已发现结节后的诊断性评估 | 结构化证据摘要；不自动作 HCC 诊断 |
| `treatment_baseline` | 治疗前基线 | 影像和检验基线记录 |
| `post_treatment_response` | 治疗后疗效评估 | 单时间点摘要；没有合格基线时禁止形成疗效结论 |
| `recurrence_surveillance` | 术后/消融后复发监测 | 趋势和当前证据摘要 |
| `unspecified` | 任务确实未知 | 仅输出保守的通用摘要 |

任务必须由提交者选择，不能仅凭 AFP 或影像文字自动猜测。

## 5. 检验数据规范

结构化 JSON 是首选格式：

```json
{
  "observations": [
    {
      "date": "2026-07-20",
      "marker": "AFP",
      "value": "96",
      "unit": "ng/mL",
      "referenceRange": "0-7"
    }
  ]
}
```

每一条检验至少应包含：

- 检验日期；
- 项目名称；
- 原始结果，保留 `<`、`>` 等比较符；
- 单位；
- 本实验室参考范围。

不要提前把“升高”“异常”当作数值传入，也不要把不同日期的结果覆盖成一条。系统当前重点识别 AFP、DCP/PIVKA-II、AFP-L3%、白蛋白、总胆红素、ALT、AST、ALP、GGT、PT/INR 和病毒学项目；无法识别或无法换算单位的项目会被保留为拒绝记录，不进入数值趋势。

## 6. 影像数据规范

- 一个 `dicom_dir` 应对应一次检查中的一个目标 CT/MR 序列；若目录包含多个序列，当前适配器会选择实例数最多的序列并给出警告。
- 优先保留原始 DICOM 元数据，不要只上传截图。
- `phase` 只作为提示，不能替代 DICOM 元数据和人工确认。
- `image_evidence` 必须是结构化观察，不能把整段报告冒充模型测量。
- `seg` 可选；没有 SEG 时系统不得生成病灶体积。

## 7. 身份、时间和隐私规则

- `patient_id` 必须是研究假名，不放姓名、身份证号、手机号或住院号。
- DICOM 中的 `PatientID` 必须与病例 JSON 一致；其他直接身份信息应在提交前去除。
- `index_date` 是本次报告的证据截止日；之后的检验或病程不得进入当前分析。
- 影像、检验、SEG 和结构化观察必须属于同一患者。不能把公开影像与合成检验包装成患者级报告。

## 8. 运行方式

```powershell
hcc-demo run-report --case-input examples\case_input.recommended.json --output case-output
```

`examples/case_input.minimum.json` 展示兼容的 DICOM 最低合同；`examples/case_input.recommended.json` 展示公开 CT＋合成检验的 1.2 未配对合同；`examples/case_input.synthetic-1.2.json` 展示全合成合同。

每次通过病例 JSON 运行时，输出目录还会写入
`case-input.normalized.json`，记录任务、上下文和本次实际提供了哪些资料，
但不复制 DICOM 路径或二进制内容。若没有单独的 `hpi`，结构化
`clinical_context.treatment_events` 会自动转换为时间线输入。

## 9. 报告能力由输入决定

| 可用资料 | 允许输出 |
|---|---|
| DICOM + 检验 | 数据质控、检验趋势、影像证据不足说明 |
| 上述 + 结构化影像观察 | 定性跨模态证据摘要 |
| 上述 + 合格 SEG | 病灶几何定量摘要 |
| 单次随访但无基线 | 不允许输出疗效反应 |
| 基线/随访 + 配准 + 治疗锚点 | 才具备纵向研究分析前提；当前需走独立纵向管线 |

输入不足应降低报告层级，而不是把缺失模态解释成阴性。
