# 虚拟 MDT 多科室会诊设计

## 1. 设计动机

现有管线是"文本 panel + 图像 panel"的二维融合，故事性较弱。升级为**虚拟 MDT（多学科会诊）**后：

- 每个科室是**信息受限的 Agent**（视野裁剪），观点天然存在差异
- 分歧被显性标注而不是被抹平，这正是真实 MDT 的价值
- 主治医师综合的是"观点"而非"新证据"，保持 fail-closed 哲学
- 看板从"数据展示"变成"会诊过程回放"，演示效果强很多

## 2. 架构（全部建在现有证据层之上，零侵入）

```
ImagingInterpretationEvidence ──┐
ClinicalLabEvidence ────────────┼→ Evidence Router（视野裁剪）
HpiTimelineEvidence ────────────┘         │
        ┌───────────┬─────────┬───────────┤
        ▼           ▼         ▼           ▼
     影像科      检验科      内科        外科
     (imaging)  (labs)    (labs+hpi)  (imaging quant + liver reserve)
        │           │         │           │
        └───────────┴────┬────┴───────────┘
                         ▼
              DepartmentOpinion × 4
              （统一 stance 词表）
                         │
                         ▼
              Consensus Engine（纯确定性）
              unanimous / majority / split / insufficient
                         │
                         ▼
              MDTVerdict（主治医师综合）
              不新增证据 · 只综合观点 · 分歧显性标注
                         │
                         ▼
              MDT 看板（单文件 HTML）
```

## 3. 科室视野定义（Evidence Scoping）

| 科室 | 输入视野 | 现实依据 | stance 决策来源 |
|------|---------|---------|----------------|
| 影像科 | ImagingInterpretationEvidence 全量 | 放射科写报告时通常不看肿瘤标志物 | 定量测量 + 定性观察 + 期相 QC；单一时间点 → insufficient |
| 检验科 | ClinicalLabEvidence 全量 | 检验科只解读血清学 | 标志物 trajectory_state、rebound_ratio、ALBI |
| 内科 | ClinicalLabEvidence + HpiTimelineEvidence | 内科医生核心关注治疗史与趋势 | 治疗锚点 + 标志物轨迹 + liver_function_state |
| 外科 | Imaging 定量 + LiverReserveEvidence + 治疗日期 | 外科关注解剖可切除性 + 肝功能储备 | 病灶数/大小/位置 + ALBI/Child-Pugh + 治疗史 |

**关键**：每个科室的 stance 必须由自己视野内的证据推导，且记录 `evidence_refs`（如 `AFP.rebound_ratio`、`imaging.quantitative_measurements`），保证可审计。

## 4. 统一 stance 词表

```python
Stance = Literal[
    "progression_suspected",   # 倾向进展
    "response_suspected",      # 倾向缓解/治疗后响应
    "stable",                  # 倾向稳定
    "insufficient_evidence",   # 证据不足，不表态
]
```

所有科室必须落到这 4 个枚举之一。**共识引擎因此是纯确定性的**——不需要 LLM 评判谁对谁错。

## 5. Schema 设计

### DepartmentOpinion

```json
{
  "schema_version": "1.0.0",
  "department": "radiology | laboratory | internal_medicine | surgery",
  "patient_id": "DEMO_HCC_001",
  "index_date": "2026-07-20",
  "stance": "progression_suspected",
  "confidence": "high | moderate | low",
  "summary": "一句话观点",
  "key_findings": ["关键发现1", "关键发现2"],
  "concerns": ["担忧点"],
  "evidence_refs": ["AFP.rebound_ratio", "timeline.treatment_dates"],
  "scoped_evidence_summary": "该科室实际看到的证据切片描述",
  "questions_to_others": ["想问其他科室的问题（可选，第二轮用）"],
  "renderer": "deterministic | external_llm",
  "quality": { "status": "pass | warning | fail" }
}
```

### ConsensusMatrix + MDTVerdict

```json
{
  "schema_version": "1.0.0",
  "patient_id": "DEMO_HCC_001",
  "index_date": "2026-07-20",
  "department_opinions": [ "<DepartmentOpinion>", "..." ],
  "agreement_pairs": [
    { "pair": ["laboratory", "internal_medicine"], "relation": "agree | disagree | not_comparable", "reason": "stance 相同/不同/一方 insufficient" }
  ],
  "consensus_level": "unanimous | majority | split | insufficient",
  "final_stance": "progression_suspected",
  "synthesis": "主治医师综合段落",
  "explicit_disagreements": ["外科与检验/内科不一致：关注点差异说明"],
  "unresolved_questions": ["影像单一时间点，缺乏纵向对照"],
  "requires_human_review": true,
  "research_disclaimer": "Research evidence summary only; not for diagnosis, staging, prognosis, or treatment decisions."
}
```

## 6. 共识计算规则（确定性）

1. `insufficient_evidence` 的科室不参与投票，但会在 `unresolved_questions` 中体现。
2. 剩余科室 stance 统计：
   - 全部一致 → `unanimous`
   - ≥2/3 一致 → `majority`
   - 对半分 → `split`
   - 有效科室 < 2 → `insufficient`
3. `final_stance` = 多数派 stance；`split` 时为 null 且必须标注。
4. 所有 stance 不同的科室对记入 `explicit_disagreements`，附"关注点差异"解释模板。

## 7. 科室 Agent 实现：先确定性模板，后可选 LLM

### 模式 A（默认，CPU 可用）

每个科室一个确定性函数，从视野证据映射到 stance + 模板化 summary：

- 影像科：`single_timepoint → insufficient`；有 longitudinal → 按 category 映射
- 检验科：`rebound_after_nadir / persistent_rising → progression_suspected`；`falling_after_treatment → response_suspected`；无足够点 → insufficient
- 内科：治疗锚点存在 + 标志物轨迹 + overall_trend 综合
- 外科：病灶数/大小 + ALBI grade + 治疗史 → stable / progression_suspected

模板文案中**数值全部来自 evidence_refs**，不经过 LLM，天然满足 locked-field 约束。

### 模式 B（可选 LLM 润色）

复用现有 `case_llm.py` 的 fail-closed 校验模式：

- 输入：科室视野证据 JSON + 确定性 stance（锁定字段）
- LLM 只允许润色 `summary` / `key_findings` 措辞
- 校验：stance、confidence、evidence_refs、所有数字必须一致；禁止诊断/分期/预后/治疗建议（复用 `_FORBIDDEN` 正则）
- 失败 → 回退模式 A 模板，`renderer` 字段记录实际渲染器

## 8. 与现有代码的映射（新增 3 个文件，改 1 处 CLI）

| 文件 | 内容 | 复用 |
|------|------|------|
| `src/hcc_multimodal/mdt_models.py`（新增） | Stance、DepartmentOpinion、AgreementPair、MDTVerdict Pydantic 模型 | 继承 `ArtifactModel`、`JsonModel`，与现有 schema 风格一致 |
| `src/hcc_multimodal/mdt_agents.py`（新增） | Evidence Router + 4 个科室确定性 Agent + 可选 LLM 润色 | 复用 `case_llm.py` 的校验思路 |
| `src/hcc_multimodal/mdt_consensus.py`（新增） | agreement_pairs + consensus_level + synthesis 模板 | 纯函数，无外部依赖 |
| `src/hcc_multimodal/cli.py`（改动） | 新增 `hcc-demo mdt-case` 子命令，参数与 `analyze-case` 相同 | 内部先调现有 3 个 parser，再进 MDT 层 |
| `demo-output/mdt-dashboard/index.html`（新增） | 单文件深色看板，读 `mdt_verdict.json` 渲染 | 复用现有 dashboard 风格 |

数据流完全复用：`analyze-case` 已能产出 3 个证据 JSON，`mdt-case` 在其后追加 MDT 层即可。

## 9. 与既有约束的一致性

- **fail-closed**：任何科室 QC fail → 该科室 stance = insufficient_evidence，不阻塞其他科室
- **不诊断/不分期/不建议治疗**：stance 词表全部是"倾向/证据不足"，synthesis 模板过滤 `_FORBIDDEN` 词汇
- **研究声明**：MDTVerdict 与看板底部固定 `research_disclaimer`
- **可追溯**：每个 key_finding 必须挂 evidence_ref；LLM 模式校验数字一致性
- **MRI 边界**：影像科对 MR 仍遵守现有 MR SEG fail-closed 边界

## 10. 两轮会诊（可选扩展，二期）

第一轮：各科室只看自己的视野（当前设计）。
第二轮：各科室看到其他科室的第一轮观点，可：
- 修正自己的 stance（必须给理由）
- 提出 `questions_to_others`（如检验科问影像科：病灶是否也在增大？）

第二轮能进一步放大"会诊过程"的演示价值，但一期先把单轮 + 共识 + 看板做扎实。

## 11. 待确认

1. 科室划分是否就定为 影像/检验/内/外 4 个？是否加"病理科"（目前病理只有 HPI 文本，证据太薄，建议暂不加）。
2. stance 词表 4 个枚举是否够用？是否需要 `mixed_signals`（部分指标升部分降）？
3. 看板是否就按"4 科室卡片 + 一致性矩阵 + 主治医师综合"的布局？是否需要第二轮时间轴视图？
4. LLM 润色是一期就上（复用 GLM-4-flash），还是先纯确定性模板跑通再加？
