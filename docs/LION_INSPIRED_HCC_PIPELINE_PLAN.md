# LiON-inspired HCC 多模态报告系统：实施与运行手册

> 状态：第一版已实现。本文既记录设计边界，也可直接作为本地运行、验收和未来替换真实 LiON/PLAN 后端的操作手册。

## 1. 目标、边界与当前实现

当前可信链路为：

```text
单期 3D CT
  ├─ LiON-inspired / precomputed-mask
  │    像素级掩膜 QC → 连通病灶定量 → 患者级汇总
  └─ GLM image-only observer
       16 层蒙太奇 + 最多 3 张病灶定位放大图
                         ↓
                 确定性影像交叉检查
                         +
                 标准化检验与 HPI 治疗锚点
                         ↓
                    确定性规则裁决
                         ↓
              DeepSeek 受控中文报告渲染
                         ↓
              JSON + Markdown + Web + 审计
```

本实现借鉴 LiON 的像素级、病灶级、患者级证据层次，以及“额外阅片者”的协作思想，但没有运行或声称复现临床 LiON。当前后端只测量用户提供且通过几何校验的 SEG，不自动分割、不输出恶性概率、HCC 分类概率、LI-RADS、分期、预后或治疗建议。

参考：[LiON 论文](https://www.nature.com/articles/s41591-026-04589-y)；[公开 PLAN 框架](https://github.com/alibaba-damo-academy/pixel-lesion-patient-network)。

## 2. 可直接执行的最短路径

### 2.1 安装和离线运行

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"

hcc-demo run-report `
  --case-input examples\case_input.recommended.json `
  --output case-output `
  --glm-mode off `
  --report-mode deterministic
```

该命令不会发起外部网络请求。`--output` 优先于病例 JSON 的 `output_dir`。

### 2.2 配置外部模型

密钥只放在当前用户环境变量中，不写入 `.env`、病例 JSON、日志、提示词或仓库：

```powershell
[Environment]::SetEnvironmentVariable("ZHIPU_API_KEY", "<rotated-key>", "User")
[Environment]::SetEnvironmentVariable("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4", "User")
[Environment]::SetEnvironmentVariable("ZHIPU_VISION_MODEL", "glm-4.6v-flash", "User")

[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "<deepseek-key>", "User")
[Environment]::SetEnvironmentVariable("DEEPSEEK_BASE_URL", "https://api.deepseek.com", "User")
[Environment]::SetEnvironmentVariable("DEEPSEEK_MODEL", "deepseek-v4-flash", "User")
```

如果 DeepSeek 模型由阿里云百炼提供，也继续使用 `DEEPSEEK_*` 作为本项目的角色配置，
但把 Key/Base URL 设置为百炼值。客户端会根据端点方言发送
`enable_thinking=false`；DeepSeek 官方端点则发送
`thinking.type=disabled`。账号必须已经开通所配置模型的调用权限。

重新打开终端后运行：

```powershell
hcc-demo run-report `
  --case-input examples\case_input.recommended.json `
  --output case-output-live `
  --glm-mode live `
  --report-mode live `
  --require-live-models
```

`--require-live-models` 要求 GLM 和 DeepSeek 都实际调用成功且输出通过确定性校验，否则命令返回非零退出码，但保留全部可生成的证据和审计产物。曾在聊天、终端或截图中出现过的密钥必须先轮换。

## 3. `case_input 1.2.0` 合同

```json
{
  "schema_version": "1.2.0",
  "case_id": "CASE_001",
  "patient_id": "RESEARCH_001",
  "clinical_task": "recurrence_surveillance",
  "index_date": "2026-07-20",
  "data_relationship": {
    "imaging_origin": "real_public",
    "laboratory_origin": "synthetic",
    "pairing_status": "unpaired_poc_composite",
    "statement": "公开影像与合成检验仅用于软件演示"
  },
  "imaging": {
    "source_type": "nifti",
    "modality": "CT",
    "nifti_image": "ct.nii.gz",
    "dicom_dir": null,
    "seg": "tumor_mask.nii.gz",
    "seg_role": "public_reference",
    "study_date": "2026-07-20",
    "phase": "portal_venous"
  },
  "laboratory": {
    "source_type": "file",
    "file_path": "labs.txt"
  }
}
```

执行规则：

- `source_type=dicom` 只允许 `dicom_dir`；`source_type=nifti` 只允许 `nifti_image`。
- `run-report` 第一版只接受一次 3D CT 和一个期相。MR 使用旧版证据摘要入口，不得标记为 LiON-inspired。
- `seg` 可选；提供时必须声明 `expert_reference`、`public_reference`、`user_supplied` 或 `model_prediction`。
- 缺少、空白或校验失败的 SEG 均生成 unavailable 证据，病灶数为 `null`，绝不生成“0 个病灶”的阴性结论。
- 公开影像和合成检验必须声明 `unpaired_poc_composite`；规则层锁定“不形成患者级诊断解释”。
- 1.0/1.1 输入继续支持；缺少数据关系时归一为 `user_supplied_unverified` 并强制人工复核。

完整 JSON Schema 位于 `schemas/case_input.schema.json`；可运行样例位于 `examples/case_input.recommended.json`。

## 4. 三类影像证据

### 4.1 `LionInspiredImagingEvidence`

- 后端固定标识为 `precomputed_mask`。
- 像素级记录掩膜角色、SHA-256、几何对齐、体素间距与质量。
- 病灶级按 26 邻域连通区域形成匿名 `LESION_###`，记录体素数、体积、最大三维包围范围、质心和体素包围框。
- 患者级记录病灶数量、总体积和最大病灶范围。
- `diagnostic_capability` 永远为 `not_available`。

未来接入 PLAN 或真实 LiON 时，只实现 `LionBackend.infer()` 并输出同一证据合同；下游交叉检查、融合、提示词和报告无需改动。

### 4.2 `GlmImagingEvidence`

GLM 只看到从 NIfTI 重新渲染的 PNG、匿名 `LESION_###` 定位和期相声明，不看到检验、HPI、患者编号、文件路径、精确体积、精确最大径、风险或分类结果。

以下情况阻断整份 GLM 证据：

- JSON 多字段、少字段或类型不符；
- 未知病灶 ID 或图片引用；
- 诊断、分期、预后、治疗建议；
- 任何无来源数值或测量；
- 提示词回显或安全边界断言。

API 失败时生成 `GLM_UNAVAILABLE`，不使用模板伪造视觉观察。

### 4.3 `ImagingCrosscheckEvidence`

交叉检查只验证 GLM 观察能否追溯到 LiON-inspired 病灶，状态为 `pass`、`warning`、`fail`、`not_comparable` 或 `unavailable`。它不判断 GLM 医学语义是否正确，也不把 LiON 与 GLM 当作两个独立模态进行概率平均或置信度加权。

## 5. 融合和报告职责

确定性规则层拥有唯一裁决权，产生锁定的：

- `state` 和 `modality_concordance`；
- 支持、冲突和缺失证据；
- reason codes；
- `requires_clinician_review`；
- 数据关系和研究用途声明。

主要新增 reason codes：

```text
LION_QUANT_AVAILABLE
LION_MASK_UNAVAILABLE
GLM_QUAL_AVAILABLE
GLM_BLOCKED
SINGLE_PHASE_LIMIT
UNPAIRED_POC_COMPOSITE
IMAGING_CROSSCHECK_WARNING
HPI_TREATMENT_ANCHOR_AVAILABLE
```

DeepSeek 只接收紧凑 LiON 定量摘要、已验证 GLM 定性观察、标准化 AFP/DCP 趋势、规则裁决、数据关系和缺失信息。原始 CT、PNG、坐标、掩膜、路径、高维向量和未验证模型原文不会进入 DeepSeek 提示词。

DeepSeek 输出必须通过 `ControlledReport`、锁定字段、证据列表、数字白名单和安全边界校验。失败时使用确定性 Markdown 兜底并标记 `degraded`；严格实时验收不把兜底算作成功。

## 6. 每次运行的固定产物

```text
case-input.normalized.json
imaging-input-qc.json
lion-inspired-evidence.json
glm-imaging-evidence.json
imaging-crosscheck.json
clinical-lab-evidence.json
clinical-verdict.json
deepseek-prompt.json
controlled-report.json
controlled-report.md
pipeline-audit.json
web_demo.json
provider-audit/glm.json
provider-audit/deepseek.json
```

如提供 HPI，还会生成 `hpi-timeline-evidence.json`。提供方审计只记录提供方、模型、耗时、调用/响应状态、提示词版本、用量、验证错误和错误码；程序会丢弃原始模型正文、请求体及任何可能的密钥字段。

## 7. Web 入口

```powershell
hcc-demo vlm-web --port 7861
```

浏览器打开 `http://127.0.0.1:7861/`。`POST /api/interpret` 调用与 CLI 相同的 `run_report_pipeline()`，同时保留旧 `interpretation` 和 `web_demo` 字段；新增 `pipeline` 与 `artifact_index`。页面展示：

- LiON-inspired 像素/病灶/患者三级证据；
- GLM 状态、模型和通过审计的观察；
- 影像交叉检查；
- 检验趋势、规则裁决和最终受控报告；
- 数据配对与降级横幅；
- 可折叠技术附录。

Mask 在单时间点模式下可选。实时 GLM 或 DeepSeek 开关只有在用户勾选“只外发重新渲染的去标识化 PNG”确认后才允许执行。`GET /api/health` 只返回服务版本、模型名和配置是否存在，不返回密钥。

## 8. 测试与验收

提交前运行：

```powershell
python -m pytest -q
python -m ruff check src tests
python -m mypy src\hcc_multimodal
python -m pytest --cov=hcc_multimodal --cov-branch --cov-report=term-missing
```

核心验收覆盖：

- 1.0/1.1 兼容及 1.2 DICOM/NIfTI 互斥、SEG role、数据关系；
- 两个连通区域形成两个病灶，空/缺失/错位掩膜 fail-closed；
- GLM 请求只有 PNG data URL 和影像提示，无路径、患者编号和检验；
- 未知病灶、诊断和数值幻觉被阻断；
- 单期、缺 SEG、GLM 不可用和未配对声明进入缺失证据；
- 固定报告字段和证据列表不可被 DeepSeek 改写；
- CLI/Web 都调用同一编排器；健康检查和审计不泄露密钥。

真实 API 冒烟只允许使用合成或公开数据：

1. `provider=Zhipu`、视觉模型为配置的 `ZHIPU_VISION_MODEL`，至少发送一张蒙太奇；有 SEG 时至少发送一张病灶放大图。
2. JSON 模式 DeepSeek 返回严格 `ControlledReport`；仅思考的 R1 Distill 只返回通过独立安全校验的
   可选中文辅助解读，权威 `ControlledReport` 仍由确定性模板生成并锁定。
3. `--require-live-models` 返回 0，固定产物齐全，Web 展示研究用途、单期限制和配对状态。

## 9. 实施状态和后续替换点

- [x] case_input 1.2、旧版兼容、归一化清单
- [x] DICOM/NIfTI 单期 CT 归一化
- [x] `LionBackend` 与 `PrecomputedMaskLionBackend`
- [x] 16 层蒙太奇、病灶放大图和独立 `ZHIPU_*` 客户端
- [x] GLM 严格结构/数值/边界校验与失败审计
- [x] 确定性影像交叉检查和保守融合 reason codes
- [x] 独立 `DEEPSEEK_*` 配置、旧 `LLM_*` 弃用回退、受控报告
- [x] `run-report` CLI、Web 共用编排、固定产物和模型健康状态
- [x] 合成及公开 CT＋合成检验样例
- [x] 使用实际 Zhipu/DeepSeek Distill 凭据完成一次严格实时冒烟并保存脱敏审计
- [ ] 实现 `PlanLionBackend` 或获得许可的真实 LiON 接口
- [ ] 下一阶段多期识别、配准和动态强化证据合同

第一版必须始终显示 `LiON-inspired / precomputed-mask`，不得显示“LiON 诊断结果”。

### 2026-08-25 真实接口冒烟记录

- 智谱用户级凭据和端点有效，请求实际包含 1 张 16 层蒙太奇与 3 张病灶放大 PNG，模型为 `glm-4.6v-flash`。首次调用被提供方以 HTTP 429/1305 限流；有限重试返回多个 JSON 片段，因不是单一严格 JSON 对象而被 `GLM_OUTPUT_BLOCKED` 拒绝。未验证正文没有进入融合或报告。
- 当前环境没有独立 `DEEPSEEK_*`，兼容的 `LLM_*` 模型为推理型 `deepseek-r1-distill-qwen-32b`。真实调用正文不含 JSON，外部输出被 `DEEPSEEK_OUTPUT_BLOCKED` 拒绝；确定性报告兜底通过全部锁定字段校验并标记 `degraded`。
- 两条调用均未泄露密钥，证明失败审计和降级链路有效；由于没有得到通过验证的两份外部输出，`--require-live-models` 的退出码 0 验收仍未完成。下一次验收应使用轮换后的智谱密钥和独立、非思考 JSON 模式的 `DEEPSEEK_*` 配置。

### 2026-08-29 重试与提供方方言修正

- GLM 增加仅针对 408/409/425/429/5xx 和连接错误的有限指数退避重试；畸形 JSON
  允许重新请求一次，Schema 不合格允许携带错误码重新生成一次，但任何一次不合格正文都不会被
  修补或进入融合。提示词新增完整合法 JSON 示例和字段类型枚举。
- 使用相同公开 CT 进行真实调用，实际发送 1 张 16 层蒙太奇与 3 张病灶放大 PNG。
  `glm-4.6v-flash` 在两次临时错误后成功，约 32 秒返回 3 条通过验证的观察；交叉检查覆盖
  `LESION_001` 至 `LESION_003`，对未提供放大图的 `LESION_004/005` 保留 warning。
  该结果说明主要故障来自提供方波动和结构化输出遵循，而非三维体数据负担。
- DeepSeek 客户端区分官方方言与百炼方言。百炼 R1/蒸馏版不进入受控 JSON 生成器，而是进入
  可选中文辅助解读通道；不关闭 thinking、不请求 JSON、不记录 reasoning。现有百炼 Key 对
  `deepseek-v4-flash`、`deepseek-v3.2` 和 `deepseek-v3` 均返回 HTTP 403
  `Model.AccessDenied`，因此这些模型仍需另行开通权限。

### 2026-08-29 HCC 第一版严格端到端验收

- `--glm-mode live --report-mode live --require-live-models` 使用公开 CT、公开 SEG 与明确未配对的
  合成检验运行，命令退出码为零；`pipeline-audit.json` 为 `pass` 且
  `strict_success=true`，总耗时约 42 秒。
- `glm-4.6v-flash` 实际接收 1 张蒙太奇与 3 张病灶放大 PNG，一次调用通过，返回 3 条可追溯
  观察；LiON-inspired 共保留 5 个病灶区域，因此交叉检查对 `LESION_004/005` 保留 warning，
  不把未覆盖解释为无病灶。
- `deepseek-r1-distill-qwen-32b` 接收已锁定且已去除具体数字的影像、检验、融合与质量摘要，
  一次调用通过；最终正文同时说明影像和 AFP/DCP 趋势、未配对演示、单期 CT 限制及人工复核。
  thinking 只以布尔审计记录是否存在，正文通过后才写入 `deepseek-narrative.json`、Markdown 和 Web。
- 权威 `controlled-report.json` 始终由确定性锁定字段构成；可选叙述出现数字、诊断、分期、治疗
  建议、遗漏跨模态证据或遗漏配对限制时，最多重新生成一次，仍不合格则阻断并降级。
