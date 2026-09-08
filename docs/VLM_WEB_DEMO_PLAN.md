# HCC 3D VLM 真实推理 Web Demo 计划（v2）

> v2 变更：采纳修正案 A1–A4（两步输出、上下文隔离、期相人工确认、禁用推理模型做受控渲染）、
> B1–B4（病例入选标准、专家复核 checklist、服务契约三补、隐私措辞）、C1–C3（M1.5 干跑、
> 演示视频脚本、许可证脚注）；新增算力策略说明（全程推理，无需训练资源）。

## 总结

首版目标锁定为：

```text
单期HCC CT
+ HPI
+ 检验结果
  -> 本地去标识化和3D预处理
  -> 租用GPU上的 M3D-LaMed-Phi-3-4B（仅推理，不训练）
  -> 英文自由文本影像观察（Pass 1，纯影像）
  -> 受控提取为结构化 VLM 证据（Pass 2，非推理模型）
  -> 中文Web界面与审计报告
```

不训练模型、不支持MRI和多期融合、不宣称诊断或临床性能。原始DICOM留在本地，只上传
归一化的去标识3D张量。Mock Demo继续保留，但与真实VLM入口严格分开。

## 算力策略（v2 新增）

- 全流程为**纯推理**，无训练环节；M3D-LaMed-Phi-3-4B fp16 权重约 8GB，24GB 显存充足。
- 本地 32GB 内存电脑承担：DICOM/SEG 处理、HPI/检验解析、Web 服务、受控提取、审计。
- GPU 按小时租用（AutoDL 4090 24GB 约 2 元/小时级别）：
  - M1 探针与联调：约 10–20 小时
  - M4 五例演示与对照实验：约 10 小时
  - M2/M3 开发期间 GPU 关机，Web 对 mock endpoint 开发，验收时才开机。
- M5 若未来做 QLoRA 微调 4B 模型，单卡 24GB 仍可租用完成，不构成资源门槛。

## 实施方案

### 1. 建立稳定基线

- 整理当前31项未提交改动，修复README、Python测试和Demo中已有的乱码/编码问题。
- 完整运行pytest、Ruff和Mypy，建立`0.4.0`基线commit/tag。
- `vlm-skeleton-web`明确标记为Mock；现有自建projector和训练代码保留为`experimental`，不进入真实Demo。
- 新增真实入口：

```powershell
hcc-demo vlm-web `
  --endpoint https://rented-gpu.example/v1 `
  --port 7860
```

### 2. 本地Web和影像处理

- 页面上传DICOM ZIP、可选SEG、HPI文本和检验报告，并选择研究假名。
- 扫描PatientName、AccessionNumber、Institution、医师和地址等PHI；命中即阻断。
- 单期CT优先：自动列出候选series。**首版一律由用户人工确认门静脉期**（修正案A3）；
  "期相自动可靠判断"降级为M5之后的优化项，不进入首版验收。
- 完整CT按DICOM几何转换，使用固定腹部窗`[-200, 300] HU`归一化到`[0,1]`，处理为`[1,32,256,256]`。
- VLM输入使用完整腹部体数据，不根据SEG裁剪，避免只有存在分割时才能运行。SEG只用于病灶数量、体积和位置交叉校验。
- HPI与检验在本地走现有解析器，转换成英文规范化证据列表；日期、数值、单位和来源ID保持锁定。
- 远程只接收压缩3D张量、结构化临床上下文和任务配置，不接收DICOM、姓名、原始文件路径或SEG。

### 3. 租用GPU推理服务

- 使用完整`GoodBaiBai88/M3D-LaMed-Phi-3-4B`，固定revision `329bed510c2cff0ad3fe6f2a1a7075f45e2706e9`，不拆换vision encoder或projector。
- 按官方接口生成256个`<im_patch>`，调用`model.generate(image_tensor, input_ids)`；`do_sample=false`，`max_new_tokens=512`。
- 服务提供：
  - `GET /health`：模型ID、revision、设备、dtype和加载状态。
  - `POST /v1/infer`：接收`volume.npz`和`request.json`，返回原始响应、解析报告、耗时与模型审计信息。
- 服务契约补充（修正案B3）：
  - `request_id` 幂等键：重复提交返回缓存结果，支撑"逐字节一致"验收。
  - prompt模板版本号写入审计（模型revision锁定不代表prompt不变）。
  - 单GPU串行队列，返回排队位置；并发请求不得交错或OOM。
- 使用Bearer Token和HTTPS/SSH隧道；服务绑定私有网络，请求结束立即清理临时文件，不记录体数据。
- **VLM 两段式输出（修正案A1+A2，核心架构变更）：**
  - Pass 1（纯影像）：`image + 中性问题`（描述腹部CT所见），产出英文自由文本
    `imaging_observation`。prompt 中**不含** HPI/检验，防止上下文数字泄漏进影像观察。
  - Pass 2（上下文整合）：Pass 1 结果 + 结构化 HPI/检验证据，产出 consistency /
    uncertainty / missing 字段。对照实验只动 Pass 2 输入。
  - Pass 2 的受控提取使用**非推理模型**（qwen-plus / deepseek-v3 / 本地小模型），
    禁用 r1 类推理模型（实测：temperature=0 数字重复循环、reasoning 烧 token 截断正文）。
    r1 类模型如需展示，只放"自由分析"位，不进审计链（修正案A4）。
- 模型输出经严格`VlmDemoReport`校验：影像观察、临床上下文摘要、证据一致性、不确定性、
  缺失信息、图像条件化说明和固定研究声明。原始响应始终保留展示；提取层失败只影响结构化视图，
  不影响真实输出展示。
- 出现畸形JSON、额外字段、数字篡改、诊断、分期、预后或治疗建议时，不生成正式报告，只保留受限的原始响应和错误审计。

### 4. Web展示重点

- 输入区：DICOM、SEG、HPI、检验、去标识化确认。
- 处理区：series选择（含人工期相确认）、几何QC、三视图预览、输入张量形状、模型revision、256个视觉token状态。
- 输出区：Pass 1 原始英文观察、Pass 2 结构化JSON、英文研究报告、安全审计。
- 中文界面解释各字段，但不对模型英文自由文本做隐式翻译。
- 增加对照实验（均在 Pass 2 层面操作）：
  - 同一CT搭配不同合成AFP/DCP和HPI，观察临床上下文如何改变综合输出。
  - 相同文字搭配不同CT，观察影像观察是否变化。
  - 零张量或错误影像作为负控；若模型输出不敏感，明确显示"未证明图像条件化"，不得伪装成成功。

## 数据集

| 数据 | 首版用途 | 是否必需 |
|---|---|---|
| [M3D-LaMed-Phi-3-4B](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B) | 已对齐的3D encoder、projector和LLM权重（仅推理） | 必需 |
| [HCC-TACE-Seg](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/) | 105例HCC多期CT、治疗前后影像和专家SEG；首版选择5例通过QC的门静脉期病例 | 必需 |
| 合成HPI/AFP/DCP场景 | 对同一影像模拟正常、升高、治疗后反弹等上下文；始终标记为不配对合成数据 | 必需 |
| [WAW-TACE](https://zenodo.org/records/12741586) | 233例、377个肿瘤分割、四期CT和临床结局，用于第二阶段丰富公开测试 | 后续 |
| [TCGA-LIHC](https://www.cancerimagingarchive.net/collection/tcga-lihc/) | 97例异质CT/MR/PT及临床、病理、组学关联 | 后续 |
| [M3D-Data](https://github.com/BAAI-DCAI/M3D) | 120K影像文字对和662K指令对；仅在以后重新训练对齐层时需要 | 首版不下载 |

**5例 HCC-TACE-Seg 入选标准（修正案B1，写死）：** 单发病灶；SEG与CT几何对齐；
层厚 ≤3mm；无明显运动/金属伪影；非治疗后即刻影像（避免栓塞剂干扰HU值）。

HCC-TACE-Seg没有完整配对HPI、放射科报告、DCP、AFP-L3%或原始肝功能面板，
但其临床表包含105例同受试者AFP、年龄、性别、OS和死亡状态。因此原VLM演示仍使用
明确未配对的纵向合成AFP/DCP场景；新增的OS研究链则使用真实配对AFP，并将结局与模型输入物理隔离。

## 技术与人员支持

- 本地：现有32GB内存电脑负责DICOM、SEG、检验/HPI解析和Web页面；Intel 2GB显卡不承担模型推理。
- GPU：按小时租用 24GB 显存实例（4090 级别）即可；48GB L40S/A6000 为舒适选项非必需。
  至少64GB系统内存、80GB磁盘。
- 环境：Ubuntu 22.04、CUDA 11.8兼容驱动、PyTorch 2.2.1、Transformers 4.39.1、MONAI 1.3.0，并将镜像和模型revision锁定。
- 存储：模型仓库约30.4GB；公共DICOM和模型权重不提交GitHub，只提交下载清单、哈希和许可证信息。
- 人员：一名工程人员可完成系统集成；至少需要一名肝脏影像专家复核5例Demo。
- 专家复核使用结构化 checklist（修正案B2）：每条 VLM 观察标注
  {正确 / 过度解读 / 遗漏 / 幻觉}，并判断"是否超出工程演示边界"；
  产出 `docs/EXPERT_REVIEW.md`，作为仓库公开的可信材料。
- 开源：公开代码、合成输入和小型派生预览；不公开原始患者数据、模型权重、API令牌或生成的高维视觉token。
  发布前核对 M3D-LaMed 代码与权重 license、TCIA 引用要求，写入 NOTICE（修正案C3）。
- 隐私措辞（修正案B4）：上传物为"不含 DICOM 头信息与身份信息的归一化张量，
  残余风险为解剖结构本身"，不使用绝对化"匿名"表述。

## 测试与验收

- 预处理验证方向、spacing、HU窗、有限值、形状`[1,32,256,256]`和范围`[0,1]`。
- PHI、混合series、错误期相、无效DICOM、空体数据和上传中断必须阻断。
- 使用官方M3D样例先证明完整checkpoint可推理，再运行现有`HCC_003`。
- 审计必须记录模型ID/revision、输入张量SHA-256、256个图像占位符、prompt模板版本、推理耗时和安全结果，但不保存visual token本体。
- 云端冷启动目标不超过5分钟，单病例热推理目标不超过90秒；超时返回结构化错误。
- 正式报告必须通过Pydantic和安全规则；失败时不回退到Mock报告。
- 同一输入重复运行时，本地预处理和结构化请求逐字节一致（含 request_id 幂等）；模型使用确定性生成。
- GitHub Demo明确说明：使用通用3D医学VLM和公开HCC影像，仅验证工程链路，不代表HCC诊断能力。
- 预期管理：M3D-LaMed 训练分布不含 HCC 门静脉期专科任务，首版输出可能为泛化描述；
  **成功标准是链路真实可审计，而非描述惊艳**——"观察平庸"也算演示成功。

## 里程碑

1. **M0 基线清理**：编码修复、测试通过、`0.4.0`基线tag。
2. **M1 GPU探针**：官方样例和`HCC_003`完成真实M3D-LaMed推理（租用GPU，约10–20小时）。
3. **M1.5 单病例干跑**（修正案C1）：HCC_003 走通"本地预处理 → 服务推理 → 审计落盘"
   全链路（手工 curl 即可），最早暴露张量格式/显存/超时问题。
4. **M2 服务契约**：私有GPU API、认证、幂等、串行队列、审计和失败关闭。
5. **M3 Web整合**：DICOM上传、HPI/检验、期相人工确认、三视图和真实模型结果页（对 mock endpoint 开发）。
6. **M4 可信演示**：5例公开CT（按入选标准）、对照实验、专家 checklist 复核、
   `EXPERT_REVIEW.md`、3分钟演示录屏（修正案C2，兼作 README GIF）和 GitHub 文档。
7. **M5 后续研究**：评估WAW-TACE；只有获得合法配对队列后才考虑HCC LoRA或projector微调
   （QLoRA 单卡24GB可租用完成）；期相自动判断优化亦在本阶段评估。
