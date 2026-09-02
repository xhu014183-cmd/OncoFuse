# HCC TACE Overall-Survival Model Card / 模型卡

## 中文

### 模型用途

本模型用于回顾性研究：比较首次TACE前的基础临床信息、公开肿瘤SEG所提取的
肿瘤负荷，以及两者融合后对总生存风险排序的能力。它不是临床预后工具，不能
用于决定治疗，也不输出患者还能生存多少个月。

### 开发与验证

- 开发队列：WAW-TACE v2，目标233例。
- 锁定外部验证：HCC-TACE-Seg v2，公开队列105例，104例通过预先指定的影像QC。
- 终点：首次TACE后的总生存期；死亡为事件。
- 主要模型：`clinical_core`、`imaging_core`、`fused_core`。
- 探索模型：WAW内部的`fused_extended`和去除白蛋白的敏感性模型。

主模型特征在训练前固定。数值变换、标准化、惩罚系数和中位风险阈值只在WAW
中确定；外部队列不参与选择或重新校准。模型包以JSON保存特征顺序、系数、
基线累积风险、训练版本和哈希，不使用Pickle。

### 输入与输出

核心输入是年龄、性别、AFP、病灶数、总体积、最大病灶三维包围盒范围及最大
病灶球形度。输出是相对风险指数、WAW参考百分位和以WAW中位数为界的研究级
分层。该分层只表示相对于开发人群的位置，不是绝对生存概率或治疗建议。

### 已知限制

数据来自回顾性公开队列，中心、年代、扫描协议和分割流程不同。模型依赖公开
参考SEG，不包含自动分割。最大范围是三维包围盒指标，不是RECIST/mRECIST径线。
WAW期相异质，门静脉HU只作探索。HCC-TACE-Seg缺少与WAW一致的完整肝功能面板。
任何比例风险假设警告、缺失、失败或外部无提升结果都必须保留。

### 2026-09-02正式验证结果

- WAW内部嵌套交叉验证C-index：临床0.5917、影像0.6047、融合0.6310。
- HCC-TACE-Seg一次性外部验证（104例，92个事件）：临床0.5860
  （95% CI 0.5065–0.6579）、影像0.5467（0.4746–0.6094）、融合0.5819
  （0.5131–0.6516）。
- 融合相对临床的差值为-0.0041（-0.0766–0.0671），没有显示外部增益；
  相对影像的差值为0.0352（0.0029–0.0691）。
- HCC_011因基线CT缺层/层距不均而在解锁外部结果前被排除，未做插值。
- 比例风险筛查对年龄、性别或病灶数发出警告；模型没有根据外部结果事后修改。

因此，当前结果支持“完成了可审计的外部验证”，不支持“多模态优于临床模型”
或临床性能声明。

## English

### Intended use

This model supports retrospective research comparing baseline clinical evidence,
public-SEG-derived tumor burden, and their fusion for overall-survival risk ranking
after first TACE. It is not a clinical prognostic device, treatment-selection tool,
or estimator of an individual's remaining lifetime.

### Development and validation

- Development: WAW-TACE v2, target n=233.
- Locked external validation: HCC-TACE-Seg v2, 105 public subjects and 104
  passing prespecified imaging QC.
- Endpoint: overall survival after first TACE, with death as the event.
- Primary models: `clinical_core`, `imaging_core`, and `fused_core`.
- Exploratory models: WAW-only `fused_extended` and the no-albumin sensitivity model.

Features are prespecified. Transformations, scaling, penalization, and the median-risk
threshold are determined only in WAW. The external cohort cannot tune or recalibrate
the model. Human-readable JSON bundles contain ordered features, coefficients,
baseline cumulative hazard, data version, and model hash; no opaque pickle is used.

### Inputs, outputs, and limitations

Core inputs are age, sex, AFP, lesion count, total tumor volume, maximum 3D bounding-box
extent, and largest-lesion sphericity. Outputs are a relative risk index, WAW reference
percentile, and research-only median split. These are not absolute survival claims.

Limitations include retrospective cohorts, center and acquisition shift, dependence
on public reference segmentation, phase heterogeneity, and incomplete cross-cohort
liver-function harmonization. The maximum extent is not a RECIST/mRECIST diameter.
Null external findings, missingness, quality warnings, and proportional-hazards
violations must be reported without post-hoc model modification.

### Formal validation results (2026-09-02)

Nested internal C-indices in WAW were 0.5917 (clinical), 0.6047 (imaging), and
0.6310 (fused). The one-time HCC-TACE-Seg test included 104 subjects and 92 events.
External C-indices were 0.5860 (95% CI 0.5065–0.6579), 0.5467
(0.4746–0.6094), and 0.5819 (0.5131–0.6516), respectively. Fused minus clinical
was -0.0041 (-0.0766–0.0671); fused minus imaging was 0.0352
(0.0029–0.0691). HCC_011 was excluded before unlock for missing/non-uniform CT
slices; no interpolation was performed. Proportional-hazards screening flagged
age, sex, or lesion-count terms in one or more primary models.

These findings document an honest external validation but do not support a claim
that fusion outperforms the clinical core or that the model has clinical utility.
