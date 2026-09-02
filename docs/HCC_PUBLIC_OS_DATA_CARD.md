# HCC Public OS Data Card

## English

| Cohort | Role | Subjects | Core paired inputs | Outcome | License |
|---|---:|---:|---|---|---|
| WAW-TACE v2 | Development | 233 | CT tumor SEG, age, sex, AFP | OS days, death | CC BY 4.0 |
| HCC-TACE-Seg v2 | External test | 105 available / 104 evaluable | baseline CT/SEG, age, sex, AFP | OS weeks, death | CC BY 4.0 |

Raw images remain outside Git. Dataset manifests record source URL, DOI, version,
checksum, license acknowledgement and acquisition tier. Source identifiers are hashed
before cohort export.

Known limitations include retrospective single-center cohorts, different acquisition
eras and segmentation workflows, phase heterogeneity, absence of DCP/AFP-L3 and raw
liver-function tests in HCC-TACE-Seg, and a WAW albumin-unit metadata inconsistency.
One HCC-TACE-Seg subject (HCC_011) failed prespecified baseline CT geometry QC because
of missing/non-uniform slices and was excluded without interpolation.

Outcome files must never be passed to GLM, DeepSeek, or single-case inference.

## 中文

| 队列 | 角色 | 目标例数 | 可配对核心输入 | 结局 | 许可 |
|---|---:|---:|---|---|---|
| WAW-TACE v2 | 开发与内部交叉验证 | 233 | CT肿瘤SEG、年龄、性别、AFP | OS天数、死亡状态 | CC BY 4.0 |
| HCC-TACE-Seg v2 | 一次性外部验证 | 公开105例 / QC合格104例 | 基线CT/SEG、年龄、性别、AFP | OS周数、死亡状态 | CC BY 4.0 |

原始影像始终存放在Git仓库之外。数据清单记录来源、DOI、版本、校验和、许可
确认和下载层级；导出的患者ID由来源ID哈希生成。

主要限制包括回顾性单中心队列、采集年代和分割流程差异、WAW期相异质、
HCC-TACE-Seg缺少DCP/AFP-L3及完整原始肝功能面板，以及WAW白蛋白数据字典的
单位不一致。HCC_011因基线CT缺层/层距不均未通过预先指定的几何QC，系统未做
插值并将其排除。结局文件禁止发送给GLM、DeepSeek或单病例推理入口。
