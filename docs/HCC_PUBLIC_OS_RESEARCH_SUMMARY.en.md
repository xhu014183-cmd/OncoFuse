# Auditable Multimodal HCC Survival Risk Stratification after TACE

## Motivation

This project extends an auditable LiON-inspired liver evidence pipeline from a
single-case demonstration to a reproducible public-cohort prognosis study. The aim is
not to reproduce LiON or build a clinical device. It asks a narrower, falsifiable
question: do baseline CT segmentation-derived tumor burden and basic laboratory
evidence provide complementary overall-survival risk information after TACE?

## Design

WAW-TACE (233 patients) is used exclusively for development. HCC-TACE-Seg (105
patients) is reserved as an independent external test. Three prespecified penalized
Cox models compare clinical, imaging, and fused evidence. A richer WAW-only laboratory
model is labelled exploratory because the external cohort lacks the corresponding raw
liver-function panel.

The imaging model uses a harmonized, low-dimensional feature set computed from public
reference segmentations. It avoids dataset-specific high-dimensional radiomics and
does not register masks across contrast phases without validation. GLM is an optional
image-only explanation layer, while DeepSeek is limited to language editing over a
locked deterministic report. Neither contributes predictors.

## Reproducibility and safety

Every dataset file is versioned and checksummed. Feature and outcome tables are
physically separated. Cross-validation folds, transformations, coefficients, baseline
hazard, model hashes, and external-unlock audit are serialized as readable JSON. The
external cohort cannot be used for tuning. Null or adverse results are reported rather
than hidden.

## Cohort and endpoint handling

The study population is restricted to patients with HCC treated with first TACE. The
index time is the first TACE date, and the endpoint is time to death or last follow-up.
WAW survival time is retained in days. HCC-TACE-Seg survival time is supplied in weeks
and converted deterministically to days. Outcome records are stored in files separate
from baseline model features. Post-treatment response, progression, number of TACE
sessions, follow-up imaging, and follow-up laboratory measurements are explicitly
forbidden predictors.

The cross-cohort clinical model uses only age, sex, and `log1p(AFP ng/mL)`, because
these variables are available in both datasets. The HCC-TACE-Seg v2 clinical table
contains paired AFP for all 105 rows, but does not provide DCP, AFP-L3, or the complete
raw liver-function panel needed for the richer WAW analysis. Albumin, bilirubin, INR,
ALT, and creatinine therefore enter a clearly labelled WAW-only exploratory model.
WAW albumin values are preserved in their source representation and accompanied by a
version-scoped unit override audit because the dictionary label and observed range are
inconsistent. A prespecified no-albumin sensitivity model tests dependence on this
correction.

## Imaging evidence

All primary imaging features are recomputed from public reference segmentations with
one implementation: 26-connected lesion count, total tumor volume, maximum lesion 3D
bounding-box extent, and largest-lesion sphericity. These deliberately simple features
reduce dependence on cohort-specific radiomics software and reconstruction settings.
Masks on a common grid are merged before relabelling. A small number of WAW patients
store separate lesion masks on different native grids; each lesion is measured in its
own validated physical geometry and aggregated without unverified interpolation. This
condition remains visible as an audit warning rather than causing silent exclusion.

WAW segmentations were drawn on different contrast phases. The primary burden and
shape model therefore retains all eligible patients without forcing unvalidated
registration. Seventy-four WAW patients have masks that directly correspond to the
portal-venous phase and form a prespecified exploratory intensity subset. Portal CT
mean, standard deviation, tenth percentile, and ninetieth percentile are calculated
only when CT and SEG geometry match. HCC external intensity analysis similarly
requires a SEG that directly references the portal-venous CT acquisition.

## Statistical analysis

Three primary L2-penalized Cox models compare clinical, imaging, and fused inputs. A
nested five-fold event-stratified procedure estimates internal performance and chooses
the penalty from `0.001`, `0.01`, `0.1`, and `1.0`. Every transformation and scaling
parameter is fitted within the corresponding WAW training fold. After internal
validation, the chosen specification is refitted on all eligible WAW patients and
serialized as a human-readable model bundle containing ordered features, parameters,
baseline cumulative hazard, development risk distribution, data version, and hash.

Harrell's C-index is the primary discrimination measure. Patient-level bootstrap with
1,000 resamples provides 95% intervals, while paired bootstrap estimates the fused
model's C-index difference from each component model. Calibration at 12 and 24 months
uses the frozen WAW baseline hazard. The WAW median risk index defines a research-only
threshold that is applied unchanged to the external cohort. Schoenfeld-residual
screening identifies possible proportional-hazards violations; flags are reported as
limitations and do not trigger outcome-driven model revision.

## Controlled multimodal reporting

The patient-level demonstration remains separate from model development. CT plus SEG
produce LiON-inspired quantitative evidence, while a portal-venous montage and lesion
crops can be sent to GLM for image-conditioned description. GLM never receives AFP,
outcomes, model scores, or treatment-response information and is not a survival
predictor. The frozen Cox model produces structured prognostic evidence consisting of
a relative risk index, a WAW reference percentile, and a development-median research
group. It does not estimate an individual's remaining months.

A deterministic report locks the case ID, model version and hash, quantitative values,
applicability, missing evidence, and limitations. DeepSeek may edit only the language
around this locked content. It receives neither raw images nor outcome data and cannot
change any number. GLM or DeepSeek failure leaves a complete deterministic report with
an explicit degraded audit state.

## Deliverables and interpretation

The software produces dataset and license manifests, cohort QC, exclusions, separate
feature and endpoint tables, reproducible fold assignments, JSON model bundles,
internal and external validation reports, paired model comparisons, calibration
plots, feature distributions, a cohort flow diagram, an external-validation forest
plot, data and model cards, and an append-only experiment log. The external evaluator
requires an explicit one-time unlock and rejects mixed development/external artifacts.

The scientific success criterion is completion of an honest external validation, not
a prespecified minimum C-index. A null result would still answer the research question
and would be retained with the same audit trail. The project demonstrates evidence
engineering, multimodal orchestration, leakage control, reproducible survival
analysis, external validation, and appropriately limited scientific communication.

## Formal results

All 233 WAW subjects entered development. Of 105 HCC-TACE-Seg subjects, 104 passed
the prespecified baseline CT/SEG QC; one was excluded for missing/non-uniform CT
slices without interpolation. Nested WAW C-indices were 0.5917 for clinical core,
0.6047 for imaging core, and 0.6310 for fused core. In the one-time external test
(104 subjects, 92 events), the corresponding C-indices were 0.5860 (95% CI
0.5065–0.6579), 0.5467 (0.4746–0.6094), and 0.5819 (0.5131–0.6516).
The fused-minus-clinical difference was -0.0041 (-0.0766–0.0671), whereas
fused-minus-imaging was 0.0352 (0.0029–0.0691). Thus, the low-dimensional imaging
features did not add external discrimination beyond age, sex, and AFP. This null
incremental result was retained without post-hoc tuning. Proportional-hazards
screening warnings are also retained as limitations.

The resulting claim is deliberately limited: an auditable research prototype for
externally validated multimodal HCC survival risk stratification after TACE. It is not
a diagnostic system, automatic segmentation model, individual life-expectancy tool,
or prospectively validated clinical product.
