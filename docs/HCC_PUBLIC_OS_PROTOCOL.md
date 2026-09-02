# HCC Public-Cohort Overall-Survival Protocol

Status: locked before model fitting. Intended use: retrospective research only.

## Question and cohorts

The primary question is whether baseline clinical evidence and segmentation-derived
tumor burden provide complementary overall-survival risk stratification after first
TACE. WAW-TACE v2 is the sole development cohort. HCC-TACE-Seg v2 is a locked,
one-time external test cohort.

The endpoint is time from first TACE to death or last follow-up. WAW supplies days;
HCC-TACE-Seg supplies weeks and is converted by multiplying by seven. Death is the
event. No progression, response, post-treatment imaging, TACE session count, or
follow-up measurement is eligible as a predictor.

## Prespecified predictors

- Clinical core: age, sex, `log1p(AFP ng/mL)`.
- Imaging core: `log1p(26-connected lesion count)`, `log1p(total tumor volume)`,
  `log1p(maximum 3D bounding-box extent)`, and largest-lesion sphericity.
- Fused core: all clinical and imaging core predictors.
- WAW-only exploratory extension: albumin, bilirubin, INR, ALT, and creatinine.

The WAW v2 dictionary labels albumin as g/L, while the observed 2.5–5.2 range is
consistent with g/dL. The adapter applies a version-scoped, audited g/dL override and
also fits a no-albumin sensitivity model. Composite scores (Child-Pugh, BCLC, HAP,
mHAP, ALBI-TAE, six-and-twelve) are excluded.

## Imaging harmonization

Masks on the same spatial grid are merged and relabelled with the same
26-connectivity algorithm. A few WAW patients store separate lesion files on
different native grids; those lesions are measured independently with the same
algorithm and then aggregated without unvalidated resampling. The cohort audit
records `SOURCE_MASK_GRIDS_DIFFER` for these patients.
The prespecified minimum connected-component volume is 0.01 mL in both cohorts;
smaller components are audited and excluded rather than treated as lesions.
Empty masks, invalid spacing, singular geometry, or image-mask mismatch fail closed.
There is no automatic segmentation.

WAW masks were drawn on the phase where each lesion was clearest. The primary shape
model therefore does not force registration to portal venous CT. Only masks already
aligned to portal venous CT enter the exploratory intensity analysis (mean, standard
deviation, tenth and ninetieth HU percentiles). A portal subset below 20 subjects is
reported as feasibility only.

## Training and locked evaluation

Models use L2-penalized Cox partial likelihood. An outer five-fold event-stratified
cross-validation estimates development performance. Each outer training fold uses an
inner five-fold search over penalties `0.001, 0.01, 0.1, 1.0`. Transformations and
standardization are fitted only inside training folds. After internal evaluation, a
final penalty is chosen with five-fold WAW-only cross-validation and the model is fit
to all eligible WAW patients.

The external HCC cohort is opened only with `--unlock-external`. It cannot select
features, penalties, transformations, risk thresholds, or recalibration. Harrell's
C-index is primary. Patient-level bootstrap confidence intervals use 1000 resamples.
Paired bootstrap differences compare fused versus clinical and fused versus imaging.
Twelve- and twenty-four-month calibration uses frozen WAW baseline hazard. The WAW
median risk threshold is applied unchanged.

Proportional-hazards screening uses Schoenfeld residual correlation with log event
time at p<0.01. Violations are reported as limitations and do not trigger post-hoc
model changes.

## Leakage and claim controls

Feature and endpoint CSV files are separate. Training accepts only the WAW
`development-cohort.json`, while the one-time evaluator accepts only the HCC
`external-test-cohort.json`; mixed artifacts are rejected. Outcome keys are rejected from
single-case input. GLM receives only rerendered CT images and lesion identifiers;
DeepSeek receives a number-redacted controlled report. Neither model is a survival
predictor. No minimum performance threshold is declared, and unfavorable or null
external results must remain in the repository artifacts.
