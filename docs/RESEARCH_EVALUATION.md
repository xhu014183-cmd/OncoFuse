# Research evaluation checklist

The first endpoint is radiographic progression within a declared follow-up
window. The window and positive-label procedure must be fixed before the test
split is inspected.

## Required cohort fields

- Legal research use and deidentification status
- Patient identifier used only for split isolation
- Center and scanner/acquisition stratum
- Baseline and follow-up dates and phase labels
- Follow-up window in days
- Blinded endpoint label and adjudication procedure
- Treatment events between imaging time points
- Image-only, lab-only, and deterministic rule-fusion scores
- Missing reason for every absent score or modality
- Inclusion and exclusion criteria
- Pre-registration identifier when the cohort is used for a formal study

## Evaluation policy

- Split by patient before tuning thresholds or learned models.
- Keep all records for one patient in exactly one split.
- Do not infer missing scores as normal or negative.
- Report AUROC and AUPRC together with sensitivity/specificity at the declared
  threshold, Brier score, calibration error, and patient bootstrap intervals.
- Report center and scanner strata even when a stratum is too small; use null
  metrics rather than suppressing the stratum.
- Record treatment timing and do not attribute longitudinal change across an
  intervening treatment without a separate analysis plan.
- Keep learned fusion as a separate experiment. It does not replace the three
  deterministic baselines in the first research release.

## Interpretation

Synthetic and public unpaired examples validate interfaces and geometry only.
They do not support clinical performance claims. Small or single-class splits
are explicitly marked as interface checks.
