# Real multicenter cohort validation

This workflow evaluates deterministic research baselines on a local,
deidentified paired cohort. It is not a clinical product or a route for
sending patient data to an external service.

## Physical data boundary

Keep labels outside the feature run directory. A recommended layout is:

```text
study-root/
  protocol.json
  research-manifest.json
  cohort-data/
    PSEUDONYM_001/
      baseline/ct/*.dcm
      baseline/seg.dcm
      followup-001/ct/*.dcm
      followup-001/seg.dcm
      labs.json
  blinded-labels/
    development.json
    external-test.json
  locked-features/
  development-evaluation/
  external-evaluation/
```

`build-research-cohort` has no labels parameter. It processes both splits
using only the protocol, manifest, DICOM/SEG, and laboratory files. The run
lock records protocol, manifest, rule, source-code, Git, and external-center
identifiers before any endpoint label is loaded.

## Fixed endpoint

- Baseline: portal-venous CT in the 42 days before index treatment, with an
  expert SEG.
- Follow-up: reliable portal-venous CT and expert SEG after treatment.
- Laboratory alignment: nearest qualified AFP/DCP within 14 days of each
  image; same day first, then the earlier sample on an equal-distance tie.
- Positive: first blinded confirmed progression at or before day 180.
- Negative: a no-progression CT from day 150 through day 210, with every prior
  review non-progressive.
- Indeterminate: first progression only after day 180, insufficient
  observation, or an image the reviewers cannot judge.

Two blinded reviewers independently label each eligible follow-up without AFP,
DCP, or system output. Disagreement requires a distinct third reviewer. The
contract rejects missing adjudication, duplicate review records, inconsistent
final labels, and labels from the wrong split.

## Data checks

Every manifest patient receives exactly one case disposition: `included`,
`excluded`, `failed`, or `indeterminate`. Validation checks protocol timing and
phase, registration status, file containment, DICOM modality and UIDs, expert
SEG presence, pseudonymous PatientID, laboratory identity and units, and PHI
header fields. PHI findings are reported by tag and filename only; values are
never copied into an audit artifact.

The external center is named in the protocol before label access. Development
adjudications cannot contain external-center patients. External evaluation
requires `--unlock-external`, writes `external_test_unlock.json`, and refuses a
second unlock in the same output directory.

## Deterministic scores

- Image-only: progression signal `1`; complete stable/response evidence `0`;
  failed QC or complex matching `null`.
- Lab-only: count of rising AFP/DCP markers above their reference upper limit,
  divided by two; incomplete two-marker evidence `null`.
- Rule-fusion: mean of complete image and lab scores; otherwise `null`.

The fixed positive threshold is `0.5`. No test-set tuning, learned fusion,
encoder fine-tuning, LLM call, diagnosis, staging, or treatment recommendation
is part of this workflow.

## Outputs

`evaluate-research-cohort` writes:

- `evaluation.json`: AUPRC with 2000-sample patient bootstrap intervals,
  AUROC, sensitivity, specificity, PPV, NPV, Brier score, 10-bin ECE, paired
  baseline differences, strata, and indeterminate sensitivity analyses.
- `research_report.md`: deterministic summary for research and clinical teams.
- `case_dispositions.json`: every endpoint disposition and structured reason.
- `missingness.json` and `missing_data.csv`: aggregate and patient-level
  missing data.
- `external_test_unlock.json`: external-only one-time label access audit.

Fewer than 20 positive or 20 negative external endpoints produces metrics and
wide intervals but forces `exploratory_only=true`; no clinical performance
claim is permitted.
