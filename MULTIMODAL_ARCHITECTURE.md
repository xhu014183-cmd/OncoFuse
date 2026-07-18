# Multimodal architecture contract

## Current architecture

```text
NIfTI + aligned mask
    +-> deterministic measurement -> ImagingEvidence
    `-> optional 3D provider ------> image_embedding.npy + manifest

AFP/DCP observations
    +-> trend evidence ------------> LabEvidence
    `-> fixed named features ------> LabFeatureVector + availability mask

ImagingEvidence + LabEvidence
    -> transparent rule fusion
    -> ClinicalVerdict
    -> controlled LLM renderer

Evidence metadata + optional representation manifests
    -> MultimodalCaseEvidence (audit/training contract, never an LLM prompt)
```

The image representation is a parallel branch. It does not replace mask-based
measurement and does not affect `ClinicalVerdict` in the current version.

## MultimodalCaseEvidence

The case envelope records information that was previously only present in
free text:

- one `index_time` for the decision;
- each modality's `observed_at` and `evidence_age_days`;
- `available` and `quality_status`;
- `data_origin`: `real_clinical`, `real_public`, `synthetic`,
  `user_supplied`, or `unknown`;
- `pairing_status`: `same_subject`, `unpaired_poc_composite`, or
  `user_supplied_unverified`;
- the human-readable `data_relationship`;
- a masked, fixed-order AFP/DCP feature vector;
- an optional image-embedding manifest, never the high-dimensional vector.

Evidence dated after `index_time` is rejected or filtered before feature
construction. A same-day timestamp never upgrades synthetic values into real
patient-level pairing.

## Laboratory feature contract

For AFP and DCP, features appear in a stable order:

```text
latest_uln_ratio
change_uln_ratio
slope_uln_per_30d
observation_count
latest_age_days
trend_rising
missing
```

Unavailable numeric features use a zero placeholder with
`availability_mask=false`. A separate `missing` feature distinguishes an
unmeasured marker from a genuinely low or normal result. ULN means the supplied
upper limit of normal; no population reference range is invented by the demo.

These values are an interoperability contract, not a trained embedding.
Cohort-fitted scaling and model calibration belong in the future training
pipeline.

## LLM boundary

The LLM receives compact `ImagingEvidence`, compact `LabEvidence`, the
deterministic `ClinicalVerdict`, and the declared data relationship. It does
not receive:

- image embedding vectors;
- raw NIfTI voxels;
- artifact paths or hashes;
- lesion world coordinates or voxel counts;
- model tensors;
- unrestricted multimodal case envelopes.

Unsupported imaging payload shapes are rejected instead of being passed
through to the model unfiltered.

## Future learned fusion

A learned fusion layer becomes meaningful only after a real paired cohort and
a specific endpoint exist. The intended progression is:

1. Define one endpoint and index-time policy.
2. Train and evaluate image-only and lab-only baselines.
3. Add a small MLP or gated adapter over image and lab representations.
4. Compare with a missing-modality-aware cross-attention model only if the
   cohort size supports it.
5. Report results separately for image-only, labs-only, paired, and each
   missing-modality pattern.
6. Use patient-level splits, external-center validation, confidence intervals,
   and calibration.

The real public CT plus synthetic AFP/DCP scenarios can validate software
plumbing and counterfactual behavior. They cannot train or validate a learned
cross-modal association.

## Ownership boundary

```text
United Imaging side:
  DICOM/NIfTI conversion, phase identity, registration, segmentation,
  image encoder, image QC, and image-model provenance

Roche side:
  assay identity, units, supplied reference range, AFP/DCP observations,
  analytical QC, trend features, and missingness

Joint responsibility:
  patient identity, index time, endpoint definition, pairing status,
  fusion training, external validation, calibration, and output governance
```
