# Schema 1.0 migration

Version 0.2.0 intentionally replaces the public 0.1 JSON and CLI contract.
There is no automatic migration because the old artifacts discarded source
semantics that cannot be reconstructed safely.

## Breaking changes

- All top-level artifacts now include `schema_version=1.0.0`,
  `pipeline_version`, `generated_at`, `sources`, and structured quality.
- Quality values are `pass`, `warning`, `fail`, or `unavailable`.
  `pass_with_warnings` is now `warning`.
- Imaging evidence adds explicit RAS geometry, affine, shape, orientation,
  phase, Frame of Reference, series/segment identifiers, and QC checks.
- Lesions add voxel and physical bounding boxes. Longitudinal matches are typed
  records rather than arbitrary dictionaries.
- Laboratory observations retain comparator, normalized/original units,
  reference low/high, parse status, source text, and nullable value.
- Rejected laboratory rows are retained under `rejected_observations`; they no
  longer silently disappear or contribute fabricated numbers.
- Fusion outputs add rule traces, threshold version, and quality status.
  Absence of a progression signal now yields `insufficient_evidence`, not
  default moderate concordance.
- LLM output adds uncertainty, data quality, and an exact research disclaimer.
  Locked-field repair was removed: a mismatch blocks output.
- `analyze` accepts explicit phase, registration status, and treatment events.
- New stage commands are `convert-dicom-seg`, `evaluate-cohort`,
  `export-schemas`, and `validate-json`.

## Required migration action

Re-run source NIfTI/DICOM and laboratory inputs through the 0.2 pipeline. Do
not transform old numeric lab JSON into the new observation contract without
the original result text and unit. Do not populate geometry UID fields from
filenames or assumptions.
