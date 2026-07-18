# Controlled report pipeline

## Boundary

The LLM is a renderer, not a decision engine. It receives only:

- compact deidentified imaging measurements and imaging QC;
- parsed AFP/DCP values, comparators, references, trends, and lab QC;
- the deterministic fusion verdict;
- the declared real/synthetic relationship;
- locked fields, required QC items, and a strict JSON Schema.

It does not receive raw DICOM/NIfTI pixels, file paths, embeddings, feature
vectors, artifact hashes, voxel counts, world coordinates, or source result
text. Unsupported evidence shapes are rejected before prompt construction.

Evidence strings are explicitly treated as untrusted data so that source text
cannot act as instructions.

## Locked fields

The following values must match the validated code output exactly:

- `case_id`
- `scenario_id`
- `report_type`
- fusion `state`
- `concordance`
- `review_required`
- `data_quality_status`
- `intended_use`
- research disclaimer

There is no repair step. Any mismatch is `LOCKED_FIELD_MISMATCH` and blocks
the report.

## Blocking validation

Validation runs in this order:

1. Parse exactly one JSON object.
2. Validate the strict Pydantic schema with extra fields forbidden.
3. Compare all locked fields.
4. Require supporting, conflicting, and missing evidence to match the
   deterministic verdict exactly.
5. Require every critical QC item verbatim.
6. Reject numeric values absent from validated evidence.
7. Reject diagnostic assertions, staging/response labels, prognosis or
   treatment advice, and prompt-injection echoes.

On failure, the audit JSON retains raw content and structured error codes. No
formal Markdown is written. On pass, the validated JSON is rendered to
`controlled_report.md`.

## External service fallback

Missing configuration, HTTP failure, or connection failure invokes a
deterministic template that copies the same locked evidence and passes the same
validator. Malformed external JSON is treated as a model failure and does not
fall back silently.

The audit artifact records `renderer=external_llm` or
`renderer=deterministic_template` so availability cannot change provenance.
