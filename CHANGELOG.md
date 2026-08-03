# Changelog

## Unreleased

### Added
- Unified case-input JSON support via `--case-input`.
- Added `schemas/case_input.schema.json` for validated multimodal pipeline inputs.
- Added `examples/case_input.example.json` as a runnable case input example.
- Added `docs/CASE_INPUT_JSON_PROPOSAL.md` describing the JSON envelope design and usage.
- Added `src/hcc_multimodal/case_input_loader.py` to load unified case JSON and materialize inline text/files.
- Added `src/hcc_multimodal/registration.py` for volume registration logic.
- Added `tests/test_registration.py` to validate registration behavior.
- Added `docs/VIRTUAL_MDT_DESIGN.md` describing the virtual MDT / multi-disciplinary team design and architecture.

### Changed
- Extended `src/hcc_multimodal/cli.py` to support the new `--case-input` workflow while preserving existing CLI behavior.
- Updated `schemas/longitudinal-imaging-evidence.schema.json` and `src/hcc_multimodal/schemas.py` to support new multimodal data flows.
- Updated `src/hcc_multimodal/imaging.py` to support the revised pipeline and new inputs.
- Updated `pyproject.toml` for package metadata or dependency compatibility related to the new functionality.

### Notes
- The new `--case-input` workflow allows one JSON package to describe imaging, lab, HPI, optional LLM response, and output directory settings.
- Inline laboratory and HPI content is now supported, making API-driven or text-based case inputs easier to use.
- Existing `analyze-case` execution path remains intact; the new loader translates JSON inputs into the same CLI inputs used today.
