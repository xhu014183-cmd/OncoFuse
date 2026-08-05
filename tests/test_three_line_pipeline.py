from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from hcc_multimodal.case_llm import render_with_optional_llm
from hcc_multimodal.case_models import ImageToolEvidence
from hcc_multimodal.case_summary import render_case_markdown, summarize_case
from hcc_multimodal.clinical_labs import parse_laboratory_report
from hcc_multimodal.hpi import parse_hpi_timeline
from hcc_multimodal.imaging_adapter import parse_imaging_study


def _write_ct(root: Path, *, patient_id: str = "P1", patient_name: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for index in range(2):
        path = root / f"slice-{index}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.Modality = "CT"
        ds.PatientID = patient_id
        if patient_name is not None:
            ds.PatientName = patient_name
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate = "20260720"
        ds.Rows, ds.Columns = 4, 5
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, index * 2.0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.PixelData = np.zeros((4, 5), dtype=np.int16).tobytes()
        pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return root


def test_laboratory_text_supports_panel_units_and_rejects_unknown(tmp_path: Path):
    path = tmp_path / "labs.txt"
    path.write_text(
        "2026-01-01 AFP 5.2E+01 ug/L 0-7\n"
        "2026-07-01 AFP 96 ng/mL 0-7\n"
        "2026-07-01 DCP >40 AU/L 0-40\n"
        "2026-07-01 AFP-L3% 12 % 0-10\n"
        "2026-07-01 Total Bilirubin 20 umol/L 0-20\n"
        "2026-07-01 Albumin 38 g/L 35-55\n"
        "2026-07-01 HBsAg positive qualitative\n"
        "2026-07-01 DCP 2 ng/mL 0-40\n",
        encoding="utf-8",
    )
    result = parse_laboratory_report(path, patient_id="P1")
    assert result.analytes["AFP"].trajectory_state == "persistent_rising"
    assert result.analytes["DCP"].observations[0].parse_status == "censored"
    assert result.analytes["AFP-L3%"].latest_above_reference is True
    assert result.liver_reserve_evidence.albi_grade in {1, 2, 3}
    assert result.analytes["HBsAg"].latest_value == 1
    assert any(item.parse_status == "unsupported_unit" for item in result.rejected_observations)
    assert all(item.source_span is not None for item in result.analytes["AFP"].observations)


def test_laboratory_json_and_csv_inputs(tmp_path: Path):
    json_path = tmp_path / "labs.json"
    json_path.write_text(json.dumps({"observations": [{"date": "2026-01-01", "marker": "AFP", "value": "<15", "unit": "ng/mL"}]}), encoding="utf-8")
    assert parse_laboratory_report(json_path, patient_id="P1").source_format == "json"
    csv_path = tmp_path / "labs.csv"
    csv_path.write_text("date,marker,value,unit\n2026-01-01,DCP,20,mAU/mL\n", encoding="utf-8")
    assert parse_laboratory_report(csv_path, patient_id="P1").source_format == "csv"


def test_hpi_relative_date_and_marker_timeline(tmp_path: Path):
    labs_path = tmp_path / "labs.txt"
    labs_path.write_text("2026-01-01 AFP 100 ng/mL 0-7\n2026-07-01 AFP 200 ng/mL 0-7\n", encoding="utf-8")
    labs = parse_laboratory_report(labs_path, patient_id="P1")
    hpi_path = tmp_path / "hpi.txt"
    hpi_path.write_text("2026-01-15 TACE treatment\n术后两个月复查 MRI", encoding="utf-8")
    timeline = parse_hpi_timeline(hpi_path, patient_id="P1", labs=labs)
    assert timeline.treatment_dates == ["2026-01-15"]
    assert any(item.date_source == "relative_resolved" and item.event_date == "2026-03-16" for item in timeline.events)
    assert timeline.marker_trajectories["AFP"].state == "persistent_rising"


def test_hpi_excludes_events_after_index_date(tmp_path: Path):
    hpi_path = tmp_path / "hpi.txt"
    hpi_path.write_text("2026-07-20 MRI follow-up\n2026-08-01 AFP 500", encoding="utf-8")
    timeline = parse_hpi_timeline(hpi_path, patient_id="P1", index_date="2026-07-20")
    assert all(item.event_date != "2026-08-01" for item in timeline.events)
    assert any("Excluded 1 event" in item for item in timeline.limitations)


def test_imaging_adapter_checks_geometry_and_tool_consistency(tmp_path: Path):
    dicom_dir = _write_ct(tmp_path / "study")
    tool_path = tmp_path / "image.json"
    tool_path.write_text(json.dumps({
        "patient_id": "P1", "study_date": "2026-07-20", "modality": "CT", "phase": "portal_venous",
        "source_tool": "fixture", "source_version": "1", "findings": [{"finding_id": "F1", "location": "S6", "observation": "stable qualitative observation", "confidence": "moderate", "source_ref": "series/1"}],
    }), encoding="utf-8")
    result = parse_imaging_study(dicom_dir, patient_id="P1", image_evidence_path=tool_path)
    assert result.interpretation_mode == "qualitative_only"
    assert result.geometry.spacing_mm == [1.0, 1.0, 2.0]
    assert result.tool_consistency == "not_comparable"


def test_imaging_adapter_fails_closed_on_phi(tmp_path: Path):
    result = parse_imaging_study(_write_ct(tmp_path / "study", patient_name="Real Name"), patient_id="P1")
    assert result.quality.status == "fail"
    assert any("PHI" in item for item in result.quality.errors)


def test_mr_metadata_does_not_guess_dynamic_phase(tmp_path: Path):
    dicom_dir = _write_ct(tmp_path / "study")
    for path in dicom_dir.glob("*.dcm"):
        dataset = pydicom.dcmread(path)
        dataset.Modality = "MR"
        dataset.SeriesDescription = "Liver dynamic sequence"
        pydicom.dcmwrite(path, dataset, enforce_file_format=True)
    result = parse_imaging_study(dicom_dir, patient_id="P1")
    assert result.modality == "MR"
    assert result.phase == "unknown"
    assert result.interpretation_mode == "unavailable"


def test_summary_is_research_only_and_renders_markdown(tmp_path: Path):
    labs_path = tmp_path / "labs.txt"
    labs_path.write_text("2026-07-20 AFP 96 ng/mL 0-7\n", encoding="utf-8")
    labs = parse_laboratory_report(labs_path, patient_id="P1")
    image_tool = ImageToolEvidence.model_validate({
        "patient_id": "P1", "study_date": "2026-07-20", "modality": "MR", "source_tool": "fixture", "source_version": "1",
        "findings": [{"finding_id": "F1", "location": "S6", "observation": "\u8bca\u65ad\u548c\u6cbb\u7597\u5efa\u8bae\u4e0d\u5f97\u8fdb\u5165\u7ed3\u8bba", "confidence": "high", "source_ref": "x"}],
    })
    image_path = tmp_path / "study"
    _write_ct(image_path, patient_id="P1")
    image_path_json = tmp_path / "image.json"
    image_tool.write_json(image_path_json)
    imaging = parse_imaging_study(image_path, patient_id="P1", image_evidence_path=image_path_json)
    summary = summarize_case(imaging, labs)
    assert summary.research_disclaimer.startswith("Research evidence summary")
    assert all("diagnosis" not in item.lower() for item in summary.key_findings)
    assert "## 当前影像" in render_case_markdown(summary)
    invalid_markdown, audit = render_with_optional_llm(summary, {"unexpected": "diagnosis"})
    assert audit["valid"] is False
    assert invalid_markdown.startswith("# 病例研究证据摘要")


def test_valid_llm_rewrite_preserves_locked_evidence(tmp_path: Path):
    labs_path = tmp_path / "labs.txt"
    labs_path.write_text("2026-07-20 AFP 5 ng/mL 0-7\n", encoding="utf-8")
    labs = parse_laboratory_report(labs_path, patient_id="P1")
    dicom_dir = _write_ct(tmp_path / "study")
    imaging = parse_imaging_study(dicom_dir, patient_id="P1")
    summary = summarize_case(imaging, labs)
    response = {
        "patient_id": "P1",
        "imaging_summary": summary.imaging_summary.findings,
        "laboratory_summary": summary.laboratory_summary.findings,
        "timeline_summary": summary.timeline_summary.findings,
        "evidence_concordance": summary.evidence_concordance,
        "data_gaps": summary.data_gaps,
        "uncertainty": summary.uncertainty,
        "disclaimer": summary.research_disclaimer,
    }
    markdown, audit = render_with_optional_llm(summary, response)
    assert audit["mode"] == "validated_llm_rewrite"
    assert "## Laboratory results" in markdown


def test_cli_parse_labs_smoke(tmp_path: Path):
    source = tmp_path / "labs.txt"
    source.write_text("2026-07-20 AFP 5 ng/mL 0-7\n", encoding="utf-8")
    output = tmp_path / "labs.json"
    completed = subprocess.run([sys.executable, "-m", "hcc_multimodal.cli", "parse-labs", "--input", str(source), "--patient-id", "P1", "--output", str(output)], check=True, capture_output=True, text=True)
    assert output.exists()
    assert "Wrote" in completed.stdout


def test_cli_analyze_case_with_generated_dicom_seg(tmp_path: Path):
    fixture = tmp_path / "fixture"
    subprocess.run([sys.executable, "examples/generate_dicom_seg_fixture.py", "--output", str(fixture)], check=True, capture_output=True, text=True)
    output = tmp_path / "output"
    subprocess.run([
        sys.executable, "-m", "hcc_multimodal.cli", "analyze-case",
        "--dicom-dir", str(fixture / "study"), "--seg", str(fixture / "seg.dcm"),
        "--image-evidence", "examples/image_evidence.synthetic.json", "--labs", "examples/lab_report.synthetic.txt",
        "--hpi", "examples/hpi.synthetic.txt", "--patient-id", "RESEARCH_001", "--output", str(output),
    ], check=True, capture_output=True, text=True)
    assert (output / "case-summary.json").exists()
    assert (output / "case-summary.md").exists()
    assert json.loads((output / "report-audit.json").read_text(encoding="utf-8"))["mode"] == "deterministic"
