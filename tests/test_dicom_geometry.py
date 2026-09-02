from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    SegmentationStorage,
    generate_uid,
)

from hcc_multimodal.tcia import convert_ct_and_mass_seg


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    return dataset


def _write_phantom(root: Path, *, positions=(0.0, 2.0, 4.0)) -> tuple[Path, Path]:
    ct_dir = root / "ct"
    ct_dir.mkdir(parents=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    source_uids: list[str] = []
    for index, z_position in enumerate(positions):
        sop_uid = generate_uid()
        source_uids.append(sop_uid)
        path = ct_dir / f"slice-{len(positions) - index}.dcm"
        dataset = _file_dataset(path, CTImageStorage, sop_uid)
        dataset.Modality = "CT"
        dataset.PatientID = "PHANTOM"
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.AcquisitionNumber = 1
        dataset.InstanceNumber = index + 1
        dataset.Rows = 4
        dataset.Columns = 5
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [10, 20, z_position]
        dataset.PixelSpacing = [2.0, 3.0]
        dataset.SliceThickness = 2.0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope = 1
        dataset.RescaleIntercept = -1000
        pixels = np.full((4, 5), index * 100, dtype=np.int16)
        dataset.PixelData = pixels.tobytes()
        pydicom.dcmwrite(path, dataset, enforce_file_format=True)

    seg_path = root / "mass-seg.dcm"
    seg = _file_dataset(seg_path, SegmentationStorage, generate_uid())
    seg.Modality = "SEG"
    seg.PatientID = "PHANTOM"
    seg.StudyInstanceUID = study_uid
    seg.SeriesInstanceUID = generate_uid()
    seg.FrameOfReferenceUID = frame_uid
    seg.Rows = 4
    seg.Columns = 5
    seg.NumberOfFrames = 2
    seg.SamplesPerPixel = 1
    seg.PhotometricInterpretation = "MONOCHROME2"
    seg.BitsAllocated = 8
    seg.BitsStored = 8
    seg.HighBit = 7
    seg.PixelRepresentation = 0
    seg.SegmentationType = "BINARY"

    segment = Dataset()
    segment.SegmentNumber = 1
    segment.SegmentLabel = "Mass"
    segment.SegmentAlgorithmType = "MANUAL"
    seg.SegmentSequence = [segment]

    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
    shared.PlaneOrientationSequence = [orientation]
    measures = Dataset()
    measures.PixelSpacing = [2.0, 3.0]
    measures.SliceThickness = 2.0
    shared.PixelMeasuresSequence = [measures]
    seg.SharedFunctionalGroupsSequence = [shared]

    frames = []
    frame_pixels = []
    for source_index in (0, 2):
        frame = Dataset()
        identification = Dataset()
        identification.ReferencedSegmentNumber = 1
        frame.SegmentIdentificationSequence = [identification]
        position = Dataset()
        position.ImagePositionPatient = [22, 26, positions[source_index]]
        frame.PlanePositionSequence = [position]
        derivation = Dataset()
        source = Dataset()
        source.ReferencedSOPClassUID = CTImageStorage
        source.ReferencedSOPInstanceUID = source_uids[source_index]
        derivation.SourceImageSequence = [source]
        frame.DerivationImageSequence = [derivation]
        frames.append(frame)
        pixels = np.zeros((4, 5), dtype=np.uint8)
        pixels[2, 3] = 1
        frame_pixels.append(pixels)
    seg.PerFrameFunctionalGroupsSequence = frames
    seg.PixelData = np.stack(frame_pixels).tobytes()
    pydicom.dcmwrite(seg_path, seg, enforce_file_format=True)
    return ct_dir, seg_path


def test_complete_ct_grid_and_flipped_seg_are_mapped_in_patient_space(tmp_path: Path):
    ct_dir, seg_path = _write_phantom(tmp_path / "input")
    image_path, mask_path, attribution_path = convert_ct_and_mass_seg(
        ct_dir,
        seg_path,
        tmp_path / "converted",
    )
    image = nib.load(image_path)
    mask = np.asarray(nib.load(mask_path).dataobj)
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))

    assert image.shape == mask.shape == (4, 5, 3)
    assert mask[1, 1, 0] == 1
    assert mask[1, 1, 2] == 1
    assert mask[:, :, 1].sum() == 0
    assert np.allclose(nib.affines.voxel_sizes(image.affine), [2.0, 3.0, 2.0])
    assert np.allclose(nib.affines.apply_affine(image.affine, [1, 1, 0]), [-13, -22, 0])
    assert attribution["geometry_qc"]["ct_instances_in_target"] == 3
    assert attribution["geometry_qc"]["seg_frames_mapped"] == 2
    assert attribution["geometry_qc"]["max_seg_landmark_error_mm"] == 0
    assert attribution["source_patient_id"] == "PHANTOM"
    assert image_path.name == "phantom_ct.nii.gz"
    assert mask_path.name == "phantom_tumor_mask.nii.gz"


def test_requested_patient_id_must_match_dicom(tmp_path: Path):
    ct_dir, seg_path = _write_phantom(tmp_path / "input")
    with pytest.raises(ValueError, match="patient_id does not match"):
        convert_ct_and_mass_seg(
            ct_dir,
            seg_path,
            tmp_path / "converted",
            patient_id="OTHER_PATIENT",
        )


def test_nonuniform_or_missing_ct_slice_is_rejected(tmp_path: Path):
    ct_dir, seg_path = _write_phantom(tmp_path / "input", positions=(0.0, 2.0, 5.0))
    with pytest.raises(ValueError, match="missing slices or non-uniform spacing"):
        convert_ct_and_mass_seg(ct_dir, seg_path, tmp_path / "converted")


def test_empty_seg_frame_outside_ct_grid_is_ignored(tmp_path: Path):
    ct_dir, seg_path = _write_phantom(tmp_path / "input")
    seg = pydicom.dcmread(seg_path)
    original_pixels = np.asarray(seg.pixel_array)
    empty_frame = deepcopy(seg.PerFrameFunctionalGroupsSequence[-1])
    empty_frame.PlanePositionSequence[0].ImagePositionPatient = [22, 26, 6.0]
    del empty_frame.DerivationImageSequence
    seg.PerFrameFunctionalGroupsSequence.append(empty_frame)
    seg.NumberOfFrames = 3
    seg.PixelData = np.concatenate(
        [original_pixels, np.zeros((1, 4, 5), dtype=np.uint8)], axis=0
    ).tobytes()
    pydicom.dcmwrite(seg_path, seg, enforce_file_format=True)

    _, mask_path, attribution_path = convert_ct_and_mass_seg(
        ct_dir,
        seg_path,
        tmp_path / "converted",
    )

    mask = np.asarray(nib.load(mask_path).dataobj)
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    assert mask.sum() == 2
    assert attribution["geometry_qc"]["seg_frames_mapped"] == 2
    assert any(
        "Ignored 1 empty Mass SEG frames" in warning
        for warning in attribution["geometry_qc"]["warnings"]
    )


def test_nonzero_seg_frames_can_use_verified_positional_fallback(tmp_path: Path):
    ct_dir, seg_path = _write_phantom(tmp_path / "input")
    seg = pydicom.dcmread(seg_path)
    for frame in seg.PerFrameFunctionalGroupsSequence:
        del frame.DerivationImageSequence
    pydicom.dcmwrite(seg_path, seg, enforce_file_format=True)

    _, mask_path, attribution_path = convert_ct_and_mass_seg(
        ct_dir,
        seg_path,
        tmp_path / "converted",
    )

    mask = np.asarray(nib.load(mask_path).dataobj)
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    assert mask.sum() == 2
    assert any(
        "selected by verified patient-space alignment" in warning
        for warning in attribution["geometry_qc"]["warnings"]
    )
