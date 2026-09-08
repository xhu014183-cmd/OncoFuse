"""Generate a tiny deidentified CT/DICOM SEG fixture for the CPU demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    SegmentationStorage,
    generate_uid,
)


def _dataset(path: Path, storage: str) -> FileDataset:
    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = storage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    result = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    result.SOPClassUID = storage
    result.SOPInstanceUID = sop_uid
    return result


def generate(output: Path) -> tuple[Path, Path]:
    ct_dir = output / "study"
    ct_dir.mkdir(parents=True, exist_ok=True)
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    source_uids: list[str] = []
    for index, z_position in enumerate((0.0, 2.0, 4.0)):
        path = ct_dir / f"slice-{index + 1}.dcm"
        ds = _dataset(path, CTImageStorage)
        source_uids.append(str(ds.SOPInstanceUID))
        ds.Modality = "CT"
        ds.PatientID = "RESEARCH_001"
        ds.StudyDate = "20260720"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.AcquisitionNumber = 1
        ds.InstanceNumber = index + 1
        ds.SeriesDescription = "Portal venous synthetic fixture"
        ds.Rows, ds.Columns = 16, 16
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, z_position]
        ds.PixelSpacing = [1.5, 1.5]
        ds.SliceThickness = 2.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1000
        pixels = np.zeros((16, 16), dtype=np.int16)
        pixels[5:10, 5:10] = 100 + index * 10
        ds.PixelData = pixels.tobytes()
        pydicom.dcmwrite(path, ds, enforce_file_format=True)

    seg_path = output / "seg.dcm"
    seg = _dataset(seg_path, SegmentationStorage)
    seg.Modality = "SEG"
    seg.PatientID = "RESEARCH_001"
    seg.StudyInstanceUID = study_uid
    seg.SeriesInstanceUID = generate_uid()
    seg.FrameOfReferenceUID = frame_uid
    seg.Rows, seg.Columns, seg.NumberOfFrames = 16, 16, 3
    seg.SamplesPerPixel = 1
    seg.PhotometricInterpretation = "MONOCHROME2"
    seg.BitsAllocated, seg.BitsStored, seg.HighBit, seg.PixelRepresentation = 8, 8, 7, 0
    seg.SegmentationType = "BINARY"
    segment = Dataset()
    segment.SegmentNumber = 1
    segment.SegmentLabel = "Mass"
    segment.SegmentAlgorithmType = "MANUAL"
    seg.SegmentSequence = [segment]
    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = [orientation]
    measures = Dataset()
    measures.PixelSpacing = [1.5, 1.5]
    measures.SliceThickness = 2.0
    shared.PixelMeasuresSequence = [measures]
    seg.SharedFunctionalGroupsSequence = [shared]
    frames: list[Dataset] = []
    masks: list[np.ndarray] = []
    for source_index in (0, 1, 2):
        frame = Dataset()
        identification = Dataset()
        identification.ReferencedSegmentNumber = 1
        frame.SegmentIdentificationSequence = [identification]
        position = Dataset()
        position.ImagePositionPatient = [0, 0, float(source_index * 2)]
        frame.PlanePositionSequence = [position]
        derivation = Dataset()
        source = Dataset()
        source.ReferencedSOPClassUID = CTImageStorage
        source.ReferencedSOPInstanceUID = source_uids[source_index]
        derivation.SourceImageSequence = [source]
        frame.DerivationImageSequence = [derivation]
        frames.append(frame)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[5:10, 5:10] = 1
        masks.append(mask)
    seg.PerFrameFunctionalGroupsSequence = frames
    seg.PixelData = np.stack(masks).tobytes()
    pydicom.dcmwrite(seg_path, seg, enforce_file_format=True)
    return ct_dir, seg_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic-case")
    args = parser.parse_args()
    study, segmentation = generate(Path(args.output))
    print(f"study: {study}")
    print(f"seg: {segmentation}")
