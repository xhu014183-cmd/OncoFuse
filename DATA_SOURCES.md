# Public data sources

## HCC-TACE-Seg

The optional public-image demo downloads one CT series and its DICOM SEG from
the NCI Imaging Data Commons copy of the HCC-TACE-Seg collection.

- Collection: HCC-TACE-Seg
- Source subject identifier: HCC_003
- Collection DOI: https://doi.org/10.7937/TCIA.5FNA-0924
- License: Creative Commons Attribution 4.0 International
- License URL: https://creativecommons.org/licenses/by/4.0/
- Study Instance UID:
  1.3.6.1.4.1.14519.5.2.1.1706.8374.128750515241701445125322964595
- CT Series Instance UID:
  1.3.6.1.4.1.14519.5.2.1.1706.8374.281650679207816520863173918688
- SEG Series Instance UID:
  1.3.6.1.4.1.14519.5.2.1.1706.8374.106355502486885782622426045632
- Selected DICOM segment: Mass

The downloader does not commit the source DICOM or generated NIfTI files to
Git. It writes an ATTRIBUTION.json beside every converted case. Generated
files remain subject to the collection license and attribution requirements.

## Conversion

The conversion code:

1. Downloads the fixed CT and SEG series through idc-index.
2. Finds the DICOM segment whose SegmentLabel is Mass.
3. Groups duplicated source instances by acquisition and selects one coherent
   acquisition deterministically.
4. Builds the target from every CT slice in that acquisition, including slices
   without a nonzero Mass frame.
5. Sorts and validates slice positions in patient coordinates.
6. Maps every SEG frame from its own plane orientation/origin onto the CT
   row/column grid, then checks plane landmark error.
7. Applies CT rescale slope/intercept and converts DICOM LPS to NIfTI RAS.

The public Series Instance UID contains two 95-slice acquisitions at the same
locations. The SEG references both acquisitions and reverses both in-plane
directions relative to CT. The converter selects `AcquisitionNumber:1`, maps
the reversed SEG pixel plane geometrically, and records the ambiguity in QC.
For this case, SEG and CT Frame of Reference UIDs differ; direct SOP references
and per-frame landmarks reconcile the mapping with 0 mm plane error, while the
UID mismatch remains a warning.

The resulting mask is an expert-provided collection annotation, not a
prediction made by this repository.

## Composite-case warning

HCC-TACE-Seg does not provide AFP or DCP values for this subject. The public
demo therefore combines the real public imaging with explicitly synthetic
marker values under patient ID COMPOSITE_PUBLIC_HCC_003.

Those marker values are not linked to, measured from, or representative of the
TCIA subject. The result demonstrates a software contract only; it is not
patient-level multimodal evidence.
