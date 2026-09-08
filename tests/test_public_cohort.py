from pathlib import Path

import pytest

from hcc_multimodal import public_cohort


def test_prepare_with_local_ct_fallback_uses_only_downloaded_ct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient_id = "HCC_089"
    seg_uid = "SEG_UID"
    matching_uid = "1.2.3.4"
    (tmp_path / "raw" / patient_id / f"CT_{matching_uid}").mkdir(parents=True)
    calls: list[str | None] = []
    expected = (
        tmp_path / "converted" / "ct.nii.gz",
        tmp_path / "converted" / "mask.nii.gz",
        tmp_path / "converted" / "ATTRIBUTION.json",
    )

    def fake_prepare(
        output_dir: str | Path,
        *,
        patient_id: str,
        seg_series_uid: str,
        ct_series_uid: str | None = None,
    ) -> tuple[Path, Path, Path]:
        del output_dir, patient_id, seg_series_uid
        calls.append(ct_series_uid)
        if ct_series_uid is None:
            raise ValueError("bad referenced CT")
        return expected

    monkeypatch.setattr(public_cohort, "prepare_public_case", fake_prepare)

    prepared, warnings = public_cohort._prepare_with_local_ct_fallback(
        tmp_path,
        patient_id=patient_id,
        seg_series_uid=seg_uid,
    )

    assert prepared == expected
    assert calls == [None, matching_uid]
    assert warnings == ["referenced CT: bad referenced CT"]


def test_prepare_with_local_ct_fallback_fails_closed_without_local_ct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prepare(*args: object, **kwargs: object) -> tuple[Path, Path, Path]:
        del args, kwargs
        raise ValueError("invalid geometry")

    monkeypatch.setattr(public_cohort, "prepare_public_case", fail_prepare)

    with pytest.raises(RuntimeError, match="referenced CT: invalid geometry"):
        public_cohort._prepare_with_local_ct_fallback(
            tmp_path,
            patient_id="HCC_011",
            seg_series_uid="SEG_UID",
        )
