"""The Hounsfield-unit guard, and the padding that used to defeat it.

`resample_to_spacing` fills anything outside the source extent with a -1024.0 air
constant, so a rounded-up output grid gains a thin pad border. Validating AFTER that
step inspects a volume the pipeline has already widened: a normalized [0, 1] input
reads as range 1025 and passes a min/max test. These tests pin both halves of the fix,
the percentile-based spread and the ordering. Pure logic; numpy only.
"""

from __future__ import annotations

import numpy as np
import pytest

from oncoct.io.dicom_nifti import assert_hounsfield_units

# The constant resample_to_spacing writes outside the source extent.
RESAMPLE_PAD_HU = -1024.0


def ct_like(shape=(16, 32, 32)) -> np.ndarray:
    """A volume with a real CT's span: air in the lungs, soft tissue, bone."""
    rng = np.random.default_rng(0)
    vol = rng.uniform(-1000, -700, size=shape)  # aerated lung
    vol[:, :16, :] = rng.uniform(-100, 100, size=vol[:, :16, :].shape)  # soft tissue
    vol[:, :2, :] = rng.uniform(300, 900, size=vol[:, :2, :].shape)  # bone
    return vol


def normalized(shape=(16, 32, 32)) -> np.ndarray:
    """The same scan after somebody min-max scaled it to [0, 1]."""
    return np.random.default_rng(0).uniform(0.0, 1.0, size=shape)


def test_real_ct_passes():
    assert_hounsfield_units(ct_like())


def test_normalized_volume_is_rejected():
    with pytest.raises(ValueError, match="Hounsfield"):
        assert_hounsfield_units(normalized())


def test_normalized_volume_is_still_rejected_after_a_resample_pad_border():
    """The regression. A single padded slice gives min/max a 1025 spread; the volume is
    still normalized garbage and must still be refused."""
    vol = normalized()
    vol[0, :, :] = RESAMPLE_PAD_HU  # the border resampling adds
    assert vol.max() - vol.min() > 1000  # min/max is satisfied...
    with pytest.raises(ValueError, match="Hounsfield"):  # ...the guard is not
        assert_hounsfield_units(vol)


def test_a_single_padded_voxel_does_not_rescue_a_normalized_volume():
    vol = normalized()
    vol[0, 0, 0] = RESAMPLE_PAD_HU
    with pytest.raises(ValueError, match="Hounsfield"):
        assert_hounsfield_units(vol)


def test_real_ct_still_passes_with_a_pad_border():
    """The guard must not become so strict that legitimately resampled CT trips it."""
    vol = ct_like()
    vol[0, :, :] = RESAMPLE_PAD_HU
    assert_hounsfield_units(vol)


def test_empty_volume_is_rejected_rather_than_silently_accepted():
    with pytest.raises(ValueError, match="Empty volume"):
        assert_hounsfield_units(np.zeros((0,)))


def test_constant_volume_is_rejected():
    assert_hounsfield_units(ct_like())  # sanity: the fixture is valid
    with pytest.raises(ValueError, match="Hounsfield"):
        assert_hounsfield_units(np.zeros((8, 8, 8)))


@pytest.mark.parametrize("rel", ["src/oncoct/agent/tools.py", "scripts/run_pipeline.py"])
def test_ingest_validates_before_resampling(rel):
    """Both drivers must validate the volume they READ, not the one they resampled.

    Asserted at source level on purpose. Reaching the real ingest path needs SimpleITK,
    and this suite deliberately runs with no imaging stack, no GPU and no API key, so a
    functional test here would cost that property. The substantive half of the fix, the
    guard surviving pad contamination, is tested for real above.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")
    validate_at = source.index("assert_hounsfield_units(sitk.GetArrayFromImage(")
    resample_at = source.index("= resample_to_spacing(")
    assert validate_at < resample_at, (
        f"{rel}: assert_hounsfield_units must run on the raw volume, before "
        "resample_to_spacing pads it with -1024.0 and widens its range"
    )
