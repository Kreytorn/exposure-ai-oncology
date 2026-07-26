"""DICOM <-> NIfTI conversion and preprocessing validation.

Preserves Hounsfield units. Pretrained downstream models are tightly coupled to HU
windowing and voxel spacing, so this module VALIDATES rather than silently "fixes":
if HU look wrong (e.g. already normalized to [0, 1]) or spacing is missing, it raises.
"""

from __future__ import annotations

from pathlib import Path


def dicom_series_to_nifti(dicom_dir: Path, out_path: Path):
    """Convert a DICOM series folder to a NIfTI volume, preserving HU.

    Uses SimpleITK's series reader (or dcm2niix). Returns the SimpleITK image and
    writes ``out_path``.
    """
    raise NotImplementedError("Wire SimpleITK ImageSeriesReader / dcm2niix.")


def read_mhd(mhd_path: Path):
    """Read a LUNA16 .mhd/.raw MetaImage. Returns a SimpleITK.Image.

    Origin/spacing/direction come from the .mhd header - pass them to the coordinate
    conversions in resample.py. Do not assume 1mm isotropic; LUNA16 spacing varies.
    """
    import SimpleITK as sitk

    mhd_path = Path(mhd_path)
    if not mhd_path.exists():
        raise FileNotFoundError(f"MetaImage header not found: {mhd_path}")
    # The .mhd header names a sibling .raw via ElementDataFile; SimpleITK resolves it
    # relative to the header, so both files must sit in the same directory.
    image = sitk.ReadImage(str(mhd_path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D volume, got {image.GetDimension()}D: {mhd_path}")
    # Cast to float32 so later interpolation isn't clamped to the int16 storage type.
    # HU are preserved exactly (no rescale applied) - assert_hounsfield_units still holds.
    return sitk.Cast(image, sitk.sitkFloat32)


def assert_hounsfield_units(volume, tolerance_frac: float = 0.01) -> None:
    """Sanity-check that a volume is in HU, not normalized.

    A real chest CT has air near -1000 HU and dense tissue well above it, so the value
    range should span far beyond [0, 1]. Raises ValueError if it looks normalized.

    Two things make this robust that a plain min/max test is not.

    First, voxels at or below the resampler's ``AIR_FILL_HU`` constant are discounted.
    `resample_to_spacing` fills anything outside the source extent with that value, so a
    rounded-up output grid gains a pad border. A volume normalized to [0, 1] then reads
    as range 1025 and sails through a min/max test on the padding alone. Real CT is
    unaffected: aerated lung sits near -900 and stays in the measured population.

    Second, the remaining spread is measured on PERCENTILES, so a few stray outliers
    cannot stand in for genuine tissue contrast. ``tolerance_frac`` sets that window.

    Call this on the RAW volume, before resampling. Validating afterwards inspects a
    volume the pipeline has already contaminated with its own padding constant.
    """
    import numpy as np

    from oncoct.io.resample import AIR_FILL_HU

    arr = np.asarray(volume, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty volume: nothing to validate.")

    core = arr[arr > AIR_FILL_HU]
    if core.size == 0:
        raise ValueError(
            f"Volume is entirely at or below {AIR_FILL_HU} HU, so it carries no tissue "
            "contrast. Downstream pretrained models require raw HU."
        )

    pct = max(0.0, min(50.0, tolerance_frac * 100.0))
    lo, hi = (float(v) for v in np.percentile(core, [pct, 100.0 - pct]))
    if hi - lo < 100:  # normalized data would have a tiny spread
        raise ValueError(
            f"Volume spread [{lo:.3f}, {hi:.3f}] between the {pct:g} and {100 - pct:g} "
            f"percentiles of voxels above {AIR_FILL_HU} does not look like Hounsfield "
            "units. Downstream pretrained models require raw HU - do not pre-normalize CT."
        )
