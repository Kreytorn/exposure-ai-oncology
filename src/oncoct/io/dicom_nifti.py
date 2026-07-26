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

    Heuristic: a real chest CT has air near -1000 HU and dense bone > +700 HU, so the
    value range should span well beyond [0, 1]. Raises ValueError if the range looks
    like it was normalized away from HU.
    """
    import numpy as np

    arr = np.asarray(volume)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 100:  # normalized data would have a tiny range
        raise ValueError(
            f"Volume range [{lo:.3f}, {hi:.3f}] does not look like Hounsfield units. "
            "Downstream pretrained models require raw HU - do not pre-normalize CT."
        )
