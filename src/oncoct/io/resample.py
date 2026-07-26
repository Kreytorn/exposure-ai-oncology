"""World <-> voxel coordinate conversion and isotropic resampling.

Coordinate conventions (read this before touching anything):
  - WORLD coordinates: physical millimetres, order (x, y, z), as stored in LUNA16
    annotations.csv / candidates_V2.csv and as returned by SimpleITK.
  - VOXEL indices in ITK order: (i, j, k) == (x, y, z).
  - VOXEL indices in NUMPY order: (k, j, i) == (z, y, x) - because a SimpleITK image
    read into numpy via GetArrayFromImage is indexed [z, y, x].

The world<->voxel maths below is deliberately pure (no I/O) so it is exhaustively
unit-tested. `world_to_voxel` / `voxel_to_world` operate in ITK (x, y, z) order;
use `itk_to_numpy_index` / `numpy_to_itk_index` at the numpy boundary. Getting this
wrong is the classic LUNA16 bug - hence the round-trip test.
"""

from __future__ import annotations

import numpy as np

# Value written outside the source extent when resampling an intensity volume: air.
# Shared with io.dicom_nifti, whose HU guard must discount it - a rounded-up output grid
# gains a pad border, and that border alone can make normalized data look like real HU.
AIR_FILL_HU = -1024.0


def world_to_voxel(
    world_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
) -> np.ndarray:
    """Convert world-mm coordinates (x, y, z) to continuous voxel indices (i, j, k).

    voxel = (world - origin) / spacing, componentwise, in ITK (x, y, z) order.

    Args:
        world_xyz: shape (..., 3) world coordinates in mm, (x, y, z).
        origin_xyz: shape (3,) image origin in mm, (x, y, z).
        spacing_xyz: shape (3,) voxel spacing in mm, (x, y, z).

    Returns:
        Continuous voxel indices (i, j, k), same leading shape as ``world_xyz``.

    Note:
        Assumes an axis-aligned (identity direction) image. LUNA16 volumes are
        resampled to identity direction upstream; assert this in the caller.
    """
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    origin_xyz = np.asarray(origin_xyz, dtype=np.float64)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)
    return (world_xyz - origin_xyz) / spacing_xyz


def voxel_to_world(
    voxel_ijk: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`world_to_voxel`: (i, j, k) -> world-mm (x, y, z)."""
    voxel_ijk = np.asarray(voxel_ijk, dtype=np.float64)
    origin_xyz = np.asarray(origin_xyz, dtype=np.float64)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)
    return voxel_ijk * spacing_xyz + origin_xyz


def itk_to_numpy_index(ijk: np.ndarray) -> np.ndarray:
    """Reverse an ITK (i, j, k)==(x, y, z) index to numpy (k, j, i)==(z, y, x)."""
    ijk = np.asarray(ijk)
    return ijk[..., ::-1]


def numpy_to_itk_index(kji: np.ndarray) -> np.ndarray:
    """Reverse a numpy (z, y, x) index back to ITK (x, y, z)."""
    kji = np.asarray(kji)
    return kji[..., ::-1]


def resample_to_spacing(
    image,  # SimpleITK.Image
    target_spacing_xyz: tuple[float, float, float],
    is_label: bool = False,
):
    """Resample a SimpleITK image to the target spacing (x, y, z) in mm.

    Uses nearest-neighbour for label images and B-spline/linear for intensity images.
    HU values are preserved (no intensity normalization here).

    Returns:
        A resampled SimpleITK.Image with identity direction.
    """
    import SimpleITK as sitk

    src_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)  # (x, y, z)
    src_size = np.asarray(image.GetSize(), dtype=np.float64)  # (x, y, z)
    dst_spacing = np.asarray(target_spacing_xyz, dtype=np.float64)
    if np.any(dst_spacing <= 0):
        raise ValueError(f"target_spacing_xyz must be positive, got {target_spacing_xyz}")

    # Physical extent is invariant under resampling:
    # size_new = size_old * spacing_old / spacing_new.
    dst_size = np.maximum(np.round(src_size * src_spacing / dst_spacing), 1).astype(int)

    # Resample into an IDENTITY-direction frame. world_to_voxel/voxel_to_world ignore the
    # direction matrix, so any flipped LUNA16 scan must be straightened HERE - before any
    # world->voxel conversion - or the annotation indices are silently wrong (see brief §7.1).
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    # Keep the same physical corner: the far corner in the source frame becomes the new
    # origin whenever an axis is flipped, so the volume occupies the same world box.
    extent = direction @ ((src_size - 1) * src_spacing)
    dst_origin = np.minimum(origin, origin + extent)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(v) for v in dst_spacing])
    resampler.SetSize([int(v) for v in dst_size])
    resampler.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    resampler.SetOutputOrigin([float(v) for v in dst_origin])
    resampler.SetTransform(sitk.Transform())  # identity: pure grid change
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    # Outside the source extent: air for CT (HU are preserved, never normalized), 0 for labels.
    resampler.SetDefaultPixelValue(0.0 if is_label else AIR_FILL_HU)
    resampler.SetOutputPixelType(image.GetPixelID())
    return resampler.Execute(image)
