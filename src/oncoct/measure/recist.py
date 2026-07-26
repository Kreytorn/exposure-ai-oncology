"""RECIST 1.1 measurements from a 3D lesion mask.

RECIST 1.1 for solid tumours uses the longest in-plane (axial) diameter of the lesion.
We compute it per axial slice and take the maximum across slices, in millimetres, using
the volume's in-plane spacing. Volume is the voxel count times voxel volume. All pure
functions of (mask, spacing) so they are deterministic and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LesionMeasurement:
    """Deterministic measurements for one lesion. All lengths in mm, volume in mm^3."""

    long_axis_mm: float          # RECIST 1.1 longest axial diameter
    short_axis_mm: float         # in-plane perpendicular on the max slice (for nodes)
    volume_mm3: float
    centroid_zyx: tuple[float, float, float]   # numpy (z, y, x) voxel index
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    key_slice_z: int             # axial slice index of the longest diameter


def _max_in_plane_diameter_mm(slice_mask: np.ndarray, spacing_yx: tuple[float, float]) -> float:
    """Longest Euclidean distance between two foreground pixels in one axial slice, mm.

    Uses the convex-hull vertices to keep it O(h log h) rather than O(n^2) over all
    foreground pixels. Returns 0.0 for an empty slice.
    """
    ys, xs = np.nonzero(slice_mask)
    if ys.size == 0:
        return 0.0
    sy, sx = spacing_yx
    pts = np.column_stack([ys * sy, xs * sx]).astype(np.float64)
    if pts.shape[0] == 1:
        return 0.0
    # Convex hull reduces the candidate set for the farthest-pair search.
    try:
        from scipy.spatial import ConvexHull

        hull = pts[ConvexHull(pts).vertices] if pts.shape[0] >= 3 else pts
    except Exception:
        hull = pts
    diffs = hull[:, None, :] - hull[None, :, :]
    dists = np.sqrt((diffs**2).sum(-1))
    return float(dists.max())


def measure_lesion(mask_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> LesionMeasurement:
    """Compute RECIST 1.1 + volume + centroid/bbox for a binary 3D mask.

    Args:
        mask_zyx: binary mask, numpy (z, y, x) order.
        spacing_xyz: voxel spacing in mm, ITK (x, y, z) order.

    Returns:
        A :class:`LesionMeasurement`. Raises ValueError on an empty mask.
    """
    mask = np.asarray(mask_zyx) > 0
    if not mask.any():
        raise ValueError("Empty lesion mask — no voxels to measure.")

    sx, sy, sz = spacing_xyz              # ITK (x, y, z)
    spacing_yx = (sy, sx)                 # in-plane, numpy (y, x) order

    # Longest axial diameter across slices (RECIST 1.1).
    long_axis, key_z = 0.0, 0
    for z in range(mask.shape[0]):
        d = _max_in_plane_diameter_mm(mask[z], spacing_yx)
        if d > long_axis:
            long_axis, key_z = d, z

    # Short axis: perpendicular extent on the key slice, approximated by the smaller
    # in-plane bbox side on that slice (full RECIST short-axis is refined in v2).
    ys, xs = np.nonzero(mask[key_z])
    y_extent = (ys.max() - ys.min() + 1) * sy
    x_extent = (xs.max() - xs.min() + 1) * sx
    short_axis = float(min(y_extent, x_extent))

    volume = float(mask.sum()) * sx * sy * sz

    zz, yy, xx = np.nonzero(mask)
    centroid = (float(zz.mean()), float(yy.mean()), float(xx.mean()))
    bbox = (
        (int(zz.min()), int(zz.max())),
        (int(yy.min()), int(yy.max())),
        (int(xx.min()), int(xx.max())),
    )

    return LesionMeasurement(
        long_axis_mm=round(long_axis, 2),
        short_axis_mm=round(short_axis, 2),
        volume_mm3=round(volume, 2),
        centroid_zyx=centroid,
        bbox_zyx=bbox,
        key_slice_z=key_z,
    )
