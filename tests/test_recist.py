"""RECIST measurement determinism and correctness on known synthetic shapes."""

import numpy as np
import pytest

from oncoct.measure.recist import measure_lesion


def _sphere_mask(radius_vox: int, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape_zyx[0], : shape_zyx[1], : shape_zyx[2]]
    cz, cy, cx = (s // 2 for s in shape_zyx)
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius_vox**2


def test_determinism():
    mask = _sphere_mask(6, (32, 40, 40))
    a = measure_lesion(mask, spacing_xyz=(1.0, 1.0, 1.0))
    b = measure_lesion(mask, spacing_xyz=(1.0, 1.0, 1.0))
    assert a == b


def test_long_axis_of_isotropic_sphere():
    # A radius-6-voxel sphere at 1mm spacing: longest axial diameter ~= 12mm (diameter),
    # within a voxel of tolerance.
    mask = _sphere_mask(6, (32, 40, 40))
    m = measure_lesion(mask, spacing_xyz=(1.0, 1.0, 1.0))
    assert 11.0 <= m.long_axis_mm <= 13.5


def test_spacing_scales_diameter():
    mask = _sphere_mask(6, (32, 40, 40))
    m1 = measure_lesion(mask, spacing_xyz=(1.0, 1.0, 1.0))
    m2 = measure_lesion(mask, spacing_xyz=(2.0, 2.0, 1.0))
    # Doubling in-plane spacing roughly doubles the in-plane long-axis diameter.
    assert m2.long_axis_mm > 1.8 * m1.long_axis_mm


def test_volume_of_cube():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:6, 2:6, 2:6] = True                     # 4x4x4 = 64 voxels
    m = measure_lesion(mask, spacing_xyz=(2.0, 2.0, 2.0))
    assert m.volume_mm3 == pytest.approx(64 * 8, rel=1e-6)   # 64 voxels * 8 mm^3


def test_recist_target_threshold_semantics():
    # long axis >= 10mm is a measurable target lesion (RECIST 1.1) — asserted here so the
    # generator's category rule stays pinned.
    mask = _sphere_mask(8, (40, 48, 48))
    m = measure_lesion(mask, spacing_xyz=(1.0, 1.0, 1.0))
    assert m.long_axis_mm >= 10.0


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        measure_lesion(np.zeros((8, 8, 8), dtype=bool), spacing_xyz=(1.0, 1.0, 1.0))
