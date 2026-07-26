"""Guard against the classic LUNA16 bug: world<->voxel + ITK/numpy axis order.

A silent axis flip here quietly wrecks detection and evaluation, so we pin both the
round-trip identity and the (x,y,z) vs (z,y,x) reversal.
"""

import numpy as np
import pytest

from oncoct.io.resample import (
    itk_to_numpy_index,
    numpy_to_itk_index,
    voxel_to_world,
    world_to_voxel,
)


def test_world_voxel_round_trip():
    origin = np.array([-198.1, -195.0, -335.2])
    spacing = np.array([0.703125, 0.703125, 1.25])
    world = np.array([12.5, -60.0, -200.0])

    voxel = world_to_voxel(world, origin, spacing)
    back = voxel_to_world(voxel, origin, spacing)
    np.testing.assert_allclose(back, world, atol=1e-9)


def test_world_to_voxel_known_value():
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([2.0, 2.0, 2.0])
    world = np.array([10.0, 4.0, 8.0])
    np.testing.assert_allclose(world_to_voxel(world, origin, spacing), [5.0, 2.0, 4.0])


def test_axis_order_is_reversed():
    itk = np.array([1, 2, 3])                      # (x, y, z)
    np.testing.assert_array_equal(itk_to_numpy_index(itk), [3, 2, 1])   # (z, y, x)
    np.testing.assert_array_equal(numpy_to_itk_index(itk_to_numpy_index(itk)), itk)


def test_batched_conversion():
    origin = np.array([-10.0, -10.0, -10.0])
    spacing = np.array([1.0, 1.0, 1.0])
    world = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-10.0, -10.0, -10.0]])
    voxel = world_to_voxel(world, origin, spacing)
    np.testing.assert_allclose(voxel, [[10, 10, 10], [15, 15, 15], [0, 0, 0]])
    np.testing.assert_allclose(voxel_to_world(voxel, origin, spacing), world, atol=1e-9)


@pytest.mark.parametrize("bad_spacing", [0.0, -1.0])
def test_spacing_zero_or_negative_is_caller_error(bad_spacing):
    # Documents the assumption: spacing must be positive. Division by zero -> inf/nan,
    # which the caller must guard (spacing comes from the .mhd header).
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([bad_spacing, 1.0, 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = world_to_voxel(np.array([1.0, 1.0, 1.0]), origin, spacing)
    assert not np.isfinite(out[0]) or bad_spacing < 0
