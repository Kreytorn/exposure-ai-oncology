"""Guards on the lesion quality flags - specifically that `propagation_drift` stays useful.

Round 1 shipped a drift test ("end slice > 50% of the peak slice") that fired on 5/6 lesions,
was judged worse than no flag, and was replaced by a physical z-extent test. The replacement was
never written back into the returned code, so the repo silently kept the rejected version and
Round 2's gallery fired it on 18/22 lesions (see results/RUN_LOG.md §3). These tests pin BOTH
directions so the regression cannot come back unnoticed: a compact nodule must not be flagged,
and a mask that ran away along the propagation axis must be.
"""

import numpy as np

from oncoct.measure.recist import measure_lesion
from oncoct.report.quality import quality_flags

SPACING = (0.703125, 0.703125, 1.25)  # the pipeline's target spacing, ITK (x, y, z)


def _cylinder(n_slices: int, radius_vox: int, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """A disc of `radius_vox` repeated over `n_slices` centred slices."""
    mask = np.zeros(shape_zyx, dtype=bool)
    yy, xx = np.ogrid[: shape_zyx[1], : shape_zyx[2]]
    cy, cx = shape_zyx[1] // 2, shape_zyx[2] // 2
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius_vox**2
    z0 = (shape_zyx[0] - n_slices) // 2
    mask[z0 : z0 + n_slices] = disc
    return mask


def _flags(mask: np.ndarray, organ: str = "lung_upper_lobe_left") -> list[str]:
    m = measure_lesion(mask, spacing_xyz=SPACING)
    return quality_flags(mask, m, {organ: 1.0}, organ, SPACING)


def test_compact_nodule_is_not_flagged_as_drift():
    # ~8mm lesion spanning 6 slices: z-extent 7.5mm vs long axis ~8mm -> ratio ~0.94, the
    # normal range Round 1 measured (0.72-1.18). The rejected heuristic flagged exactly this.
    mask = _cylinder(n_slices=6, radius_vox=6, shape_zyx=(48, 64, 64))
    assert "propagation_drift" not in _flags(mask)


def test_thin_few_slice_nodule_is_not_flagged_as_drift():
    # The specific false positive that killed the old test: a small nodule only 4 slices thick
    # has end slices nearly as large as its peak slice, by construction.
    mask = _cylinder(n_slices=4, radius_vox=5, shape_zyx=(48, 64, 64))
    assert "propagation_drift" not in _flags(mask)


def test_mask_running_away_along_z_is_flagged():
    # A narrow tube tracked far along the propagation axis: z-extent 50mm vs long axis ~4mm.
    mask = _cylinder(n_slices=40, radius_vox=2, shape_zyx=(64, 64, 64))
    assert "propagation_drift" in _flags(mask)


def test_drift_test_uses_physical_z_extent_not_slice_count():
    # Same voxel geometry, coarser z spacing -> larger physical extent -> flag flips on.
    mask = _cylinder(n_slices=10, radius_vox=6, shape_zyx=(48, 64, 64))
    m = measure_lesion(mask, spacing_xyz=SPACING)
    fine = quality_flags(mask, m, {"lung": 1.0}, "lung", (0.703125, 0.703125, 1.25))
    coarse = quality_flags(mask, m, {"lung": 1.0}, "lung", (0.703125, 0.703125, 5.0))
    assert "propagation_drift" not in fine
    assert "propagation_drift" in coarse


def test_non_lung_organ_is_flagged_outside_parenchyma():
    # This flag is correct behaviour, not a bug: Round 2 fired it on 4/22 lesions (2 background,
    # 2 rib) at the deliberately low 0.02 detection threshold. Pin it so it stays.
    mask = _cylinder(n_slices=6, radius_vox=6, shape_zyx=(48, 64, 64))
    assert "outside_lung_parenchyma" in _flags(mask, organ="rib_left_1")
    assert "outside_lung_parenchyma" not in _flags(mask, organ="lung_upper_lobe_left")
