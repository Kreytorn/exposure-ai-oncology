"""Machine-checkable caveats attached to a lesion, so the report can flag its own doubts.

Imaging-plane logic: pure function of a mask + its measurement + organ attribution. It lives
here rather than in `scripts/run_pipeline.py` so it is importable and unit-testable without the
heavy runtime stack (see `tests/test_quality_flags.py`).

The agent never computes these — it only transcribes them, like every other tool output.
"""

from __future__ import annotations

import numpy as np

# Drift runs ALONG the propagation axis, so the test is the mask's physical shape: a lung nodule
# is roughly isotropic, so its z-extent should track its in-plane long axis. Measured on the
# Round-1 gallery the ratio sits at 0.72-1.18 (tightly clustered at 1.0), so this threshold is
# well clear of normal variation and only fires on a mask that genuinely ran away in z.
DRIFT_Z_EXTENT_RATIO = 2.5

# LUNA16 and RECIST 1.1 both treat sub-3mm findings as noise rather than measurable lesions.
MIN_MEASURABLE_MM = 3.0

# Below this the organ map's top candidate is not a majority, so attribution is a coin-flip.
MIN_ORGAN_FRACTION = 0.5


def quality_flags(
    mask_zyx: np.ndarray,
    measurement,
    fractions: dict[str, float],
    organ: str,
    spacing_xyz: tuple[float, float, float],
) -> list[str]:
    """Return the machine-checkable caveats for one lesion.

    Args:
        mask_zyx: binary lesion mask, numpy (z, y, x) order.
        measurement: a :class:`oncoct.measure.recist.LesionMeasurement`.
        fractions: organ-name -> overlap fraction, from `attribute_organ`.
        organ: the winning organ name.
        spacing_xyz: voxel spacing in mm, ITK (x, y, z) order.
    """
    flags: list[str] = []
    mask = np.asarray(mask_zyx) > 0
    z_idx = np.flatnonzero(mask.sum(axis=(1, 2)) > 0)

    if z_idx.size and measurement.long_axis_mm > 0:
        # Do NOT go back to the earlier "end slice > 50% of the peak slice" test: a small nodule
        # spanning ~4 slices always has large end slices, so that version fired on 5/6 lesions in
        # Round 1 and on 18/22 in Round 2 (see results/RUN_LOG.md §3). A flag that fires on
        # everything is worse than no flag at all.
        z_extent_mm = (z_idx[-1] - z_idx[0] + 1) * spacing_xyz[2]
        if z_extent_mm > DRIFT_Z_EXTENT_RATIO * measurement.long_axis_mm:
            flags.append("propagation_drift")

    if measurement.long_axis_mm < MIN_MEASURABLE_MM:
        flags.append("below_3mm_not_measurable")
    if organ in ("unknown", "background") or not organ.startswith("lung"):
        flags.append("outside_lung_parenchyma")
    if fractions and max(fractions.values()) < MIN_ORGAN_FRACTION:
        flags.append("ambiguous_organ_attribution")
    return flags
