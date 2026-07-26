"""Lesion segmentation: detection box -> precise 3D mask.

A common interface (`Segmenter` protocol + shared `LesionPrompt`) with two backends
selected by config:
  - MedSAM2 (primary): promptable, no training, research/education-only weights.
  - VISTA3D (fallback): MONAI foundation model, Apache-2.0, config-swappable.

Both backends take the SAME `LesionPrompt` and return a binary (z, y, x) mask, so the
caller swaps them by config flag with no signature change. Each backend uses whichever
fields of the prompt it needs (MedSAM2 uses the box + key slice; VISTA3D uses the point
+ class), and derives missing fields from what's present (box center -> point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LesionPrompt:
    """Backend-agnostic segmentation prompt for one lesion.

    Populated from a `Detection` (see oncoct.detect). Coordinates are in the SAME grid
    as the volume passed to `segment` (the resampled isotropic grid). `box_xyxy` is in
    ORIGINAL in-plane pixel coords on `key_slice_z`; MedSAM2 rescales it to 512-space
    internally. VISTA3D ignores the box and uses `point_zyx` (defaults to the box center).
    """

    key_slice_z: int
    box_xyxy: tuple[float, float, float, float]        # x_min, y_min, x_max, y_max (original px)
    point_zyx: tuple[int, int, int]                    # a point inside the lesion (z, y, x)
    vista3d_class: int | None = None                   # VISTA3D lung-lesion class index

    @classmethod
    def from_detection(cls, det, vista3d_class: int | None = None) -> "LesionPrompt":
        """Build a prompt from a detect.Detection (bbox_zyx + center_zyx)."""
        (z0, z1), (y0, y1), (x0, x1) = det.bbox_zyx
        cz, cy, cx = det.center_zyx
        key_z = int(round(cz))
        return cls(
            key_slice_z=key_z,
            box_xyxy=(float(x0), float(y0), float(x1), float(y1)),
            point_zyx=(key_z, int(round(cy)), int(round(cx))),
            vista3d_class=vista3d_class,
        )


class Segmenter(Protocol):
    """Both backends implement this. One method, one prompt type, one return type."""

    def segment(self, volume_zyx: np.ndarray, prompt: LesionPrompt) -> np.ndarray:
        """Return a binary 3D mask (z, y, x) on the same grid as `volume_zyx`."""
        ...
