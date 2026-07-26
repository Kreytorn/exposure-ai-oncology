"""VISTA3D segmentation backend (Apache-2.0 fallback for MedSAM2).

MONAI's VISTA3D foundation model does automatic (127-class) + interactive (3D point)
3D CT segmentation. Used when a commercially-safe / no-license-caveat path is required,
or when the MedSAM2 env pin causes trouble.

Implements the shared Segmenter interface (segment(volume, LesionPrompt) -> mask). It
uses the prompt's `point_zyx` (a point inside the lesion, defaulted to the box center by
LesionPrompt) and, if set, `vista3d_class`. NOTE: VISTA3D's automatic class set covers
127 anatomical structures; lung lesions/tumors may fall under a specific class index OR
require the interactive point branch. The Round-1 executor must confirm the correct
lung-lesion class index (or use pure point-prompt) and record it in the RUN_LOG.
"""

from __future__ import annotations

import numpy as np

from oncoct.segment import LesionPrompt


class Vista3DSegmenter:
    """Point/automatic 3D CT lesion segmentation via the MONAI VISTA3D bundle."""

    def __init__(self, bundle: str = "vista3d", device: str = "cuda"):
        self.bundle = bundle
        self.device = device

    def segment(self, volume_zyx: np.ndarray, prompt: LesionPrompt) -> np.ndarray:
        """Return a binary 3D mask (z, y, x) for the lesion at `prompt.point_zyx`.

        Uses the interactive point branch (robust for arbitrary lesions). If
        `prompt.vista3d_class` is set, may use the automatic branch for that class.
        """
        raise NotImplementedError(
            "Load the MONAI VISTA3D bundle; run point-prompt (prompt.point_zyx) inference. "
            "Confirm the lung-lesion class index if using the automatic branch; log it."
        )
