"""MedSAM2 segmentation backend: detection box on a key slice -> 3D lesion mask.

Because MedSAM2 pins an incompatible torch/CUDA stack (torch 2.5.1 / cu124 / py3.12),
this backend runs in the isolated `medsam2` conda env. The `segment()` here is the
in-env implementation; the primary `oncoct` env calls it via the subprocess worker
(scripts/medsam2_worker.py) so the two envs never share a process. See that worker for
the on-disk contract (volume.npz + prompt.json -> mask.npz).

The MedSAM2 model config and checkpoint live inside the CLONED MedSAM2 repo, not in
oncoct/configs/. The config is a Hydra config-name (e.g. "configs/sam2.1_hiera_t512.yaml"
resolved relative to the MedSAM2 package), and the checkpoint is checkpoints/MedSAM2_CTLesion.pt
fetched by MedSAM2's download.sh.

Critical steps (each is a silent-failure source if skipped):
  1. Preprocess EXACTLY as training: HU-window clip -> uint8 0-255 -> grayscale->RGB ->
     resize each slice to 512x512 -> ImageNet normalize.
  2. Place the box on the lesion's KEY slice, rescaled to 512x512 space
     (x * 512/orig_W, y * 512/orig_H).
  3. Propagate in BOTH directions (reverse=False AND reverse=True) -- forgetting the
     reverse pass leaves half the lesion unsegmented.
  4. Threshold logits > 0. Return the mask ON THE SAME (resampled) GRID as the input
     volume -- do NOT resize back to a different resolution here; the caller measures at
     the volume's spacing (see plan.md / measure.recist). If you resize, you must pass
     the matching spacing to measure_lesion.
"""

from __future__ import annotations

import numpy as np

from oncoct.segment import LesionPrompt


class MedSAM2Segmenter:
    """box -> 3D mask via SAM2 memory-propagation across axial slices."""

    def __init__(self, checkpoint: str, config: str, device: str = "cuda"):
        self.checkpoint = checkpoint    # path to MedSAM2_CTLesion.pt in the MedSAM2 clone
        self.config = config            # Hydra config-name within the MedSAM2 package
        self.device = device
        self._predictor = None

    def _lazy_build(self):
        # from sam2.build_sam import build_sam2_video_predictor_npz
        # self._predictor = build_sam2_video_predictor_npz(self.config, self.checkpoint)
        raise NotImplementedError("Build build_sam2_video_predictor_npz with the t512 config.")

    def segment(
        self,
        volume_zyx: np.ndarray,
        prompt: LesionPrompt,
        hu_window: tuple[int, int] = (-1000, 400),
    ) -> np.ndarray:
        """Return a binary 3D mask (z, y, x) on the same grid as `volume_zyx`.

        Uses `prompt.box_xyxy` (original in-plane pixel coords) on `prompt.key_slice_z`;
        rescales the box to 512-space internally, propagates BOTH directions.
        """
        raise NotImplementedError(
            "Preprocess -> init_state -> add_new_points_or_box(box rescaled to 512 on key "
            "slice) -> propagate_in_video BOTH directions -> threshold. Return on input grid."
        )
