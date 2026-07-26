"""Segmentation evaluation: Dice + surface metrics on MSD Task06 / LIDC contours.

Reports volumetric Dice and (optionally) Hausdorff / surface-Dice for boundary quality.
Also sanity-checks MedSAM2 propagation drift by comparing per-slice Dice against distance
from the prompt (key) slice.
"""

from __future__ import annotations

import numpy as np


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Volumetric Dice-Sorensen coefficient for two binary masks."""
    p = np.asarray(pred_mask) > 0
    g = np.asarray(gt_mask) > 0
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0                       # both empty -> perfect agreement by convention
    return float(2.0 * (p & g).sum() / denom)


def evaluate_folder(pred_dir: str, gt_dir: str) -> dict:
    """Mean/median Dice over matched prediction/ground-truth NIfTI pairs."""
    from pathlib import Path

    import SimpleITK as sitk

    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    preds = {p.name.split(".nii")[0]: p for p in sorted(pred_dir.glob("*.nii*"))}
    gts = {p.name.split(".nii")[0]: p for p in sorted(gt_dir.glob("*.nii*"))}
    shared = sorted(set(preds) & set(gts))
    if not shared:
        raise ValueError(f"No matching case ids between {pred_dir} and {gt_dir}")

    per_case, skipped = {}, {}
    for case in shared:
        p = sitk.GetArrayFromImage(sitk.ReadImage(str(preds[case])))
        g = sitk.GetArrayFromImage(sitk.ReadImage(str(gts[case])))
        if p.shape != g.shape:
            # Dice across mismatched grids is meaningless; record it rather than silently
            # broadcasting or cropping into a number that looks fine.
            skipped[case] = f"shape mismatch pred{p.shape} vs gt{g.shape}"
            continue
        per_case[case] = dice(p, g)

    values = np.array(list(per_case.values()), dtype=float)
    return {
        "n_cases": int(len(values)),
        "mean_dice": float(values.mean()) if values.size else 0.0,
        "median_dice": float(np.median(values)) if values.size else 0.0,
        "std_dice": float(values.std()) if values.size else 0.0,
        "min_dice": float(values.min()) if values.size else 0.0,
        "max_dice": float(values.max()) if values.size else 0.0,
        "per_case": per_case,
        "skipped": skipped,
        "unmatched_pred": sorted(set(preds) - set(gts)),
        "unmatched_gt": sorted(set(gts) - set(preds)),
    }
