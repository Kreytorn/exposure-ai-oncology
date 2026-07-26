"""Organ context + lesion->organ attribution.

Runs TotalSegmentator `total` once per study (cached) to get a 117-class organ label
map, then attributes each lesion to an organ/lobe by MAJORITY MASK OVERLAP against that
map. This is the non-hallucinated mechanism for "which organ/lobe is this lesion in":
the agent reads a mapped name, it does not guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_organ_map(volume_path: Path, cache_dir: Path, fast: bool = False) -> np.ndarray:
    """Run TotalSegmentator `total` (multilabel) and cache the result.

    Returns the multilabel array in numpy (z, y, x) order; label index -> organ name is
    read from totalsegmentator's map_to_binary. Cached per study UID.
    """
    import SimpleITK as sitk

    volume_path = Path(volume_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache per study: TotalSegmentator is ~1-2 min/scan, and re-running it is pure waste.
    stem = volume_path.name.split(".nii")[0]
    cached = cache_dir / f"{stem}_organmap.nii.gz"

    if not cached.exists():
        from totalsegmentator.python_api import totalsegmentator

        totalsegmentator(
            input=str(volume_path),
            output=str(cached),
            task="total",
            ml=True,          # one multilabel volume, not 117 separate binary masks
            fast=fast,
            device="gpu",
            quiet=True,
        )
        if not cached.exists():
            raise RuntimeError(f"TotalSegmentator produced no output at {cached}")

    seg = sitk.ReadImage(str(cached))
    ref = sitk.ReadImage(str(volume_path))

    # GRID CONTRACT (brief §5.2): attribute_organ does organ_map[lesion_mask], so the organ
    # map must sit on exactly the volume's grid. TotalSegmentator resamples internally
    # (1.5 mm for `total`) and normally restores the input geometry — but if any version
    # hands back a different grid, resample here rather than let a silent mismatch through.
    same_grid = (
        seg.GetSize() == ref.GetSize()
        and np.allclose(seg.GetSpacing(), ref.GetSpacing(), atol=1e-4)
        and np.allclose(seg.GetOrigin(), ref.GetOrigin(), atol=1e-3)
        and np.allclose(seg.GetDirection(), ref.GetDirection(), atol=1e-4)
    )
    if not same_grid:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)   # labels: never interpolate
        resampler.SetDefaultPixelValue(0)
        seg = resampler.Execute(seg)

    # sitk -> numpy already yields (z, y, x), matching volume_zyx in run_pipeline.
    return sitk.GetArrayFromImage(seg).astype(np.int16)


def attribute_organ(
    lesion_mask_zyx: np.ndarray,
    organ_map_zyx: np.ndarray,
    label_to_name: dict[int, str],
) -> tuple[str, dict[str, float]]:
    """Attribute a lesion to an organ by majority overlap with the organ map.

    Returns (majority_organ_name, {organ_name: overlap_fraction}). Background (label 0)
    is ignored unless the lesion overlaps nothing but background.
    """
    mask = np.asarray(lesion_mask_zyx) > 0
    labels = np.asarray(organ_map_zyx)[mask]
    if labels.size == 0:
        return "unknown", {}
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    fractions = {
        label_to_name.get(int(v), f"label_{int(v)}"): float(c) / float(total)
        for v, c in zip(values, counts, strict=True)
        if int(v) != 0
    }
    if not fractions:
        return "background", {}
    majority = max(fractions, key=fractions.get)
    return majority, fractions


def split_organ_lobe(attributed: str) -> tuple[str, str | None]:
    """TotalSegmentator label name -> (organ, lobe_or_segment).

    `attribute_organ` returns TotalSegmentator's own name, which for lungs is already
    lobe-level ("lung_lower_lobe_left"). The report wants the organ and the sub-segment
    separately, so split rather than throw the lobe away — lobe is clinically the useful
    half, and it came from a segmentation, not from the model guessing.
    """
    if attributed.startswith("lung_") and "lobe" in attributed:
        parts = attributed.split("_")            # lung_lower_lobe_left
        side = parts[-1]
        position = parts[1]
        return "lung", f"{position} lobe, {side}"
    return attributed, None


def label_to_name() -> dict[int, str]:
    """TotalSegmentator label-index -> organ-name mapping.

    The `class_map` layout is VERSION-DEPENDENT (task-keyed dict vs flat dict). We try the
    common accessors and fall back to an empty map — `attribute_organ` already degrades to
    `label_{n}` names, so a bad accessor must NOT hard-abort the run. Verify at runtime with
    `from totalsegmentator.map_to_binary import class_map; class_map.keys()`.
    """
    try:
        from totalsegmentator.map_to_binary import class_map

        # Kept as an explicit membership test rather than .get(): class_map's type is
        # version-dependent and is not guaranteed to be a plain dict with .get().
        cm = class_map["total"] if "total" in class_map else class_map  # noqa: SIM401
        return {int(k): str(v) for k, v in dict(cm).items()}
    except Exception as e:  # noqa: BLE001
        print(f"[organ_map] WARN: label map unavailable ({e}); using label_N fallbacks")
        return {}
