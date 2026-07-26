"""RECIST measurement agreement vs reference readers.

Reports mean absolute long-axis error (mm), Bland-Altman limits of agreement, and
run-to-run reproducibility. Validate against the public CT dataset that ships RECIST
measurements + segmentation masks, following the multi-observer protocol.
"""

from __future__ import annotations

import numpy as np


def mean_abs_error(pred_mm: np.ndarray, ref_mm: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred_mm) - np.asarray(ref_mm))))


def bland_altman(pred_mm: np.ndarray, ref_mm: np.ndarray) -> dict:
    """Return {'bias': float, 'loa_lower': float, 'loa_upper': float}."""
    pred = np.asarray(pred_mm, dtype=float)
    ref = np.asarray(ref_mm, dtype=float)
    diff = pred - ref
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return {"bias": bias, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd}
