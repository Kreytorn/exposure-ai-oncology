"""Run the detector over a folder of LUNA16 .mhd scans and emit a world-coord preds CSV.

This is the input to the FROC evaluation. Detections are kept at a LOW score threshold —
FROC sweeps the whole operating curve, so thresholding at 0.5 here would truncate the
curve and understate sensitivity at the higher FP/scan operating points.

    python scripts/run_detection_subset.py --series-dir data/luna16/subset0 --out results/metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# `eval/` lives at the repo root and is NOT pip-installed (pyproject installs only src/).
# Running `python scripts/run_detection_subset.py` puts scripts/ on sys.path, not the repo
# root, so `from eval...` would fail. Add the repo root explicitly so the script works no
# matter how it's invoked (don't rely on PYTHONPATH being set).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.froc_luna16 import write_predictions_csv
from oncoct.detect.nodule_detector import detect_nodules
from oncoct.io.dicom_nifti import read_mhd
from oncoct.io.resample import resample_to_spacing

TARGET_SPACING = (0.703125, 0.703125, 1.25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series-dir", required=True)
    ap.add_argument("--out", default="results/metrics")
    ap.add_argument("--score-threshold", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scans = sorted(Path(args.series_dir).glob("*.mhd"))
    if args.limit:
        scans = scans[: args.limit]
    print(f"[detect] {len(scans)} scans -> {out}", flush=True)

    dets_by_series: dict[str, list] = {}
    geom_by_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    for i, p in enumerate(scans, 1):
        uid = p.stem
        img = resample_to_spacing(read_mhd(p), TARGET_SPACING)
        vol = sitk.GetArrayFromImage(img)
        dets = detect_nodules(vol, TARGET_SPACING, score_threshold=args.score_threshold)
        dets_by_series[uid] = dets
        # Geometry of the grid the detector ACTUALLY ran on (post-resample), which is what
        # voxel->world must use. Passing the raw .mhd origin/spacing here would shift every
        # prediction and silently collapse the FROC.
        geom_by_series[uid] = (np.array(img.GetOrigin()), np.array(img.GetSpacing()))
        print(f"  [{i}/{len(scans)}] {uid[-16:]} {len(dets):4d} dets "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    csv_path = out / "preds.csv"
    write_predictions_csv(dets_by_series, geom_by_series, str(csv_path))
    # Every scan RUN, including any that yielded no candidates — this is the correct
    # FROC denominator, and preds.csv alone cannot reconstruct it.
    import csv as _csv
    with (out / "evaluated_seriesuids.csv").open("w", newline="") as f:
        _csv.writer(f).writerows([[p.stem] for p in scans])
    n = sum(len(v) for v in dets_by_series.values())
    (out / "detection_run.json").write_text(json.dumps({
        "n_scans": len(scans), "n_detections": n,
        "score_threshold": args.score_threshold,
        "mean_detections_per_scan": n / max(len(scans), 1),
        "target_spacing_xyz_mm": list(TARGET_SPACING),
        "seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"[detect] {n} detections over {len(scans)} scans "
          f"({n/max(len(scans),1):.1f}/scan) in {time.time()-t0:.0f}s -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
