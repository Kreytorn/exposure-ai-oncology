"""LUNA16 FROC / CPM evaluation wrapper.

Wraps the OFFICIAL noduleCADEvaluation script - rolling your own FROC inflates CPM.
CPM = mean sensitivity at {1/8, 1/4, 1/2, 1, 2, 4, 8} FP/scan, with 95% CI via
bootstrapping. Respect the 10-fold subset boundaries; treat "irrelevant findings"
(<3mm, non-nodule, 1-2 reader agreement) as ignored (neither TP nor FP).

Usage:
    python eval/froc_luna16.py --predictions preds.csv --annotations annotations.csv \
        --annotations-excluded annotations_excluded.csv --output froc_out/
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from oncoct.io.resample import numpy_to_itk_index, voxel_to_world


def write_predictions_csv(
    detections_by_series: dict[str, list],  # seriesuid -> list[detect.Detection]
    origin_spacing_by_series: dict[str, tuple[np.ndarray, np.ndarray]],  # -> (origin_xyz, spacing_xyz)
    out_csv: str,
) -> None:
    """Emit LUNA16-format predictions CSV: seriesuid,coordX,coordY,coordZ,probability (WORLD mm).

    The detector returns centers in numpy (z, y, x) VOXEL space. LUNA16 evaluation expects
    WORLD-mm (x, y, z). We flip axis order (numpy_to_itk_index) BEFORE voxel_to_world - this
    is the exact seam where the classic axis bug reappears, so it is done explicitly here.
    IMPORTANT: origin/spacing must be for the SAME grid the detector ran on (the resampled
    isotropic grid if detection ran post-resample).
    """
    rows = [("seriesuid", "coordX", "coordY", "coordZ", "probability")]
    for uid, dets in detections_by_series.items():
        origin_xyz, spacing_xyz = origin_spacing_by_series[uid]
        for det in dets:
            center_ijk = numpy_to_itk_index(np.asarray(det.center_zyx))  # (z,y,x) -> (x,y,z)
            wx, wy, wz = voxel_to_world(center_ijk, origin_xyz, spacing_xyz)
            rows.append((uid, f"{wx:.4f}", f"{wy:.4f}", f"{wz:.4f}", f"{det.score:.6f}"))
    with Path(out_csv).open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def evaluate_froc(
    predictions_csv: str,
    annotations_csv: str,
    excluded_csv: str,
    out_dir: str,
    series_uids_csv: str | None = None,
) -> dict:
    """Return {'cpm': float, 'sensitivities': {fp_rate: sens}, 'ci95': (lo, hi)}.

    Wraps the OFFICIAL LUNA16 noduleCADEvaluation. `excluded_csv` is the irrelevant-findings
    set (annotations_excluded.csv) - required for a correct CPM; do not omit it.
    """
    import sys

    import matplotlib

    matplotlib.use("Agg")

    official = Path(__file__).resolve().parent / "luna16_official"
    if not (official / "noduleCADEvaluationLUNA16.py").exists():
        raise FileNotFoundError(
            f"Official LUNA16 evaluation script not found at {official}. Copy "
            "evaluationScript/{noduleCADEvaluationLUNA16.py,NoduleFinding.py,tools} there. "
            "It ships as Python 2 - see eval/README_froc.md for the port."
        )
    sys.path.insert(0, str(official))
    import noduleCADEvaluationLUNA16 as luna  # noqa: E402

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The official seriesuids.csv lists all 888 scans. Scoring a subset against that list
    # would count every un-run scan as a scan with zero predictions, which drags the FP/scan
    # denominator up and sensitivity down. So evaluate against exactly the scans we ran.
    if series_uids_csv:
        # Preferred: the list of scans the detector actually RAN. A scan that produced zero
        # candidates still belongs in the denominator - dropping it would shrink the FP/scan
        # divisor and, if it held nodules, silently hide those misses.
        with Path(series_uids_csv).open() as f:
            evaluated = sorted({r[0].strip() for r in csv.reader(f) if r and r[0].strip()})
    else:
        with Path(predictions_csv).open() as f:
            rows = list(csv.reader(f))
        evaluated = sorted({r[0] for r in rows[1:] if r})
        print(
            "[froc] WARNING: scan list derived from predictions.csv, so any scan with zero "
            "detections is excluded from the FP/scan denominator. Pass series_uids_csv with "
            "every scan that was run for a correct CPM."
        )
    if not evaluated:
        raise ValueError(f"No scans to evaluate (predictions: {predictions_csv})")
    series_file = out / "seriesuids_evaluated.csv"
    with series_file.open("w", newline="") as f:
        csv.writer(f).writerows([[uid] for uid in evaluated])

    all_nodules, series_uids = luna.collect(annotations_csv, excluded_csv, str(series_file))
    fps, sens, thresholds, fps_bs, sens_mean, sens_lb, sens_ub = luna.evaluateCAD(
        series_uids, str(predictions_csv), str(out), all_nodules,
        os.path.splitext(os.path.basename(str(predictions_csv)))[0],
        maxNumberOfCADMarks=100, performBootstrapping=True,
        numberOfBootstrapSamples=1000, confidence=0.95,
    )

    # CPM = mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan (LUNA16 definition).
    fp_points = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sens_at = {fp: float(np.interp(fp, fps, sens)) for fp in fp_points}
    cpm = float(np.mean(list(sens_at.values())))

    ci95 = None
    if fps_bs is not None:
        lo = float(np.mean([np.interp(fp, fps_bs, sens_lb) for fp in fp_points]))
        hi = float(np.mean([np.interp(fp, fps_bs, sens_ub) for fp in fp_points]))
        ci95 = (lo, hi)

    # Count the INCLUDED nodules only. `all_nodules` also carries the excluded/irrelevant
    # findings (<3 mm, non-nodule, 1-2 reader agreement), which are ignored by the metric -
    # reporting their total would overstate the evaluated nodule count by ~30x.
    with Path(annotations_csv).open() as f:
        included = sum(1 for r in csv.DictReader(f) if r["seriesuid"] in set(evaluated))

    return {
        "cpm": cpm,
        "sensitivities": {str(k): v for k, v in sens_at.items()},
        "ci95": ci95,
        "n_scans": len(series_uids),
        "n_nodules_included": included,
        "max_sensitivity": float(np.max(sens)) if len(sens) else 0.0,
    }


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--annotations-excluded", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--series-uids", default=None,
                    help="CSV of every seriesuid the detector RAN (not just those with hits)")
    args = ap.parse_args()

    result = evaluate_froc(
        args.predictions, args.annotations, args.annotations_excluded, args.output,
        series_uids_csv=args.series_uids,
    )
    Path(args.output, "froc_subset.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
