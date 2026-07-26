"""Cache 48^3 lesion patches + LIDC malignancy labels for the classifier fine-tune.

Patches are cropped from volumes resampled to the SAME isotropic spacing the pipeline uses
at inference (0.703125 x 0.703125 x 1.25 mm), with the SAME `_crop_patch` helper that
run_pipeline.py calls. Train/inference patch skew is the classic way a nodule classifier
silently scores garbage, so both sides share one code path by construction.

    python scripts/build_malignancy_patches.py --out artifacts/patches --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import _crop_patch  # noqa: E402  - one crop spec for train AND inference

from oncoct.io.resample import resample_to_spacing, world_to_voxel  # noqa: E402
from oncoct.labels.lidc_malignancy import patch_pylidc  # noqa: E402

TARGET_SPACING = (0.703125, 0.703125, 1.25)
PATCH = 48


def _scan_records(series_uid: str):
    """(centroid_world_xyz, median, is_malignant, n_readers) for one scan's nodules."""
    import pylidc as pl

    from oncoct.labels.lidc_malignancy import _centroid_to_world, _scan_geometry

    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
    if scan is None:
        return None, []
    geom = _scan_geometry(scan)
    if geom is None:
        return None, []
    out = []
    for cluster in scan.cluster_annotations():
        scores = [a.malignancy for a in cluster if a.malignancy is not None]
        if not scores:
            continue
        median = float(np.median(scores))
        if median == 3.0:                     # ambiguous - dropped, see plan.md §4
            continue
        c = np.mean([a.centroid for a in cluster], axis=0)
        out.append((_centroid_to_world(c, geom), median, median >= 4.0, len(scores)))
    return scan, out


def process_series(args_tuple):
    series_uid, patient_id, out_dir = args_tuple
    patch_pylidc()
    dst = Path(out_dir) / f"{series_uid}.npz"
    if dst.exists():
        return series_uid, "cached", 0

    try:
        scan, records = _scan_records(series_uid)
        if scan is None or not records:
            return series_uid, "no-labels", 0

        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(scan.get_path_to_dicom_files(), series_uid)
        reader.SetFileNames(files)
        img = reader.Execute()
        img = sitk.Cast(img, sitk.sitkFloat32)
        # Identical resample to the inference path - this is what makes the patch spec match.
        img = resample_to_spacing(img, TARGET_SPACING, is_label=False)
        vol = sitk.GetArrayFromImage(img)                       # (z, y, x), HU
        origin = np.array(img.GetOrigin())
        spacing = np.array(img.GetSpacing())

        patches, ys, medians, nreaders = [], [], [], []
        for world, median, is_mal, n_read in records:
            ijk = world_to_voxel(np.asarray(world), origin, spacing)   # (x, y, z)
            centroid_zyx = (ijk[2], ijk[1], ijk[0])
            if not (0 <= centroid_zyx[0] < vol.shape[0]
                    and 0 <= centroid_zyx[1] < vol.shape[1]
                    and 0 <= centroid_zyx[2] < vol.shape[2]):
                continue
            p = _crop_patch(vol, centroid_zyx, size=PATCH)
            if p.shape != (PATCH, PATCH, PATCH):
                continue
            patches.append(p.astype(np.float32))
            ys.append(bool(is_mal))
            medians.append(median)
            nreaders.append(n_read)

        if not patches:
            return series_uid, "no-patches", 0

        np.savez_compressed(
            dst,
            patches=np.stack(patches), y=np.array(ys), median=np.array(medians),
            n_readers=np.array(nreaders), patient_id=patient_id, series_uid=series_uid,
        )
        return series_uid, "ok", len(patches)
    except Exception as e:  # noqa: BLE001
        return series_uid, f"ERROR {type(e).__name__}: {e}", 0


def main() -> None:
    import csv
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="/content/drive/MyDrive/exposure_ai_Oncology/"
                                          "upload_to_drive/LIDC-IDRI/lidc_manifest.csv")
    ap.add_argument("--out", default="artifacts/patches")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest)))
    jobs = [(r["seriesuid"], r["patient_id"], str(out)) for r in rows]
    print(f"[patches] {len(jobs)} LIDC series -> {out}", flush=True)

    total = 0
    with Pool(args.workers) as pool:
        for i, (uid, status, n) in enumerate(pool.imap_unordered(process_series, jobs), 1):
            total += n
            if status.startswith("ERROR") or i % 20 == 0 or n == 0:
                print(f"  [{i}/{len(jobs)}] {uid[-16:]} {status} n={n}", flush=True)
    print(f"[patches] done: {total} labelled patches cached", flush=True)


if __name__ == "__main__":
    main()
