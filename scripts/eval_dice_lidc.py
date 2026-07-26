"""Generate pred/GT NIfTI pairs for segmentation Dice, using LIDC radiologist contours.

LUNA16 ships nodule centroids and diameters, not voxel masks, so it cannot score Dice.
MSD Task06 could, but it is a separate ~9 GB download. LIDC-IDRI already gives us what we
need: the 4 readers' contours, from which pylidc builds a consensus mask.

Everything happens in the grid derived from ONE SimpleITK read of the DICOM series, then
resampled once - so prediction and ground truth share geometry by construction rather than
by assumption. Per lesion we crop a slab around the nodule so Dice measures the lesion, not
the surrounding empty chest.

    python scripts/eval_dice_lidc.py --out results/dice_pairs --limit 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import MedSAM2SubprocessClient  # noqa: E402

from oncoct.detect.nodule_detector import detect_nodules  # noqa: E402
from oncoct.io.resample import resample_to_spacing, world_to_voxel  # noqa: E402
from oncoct.labels.lidc_malignancy import patch_pylidc  # noqa: E402
from oncoct.segment import LesionPrompt  # noqa: E402

TARGET_SPACING = (0.703125, 0.703125, 1.25)


def consensus_mask_full(scan, cluster, shape_zyx):
    """Consensus mask of one nodule's readers, placed into a full (z, y, x) volume."""
    import pylidc as pl
    from pylidc.utils import consensus

    cmask, cbbox, _ = consensus(cluster, clevel=0.5)
    full = np.zeros(shape_zyx, dtype=np.uint8)
    # pylidc works in (row, col, slice) = (y, x, z); our volume is (z, y, x).
    ys, xs, zs = cbbox
    full[zs.start:zs.stop, ys.start:ys.stop, xs.start:xs.stop] = np.transpose(
        cmask.astype(np.uint8), (2, 0, 1)
    )
    return full


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="/content/drive/MyDrive/exposure_ai_Oncology/"
                                          "upload_to_drive/LIDC-IDRI/lidc_manifest.csv")
    ap.add_argument("--out", default="results/dice_pairs")
    ap.add_argument("--limit", type=int, default=12, help="max lesions to evaluate")
    ap.add_argument("--medsam2-checkpoint", required=True)
    ap.add_argument("--medsam2-config", default="configs/sam2.1_hiera_t512.yaml")
    ap.add_argument("--slab", type=int, default=32, help="+/- slices kept around the nodule")
    args = ap.parse_args()

    import csv

    patch_pylidc()
    import pylidc as pl

    out = Path(args.out)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    seg = MedSAM2SubprocessClient(
        checkpoint=args.medsam2_checkpoint, config=args.medsam2_config,
        workdir=out / "_work", hu_window=(-1000, 400),
    )

    rows = [r for r in csv.DictReader(open(args.manifest))
            if r["in_luna_subset0"] == "1" and int(r["n_annotations"]) > 0]
    done = 0
    for row in rows:
        if done >= args.limit:
            break
        uid = row["seriesuid"]
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
        if scan is None:
            continue
        try:
            reader = sitk.ImageSeriesReader()
            files = reader.GetGDCMSeriesFileNames(scan.get_path_to_dicom_files(), uid)
            reader.SetFileNames(files)
            raw = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
            clusters = [c for c in scan.cluster_annotations() if len(c) >= 3]
            if not clusters:
                continue

            gt_raw_shape = sitk.GetArrayFromImage(raw).shape
            img = resample_to_spacing(raw, TARGET_SPACING)
            vol = sitk.GetArrayFromImage(img)
            origin, spacing = np.array(img.GetOrigin()), np.array(img.GetSpacing())
            dets = detect_nodules(vol, TARGET_SPACING, score_threshold=0.5)
            if not dets:
                continue

            for ci, cluster in enumerate(clusters):
                if done >= args.limit:
                    break
                gt_full = consensus_mask_full(scan, cluster, gt_raw_shape)
                gt_img = sitk.GetImageFromArray(gt_full)
                gt_img.CopyInformation(raw)
                # Same resample as the image, nearest-neighbour: GT lands on the pred grid.
                gt_rs = sitk.GetArrayFromImage(
                    resample_to_spacing(gt_img, TARGET_SPACING, is_label=True)
                )
                if gt_rs.sum() == 0:
                    continue
                gz, gy, gx = (c.mean() for c in np.nonzero(gt_rs))

                # Match the detection whose centre is nearest this nodule; skip if the
                # detector missed it (that is a DETECTION miss, and belongs in FROC, not
                # in a segmentation-quality number).
                best = min(dets, key=lambda d: np.linalg.norm(
                    (np.array(d.center_zyx) - np.array([gz, gy, gx])) * np.array([1.25, .703125, .703125])))
                dist = float(np.linalg.norm(
                    (np.array(best.center_zyx) - np.array([gz, gy, gx])) * np.array([1.25, .703125, .703125])))
                if dist > 15.0:
                    continue

                mask = seg.segment(vol, LesionPrompt.from_detection(best))
                if mask.sum() == 0:
                    continue

                z0 = max(0, int(gz) - args.slab)
                z1 = min(vol.shape[0], int(gz) + args.slab + 1)
                case = f"{uid[-16:]}_n{ci}"
                for name, arr in (("pred", mask), ("gt", gt_rs)):
                    im = sitk.GetImageFromArray(arr[z0:z1].astype(np.uint8))
                    sitk.WriteImage(im, str(out / name / f"{case}.nii.gz"))
                done += 1
                print(f"  [{done}/{args.limit}] {case} det_dist={dist:.1f}mm "
                      f"gt_vox={int(gt_rs.sum())} pred_vox={int(mask.sum())}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {uid[-16:]}: {type(e).__name__}: {e}", flush=True)

    print(f"[dice] wrote {done} pred/gt pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
