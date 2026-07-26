"""Fine-tune the malignancy classifier head (the only training step in the pipeline).

Freezes a 3D CNN encoder and trains a small head on LIDC malignancy labels
(median-of-4, drop median==3), joined to LUNA16 scans by SeriesInstanceUID. Designed
for a short A100 run (minutes to ~1-2h). Checkpoints frequently — Colab sessions die.

    python scripts/finetune_malignancy.py --epochs 30 --ckpt artifacts/weights/malignancy

Evaluation is patient-grouped k-fold: a patient's nodules never straddle the train/val
split. With ~200 nodules a single hold-out split would swing several AUROC points on the
luck of the draw, so the headline number is the cross-validated mean, and the shipped
checkpoint is refit on all folds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oncoct.classify.malignancy import (
    ENCODER_HU_WINDOW,
    FEATURE_DIM,
    PATCH_SIZE,
    TARGET_SPACING_XYZ,
    MalignancyHead,
    _flip_views,
    build_encoder,
    encode,
)


def load_dataset(patch_dir: Path):
    """Stack the per-series patch caches into arrays + patient groups."""
    patches, ys, groups, medians = [], [], [], []
    for f in sorted(patch_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        patches.append(d["patches"])
        ys.append(d["y"])
        medians.append(d["median"])
        groups += [str(d["patient_id"])] * len(d["y"])
    if not patches:
        raise SystemExit(f"No patch caches in {patch_dir}; run build_malignancy_patches.py first.")
    return (np.concatenate(patches), np.concatenate(ys).astype(np.float32),
            np.array(groups), np.concatenate(medians))


def precompute_features(trunk, patches, device):
    """(N, 48^3) -> (N, 8, FEATURE_DIM): frozen features for all 8 flip views.

    The encoder never changes, so its outputs are computed once and reused for every epoch
    and every fold. That turns "train a 3D CNN" into "train a 35k-parameter MLP on cached
    vectors" — seconds instead of GPU-hours, which is exactly the budget this round has.
    """
    views = np.stack([np.stack(list(_flip_views(p))) for p in patches])   # (N, 8, 48,48,48)
    n, v = views.shape[:2]
    feats = encode(trunk, views.reshape(n * v, *views.shape[2:]), device)
    return feats.reshape(n, v, -1)


def train_head(feats_train, y_train, feats_val, y_val, epochs, lr, device, seed=0):
    """Train the MLP head on flip-augmented features; return (head, TTA-averaged val probs)."""
    torch.manual_seed(seed)
    n, v, dim = feats_train.shape
    x = torch.from_numpy(feats_train.reshape(n * v, dim)).float().to(device)
    y = torch.from_numpy(np.repeat(y_train, v)).float().to(device)

    head = MalignancyHead(in_dim=dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # The set is ~28% positive; without pos_weight the head can score well by calling
    # everything benign, which is exactly the failure AUPRC is meant to expose.
    pos_weight = torch.tensor([float((len(y) - y.sum()) / max(float(y.sum()), 1.0))]).to(device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ep in range(epochs):
        head.train()
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:            # BatchNorm1d needs more than one sample
                continue
            opt.zero_grad()
            lossf(head(x[idx]), y[idx]).backward()
            opt.step()
        sched.step()

    head.eval()
    if feats_val is None:
        return head, None
    with torch.no_grad():
        nv, vv, _ = feats_val.shape
        xv = torch.from_numpy(feats_val.reshape(nv * vv, dim)).float().to(device)
        probs = torch.sigmoid(head(xv)).reshape(nv, vv).mean(dim=1)   # TTA over the 8 views
    return head, probs.cpu().numpy()


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patches", default="artifacts/patches")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ckpt", default="artifacts/weights/malignancy")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--checkpoint-every-min", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    patches, y, groups, _medians = load_dataset(Path(args.patches))
    print(f"[finetune] {len(y)} nodules | {int(y.sum())} malignant / {int((1-y).sum())} benign "
          f"| {len(set(groups))} patients", flush=True)

    from oncoct.detect.nodule_detector import default_bundle_dir
    trunk = build_encoder(default_bundle_dir(), device)
    t0 = time.time()
    feats = precompute_features(trunk, patches, device)
    print(f"[finetune] frozen features {feats.shape} in {time.time()-t0:.0f}s", flush=True)

    # ---- patient-grouped cross-validation --------------------------------------------
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    fold_metrics, oof = [], np.zeros(len(y))
    last_ckpt = time.time()
    for k, (tr, va) in enumerate(skf.split(feats, y, groups), 1):
        assert not (set(groups[tr]) & set(groups[va])), "patient leaked across the split"
        head, probs = train_head(feats[tr], y[tr], feats[va], y[va],
                                 args.epochs, args.lr, device, seed=k)
        oof[va] = probs
        m = {"fold": k, "n_val": int(len(va)), "n_pos": int(y[va].sum()),
             "auroc": float(roc_auc_score(y[va], probs)),
             "auprc": float(average_precision_score(y[va], probs))}
        fold_metrics.append(m)
        print(f"  fold {k}: n={m['n_val']:3d} pos={m['n_pos']:2d} "
              f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}", flush=True)
        if time.time() - last_ckpt > args.checkpoint_every_min * 60:
            torch.save({"state_dict": head.state_dict(), "fold": k}, ckpt_dir / "fold_latest.pt")
            last_ckpt = time.time()

    aurocs = np.array([m["auroc"] for m in fold_metrics])
    auprcs = np.array([m["auprc"] for m in fold_metrics])
    oof_auroc = float(roc_auc_score(y, oof))
    oof_auprc = float(average_precision_score(y, oof))
    print(f"[finetune] CV AUROC {aurocs.mean():.3f} +/- {aurocs.std():.3f} | "
          f"AUPRC {auprcs.mean():.3f} +/- {auprcs.std():.3f}", flush=True)
    print(f"[finetune] pooled out-of-fold AUROC {oof_auroc:.3f} AUPRC {oof_auprc:.3f}", flush=True)

    # ---- logistic-regression reference on the same folds ------------------------------
    lr_oof = np.zeros(len(y))
    for tr, va in skf.split(feats, y, groups):
        n, v, dim = feats[tr].shape
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(feats[tr].reshape(n * v, dim), np.repeat(y[tr], v))
        lr_oof[va] = clf.predict_proba(feats[va].mean(axis=1))[:, 1]
    lr_auroc = float(roc_auc_score(y, lr_oof))
    lr_auprc = float(average_precision_score(y, lr_oof))
    print(f"[finetune] logistic-regression reference: out-of-fold "
          f"AUROC {lr_auroc:.3f} AUPRC {lr_auprc:.3f}", flush=True)

    # ---- refit on everything; this is the shipped checkpoint ---------------------------
    final, _ = train_head(feats, y, None, None, args.epochs, args.lr, device, seed=1234)
    patch_spec = {
        "patch_size": PATCH_SIZE, "spacing_xyz_mm": list(TARGET_SPACING_XYZ),
        "hu_window": list(ENCODER_HU_WINDOW), "tta_flip_views": 8,
        "encoder": "lung_nodule_ct_detection resnet50 trunk through layer2 (frozen)",
    }
    out = ckpt_dir / "malignancy_head.pt"
    torch.save({"state_dict": final.state_dict(), "in_dim": FEATURE_DIM, "hidden": 32,
                "patch_spec": patch_spec}, out)
    print(f"[finetune] saved {out} ({out.stat().st_size/1e6:.2f} MB)", flush=True)

    metrics = {
        "n_samples": int(len(y)), "n_malignant": int(y.sum()), "n_benign": int((1 - y).sum()),
        "n_patients": int(len(set(groups))),
        "label_policy": {
            "aggregate": "median of per-reader LIDC malignancy (1-5)",
            "drop_ambiguous_median_3": True,
            "malignant_if": "median >= 4",
            "note": ("median==3 nodules are dropped as indeterminate. They are the single "
                     "largest bucket in LIDC, so this both shrinks the usable set and "
                     "removes the hardest cases; reported metrics are optimistic relative "
                     "to an unfiltered screening population."),
        },
        "patch_spec": patch_spec,
        "backend": "cnn_head",
        "validation": "patient-grouped StratifiedGroupKFold; no patient in both splits",
        "folds": fold_metrics,
        "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
        "auprc_mean": float(auprcs.mean()), "auprc_std": float(auprcs.std()),
        "auroc_out_of_fold": oof_auroc, "auprc_out_of_fold": oof_auprc,
        "logreg_reference_auroc_out_of_fold": lr_auroc,
        "logreg_reference_auprc_out_of_fold": lr_auprc,
        "prevalence": float(y.mean()),
        "trainable_params": int(sum(p.numel() for p in final.parameters())),
        "shipped_checkpoint": ("refit on all folds after CV; the AUROC/AUPRC above are the "
                               "cross-validated estimates, NOT scores of this checkpoint on "
                               "held-out data"),
    }
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
        print(f"[finetune] metrics -> {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
