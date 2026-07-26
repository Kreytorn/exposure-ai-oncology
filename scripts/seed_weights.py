"""Pre-seed model weight caches so ephemeral Colab sessions don't re-download on reset.

On Colab, point the cache dirs at Google Drive (configs/paths.yaml) and run once; later
sessions reuse the cached weights instead of downloading. Covers TotalSegmentator, the
MONAI detection bundle, and MedSAM2 checkpoints.

    python scripts/seed_weights.py --to artifacts/weights
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--to", default="artifacts/weights")
    ap.parse_args()
    raise NotImplementedError(
        "Download: TotalSegmentator weights (totalseg_download_weights / TOTALSEG_WEIGHTS_PATH), "
        "MONAI lung_nodule_ct_detection bundle, MedSAM2 checkpoints (bash download.sh -> checkpoints/)."
    )


if __name__ == "__main__":
    main()
