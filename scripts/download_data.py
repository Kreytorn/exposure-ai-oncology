"""Download datasets: LUNA16, MSD Task06_Lung, and LIDC-IDRI (for malignancy labels).

LUNA16 raw is ~120GB - plan storage. LIDC DICOM+XML is fetched separately for pylidc.

    python scripts/download_data.py --dataset luna16    --out data/luna16
    python scripts/download_data.py --dataset msd_task06 --out data/Task06_Lung
    python scripts/download_data.py --dataset lidc        --out data/lidc-idri
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["luna16", "msd_task06", "lidc"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raise NotImplementedError(
        f"Fetch {args.dataset} to {args.out}. LUNA16: subset0..9 + annotations.csv + "
        "candidates_V2.csv (Zenodo/grand-challenge). MSD: DecathlonDataset Task06_Lung. "
        "LIDC: TCIA DICOM+XML for pylidc."
    )


if __name__ == "__main__":
    main()
