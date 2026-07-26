"""Preprocess LUNA16: read .mhd/.raw, resample to isotropic, cache to disk.

Resampling all 888 volumes is the main CPU/RAM cost — cache it, never recompute.
Converts annotation world-coords to voxel indices using each scan's origin/spacing
(via oncoct.io.resample), preserving HU. Output goes to the cache_root in paths.yaml.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--luna16-root", default="data/luna16")
    ap.add_argument("--cache-root", default="artifacts/cache")
    ap.add_argument("--spacing", nargs=3, type=float, default=[0.703125, 0.703125, 1.25])
    ap.parse_args()
    raise NotImplementedError(
        "For each .mhd: read_mhd -> assert_hounsfield_units -> resample_to_spacing -> "
        "cache .npz; convert annotations.csv world coords -> voxel; write a manifest."
    )


if __name__ == "__main__":
    main()
