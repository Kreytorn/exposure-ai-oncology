"""Extract nodule malignancy labels from LIDC-IDRI and join to LUNA16 scans.

LUNA16 has NO malignancy labels. The parent LIDC-IDRI has 4 radiologists rating each
>=3mm nodule 1-5 for malignancy, stored in per-scan XML. We use pylidc to cluster the
4 readers into nodules, aggregate malignancy by MEDIAN, DROP median==3 (ambiguous), and
join to LUNA16 scans on SeriesInstanceUID.

The drop-median==3 policy shrinks and biases the usable set - it is a stated modelling
choice (plan.md §4), configurable in configs/pipeline.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoduleLabel:
    series_uid: str
    centroid_world_xyz: tuple[float, float, float]
    malignancy_median: float          # 1..5 (median of 4 readers)
    is_malignant: bool                # median >= 4
    n_readers: int


def extract_labels(
    lidc_root: str,
    aggregate: str = "median",
    drop_ambiguous_median_3: bool = True,
) -> list[NoduleLabel]:
    """Cluster LIDC annotations into nodules and derive binary malignancy labels.

    Requires a configured ``~/.pylidcrc`` pointing at the LIDC DICOM directory.
    """
    import numpy as np

    patch_pylidc()
    import pylidc as pl

    if aggregate != "median":
        raise ValueError(f"only 'median' aggregation is implemented, got {aggregate!r}")

    labels: list[NoduleLabel] = []
    for scan in pl.query(pl.Scan).all():
        try:
            if not Path(scan.get_path_to_dicom_files()).is_dir():
                continue                       # pylidc's DB covers all 1018 scans; we only
        except Exception:                      # hold DICOM for the subset we downloaded.
            continue

        try:
            clusters = scan.cluster_annotations()
        except Exception as e:                 # noqa: BLE001
            print(f"[lidc] skipping {scan.series_instance_uid}: clustering failed ({e})")
            continue

        geom = _scan_geometry(scan)
        if geom is None:
            continue

        for cluster in clusters:
            scores = [a.malignancy for a in cluster if a.malignancy is not None]
            if not scores:
                continue
            median = float(np.median(scores))
            # median == 3 is "indeterminate": the readers neither called it benign nor
            # malignant. Dropping it removes the hardest cases and inflates the reported
            # AUROC relative to a real screening population - a stated bias, not a bug.
            if drop_ambiguous_median_3 and median == 3.0:
                continue
            centroid_ijk = np.mean([a.centroid for a in cluster], axis=0)
            labels.append(
                NoduleLabel(
                    series_uid=scan.series_instance_uid,
                    centroid_world_xyz=_centroid_to_world(centroid_ijk, geom),
                    malignancy_median=median,
                    is_malignant=median >= 4.0,
                    n_readers=len(scores),
                )
            )
    return labels


def patch_pylidc() -> None:
    """Make pylidc 0.2.x importable on Python 3.12 + numpy 2.x.

    pylidc is unmaintained and still calls `configparser.SafeConfigParser` (removed in 3.12)
    and the numpy aliases removed in numpy 2. Both are source bugs in pylidc, not data
    problems, so we patch rather than pin the whole stack backwards.
    """
    import configparser

    import numpy as np

    configparser.SafeConfigParser = configparser.ConfigParser        # Scan.py path resolution
    for alias, target in (("int", np.int_), ("float", np.float64), ("bool", np.bool_)):
        if not hasattr(np, alias):                                    # Contour/Annotation/utils
            setattr(np, alias, target)


def _scan_geometry(scan):
    """SimpleITK image geometry for a pylidc scan, or None if the series won't load."""
    import SimpleITK as sitk

    try:
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(
            scan.get_path_to_dicom_files(), scan.series_instance_uid
        )
        if not files:
            return None
        reader.SetFileNames(files)
        return reader.Execute()
    except Exception:  # noqa: BLE001
        return None


def _centroid_to_world(centroid_ijk, geom) -> tuple[float, float, float]:
    """pylidc (i, j, k) = (row, col, slice) -> world mm (x, y, z).

    pylidc indexes `scan.to_volume()` as [row, col, slice] while SimpleITK indexes
    (col, row, slice) - hence the i/j swap. The slice axis is NOT reversed: validated
    against LUNA16's independent world coordinates, forward ordering matches to a median
    of 0.29 mm while reversed ordering is off by >100 mm.
    """
    i, j, k = (float(v) for v in centroid_ijk)
    return tuple(float(v) for v in geom.TransformContinuousIndexToPhysicalPoint((j, i, k)))
