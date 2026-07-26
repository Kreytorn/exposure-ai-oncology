"""End-to-end DETERMINISTIC pipeline driver (Round 1 - no live LLM agent).

Chains the imaging-plane stages in order and feeds the deterministic report assembler:

    ingest -> organ context -> detect -> (per lesion) segment -> measure -> attribute
           -> classify -> assemble StudyReport -> write JSON + text + overlay PNGs

This is the Round-1 stand-in for the LLM orchestrator: it calls the SAME tools the agent
will later call, in a fixed order, and produces the same StudyReport. The `*_source` ids
are stage+lesion tags (e.g. "measure_L1") so verify_traceability passes without a live
agent trace.

Fill in the stage calls marked `TODO(round1)` - the surrounding orchestration, the
record-dict shape for build_study_report, and the traceability tags are already correct.

Usage:
    python scripts/run_pipeline.py --series <path/to/scan.mhd> --config configs/pipeline.yaml \
        --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from oncoct.classify.malignancy import MalignancyClassifier, crop_patch
from oncoct.context.organ_map import (
    attribute_organ,
    compute_organ_map,
    label_to_name,
    split_organ_lobe,
)
from oncoct.detect.nodule_detector import detect_nodules
from oncoct.io.dicom_nifti import assert_hounsfield_units, read_mhd
from oncoct.io.resample import resample_to_spacing
from oncoct.measure.recist import measure_lesion
from oncoct.report.generator import build_study_report
from oncoct.report.quality import quality_flags
from oncoct.report.text import render_text_report
from oncoct.segment import LesionPrompt
from oncoct.segment.factory import build_segmenter


def run_one(series_path: Path, config: dict, out_dir: Path, cache_dir: Path | None = None) -> None:
    study_uid = series_path.stem
    spacing_xyz = tuple(config["preprocess"]["target_spacing_mm"])

    # 1. INGEST ---------------------------------------------------------------
    import SimpleITK as sitk

    img = read_mhd(series_path)  # SimpleITK image
    # Validate BEFORE resampling. resample_to_spacing pads outside-extent voxels with a
    # -1024.0 air constant, so a post-resample check inspects a volume this pipeline has
    # already widened, and normalized input would pass on the padding alone.
    assert_hounsfield_units(sitk.GetArrayFromImage(img))
    img = resample_to_spacing(img, spacing_xyz, is_label=False)  # to isotropic
    volume_zyx = sitk.GetArrayFromImage(img)  # numpy (z, y, x); HU preserved

    # Persist the RESAMPLED volume so every grid-coupled stage (organ map, segmentation)
    # operates on the SAME grid as detection/measurement. Passing the raw .mhd here would
    # put the organ map on a different grid than the lesion masks -> attribute_organ
    # (which does organ_map[mask]) would overlap mismatched grids -> wrong organ.
    # Resampled volumes and organ maps are hundreds of MB per scan. They are scratch, not
    # deliverables, so they default beside results/ but stay OUT of it (--cache to relocate).
    cache = Path(cache_dir) if cache_dir else out_dir.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    resampled_path = cache / f"{study_uid}_resampled.nii.gz"
    sitk.WriteImage(img, str(resampled_path))

    # 2. ORGAN CONTEXT --------------------------------------------------------
    # compute_organ_map MUST return an organ map on the SAME (resampled) grid as volume_zyx.
    organ_map = compute_organ_map(resampled_path, cache, fast=False)  # stub -> implement
    assert organ_map.shape == volume_zyx.shape, (
        f"organ map {organ_map.shape} != volume {volume_zyx.shape}: compute_organ_map must "
        "return on the resampled grid (see run_pipeline ingest)."
    )
    l2n = label_to_name()

    # 3. DETECT ---------------------------------------------------------------
    thr = config["detect"].get("score_threshold", 0.5)
    detections = detect_nodules(volume_zyx, spacing_xyz, score_threshold=thr)  # stub -> implement
    # Segmentation costs ~10 s/lesion, so a pathological scan with 100 candidates would
    # stall the study. Detections are score-sorted, so this keeps the most confident ones.
    max_lesions = config["detect"].get("max_lesions_per_study", 10)
    if max_lesions and len(detections) > max_lesions:
        print(f"[run_pipeline]   capping {len(detections)} detections at top {max_lesions}")
        detections = detections[:max_lesions]

    # 4. SEGMENT -> MEASURE -> ATTRIBUTE -> CLASSIFY (per lesion) -------------
    segmenter = build_segmenter(config, cache)
    classifier = MalignancyClassifier(
        backend=config["classify"]["backend"],
        weights_path=str(out_dir / "weights" / "malignancy_head.pt"),
    )

    records: list[dict] = []
    overlays: list[tuple[str, int, np.ndarray, np.ndarray]] = []  # (lesion_id, key_z, box, mask)
    for i, det in enumerate(detections):
        lid = f"L{i + 1}"
        prompt = LesionPrompt.from_detection(det, vista3d_class=None)
        mask = segmenter.segment(volume_zyx, prompt)  # stub -> implement
        if mask.sum() == 0:
            continue
        m = measure_lesion(mask, spacing_xyz)  # implemented
        organ, frac = attribute_organ(mask, organ_map, l2n)  # implemented
        patch = crop_patch(volume_zyx, m.centroid_zyx, size=48)  # helper below
        mal = classifier.predict(patch)  # stub -> implement
        organ_name, lobe = split_organ_lobe(organ)

        records.append(
            {
                "lesion_id": lid,
                "organ": organ_name,
                "lobe_or_segment": lobe,
                "long_axis_mm": m.long_axis_mm,
                "long_axis_source": f"measure_{lid}",
                "short_axis_mm": m.short_axis_mm,
                "short_axis_source": f"measure_{lid}",
                "volume_mm3": m.volume_mm3,
                "volume_source": f"measure_{lid}",
                "malignancy_score": mal.score,
                "malignancy_source": f"classify_{lid}",
                "malignancy_confidence": mal.confidence,
                "detector_score": det.score,
                "quality_flags": quality_flags(mask, m, frac, organ, spacing_xyz),
            }
        )
        overlays.append((lid, m.key_slice_z, np.array(det.bbox_zyx), mask))

    # 5. ASSEMBLE + WRITE -----------------------------------------------------
    report = build_study_report(study_uid, records)  # implemented
    _write_outputs(report, volume_zyx, overlays, study_uid, out_dir)


def _write_outputs(report, volume_zyx, overlays, study_uid, out_dir: Path) -> None:
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / f"{study_uid}.json").write_text(report.model_dump_json(indent=2))
    (out_dir / "reports" / f"{study_uid}.txt").write_text(render_text_report(report))
    for lid, key_z, bbox_zyx, mask in overlays:
        # bbox_zyx = ((z0,z1),(y0,y1),(x0,x1)) in resampled-voxel space; slice is the same grid.
        (y0, y1), (x0, x1) = bbox_zyx[1], bbox_zyx[2]
        _save_overlay(
            volume_zyx[key_z],
            mask[key_z],
            (x0, y0, x1, y1),
            out_dir / "overlays" / f"{study_uid}_{lid}.png",
        )


def _save_overlay(slice_2d: np.ndarray, mask_2d: np.ndarray, box_xyxy, path: Path) -> None:
    """Matplotlib overlay: CT slice (lung window) + detection box + mask contour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.clip(slice_2d, -1000, 400), cmap="gray")
    ax.contour(mask_2d, colors="r", linewidths=0.8)
    x0, y0, x1, y1 = box_xyxy
    ax.add_patch(
        mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1.0)
    )
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", required=True, help="path to a .mhd scan (or a dir of them)")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default=None, help="scratch dir for resampled volumes/organ maps")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out)
    series = Path(args.series)
    paths = sorted(series.glob("*.mhd")) if series.is_dir() else [series]
    for p in paths:
        print(f"[run_pipeline] {p.name}")
        run_one(p, config, out_dir, Path(args.cache) if args.cache else None)
    print(f"[run_pipeline] done -> {out_dir}")


if __name__ == "__main__":
    main()
