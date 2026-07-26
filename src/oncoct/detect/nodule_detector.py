"""Lung nodule detection via the pretrained MONAI RetinaNet bundle.

Uses the model-zoo `lung_nodule_ct_detection` bundle (pretrained on LUNA16). Detection,
not segmentation, is the correct paradigm for nodules and for the LUNA16 FROC metric.
Input must be resampled to the bundle's expected spacing (0.703x0.703x1.25mm); a spacing
mismatch silently degrades detection.

Honesty note: the bundle is trained on LUNA16 fold 0 only; its reported COCO mAP (0.852)
is NOT the LUNA16 leaderboard FROC number. Report them separately (see eval/froc_luna16.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The bundle's own intensity window (configs/inference.json::ScaleIntensityRanged). This is
# NOT the pipeline's hu_window [-1000, 400] - that one belongs to MedSAM2/the classifier.
# Feeding the detector a different window silently degrades it, so it is pinned here.
_DETECTOR_HU_WINDOW = (-1024.0, 300.0)
_BUNDLE_SPACING_XYZ = (0.703125, 0.703125, 1.25)

# The bundle trains and infers in RAS (configs/inference.json applies Orientationd(RAS)).
# Our resampled volumes carry an identity direction in ITK's LPS frame, so reaching RAS is
# exactly a flip of the x and y index axes. Held as a constant so the box inverse below
# cannot drift out of sync with the image transform.
_RAS_FLIP_AXES = (0, 1)

_DETECTOR_CACHE: dict[tuple[str, str], object] = {}


@dataclass(frozen=True)
class Detection:
    """One detected nodule candidate in numpy (z, y, x) voxel space."""

    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    center_zyx: tuple[float, float, float]
    score: float


def default_bundle_dir() -> Path:
    """Where the `lung_nodule_ct_detection` bundle lives. Override with ONCOCT_BUNDLE_DIR."""
    env = os.environ.get("ONCOCT_BUNDLE_DIR")
    if env:
        return Path(env)
    return Path("artifacts/weights/lung_nodule_ct_detection")


def _build_detector(bundle_dir: Path, device: str):
    """Build the RetinaNet detector from the bundle's OWN inference.json.

    Parsing the shipped config (rather than re-declaring the architecture here) keeps the
    backbone, anchor shapes, box-selector thresholds and sliding-window settings exactly as
    published - a hand-transcribed anchor shape would degrade detection silently.
    """
    import torch
    from monai.bundle import ConfigParser

    key = (str(bundle_dir), device)
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]

    bundle_dir = Path(bundle_dir)
    cfg = bundle_dir / "configs" / "inference.json"
    if not cfg.exists():
        raise FileNotFoundError(
            f"Detection bundle not found at {bundle_dir}. Fetch it with:\n"
            '  python -m monai.bundle download "lung_nodule_ct_detection" '
            "--bundle_dir <weights_dir>/"
        )

    parser = ConfigParser()
    parser.read_config(str(cfg))
    parser["bundle_root"] = str(bundle_dir)
    parser["device"] = f"$torch.device('{device}')"

    detector = parser.get_parsed_content("detector")
    # Applies set_target_keys, the box-selector thresholds, and the sliding-window inferer.
    parser.get_parsed_content("detector_ops")

    state = torch.load(bundle_dir / "models" / "model.pt", map_location="cpu", weights_only=True)
    detector.network.load_state_dict(state)
    detector.to(torch.device(device))
    detector.eval()

    _DETECTOR_CACHE[key] = detector
    return detector


def _to_network_input(volume_zyx: np.ndarray) -> np.ndarray:
    """(z, y, x) HU on an LPS identity grid -> (x, y, z) intensity-scaled RAS array."""
    arr = np.ascontiguousarray(np.asarray(volume_zyx).transpose(2, 1, 0), dtype=np.float32)
    for ax in _RAS_FLIP_AXES:
        arr = np.flip(arr, axis=ax)
    lo, hi = _DETECTOR_HU_WINDOW
    arr = (np.clip(arr, lo, hi) - lo) / (hi - lo)          # ScaleIntensityRanged, clip=True
    return np.ascontiguousarray(arr, dtype=np.float32)


def _box_ras_to_numpy_zyx(
    box_xyzxyz: np.ndarray, shape_zyx: tuple[int, int, int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Invert `_to_network_input` for one box: RAS (x0,y0,z0,x1,y1,z1) -> numpy bbox_zyx.

    A flipped axis reverses the interval, so the low and high edges swap: an index i maps
    to (N - 1 - i), hence [x0, x1] -> [X-1-x1, X-1-x0]. Getting this backwards puts every
    lesion in the mirror-image lung, which is exactly the silent failure the brief warns of.
    """
    nz, ny, nx = shape_zyx
    x0, y0, z0, x1, y1, z1 = (float(v) for v in box_xyzxyz)

    x0, x1 = (nx - 1) - x1, (nx - 1) - x0      # x axis was flipped
    y0, y1 = (ny - 1) - y1, (ny - 1) - y0      # y axis was flipped
    # z was not flipped.

    def _clamp(lo: float, hi: float, n: int) -> tuple[int, int]:
        a = int(np.clip(np.floor(lo), 0, n - 1))
        b = int(np.clip(np.ceil(hi), 0, n - 1))
        return (a, b) if b >= a else (b, a)

    return (_clamp(z0, z1, nz), _clamp(y0, y1, ny), _clamp(x0, x1, nx))


def detect_nodules(
    volume_zyx,
    spacing_xyz,
    score_threshold: float = 0.5,
    bundle_dir: str | Path | None = None,
    device: str | None = None,
) -> list[Detection]:
    """Run the pretrained detector and return candidates above ``score_threshold``.

    Args:
        volume_zyx: preprocessed CT volume, numpy (z, y, x), HU preserved.
        spacing_xyz: voxel spacing (x, y, z); must match the bundle's expectation.
    """
    import torch

    volume_zyx = np.asarray(volume_zyx)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if not np.allclose(spacing, _BUNDLE_SPACING_XYZ, atol=1e-3):
        raise ValueError(
            f"Detector expects spacing {_BUNDLE_SPACING_XYZ} mm (x, y, z), got "
            f"{tuple(spacing)}. Resample first - a spacing mismatch degrades detection "
            "silently rather than erroring (see brief §7.4)."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = _build_detector(Path(bundle_dir or default_bundle_dir()), device)

    net_in = _to_network_input(volume_zyx)
    tensor = torch.from_numpy(net_in)[None, None].to(device)   # (B=1, C=1, X, Y, Z)

    with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], enabled=device != "cpu"):
        outputs = detector(tensor, use_inferer=True)

    pred = outputs[0]
    boxes = pred[detector.target_box_key].detach().float().cpu().numpy()      # (N, 6) xyzxyz
    scores = pred[detector.pred_score_key].detach().float().cpu().numpy()     # (N,)

    shape_zyx = tuple(int(v) for v in volume_zyx.shape)
    detections: list[Detection] = []
    for box, score in zip(boxes, scores, strict=True):
        if float(score) < score_threshold:
            continue
        bbox = _box_ras_to_numpy_zyx(box, shape_zyx)
        center = tuple(float(lo + hi) / 2.0 for lo, hi in bbox)
        detections.append(Detection(bbox_zyx=bbox, center_zyx=center, score=float(score)))

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections
