"""MedSAM2 subprocess worker — runs INSIDE the `medsam2` conda env.

The primary `oncoct` env cannot import `sam2` (hard torch/CUDA version pin), so it shells
out to this worker. Cross-env contract (all arrays in numpy (z, y, x) order to avoid the
axis-flip bug at the seam):

    INPUT   volume.npz   -> key "volume": float array (z, y, x), HU preserved
            prompt.json  -> {"key_slice_z": int,
                             "box_xyxy": [x_min, y_min, x_max, y_max],  # original px on key slice
                             "hu_window": [lo, hi]}
    OUTPUT  mask.npz     -> key "mask": uint8 array (z, y, x), same grid as input volume

Invoked by the oncoct env as:
    conda run -n medsam2 python scripts/medsam2_worker.py \
        --volume /tmp/volume.npz --prompt /tmp/prompt.json --out /tmp/mask.npz \
        --checkpoint <MedSAM2_clone>/checkpoints/MedSAM2_CTLesion.pt \
        --config configs/sam2.1_hiera_t512.yaml
The config is a Hydra config-NAME resolved within the MedSAM2 package, not an oncoct path.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_IMAGE_SIZE = 512

# Propagation is limited to a slab centred on the key slice. A lung nodule is at most a few
# cm, so tracking to both ends of a 300-slice chest CT only buys propagation drift and
# wasted compute. 48 slices at 1.25 mm = 60 mm each way — far beyond any lung nodule.
DEFAULT_Z_MARGIN = 48


def _preprocess(volume_zyx: np.ndarray, hu_window) -> np.ndarray:
    """HU volume -> (D, 3, 512, 512) float32, ImageNet-normalized.

    Mirrors MedSAM2's own medsam2_infer_3D_CT.py exactly: clip to the window, min-max to
    0-255 uint8 over the CLIPPED volume, grayscale->RGB via PIL, resize to 512, /255, then
    ImageNet normalize. Any deviation here silently shifts the input distribution away from
    what the CTLesion checkpoint was trained on.
    """
    from PIL import Image

    lo, hi = float(hu_window[0]), float(hu_window[1])
    vol = np.clip(volume_zyx, lo, hi)
    vmin, vmax = float(vol.min()), float(vol.max())
    denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
    vol = np.uint8((vol - vmin) / denom * 255.0)

    d = vol.shape[0]
    out = np.zeros((d, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.float32)
    for i in range(d):
        rgb = Image.fromarray(vol[i]).convert("RGB").resize(
            (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        )
        out[i] = np.array(rgb, dtype=np.float32).transpose(2, 0, 1)
    out /= 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (out - mean) / std


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest 3D blob — drops propagation spill into neighbouring structures."""
    from skimage import measure

    if mask.max() == 0:
        return mask
    labels = measure.label(mask)
    counts = np.bincount(labels.flat)[1:]
    if counts.size == 0:
        return mask
    return (labels == (int(np.argmax(counts)) + 1)).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--z-margin", type=int, default=DEFAULT_Z_MARGIN)
    args = ap.parse_args()

    import torch
    from sam2.build_sam import build_sam2_video_predictor_npz

    volume = np.load(args.volume)["volume"]            # (z, y, x)
    prompt = json.loads(open(args.prompt).read())      # noqa: SIM115

    key_z = int(prompt["key_slice_z"])
    x0, y0, x1, y1 = (float(v) for v in prompt["box_xyxy"])
    hu_window = prompt.get("hu_window", (-1000, 400))

    nz, height, width = volume.shape
    key_z = int(np.clip(key_z, 0, nz - 1))

    # Slab around the key slice; the box's frame index is relative to the slab.
    z_lo = max(0, key_z - args.z_margin)
    z_hi = min(nz, key_z + args.z_margin + 1)
    slab = volume[z_lo:z_hi]
    key_idx = key_z - z_lo

    images = torch.from_numpy(_preprocess(slab, hu_window))
    if torch.cuda.is_available():
        images = images.cuda()

    # The box stays in ORIGINAL pixel coords. init_state is told the true video_height/width,
    # and add_new_points_or_box divides by those then multiplies by image_size internally
    # (sam2_video_predictor_npz.py). Pre-scaling the box to 512 here would scale it TWICE.
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    predictor = build_sam2_video_predictor_npz(args.config, args.checkpoint)
    seg = np.zeros(slab.shape, dtype=np.uint8)

    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
        state = predictor.init_state(images, height, width)
        # The box sits on ONE key slice; propagate_in_video only walks one way per call, so
        # the reverse pass is mandatory — without it the lesion is segmented only on the
        # superior side of the prompt (brief §7.3).
        for reverse in (False, True):
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=key_idx, obj_id=1, box=box
            )
            for frame_idx, _obj_ids, logits in predictor.propagate_in_video(
                state, reverse=reverse
            ):
                seg[frame_idx, (logits[0] > 0.0).cpu().numpy()[0]] = 1
            predictor.reset_state(state)

    seg = _largest_connected_component(seg)

    # Back onto the full input grid — the caller measures at the volume's spacing, so the
    # mask must share the volume's shape exactly (brief §7.7).
    mask = np.zeros(volume.shape, dtype=np.uint8)
    mask[z_lo:z_hi] = seg
    np.savez_compressed(args.out, mask=mask)
    print(
        f"[medsam2_worker] key_z={key_z} slab=[{z_lo},{z_hi}) "
        f"voxels={int(mask.sum())} slices={int((mask.sum(axis=(1, 2)) > 0).sum())}"
    )


if __name__ == "__main__":
    main()
