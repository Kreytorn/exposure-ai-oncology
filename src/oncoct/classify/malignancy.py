"""Malignancy classification: benign vs malignant nodule.

Two backends (config-selected):
  - cnn_head: freeze a 3D CNN encoder, train a small head on LIDC malignancy labels.
    This is the ONLY model we fine-tune (hours on an A100, frozen encoder).
  - radiomics_gbm: pyradiomics features + gradient-boosted trees - cheap, interpretable
    baseline that runs on a weak GPU/CPU.

Labels come from LIDC-IDRI (see labels/lidc_malignancy.py), aggregated by median of the
4 readers with median==3 dropped as ambiguous - a policy that biases the usable set and
must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------------------
# Patch specification. Training and inference MUST agree on every value here or the scores
# are meaningless (brief §5.6), so both sides import these constants rather than restating
# them. The window is the DETECTION BUNDLE's window, not the pipeline's [-1000, 400]:
# the frozen encoder below is that bundle's backbone, so its inputs must match what it saw.
PATCH_SIZE = 48                       # voxels, cubic, at TARGET_SPACING
TARGET_SPACING_XYZ = (0.703125, 0.703125, 1.25)
ENCODER_HU_WINDOW = (-1024.0, 300.0)
FEATURE_DIM = 1024                    # 512 channels, avg- and max-pooled, concatenated
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MalignancyResult:
    score: float          # probability of malignant, 0..1
    confidence: float     # calibrated confidence / entropy-derived


def crop_patch(volume_zyx: np.ndarray, centroid_zyx, size: int = 48) -> np.ndarray:
    """Center-crop a size^3 patch around the lesion centroid (pad if near an edge)."""
    cz, cy, cx = (int(round(c)) for c in centroid_zyx)
    h = size // 2
    padded = np.pad(volume_zyx, h, mode="constant", constant_values=volume_zyx.min())
    cz, cy, cx = cz + h, cy + h, cx + h
    return padded[cz - h : cz + h, cy - h : cy + h, cx - h : cx + h]


def normalize_patch(patch_zyx: np.ndarray) -> np.ndarray:
    """HU patch -> [0, 1] using the encoder's own intensity window. Shared by train+infer."""
    lo, hi = ENCODER_HU_WINDOW
    return ((np.clip(np.asarray(patch_zyx, dtype=np.float32), lo, hi) - lo) / (hi - lo)).astype(
        np.float32
    )


def build_encoder(bundle_dir, device: str = "cuda"):
    """Frozen 3D CNN encoder: the detection bundle's LUNA16-pretrained ResNet-50 trunk.

    Reusing the detector's backbone means the encoder is already tuned on lung nodules at
    exactly our spacing - far better than an ImageNet-ish 3D model, and it needs no extra
    download. NOTE: the bundle's FPN keeps only `returned_layers=[1, 2]`, so the checkpoint
    contains conv1/bn1/layer1/layer2 and nothing deeper. We therefore truncate the trunk at
    layer2; going deeper would silently mix in randomly-initialised layer3/layer4 weights.
    """
    from monai.networks.nets.resnet import resnet50

    net = resnet50(
        spatial_dims=3, n_input_channels=1, conv1_t_stride=[2, 2, 1], conv1_t_size=[7, 7, 7]
    )
    state = torch.load(Path(bundle_dir) / "models" / "model.pt", map_location="cpu",
                       weights_only=True)
    prefix = "feature_extractor.body."
    body = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not body:
        raise RuntimeError(f"No '{prefix}*' weights in the detection checkpoint at {bundle_dir}")
    net.load_state_dict(body, strict=False)

    trunk = torch.nn.Sequential(
        net.conv1, net.bn1, net.act, net.maxpool, net.layer1, net.layer2
    ).to(device).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


def encode(trunk, patches_zyx, device: str = "cuda", batch_size: int = 32):
    """(N, 48, 48, 48) HU patches -> (N, FEATURE_DIM) frozen features."""
    arr = np.asarray(patches_zyx, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    feats = []
    with torch.no_grad():
        for i in range(0, len(arr), batch_size):
            x = torch.from_numpy(normalize_patch(arr[i:i + batch_size]))[:, None].to(device)
            h = trunk(x)
            # avg captures the general texture, max the focal bright/dense component -
            # concatenating both is a standard, cheap way to keep focal signal a mean loses.
            feats.append(
                torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], dim=1).float().cpu()
            )
    return torch.cat(feats).numpy()


class MalignancyHead(torch.nn.Module):
    """The only trained parameters in the pipeline: a small MLP over frozen features."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_dim),
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _flip_views(patch_zyx: np.ndarray):
    """The 8 axis-flip views of a patch. Nodule malignancy is flip-invariant, so these are
    label-preserving - used to augment training and to average at inference (TTA)."""
    for fz in (False, True):
        for fy in (False, True):
            for fx in (False, True):
                v = patch_zyx
                if fz:
                    v = v[::-1]
                if fy:
                    v = v[:, ::-1]
                if fx:
                    v = v[:, :, ::-1]
                yield np.ascontiguousarray(v)


class MalignancyClassifier:
    def __init__(self, backend: str = "cnn_head", weights_path: str | None = None,
                 bundle_dir: str | None = None, device: str | None = None, tta: bool = True):
        self.backend = backend
        self.weights_path = weights_path
        self.bundle_dir = bundle_dir
        self.device = device
        self.tta = tta
        self._trunk = None
        self._head = None

    def _lazy_load(self):
        if self._head is not None:
            return
        if self.backend != "cnn_head":
            raise NotImplementedError(
                f"backend {self.backend!r} is not implemented; this round ships 'cnn_head' "
                "(frozen detection-bundle encoder + trained MLP head)."
            )
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        from oncoct.detect.nodule_detector import default_bundle_dir

        bundle = self.bundle_dir or default_bundle_dir()
        self._trunk = build_encoder(bundle, self.device)

        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        head = MalignancyHead(
            in_dim=ckpt.get("in_dim", FEATURE_DIM), hidden=ckpt.get("hidden", 32)
        ).to(self.device)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        self._head = head
        self._patch_spec = ckpt.get("patch_spec", {})

    def predict(self, patch_zyx: np.ndarray) -> MalignancyResult:
        """Score a lesion patch (cropped around the nodule) as malignant vs benign."""
        self._lazy_load()
        patch = np.asarray(patch_zyx, dtype=np.float32)
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Expected a {PATCH_SIZE}^3 patch at {TARGET_SPACING_XYZ} mm spacing, got "
                f"{patch.shape}. The crop spec must match training exactly (brief §5.6)."
            )

        views = list(_flip_views(patch)) if self.tta else [patch]
        feats = encode(self._trunk, np.stack(views), self.device)
        with torch.no_grad():
            logits = self._head(torch.from_numpy(feats).to(self.device))
            score = float(torch.sigmoid(logits).mean().item())

        # Confidence = 1 - normalized binary entropy: 0 at p=0.5 (model abstaining), 1 at a
        # saturated call. It is a sharpness measure, NOT a calibrated reliability estimate -
        # with n=196 training nodules there is not enough data to calibrate honestly.
        p = min(max(score, 1e-6), 1 - 1e-6)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)
        return MalignancyResult(score=score, confidence=float(1.0 - entropy))
