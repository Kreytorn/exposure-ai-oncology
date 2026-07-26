"""Segmenter construction + the MedSAM2 cross-environment bridge.

This lives in the library rather than in a script because BOTH callers of the imaging
plane need it: the deterministic driver (`scripts/run_pipeline.py`) and the LLM
orchestration agent (`oncoct.agent.tools`). Two copies would drift, and a drifting
segmenter is exactly the kind of silent difference this project cannot afford — the
agent must call the SAME tools the deterministic pipeline calls, not lookalikes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from oncoct.segment import LesionPrompt


def medsam2_launcher() -> list[str]:
    """Argv prefix that runs a script inside the isolated medsam2 environment.

    Two ways to isolate MedSAM2's torch pin. `conda run -n medsam2` is the documented one,
    but a plain virtualenv works identically and is what environments without conda (e.g.
    Colab) actually have. Set ONCOCT_MEDSAM2_PYTHON to that interpreter to use it.
    """
    py = os.environ.get("ONCOCT_MEDSAM2_PYTHON")
    if py:
        return [py]
    return ["conda", "run", "-n", "medsam2", "python"]


def medsam2_worker_path() -> Path:
    """Absolute path to scripts/medsam2_worker.py.

    The worker is a script, not an installed module, so it cannot be imported — it has to
    be located on disk. Resolving it relative to this file assumes the editable/src layout
    the project actually uses (`pip install -e .`); ONCOCT_MEDSAM2_WORKER overrides that for
    any layout where it doesn't hold. Raises rather than shelling out to a missing path, so
    the failure names the real problem instead of surfacing as a subprocess exit code.
    """
    env = os.environ.get("ONCOCT_MEDSAM2_WORKER")
    repo_root = Path(__file__).resolve().parents[3]
    worker = Path(env) if env else repo_root / "scripts" / "medsam2_worker.py"
    if not worker.exists():
        raise FileNotFoundError(
            f"MedSAM2 worker not found at {worker}. Set ONCOCT_MEDSAM2_WORKER to "
            "scripts/medsam2_worker.py if oncoct is installed outside the repo tree."
        )
    return worker


class MedSAM2SubprocessClient:
    """In-`oncoct`-env stand-in for MedSAM2, which can't be imported here (torch/CUDA pin).

    Implements the shared Segmenter interface but runs the actual model in the isolated
    `medsam2` env by shelling out to scripts/medsam2_worker.py. Serializes the volume +
    prompt to disk (all arrays (z,y,x)), invokes the worker, reads the mask back — this IS
    the cross-env glue so the caller still just does `segmenter.segment(volume, prompt)`.
    """

    def __init__(self, checkpoint: str, config: str, workdir: Path, hu_window=(-1000, 400)):
        self.checkpoint = checkpoint
        self.config = config
        self.workdir = workdir
        self.hu_window = hu_window

    def segment(self, volume_zyx: np.ndarray, prompt: LesionPrompt) -> np.ndarray:
        import json
        import subprocess

        self.workdir.mkdir(parents=True, exist_ok=True)
        vol_p = self.workdir / "volume.npz"
        prm_p = self.workdir / "prompt.json"
        out_p = self.workdir / "mask.npz"
        # Every lesion in a study shares one volume, so serialize it once per scan rather
        # than re-writing ~300 MB per lesion. Uncompressed: this is a scratch handoff file,
        # and zlib on 300 MB costs more than the disk write it saves.
        fingerprint = (volume_zyx.shape, float(volume_zyx[0, 0, 0]), float(volume_zyx.sum()))
        if getattr(self, "_volume_fingerprint", None) != fingerprint or not vol_p.exists():
            np.savez(vol_p, volume=volume_zyx)
            self._volume_fingerprint = fingerprint
        prm_p.write_text(
            json.dumps(
                {
                    "key_slice_z": prompt.key_slice_z,
                    "box_xyxy": list(prompt.box_xyxy),
                    "hu_window": list(self.hu_window),
                }
            )
        )
        subprocess.run(
            [
                *medsam2_launcher(),
                str(medsam2_worker_path()),
                "--volume",
                str(vol_p),
                "--prompt",
                str(prm_p),
                "--out",
                str(out_p),
                "--checkpoint",
                self.checkpoint,
                "--config",
                self.config,
            ],
            check=True,
        )
        return np.load(out_p)["mask"]


def build_segmenter(config: dict, workdir: Path):
    """Return a Segmenter. Both branches expose segment(volume, LesionPrompt) -> mask.

    NOTE: MedSAM2 cannot run in-process in the `oncoct` env (it needs the `medsam2` env),
    so the medsam2 backend returns a subprocess client, NOT the in-env MedSAM2Segmenter
    class. The VISTA3D backend runs in-process (Apache, no pin) and is the simplest default.
    """
    backend = config["segment"]["backend"]
    if backend == "medsam2":
        m = config["segment"]["medsam2"]
        # The checkpoint and the Hydra config-name both live in the CLONED MedSAM2 repo, not
        # under oncoct/. ONCOCT_MEDSAM2_ROOT points at that clone so the config can stay a
        # bare filename and remain portable across machines.
        ckpt = Path(m["checkpoint"])
        if not ckpt.is_absolute():
            root = os.environ.get("ONCOCT_MEDSAM2_ROOT")
            ckpt = Path(root) / "checkpoints" / ckpt.name if root else ckpt
        return MedSAM2SubprocessClient(
            checkpoint=str(ckpt),
            config=m["config"],
            workdir=workdir / "medsam2",
            hu_window=tuple(config["preprocess"]["hu_window"]),
        )
    from oncoct.segment.vista3d_fallback import Vista3DSegmenter

    return Vista3DSegmenter(bundle=config["segment"]["vista3d"]["bundle"])
