"""Typed tool definitions the orchestrator agent can call.

Each tool wraps one deterministic imaging-plane stage, takes/returns JSON-serializable
data, and records its output in the trace under a unique ``tool_call_id`` so the report
can cite it. Tools are the ONLY way a number enters the pipeline output.

The `TOOL_SCHEMAS` list is what gets handed to the LLM as its tool set (Anthropic
tool-use format). Implementations dispatch into the imaging-plane modules.

Two design rules make the traceability invariant hold at runtime rather than by convention:

1. **The agent never carries a number between tools.** Tools address each other through
   opaque handles (`study_uid`, `detection_id`, `lesion_id`); measurements stay inside the
   executor's state. The model can reorder or skip stages, but it cannot retype a value and
   it cannot smuggle one in through a tool argument.
2. **`assemble_report` builds the report from the trace, not from the model's tokens.** The
   agent supplies only prose (the impression) and which lesions to include; every numeric
   field is filled in from the recorded tool outputs, tagged with the id of the call that
   produced it. That is what `verify_traceability` then re-checks.

Heavy imaging dependencies (monai, SimpleITK, TotalSegmentator, torch) are imported inside
the handlers, not at module scope, so the agent layer stays importable — and unit-testable —
in an environment that has none of them installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TOOL_SCHEMAS = [
    {
        "name": "ingest",
        "description": (
            "Convert/validate a study to a preprocessed CT volume (HU preserved, resampled). "
            "Must be called first; every other tool needs the study_uid it returns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"study_path": {"type": "string"}},
            "required": ["study_path"],
        },
    },
    {
        "name": "organ_context",
        "description": (
            "Compute (and cache) the whole-body organ label map for the study. Required "
            "before attribute_organ."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"study_uid": {"type": "string"}},
            "required": ["study_uid"],
        },
    },
    {
        "name": "detect_nodules",
        "description": (
            "Run the pretrained nodule detector; return candidate boxes + scores, each with a "
            "detection_id to pass to segment_lesion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"study_uid": {"type": "string"}, "score_threshold": {"type": "number"}},
            "required": ["study_uid"],
        },
    },
    {
        "name": "segment_lesion",
        "description": (
            "Turn a detection box into a precise 3D lesion mask (box on key slice). Returns a "
            "lesion_id used by measure, attribute_organ and classify_malignancy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"study_uid": {"type": "string"}, "detection_id": {"type": "string"}},
            "required": ["study_uid", "detection_id"],
        },
    },
    {
        "name": "measure",
        "description": "Compute RECIST 1.1 long/short axis, volume, centroid from a lesion mask.",
        "input_schema": {
            "type": "object",
            "properties": {"lesion_id": {"type": "string"}},
            "required": ["lesion_id"],
        },
    },
    {
        "name": "attribute_organ",
        "description": "Attribute a lesion to an organ/lobe via overlap with the cached organ map.",
        "input_schema": {
            "type": "object",
            "properties": {"lesion_id": {"type": "string"}},
            "required": ["lesion_id"],
        },
    },
    {
        "name": "classify_malignancy",
        "description": "Score a lesion as malignant vs benign (returns score + confidence).",
        "input_schema": {
            "type": "object",
            "properties": {"lesion_id": {"type": "string"}},
            "required": ["lesion_id"],
        },
    },
    {
        "name": "assemble_report",
        "description": (
            "Finalize the study. Provide the impression prose and the lesions to include; every "
            "number is filled in from this session's tool outputs and tagged with the call that "
            "produced it. Each lesion must already have measure, attribute_organ and "
            "classify_malignancy results. Call this exactly once, last. Do NOT write any "
            "measurement into the impression that a tool did not return."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "study_uid": {"type": "string"},
                "lesion_ids": {"type": "array", "items": {"type": "string"}},
                "impression": {"type": "string"},
            },
            "required": ["study_uid", "lesion_ids", "impression"],
        },
    },
]


class ToolError(RuntimeError):
    """A tool was called out of order or with an unknown handle.

    Raised (rather than returned) so the orchestrator can surface it to the model as an
    `is_error` tool_result — the agent is expected to read the message and recover, e.g. by
    calling `ingest` before `detect_nodules`.
    """


class ToolExecutor:
    """Dispatches tool calls to imaging-plane implementations and records the trace.

    ``trace`` maps tool_call_id -> {"tool": name, "input": ..., "output": ...}. The
    traceability guard reads this to verify every reported number is sourced.
    """

    def __init__(
        self,
        config: dict | None = None,
        cache_dir: Path | None = None,
        out_dir: Path | None = None,
    ):
        # `config` stays optional so a bare ToolExecutor() is still a usable trace holder —
        # the traceability guard and its tests only ever read `.trace`, and requiring a full
        # pipeline config to construct one would couple that check to the imaging config.
        self.config = config if config is not None else {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/cache")
        self.out_dir = Path(out_dir) if out_dir else Path("results")
        self.trace: dict[str, dict] = {}
        self.report = None  # set by assemble_report

        # Per-study state. The agent addresses these through handles; it never sees the arrays.
        self.study_uid: str | None = None
        self._volume = None  # numpy (z, y, x), HU preserved
        self._spacing_xyz: tuple[float, float, float] | None = None
        self._resampled_path: Path | None = None
        self._organ_map = None
        self._label_to_name: dict[int, str] = {}
        self._detections: dict[str, Any] = {}
        self._lesions: dict[str, dict] = {}
        self._segmenter = None
        self._classifier = None

    # -- dispatch -------------------------------------------------------------------------

    def call(self, tool_name: str, tool_call_id: str, **kwargs) -> dict:
        """Run one tool, record it in the trace under `tool_call_id`, return its output."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            raise ToolError(
                f"Unknown tool {tool_name!r}. Available: "
                + ", ".join(s["name"] for s in TOOL_SCHEMAS)
            )
        output = handler(tool_call_id=tool_call_id, **kwargs)
        self.trace[tool_call_id] = {"tool": tool_name, "input": kwargs, "output": output}
        return output

    def _require_study(self, study_uid: str | None = None):
        if self.study_uid is None:
            raise ToolError("No study loaded — call ingest(study_path) first.")
        if study_uid is not None and study_uid != self.study_uid:
            raise ToolError(
                f"Unknown study_uid {study_uid!r}; this session holds {self.study_uid!r}."
            )

    def _require_lesion(self, lesion_id: str) -> dict:
        lesion = self._lesions.get(lesion_id)
        if lesion is None:
            known = ", ".join(self._lesions) or "none yet"
            raise ToolError(f"Unknown lesion_id {lesion_id!r}. Segmented lesions: {known}.")
        return lesion

    # -- tools ----------------------------------------------------------------------------

    def _tool_ingest(self, tool_call_id: str, study_path: str) -> dict:
        import SimpleITK as sitk

        from oncoct.io.dicom_nifti import assert_hounsfield_units, read_mhd
        from oncoct.io.resample import resample_to_spacing

        path = Path(study_path)
        if not path.exists():
            raise ToolError(f"Study not found: {path}")
        spacing = tuple(self.config["preprocess"]["target_spacing_mm"])
        img = resample_to_spacing(read_mhd(path), spacing, is_label=False)
        volume = sitk.GetArrayFromImage(img)
        assert_hounsfield_units(volume)  # never silently accept non-HU input

        self.study_uid = path.stem
        self._volume = volume
        self._spacing_xyz = spacing
        # Persist the RESAMPLED volume: every grid-coupled stage downstream (organ map,
        # segmentation) must operate on this grid, not on the raw one.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._resampled_path = self.cache_dir / f"{self.study_uid}_resampled.nii.gz"
        sitk.WriteImage(img, str(self._resampled_path))
        return {
            "study_uid": self.study_uid,
            "shape_zyx": [int(n) for n in volume.shape],
            "spacing_xyz_mm": [float(s) for s in spacing],
            "hu_range": [float(volume.min()), float(volume.max())],
        }

    def _tool_organ_context(self, tool_call_id: str, study_uid: str) -> dict:
        from oncoct.context.organ_map import compute_organ_map, label_to_name

        self._require_study(study_uid)
        self._organ_map = compute_organ_map(self._resampled_path, self.cache_dir, fast=False)
        if self._organ_map.shape != self._volume.shape:
            raise ToolError(
                f"Organ map {self._organ_map.shape} does not match volume {self._volume.shape}; "
                "it must be computed on the resampled grid."
            )
        self._label_to_name = label_to_name()
        present = sorted(
            {
                self._label_to_name.get(int(v), f"label_{int(v)}")
                for v in set(self._organ_map.ravel().tolist())
                if v != 0
            }
        )
        return {"study_uid": study_uid, "n_structures": len(present), "structures_present": present}

    def _tool_detect_nodules(
        self, tool_call_id: str, study_uid: str, score_threshold: float | None = None
    ) -> dict:
        from oncoct.detect.nodule_detector import detect_nodules

        self._require_study(study_uid)
        thr = float(
            score_threshold
            if score_threshold is not None
            else self.config["detect"].get("score_threshold", 0.5)
        )
        detections = detect_nodules(self._volume, self._spacing_xyz, score_threshold=thr)
        # Segmentation costs ~10 s/lesion, so a pathological scan with 100 candidates would
        # stall the study. Detections are score-sorted, so this keeps the most confident ones.
        cap = self.config["detect"].get("max_lesions_per_study", 10)
        capped = bool(cap and len(detections) > cap)
        if capped:
            detections = detections[:cap]

        self._detections = {}
        listed = []
        for i, det in enumerate(detections):
            did = f"D{i + 1}"
            self._detections[did] = det
            listed.append(
                {
                    "detection_id": did,
                    "score": float(det.score),
                    "center_zyx": [float(c) for c in det.center_zyx],
                }
            )
        return {
            "study_uid": study_uid,
            "score_threshold": thr,
            "n_candidates": len(listed),
            "capped_to_max_lesions_per_study": capped,
            "detections": listed,
        }

    def _tool_segment_lesion(self, tool_call_id: str, study_uid: str, detection_id: str) -> dict:
        from oncoct.segment import LesionPrompt
        from oncoct.segment.factory import build_segmenter

        self._require_study(study_uid)
        det = self._detections.get(detection_id)
        if det is None:
            known = ", ".join(self._detections) or "none — call detect_nodules first"
            raise ToolError(f"Unknown detection_id {detection_id!r}. Detections: {known}.")
        if self._segmenter is None:
            self._segmenter = build_segmenter(self.config, self.cache_dir)

        mask = self._segmenter.segment(self._volume, LesionPrompt.from_detection(det))
        n_vox = int((mask > 0).sum())
        if n_vox == 0:
            # An empty mask is a real outcome, not a crash: the detection had no segmentable
            # lesion. Report it so the agent drops the candidate instead of measuring nothing.
            return {"detection_id": detection_id, "lesion_id": None, "n_voxels": 0, "empty": True}

        lesion_id = f"L{len(self._lesions) + 1}"
        self._lesions[lesion_id] = {
            "detection_id": detection_id,
            "mask": mask,
            "detector_score": float(det.score),
        }
        return {
            "detection_id": detection_id,
            "lesion_id": lesion_id,
            "n_voxels": n_vox,
            "empty": False,
        }

    def _tool_measure(self, tool_call_id: str, lesion_id: str) -> dict:
        from oncoct.measure.recist import measure_lesion

        lesion = self._require_lesion(lesion_id)
        m = measure_lesion(lesion["mask"], self._spacing_xyz)
        lesion["measurement"] = m
        lesion["measure_source"] = tool_call_id  # what the report will cite
        return {
            "lesion_id": lesion_id,
            "long_axis_mm": round(float(m.long_axis_mm), 2),
            "short_axis_mm": round(float(m.short_axis_mm), 2),
            "volume_mm3": round(float(m.volume_mm3), 2),
            "key_slice_z": int(m.key_slice_z),
        }

    def _tool_attribute_organ(self, tool_call_id: str, lesion_id: str) -> dict:
        from oncoct.context.organ_map import attribute_organ, split_organ_lobe

        lesion = self._require_lesion(lesion_id)
        if self._organ_map is None:
            raise ToolError("No organ map — call organ_context(study_uid) first.")
        organ, fractions = attribute_organ(lesion["mask"], self._organ_map, self._label_to_name)
        organ_name, lobe = split_organ_lobe(organ)
        lesion["organ"] = organ_name
        lesion["lobe_or_segment"] = lobe
        lesion["organ_fractions"] = fractions
        lesion["organ_attributed"] = organ
        return {
            "lesion_id": lesion_id,
            "organ": organ_name,
            "lobe_or_segment": lobe,
            "overlap_fractions": {k: round(float(v), 3) for k, v in fractions.items()},
        }

    def _tool_classify_malignancy(self, tool_call_id: str, lesion_id: str) -> dict:
        from oncoct.classify.malignancy import MalignancyClassifier, crop_patch

        lesion = self._require_lesion(lesion_id)
        if "measurement" not in lesion:
            raise ToolError(
                f"Call measure({lesion_id!r}) first — the patch is cropped at its centroid."
            )
        if self._classifier is None:
            self._classifier = MalignancyClassifier(
                backend=self.config["classify"]["backend"],
                weights_path=str(self.out_dir / "weights" / "malignancy_head.pt"),
            )
        patch = crop_patch(self._volume, lesion["measurement"].centroid_zyx, size=48)
        result = self._classifier.predict(patch)
        lesion["malignancy"] = result
        lesion["classify_source"] = tool_call_id  # what the report will cite
        return {
            "lesion_id": lesion_id,
            "malignancy_score": round(float(result.score), 4),
            "malignancy_confidence": round(float(result.confidence), 4),
            # Stated on every call so the agent cannot narrate this as a reliability estimate.
            "confidence_note": "sharpness (1 - normalized entropy), NOT calibrated reliability",
        }

    def _tool_assemble_report(
        self, tool_call_id: str, study_uid: str, lesion_ids: list[str], impression: str
    ) -> dict:
        from oncoct.report.generator import build_study_report
        from oncoct.report.quality import quality_flags

        self._require_study(study_uid)
        if not lesion_ids:
            raise ToolError(
                "lesion_ids is empty — a study with no lesions still needs an explicit list."
            )

        records = []
        for lid in lesion_ids:
            lesion = self._require_lesion(lid)
            missing = [
                stage
                for stage, key in (
                    ("measure", "measurement"),
                    ("classify_malignancy", "malignancy"),
                )
                if key not in lesion
            ]
            if "organ" not in lesion:
                missing.append("attribute_organ")
            if missing:
                raise ToolError(f"{lid} is missing tool results from: {', '.join(missing)}.")

            m, mal = lesion["measurement"], lesion["malignancy"]
            records.append(
                {
                    "lesion_id": lid,
                    "organ": lesion["organ"],
                    "lobe_or_segment": lesion["lobe_or_segment"],
                    "long_axis_mm": float(m.long_axis_mm),
                    "long_axis_source": lesion["measure_source"],
                    "short_axis_mm": float(m.short_axis_mm),
                    "short_axis_source": lesion["measure_source"],
                    "volume_mm3": float(m.volume_mm3),
                    "volume_source": lesion["measure_source"],
                    "malignancy_score": float(mal.score),
                    "malignancy_source": lesion["classify_source"],
                    "malignancy_confidence": float(mal.confidence),
                    "detector_score": lesion["detector_score"],
                    "quality_flags": quality_flags(
                        lesion["mask"],
                        m,
                        lesion.get("organ_fractions", {}),
                        lesion.get("organ_attributed", lesion["organ"]),
                        self._spacing_xyz,
                    ),
                }
            )

        report = build_study_report(study_uid, records)
        report.impression = impression  # the one field the agent authors
        self.report = report
        return {
            "study_uid": study_uid,
            "n_findings": len(report.findings),
            "n_recist_targets": sum(1 for f in report.findings if f.recist_category == "target"),
            "status": "report assembled; every numeric field is tagged with its tool_call id",
        }
