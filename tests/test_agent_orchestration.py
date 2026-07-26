"""The live orchestration loop, and the invariant applied to the path the agent actually takes.

`test_agent_no_unsourced_numbers.py` pins the contract on a hand-built report. These tests
pin it on the real loop: a scripted model drives `ToolExecutor`, the report is assembled from
the recorded trace, and `verify_traceability` runs on the result - including the impression,
which is the one part of the report the model writes in its own words and therefore the one
place a number can enter without passing through a tool.

No API key, no GPU, no imaging stack: the model is a scripted stub and the imaging stages are
faked, so this runs anywhere `pytest` does.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from oncoct.agent.orchestrator import run, verify_traceability
from oncoct.agent.tools import ToolError, ToolExecutor
from oncoct.measure.recist import measure_lesion

CONFIG = {
    "preprocess": {"target_spacing_mm": [0.703125, 0.703125, 1.25], "hu_window": [-1000, 400]},
    "detect": {"score_threshold": 0.5, "max_lesions_per_study": 10},
    "segment": {"backend": "vista3d", "vista3d": {"bundle": "vista3d"}},
    "classify": {"backend": "cnn_head"},
    "report": {"llm": "claude"},
    "agent": {"model": "test-model", "prompt_path": "configs/agent_prompt.md", "max_turns": 20},
}


# --- scripted model ---------------------------------------------------------------------


@dataclass
class _ToolUse:
    name: str
    input: dict
    id: str
    type: str = "tool_use"


@dataclass
class _Response:
    content: list
    stop_reason: str


class ScriptedClient:
    """Replays a fixed list of assistant turns, recording what it was sent."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.sent = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.sent.append(kwargs)
        if not self._turns:
            return _Response(content=[], stop_reason="end_turn")
        return self._turns.pop(0)


# --- fake imaging plane -----------------------------------------------------------------


@dataclass(frozen=True)
class _FakeMalignancy:
    score: float
    confidence: float


def _blob(n_slices: int = 6, radius: int = 6, shape=(48, 64, 64)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    disc = ((yy - shape[1] // 2) ** 2 + (xx - shape[2] // 2) ** 2) <= radius**2
    z0 = (shape[0] - n_slices) // 2
    mask[z0 : z0 + n_slices] = disc
    return mask


class FakeExecutor(ToolExecutor):
    """Real dispatch, trace and report assembly; imaging stages replaced by fixtures.

    Only the four stages that need monai/SimpleITK/torch are overridden. `measure`,
    `attribute_organ` and `assemble_report` run the REAL implementations, so the wiring
    under test - that each number reaches the report tagged with the id of the call that
    produced it - is the production wiring.
    """

    def __init__(self, config=CONFIG, malignancy=(0.73, 0.66)):
        super().__init__(config)
        self._malignancy = malignancy

    def _tool_ingest(self, tool_call_id, study_path):
        self.study_uid = "1.2.3"
        self._volume = np.full((48, 64, 64), -800.0)
        self._spacing_xyz = tuple(self.config["preprocess"]["target_spacing_mm"])
        return {"study_uid": self.study_uid, "shape_zyx": [48, 64, 64]}

    def _tool_organ_context(self, tool_call_id, study_uid):
        self._require_study(study_uid)
        self._organ_map = np.full((48, 64, 64), 13, dtype=np.int16)
        self._label_to_name = {13: "lung_upper_lobe_left"}
        return {"study_uid": study_uid, "n_structures": 1}

    def _tool_detect_nodules(self, tool_call_id, study_uid, score_threshold=None):
        self._require_study(study_uid)
        self._detections = {"D1": object()}
        return {"study_uid": study_uid, "n_candidates": 1, "detections": [{"detection_id": "D1"}]}

    def _tool_segment_lesion(self, tool_call_id, study_uid, detection_id):
        self._require_study(study_uid)
        if detection_id not in self._detections:
            raise ToolError(f"Unknown detection_id {detection_id!r}.")
        lesion_id = f"L{len(self._lesions) + 1}"
        self._lesions[lesion_id] = {
            "detection_id": detection_id,
            "mask": _blob(),
            "detector_score": 0.91,
        }
        return {"detection_id": detection_id, "lesion_id": lesion_id, "empty": False}

    def _tool_classify_malignancy(self, tool_call_id, lesion_id):
        lesion = self._require_lesion(lesion_id)
        if "measurement" not in lesion:
            raise ToolError(f"Call measure({lesion_id!r}) first.")
        score, confidence = self._malignancy
        lesion["malignancy"] = _FakeMalignancy(score=score, confidence=confidence)
        lesion["classify_source"] = tool_call_id
        return {"lesion_id": lesion_id, "malignancy_score": score}


def _happy_path_turns(impression: str):
    """The canonical call sequence a well-behaved agent makes on a one-lesion study."""
    calls = [
        ("ingest", {"study_path": "scan.mhd"}),
        ("organ_context", {"study_uid": "1.2.3"}),
        ("detect_nodules", {"study_uid": "1.2.3"}),
        ("segment_lesion", {"study_uid": "1.2.3", "detection_id": "D1"}),
        ("measure", {"lesion_id": "L1"}),
        ("attribute_organ", {"lesion_id": "L1"}),
        ("classify_malignancy", {"lesion_id": "L1"}),
        (
            "assemble_report",
            {"study_uid": "1.2.3", "lesion_ids": ["L1"], "impression": impression},
        ),
    ]
    turns = [
        _Response(content=[_ToolUse(name=n, input=i, id=f"toolu_{k}")], stop_reason="tool_use")
        for k, (n, i) in enumerate(calls)
    ]
    turns.append(_Response(content=[], stop_reason="end_turn"))
    return turns


# --- the loop ---------------------------------------------------------------------------


def test_loop_produces_a_fully_sourced_report():
    executor = FakeExecutor()
    report = run(
        "scan.mhd",
        CONFIG,
        client=ScriptedClient(_happy_path_turns("Single left upper lobe nodule, indeterminate.")),
        executor=executor,
    )

    assert report.study_uid == "1.2.3"
    assert len(report.findings) == 1
    lesion = report.findings[0]
    assert lesion.organ == "lung"
    assert lesion.lobe_or_segment == "upper lobe, left"
    # Every numeric field cites a real tool call, and the measure/classify ids are distinct
    # calls - not one blanket source stamped over everything.
    assert lesion.long_axis_mm.source in executor.trace
    assert lesion.malignancy_score.source in executor.trace
    assert lesion.long_axis_mm.source != lesion.malignancy_score.source
    assert executor.trace[lesion.long_axis_mm.source]["tool"] == "measure"
    assert executor.trace[lesion.malignancy_score.source]["tool"] == "classify_malignancy"


def test_measurements_come_from_the_tool_not_the_model():
    """The report's numbers equal what measure_lesion computes, to the digit."""
    executor = FakeExecutor()
    report = run(
        "scan.mhd",
        CONFIG,
        client=ScriptedClient(_happy_path_turns("Nodule present.")),
        executor=executor,
    )
    expected = measure_lesion(_blob(), spacing_xyz=tuple(CONFIG["preprocess"]["target_spacing_mm"]))
    assert report.findings[0].long_axis_mm.value == pytest.approx(expected.long_axis_mm)
    assert report.findings[0].volume_mm3.value == pytest.approx(expected.volume_mm3)


def test_tool_results_are_returned_in_one_user_message_per_turn():
    """Splitting results across messages trains the model out of parallel tool calls."""
    client = ScriptedClient(_happy_path_turns("Nodule present."))
    run("scan.mhd", CONFIG, client=client, executor=FakeExecutor())
    for request in client.sent:
        results = [
            m for m in request["messages"] if m["role"] == "user" and isinstance(m["content"], list)
        ]
        for message in results:
            assert all(b["type"] == "tool_result" for b in message["content"])


# --- the invariant, on the path the agent actually takes --------------------------------


def test_hallucinated_number_in_impression_is_rejected():
    turns = _happy_path_turns("Dominant lesion measures 42.0 mm in long axis.")
    with pytest.raises(ValueError, match="Untraceable number in impression"):
        run("scan.mhd", CONFIG, client=ScriptedClient(turns), executor=FakeExecutor())


def test_rounded_restatement_of_a_real_measurement_is_accepted():
    executor = FakeExecutor()
    truth = measure_lesion(_blob(), spacing_xyz=tuple(CONFIG["preprocess"]["target_spacing_mm"]))
    rounded = f"{truth.long_axis_mm:.1f}"
    report = run(
        "scan.mhd",
        CONFIG,
        client=ScriptedClient(_happy_path_turns(f"Lesion measures {rounded} mm.")),
        executor=executor,
    )
    assert rounded in report.impression


def test_lesion_handles_are_not_read_as_measurements():
    """ "L1" must not trip the numeric guard - it is a handle, not a number."""
    report = run(
        "scan.mhd",
        CONFIG,
        client=ScriptedClient(_happy_path_turns("L1 is the only target, per RECIST 1.1.")),
        executor=FakeExecutor(),
    )
    assert "L1" in report.impression


def test_malignancy_score_may_be_narrated_as_a_percentage():
    report = run(
        "scan.mhd",
        CONFIG,
        client=ScriptedClient(_happy_path_turns("Malignancy probability 73%.")),
        executor=FakeExecutor(malignancy=(0.73, 0.66)),
    )
    assert "73%" in report.impression


def test_percentage_that_matches_no_score_is_rejected():
    turns = _happy_path_turns("Malignancy probability 15%.")
    with pytest.raises(ValueError, match="Untraceable number in impression"):
        run(
            "scan.mhd",
            CONFIG,
            client=ScriptedClient(turns),
            executor=FakeExecutor(malignancy=(0.73, 0.66)),
        )


def test_verify_traceability_still_rejects_an_unsourced_structured_field():
    """The original contract is unchanged by the narrative check being added alongside it."""
    executor = FakeExecutor()
    report = run(
        "scan.mhd",
        CONFIG,
        client=ScriptedClient(_happy_path_turns("Nodule present.")),
        executor=executor,
    )
    report.findings[0].malignancy_score.source = "toolu_fabricated"
    with pytest.raises(ValueError, match="Untraceable number in report"):
        verify_traceability(report, executor)


# --- failure modes ----------------------------------------------------------------------


def test_ending_without_assembling_is_a_hard_failure():
    """A partial narration is not a study report."""
    turns = [
        _Response(
            content=[_ToolUse(name="ingest", input={"study_path": "scan.mhd"}, id="toolu_0")],
            stop_reason="tool_use",
        ),
        _Response(content=[], stop_reason="end_turn"),
    ]
    with pytest.raises(RuntimeError, match="without calling assemble_report"):
        run("scan.mhd", CONFIG, client=ScriptedClient(turns), executor=FakeExecutor())


def test_tool_error_is_handed_back_to_the_model_not_raised():
    """A bad call order is recoverable - the agent must get the message and adapt."""
    bad_then_good = [
        _Response(
            content=[_ToolUse(name="measure", input={"lesion_id": "L9"}, id="toolu_bad")],
            stop_reason="tool_use",
        ),
        *_happy_path_turns("Nodule present."),
    ]
    client = ScriptedClient(bad_then_good)
    report = run("scan.mhd", CONFIG, client=client, executor=FakeExecutor())
    assert report is not None  # the run survived the bad call

    errors = [
        block
        for request in client.sent
        for message in request["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("is_error")
    ]
    assert errors, "the failing tool call was never reported back to the model"
    assert "Unknown lesion_id" in errors[0]["content"]


def test_refusal_is_surfaced_not_silently_swallowed():
    turns = [_Response(content=[], stop_reason="refusal")]
    with pytest.raises(RuntimeError, match="declined this request"):
        run("scan.mhd", CONFIG, client=ScriptedClient(turns), executor=FakeExecutor())


def test_truncated_turn_is_surfaced():
    turns = [_Response(content=[], stop_reason="max_tokens")]
    with pytest.raises(RuntimeError, match="max_tokens"):
        run("scan.mhd", CONFIG, client=ScriptedClient(turns), executor=FakeExecutor())


def test_runaway_agent_hits_the_turn_cap():
    looping = [
        _Response(
            content=[_ToolUse(name="ingest", input={"study_path": "scan.mhd"}, id=f"toolu_{i}")],
            stop_reason="tool_use",
        )
        for i in range(50)
    ]
    config = {**CONFIG, "agent": {**CONFIG["agent"], "max_turns": 5}}
    with pytest.raises(RuntimeError, match="did not finish within 5 turns"):
        run("scan.mhd", config, client=ScriptedClient(looping), executor=FakeExecutor(config))


def test_assemble_rejects_a_lesion_missing_a_stage():
    """Every lesion in the report must have been measured, attributed AND classified."""
    executor = FakeExecutor()
    executor.call("ingest", "t0", study_path="scan.mhd")
    executor.call("organ_context", "t1", study_uid="1.2.3")
    executor.call("detect_nodules", "t2", study_uid="1.2.3")
    executor.call("segment_lesion", "t3", study_uid="1.2.3", detection_id="D1")
    executor.call("measure", "t4", lesion_id="L1")
    with pytest.raises(ToolError, match="missing tool results from"):
        executor.call("assemble_report", "t5", study_uid="1.2.3", lesion_ids=["L1"], impression="x")


def test_unknown_tool_names_the_available_ones():
    with pytest.raises(ToolError, match="Unknown tool"):
        FakeExecutor().call("compute_diameter_yourself", "t0")
