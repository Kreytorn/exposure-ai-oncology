"""Shape translation between the Anthropic Messages interface and Gemini's REST API.

The orchestrator is provider-agnostic - it only needs `.content` blocks and a `.stop_reason`.
These tests pin the translation in both directions against payloads captured from the live
v1beta API, so a provider swap cannot silently change what the loop sees. No network.
"""

import io
import json
from unittest import mock

import pytest

from oncoct.agent.gemini_client import (
    Block,
    GeminiClient,
    _to_function_declarations,
    _wrap_result,
)
from oncoct.agent.tools import TOOL_SCHEMAS


class _FakeHTTP:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _client() -> GeminiClient:
    return GeminiClient(api_key="test-key")


# --- request translation ------------------------------------------------------------------


def test_tool_schemas_translate_to_function_declarations():
    decls = _to_function_declarations(TOOL_SCHEMAS)
    assert len(decls) == len(TOOL_SCHEMAS)
    by_name = {d["name"]: d for d in decls}
    assert "assemble_report" in by_name
    # input_schema -> parameters, passed through unchanged (it is inside Gemini's subset).
    assert by_name["measure"]["parameters"] == {
        "type": "object",
        "properties": {"lesion_id": {"type": "string"}},
        "required": ["lesion_id"],
    }
    assert by_name["ingest"]["description"]


def test_user_string_becomes_a_text_part():
    contents = _client()._to_contents([{"role": "user", "content": "hello"}])
    assert contents == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_assistant_tool_use_becomes_a_function_call():
    c = _client()
    contents = c._to_contents(
        [{"role": "assistant",
          "content": [Block(type="tool_use", id="abc", name="measure", input={"lesion_id": "L1"})]}]
    )
    assert contents == [
        {"role": "model",
         "parts": [{"functionCall": {"id": "abc", "name": "measure", "args": {"lesion_id": "L1"}}}]}
    ]


def test_tool_result_is_routed_back_by_name_not_id():
    """Gemini keys a functionResponse by tool NAME; Anthropic keys tool_result by id."""
    c = _client()
    c._to_contents(
        [{"role": "assistant",
          "content": [Block(type="tool_use", id="xyz", name="measure", input={})]}]
    )  # teaches the adapter that xyz -> measure
    contents = c._to_contents(
        [{"role": "user",
          "content": [{"type": "tool_result", "tool_use_id": "xyz",
                       "content": json.dumps({"long_axis_mm": 8.4})}]}]
    )
    fr = contents[0]["parts"][0]["functionResponse"]
    assert fr["name"] == "measure"
    assert fr["response"] == {"long_axis_mm": 8.4}


def test_unknown_tool_result_id_does_not_crash():
    contents = _client()._to_contents(
        [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "never-seen",
                                       "content": "{}"}]}]
    )
    assert contents[0]["parts"][0]["functionResponse"]["name"] == "unknown_tool"


@pytest.mark.parametrize(
    "content,expected",
    [
        (json.dumps({"a": 1}), {"a": 1}),
        ("ToolError: boom", {"result": "ToolError: boom"}),   # error strings are not JSON
        (json.dumps([1, 2]), {"result": [1, 2]}),             # response must be an object
        ({"already": "dict"}, {"already": "dict"}),
    ],
)
def test_tool_result_payload_is_always_an_object(content, expected):
    assert _wrap_result(content) == expected


# --- response translation -----------------------------------------------------------------


def _payload(parts, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"role": "model", "parts": parts}}]}


def test_function_call_becomes_a_tool_use_block():
    r = _client()._to_response(
        _payload([{"functionCall": {"id": "sApsicfs", "name": "ingest",
                                    "args": {"study_path": "scan.mhd"}}}])
    )
    assert r.stop_reason == "tool_use"
    assert len(r.content) == 1
    block = r.content[0]
    assert (block.type, block.id, block.name) == ("tool_use", "sApsicfs", "ingest")
    assert block.input == {"study_path": "scan.mhd"}


def test_text_only_turn_is_end_turn():
    r = _client()._to_response(_payload([{"text": "All done."}]))
    assert r.stop_reason == "end_turn"
    assert r.content[0].type == "text"


def test_missing_call_id_is_synthesized():
    """The trace and the report's provenance are keyed on the id - it cannot be None."""
    r = _client()._to_response(_payload([{"functionCall": {"name": "measure", "args": {}}}]))
    assert r.content[0].id
    assert r.content[0].type == "tool_use"


@pytest.mark.parametrize(
    "finish,expected",
    [("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal"), ("PROHIBITED_CONTENT", "refusal")],
)
def test_finish_reasons_map_to_orchestrator_stop_reasons(finish, expected):
    assert _client()._to_response(_payload([{"text": "x"}], finish=finish)).stop_reason == expected


def test_blocked_prompt_with_no_candidate_is_a_refusal():
    r = _client()._to_response({"promptFeedback": {"blockReason": "SAFETY"}})
    assert r.stop_reason == "refusal"
    # The reason is carried through: an empty refusal is undiagnosable from a notebook.
    assert "SAFETY" in r.content[0].text


def test_non_gemini_model_name_does_not_reach_the_api():
    """A shared config carries a Claude model id; sending it to Gemini would 404."""
    c = _client()
    seen = {}

    def fake_post(model, body):
        seen["model"] = model
        return _payload([{"text": "ok"}])

    c._post = fake_post
    c.messages.create(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    assert seen["model"] == "gemini-flash-latest"
    c.messages.create(model="gemini-2.5-pro", messages=[{"role": "user", "content": "hi"}])
    assert seen["model"] == "gemini-2.5-pro"


@pytest.mark.parametrize(
    "body,expected",
    [
        ('{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo",'
         '"retryDelay":"37s"}]}}', 37.0),
        ('{"error":{"code":429}}', 0.0),          # no RetryInfo -> fall back to our backoff
        ("not json", 0.0),
    ],
)
def test_retry_delay_is_taken_from_the_server_when_offered(body, expected):
    """Round 4 lost the agent job to a free-tier limit; blind backoff can undershoot the
    per-minute window, so the server's own RetryInfo wins when it is present."""
    from oncoct.agent.gemini_client import _retry_after_seconds

    assert _retry_after_seconds(body) == expected


_DAILY_429 = json.dumps({"error": {"code": 429, "message": "You exceeded your current quota, "
    + "x" * 300, "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    "quotaValue": "20"}]}]}})
_MINUTE_429 = json.dumps({"error": {"code": 429, "message": "rate", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}})


def test_daily_cap_is_distinguished_from_a_per_minute_limit():
    """A per-minute limit is worth sleeping through; a per-day cap is not.

    The full body must be parsed: the quota id lives in `details`, after a ~300-char message.
    Truncating first (as an early version did) hides it and sends a daily cap down the retry
    path, where no amount of waiting can help - which is what broke Round 4's agent job.
    """
    from oncoct.agent.gemini_client import _is_daily_quota

    assert _is_daily_quota(_DAILY_429) is True
    assert _is_daily_quota(_MINUTE_429) is False
    assert _is_daily_quota("not json") is False


def test_exhausted_model_rolls_over_to_the_next_one():
    """Free-tier quota is per MODEL, so a different model is a fresh budget, not a retry."""
    import urllib.error

    c = GeminiClient(api_key="k", model="gemini-flash-latest")
    tried = []

    def fake_urlopen(req, timeout=None):
        model = req.full_url.split("/models/")[1].split(":")[0]
        tried.append(model)
        if model == "gemini-flash-latest":
            raise urllib.error.HTTPError(
                req.full_url, 429, "quota", {}, io.BytesIO(_DAILY_429.encode())
            )
        return _FakeHTTP(json.dumps(_payload([{"text": "ok"}])).encode())

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        r = c.messages.create(messages=[{"role": "user", "content": "hi"}])

    assert r.stop_reason == "end_turn"
    assert tried[0] == "gemini-flash-latest"          # tried the configured model first
    assert tried[1] == "gemini-flash-lite-latest"     # then the next budget
    assert "gemini-flash-latest" in c._exhausted      # and took it out of rotation


def test_all_models_exhausted_says_so_plainly():
    import urllib.error

    c = GeminiClient(api_key="k")

    def always_daily(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 429, "quota", {}, io.BytesIO(_DAILY_429.encode())
        )

    with mock.patch("urllib.request.urlopen", always_daily), pytest.raises(
        Exception, match="out of free-tier quota for today"
    ):
        c.messages.create(messages=[{"role": "user", "content": "hi"}])


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(Exception, match="No Gemini API key"):
        GeminiClient()


# --- the whole loop, on a fake Gemini transport --------------------------------------------


def test_orchestrator_drives_a_full_study_through_the_gemini_adapter():
    """End-to-end through the real adapter: only the HTTP call is faked."""
    from test_agent_orchestration import CONFIG, FakeExecutor

    from oncoct.agent.orchestrator import run

    calls = [
        ("ingest", {"study_path": "scan.mhd"}),
        ("organ_context", {"study_uid": "1.2.3"}),
        ("detect_nodules", {"study_uid": "1.2.3"}),
        ("segment_lesion", {"study_uid": "1.2.3", "detection_id": "D1"}),
        ("measure", {"lesion_id": "L1"}),
        ("attribute_organ", {"lesion_id": "L1"}),
        ("classify_malignancy", {"lesion_id": "L1"}),
        ("assemble_report", {"study_uid": "1.2.3", "lesion_ids": ["L1"],
                             "impression": "Single left upper lobe nodule."}),
    ]
    client = _client()
    turns = iter(
        [_payload([{"functionCall": {"id": f"g{i}", "name": n, "args": a}}])
         for i, (n, a) in enumerate(calls)]
        + [_payload([{"text": "Report assembled."}])]
    )
    client._post = lambda model, body: next(turns)

    executor = FakeExecutor()
    report = run("scan.mhd", {**CONFIG, "report": {"llm": "gemini"}},
                 client=client, executor=executor)

    assert len(report.findings) == 1
    assert report.findings[0].long_axis_mm.source in executor.trace
    # Provenance survives the provider swap: the id Gemini issued is the id the report cites.
    assert report.findings[0].long_axis_mm.source == "g4"
    assert executor.trace["g4"]["tool"] == "measure"
