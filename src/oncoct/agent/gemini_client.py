"""Gemini adapter presenting the Anthropic Messages interface the orchestrator expects.

`orchestrator.run()` takes an injectable `client` — the seam that lets the tests drive the
loop with a scripted model. The same seam makes the provider swappable: anything exposing
`.messages.create(model=, max_tokens=, system=, tools=, messages=)` and returning an object
with `.content` (blocks) and `.stop_reason` can drive the pipeline. Nothing in the loop, the
tools, or the traceability guard is provider-specific, so this file is the entire cost of
running the agent on Gemini instead of Claude.

Deliberately built on `urllib` rather than `google-genai`: the imaging environment already
juggles a numpy/torch pin against MedSAM2's, and the one thing this adapter must not do is
add another package that can conflict on Colab. The REST surface it targets is small and
was verified live before this was written.

Shape translation (verified against the live v1beta API):

    Anthropic                          Gemini
    system="..."                    -> systemInstruction.parts[0].text
    tools[].input_schema            -> tools[0].functionDeclarations[].parameters
    tool_use  {id, name, input}     -> functionCall {id, name, args}
    tool_result {tool_use_id, ...}  -> functionResponse {name, response}
    stop_reason                     <- finishReason (+ whether a functionCall is present)

The one asymmetry worth knowing: a Gemini `functionResponse` is keyed by tool NAME, while an
Anthropic `tool_result` is keyed by tool_use_id. The adapter therefore remembers id -> name
for every call it has seen this session, so results can be routed back correctly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
# Pro-tier models are commonly out of free-tier quota; flash-latest is the model this was
# verified against. Override per call or via ONCOCT_GEMINI_MODEL.
DEFAULT_MODEL = "gemini-flash-latest"

# Free-tier quota is `GenerateRequestsPerDayPerProjectPerModel` — 20 requests per day, and
# critically it is scoped PER MODEL. One study is ~9 round trips, so two studies exhaust a
# model for the rest of the day; that is exactly how Round 4's agent job died. Rolling over
# to the next model is therefore a real recovery and not a retry dressed up as one: backoff
# cannot help a daily cap, but a different model has its own untouched budget.
FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


@dataclass
class Block:
    """One content block, shaped like the Anthropic blocks the orchestrator reads.

    `raw` keeps the original Gemini part so a model turn can be replayed byte-for-byte.
    That is not an optimization: Gemini rejects a replayed `functionCall` that has lost its
    `thoughtSignature` ("Function call is missing a thought_signature", HTTP 400), and
    reconstructing parts from the fields we happen to model would drop any such metadata —
    including fields added after this was written.
    """

    type: str
    id: str | None = None
    name: str | None = None
    input: dict = field(default_factory=dict)
    text: str = ""
    raw: dict | None = None


@dataclass
class Response:
    content: list[Block]
    stop_reason: str
    raw: dict = field(default_factory=dict)


class GeminiError(RuntimeError):
    pass


class _DailyQuotaExhausted(RuntimeError):
    """This model's per-day free-tier budget is gone. Waiting will not help; switch models."""


def _is_daily_quota(error_body: str) -> bool:
    """True when a 429 is the per-DAY cap rather than a per-minute rate limit.

    The distinction decides the response: a per-minute limit is worth sleeping through, a
    per-day cap is not (Google still returns a RetryInfo of ~55s for it, which is misleading).
    """
    try:
        details = json.loads(error_body)["error"].get("details", [])
    except (TypeError, ValueError, KeyError):
        return False
    for detail in details:
        if detail.get("@type", "").endswith("QuotaFailure"):
            for violation in detail.get("violations", []):
                if "PerDay" in str(violation.get("quotaId", "")):
                    return True
    return False


def _to_function_declarations(tools: list[dict]) -> list[dict]:
    """Anthropic tool schemas -> Gemini functionDeclarations.

    Gemini accepts an OpenAPI subset. Our schemas only use type/properties/required/items/
    description, which is inside it, so the parameters object passes through unchanged.
    """
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


class _Messages:
    def __init__(self, client: GeminiClient):
        self._client = client

    def create(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        system: str = "",
        tools: list[dict] | None = None,
        messages: list[dict],
        **_ignored,
    ) -> Response:
        body: dict = {
            "contents": self._client._to_contents(messages),
            "generationConfig": {"maxOutputTokens": int(max_tokens)},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": _to_function_declarations(tools)}]

        # The orchestrator passes whatever `agent.model` says, which in a shared config is
        # usually a Claude id. Sending that to Gemini yields a baffling 404, so ignore any
        # model name that clearly is not Gemini's and use this client's own.
        wanted = model if (model and "gemini" in model.lower()) else self._client.model
        payload = self._client._post(wanted, body)
        return self._client._to_response(payload)


class GeminiClient:
    """Drop-in stand-in for `anthropic.Anthropic()` covering what the orchestrator uses."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 5,
        timeout: int = 180,
        fallback_models: list[str] | None = None,
    ):
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise GeminiError(
                "No Gemini API key. Pass api_key= or set GEMINI_API_KEY in the environment."
            )
        self.model = os.environ.get("ONCOCT_GEMINI_MODEL", model)
        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback_models = list(
            fallback_models if fallback_models is not None else FALLBACK_MODELS
        )
        self.model_in_use = self.model  # which model actually served the last turn
        self._exhausted: set[str] = set()  # models whose daily budget ran out this session
        self._id_to_name: dict[str, str] = {}  # tool_use_id -> tool name, for functionResponse
        self.messages = _Messages(self)

    # -- transport -------------------------------------------------------------------------

    def _post(self, model: str, body: dict) -> dict:
        """POST one turn, retrying transient errors and rolling over exhausted models."""
        candidates = [model] + [m for m in self.fallback_models if m != model]
        last = ""
        for candidate in candidates:
            if candidate in self._exhausted:
                continue
            try:
                return self._post_one(candidate, body)
            except _DailyQuotaExhausted as e:
                # Not retryable in any amount of time short of tomorrow — take the model out
                # of rotation for this session and try the next one's separate budget.
                self._exhausted.add(candidate)
                last = str(e)
                print(f"[gemini] {candidate} is out of daily free-tier quota; trying next model")
                continue
        raise GeminiError(
            "All Gemini models are out of free-tier quota for today "
            f"({', '.join(candidates)}). The cap is ~20 requests/day per model and one study "
            f"is ~9 round trips. Use a paid key or wait for the daily reset. Last: {last}"
        )

    def _post_one(self, model: str, body: dict) -> dict:
        url = f"{API_ROOT}/{model}:generateContent?key={self.api_key}"
        data = json.dumps(body).encode()
        delay = 2.0
        last = ""
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.model_in_use = model
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                # Parse the FULL body — the quota id lives in `details`, which sits after a
                # ~250-char message. Truncating before parsing silently hides it and sends a
                # per-day cap down the retry path, where no amount of waiting can help.
                body_text = e.read().decode()
                last = body_text[:400]
                if e.code == 429 and _is_daily_quota(body_text):
                    raise _DailyQuotaExhausted(f"{model}: {last[:200]}") from e
                # Per-minute limits and 5xx ARE worth waiting out. Prefer the server's own
                # RetryInfo delay: blind exponential backoff can undershoot the window.
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(max(delay, _retry_after_seconds(body_text)))
                    delay *= 2
                    continue
                raise GeminiError(f"Gemini HTTP {e.code}: {last}") from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise GeminiError(f"Gemini connection error: {e}") from e
        raise GeminiError(f"Gemini failed after {self.max_retries} attempts: {last}")

    # -- request translation ---------------------------------------------------------------

    def _to_contents(self, messages: list[dict]) -> list[dict]:
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            if isinstance(content, str):
                contents.append({"role": role, "parts": [{"text": content}]})
                continue

            parts = []
            for block in content:
                btype = block.type if isinstance(block, Block) else block.get("type")
                # Replay a model turn exactly as it arrived. Anything we rebuild by hand
                # loses metadata Gemini requires back (thoughtSignature) — see Block.raw.
                if role == "model" and isinstance(block, Block) and block.raw is not None:
                    parts.append(block.raw)
                    if block.type == "tool_use" and block.id:
                        self._id_to_name[block.id] = block.name
                elif btype == "tool_use":
                    bid = block.id if isinstance(block, Block) else block["id"]
                    bname = block.name if isinstance(block, Block) else block["name"]
                    binput = block.input if isinstance(block, Block) else block["input"]
                    self._id_to_name[bid] = bname
                    parts.append({"functionCall": {"id": bid, "name": bname, "args": binput}})
                elif btype == "tool_result":
                    tool_use_id = block["tool_use_id"]
                    parts.append(
                        {
                            "functionResponse": {
                                "name": self._id_to_name.get(tool_use_id, "unknown_tool"),
                                "response": _wrap_result(block.get("content", "")),
                            }
                        }
                    )
                elif btype == "text":
                    text = block.text if isinstance(block, Block) else block.get("text", "")
                    if text:
                        parts.append({"text": text})
            if parts:
                contents.append({"role": role, "parts": parts})
        return contents

    # -- response translation --------------------------------------------------------------

    def _to_response(self, payload: dict) -> Response:
        candidates = payload.get("candidates") or []
        if not candidates:
            # No candidate at all means the prompt itself was blocked upstream. Carry the
            # reason in a text block: an empty refusal with no explanation is the hardest
            # kind of failure to diagnose from a notebook.
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "unknown")
            return Response(
                content=[Block(type="text", text=f"blocked by provider: {reason}")],
                stop_reason="refusal",
                raw=payload,
            )

        candidate = candidates[0]
        finish = candidate.get("finishReason", "STOP")
        blocks: list[Block] = []
        for i, part in enumerate((candidate.get("content") or {}).get("parts") or []):
            if "functionCall" in part:
                call = part["functionCall"]
                # Gemini usually supplies an id; synthesize a stable one when it does not,
                # because the trace and the report's provenance are keyed on it.
                call_id = call.get("id") or f"gemini_{len(self._id_to_name)}_{i}"
                self._id_to_name[call_id] = call["name"]
                blocks.append(
                    Block(
                        type="tool_use",
                        id=call_id,
                        name=call["name"],
                        input=dict(call.get("args") or {}),
                        raw=part,
                    )
                )
            elif "text" in part:
                blocks.append(Block(type="text", text=part["text"], raw=part))
            else:
                # Unrecognized part kind (e.g. a thought part). Keep it so the turn can be
                # replayed intact; the orchestrator ignores blocks whose type it not know.
                blocks.append(Block(type="other", raw=part))

        has_tool_use = any(b.type == "tool_use" for b in blocks)
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            stop = "refusal"
        elif finish == "MAX_TOKENS":
            stop = "max_tokens"
        elif has_tool_use:
            stop = "tool_use"
        else:
            stop = "end_turn"
        return Response(content=blocks, stop_reason=stop, raw=payload)


def _retry_after_seconds(error_body: str) -> float:
    """Seconds Google asks us to wait, from a 429's RetryInfo detail. 0 if unstated."""
    try:
        details = json.loads(error_body)["error"].get("details", [])
    except (TypeError, ValueError, KeyError):
        return 0.0
    for detail in details:
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(raw)
            except ValueError:
                return 0.0
    return 0.0


def _wrap_result(content) -> dict:
    """A Gemini functionResponse.response must be an object; ours arrives as a JSON string."""
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return {"result": str(content)}
    return parsed if isinstance(parsed, dict) else {"result": parsed}
