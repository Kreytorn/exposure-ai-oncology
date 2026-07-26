"""The tool-calling orchestration loop.

Drives an LLM (Claude by default; MedGemma fallback) through the diagnostic checklist in
configs/agent_prompt.md: the model plans, emits tool calls, reads results, and finally
narrates a StudyReport. The loop enforces two invariants:
  1. the model may only call tools in agent/tools.py (no free-form computation);
  2. every number in the final report must be present in the tool-call trace
     (checked by verify_traceability before the report is accepted).
"""

from __future__ import annotations

import re
from pathlib import Path

from oncoct.agent.tools import TOOL_SCHEMAS, ToolError, ToolExecutor
from oncoct.report.schema import StudyReport

# Anthropic tool-use loop defaults. Claude Opus 5 runs adaptive thinking by default, and
# max_tokens caps thinking + text together — hence the headroom.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_MAX_TURNS = 60

# Numeric literals in prose, ignoring anything glued to a word character (so lesion handles
# like "L1" and file stems do not read as measurements).
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
# Tokens that legitimately contain digits but are names, not measurements.
_NON_MEASUREMENT_TOKENS = re.compile(r"\bRECIST\s*1\.1\b|\b[LD]\d+\b", re.IGNORECASE)


def verify_traceability(report: StudyReport, executor: ToolExecutor) -> None:
    """Raise if any numeric field in the report cites a tool_call id not in the trace.

    This is the runtime twin of tests/test_agent_no_unsourced_numbers.py: the invariant
    that makes an autonomous report trustworthy. A ';'-joined source (RECIST sum) is
    split and each component checked.

    The structured fields carry their own provenance, so checking them is a set membership
    test. The **impression is free prose the model authored**, which is the one place a
    number could enter the report without passing through a tool — so every numeric literal
    in it must also match a value the tools actually returned.
    """
    known = set(executor.trace.keys())
    for src in report.all_numeric_sources():
        for cid in src.split(";"):
            if cid and cid not in known:
                raise ValueError(
                    f"Untraceable number in report: source '{cid}' has no matching tool call. "
                    "The agent may not state numbers it did not obtain from a tool."
                )
    _verify_narrative_numbers(report, executor)


def _verify_narrative_numbers(report: StudyReport, executor: ToolExecutor) -> None:
    """Every number written into the impression must match one a tool returned."""
    allowed = _allowed_numbers(report, executor)
    prose = _NON_MEASUREMENT_TOKENS.sub(" ", report.impression or "")
    for literal in _NUMBER_RE.findall(prose):
        if not _matches_any(literal, allowed):
            raise ValueError(
                f"Untraceable number in impression: '{literal}' does not match any value "
                "returned by a tool call. The agent may not state numbers it did not obtain "
                "from a tool — cite the structured findings instead."
            )


def _allowed_numbers(report: StudyReport, executor: ToolExecutor) -> set[float]:
    """Numbers the agent is permitted to restate in prose."""
    allowed: set[float] = set()
    for entry in executor.trace.values():
        _collect_numbers(entry.get("output"), allowed)
    for lesion in report.findings:
        allowed.update(
            {
                lesion.long_axis_mm.value,
                lesion.short_axis_mm.value,
                lesion.volume_mm3.value,
                lesion.malignancy_score.value,
                lesion.malignancy_confidence,
                lesion.detector_score,
            }
        )
    if report.recist_sum_of_diameters_mm is not None:
        allowed.add(report.recist_sum_of_diameters_mm.value)
    # Structural counts are derived from the report itself, not asserted by the model.
    allowed.add(float(len(report.findings)))
    # Probabilities are commonly narrated as percentages.
    allowed |= {v * 100.0 for v in tuple(allowed) if 0.0 <= v <= 1.0}
    return allowed


def _collect_numbers(obj, into: set[float]) -> None:
    if isinstance(obj, bool):  # bool is an int subclass; not a measurement
        return
    if isinstance(obj, (int, float)):
        into.add(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, into)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, into)


def _matches_any(literal: str, allowed: set[float]) -> bool:
    """Does `literal` match an allowed value at the precision it was written to?

    "14.2" matches 14.23 (the agent rounded); "14" matches 14.2; "15" matches neither. The
    tolerance is half an ulp at the written number of decimals, so a rounded restatement
    passes and a fabricated one does not.
    """
    value = float(literal)
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    tolerance = max(0.5 * (10.0**-decimals), 1e-9)
    return any(abs(value - candidate) <= tolerance for candidate in allowed)


def _load_system_prompt(config: dict) -> str:
    path = Path(config.get("agent", {}).get("prompt_path", "configs/agent_prompt.md"))
    if not path.exists():
        raise FileNotFoundError(f"Agent system prompt not found at {path}")
    return path.read_text(encoding="utf-8")


def _build_client(config: dict):
    """Construct the LLM client for the configured backend.

    Both branches import lazily so the agent layer stays importable — and unit-testable —
    without either SDK installed. The loop below is provider-agnostic: it only needs
    `.messages.create(...)` returning `.content` blocks and a `.stop_reason`.
    """
    backend = config.get("report", {}).get("llm", "claude")
    if backend == "claude":
        import anthropic

        return anthropic.Anthropic()
    if backend == "gemini":
        from oncoct.agent.gemini_client import GeminiClient

        agent_cfg = config.get("agent", {})
        return GeminiClient(model=agent_cfg.get("gemini_model", "gemini-flash-latest"))
    raise NotImplementedError(
        f"Orchestrator backend {backend!r} is not wired; choose 'claude' or 'gemini' "
        "via report.llm in configs/pipeline.yaml."
    )


def run(
    study_path: str,
    config: dict,
    client=None,
    executor: ToolExecutor | None = None,
) -> StudyReport:
    """Run the full agentic pipeline on one study and return a validated StudyReport.

    `client` and `executor` are injectable so the loop can be exercised without an API key
    or the imaging stack; production callers pass neither.
    """
    agent_cfg = config.get("agent", {})
    client = client or _build_client(config)
    executor = executor or ToolExecutor(config)
    system = _load_system_prompt(config)
    max_turns = int(agent_cfg.get("max_turns", DEFAULT_MAX_TURNS))

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Analyze the chest CT study at: {study_path}\n\n"
                "Work the diagnostic checklist in order. Every number you report must come "
                "from a tool call. When you have measured, attributed and classified every "
                "lesion worth reporting, call assemble_report exactly once to finalize."
            ),
        }
    ]

    for _ in range(max_turns):
        response = client.messages.create(
            model=agent_cfg.get("model", DEFAULT_MODEL),
            max_tokens=int(agent_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)),
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        stop = getattr(response, "stop_reason", None)

        if stop == "refusal":
            raise RuntimeError(
                "The model declined this request (stop_reason='refusal'); no report produced."
            )
        if stop == "max_tokens":
            raise RuntimeError(
                "The model hit max_tokens mid-turn; the response is truncated and cannot be "
                "trusted. Raise agent.max_tokens in configs/pipeline.yaml."
            )
        if stop == "pause_turn":
            # Server-side tool loop paused; re-send so it resumes where it left off.
            messages.append({"role": "assistant", "content": response.content})
            continue
        if stop == "end_turn":
            break

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            try:
                output = executor.call(block.name, block.id, **dict(block.input))
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": _as_text(output)}
                )
            except (ToolError, ValueError, KeyError, FileNotFoundError) as exc:
                # Hand the failure back to the model rather than aborting: a wrong call order
                # or a dead detection is recoverable, and the agent is expected to adapt.
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"{type(exc).__name__}: {exc}",
                        "is_error": True,
                    }
                )
        # All results for one assistant turn go back in a SINGLE user message — splitting
        # them trains the model out of making parallel tool calls.
        messages.append({"role": "user", "content": results})
    else:
        raise RuntimeError(
            f"Agent did not finish within {max_turns} turns; no report assembled. "
            "Raise agent.max_turns or narrow the study."
        )

    if executor.report is None:
        raise RuntimeError(
            "The agent ended its turn without calling assemble_report, so there is no report. "
            "This is a hard failure: a partial narration is not a study report."
        )
    verify_traceability(executor.report, executor)
    return executor.report


def _as_text(output) -> str:
    import json

    return json.dumps(output, default=str)
