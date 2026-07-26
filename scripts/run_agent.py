"""Run the LLM orchestration agent on one study and write the report + its audit trail.

The agentic counterpart to `run_pipeline.py`. Same imaging plane, same StudyReport - the
difference is who decides the call order: here an LLM does, through the typed tools in
`oncoct.agent.tools`, and the run is rejected unless every number in the report traces to a
tool call it actually made.

The trace is written alongside the report on purpose. "Every number is grounded" is the
project's central claim, and a claim you cannot audit after the fact is a slogan - the trace
is the evidence, one entry per tool call, with the ids the report cites.

    ANTHROPIC_API_KEY=... python scripts/run_agent.py --series <scan.mhd> --out results

Requires `if __name__ == "__main__"`: TotalSegmentator's nnUNet backend uses spawn, and an
unguarded entry point makes the workers re-import this module and die misleadingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oncoct.agent.orchestrator import run  # noqa: E402
from oncoct.agent.tools import ToolExecutor  # noqa: E402


def _serializable(value):
    """Trace entries hold whatever a tool returned; keep the JSON dump total."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", required=True, help="path to a .mhd study")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default=None, help="scratch dir for resampled volumes / organ maps")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else out_dir.parent / "cache"
    executor = ToolExecutor(config, cache_dir=cache_dir, out_dir=out_dir)

    failure = None
    try:
        report = run(args.series, config, executor=executor)
    except BaseException as exc:
        # Record WHY, then re-raise. Round 4's agent job stopped after 8 tool calls and the
        # trace showed only that it stopped - the reason lived in Colab stdout that was gone
        # by the time anyone looked. A trace that cannot explain its own failure is half an
        # audit trail, and this is the half you need when something goes wrong.
        failure = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise
    finally:
        # Write the trace even on failure - a rejected run is exactly when you want to see
        # which tool calls were made and what the agent tried to claim from them.
        trace_dir = out_dir / "agent"
        trace_dir.mkdir(parents=True, exist_ok=True)
        study = Path(args.series).stem
        (trace_dir / f"{study}_trace.json").write_text(
            json.dumps(
                {
                    "study": study,
                    "status": "failed" if failure else "ok",
                    "failure": failure,
                    "n_tool_calls": len(executor.trace),
                    "calls": [
                        {
                            "tool_call_id": cid,
                            "tool": entry["tool"],
                            "input": _serializable(entry["input"]),
                            "output": _serializable(entry["output"]),
                        }
                        for cid, entry in executor.trace.items()
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    (out_dir / "agent").mkdir(parents=True, exist_ok=True)
    (out_dir / "agent" / f"{report.study_uid}_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"\nAgent run OK - {len(executor.trace)} tool calls, {len(report.findings)} findings.")
    print(f"  report  {out_dir / 'agent' / (report.study_uid + '_report.json')}")
    print(f"  trace   {out_dir / 'agent' / (Path(args.series).stem + '_trace.json')}")
    print(f"  impression: {report.impression}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
