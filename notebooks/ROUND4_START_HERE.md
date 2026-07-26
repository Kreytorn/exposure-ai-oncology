# START HERE — Round 4 (A100)

**One notebook, one step. No CPU prep session, no downloads.** Connect an A100, open
`ROUND4_A100.ipynb`, and Run all (~1–2 h). Everything this round needs — subset0, the LUNA16
annotations, the detection bundle, and Round 2's retrained malignancy head — is already in
Drive from earlier rounds.

> This is the **third** handoff folder (`upload_to_drive_3.0`) but **Round 4** of the roadmap.
> They diverged because Round 3 — building the live LLM orchestration agent — was done
> entirely on the laptop and needed no GPU.

## What this round is for

Round 3 built the agent but never ran it on real voxels, and two fixes made on the laptop
are still unverified on real data. This round is cheap and closes all three:

| # | Job | Why it matters |
|---|-----|----------------|
| 1 | **subset0 FROC re-confirm** | A **debt**. Round 2's cell died in 0.24 s on a stale snapshot. Should land near Round 1's **CPM 0.776**. |
| 2 | **Demo gallery + `propagation_drift` verdict** | Round 2 fired that flag on **18/22 lesions (82%)** using a heuristic Round 1 had already rejected. The rewrite is unit-tested but has never seen a real mask. **Expect a low rate. If it is still ~80%, the fix is wrong — say so.** |
| 3 | **First live agent run** | Claude drives the typed tools and writes the report *plus its tool-call trace*. Needs an API key; skipped non-fatally without one. |

Jobs 1 and 2 are the ones that matter. Job 3 is the interesting one.

## Before you Run all

- **Job 3 needs an API key.** Colab → key icon (Secrets) → add `ANTHROPIC_API_KEY` → enable
  for this notebook. Costs a few cents per scan. Skip it and Jobs 1–2 still stand.
- **The CONFIG cell is pre-filled** with the paths Round 2 actually used. The preflight cell
  checks every one of them and stops with a specific message if anything moved.
- **Do not modify `repo_snapshot/`.** It is an exact `git archive` of a laptop commit; its SHA
  is recorded in `SNAPSHOT.txt` and re-checked by the preflight.

## The preflight cell is not boilerplate — read it if it stops you

Round 2 lost a deliverable because nobody could tell the snapshot was ~18 h stale. The
preflight now refuses to run if the two fixes this round exists to verify are missing from
the snapshot, naming which one and why. If it stops you, the snapshot is old: rebuild it on
the laptop with `python scripts/build_handoff.py` and re-upload. Do not work around it — a
stale snapshot is precisely how Round 2 wasted a GPU session.

## When it finishes

Download `results/` and hand it to laptop Claude with three one-line answers:

1. **subset0 CPM** — and whether the debt is paid (near 0.776?).
2. **The `propagation_drift` rate** — fix confirmed, or not.
3. **Did the agent produce a report?** If not, what did it fail on — the trace is written
   even on failure, so say what it tried to claim.

Report all three honestly, including failures. A round that says "job 2 did not work" is
worth more than one that quietly drops it — Round 2's most valuable output was a crash.

## Files

- `ROUND4_A100.ipynb` — the runnable notebook (the whole round).
- `repo_snapshot/` — the pipeline at a known commit. Don't edit.
- `SNAPSHOT.txt` — which commit, and when it was built.
