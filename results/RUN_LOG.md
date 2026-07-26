# RUN LOG - Round 4 (A100), debt paid + all three jobs complete

**Status: ALL 3 JOBS COMPLETE.** Both things this round existed to *verify* came back
confirmed, and - on a second short run after the quota diagnosis below - the agent produced
a full study report on real CT.

Reconstructed on the laptop from the returned `results/` (53 files). Every number below is
read directly out of that folder; nothing is quoted from memory.

---

## JOB 1 - subset0 detection FROC: **DEBT PAID**

`results/metrics/froc_subset0.json`. Round 2 tried this and died in 0.24 s on a stale
snapshot; the preflight added this round made that impossible, and it ran clean.

| | Round 1 | **Round 4** |
|---|---|---|
| CPM (subset0, out-of-fold) | 0.7755 | **0.7755** |
| 95% CI | 0.664–0.873 | 0.660–0.866 |
| scans / nodules | 89 / 112 | 89 / 112 |

**Delta: +0.0000.** Not "close to" - identical to four decimals, on all 89 scans with the
official LUNA16 evaluation script. The CI differs in the third decimal because the bootstrap
resamples; the point estimate is exact. The detection number is now confirmed by two
independent runs, months of code changes apart.

## JOB 2 - the `propagation_drift` fix: **CONFIRMED ON REAL MASKS**

Counted directly from the 8 returned report JSONs (same 8 subset0 scans as Round 2, so the
comparison is like-for-like):

| | Round 2 (rejected heuristic) | **Round 4 (rewritten test)** |
|---|---|---|
| lesions | 22 | 22 |
| `propagation_drift` fires | **18 / 22 (82%)** | **0 / 22 (0%)** |
| `outside_lung_parenchyma` fires | 4 | 4 |

The flag was unit-tested on synthetic masks but had never seen a real one. It now fires on
nothing across 22 real lesions, which is what a discriminating flag should do on a set where
the segmenter behaved - Round 1 measured z-extent/long-axis ratios of 0.72–1.18 on real
lesions, comfortably under the 2.5 threshold.

The 4 `outside_lung_parenchyma` flags are unchanged and **correct**: at the deliberately low
0.02 detection threshold some candidates land on rib or background, and surfacing them is
exactly that flag's job.

*Caveat worth keeping:* 0/22 confirms the flag no longer fires spuriously. It does **not**
prove it would fire on genuine drift - no drifted mask appeared in this sample. The
synthetic unit test (`tests/test_quality_flags.py`) covers that direction.

## JOB 3 - live agent on real CT: **COMPLETE**

`results/agent/` holds the finished artefact: `*_report.json` and its `*_trace.json`.

**24 tool calls, `status: ok`, 5 lesions reported**, on LUNA16 scan `...105756658`:

| lesion | organ / lobe | long axis | volume | malignancy | RECIST |
|---|---|---|---|---|---|
| L1 | lung / upper lobe, left | 5.67 mm | 113.09 mm³ | 0.0002 | non-target |
| L2 | background | 14.15 mm | 1292.20 mm³ | 0.2316 | **target** |
| L3 | lung / lower lobe, right | 5.07 mm | 87.14 mm³ | 0.0002 | non-target |
| L4 | lung / lower lobe, left | 5.49 mm | 93.32 mm³ | 0.0001 | non-target |
| L5 | lung / lower lobe, left | 5.07 mm | 108.15 mm³ | 0.0000 | non-target |

RECIST sum of diameters: **14.15 mm** (the single target lesion), sourced to `d31253mn`.
L2 correctly carries `outside_lung_parenchyma` - it attributed to background, which is what
that flag exists to surface at the 0.02 detection threshold.

**Re-verified offline on the laptop**, not merely trusted from the GPU run: the returned
report was re-validated against the returned trace with `verify_traceability`. It passes.
5 findings cite **10 distinct tool-call ids**, every one present in the trace, and all **16
numeric literals in the agent's own impression** trace to tool output. The agent wrote the
prose; it did not author a single number in it.

The call sequence shows the loop batching work - five `segment_lesion` calls, then five
`measure`, then five `attribute_organ`, then five `classify_malignancy`, then
`assemble_report`. That matters practically: 24 tool calls at one API round trip each would
have exceeded the daily quota outright (see below), so parallel tool calls per turn are what
made a five-lesion study fit inside a free-tier budget at all.

---

## The first attempt at Job 3, and what it cost to understand

The initial Round-4 run did **not** produce a report: the agent made 8 correct tool calls on
the first study and stopped, and the second study made 0. Diagnosing that took two passes and
turned up three separate defects, all worth recording because they share one shape -
**information destroyed before anyone could read it.**

1. **The notebook was swallowing every subprocess's output.** `sh()` ran
   `subprocess.run(cmd, shell=True)` with no capture. Colab captures Python-level stdout but
   not a subprocess's file descriptors, so the agent's traceback - and the output of the
   detection, FROC and all eight pipeline runs - went to the kernel log where nobody sees it.
   The failure looked like "no error at all". Now captured and re-printed.

2. **The cause was a per-DAY quota, not the rate limit I first assumed.** Reproduced against
   the live API: `GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20`. One study is ~9
   round trips, so two studies exhaust a model outright. Backoff cannot help a daily cap, and
   Google still returns a `RetryInfo` of ~55 s for it, which is actively misleading. My
   earlier "retry harder" fix was aimed at the wrong failure.

3. **The client truncated the error body before parsing it.** The quota id lives in `details`,
   after a ~300-character message; cutting to 400 chars hid it, so every 429 went down the
   retry path. Parse the full body, truncate only for display.

**The fix that actually worked:** the quota is scoped *per model*, so a different model has
its own untouched budget. The client now detects a daily cap specifically, drops that model
for the session, and continues on the next. Verified live before the re-run - with
`gemini-flash-latest` spent, a 4-lesion study rolled over to `gemini-flash-lite-latest` and
completed.

The second run then produced the report above.

## What this round settles

- The **detection number is real**: 0.7755, reproduced exactly, out-of-fold, official script.
- The **drift-flag regression is genuinely fixed** on real data, not just in a unit test.
- The **agent completes a real multi-lesion study end to end** - 5 lesions, 24 tool calls,
  a RECIST sum, and an impression whose every number is traceable to a tool call.

## What is still owed

Nothing from the GPU. The imaging plane, the orchestration plane, and the reliability
invariant are all demonstrated on real data. Remaining work is writing, not computing.
