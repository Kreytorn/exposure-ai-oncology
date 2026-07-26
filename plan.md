# oncoct: an autonomous CT pipeline for lung nodules

An LLM agent reads a chest CT, finds lung nodules, measures them, scores them, and writes a
radiology report. The point of the project is not the individual models. It is that **the
agent cannot state a number it did not get from a tool**, and that this is enforced by the
code rather than promised in a prompt.

> Research and decision support only. Not a medical device. Every finding is a candidate that
> needs a radiologist.

## 1. Where it stands

Everything below was measured by running the code, not quoted from a paper.

| Stage | Metric | Result |
|-------|--------|--------|
| Detection | FROC / CPM, LUNA16 subset0, 89 scans | **0.7755** (95% CI 0.66 to 0.87) |
| Segmentation | Dice vs LIDC contours, 15 lesions | **0.70** mean, 0.73 median |
| Malignancy | AUROC, patient-grouped 5-fold CV, n=1353 | **0.876 ± 0.045** |
| Agent | full study on real CT | **5 lesions, 24 tool calls, 0 untraced numbers** |

Two things are worth knowing about these.

The detection number was measured twice, in separate runs weeks apart, and came back
identical to four decimal places. It is out of fold: the pretrained bundle trained on subsets
1 to 9 and validated on subset0, so subset0 is the honest split. Reporting subset1 would look
better and mean nothing.

The malignancy classifier ties a plain logistic regression on the same frozen features
(pooled out-of-fold 0.869 for the probe, 0.860 for the trained head). At the earlier n=196
the head led by 0.028, and that gap did not survive more data. So the useful signal is in the
pretrained encoder, not in the head I trained. I kept the head and say so rather than quoting
the old gap.

## 2. The design: two planes

```
ORCHESTRATION PLANE (the LLM)
  picks what to do next, calls typed tools, reads their JSON, writes the prose.
  Never touches voxels. Never computes a number.
        |
        |  typed tool calls, JSON in and JSON out
        v
IMAGING PLANE (deterministic)
  ingest -> organ context -> detect -> segment -> measure -> attribute -> classify
  Each stage is cached, unit tested, and swappable on its own.
```

The failure mode this design targets is the one that matters clinically: a model turning a
vague visual impression into a confident, wrong sentence. Prompting a model to "only use tool
outputs" does not prevent that, because nothing checks it.

## 3. How the guarantee is actually enforced

Three mechanisms, all tested.

**Tools talk in handles, not values.** Tools reference each other with `study_uid`,
`detection_id`, `lesion_id`. Measurements live in the executor's state and never appear as a
tool argument, so there is no channel through which the model could retype or invent one.

**The report is built from the trace.** The agent calls `assemble_report` with a lesion list
and its prose. It does not supply any figures. Every numeric field is filled in from the
recorded tool outputs and tagged with the id of the call that produced it.

**The prose is checked too.** The impression is the one thing the model writes freely, so
every number in it has to match a value some tool returned during that run. Rounded
restatements pass (8.4 for 8.43). Percentages pass (73% for 0.73). Anything else fails the
run and no report is produced.

`tests/test_agent_orchestration.py` drives the real loop with a scripted model and checks
both directions: a rounded restatement is accepted, a fabricated figure is rejected. It needs
no API key and no GPU.

## 4. The pipeline

| # | Stage | Tool | Trained here? |
|---|-------|------|---------------|
| 1 | Ingest | SimpleITK, resample to 0.703 x 0.703 x 1.25 mm, HU preserved | no |
| 2 | Organ context | TotalSegmentator (117 structures) | no, pretrained |
| 3 | Detection | MONAI `lung_nodule_ct_detection` (3D RetinaNet) | no, pretrained |
| 4 | Segmentation | MedSAM2, box to 3D mask (VISTA3D as fallback) | no, pretrained |
| 5 | Measurement | RECIST 1.1 long and short axis, volume | no, rule based |
| 6 | Organ attribution | mask overlap against the cached organ map | no, rule based |
| 7 | Malignancy | frozen detector trunk plus a small head, LIDC labels | **yes, the only training** |
| 8 | Orchestration | LLM tool calling, Claude or Gemini | no |

Only stage 7 is trained, and only its head, on labels that already exist. Everything else is
pretrained inference. That was a deliberate scoping decision: a full 10-fold RetinaNet would
cost roughly 55 GPU hours per fold and would not have made the system any more of a system.

## 5. What it produced

One LUNA16 scan, analysed end to end by the agent. It chose the order of operations, made 24
tool calls, and wrote the impression itself.

| | |
|---|---|
| ![lung nodule](docs/img/agent_L1_lung_nodule.png) | ![flagged](docs/img/agent_L2_flagged_outside_lung.png) |
| **L1**: 5.67 mm, 113.09 mm3, lung / upper lobe left, malignancy 0.0002 | **L2**: 14.15 mm, attributed to background, flagged `outside_lung_parenchyma` |

The right hand case is the one I would point at. The detector fired on something in the chest
wall, outside the lung. Organ attribution caught it, the quality flag surfaced it, and it
reached the report labelled as a candidate needing review instead of as a lung nodule. The
0.02 detection threshold is deliberately low, so false positives are expected. What matters
is that the system says so.

All 16 numbers in the agent's impression trace back to recorded tool calls. I re-checked that
on the laptop against the saved trace rather than trusting the run that produced it. The
report and its full audit trail are in `results/agent/`.

## 6. Data, and the traps in it

**LUNA16** (888 chest CTs, CC BY 4.0) for detection. Annotations are world coordinates in
millimetres, not voxel indices.

**LIDC-IDRI** (CC BY 3.0) for malignancy labels. LUNA16 itself has no malignancy labels,
which is a common mistake. Four radiologists rated each nodule 1 to 5. I take the median and
drop median==3 as ambiguous. That policy removes the hardest cases, so the AUROC is optimistic
compared to an unfiltered screening population. It is stated here because it changes any
comparison.

Two things that fail silently rather than loudly, so both have unit tests:

1. **Coordinates.** LUNA16 gives world millimetres, ITK indexes as (x, y, z), numpy as
   (z, y, x). One axis mix up puts every lesion in the wrong place with no error.
2. **Orientation.** SimpleITK loads LPS, the detector expects RAS. Without the flip, every
   lesion is reported in the mirror image lung. Tested A/B: with the flip, detections land
   0.1 to 0.3 mm from ground truth; without it, 20 to 118 mm away.

## 7. Compute

About 6 hours of A100 total, across four rounds, plus a laptop. There is no from scratch
training, so this was comfortable. The malignancy head trains on cached frozen features in
seconds because the encoder never changes.

MedSAM2 pins torch 2.5.1 with CUDA 12.4, which conflicts with the MONAI stack. It runs in its
own environment through a subprocess worker rather than being forced into one environment.

## 8. What I would do next

- **Longitudinal RECIST.** Match lesions across timepoints and produce a response category.
  This is the part clinicians actually use and the current system is single timepoint.
- **Calibrate the malignancy confidence.** Today it is sharpness, `1 - normalized entropy`,
  not a calibrated probability. It should not be shown to a clinician as reliability.
- **Decide the head.** It ties a linear probe. Either find data or architecture that beats
  one, or drop to the probe and describe it accurately.
- **More segmentation ground truth.** Dice on 15 lesions is thin.

## 9. Honest limitations

- The malignancy AUROC is optimistic because median==3 nodules are dropped.
- Segmentation Dice is measured on 15 lesions, which is a small sample.
- The detection number comes from a pretrained bundle evaluated on its own validation fold.
  It carries model selection bias, though not gradient exposure.
- The shipped malignancy checkpoint is refit on all folds. The reported AUROC is the cross
  validated estimate, not a score of that checkpoint on held out data.
- One complete agent study has been run on real CT, not a large batch of them.

## 10. Layout

```
src/oncoct/      io, context, detect, segment, measure, classify, report, agent
eval/            LUNA16 FROC (official script, ported), Dice
scripts/         run_pipeline (deterministic), run_agent (LLM driven), training, eval
tests/           coordinates, RECIST, quality flags, traceability, the agent loop
configs/         pipeline.yaml, agent_prompt.md
results/agent/   the agent's report and its tool call trace
```

61 tests, all passing, none needing a GPU or an API key.
