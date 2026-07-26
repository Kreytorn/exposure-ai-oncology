"""Render a StudyReport as an abnormality-first, radiology-style text report.

Lives in the library rather than inside `scripts/run_pipeline.py` so it can be imported
and unit-tested without the imaging stack, and so any future caller renders through the
same implementation instead of a copy that drifts.

Every number printed here is copied from a `Sourced` field, and each finding prints the
tool-call id that produced it. Nothing is narrated that a tool did not measure.
"""

from __future__ import annotations

from oncoct.report.generator import MIN_MEASURABLE_LONG_AXIS_MM
from oncoct.report.schema import StudyReport


def _recist_summary(report: StudyReport, targets: list) -> list[str]:
    """The RECIST block, including WHY there is no sum when there is no sum.

    A study can lack a target for two very different reasons, and saying the wrong one
    is a clinically misleading statement rather than a cosmetic slip: either nothing was
    big enough, or something was big enough and the pipeline distrusted it.
    """
    if report.recist_sum_of_diameters_mm:
        return [
            f"  Sum of target-lesion long axes: "
            f"{report.recist_sum_of_diameters_mm.value:.1f} mm "
            f"(n={len(targets)}, source={report.recist_sum_of_diameters_mm.source})",
            "  Baseline study - no prior for comparison, so no response category.",
        ]

    excluded = [f for f in report.findings if f.recist_exclusion_reason]
    if not excluded:
        return [
            f"  No measurable target lesion (all candidates < "
            f"{MIN_MEASURABLE_LONG_AXIS_MM:.0f} mm long axis)."
        ]

    lines = [
        f"  No target lesion. {len(excluded)} candidate(s) met the "
        f"{MIN_MEASURABLE_LONG_AXIS_MM:.0f} mm threshold but were not selected:"
    ]
    lines += [f"    {f.lesion_id}: {f.recist_exclusion_reason}" for f in excluded]
    lines.append("  No sum of diameters is reported, rather than one built on those.")
    return lines


def _impression(report: StudyReport, targets: list) -> list[str]:
    if report.impression:
        return [f"  {report.impression}"]

    if targets:
        big = max(targets, key=lambda r: r.long_axis_mm.value)
        lobe = f" ({big.lobe_or_segment})" if big.lobe_or_segment else ""
        return [
            f"  {len(report.findings)} pulmonary nodule candidate(s). Largest target: "
            f"{big.lesion_id}, {big.long_axis_mm.value:.1f} mm in {big.organ}{lobe}, "
            f"model malignancy score {big.malignancy_score.value:.2f}.",
            "  (Deterministic assembly - no LLM narration in this round.)",
        ]

    excluded = [f for f in report.findings if f.recist_exclusion_reason]
    if excluded:
        # Do NOT call these sub-centimetre: they cleared the size threshold and were
        # dropped for quality or by a RECIST cap.
        return [
            f"  {len(report.findings)} nodule candidate(s); none selected as a RECIST "
            f"target lesion ({len(excluded)} large enough but excluded, see above). "
            "(Deterministic assembly - no LLM narration.)"
        ]
    return [
        f"  {len(report.findings)} sub-centimetre nodule candidate(s); none measurable "
        "as a RECIST target lesion. (Deterministic assembly - no LLM narration.)"
    ]


def render_text_report(report: StudyReport) -> str:
    """Render `report` as plain text. Pure function of the report."""
    targets = [f for f in report.findings if f.recist_category.value == "target"]
    lines = [
        f"STUDY {report.study_uid}",
        f"{len(report.findings)} lesion candidate(s); "
        f"{len(targets)} RECIST target, {len(report.findings) - len(targets)} non-target.",
        "",
        "FINDINGS:",
    ]
    if not report.findings:
        lines.append("  No lesion candidates above the detector threshold.")

    for f in sorted(report.findings, key=lambda r: -r.long_axis_mm.value):
        site = f.organ + (f" ({f.lobe_or_segment})" if f.lobe_or_segment else "")
        lines.append(
            f"  {f.lesion_id}  {site} - {f.long_axis_mm.value:.1f} x "
            f"{f.short_axis_mm.value:.1f} mm, volume {f.volume_mm3.value:.0f} mm^3"
        )
        lines.append(
            f"        RECIST 1.1: {f.recist_category.value}  |  "
            f"malignancy {f.malignancy_score.value:.2f} "
            f"(confidence {f.malignancy_confidence:.2f})  |  "
            f"detector {f.detector_score:.2f}"
        )
        lines.append(
            f"        sources: size={f.long_axis_mm.source}, volume={f.volume_mm3.source}, "
            f"malignancy={f.malignancy_score.source}"
        )
        if f.quality_flags:
            lines.append(f"        QUALITY FLAGS: {', '.join(f.quality_flags)}")
        if f.recist_exclusion_reason:
            lines.append(f"        NOT A TARGET: {f.recist_exclusion_reason}")

    lines += ["", "RECIST SUMMARY:"] + _recist_summary(report, targets)
    lines += ["", "IMPRESSION:"] + _impression(report, targets)
    lines += ["", report.disclaimer]
    return "\n".join(lines)
