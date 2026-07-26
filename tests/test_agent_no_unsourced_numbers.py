"""The reliability invariant: no number in the report without a tool-call source.

This is the test that makes an "autonomous" report trustworthy. It exercises the same
check the runtime orchestrator runs (orchestrator.verify_traceability).
"""

import pytest

from oncoct.agent.orchestrator import verify_traceability
from oncoct.agent.tools import ToolExecutor
from oncoct.report.schema import LesionReport, RecistCategory, Sourced, StudyReport


def _report_with_sources(long_src, vol_src, mal_src, short_src) -> StudyReport:
    return StudyReport(
        study_uid="1.2.3",
        findings=[
            LesionReport(
                lesion_id="L1",
                organ="lung_upper_lobe_right",
                lobe_or_segment="RUL",
                long_axis_mm=Sourced(value=14.2, source=long_src),
                short_axis_mm=Sourced(value=9.1, source=short_src),
                volume_mm3=Sourced(value=812.0, source=vol_src),
                malignancy_score=Sourced(value=0.73, source=mal_src),
                malignancy_confidence=0.66,
                recist_category=RecistCategory.TARGET,
                detector_score=0.91,
            )
        ],
        impression="Single RUL nodule, indeterminate.",
    )


def _executor_with(ids) -> ToolExecutor:
    ex = ToolExecutor()
    ex.trace = {i: {"tool": "measure", "input": {}, "output": {}} for i in ids}
    return ex


def test_fully_sourced_report_passes():
    ex = _executor_with(["c_measure_L1", "c_classify_L1"])
    report = _report_with_sources(
        long_src="c_measure_L1",
        short_src="c_measure_L1",
        vol_src="c_measure_L1",
        mal_src="c_classify_L1",
    )
    verify_traceability(report, ex)   # must not raise


def test_unsourced_number_is_rejected():
    ex = _executor_with(["c_measure_L1"])            # classify id is missing from trace
    report = _report_with_sources(
        long_src="c_measure_L1",
        short_src="c_measure_L1",
        vol_src="c_measure_L1",
        mal_src="c_hallucinated",                     # not a real tool call
    )
    with pytest.raises(ValueError, match="Untraceable number"):
        verify_traceability(report, ex)


def test_recist_sum_sources_are_all_checked():
    ex = _executor_with(["c_measure_L1"])            # only one of the two summed ids exists
    report = _report_with_sources(
        long_src="c_measure_L1",
        short_src="c_measure_L1",
        vol_src="c_measure_L1",
        mal_src="c_measure_L1",
    )
    report.recist_sum_of_diameters_mm = Sourced(value=28.4, source="c_measure_L1;c_missing")
    with pytest.raises(ValueError, match="Untraceable number"):
        verify_traceability(report, ex)
