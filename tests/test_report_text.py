"""The plain-text report renderer.

Newly testable: it used to live inside scripts/run_pipeline.py behind a torch import, so
nothing could exercise it. These tests pin the two statements that went wrong once a
>=10mm lesion could be excluded for quality rather than for size.
"""

from __future__ import annotations

from test_recist_target_selection import record  # sibling test module, not a package

from oncoct.report.generator import build_study_report
from oncoct.report.text import render_text_report


def render(records) -> str:
    return render_text_report(build_study_report("study", records))


def test_no_target_because_everything_is_small_says_so():
    text = render([record("L1", 5.0), record("L2", 7.0)])
    assert "all candidates < 10 mm long axis" in text
    assert "0 RECIST target, 2 non-target" in text


def test_no_target_because_of_a_flag_does_not_claim_everything_was_small():
    """The bug: a 14.15mm lesion excluded for a quality flag was reported as though no
    candidate had reached the threshold, and as 'sub-centimetre' in the impression."""
    text = render([record("L1", 5.67), record("L2", 14.15, flags=["outside_lung_parenchyma"])])
    assert "all candidates < 10 mm" not in text
    assert "sub-centimetre" not in text
    assert "met the 10 mm threshold but were not selected" in text
    assert "outside_lung_parenchyma" in text


def test_excluded_lesion_is_labelled_in_its_finding_block():
    text = render([record("L2", 14.15, flags=["propagation_drift"])])
    assert "NOT A TARGET:" in text
    assert "propagation_drift" in text
    # And it is not double-prefixed by the reason string.
    assert "NOT A TARGET: not a RECIST target" not in text


def test_no_sum_is_printed_when_the_only_big_lesion_was_excluded():
    text = render([record("L2", 14.15, flags=["outside_lung_parenchyma"])])
    assert "Sum of target-lesion long axes" not in text
    assert "No sum of diameters is reported" in text
    assert "14.1" not in text.split("RECIST SUMMARY:")[1]


def test_a_real_target_still_prints_its_sum_and_source():
    text = render([record("L1", 22.0), record("L2", 5.0)])
    assert "Sum of target-lesion long axes: 22.0 mm (n=1, source=src_L1)" in text
    assert "1 RECIST target, 1 non-target" in text


def test_every_printed_finding_carries_its_sources():
    text = render([record("L1", 22.0), record("L2", 5.0)])
    assert text.count("sources: size=") == 2


def test_disclaimer_is_always_present():
    assert "Not a medical device" in render([record("L1", 22.0)])
    assert "Not a medical device" in render([])


def test_empty_study_renders_without_crashing():
    text = render([])
    assert "No lesion candidates above the detector threshold." in text
    assert "0 RECIST target, 0 non-target" in text
