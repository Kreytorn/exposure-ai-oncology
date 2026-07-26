"""RECIST 1.1 target-lesion selection.

Pins the three rules the naive "long_axis >= 10mm" test got wrong: a flagged lesion is
never a target, at most two targets per organ, at most five in total. Pure logic; no
imaging stack, GPU or API key required.
"""

from __future__ import annotations

from oncoct.report.generator import (
    MAX_TARGET_LESIONS,
    MAX_TARGETS_PER_ORGAN,
    build_study_report,
    select_recist_targets,
)
from oncoct.report.schema import RecistCategory


def record(lesion_id: str, long_axis: float, organ: str = "lung", flags=None) -> dict:
    """One lesion_records entry, with the *_source tags build_study_report requires."""
    return {
        "lesion_id": lesion_id,
        "organ": organ,
        "lobe_or_segment": None,
        "long_axis_mm": long_axis,
        "long_axis_source": f"src_{lesion_id}",
        "short_axis_mm": long_axis * 0.8,
        "short_axis_source": f"src_{lesion_id}",
        "volume_mm3": long_axis**3,
        "volume_source": f"src_{lesion_id}",
        "malignancy_score": 0.5,
        "malignancy_source": f"mal_{lesion_id}",
        "malignancy_confidence": 0.5,
        "detector_score": 0.9,
        "quality_flags": list(flags or []),
    }


def targets(report) -> list[str]:
    return [f.lesion_id for f in report.findings if f.recist_category is RecistCategory.TARGET]


# -- rule 1: a flagged lesion is never a target -------------------------------------------


def test_flagged_lesion_is_not_a_target():
    """The shipped-report bug: the sum of diameters rested entirely on the one lesion
    the pipeline had flagged as outside the lung."""
    report = build_study_report(
        "study",
        [
            record("L1", 5.67),
            record("L2", 14.15, organ="background", flags=["outside_lung_parenchyma"]),
        ],
    )
    assert targets(report) == []
    # No eligible target means no headline number at all, rather than a number sourced
    # from a finding the same report warns about.
    assert report.recist_sum_of_diameters_mm is None
    l2 = next(f for f in report.findings if f.lesion_id == "L2")
    assert l2.recist_category is RecistCategory.NON_TARGET
    assert "outside_lung_parenchyma" in l2.recist_exclusion_reason


def test_unflagged_lesion_of_the_same_size_is_a_target():
    """Guards the fix from over-firing: size alone still qualifies when nothing is wrong."""
    report = build_study_report("study", [record("L2", 14.15)])
    assert targets(report) == ["L2"]
    assert report.recist_sum_of_diameters_mm.value == 14.15
    assert report.recist_sum_of_diameters_mm.source == "src_L2"


def test_sum_excludes_the_flagged_lesion_but_keeps_the_clean_one():
    report = build_study_report(
        "study",
        [
            record("L1", 20.0, flags=["propagation_drift"]),
            record("L2", 12.0),
        ],
    )
    assert targets(report) == ["L2"]
    assert report.recist_sum_of_diameters_mm.value == 12.0
    # The sum cites only the surviving target, not the larger flagged lesion.
    assert report.recist_sum_of_diameters_mm.source == "src_L2"


# -- rule 2: at most two targets per organ ------------------------------------------------


def test_at_most_two_targets_per_organ():
    records = [record(f"L{i}", 30.0 - i, organ="lung") for i in range(4)]
    report = build_study_report("study", records)
    assert len(targets(report)) == MAX_TARGETS_PER_ORGAN
    # Largest first: L0 (30.0) and L1 (29.0).
    assert targets(report) == ["L0", "L1"]
    l2 = next(f for f in report.findings if f.lesion_id == "L2")
    assert "maximum of 2 target lesions" in l2.recist_exclusion_reason


def test_the_per_organ_cap_is_per_organ_not_global():
    report = build_study_report(
        "study",
        [
            record("A1", 20.0, organ="lung"),
            record("A2", 19.0, organ="lung"),
            record("B1", 18.0, organ="liver"),
            record("B2", 17.0, organ="liver"),
        ],
    )
    assert sorted(targets(report)) == ["A1", "A2", "B1", "B2"]


# -- rule 3: at most five targets in total ------------------------------------------------


def test_at_most_five_targets_in_total():
    """Three organs at two each would be six; RECIST 1.1 caps the study at five."""
    records = [
        record(f"{organ}{i}", 30.0 - n, organ=organ)
        for n, (organ, i) in enumerate(
            [(o, i) for o in ("lung", "liver", "kidney") for i in (1, 2)]
        )
    ]
    report = build_study_report("study", records)
    assert len(targets(report)) == MAX_TARGET_LESIONS
    dropped = [f for f in report.findings if f.recist_category is RecistCategory.NON_TARGET]
    assert len(dropped) == 1
    assert "maximum of 5 target lesions" in dropped[0].recist_exclusion_reason


def test_sum_of_diameters_adds_only_the_selected_targets():
    records = [record(f"L{i}", 20.0, organ=f"organ{i}") for i in range(6)]
    report = build_study_report("study", records)
    assert report.recist_sum_of_diameters_mm.value == 20.0 * MAX_TARGET_LESIONS
    assert len(report.recist_sum_of_diameters_mm.source.split(";")) == MAX_TARGET_LESIONS


# -- selection is deterministic and explains itself ---------------------------------------


def test_selection_is_largest_first_and_deterministic():
    """The two largest win the per-organ cap, and input order does not change WHICH.

    `findings` deliberately preserves the caller's lesion order, so compare the selected
    set rather than its position in the list.
    """
    records = [record("small", 11.0), record("big", 25.0), record("mid", 18.0)]
    assert sorted(targets(build_study_report("study", records))) == ["big", "mid"]
    assert sorted(targets(build_study_report("study", list(reversed(records))))) == ["big", "mid"]


def test_findings_preserve_the_caller_order():
    records = [record("small", 11.0), record("big", 25.0), record("mid", 18.0)]
    report = build_study_report("study", records)
    assert [f.lesion_id for f in report.findings] == ["small", "big", "mid"]


def test_sub_threshold_lesions_are_non_target_without_an_excuse():
    """A 5mm nodule is not "excluded", it simply never qualified. No reason string."""
    report = build_study_report("study", [record("L1", 5.0)])
    finding = report.findings[0]
    assert finding.recist_category is RecistCategory.NON_TARGET
    assert finding.recist_exclusion_reason is None


def test_select_recist_targets_marks_selected_lesions_with_none():
    verdict = select_recist_targets([record("L1", 20.0), record("L2", 14.0, flags=["x"])])
    assert verdict["L1"] is None
    assert verdict["L2"] is not None
