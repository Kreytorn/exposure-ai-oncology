"""Structured report schema (pydantic).

Every numeric field carries a ``source`` - the id of the tool call that produced it.
This is what the traceability guard (tests/test_agent_no_unsourced_numbers.py) checks:
no number reaches the report without a tool-call provenance.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RecistCategory(str, Enum):
    TARGET = "target"
    NON_TARGET = "non_target"


class Sourced(BaseModel):
    """A value paired with the tool-call id that produced it."""

    value: float
    source: str = Field(..., description="tool_call id in the agent trace")


class LesionReport(BaseModel):
    lesion_id: str
    organ: str                                  # from attribute_organ tool
    lobe_or_segment: str | None = None
    long_axis_mm: Sourced
    short_axis_mm: Sourced
    volume_mm3: Sourced
    malignancy_score: Sourced                   # 0..1
    malignancy_confidence: float
    recist_category: RecistCategory
    detector_score: float
    requires_review: bool = True                # candidate - radiologist review
    quality_flags: list[str] = Field(default_factory=list)   # e.g. "propagation_drift"


class StudyReport(BaseModel):
    study_uid: str
    findings: list[LesionReport]
    impression: str
    recist_sum_of_diameters_mm: Sourced | None = None
    disclaimer: str = (
        "Research/decision-support output. Not a medical device. "
        "All findings are candidates requiring radiologist review."
    )

    def all_numeric_sources(self) -> list[str]:
        """Every tool-call id cited by any numeric field - used by the traceability test."""
        srcs: list[str] = []
        for lesion in self.findings:
            srcs += [
                lesion.long_axis_mm.source,
                lesion.short_axis_mm.source,
                lesion.volume_mm3.source,
                lesion.malignancy_score.source,
            ]
        if self.recist_sum_of_diameters_mm is not None:
            srcs.append(self.recist_sum_of_diameters_mm.source)
        return srcs
