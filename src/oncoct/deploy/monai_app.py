"""MONAI Deploy application (optional): wrap the pipeline as a MAP container.

DICOM series in -> DICOM SEG (masks) + DICOM SR (structured report) out, for a
hospital-integration story. For the hackathon demo, a plain FastAPI + Docker service
(see plan.md §3, stage 10) is the faster path; this is the clinical-integration variant.
"""

from __future__ import annotations


def build_app():
    """Construct the MONAI Deploy Application DAG wrapping the oncoct pipeline."""
    raise NotImplementedError("Wire monai.deploy operators: DICOMDataLoader -> pipeline -> DICOMSeg/SR writer.")
