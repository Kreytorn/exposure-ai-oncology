"""oncoct — autonomous oncology CT pipeline.

Two planes (see plan.md):
  - imaging plane: deterministic tool nodes (io, context, detect, segment, measure,
    classify, labels) that touch voxels;
  - orchestration plane: an LLM agent (agent/) that calls typed tools and writes the
    report, and never touches voxels or invents a measurement.
"""

__version__ = "0.1.0"
