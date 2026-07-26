"""I/O and coordinate handling.

The single most important correctness concern in a LUNA16 pipeline lives here:
LUNA16 annotations are in WORLD coordinates (mm), but arrays are indexed in VOXELS,
and ITK reports axes as (x, y, z) while numpy indexes as (z, y, x). A silent axis
flip here quietly wrecks detection and evaluation. The conversions in `resample.py`
are pure functions with a round-trip unit test (tests/test_coord_conversion.py).
"""
