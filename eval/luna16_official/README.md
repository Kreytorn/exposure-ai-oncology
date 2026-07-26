# LUNA16 official evaluation - vendored & ported

These files (`noduleCADEvaluationLUNA16.py`, `NoduleFinding.py`, `tools/csvTools.py`) are the
**official LUNA16 challenge FROC/CPM evaluation script**, from the LUNA16 grand-challenge
organizers (luna16.grand-challenge.org). They are vendored here so our FROC numbers use the
canonical scoring code rather than a re-implementation (a home-rolled FROC inflates CPM).

**Not our original work** - provenance is the LUNA16 challenge. They carry the challenge's own
terms, separate from this repo's Apache-2.0 license; do not treat them as Apache-2.0.

**Port:** the original ships as **Python 2**. Changes made to run under Python 3, all
behaviour-preserving:
- `print` statements -> `print()`
- `dict.iteritems()` -> `dict.items()`
- CSV opened in text mode (was binary)
- removed two deprecated matplotlib kwargs (`basex=`, `grid(b=)`)
- audited every division for Py2 floor-division: each site has a float operand, so no numeric
  behaviour changed.

Called via `eval/froc_luna16.py::evaluate_froc`.
