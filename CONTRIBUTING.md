# Contributing

Bug reports and pull requests are welcome.

**Never paste real data** into an issue, a test or a pull request: no values
from a real database, not even masked. Describe a column by its name, its
type and what kind of values it holds. If a test needs data, generate it.

## A false alarm or a miss

Say which type was reported (or missed), the column name and type, and what
the values look like ("12 digits, starting 2026..."). `scripts/explain.py`
prints why a column was or was not flagged, with values masked; its output
is the most useful thing to include.

## A pull request

- `pytest` passes. Tests that need a database skip without one; see the
  README for loading the synthetic data.
- A new detector comes with tests for what it must find *and* for what it
  must not (`tests/test_layer1_names.py` has the anti-substring cases:
  `pan` must not fire on `japan_office`).
- The guarantees in the README hold: no write, no value kept or logged, no
  network but the database.

By contributing you agree your contribution is licensed under Apache 2.0.
