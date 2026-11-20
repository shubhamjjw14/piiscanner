# Changelog

## 1.0.0

First public release.

- Scans MySQL, PostgreSQL and MongoDB for 26 kinds of personal data,
  including Aadhaar, PAN, GSTIN, voter ID, passport, UPI, IFSC, bank and
  card numbers.
- Read-only sessions, a read-only guard on every statement, masked evidence,
  no values in logs.
- Bounded sampling: index seeks on large MySQL tables, `TABLESAMPLE` on
  PostgreSQL, `$sample` or `_id` ranges on MongoDB.
- `import piiscanner; piiscanner.scan(dsn)` from Python, returning the
  findings and the tables that could not be read.
- `benchmark` against a synthetic Indian dataset; `audit` for measuring
  precision and recall on a real schema by hand-labelling a sample.
