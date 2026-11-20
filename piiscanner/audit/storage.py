"""On-disk layout for an audit.

    audit/<name>-<YYYYMMDD>/
        sample.json      the drawn sample, seed, stratum populations, versions
        worksheet.csv    blind labelling sheet -- edited in place by the auditor
        scores.json      computed scores (written by `audit score`)
        misses.md        the work queue: every disagreement, grouped by cause

One directory per audit, named for the source and the date, so a second audit of
the same database months later sits beside the first and the two are comparable.
The classifier and ruleset versions are recorded in ``sample.json`` because a
score is only meaningful against the code that produced it.

Same masking invariant as everywhere else: ``SampledField.evidence`` inherits the
validator from ``MaskedEvidenceModel``, so nothing written here can hold a raw
value.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from piiscanner.audit.sampling import AuditSample

SAMPLE_FILE = "sample.json"
WORKSHEET_FILE = "worksheet.csv"
SCORES_FILE = "scores.json"
MISSES_FILE = "misses.md"


def audit_dir(root: str | Path, name: str, on: date | None = None) -> Path:
    """``<root>/<name>-<YYYYMMDD>``."""
    stamp = (on or date.today()).strftime("%Y%m%d")
    return Path(root) / f"{name}-{stamp}"


def save_audit(sample: AuditSample, directory: str | Path) -> Path:
    """Write ``sample.json`` into ``directory``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / SAMPLE_FILE
    path.write_text(
        json.dumps(sample.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_audit(directory: str | Path) -> AuditSample:
    """Read ``sample.json`` back."""
    path = Path(directory) / SAMPLE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `piiscanner audit sample` first."
        )
    return AuditSample.model_validate(json.loads(path.read_text(encoding="utf-8")))
