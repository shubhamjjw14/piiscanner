"""JSON findings output.

The written file is the artefact a customer keeps, attaches to an audit, or
feeds into a ticketing system.  It contains masked evidence only -- the
masking is enforced upstream by the ``Finding`` model, and re-asserted here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from piiscanner.models import ScanResult


def to_dict(result: ScanResult) -> dict[str, Any]:
    """Render a scan result as plain JSON-safe data, findings sorted by rank."""
    payload = result.model_dump(mode="json")
    payload["findings"] = [f.model_dump(mode="json") for f in result.sorted_findings()]
    payload["summary"] = {
        "total_findings": len(result.findings),
        "by_pii_type": result.counts_by_type(),
        "datasets_scanned": result.datasets_scanned,
        "collections_scanned": result.collections_scanned,
        "fields_examined": result.fields_examined,
        "fields_skipped": result.fields_skipped,
    }
    return payload


def write_json(result: ScanResult, path: str | Path) -> Path:
    """Write findings to ``path`` and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(to_dict(result), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return target


def read_json(path: str | Path) -> ScanResult:
    """Load a findings file back into a :class:`ScanResult`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("summary", None)
    return ScanResult.model_validate(data)
