"""Stratified sampling of a scan's field inventory.

Why stratified.  On the real schema this was built for, 6 of 591 examined fields
were flagged.  A uniform sample of 60 fields would contain 0 or 1 flagged
field, so precision -- the number a buyer cares about most -- would rest on a
single label.  Meanwhile the 585 unflagged fields are where recall hides, and
labelling all of them by hand is not going to happen.

So: take the flagged fields as a *census* (all of them, precision is exact) and
*sample* the other two strata (recall is estimated, with an interval).

The population size of each stratum is recorded because the extrapolation
depends on it: an observed false-negative rate of 2/60 means something very
different against 585 unflagged fields than against 60.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from piiscanner.models import (
    CLASSIFIER_VERSION,
    RULESET_VERSION,
    FieldRecord,
    Finding,
    MaskedEvidenceModel,
    PIIType,
    ScanResult,
)


class Stratum(str, Enum):
    """Which population a sampled field was drawn from."""

    FLAGGED = "flagged"      # a finding above threshold
    UNFLAGGED = "unflagged"  # examined, nothing reported
    SKIPPED = "skipped"      # excluded before classification


#: Default draw per stratum.  ``None`` means census -- take every one.
STRATA: dict[Stratum, int | None] = {
    Stratum.FLAGGED: None,
    Stratum.UNFLAGGED: 60,
    Stratum.SKIPPED: 15,
}


class SampledField(MaskedEvidenceModel):
    """One field drawn for labelling, with the scanner's verdict alongside.

    The verdict lives here, in the audit record, and is deliberately *not*
    written to the worksheet -- see :mod:`piiscanner.audit.worksheet`.
    """

    #: Opaque, stable, and carries no information about stratum or verdict.
    #: The worksheet is keyed on this and nothing else.
    row_id: str

    dataset: str
    collection: str
    field: str
    native_type: str
    stratum: Stratum

    # -- what the labeller sees -------------------------------------------
    sample_count: int = 0
    null_rate: float = 0.0
    blank_rate: float = 0.0
    distinct_ratio: float = 0.0
    approx_row_count: int | None = None
    evidence: list[str] = Field(default_factory=list, max_length=3)

    # -- what the labeller must NOT see -----------------------------------
    scanner_flagged: bool = False
    scanner_pii_type: PIIType | None = None
    scanner_confidence: float | None = None
    scanner_unverified: bool = False
    scanner_detected_by: list[str] = Field(default_factory=list)
    scanner_match_rate: float | None = None
    scanner_notes: str | None = None
    skip_reason: str | None = None

    @property
    def path(self) -> str:
        return f"{self.dataset}.{self.collection}.{self.field}"


class AuditSample(BaseModel):
    """A drawn sample, plus everything needed to reproduce and score it."""

    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    #: Provenance of the scan being audited.
    source_alias: str
    scan_run_id: str
    classifier_version: str = CLASSIFIER_VERSION
    ruleset_version: str = RULESET_VERSION

    seed: int
    #: Stratum -> size of the *whole population*, not the sample.  The recall
    #: extrapolation multiplies by this, so getting it wrong silently corrupts
    #: the headline number.
    population: dict[Stratum, int] = Field(default_factory=dict)
    #: Stratum -> how many were actually drawn.
    drawn: dict[Stratum, int] = Field(default_factory=dict)

    fields: list[SampledField] = Field(default_factory=list)

    def by_stratum(self, stratum: Stratum) -> list[SampledField]:
        return [f for f in self.fields if f.stratum is stratum]

    def by_row_id(self) -> dict[str, SampledField]:
        return {f.row_id: f for f in self.fields}


def _best_finding(findings: list[Finding]) -> Finding | None:
    """The highest-confidence finding for a field, which is its verdict."""
    return max(findings, key=lambda f: f.confidence, default=None)


def _classify_strata(result: ScanResult) -> dict[Stratum, list[FieldRecord]]:
    """Split the inventory into the three populations."""
    buckets: dict[Stratum, list[FieldRecord]] = {s: [] for s in Stratum}
    for record in result.fields:
        if not record.examined:
            buckets[Stratum.SKIPPED].append(record)
        elif record.flagged:
            buckets[Stratum.FLAGGED].append(record)
        else:
            buckets[Stratum.UNFLAGGED].append(record)
    return buckets


def draw_sample(
    result: ScanResult,
    name: str,
    seed: int,
    sizes: dict[Stratum, int | None] | None = None,
) -> AuditSample:
    """Draw a stratified sample from a scan's field inventory.

    Raises ``ValueError`` when the scan carries no inventory: recall cannot be
    measured from a findings list, and silently producing a flagged-only sample
    would give a precision-shaped number that looks like an audit.
    """
    if not result.has_inventory:
        raise ValueError(
            "this scan has no field inventory, so unflagged and skipped fields "
            "cannot be sampled. Re-run: piiscanner scan --inventory ..."
        )

    sizes = {**STRATA, **(sizes or {})}
    buckets = _classify_strata(result)

    # Findings indexed by field, so each sampled field carries its verdict.
    findings_by_path: dict[str, list[Finding]] = {}
    for finding in result.findings:
        findings_by_path.setdefault(finding.path, []).append(finding)

    rng = random.Random(seed)
    sample = AuditSample(
        name=name,
        source_alias=result.connectors[0].alias if result.connectors else "unknown",
        scan_run_id=result.run_id,
        classifier_version=result.classifier_version,
        ruleset_version=result.ruleset_version,
        seed=seed,
        population={s: len(records) for s, records in buckets.items()},
    )

    seen_ids: set[str] = set()

    for stratum, records in buckets.items():
        want = sizes.get(stratum)
        # Sort before sampling: the inventory arrives in whatever order the
        # connector produced, which is not guaranteed stable across runs. A
        # seeded draw over an unstable order is not reproducible.
        ordered = sorted(records, key=lambda r: r.path)
        chosen = (
            ordered
            if want is None or want >= len(ordered)
            else rng.sample(ordered, want)
        )
        sample.drawn[stratum] = len(chosen)
        for record in chosen:
            verdict = _best_finding(findings_by_path.get(record.path, []))
            sample.fields.append(
                SampledField(
                    row_id=_row_id(rng, seen_ids),
                    dataset=record.dataset,
                    collection=record.collection,
                    field=record.field,
                    native_type=record.native_type,
                    stratum=stratum,
                    sample_count=record.sample_count,
                    null_rate=record.null_rate,
                    blank_rate=record.blank_rate,
                    distinct_ratio=record.distinct_ratio,
                    approx_row_count=record.approx_row_count,
                    evidence=record.evidence,
                    skip_reason=record.skip_reason,
                    scanner_flagged=record.flagged,
                    scanner_pii_type=verdict.pii_type if verdict else None,
                    scanner_confidence=verdict.confidence if verdict else None,
                    scanner_unverified=verdict.unverified if verdict else False,
                    scanner_detected_by=(
                        [d.value for d in verdict.detected_by] if verdict else []
                    ),
                    scanner_match_rate=verdict.match_rate if verdict else None,
                    scanner_notes=verdict.notes if verdict else None,
                )
            )

    # Shuffle across strata so worksheet order leaks nothing.
    rng.shuffle(sample.fields)
    return sample


#: Consonants and unambiguous digits: no vowels (so ids cannot spell anything
#: suggestive), no 0/1/l/o/i (so they survive being read aloud or retyped).
_ID_ALPHABET = "bcdfghjkmnpqrstvwxz23456789"
_ID_LENGTH = 8


def _row_id(rng: random.Random, seen: set[str]) -> str:
    """An opaque, unique row id.

    Not a hash of the field path: a labeller who guessed the scheme could
    recompute it, and ids must not sort into stratum order.  Drawn from the
    seeded RNG so the whole sample stays reproducible.

    Uniqueness is enforced rather than assumed.  A collision is unlikely at
    these sizes, but the worksheet is keyed on this value alone -- two rows
    sharing an id would silently merge one label onto both fields, and the audit
    would be quietly wrong rather than loudly broken.
    """
    for _ in range(1000):
        candidate = "".join(rng.choice(_ID_ALPHABET) for _ in range(_ID_LENGTH))
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    raise RuntimeError("could not generate a unique row id")  # pragma: no cover
