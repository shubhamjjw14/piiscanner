"""Merge Layer 1 and Layer 2 into a confidence-scored verdict.

The scoring rules come from the spec.  Three additions were needed to make the
false-positive suite pass, and each is called out below where it applies:

*refutation*
    A Layer 1 hint that the values actively contradict is dropped, not
    reported at 0.5.  Without this, any column named ``pan`` is a finding
    regardless of what it holds.

*name support gating*
    Generic patterns (bank account, pincode, age) are only reported when the
    field name corroborates.  Without this, every 16-digit transaction id is a
    bank account.

*one verdict per field*
    When several types are candidates for one field, verified candidates
    displace unverified ones.  Without this, ``ip_addr`` yields both
    IP_ADDRESS (correct) and ADDRESS (noise).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from piiscanner.classify.layer1_names import classify_name
from piiscanner.classify.layer2_values import (
    MIN_CONFIDENT_SAMPLE,
    ValueVerdict,
    classify_values,
    refutes,
)
from piiscanner.classify.patterns import (
    NAME_ONLY_TYPES,
    looks_like_indian_state,
    normalise,
)
from piiscanner.connectors.base import Connector, FieldSample
from piiscanner.models import (
    DetectionLayer,
    FieldRecord,
    FieldSchema,
    Finding,
    PIIType,
    ScanResult,
    mask_value,
)

if TYPE_CHECKING:  # pragma: no cover
    from piiscanner.config import ScanConfig

log = logging.getLogger("piiscanner.classify.engine")

#: Findings below this confidence are not reported.
REPORT_THRESHOLD = 0.4

#: Confidence awarded by each scoring rule.
CONF_CHECKSUM_STRONG = 0.95      # checksum type, match rate >= 0.8
CONF_CHECKSUM_MIXED = 0.70       # checksum type, match rate 0.3-0.8
CONF_PATTERN_STRONG = 0.75       # non-checksum type, match rate >= 0.9
CONF_PATTERN_MIXED = 0.60        # non-checksum type, match rate 0.3-0.9
CONF_PATTERN_SPARSE = 0.45       # non-checksum type, match rate 0.1-0.3
CONF_NAME_ONLY = 0.50            # Layer 1 with nothing to confirm it
CONF_WEAK_CAP = 0.50             # weak types cannot exceed this on values alone
AGREEMENT_BONUS = 0.10
CONF_CAP = 0.99

#: A checksum type below this match rate is noise, not a mixed column.  Set
#: above the ~10% of random 12-digit numbers that pass Verhoeff by chance.
CHECKSUM_NOISE_FLOOR = 0.30

#: Thin samples cannot support a confident verdict.
SMALL_SAMPLE_PENALTY = 0.15

#: A free-text hit needs a real sample behind it, not just a high rate on a
#: handful of values.  Found on real data: a JSON blob column with 7 non-null
#: sampled values and 3 chance Verhoeff passes scored a 43% "match rate".
MIN_FREE_TEXT_MATCHES = 5
MIN_FREE_TEXT_CONSIDERED = 20


@dataclass
class Candidate:
    """A type under consideration for a field, before it becomes a finding."""

    pii_type: PIIType
    confidence: float
    match_rate: float
    evidence: list[str]
    detected_by: list[DetectionLayer]
    unverified: bool
    notes: list[str]

    @property
    def verified(self) -> bool:
        return DetectionLayer.VALUE in self.detected_by


def _score_values(verdict: ValueVerdict, name_supports: bool) -> float | None:
    """Confidence from values alone, or ``None`` if this verdict is not reportable."""
    rate = verdict.match_rate

    if verdict.checksum:
        # A passing checksum is the strongest signal available, but only when
        # enough of the column passes.  Below the floor we are looking at
        # chance passes on non-PII data.
        #
        # In free-text mode the floor is applied to the *candidate* pass rate,
        # not the raw match rate: most sentences contain no 12-digit number at
        # all, so `rate` is diluted by them and would hide a real finding. What
        # matters is how many of the right-shaped substrings passed the check.
        effective = (
            verdict.stats.get("candidate_pass_rate", rate)
            if verdict.free_text
            else rate
        )
        if effective >= 0.8:
            return CONF_CHECKSUM_STRONG if not verdict.free_text else CONF_CHECKSUM_MIXED
        if effective >= CHECKSUM_NOISE_FLOOR:
            return CONF_CHECKSUM_MIXED
        return None

    if verdict.requires_name_support and not name_supports:
        # Too generic to report on values alone.
        return None

    if verdict.weak:
        return CONF_WEAK_CAP if rate >= 0.5 else None

    if rate >= 0.9:
        return CONF_PATTERN_STRONG
    if rate >= 0.3:
        return CONF_PATTERN_MIXED
    if rate >= 0.1:
        # Mixed free-text columns live here: a notes field where 15% of rows
        # contain a phone number is a real finding, just a weaker one.
        return CONF_PATTERN_SPARSE
    return None


def _dob_gate(verdict: ValueVerdict, name_supports: bool) -> bool:
    """Date columns are only birth dates with name support or a birth-year skew."""
    if name_supports:
        return True
    return verdict.stats.get("birth_window_share", 0.0) > 0.8


def _free_text_gate(verdict: ValueVerdict) -> bool:
    """Free-text findings need enough absolute evidence, not just a rate."""
    if not verdict.free_text:
        return True
    return (
        verdict.matched_count >= MIN_FREE_TEXT_MATCHES
        and verdict.considered >= MIN_FREE_TEXT_CONSIDERED
    )


def _state_token_gate(hint, sample: FieldSample) -> bool:
    """Distinguish a geographic `state` column from a workflow `state` column.

    Both names are ubiquitous and only the values tell them apart, so an
    ADDRESS hint that rests solely on the token `state` has to see at least
    one recognisable state or union territory among the sampled values.
    """
    if hint.pii_type is not PIIType.ADDRESS or hint.matched_token != "state":
        return True
    texts = [normalise(v) for v in sample.non_null]
    if not texts:
        return False
    return any(looks_like_indian_state(t) for t in texts)


def classify_field(sample: FieldSample, source: str) -> list[Finding]:
    """Produce the findings for one sampled field."""
    schema = sample.schema
    # An entirely empty column holds no personal data, whatever it is called.
    # Reporting one is noise a reviewer has to chase down for nothing.
    if not sample.non_null:
        return []
    hints = classify_name(schema.name)
    verdicts = classify_values(sample)
    verdict_by_type = {v.pii_type: v for v in verdicts}
    hinted_types = {h.pii_type for h in hints}

    candidates: dict[PIIType, Candidate] = {}

    # -- Layer 2 candidates ------------------------------------------------
    for verdict in verdicts:
        name_supports = verdict.pii_type in hinted_types
        if verdict.pii_type is PIIType.DATE_OF_BIRTH and not _dob_gate(
            verdict, name_supports
        ):
            continue
        if not _free_text_gate(verdict):
            continue
        score = _score_values(verdict, name_supports)
        if score is None:
            continue

        notes: list[str] = []
        detected_by = [DetectionLayer.VALUE]
        if name_supports:
            # Both layers agree: take the higher and add the bonus.
            score = min(CONF_CAP, max(score, CONF_NAME_ONLY) + AGREEMENT_BONUS)
            detected_by.insert(0, DetectionLayer.NAME)
        if verdict.free_text:
            notes.append(
                f"found inside free text in {verdict.match_rate:.0%} of sampled values"
            )
        if verdict.pii_type is PIIType.IP_ADDRESS and "private_share" in verdict.stats:
            notes.append(
                f"{verdict.stats['private_share']:.0%} of addresses are private-range"
            )

        candidates[verdict.pii_type] = Candidate(
            pii_type=verdict.pii_type,
            confidence=score,
            match_rate=verdict.match_rate,
            evidence=verdict.evidence,
            detected_by=detected_by,
            unverified=False,
            notes=notes,
        )

    # -- Layer 1 candidates the values did not already confirm --------------
    for hint in hints:
        if hint.pii_type in candidates:
            continue
        if not _state_token_gate(hint, sample):
            log.debug(
                "dropped ADDRESS hint on %s.%s: `state` holds no state names",
                schema.collection, schema.name,
            )
            continue
        if refutes(sample, hint.pii_type, verdicts):
            log.debug(
                "refuted name hint %s on %s.%s: values do not match",
                hint.pii_type.value, schema.collection, schema.name,
            )
            continue
        note = "field name only; no value-level confirmation"
        if hint.pii_type not in NAME_ONLY_TYPES:
            note += " (values were inconclusive)"
        candidates[hint.pii_type] = Candidate(
            pii_type=hint.pii_type,
            confidence=CONF_NAME_ONLY,
            match_rate=verdict_by_type[hint.pii_type].match_rate
            if hint.pii_type in verdict_by_type
            else 0.0,
            evidence=[],
            detected_by=[DetectionLayer.NAME],
            unverified=True,
            notes=[note],
        )

    if not candidates:
        return []

    # A verified candidate displaces unverified ones on the same field.
    if any(c.verified for c in candidates.values()):
        candidates = {t: c for t, c in candidates.items() if c.verified}

    # -- penalties and thresholding ----------------------------------------
    findings: list[Finding] = []
    non_null = len(sample.non_null)
    for candidate in candidates.values():
        confidence = candidate.confidence
        notes = list(candidate.notes)
        if 0 < non_null < MIN_CONFIDENT_SAMPLE:
            # The penalty applies to value evidence only.  A cardholder-name
            # column is named that whether it holds 11 rows or 11 million;
            # discounting the name because the table is small dropped genuine
            # PII below the reporting threshold on a real schema.
            if candidate.verified:
                confidence = max(0.0, confidence - SMALL_SAMPLE_PENALTY)
            notes.append(f"only {non_null} non-null values sampled")
        if confidence < REPORT_THRESHOLD:
            continue
        findings.append(
            Finding(
                source=source,
                dataset=schema.dataset,
                collection=schema.collection,
                field=schema.name,
                pii_type=candidate.pii_type,
                confidence=round(confidence, 2),
                match_rate=round(candidate.match_rate, 4),
                sample_count=sample.total_sampled,
                null_rate=round(sample.null_rate, 4),
                blank_rate=round(sample.blank_rate, 4),
                distinct_ratio=round(sample.distinct_ratio, 4),
                evidence=candidate.evidence,
                detected_by=candidate.detected_by,
                unverified=candidate.unverified,
                approx_row_count=schema.approx_row_count,
                notes="; ".join(notes) or None,
            )
        )
    return findings


def _sampled_record(sample: FieldSample, flagged: bool) -> FieldRecord:
    """Inventory entry for a field that was sampled and classified.

    ``evidence`` here is the first few *sampled* values, masked -- not the
    values some rule matched.  An auditor labelling an unflagged field needs to
    see what it holds, and for an unflagged field no rule fired at all.
    """
    schema = sample.schema
    return FieldRecord(
        dataset=schema.dataset,
        collection=schema.collection,
        field=schema.name,
        native_type=schema.native_type,
        examined=True,
        flagged=flagged,
        approx_row_count=schema.approx_row_count,
        presence_count=schema.presence_count,
        sample_count=sample.total_sampled,
        null_rate=round(sample.null_rate, 4),
        blank_rate=round(sample.blank_rate, 4),
        distinct_ratio=round(sample.distinct_ratio, 4),
        evidence=[mask_value(normalise(v)) for v in sample.non_null[:3]],
    )


def _skipped_record(schema: FieldSchema) -> FieldRecord:
    """Inventory entry for a field excluded before sampling.

    No evidence: skipped fields are never read, which is the point of skipping
    them.  An auditor judges these on name and declared type alone -- the same
    information the skip rule itself had.
    """
    return FieldRecord(
        dataset=schema.dataset,
        collection=schema.collection,
        field=schema.name,
        native_type=schema.native_type,
        examined=False,
        skip_reason=schema.skip_reason,
        flagged=False,
        approx_row_count=schema.approx_row_count,
        presence_count=schema.presence_count,
    )


def run_scan(
    connector: Connector,
    config: "ScanConfig",
    on_field: "Callable[[str, str, str], None] | None" = None,
) -> ScanResult:
    """Scan every in-scope field reachable through ``connector``.

    ``on_field`` is called with (dataset, collection, field) as each field is
    classified, so a caller can drive a progress display without this module
    depending on a UI library.
    """
    from datetime import datetime, timezone

    result = ScanResult(connectors=[connector.info()])
    datasets = config.datasets or connector.list_datasets()

    for dataset in datasets:
        schemas = connector.get_schema(dataset)
        in_scope = [s for s in schemas if config.matches(dataset, s.collection)]
        if not in_scope:
            continue
        result.datasets_scanned += 1
        result.collections_scanned += len({s.collection for s in in_scope})
        result.fields_skipped += sum(1 for s in in_scope if s.skip)

        if config.record_inventory:
            result.fields.extend(
                _skipped_record(s) for s in in_scope if s.skip
            )

        scoped_tables = {s.collection for s in in_scope if not s.skip}
        for sample in connector.iter_samples(dataset, config.sample_size):
            if sample.schema.collection not in scoped_tables:
                continue
            result.fields_examined += 1
            if on_field is not None:
                on_field(dataset, sample.schema.collection, sample.schema.name)
            findings = classify_field(sample, source=connector.alias)
            result.findings.extend(findings)
            if config.record_inventory:
                result.fields.append(_sampled_record(sample, flagged=bool(findings)))

    result.tables_not_read = sorted({f"{dataset}.{collection}" for dataset, collection, _
                                     in getattr(connector, "sampling_errors", [])})
    result.finished_at = datetime.now(timezone.utc)
    log.info(
        "scan complete: %d datasets, %d fields examined, %d findings",
        result.datasets_scanned, result.fields_examined, len(result.findings),
    )
    return result
