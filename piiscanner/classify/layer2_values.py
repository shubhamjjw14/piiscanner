"""Layer 2 -- value pattern matching with checksum validation.

This is what finds PII in ``col_17`` and what keeps ``order_reference`` from
being reported as Aadhaar.  Layer 1 can only ever suggest; this layer is the
evidence.

Two matching modes, chosen per field rather than per value:

*whole-value*
    The default.  The entire cell must satisfy the rule.  A 16-digit account
    number is therefore not a 10-digit mobile number that happens to be
    embedded in it.

*free-text*
    Used when a column looks like prose.  The rule is searched for inside the
    value, with lookaround-anchored patterns so a match cannot straddle a
    longer run of digits.  This is what finds the phone number in a support
    ticket note where only 15% of rows contain one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from piiscanner.classify.patterns import (
    RULES,
    PatternRule,
    in_birth_window,
    is_private_ip,
    normalise,
)
from piiscanner.connectors.base import FieldSample
from piiscanner.models import PIIType, mask_value

#: A field is treated as prose when its typical value is at least this long.
FREE_TEXT_MEDIAN_LENGTH = 40

#: Below this many non-null values, a verdict is too thin to be confident in.
#: The finding is still produced, but flagged and penalised.
MIN_CONFIDENT_SAMPLE = 20

#: Free-text hits are rarer by nature, so they need a lower bar than
#: whole-value matches to be worth reporting.
MIN_FREE_TEXT_RATE = 0.05


@dataclass
class ValueVerdict:
    """What Layer 2 concluded about one type on one field."""

    pii_type: PIIType
    match_rate: float
    matched_count: int
    considered: int
    #: Masked, never raw -- these become the finding's evidence.
    evidence: list[str] = field(default_factory=list)
    checksum: bool = False
    weak: bool = False
    requires_name_support: bool = False
    free_text: bool = False
    #: Type-specific extras, e.g. the share of IPs that are private.
    stats: dict[str, float] = field(default_factory=dict)


def looks_like_free_text(texts: list[str]) -> bool:
    """True when a column holds prose rather than discrete identifiers."""
    if not texts:
        return False
    lengths = [len(t) for t in texts]
    if median(lengths) >= FREE_TEXT_MEDIAN_LENGTH:
        return True
    # Short but sentence-shaped: several words in most values.
    wordy = sum(1 for t in texts if t.count(" ") >= 3)
    return wordy / len(texts) >= 0.5


def _evidence_for(rule: PatternRule, texts: list[str], free_text: bool) -> list[str]:
    """Up to three masked examples of the rule firing.

    In free-text mode the masked example is the extracted substring, not the
    whole sentence -- masking a 90-character note would leak most of it.
    """
    out: list[str] = []
    for text in texts:
        if len(out) >= 3:
            break
        if free_text:
            hit = rule.find_in_text(text)
            if hit:
                out.append(mask_value(hit))
        elif rule.matches(text):
            out.append(mask_value(text))
    return out


def classify_values(sample: FieldSample) -> list[ValueVerdict]:
    """Run every rule against a field's sampled values.

    Returns one verdict per type that fired at all; the merge logic decides
    which of them survive into findings.
    """
    texts = [normalise(v) for v in sample.non_null]
    texts = [t for t in texts if t]
    if not texts:
        return []

    free_text = looks_like_free_text(texts)
    verdicts: list[ValueVerdict] = []

    for rule in RULES:
        matched = sum(1 for t in texts if rule.matches(t))
        rate = matched / len(texts)
        mode_free_text = False
        candidates = 0

        # Fall back to searching inside values only when whole-value matching
        # found essentially nothing and the column reads like prose.
        if rate < 0.05 and free_text and rule.free_text_regex is not None:
            found = sum(1 for t in texts if rule.find_in_text(t) is not None)
            if found and found / len(texts) >= MIN_FREE_TEXT_RATE:
                matched, rate, mode_free_text = found, found / len(texts), True
                candidates = sum(1 for t in texts if rule.has_candidate_in_text(t))

        if matched == 0:
            continue

        verdict = ValueVerdict(
            pii_type=rule.pii_type,
            match_rate=rate,
            matched_count=matched,
            considered=len(texts),
            evidence=_evidence_for(rule, texts, mode_free_text),
            checksum=rule.checksum,
            weak=rule.weak,
            requires_name_support=rule.requires_name_support,
            free_text=mode_free_text,
        )
        if mode_free_text and candidates:
            # How many of the shape-matching substrings actually passed the
            # validator.  For a checksum type this is the meaningful precision:
            # `match_rate` is diluted by every value with no candidate at all.
            verdict.stats["candidate_pass_rate"] = matched / candidates
        _attach_stats(verdict, rule, texts)
        verdicts.append(verdict)

    return sorted(verdicts, key=lambda v: -v.match_rate)


def _attach_stats(verdict: ValueVerdict, rule: PatternRule, texts: list[str]) -> None:
    """Compute the column-level statistics some types need."""
    if rule.pii_type is PIIType.IP_ADDRESS:
        matching = [t for t in texts if rule.matches(t)]
        if matching:
            private = sum(1 for t in matching if is_private_ip(t))
            verdict.stats["private_share"] = private / len(matching)

    if rule.pii_type is PIIType.DATE_OF_BIRTH:
        parsed = [t for t in texts if rule.matches(t)]
        if parsed:
            in_window = sum(1 for t in parsed if in_birth_window(t))
            # The spec's alternative to name support: if over 80% of the dates
            # fall in a plausible birth-year range, the column is a birth date
            # even if nobody named it one.
            verdict.stats["birth_window_share"] = in_window / len(parsed)


def refutes(sample: FieldSample, pii_type: PIIType, verdicts: list[ValueVerdict]) -> bool:
    """True when the values actively contradict a Layer 1 hint.

    This is what stops a column named ``pan`` full of solar panel SKUs from
    being reported.  Refutation requires three things: the type must be
    testable by value, we must have enough values to be sure, and essentially
    none of them may match.
    """
    from piiscanner.classify.patterns import NAME_ONLY_TYPES, RULES_BY_TYPE

    if pii_type in NAME_ONLY_TYPES or pii_type not in RULES_BY_TYPE:
        return False  # no value-level test exists, so nothing can refute it
    if len(sample.non_null) < MIN_CONFIDENT_SAMPLE:
        return False  # too little data to contradict anything
    for verdict in verdicts:
        if verdict.pii_type is pii_type:
            return verdict.match_rate < 0.05
    return True  # the rule never fired on a healthy sample
