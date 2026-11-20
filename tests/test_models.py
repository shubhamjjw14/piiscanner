"""Tests for the data model, focused on the no-raw-values invariant."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from piiscanner.connectors.base import FieldSample
from piiscanner.models import (
    DetectionLayer,
    Finding,
    FieldSchema,
    PIIType,
    ScanResult,
    mask_value,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("9876543210", "98XXXXXX10"),
        ("9812345621", "98XXXXXX21"),  # the shape from the spec
        ("234567890123", "23XXXXXXXX23"),
        ("priya@example.com", "prXXXXXXXXXXXXXom"),
        ("abcdef", "abXXef"),
        ("abcde", "XXXXX"),  # under 6 characters: fully masked
        ("a", "X"),
        ("", ""),
        (None, ""),
        (9876543210, "98XXXXXX10"),  # non-string values are stringified first
    ],
)
def test_mask_value(raw: object, expected: str) -> None:
    assert mask_value(raw) == expected


def test_mask_value_is_idempotent() -> None:
    """Idempotence is what lets the validator mask defensively."""
    for raw in ("9876543210", "234567890123", "abcdef", "abc"):
        once = mask_value(raw)
        assert mask_value(once) == once


def _finding(**overrides: object) -> Finding:
    kwargs: dict[str, object] = dict(
        source="prod",
        dataset="app",
        collection="users",
        field="mobile",
        pii_type=PIIType.INDIAN_MOBILE,
        confidence=0.95,
        match_rate=0.98,
        sample_count=500,
        null_rate=0.02,
        distinct_ratio=0.99,
        evidence=["9876543210"],
        detected_by=[DetectionLayer.NAME, DetectionLayer.VALUE],
    )
    kwargs.update(overrides)
    return Finding(**kwargs)  # type: ignore[arg-type]


def test_finding_masks_evidence_even_when_given_raw_values() -> None:
    finding = _finding(evidence=["9876543210", "priya@example.com"])
    assert finding.evidence == ["98XXXXXX10", "prXXXXXXXXXXXXXom"]


def test_finding_masks_evidence_in_serialised_output() -> None:
    """The invariant has to hold at the boundary, not just in memory."""
    dumped = _finding(evidence=["234567890123"]).model_dump_json()
    assert "234567890123" not in dumped
    assert "23XXXXXXXX23" in dumped


def test_finding_evidence_is_capped_at_three() -> None:
    finding = _finding(evidence=[f"98765432{i:02d}" for i in range(10)])
    assert len(finding.evidence) == 3


def test_finding_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        _finding(confidence=1.5)


def test_findings_sort_by_confidence_then_row_count() -> None:
    small = _finding(confidence=0.95, collection="lookup", approx_row_count=200)
    large = _finding(confidence=0.95, collection="events", approx_row_count=40_000_000)
    weak = _finding(confidence=0.5, collection="misc", approx_row_count=90_000_000)
    result = ScanResult(findings=[small, weak, large])
    assert [f.collection for f in result.sorted_findings()] == [
        "events",
        "lookup",
        "misc",
    ]


def test_counts_by_type() -> None:
    result = ScanResult(
        findings=[
            _finding(pii_type=PIIType.EMAIL),
            _finding(pii_type=PIIType.EMAIL),
            _finding(pii_type=PIIType.AADHAAR),
        ]
    )
    assert result.counts_by_type() == {"EMAIL": 2, "AADHAAR": 1}


def test_field_sample_repr_does_not_leak_values() -> None:
    """An accidental log of a sample must not print PII."""
    schema = FieldSchema(
        dataset="app", collection="users", name="aadhaar", native_type="bigint"
    )
    sample = FieldSample(
        schema=schema,
        values=["234567890123", None, "876543210987"],
    )
    for rendering in (repr(sample), str(sample), f"{sample}"):
        assert "234567890123" not in rendering
        assert "<redacted>" in rendering


def test_field_sample_statistics() -> None:
    schema = FieldSchema(
        dataset="app", collection="users", name="city", native_type="varchar"
    )
    sample = FieldSample(
        schema=schema,
        values=["Pune", "Pune", "Kochi", None, ""],
    )
    assert sample.non_null == ["Pune", "Pune", "Kochi"]
    assert sample.total_sampled == 5
    assert sample.null_count == 1               # only the None
    assert sample.empty_count == 1              # only the ""
    assert sample.non_null_count == 3
    assert sample.null_rate == pytest.approx(0.2)     # 1/5
    assert sample.blank_rate == pytest.approx(0.4)    # (1 + 1)/5
    assert sample.distinct_ratio == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Blank accounting.
#
# `null_count` used to be passed in by each connector counting `None`, while
# `non_null` was derived here and excluded `""` as well.  The two disagreed, so
# a column that was 62% empty string reported `null_rate == 0.0` and read as
# fully populated.  Every count is now derived from one list.
# --------------------------------------------------------------------------


def _sample(values: list[object], name: str = "col") -> FieldSample:
    return FieldSample(
        schema=FieldSchema(
            dataset="db", collection="t", name=name, native_type="varchar"
        ),
        values=list(values),
    )


@pytest.mark.parametrize(
    "values,total,nulls,empties,usable",
    [
        ([None] * 10, 10, 10, 0, 0),                       # all NULL
        ([""] * 10, 10, 0, 10, 0),                         # all empty string
        ([None] * 3 + [""] * 2 + ["Pune"] * 5, 10, 3, 2, 5),  # mixed
        (["Pune"] * 10, 10, 0, 0, 10),                     # fully populated
        ([], 0, 0, 0, 0),                                  # nothing sampled
    ],
)
def test_blank_buckets_are_disjoint_and_total(
    values: list[object], total: int, nulls: int, empties: int, usable: int
) -> None:
    sample = _sample(values)
    assert sample.total_sampled == total
    assert sample.null_count == nulls
    assert sample.empty_count == empties
    assert sample.non_null_count == usable
    # The invariant that makes the two rates reconcilable.
    assert sample.null_count + sample.empty_count + sample.non_null_count == total


def test_all_null_column_rates() -> None:
    sample = _sample([None] * 10)
    assert sample.null_rate == 1.0
    assert sample.blank_rate == 1.0
    assert sample.distinct_ratio == 0.0


def test_all_empty_string_column_rates() -> None:
    """The case the old code got wrong: not one NULL, yet entirely blank."""
    sample = _sample([""] * 10)
    assert sample.null_rate == 0.0      # strictly None -- there are none
    assert sample.blank_rate == 1.0     # ...but the column is completely empty
    assert sample.distinct_ratio == 0.0


def test_sixty_percent_empty_string_column() -> None:
    """The real-schema shape: mostly `''`, the rest genuine values."""
    values = [""] * 60 + [f"name{i}" for i in range(40)]
    sample = _sample(values, name="customer_name")

    assert sample.null_rate == 0.0                     # would have read as "full"
    assert sample.blank_rate == pytest.approx(0.60)    # what a reviewer needs
    assert sample.non_null_count == 40
    # Denominator is non_null_count, not total_sampled: of the values that
    # exist, all 40 are unique.
    assert sample.distinct_ratio == 1.0


def test_empty_string_is_not_counted_as_null() -> None:
    sample = _sample(["a", "", None])
    assert sample.null_count == 1
    assert sample.empty_count == 1
    assert sample.null_rate == pytest.approx(1 / 3)
    assert sample.blank_rate == pytest.approx(2 / 3)


def test_finding_blank_rate_reaches_the_report() -> None:
    """The fix is worthless if the number never leaves the classifier."""
    from piiscanner.classify.engine import classify_field

    values = [""] * 60 + ["PRIYA SHARMA", "RAJESH KUMAR"] * 20
    findings = classify_field(_sample(values, name="customer_name"), "test")
    finding = next(f for f in findings if f.pii_type is PIIType.PERSON_NAME)
    assert finding.null_rate == 0.0
    assert finding.blank_rate == pytest.approx(0.6)
    assert finding.sample_count == 100


def test_falsy_non_string_values_are_not_blank() -> None:
    """`0` and `False` are values, not blanks -- `0 == ""` is False in Python."""
    sample = _sample([0, False, 0.0])
    assert sample.null_count == 0
    assert sample.empty_count == 0
    assert sample.blank_rate == 0.0


def test_evidence_is_masked_on_assignment_not_only_on_construction() -> None:
    """Pydantic validates at construction by default, so `finding.evidence = raw`
    used to store a raw value. `validate_assignment=True` closes that."""
    finding = Finding(
        source="s", dataset="d", collection="c", field="f",
        pii_type=PIIType.EMAIL, confidence=0.9, match_rate=0.9,
        sample_count=1, null_rate=0.0, distinct_ratio=1.0,
        evidence=["9812345621"],
    )
    assert finding.evidence == ["98XXXXXX21"]

    finding.evidence = ["9812345621", "priya@example.com"]
    assert finding.evidence == ["98XXXXXX21", "prXXXXXXXXXXXXXom"]


def test_field_record_shares_the_masking_invariant() -> None:
    from piiscanner.models import FieldRecord

    record = FieldRecord(
        dataset="d", collection="c", field="f", native_type="varchar",
        evidence=["9812345621"],
    )
    assert record.evidence == ["98XXXXXX21"]
    record.evidence = ["9812345621"]
    assert record.evidence == ["98XXXXXX21"]
