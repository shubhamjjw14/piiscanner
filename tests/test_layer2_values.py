"""Layer 2 and merge-logic tests that need no database."""

from __future__ import annotations

import random
from decimal import Decimal
from typing import Any

import pytest

from piiscanner.classify.engine import classify_field
from piiscanner.classify.layer2_values import classify_values, looks_like_free_text
from piiscanner.classify.patterns import normalise, parse_date
from piiscanner.connectors.base import FieldSample
from piiscanner.models import DetectionLayer, FieldSchema, PIIType
from piiscanner.testdata.generate import (
    gen_aadhaar,
    gen_bad_transaction_id,
    gen_card,
    gen_email,
    gen_gstin,
    gen_ifsc,
    gen_mobile,
    gen_order_reference,
    gen_pan,
    gen_upi,
    gen_vehicle_reg,
)

SEED = 4242


def make_sample(name: str, values: list[Any], native_type: str = "varchar") -> FieldSample:
    return FieldSample(
        schema=FieldSchema(
            dataset="db", collection="t", name=name, native_type=native_type,
            approx_row_count=len(values),
        ),
        values=values,
    )


def types_found(name: str, values: list[Any]) -> set[PIIType]:
    return {f.pii_type for f in classify_field(make_sample(name, values), "test")}


def gen(fn, n: int = 200, **kwargs: Any) -> list[str]:
    rng = random.Random(SEED)
    return [fn(rng, **kwargs) for _ in range(n)]


# --------------------------------------------------------------------------
# Value normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (Decimal("234567890123"), "234567890123"),  # numeric-typed Aadhaar
        (Decimal("9876543210"), "9876543210"),
        (Decimal("123456.78"), "123456.78"),
        (234567890123, "234567890123"),
        ("  padded  ", "padded"),
        (None, ""),
        (True, ""),  # booleans carry no PII
    ],
)
def test_normalise(raw: Any, expected: str) -> None:
    assert normalise(raw) == expected


def test_bare_numbers_are_not_parsed_as_dates() -> None:
    """A short integer id must not become a date of birth."""
    for value in ("12345", "2024", "999999", "42"):
        assert parse_date(value) is None
    assert parse_date("1985-03-12") is not None


# --------------------------------------------------------------------------
# Detection on clean columns
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,generator,expected",
    [
        ("col_17", gen_aadhaar, PIIType.AADHAAR),
        ("data1", None, PIIType.EMAIL),
        ("extra_field_3", gen_mobile, PIIType.INDIAN_MOBILE),
        ("field_x", gen_card, PIIType.CARD_NUMBER),
        ("blob_9", gen_gstin, PIIType.GSTIN),
        ("zz", gen_pan, PIIType.PAN),
        ("qq", gen_ifsc, PIIType.IFSC),
        ("rr", gen_vehicle_reg, PIIType.VEHICLE_REG),
    ],
)
def test_layer2_finds_pii_in_meaninglessly_named_columns(
    field_name: str, generator: Any, expected: PIIType
) -> None:
    """The whole justification for value scanning: Layer 1 is blind here."""
    rng = random.Random(SEED)
    if generator is None:
        values = [gen_email(rng, "priya", "sharma") for _ in range(200)]
    else:
        values = [generator(rng) for _ in range(200)]
    assert expected in types_found(field_name, values)


def test_aadhaar_accepts_spaced_formatting() -> None:
    rng = random.Random(SEED)
    values = [gen_aadhaar(rng, spaced=True) for _ in range(100)]
    assert PIIType.AADHAAR in types_found("col_9", values)


def test_numeric_typed_aadhaar_is_found() -> None:
    """Aadhaar as BIGINT is common and must not be missed."""
    rng = random.Random(SEED)
    values = [int(gen_aadhaar(rng)) for _ in range(200)]
    assert PIIType.AADHAAR in types_found("col_4", values)


def test_numeric_typed_mobile_is_found() -> None:
    rng = random.Random(SEED)
    values = [Decimal(gen_mobile(rng)) for _ in range(200)]
    assert PIIType.INDIAN_MOBILE in types_found("col_5", values)


def test_upi_is_not_reported_as_email() -> None:
    rng = random.Random(SEED)
    values = [gen_upi(rng, "priya") for _ in range(200)]
    found = types_found("handle", values)
    assert PIIType.UPI_VPA in found
    assert PIIType.EMAIL not in found


def test_email_is_not_reported_as_upi() -> None:
    rng = random.Random(SEED)
    values = [gen_email(rng, "priya", "sharma") for _ in range(200)]
    found = types_found("contact_handle", values)
    assert PIIType.EMAIL in found
    assert PIIType.UPI_VPA not in found


# --------------------------------------------------------------------------
# False positives -- the decoys, at unit level
# --------------------------------------------------------------------------


def test_order_reference_is_not_aadhaar() -> None:
    """12 digits that fail Verhoeff. Regex alone would flag every one."""
    values = gen(gen_order_reference, 300)
    assert types_found("order_reference", values) == set()


def test_transaction_id_is_not_a_card_number() -> None:
    """16 digits that fail Luhn."""
    values = gen(gen_bad_transaction_id, 300)
    assert types_found("transaction_id", values) == set()


def test_column_named_pan_holding_panel_skus_is_refuted() -> None:
    """Layer 1 fires; the values contradict it; nothing is reported."""
    values = ["SOLR-450W-MONO", "PNL-550W-BIFA", "SOLR-330W-POLY"] * 100
    assert types_found("pan", values) == set()


def test_bank_account_digits_are_not_reported_without_name_support() -> None:
    """A 16-digit id is a valid-looking account number; the name must agree."""
    rng = random.Random(SEED)
    values = ["".join(rng.choice("0123456789") for _ in range(14)) for _ in range(200)]
    assert PIIType.BANK_ACCOUNT not in types_found("internal_ref", values)
    assert PIIType.BANK_ACCOUNT in types_found("account_no", values)


def test_six_digit_ids_are_not_pincodes_without_name_support() -> None:
    values = [str(n) for n in range(100_000, 100_300)]
    assert PIIType.PINCODE not in types_found("id", values)
    assert PIIType.PINCODE in types_found("pincode", values)


def test_audit_timestamps_are_not_birth_dates() -> None:
    """Dates outside the birth window, in a column nobody named `dob`."""
    values = [f"2024-{m:02d}-15 10:00:00" for m in range(1, 13)] * 20
    assert PIIType.DATE_OF_BIRTH not in types_found("created_at", values)


def test_birth_window_dates_are_found_even_without_name_support() -> None:
    """The spec's alternative route: >80% of values in a plausible birth range."""
    values = [f"19{y:02d}-06-15" for y in range(50, 99)] * 5
    assert PIIType.DATE_OF_BIRTH in types_found("legacy_col_2", values)


def test_low_cardinality_labels_produce_nothing() -> None:
    for values in (
        ["open", "closed", "escalated"] * 100,
        ["low", "medium", "high"] * 100,
        ["alpha", "beta", "gamma", "delta"] * 75,
    ):
        assert types_found("bucket", values) == set()


# --------------------------------------------------------------------------
# Free text
# --------------------------------------------------------------------------


def test_free_text_detection() -> None:
    assert looks_like_free_text(
        ["Customer reported the app crashing on checkout, asked for a callback"] * 10
    )
    assert not looks_like_free_text(["9876543210"] * 10)


def test_phone_number_embedded_in_free_text_is_found() -> None:
    """The mixed-column case: only 15% of rows contain a number."""
    rng = random.Random(SEED)
    values = []
    for i in range(400):
        if i % 20 < 3:  # 15%
            values.append(
                f"Customer asked for a callback on {gen_mobile(rng)} after 6pm."
            )
        else:
            values.append("Refund processed for the order placed last week.")
    findings = classify_field(make_sample("notes", values), "test")
    mobiles = [f for f in findings if f.pii_type is PIIType.INDIAN_MOBILE]
    assert mobiles, "phone number in free text was missed"
    assert 0.10 <= mobiles[0].match_rate <= 0.20
    assert "free text" in (mobiles[0].notes or "")


def test_free_text_evidence_is_the_extracted_value_not_the_sentence() -> None:
    """Masking a whole 90-character note would leak most of it."""
    rng = random.Random(SEED)
    values = [f"Please call {gen_mobile(rng)} to confirm the delivery slot." for _ in range(50)]
    findings = classify_field(make_sample("notes", values), "test")
    mobile = next(f for f in findings if f.pii_type is PIIType.INDIAN_MOBILE)
    for item in mobile.evidence:
        assert len(item) <= 14
        assert "call" not in item


def test_ten_digit_run_inside_a_longer_number_is_not_a_mobile() -> None:
    """Lookarounds, not \\b: a 16-digit account contains a 10-digit substring."""
    rng = random.Random(SEED)
    values = [
        "Reference number " + "".join(rng.choice("0123456789") for _ in range(16))
        for _ in range(200)
    ]
    assert PIIType.INDIAN_MOBILE not in types_found("notes", values)


# --------------------------------------------------------------------------
# Confidence and merge behaviour
# --------------------------------------------------------------------------


def test_agreement_between_layers_beats_either_alone() -> None:
    rng = random.Random(SEED)
    values = [gen_aadhaar(rng) for _ in range(200)]
    named = classify_field(make_sample("aadhaar_no", values), "test")[0]
    unnamed = classify_field(make_sample("col_17", values), "test")[0]
    assert named.confidence > unnamed.confidence
    assert named.detected_by == [DetectionLayer.NAME, DetectionLayer.VALUE]
    assert unnamed.detected_by == [DetectionLayer.VALUE]


def test_name_only_findings_are_marked_unverified() -> None:
    values = ["Priya Sharma", "Rajesh Kumar", "Rajinikanth"] * 60
    finding = next(
        f for f in classify_field(make_sample("full_name", values), "test")
        if f.pii_type is PIIType.PERSON_NAME
    )
    assert finding.unverified
    assert finding.confidence == 0.5
    assert finding.detected_by == [DetectionLayer.NAME]


def test_mixed_checksum_column_is_still_reported() -> None:
    """Half a column of real Aadhaar numbers is a finding, at lower confidence."""
    rng = random.Random(SEED)
    values = [gen_aadhaar(rng) for _ in range(100)] + [
        gen_order_reference(rng) for _ in range(100)
    ]
    finding = next(
        f for f in classify_field(make_sample("col_17", values), "test")
        if f.pii_type is PIIType.AADHAAR
    )
    assert finding.confidence == pytest.approx(0.70)


def test_thin_samples_are_penalised() -> None:
    """A verdict from 5 values is not the same as one from 500."""
    rng = random.Random(SEED)
    many = [gen_aadhaar(rng) for _ in range(200)]
    few = [gen_aadhaar(rng) for _ in range(5)]
    big = classify_field(make_sample("aadhaar_no", many), "test")[0]
    small = classify_field(make_sample("aadhaar_no", few), "test")[0]
    assert small.confidence < big.confidence
    assert "5 non-null values" in (small.notes or "")


def test_empty_and_all_null_fields_produce_nothing() -> None:
    assert classify_field(make_sample("col_1", []), "test") == []
    assert classify_values(make_sample("col_1", [None, None, None])) == []


def test_private_ip_share_is_reported() -> None:
    values = ["10.0.0.5", "192.168.1.7", "8.8.8.8", "1.1.1.1"] * 50
    finding = next(
        f for f in classify_field(make_sample("ip_addr", values), "test")
        if f.pii_type is PIIType.IP_ADDRESS
    )
    assert "private-range" in (finding.notes or "")


# --------------------------------------------------------------------------
# Regressions found by scanning a real production-shaped schema
#
# Every case below was a false positive on a live fintech database. The
# synthetic dataset did not contain any of these shapes, which is exactly why
# they survived to be found the hard way.
# --------------------------------------------------------------------------


def test_all_null_column_produces_no_findings() -> None:
    """An empty column holds no personal data, whatever it is called.

    Real case: a 100%-NULL card-metadata column was reported as CARD_NUMBER at
    0.5 -- refutation could not fire because there were no values to refute it
    with.
    """
    assert types_found("card_number", [None] * 200) == set()
    assert types_found("aadhaar_number", [None] * 200) == set()
    assert types_found("email", []) == set()


def test_trace_ids_are_not_device_ids() -> None:
    """A 32-hex string is a trace id, a request id, an MD5, or a device id.

    Nothing in the value distinguishes them, so the name must. Real case: 8
    separate tracing correlation-id columns across a service schema were all
    reported as DEVICE_ID at 0.75.
    """
    rng = random.Random(SEED)
    hexes = [
        "".join(rng.choice("0123456789abcdef") for _ in range(32))
        for _ in range(200)
    ]
    assert PIIType.DEVICE_ID not in types_found("trace_id", hexes)
    assert PIIType.DEVICE_ID not in types_found("request_id", hexes)
    assert PIIType.DEVICE_ID not in types_found("correlation_id", hexes)
    assert PIIType.DEVICE_ID not in types_found("checksum", hexes)
    # ...but the name saying `device` is exactly the corroboration needed.
    assert PIIType.DEVICE_ID in types_found("device_id", hexes)


def test_workflow_state_column_is_not_an_address() -> None:
    """`state` is a workflow status at least as often as it is a place.

    Only the values can tell them apart, so an ADDRESS hint resting solely on
    the token `state` has to see a real state name.
    """
    workflow = ["CREATED", "IN_PROGRESS", "APPROVED", "REJECTED"] * 50
    assert PIIType.ADDRESS not in types_found("state", workflow)
    assert PIIType.ADDRESS not in types_found("sub_state", workflow)
    geographic = ["Maharashtra", "Karnataka", "Tamil Nadu", "West Bengal"] * 50
    assert PIIType.ADDRESS in types_found("state", geographic)


def test_organisation_names_are_not_person_names() -> None:
    """Real case: `<org>_name` columns in a payments schema were read as people.

    Card issuers, payment gateways, programme partners and product names all
    live in columns whose name ends `_name`, and none of them is personal data.
    """
    values = ["HDFC Bank", "ICICI Bank", "Axis Bank"] * 60
    for field_name in (
        "issuer_name", "pg_name", "gateway_name", "acquirer_name",
        "network_name", "scheme_name", "partner_program_name", "program_name",
    ):
        assert PIIType.PERSON_NAME not in types_found(field_name, values), field_name
    # The genuine cases must survive.
    for field_name in ("customer_name", "account_holder_name", "full_name"):
        assert PIIType.PERSON_NAME in types_found(field_name, values), field_name


def test_free_text_findings_need_enough_absolute_evidence() -> None:
    """A high rate over a handful of values is not evidence.

    Real case: a JSON blob column with 7 non-null sampled values contained 3
    chance Verhoeff passes, scoring a 43% AADHAAR "match rate".
    """
    rng = random.Random(SEED)
    blob = "Refund note for the order placed last week, no further action."
    thin = [blob] * 4 + [
        f"{blob} ref {gen_aadhaar(rng)}" for _ in range(3)
    ]
    assert PIIType.AADHAAR not in types_found("event_metadata", thin)

    # The same shape with a real sample behind it is still reported.
    thick = [blob] * 400 + [f"{blob} ref {gen_aadhaar(rng)}" for _ in range(100)]
    assert PIIType.AADHAAR in types_found("event_metadata", thick)


def test_card_qualifiers_are_not_card_numbers() -> None:
    """Real case: card-attribute and cardholder-name columns read as CARD_NUMBER.

    A card *number* column is `card_number` or `cc_num`. Bare `card` appears in
    a dozen columns that hold product metadata instead.
    """
    labels = ["PLATINUM", "GOLD", "TITANIUM"] * 40
    for field_name in (
        "card_variant", "premium_card_variant", "card_network", "card_type",
        "card_offering", "card_type_id", "internal_card_id",
    ):
        assert PIIType.CARD_NUMBER not in types_found(field_name, labels), field_name

    names = ["PRIYA SHARMA", "RAJESH KUMAR", "A IYER"] * 40
    found = types_found("card_holder_name", names)
    assert PIIType.PERSON_NAME in found       # it IS a person's name
    assert PIIType.CARD_NUMBER not in found   # it is NOT a card number

    # The precise forms still work.
    rng = random.Random(SEED)
    cards = [gen_card(rng) for _ in range(100)]
    for field_name in ("card_number", "card_no", "cc_num", "credit_card"):
        assert PIIType.CARD_NUMBER in types_found(field_name, cards), field_name


def test_name_only_findings_survive_on_small_tables() -> None:
    """Real case: a cardholder-name column on a 22-row table was silently dropped.

    The small-sample penalty belongs on value evidence. A column's name is
    just as informative at 11 rows as at 11 million.
    """
    names = ["PRIYA SHARMA", "RAJESH KUMAR"] * 5   # 10 values, under the floor
    finding = next(
        f for f in classify_field(make_sample("card_holder_name", names), "test")
        if f.pii_type is PIIType.PERSON_NAME
    )
    assert finding.confidence == 0.5
    assert "10 non-null values sampled" in (finding.notes or "")
