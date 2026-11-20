"""Layer 1 tests.

The substring-versus-token distinction gets its own block, because getting it
wrong is the single most common defect in name-based PII classifiers and it is
invisible until a customer scans a table with a column called `japan_office`.
"""

from __future__ import annotations

import pytest

from piiscanner.classify.layer1_names import (
    MAX_NAME_CONFIDENCE,
    classify_name,
    suggests,
    tokenise,
)
from piiscanner.models import PIIType


@pytest.mark.parametrize(
    "name,expected",
    [
        ("customer_email", ["customer", "email"]),
        ("customerEmail", ["customer", "email"]),
        ("CustomerEmail", ["customer", "email"]),
        ("IPAddress", ["ip", "address"]),
        ("extra_field_3", ["extra", "field", "3"]),
        ("col_17", ["col", "17"]),
        ("data1", ["data", "1"]),
        ("company_panel", ["company", "panel"]),
        ("dobbs_lane", ["dobbs", "lane"]),
        ("japan_office", ["japan", "office"]),
        ("user.contact.phone", ["user", "contact", "phone"]),
        ("orders[].shipping.pincode", ["orders", "shipping", "pincode"]),
        ("PAN", ["pan"]),
        ("", []),
    ],
)
def test_tokenise(name: str, expected: list[str]) -> None:
    assert tokenise(name) == expected


# --------------------------------------------------------------------------
# Whole-token matching -- the anti-substring suite
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "panel", "company_panel", "japan", "japan_office", "panasonic",
        "solar_panel_id", "panic_flag", "span", "expander",
    ],
)
def test_pan_does_not_fire_on_substrings(name: str) -> None:
    assert not suggests(name, PIIType.PAN), name


@pytest.mark.parametrize(
    "name", ["dobbs", "dobbs_lane", "hadoop_job", "adobe_id", "dobro"]
)
def test_dob_does_not_fire_on_substrings(name: str) -> None:
    assert not suggests(name, PIIType.DATE_OF_BIRTH), name


@pytest.mark.parametrize(
    "name", ["shipment", "equipment", "zipper", "unzip_flag"]
)
def test_pincode_does_not_fire_on_substrings(name: str) -> None:
    assert not suggests(name, PIIType.PINCODE), name


@pytest.mark.parametrize("name", ["uidai_ref", "guid", "druid", "liquid", "squid"])
def test_uid_does_not_fire_on_substrings(name: str) -> None:
    """`uid` is a real Aadhaar keyword; `guid` and `liquid` merely contain it."""
    fires = suggests(name, PIIType.AADHAAR)
    assert fires is (name == "uidai_ref"), name


@pytest.mark.parametrize("name", ["gstin_no", "gst_rate", "august_total", "gusto"])
def test_gst_does_not_fire_on_substrings(name: str) -> None:
    fires = suggests(name, PIIType.GSTIN)
    assert fires is (name in {"gstin_no", "gst_rate"}), name


# --------------------------------------------------------------------------
# Positive coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,pii_type",
    [
        ("email", PIIType.EMAIL),
        ("user_email", PIIType.EMAIL),
        ("emailAddress", PIIType.EMAIL),
        ("mobile", PIIType.INDIAN_MOBILE),
        ("phone_number", PIIType.INDIAN_MOBILE),
        ("contact_num", PIIType.INDIAN_MOBILE),
        ("msisdn", PIIType.INDIAN_MOBILE),
        ("full_name", PIIType.PERSON_NAME),
        ("customer_name", PIIType.PERSON_NAME),
        ("fname", PIIType.PERSON_NAME),
        ("lname", PIIType.PERSON_NAME),
        ("address_line", PIIType.ADDRESS),
        ("addr1", PIIType.ADDRESS),
        ("street", PIIType.ADDRESS),
        ("city", PIIType.ADDRESS),
        ("pincode", PIIType.PINCODE),
        ("zip_code", PIIType.PINCODE),
        ("area_pin", PIIType.PINCODE),
        ("dob", PIIType.DATE_OF_BIRTH),
        ("date_of_birth", PIIType.DATE_OF_BIRTH),
        ("birthDate", PIIType.DATE_OF_BIRTH),
        ("age", PIIType.AGE),
        ("aadhaar_no", PIIType.AADHAAR),
        ("aadhar", PIIType.AADHAAR),
        ("uid_number", PIIType.AADHAAR),
        ("pan_number", PIIType.PAN),
        ("pan_card", PIIType.PAN),
        ("gstin", PIIType.GSTIN),
        ("ifsc_code", PIIType.IFSC),
        ("account_no", PIIType.BANK_ACCOUNT),
        ("acct_number", PIIType.BANK_ACCOUNT),
        ("passport_no", PIIType.INDIAN_PASSPORT),
        ("voter_epic_id", PIIType.VOTER_ID),
        ("upi_id", PIIType.UPI_VPA),
        ("vpa", PIIType.UPI_VPA),
        ("ip_addr", PIIType.IP_ADDRESS),
        ("device_id", PIIType.DEVICE_ID),
        ("latitude", PIIType.GEO_LOCATION),
        ("lng", PIIType.GEO_LOCATION),
        ("annual_ctc", PIIType.SALARY),
        ("salary", PIIType.SALARY),
        ("gender", PIIType.GENDER),
        ("religion", PIIType.RELIGION),
        ("caste", PIIType.CASTE),
        ("diagnosis", PIIType.HEALTH_DATA),
        ("medical_history", PIIType.HEALTH_DATA),
        ("blood_group", PIIType.HEALTH_DATA),
    ],
)
def test_keyword_coverage(name: str, pii_type: PIIType) -> None:
    assert suggests(name, pii_type), f"{name} should suggest {pii_type.value}"


def test_layer1_confidence_is_capped_at_half() -> None:
    """Layer 1 is a hint, never a verdict."""
    for hint in classify_name("aadhaar_number"):
        assert hint.confidence <= MAX_NAME_CONFIDENCE


def test_negating_tokens_suppress_non_personal_names() -> None:
    """`company_name` is not a person."""
    for name in ("company_name", "product_name", "file_name", "table_name",
                 "bank_name", "event_name", "status_name"):
        assert not suggests(name, PIIType.PERSON_NAME), name
    assert suggests("customer_name", PIIType.PERSON_NAME)


def test_audit_timestamps_are_not_birth_dates() -> None:
    for name in ("created_at", "updated_at", "order_date", "delivery_date",
                 "issue_date", "expiry_date", "joining_date"):
        assert not suggests(name, PIIType.DATE_OF_BIRTH), name


def test_ip_address_keeps_ip_and_drops_generic_address() -> None:
    """`ip_addr` means a network address, not a postal one."""
    types = {h.pii_type for h in classify_name("ip_address")}
    assert PIIType.IP_ADDRESS in types
    assert PIIType.ADDRESS not in types


@pytest.mark.parametrize(
    "name", ["col_17", "data1", "extra_field_3", "field_x", "bucket",
             "quantity", "amount", "status", "priority", "event_type", "id"]
)
def test_meaningless_names_produce_no_hints(name: str) -> None:
    """These are exactly the columns Layer 2 has to carry alone."""
    assert classify_name(name) == []
