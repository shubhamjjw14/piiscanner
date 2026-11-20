"""Deterministic synthetic Indian PII dataset.

This is what the scanner is developed and measured against.  Never point the
scanner at real data.

Everything here is generated from a fixed seed, so the precision/recall number
the harness reports is reproducible run to run.  The dataset is deliberately
messy: it contains clean cases the scanner should obviously find, badly named
columns it can only find by looking at values, and decoys that look like PII
but are not.  The decoys are the important part -- they are the regression
tests for false positives.

``GROUND_TRUTH`` at the bottom is the answer key the harness scores against.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from piiscanner.classify.checksums import (
    PAN_ENTITY_TYPES,
    gstin_checksum,
    luhn_checksum,
    verhoeff_checksum,
)
from piiscanner.models import PIIType

SEED = 20260802
DATABASE_NAME = "piiscanner_testdata"

# --------------------------------------------------------------------------
# Vocabulary -- Indian names, places, and institutions
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Krishna",
    "Ishaan", "Rohan", "Kabir", "Ananya", "Diya", "Aadhya", "Saanvi", "Priya",
    "Meera", "Kavya", "Anika", "Riya", "Ishita", "Rajesh", "Suresh", "Ramesh",
    "Mahesh", "Deepak", "Sunita", "Kavita", "Anjali", "Pooja", "Neha",
    "Abdul", "Fatima", "Imran", "Zoya", "Farhan", "Ayesha", "Gurpreet",
    "Harpreet", "Manjeet", "Simran", "Lakshmi", "Venkatesh", "Srinivas",
    "Padma", "Ravi", "Vijay", "Sanjay", "Amit", "Vikram", "Nitin",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Shah", "Mehta",
    "Joshi", "Desai", "Nair", "Menon", "Pillai", "Iyer", "Iyengar", "Rao",
    "Reddy", "Naidu", "Chatterjee", "Banerjee", "Mukherjee", "Das", "Bose",
    "Ghosh", "Khan", "Ahmed", "Sheikh", "Ansari", "Kaur", "Gill", "Dhillon",
    "Bhattacharya", "Chauhan", "Yadav", "Mishra", "Tiwari", "Pandey",
]

#: Some Indians go by a single name, with no surname.  A name classifier that
#: assumes "two words" will get these wrong, so the dataset includes them.
MONONYMS = [
    "Rajinikanth", "Nagarjuna", "Chiranjeevi", "Karthi", "Trisha", "Sridevi",
    "Dhanush", "Prabhas", "Kajol", "Tabu", "Yesudas", "Ilaiyaraaja",
]

CITIES = [
    ("Mumbai", "Maharashtra", "4000"), ("Pune", "Maharashtra", "4110"),
    ("Bengaluru", "Karnataka", "5600"), ("Mysuru", "Karnataka", "5700"),
    ("Chennai", "Tamil Nadu", "6000"), ("Coimbatore", "Tamil Nadu", "6410"),
    ("Hyderabad", "Telangana", "5000"), ("Kochi", "Kerala", "6820"),
    ("Thiruvananthapuram", "Kerala", "6950"), ("Kolkata", "West Bengal", "7000"),
    ("New Delhi", "Delhi", "1100"), ("Gurugram", "Haryana", "1220"),
    ("Noida", "Uttar Pradesh", "2013"), ("Lucknow", "Uttar Pradesh", "2260"),
    ("Jaipur", "Rajasthan", "3020"), ("Ahmedabad", "Gujarat", "3800"),
    ("Surat", "Gujarat", "3950"), ("Indore", "Madhya Pradesh", "4520"),
    ("Bhopal", "Madhya Pradesh", "4620"), ("Patna", "Bihar", "8000"),
    ("Guwahati", "Assam", "7810"), ("Chandigarh", "Chandigarh", "1600"),
]

STREET_TYPES = ["Road", "Marg", "Street", "Lane", "Nagar", "Colony", "Layout"]
STREET_NAMES = [
    "MG", "Gandhi", "Nehru", "Shivaji", "Tilak", "Ambedkar", "Subhash",
    "Station", "Church", "Market", "Temple", "Park", "Hill", "Lake",
]
LOCALITIES = [
    "Andheri East", "Bandra West", "Koramangala", "Indiranagar", "Adyar",
    "T Nagar", "Banjara Hills", "Salt Lake", "Vasant Kunj", "Sector 62",
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.in", "rediffmail.com", "outlook.com", "hotmail.com",
    "protonmail.com", "zoho.in", "infotech.co.in", "brightretail.in",
]

BANK_CODES = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB", "CNRB"]
UPI_HANDLES = ["okhdfcbank", "okicici", "oksbi", "paytm", "ybl", "upi", "axl"]

#: Two-letter registration codes for Indian states and UTs.
VEHICLE_STATE_CODES = [
    "MH", "KA", "TN", "TS", "KL", "WB", "DL", "HR", "UP", "RJ", "GJ", "MP",
    "BR", "AS", "CH", "PB", "OD", "JH", "CG", "UK", "HP", "GA", "AP",
]

GENDERS = ["Male", "Female", "M", "F", "Other", "Transgender"]

DIAGNOSES = [
    "Type 2 Diabetes Mellitus", "Hypertension Stage 1", "Asthma",
    "Hypothyroidism", "Iron Deficiency Anaemia", "Migraine",
    "Vitamin D Deficiency", "Coronary Artery Disease", "Dengue Fever",
]

#: Solar panel product codes for the `pan` decoy column.  A name-based
#: classifier fires on the column name; the values must stop it.
PANEL_SKUS = [
    "SOLR-450W-MONO", "SOLR-330W-POLY", "PNL-550W-BIFA", "SOLR-120W-FLEX",
    "PNL-400W-MONO", "SOLR-250W-POLY", "PNL-600W-TOPC", "SOLR-180W-THIN",
]

TICKET_TEMPLATES = [
    "Customer reported the app crashing on checkout. Asked for a callback.",
    "Refund processed for order placed last week. Awaiting confirmation.",
    "Delivery delayed due to weather. Informed the customer over chat.",
    "Login issue resolved after password reset. No further action needed.",
    "Customer wants to change the delivery address before dispatch.",
    "Damaged item received. Replacement shipped the same day.",
    "Query about warranty coverage. Shared the policy document.",
    "Duplicate charge reported. Escalated to the payments team.",
]

#: 15% of support notes embed a phone number in free text.  Layer 2 has to
#: find it by searching inside the value, not by matching the whole value.
TICKET_TEMPLATES_WITH_PHONE = [
    "Customer asked for a callback on {phone} after 6pm.",
    "Reached out to the customer at {phone}, no answer. Will retry tomorrow.",
    "Alternate contact number shared by the customer: {phone}.",
    "Please call {phone} to confirm the delivery slot.",
    "Customer's registered number {phone} is not reachable, used email instead.",
]


# --------------------------------------------------------------------------
# PII value generators
# --------------------------------------------------------------------------


def gen_aadhaar(rng: random.Random, spaced: bool = False) -> str:
    """A 12-digit Aadhaar with a valid Verhoeff digit and a 2-9 leading digit."""
    base = rng.choice("23456789") + "".join(
        rng.choice(string.digits) for _ in range(10)
    )
    number = base + verhoeff_checksum(base)
    if spaced:
        return f"{number[:4]} {number[4:8]} {number[8:]}"
    return number


def gen_pan(rng: random.Random) -> str:
    return (
        "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        + rng.choice(sorted(PAN_ENTITY_TYPES))
        + rng.choice(string.ascii_uppercase)
        + "".join(rng.choice(string.digits) for _ in range(4))
        + rng.choice(string.ascii_uppercase)
    )


def gen_gstin(rng: random.Random) -> str:
    state = f"{rng.randint(1, 38):02d}"
    entity = rng.choice("123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    base = f"{state}{gen_pan(rng)}{entity}Z"
    return base + gstin_checksum(base)


def gen_mobile(rng: random.Random, style: str = "plain") -> str:
    number = rng.choice("6789") + "".join(
        rng.choice(string.digits) for _ in range(9)
    )
    if style == "plain":
        return number
    if style == "country":
        return f"+91{number}"
    if style == "spaced":
        return f"+91 {number}"
    if style == "dashed":
        return f"+91-{number}"
    return number


def gen_email(rng: random.Random, first: str, last: str) -> str:
    handle_styles = [
        f"{first}.{last}",
        f"{first}{last}",
        f"{first}_{last}",
        f"{first}{rng.randint(1, 9999)}",
        f"{first[0]}{last}",
    ]
    handle = rng.choice(handle_styles).lower()
    return f"{handle}@{rng.choice(EMAIL_DOMAINS)}"


def gen_ifsc(rng: random.Random) -> str:
    return (
        rng.choice(BANK_CODES)
        + "0"
        + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    )


def gen_bank_account(rng: random.Random) -> str:
    """11-16 digits.

    Deliberately never 10 digits starting 6-9, which would be indistinguishable
    from a mobile number -- that ambiguity is real but belongs in its own test,
    not scattered through the recall suite.
    """
    return "".join(rng.choice(string.digits) for _ in range(rng.randint(11, 16)))


def gen_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "53", "55", "6011", "3528"])
    length = 16 if not prefix.startswith("3") else 15
    body = prefix + "".join(
        rng.choice(string.digits) for _ in range(length - len(prefix) - 1)
    )
    return body + luhn_checksum(body)


def gen_passport(rng: random.Random) -> str:
    return rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + "".join(
        rng.choice(string.digits) for _ in range(7)
    )


def gen_voter_id(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + "".join(
        rng.choice(string.digits) for _ in range(7)
    )


def gen_vehicle_reg(rng: random.Random) -> str:
    return (
        rng.choice(VEHICLE_STATE_CODES)
        + f"{rng.randint(1, 99):02d}"
        + "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(1, 2)))
        + f"{rng.randint(0, 9999):04d}"
    )


def gen_upi(rng: random.Random, first: str) -> str:
    """A VPA whose domain has no dot, so it must not be read as an email."""
    handle = rng.choice(
        [f"{first.lower()}{rng.randint(1, 999)}", gen_mobile(rng)]
    )
    return f"{handle}@{rng.choice(UPI_HANDLES)}"


def gen_pincode(rng: random.Random, prefix: str) -> str:
    return prefix + f"{rng.randint(1, 99):02d}"


def gen_ipv4(rng: random.Random, private: bool = False) -> str:
    if private:
        return rng.choice(
            [
                f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
                f"192.168.{rng.randint(0,255)}.{rng.randint(1,254)}",
                f"172.{rng.randint(16,31)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            ]
        )
    return ".".join(str(rng.randint(1, 223)) for _ in range(4))


def gen_name(rng: random.Random) -> tuple[str, str, str]:
    """Return (first, last, display).  Roughly 8% of records are mononyms."""
    if rng.random() < 0.08:
        name = rng.choice(MONONYMS)
        return name, "", name
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return first, last, f"{first} {last}"


def gen_address(rng: random.Random) -> tuple[str, str, str, str]:
    city, state, pin_prefix = rng.choice(CITIES)
    line = (
        f"{rng.randint(1, 400)}, {rng.choice(STREET_NAMES)} "
        f"{rng.choice(STREET_TYPES)}, {rng.choice(LOCALITIES)}"
    )
    return line, city, state, gen_pincode(rng, pin_prefix)


def gen_dob(rng: random.Random) -> str:
    year = rng.randint(1945, 2006)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def gen_device_id(rng: random.Random) -> str:
    return "-".join(
        "".join(rng.choice(string.hexdigits.lower()[:16]) for _ in range(n))
        for n in (8, 4, 4, 4, 12)
    )


# --------------------------------------------------------------------------
# Decoy generators -- these must NOT be flagged
# --------------------------------------------------------------------------


def gen_order_reference(rng: random.Random) -> str:
    """12 digits that deliberately fail Verhoeff.

    Roughly one random 12-digit number in ten passes Verhoeff by chance, so
    the generator retries until it produces one that fails.  That makes this
    column a clean false-positive test: every single value is a non-Aadhaar.
    """
    from piiscanner.classify.checksums import aadhaar_validate

    while True:
        candidate = rng.choice("123456789") + "".join(
            rng.choice(string.digits) for _ in range(11)
        )
        if not aadhaar_validate(candidate):
            return candidate


def gen_bad_transaction_id(rng: random.Random) -> str:
    """16 digits that deliberately fail Luhn."""
    from piiscanner.classify.checksums import luhn_validate

    while True:
        candidate = "".join(rng.choice(string.digits) for _ in range(16))
        if not luhn_validate(candidate):
            return candidate


# --------------------------------------------------------------------------
# Table definitions
# --------------------------------------------------------------------------


@dataclass
class Table:
    """One synthetic table: its DDL, its rows, and how many to generate."""

    name: str
    ddl: str
    columns: list[str]
    row_count: int
    make_row: Callable[[random.Random, int], tuple[Any, ...]]
    comment: str = ""
    #: Rows per INSERT batch.  Lower for wide tables to stay under packet limits.
    batch_size: int = 2000


def _users_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    first, last, display = gen_name(rng)
    line, city, state, pincode = gen_address(rng)
    return (
        i,
        display,
        first,
        last or None,
        gen_email(rng, first, last or first),
        gen_mobile(rng, rng.choice(["plain", "plain", "plain", "country", "dashed"])),
        gen_dob(rng),
        rng.choice(GENDERS),
        line,
        city,
        state,
        pincode,
        rng.choice([0, 1]),
        f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d} 10:00:00",
    )


def _kyc_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    first, last, _ = gen_name(rng)
    return (
        i,
        rng.randint(1, 5000),
        gen_pan(rng),
        gen_aadhaar(rng, spaced=rng.random() < 0.25),
        gen_gstin(rng) if rng.random() < 0.4 else None,
        gen_ifsc(rng),
        gen_bank_account(rng),
        gen_upi(rng, first),
        gen_passport(rng) if rng.random() < 0.3 else None,
        gen_voter_id(rng) if rng.random() < 0.35 else None,
        rng.choice(["verified", "pending", "rejected"]),
    )


def _weird_schema_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    """Badly named columns holding real PII.

    Layer 1 must find nothing here.  Everything found in this table is a
    Layer 2 win, which is the whole reason value scanning exists.
    """
    first, last, _ = gen_name(rng)
    return (
        i,
        gen_aadhaar(rng),
        gen_email(rng, first, last or first),
        gen_mobile(rng),
        gen_card(rng),
        rng.choice(["alpha", "beta", "gamma", "delta"]),
    )


def _decoys_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    """Things that look like PII to a naive scanner but are not.

    Each column here targets a specific failure mode:
      order_reference  -- 12 digits, fails Verhoeff (regex-only Aadhaar bug)
      transaction_id   -- 16 digits, fails Luhn (regex-only card bug)
      pan              -- column named `pan`, holds solar panel SKUs
      company_panel    -- token `panel` must not fire the `pan` keyword
      japan_office     -- substring `pan` inside a word
      dobbs_lane       -- substring `dob` inside a word
    """
    return (
        i,
        gen_order_reference(rng),
        gen_bad_transaction_id(rng),
        rng.choice(PANEL_SKUS),
        f"PANEL-{rng.randint(100, 999)}",
        rng.choice(["Osaka", "Tokyo", "Kyoto", "Nagoya"]),
        f"{rng.randint(1, 99)} Dobbs Lane",
        rng.randint(1, 500),
        round(rng.uniform(10.0, 9999.0), 2),
    )


def _support_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    if rng.random() < 0.15:
        note = rng.choice(TICKET_TEMPLATES_WITH_PHONE).format(
            phone=gen_mobile(rng, rng.choice(["plain", "country", "spaced"]))
        )
    else:
        note = rng.choice(TICKET_TEMPLATES)
    return (
        i,
        rng.randint(1, 5000),
        rng.choice(["open", "closed", "escalated"]),
        note,
        rng.choice(["low", "medium", "high"]),
    )


def _numeric_pii_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    """PII stored in numeric columns -- extremely common in the wild.

    A scanner that skips numeric types to save time misses Aadhaar entirely.
    """
    return (
        i,
        int(gen_aadhaar(rng)),
        int(gen_mobile(rng)),
        int(gen_pincode(rng, rng.choice(CITIES)[2])),
        rng.randint(18, 85),
        round(rng.uniform(180000, 4500000), 2),
    )


def _events_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    """The large table -- exercises sampling rather than full scans."""
    first, last, _ = gen_name(rng)
    return (
        i,
        rng.randint(1, 5000),
        rng.choice(["login", "logout", "purchase", "view", "search"]),
        gen_email(rng, first, last or first),
        gen_ipv4(rng, private=rng.random() < 0.3),
        gen_device_id(rng),
        f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d} 12:00:00",
    )


def _health_row(rng: random.Random, i: int) -> tuple[Any, ...]:
    _, _, display = gen_name(rng)
    return (
        i,
        rng.randint(1, 5000),
        display,
        rng.choice(DIAGNOSES),
        rng.choice(["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]),
        gen_vehicle_reg(rng),
    )


TABLES: list[Table] = [
    Table(
        name="users",
        comment="Clean cases -- honest column names holding the obvious things",
        ddl="""
            CREATE TABLE users (
                id            INT PRIMARY KEY,
                full_name     VARCHAR(120),
                first_name    VARCHAR(60),
                last_name     VARCHAR(60),
                email         VARCHAR(160),
                mobile        VARCHAR(20),
                dob           DATE,
                gender        VARCHAR(16),
                address_line  VARCHAR(200),
                city          VARCHAR(60),
                state         VARCHAR(60),
                pincode       CHAR(6),
                is_active     BOOLEAN,
                created_at    DATETIME
            )
        """,
        columns=[
            "id", "full_name", "first_name", "last_name", "email", "mobile",
            "dob", "gender", "address_line", "city", "state", "pincode",
            "is_active", "created_at",
        ],
        row_count=5000,
        make_row=_users_row,
    ),
    Table(
        name="kyc",
        comment="Identity documents -- the checksum-bearing types",
        ddl="""
            CREATE TABLE kyc (
                id             INT PRIMARY KEY,
                user_id        INT,
                pan_number     CHAR(10),
                aadhaar_no     VARCHAR(14),
                gstin          CHAR(15),
                ifsc_code      CHAR(11),
                account_no     VARCHAR(20),
                upi_id         VARCHAR(80),
                passport_no    CHAR(8),
                voter_epic_id  CHAR(10),
                kyc_status     VARCHAR(16)
            )
        """,
        columns=[
            "id", "user_id", "pan_number", "aadhaar_no", "gstin", "ifsc_code",
            "account_no", "upi_id", "passport_no", "voter_epic_id", "kyc_status",
        ],
        row_count=3000,
        make_row=_kyc_row,
    ),
    Table(
        name="legacy_import",
        comment="Badly named columns -- Layer 1 blind, Layer 2 must carry it",
        ddl="""
            CREATE TABLE legacy_import (
                id             INT PRIMARY KEY,
                col_17         VARCHAR(20),
                data1          VARCHAR(160),
                extra_field_3  VARCHAR(20),
                field_x        VARCHAR(24),
                bucket         VARCHAR(16)
            )
        """,
        columns=["id", "col_17", "data1", "extra_field_3", "field_x", "bucket"],
        row_count=2000,
        make_row=_weird_schema_row,
    ),
    Table(
        name="decoys",
        comment="False-positive regression tests -- must produce zero findings",
        ddl="""
            CREATE TABLE decoys (
                id                INT PRIMARY KEY,
                order_reference   CHAR(12),
                transaction_id    CHAR(16),
                pan               VARCHAR(32),
                company_panel     VARCHAR(32),
                japan_office      VARCHAR(32),
                dobbs_lane        VARCHAR(64),
                quantity          INT,
                amount            DECIMAL(10,2)
            )
        """,
        columns=[
            "id", "order_reference", "transaction_id", "pan", "company_panel",
            "japan_office", "dobbs_lane", "quantity", "amount",
        ],
        row_count=2000,
        make_row=_decoys_row,
    ),
    Table(
        name="support_tickets",
        comment="Mixed column -- 15% of notes embed a phone number in free text",
        ddl="""
            CREATE TABLE support_tickets (
                id         INT PRIMARY KEY,
                user_id    INT,
                status     VARCHAR(16),
                notes      TEXT,
                priority   VARCHAR(8)
            )
        """,
        columns=["id", "user_id", "status", "notes", "priority"],
        row_count=4000,
        make_row=_support_row,
    ),
    Table(
        name="numeric_pii",
        comment="PII in numeric columns -- Aadhaar as BIGINT, mobile as DECIMAL",
        ddl="""
            CREATE TABLE numeric_pii (
                id              INT PRIMARY KEY,
                uid_number      BIGINT,
                contact_num     DECIMAL(12,0),
                area_pin        INT,
                age             SMALLINT,
                annual_ctc      DECIMAL(12,2)
            )
        """,
        columns=["id", "uid_number", "contact_num", "area_pin", "age", "annual_ctc"],
        row_count=2000,
        make_row=_numeric_pii_row,
    ),
    Table(
        name="events",
        comment="Large table -- 120k rows, forces sampling instead of full scan",
        ddl="""
            CREATE TABLE events (
                id           INT PRIMARY KEY,
                user_id      INT,
                event_type   VARCHAR(24),
                user_email   VARCHAR(160),
                ip_addr      VARCHAR(45),
                device_id    VARCHAR(48),
                occurred_at  DATETIME,
                INDEX idx_events_user (user_id)
            )
        """,
        columns=[
            "id", "user_id", "event_type", "user_email", "ip_addr",
            "device_id", "occurred_at",
        ],
        row_count=120_000,
        make_row=_events_row,
        batch_size=5000,
    ),
    Table(
        name="patient_records",
        comment="Sensitive-category data under the DPDP Act",
        ddl="""
            CREATE TABLE patient_records (
                id            INT PRIMARY KEY,
                user_id       INT,
                patient_name  VARCHAR(120),
                diagnosis     VARCHAR(120),
                blood_group   VARCHAR(4),
                vehicle_no    VARCHAR(16)
            )
        """,
        columns=[
            "id", "user_id", "patient_name", "diagnosis", "blood_group",
            "vehicle_no",
        ],
        row_count=1500,
        make_row=_health_row,
    ),
]


# --------------------------------------------------------------------------
# Ground truth -- the answer key for the precision/recall harness
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Expectation:
    """One column the scanner is expected to flag, and as what."""

    collection: str
    field: str
    pii_type: PIIType
    #: Set when Layer 1 cannot possibly help, so the case measures Layer 2 alone.
    layer2_only: bool = False
    note: str = ""


#: Columns that MUST be found.  Missing one is a recall failure.
EXPECTED: list[Expectation] = [
    # -- users: clean, well-named
    Expectation("users", "full_name", PIIType.PERSON_NAME),
    Expectation("users", "first_name", PIIType.PERSON_NAME),
    Expectation("users", "last_name", PIIType.PERSON_NAME),
    Expectation("users", "email", PIIType.EMAIL),
    Expectation("users", "mobile", PIIType.INDIAN_MOBILE),
    Expectation("users", "dob", PIIType.DATE_OF_BIRTH),
    Expectation("users", "gender", PIIType.GENDER),
    Expectation("users", "address_line", PIIType.ADDRESS),
    Expectation("users", "city", PIIType.ADDRESS),
    Expectation("users", "state", PIIType.ADDRESS),
    Expectation("users", "pincode", PIIType.PINCODE),
    # -- kyc: the checksum types
    Expectation("kyc", "pan_number", PIIType.PAN),
    Expectation("kyc", "aadhaar_no", PIIType.AADHAAR),
    Expectation("kyc", "gstin", PIIType.GSTIN),
    Expectation("kyc", "ifsc_code", PIIType.IFSC),
    Expectation("kyc", "account_no", PIIType.BANK_ACCOUNT),
    Expectation("kyc", "upi_id", PIIType.UPI_VPA),
    Expectation("kyc", "passport_no", PIIType.INDIAN_PASSPORT),
    Expectation("kyc", "voter_epic_id", PIIType.VOTER_ID),
    # -- legacy_import: Layer 2 only, the hardest and most valuable cases
    Expectation(
        "legacy_import", "col_17", PIIType.AADHAAR, layer2_only=True,
        note="Aadhaar in a meaningless column name",
    ),
    Expectation(
        "legacy_import", "data1", PIIType.EMAIL, layer2_only=True,
        note="email in a meaningless column name",
    ),
    Expectation(
        "legacy_import", "extra_field_3", PIIType.INDIAN_MOBILE, layer2_only=True,
        note="mobile in a meaningless column name",
    ),
    Expectation(
        "legacy_import", "field_x", PIIType.CARD_NUMBER, layer2_only=True,
        note="Luhn-valid card number in a meaningless column name",
    ),
    # -- free text
    Expectation(
        "support_tickets", "notes", PIIType.INDIAN_MOBILE,
        note="only ~15% of rows contain a phone number",
    ),
    # -- numeric-typed PII
    Expectation("numeric_pii", "uid_number", PIIType.AADHAAR),
    Expectation("numeric_pii", "contact_num", PIIType.INDIAN_MOBILE),
    Expectation("numeric_pii", "area_pin", PIIType.PINCODE),
    Expectation("numeric_pii", "age", PIIType.AGE),
    Expectation("numeric_pii", "annual_ctc", PIIType.SALARY),
    # -- large table
    Expectation("events", "user_email", PIIType.EMAIL),
    Expectation("events", "ip_addr", PIIType.IP_ADDRESS),
    Expectation("events", "device_id", PIIType.DEVICE_ID),
    # -- sensitive categories
    Expectation("patient_records", "patient_name", PIIType.PERSON_NAME),
    Expectation("patient_records", "diagnosis", PIIType.HEALTH_DATA),
    Expectation("patient_records", "blood_group", PIIType.HEALTH_DATA),
    Expectation("patient_records", "vehicle_no", PIIType.VEHICLE_REG),
]


#: Columns that must produce ZERO findings.  Any finding here is a false
#: positive and fails the regression suite.
FORBIDDEN: list[tuple[str, str, str]] = [
    ("decoys", "order_reference", "12 digits but fails Verhoeff"),
    ("decoys", "transaction_id", "16 digits but fails Luhn"),
    ("decoys", "pan", "column named `pan`, holds solar panel SKUs"),
    ("decoys", "company_panel", "token `panel` must not fire the `pan` keyword"),
    ("decoys", "japan_office", "substring `pan` inside `japan`"),
    ("decoys", "dobbs_lane", "substring `dob` inside `dobbs`"),
    ("decoys", "quantity", "plain integer"),
    ("decoys", "amount", "plain decimal"),
    ("users", "is_active", "boolean, cannot hold PII"),
    ("users", "created_at", "a timestamp, not a birth date"),
    ("kyc", "kyc_status", "low-cardinality status string"),
    ("legacy_import", "bucket", "low-cardinality label"),
    ("support_tickets", "status", "low-cardinality status string"),
    ("support_tickets", "priority", "low-cardinality label"),
    ("events", "event_type", "low-cardinality label"),
]


@dataclass
class GroundTruth:
    """The answer key, keyed for quick lookup by the scoring harness."""

    expected: list[Expectation] = field(default_factory=lambda: list(EXPECTED))
    forbidden: list[tuple[str, str, str]] = field(
        default_factory=lambda: list(FORBIDDEN)
    )

    def expected_pairs(self) -> set[tuple[str, str]]:
        return {(e.collection, e.field) for e in self.expected}

    def forbidden_pairs(self) -> set[tuple[str, str]]:
        return {(c, f) for c, f, _ in self.forbidden}

    def expected_types(self) -> dict[tuple[str, str], PIIType]:
        return {(e.collection, e.field): e.pii_type for e in self.expected}


GROUND_TRUTH = GroundTruth()


# --------------------------------------------------------------------------
# Row generation
# --------------------------------------------------------------------------


def generate_rows(table: Table, seed: int = SEED) -> Iterable[tuple[Any, ...]]:
    """Yield rows for ``table``, deterministically.

    Each table gets its own RNG derived from the table name, so adding or
    resizing one table does not shift the data in any other.
    """
    rng = random.Random(f"{seed}:{table.name}")
    for i in range(1, table.row_count + 1):
        yield table.make_row(rng, i)


# --------------------------------------------------------------------------
# MongoDB documents
# --------------------------------------------------------------------------
#
# The Mongo dataset mirrors the MySQL one and adds what only a document store
# can get wrong: PII several levels deep, PII inside arrays of objects, and
# sparse paths that appear in only a fraction of documents.


@dataclass
class Collection:
    """One synthetic MongoDB collection."""

    name: str
    doc_count: int
    make_doc: Callable[[random.Random, int], dict[str, Any]]
    comment: str = ""


def _mongo_user_doc(rng: random.Random, i: int) -> dict[str, Any]:
    first, last, display = gen_name(rng)
    line, city, state, pincode = gen_address(rng)
    return {
        "_id": i,
        "full_name": display,
        "email": gen_email(rng, first, last or first),
        "mobile": gen_mobile(rng),
        "dob": gen_dob(rng),
        "gender": rng.choice(GENDERS),
        "address": {
            "line": line,
            "city": city,
            "state": state,
            "pincode": pincode,
        },
        "is_active": rng.choice([True, False]),
    }


def _mongo_profile_doc(rng: random.Random, i: int) -> dict[str, Any]:
    """PII three and four levels deep."""
    first, last, display = gen_name(rng)
    doc: dict[str, Any] = {
        "_id": i,
        "account": {
            "tier": rng.choice(["basic", "gold", "platinum"]),
            "owner": {
                "display_name": display,
                "identity": {
                    "aadhaar": gen_aadhaar(rng),
                    "pan": gen_pan(rng),
                },
                "contact": {
                    "email": gen_email(rng, first, last or first),
                    "phone": gen_mobile(rng),
                },
            },
        },
    }
    # A sparse path: present in roughly 8% of documents.  A scanner that
    # reports it with the same confidence as a dense field is lying.
    if rng.random() < 0.08:
        doc["account"]["owner"]["identity"]["passport"] = gen_passport(rng)
    return doc


def _mongo_order_doc(rng: random.Random, i: int) -> dict[str, Any]:
    """PII inside arrays of objects."""
    line, city, state, pincode = gen_address(rng)
    first, _, _ = gen_name(rng)
    return {
        "_id": i,
        "order_reference": gen_order_reference(rng),
        "items": [
            {
                "sku": rng.choice(PANEL_SKUS),
                "qty": rng.randint(1, 5),
            }
            for _ in range(rng.randint(1, 4))
        ],
        "shipping": {
            "address_line": line,
            "city": city,
            "pincode": pincode,
        },
        "recipients": [
            {
                "contact": {
                    "mobile": gen_mobile(rng),
                    "upi": gen_upi(rng, first),
                }
            }
            for _ in range(rng.randint(1, 3))
        ],
    }


def _mongo_legacy_doc(rng: random.Random, i: int) -> dict[str, Any]:
    first, last, _ = gen_name(rng)
    return {
        "_id": i,
        "col_17": gen_aadhaar(rng),
        "data1": gen_email(rng, first, last or first),
        "extra_field_3": gen_mobile(rng),
        "nested": {"blob": {"v2": gen_card(rng)}},
        "bucket": rng.choice(["alpha", "beta", "gamma"]),
    }


def _mongo_decoy_doc(rng: random.Random, i: int) -> dict[str, Any]:
    return {
        "_id": i,
        "order_reference": gen_order_reference(rng),
        "transaction_id": gen_bad_transaction_id(rng),
        "pan": rng.choice(PANEL_SKUS),
        "company_panel": f"PANEL-{rng.randint(100, 999)}",
        "japan_office": rng.choice(["Osaka", "Tokyo", "Kyoto"]),
        "dobbs_lane": f"{rng.randint(1, 99)} Dobbs Lane",
        "quantity": rng.randint(1, 500),
    }


def _mongo_ticket_doc(rng: random.Random, i: int) -> dict[str, Any]:
    if rng.random() < 0.15:
        note = rng.choice(TICKET_TEMPLATES_WITH_PHONE).format(
            phone=gen_mobile(rng, rng.choice(["plain", "country"]))
        )
    else:
        note = rng.choice(TICKET_TEMPLATES)
    return {
        "_id": i,
        "status": rng.choice(["open", "closed", "escalated"]),
        "notes": note,
    }


MONGO_COLLECTIONS: list[Collection] = [
    Collection("users", 5000, _mongo_user_doc, "Flat documents, honest names"),
    Collection("profiles", 3000, _mongo_profile_doc, "PII 3-4 levels deep, plus a sparse path"),
    Collection("orders", 4000, _mongo_order_doc, "PII inside arrays of objects"),
    Collection("legacy_import", 2000, _mongo_legacy_doc, "Badly named paths"),
    Collection("decoys", 2000, _mongo_decoy_doc, "Must produce zero findings"),
    Collection("support_tickets", 4000, _mongo_ticket_doc, "Phone numbers in free text"),
]


#: Answer key for the MongoDB dataset.  Paths use the connector's flattened
#: notation, so `[]` marks an array segment.
MONGO_EXPECTED: list[Expectation] = [
    Expectation("users", "full_name", PIIType.PERSON_NAME),
    Expectation("users", "email", PIIType.EMAIL),
    Expectation("users", "mobile", PIIType.INDIAN_MOBILE),
    Expectation("users", "dob", PIIType.DATE_OF_BIRTH),
    Expectation("users", "gender", PIIType.GENDER),
    Expectation("users", "address.line", PIIType.ADDRESS),
    Expectation("users", "address.city", PIIType.ADDRESS),
    Expectation("users", "address.state", PIIType.ADDRESS),
    Expectation("users", "address.pincode", PIIType.PINCODE),
    # -- deep nesting
    Expectation("profiles", "account.owner.display_name", PIIType.PERSON_NAME),
    Expectation(
        "profiles", "account.owner.identity.aadhaar", PIIType.AADHAAR,
        note="four levels deep",
    ),
    Expectation("profiles", "account.owner.identity.pan", PIIType.PAN),
    Expectation("profiles", "account.owner.contact.email", PIIType.EMAIL),
    Expectation("profiles", "account.owner.contact.phone", PIIType.INDIAN_MOBILE),
    Expectation(
        "profiles", "account.owner.identity.passport", PIIType.INDIAN_PASSPORT,
        note="sparse: present in ~8% of documents",
    ),
    # -- inside arrays
    Expectation("orders", "shipping.address_line", PIIType.ADDRESS),
    Expectation("orders", "shipping.city", PIIType.ADDRESS),
    Expectation("orders", "shipping.pincode", PIIType.PINCODE),
    Expectation(
        "orders", "recipients[].contact.mobile", PIIType.INDIAN_MOBILE,
        note="inside an array of objects",
    ),
    Expectation("orders", "recipients[].contact.upi", PIIType.UPI_VPA),
    # -- Layer 2 only
    Expectation("legacy_import", "col_17", PIIType.AADHAAR, layer2_only=True),
    Expectation("legacy_import", "data1", PIIType.EMAIL, layer2_only=True),
    Expectation("legacy_import", "extra_field_3", PIIType.INDIAN_MOBILE, layer2_only=True),
    Expectation(
        "legacy_import", "nested.blob.v2", PIIType.CARD_NUMBER, layer2_only=True,
        note="card number on a meaningless nested path",
    ),
    # -- free text
    Expectation("support_tickets", "notes", PIIType.INDIAN_MOBILE),
]


MONGO_FORBIDDEN: list[tuple[str, str, str]] = [
    ("decoys", "order_reference", "12 digits but fails Verhoeff"),
    ("decoys", "transaction_id", "16 digits but fails Luhn"),
    ("decoys", "pan", "path named `pan`, holds solar panel SKUs"),
    ("decoys", "company_panel", "token `panel` must not fire the `pan` keyword"),
    ("decoys", "japan_office", "substring `pan` inside `japan`"),
    ("decoys", "dobbs_lane", "substring `dob` inside `dobbs`"),
    ("decoys", "quantity", "plain integer"),
    ("orders", "order_reference", "12 digits but fails Verhoeff"),
    ("orders", "items[].sku", "product codes"),
    ("orders", "items[].qty", "plain integer"),
    ("legacy_import", "bucket", "low-cardinality label"),
    ("support_tickets", "status", "low-cardinality status string"),
    ("users", "is_active", "boolean, cannot hold PII"),
    ("profiles", "account.tier", "low-cardinality label"),
]


MONGO_GROUND_TRUTH = GroundTruth(
    expected=list(MONGO_EXPECTED), forbidden=list(MONGO_FORBIDDEN)
)


def generate_documents(
    collection: Collection, seed: int = SEED
) -> Iterable[dict[str, Any]]:
    """Yield documents for ``collection``, deterministically."""
    rng = random.Random(f"{seed}:mongo:{collection.name}")
    for i in range(1, collection.doc_count + 1):
        yield collection.make_doc(rng, i)
