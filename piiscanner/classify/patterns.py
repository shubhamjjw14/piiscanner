"""PII type registry -- the value-level rules Layer 2 applies.

Each rule pairs a regex with an optional validator.  The regex is the cheap
filter; the validator is what makes the finding credible.  For the three types
that carry a check digit (Aadhaar, GSTIN, card numbers) the validator is not
optional -- a match without it must never be reported.

Three flags shape how the merge logic treats a rule:

``checksum``
    The type carries a real check digit.  A high match rate is strong
    evidence, and a low match rate is strong evidence *against*.

``weak``
    The pattern is common enough that values alone cannot justify a confident
    verdict.  Confidence is capped unless the field name agrees.

``requires_name_support``
    The pattern is so generic that reporting it on values alone would produce
    noise.  A 16-digit transaction id is a valid-looking bank account; a
    six-digit primary key is a valid-looking pincode.  These are only reported
    when the field name corroborates.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

from piiscanner.classify.checksums import (
    aadhaar_validate,
    gstin_validate,
    luhn_validate,
    pan_validate,
)
from piiscanner.models import PIIType

#: Birth dates realistically fall in this window.  Outside it, a date column
#: is far more likely to be an audit timestamp than a date of birth.
BIRTH_YEAR_MIN = 1920
BIRTH_YEAR_MAX = 2015

#: Email local parts and domains that are placeholders, not real people.
EMAIL_PLACEHOLDER_PREFIXES = (
    "test@", "noreply@", "no-reply@", "donotreply@", "do-not-reply@",
    "admin@", "postmaster@", "mailer-daemon@", "support@", "info@",
    "example@", "dummy@", "sample@", "null@", "none@", "user@",
)
EMAIL_PLACEHOLDER_DOMAINS = (
    "example.com", "example.org", "example.net", "test.com", "localhost",
    "invalid", "domain.com", "email.com", "mail.com", "yourdomain.com",
)

#: Two-letter vehicle registration codes for Indian states and UTs.
VEHICLE_STATE_CODES = frozenset(
    {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA",
        "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML",
        "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN",
        "TR", "TS", "UK", "UA", "UP", "WB",
    }
)

#: States and union territories.  Used to tell a geographic `state` column from
#: a workflow `state` column -- both names are extremely common, and only the
#: values can distinguish them.
INDIAN_STATE_LEXICON = frozenset(
    {
        "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
        "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
        "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
        "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
        "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
        "uttar pradesh", "uttarakhand", "uttaranchal", "west bengal",
        "andaman and nicobar islands", "chandigarh", "delhi", "new delhi",
        "dadra and nagar haveli", "daman and diu", "jammu and kashmir",
        "ladakh", "lakshadweep", "puducherry", "pondicherry",
    }
)


def looks_like_indian_state(text: str) -> bool:
    return text.strip().lower() in INDIAN_STATE_LEXICON


GENDER_LEXICON = frozenset(
    {
        "m", "f", "male", "female", "man", "woman", "other", "trans",
        "transgender", "non-binary", "nonbinary", "nb", "prefer not to say",
        "unspecified", "o",
    }
)

#: Date formats seen in the wild when dates are stored as text.
_DATE_FORMATS = (
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%Y-%m-%d %H:%M:%S",
)


def normalise(value: Any) -> str:
    """Render a sampled value as the string the rules will see.

    Numeric columns need care: a ``DECIMAL(12,0)`` Aadhaar arrives as
    ``Decimal('234567890123')`` and a float would render it in scientific
    notation, so integral decimals are rendered without a fractional part.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, Decimal):
        return str(value.to_integral_value()) if value == value.to_integral_value() else str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - defensive
            return ""
    return str(value).strip()


def parse_date(value: Any) -> date | None:
    """Best-effort date parse.  Returns ``None`` when the value is not a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = normalise(value)
    if not text or len(text) > 32:
        return None
    # A bare number is an id or an amount, not a date, whatever strptime says.
    if text.replace("-", "").replace("/", "").replace(".", "").isdigit() and len(text) <= 8:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


def _valid_email(text: str) -> bool:
    lowered = text.lower()
    if any(lowered.startswith(p) for p in EMAIL_PLACEHOLDER_PREFIXES):
        return False
    domain = lowered.rsplit("@", 1)[-1]
    if domain in EMAIL_PLACEHOLDER_DOMAINS:
        return False
    # A real address has a dotted domain; this is also what keeps UPI VPAs
    # (handle@okhdfcbank) from being read as email addresses.
    return "." in domain


def _valid_mobile(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    subscriber = digits[-10:]
    if len(subscriber) != 10:
        return False
    # 9999999999 and friends are test data or placeholders, not real numbers.
    return len(set(subscriber)) > 1


def _valid_vehicle_reg(text: str) -> bool:
    return text[:2].upper() in VEHICLE_STATE_CODES


def _valid_upi(text: str) -> bool:
    """A VPA, and specifically not an email address."""
    if "@" not in text:
        return False
    handle, _, domain = text.rpartition("@")
    if "." in domain:  # dotted domain means it is an email
        return False
    return len(handle) >= 2 and domain.isalpha() and len(domain) >= 3


def _valid_card(text: str) -> bool:
    return luhn_validate(text)


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def is_private_ip(text: str) -> bool:
    """True for RFC1918 / loopback / link-local addresses.

    Reported separately: an internal address is far less identifying than a
    public one, and the distinction matters for a DPDP risk assessment.
    """
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_link_local


def _valid_age(text: str) -> bool:
    if not text.isdigit():
        return False
    return 0 <= int(text) <= 120


def _valid_gender(text: str) -> bool:
    return text.strip().lower() in GENDER_LEXICON


def _valid_dob(text: str) -> bool:
    return parse_date(text) is not None


def in_birth_window(value: Any) -> bool:
    """True when a parsed date falls in a plausible birth-year range."""
    parsed = parse_date(value)
    return parsed is not None and BIRTH_YEAR_MIN <= parsed.year <= BIRTH_YEAR_MAX


# --------------------------------------------------------------------------
# Rule registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternRule:
    """One value-level detection rule."""

    pii_type: PIIType
    #: Anchored against the whole value.
    regex: re.Pattern[str]
    validator: Callable[[str], bool] | None = None
    #: Unanchored variant used to find the type inside free text.  ``None``
    #: means the type is only meaningful as a whole-value match.
    free_text_regex: re.Pattern[str] | None = None
    checksum: bool = False
    weak: bool = False
    requires_name_support: bool = False
    #: Longest value worth testing; guards against running regexes over blobs.
    max_length: int = 512

    def matches(self, text: str) -> bool:
        """True if ``text`` matches the rule as a whole value."""
        if not text or len(text) > self.max_length:
            return False
        if not self.regex.fullmatch(text):
            return False
        return self.validator is None or self.validator(text)

    def has_candidate_in_text(self, text: str) -> bool:
        """True if the regex matches inside ``text``, ignoring the validator.

        The denominator for a checksum's pass rate in free text: how many
        values contained something of the right *shape*, before checking
        whether the check digit holds.
        """
        if not self.free_text_regex or not text or len(text) > self.max_length:
            return False
        return self.free_text_regex.search(text) is not None

    def find_in_text(self, text: str) -> str | None:
        """Return the first substring of ``text`` satisfying the rule.

        Used for free-text columns, where the PII is embedded in a sentence
        rather than being the whole value.
        """
        if not self.free_text_regex or not text or len(text) > self.max_length:
            return None
        for match in self.free_text_regex.finditer(text):
            candidate = match.group(0)
            if self.validator is None or self.validator(candidate):
                return candidate
        return None


RULES: tuple[PatternRule, ...] = (
    PatternRule(
        pii_type=PIIType.AADHAAR,
        regex=re.compile(r"\d{4}\s?\d{4}\s?\d{4}"),
        validator=aadhaar_validate,
        free_text_regex=re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"),
        checksum=True,
    ),
    PatternRule(
        pii_type=PIIType.GSTIN,
        regex=re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]", re.I),
        validator=gstin_validate,
        free_text_regex=re.compile(
            r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", re.I
        ),
        checksum=True,
    ),
    PatternRule(
        pii_type=PIIType.CARD_NUMBER,
        regex=re.compile(r"(?:\d[ -]?){12,18}\d"),
        validator=_valid_card,
        free_text_regex=re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
        checksum=True,
    ),
    PatternRule(
        pii_type=PIIType.PAN,
        regex=re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]", re.I),
        validator=pan_validate,
        free_text_regex=re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    ),
    PatternRule(
        pii_type=PIIType.EMAIL,
        regex=re.compile(r"[^@\s,;<>]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}"),
        validator=_valid_email,
        free_text_regex=re.compile(
            r"\b[^@\s,;<>]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}\b"
        ),
    ),
    PatternRule(
        pii_type=PIIType.INDIAN_MOBILE,
        regex=re.compile(r"(?:\+?91[\-\s]?)?[6-9]\d{9}"),
        validator=_valid_mobile,
        # Lookarounds rather than \b: \b does not fire next to '+', and this
        # also stops a 10-digit run inside a 16-digit account number matching.
        free_text_regex=re.compile(r"(?<!\d)(?:\+?91[\-\s]?)?[6-9]\d{9}(?!\d)"),
    ),
    PatternRule(
        pii_type=PIIType.IFSC,
        regex=re.compile(r"[A-Z]{4}0[A-Z0-9]{6}", re.I),
        free_text_regex=re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    ),
    PatternRule(
        pii_type=PIIType.UPI_VPA,
        regex=re.compile(r"[\w.\-]{2,}@[a-zA-Z]{3,}"),
        validator=_valid_upi,
        free_text_regex=re.compile(r"\b[\w.\-]{2,}@[a-zA-Z]{3,}\b"),
    ),
    PatternRule(
        pii_type=PIIType.IP_ADDRESS,
        regex=re.compile(r"[0-9a-fA-F:.]{3,45}"),
        validator=_valid_ip,
        free_text_regex=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    PatternRule(
        pii_type=PIIType.VEHICLE_REG,
        regex=re.compile(r"[A-Z]{2}[\s-]?[0-9]{1,2}[\s-]?[A-Z]{1,3}[\s-]?[0-9]{4}", re.I),
        validator=_valid_vehicle_reg,
        free_text_regex=re.compile(r"\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}\b"),
    ),
    PatternRule(
        pii_type=PIIType.DEVICE_ID,
        regex=re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
            r"|[0-9a-f]{32}",
            re.I,
        ),
        # An opaque hex identifier carries no evidence of *what* it identifies.
        # A trace id, a request id, an MD5 hash and a device id are byte-for-byte
        # indistinguishable, so the field name has to say.  Found on real data:
        # without this, every distributed-tracing correlation-id column in a
        # service schema is a finding.
        requires_name_support=True,
    ),
    # -- weak types: structure alone is not enough -------------------------
    PatternRule(
        pii_type=PIIType.INDIAN_PASSPORT,
        regex=re.compile(r"[A-Z][0-9]{7}", re.I),
        weak=True,
    ),
    PatternRule(
        pii_type=PIIType.VOTER_ID,
        regex=re.compile(r"[A-Z]{3}[0-9]{7}", re.I),
        weak=True,
    ),
    PatternRule(
        pii_type=PIIType.PINCODE,
        regex=re.compile(r"[1-9][0-9]{5}"),
        weak=True,
        requires_name_support=True,
    ),
    PatternRule(
        pii_type=PIIType.BANK_ACCOUNT,
        regex=re.compile(r"\d{9,18}"),
        weak=True,
        requires_name_support=True,
    ),
    PatternRule(
        pii_type=PIIType.AGE,
        regex=re.compile(r"\d{1,3}"),
        validator=_valid_age,
        weak=True,
        requires_name_support=True,
    ),
    PatternRule(
        pii_type=PIIType.GENDER,
        regex=re.compile(r"[A-Za-z \-]{1,20}"),
        validator=_valid_gender,
        weak=True,
        requires_name_support=True,
    ),
    PatternRule(
        pii_type=PIIType.DATE_OF_BIRTH,
        regex=re.compile(r"[0-9A-Za-z,./\- :]{6,32}"),
        validator=_valid_dob,
        weak=True,
        # Deliberately NOT requires_name_support: the engine's `_dob_gate`
        # implements the richer rule, admitting a date column either when the
        # name suggests birth or when the years cluster in a birth range.
        # Setting the flag here would short-circuit that second route.
    ),
)

#: Rules indexed by type, for the merge logic.
RULES_BY_TYPE: dict[PIIType, PatternRule] = {r.pii_type: r for r in RULES}

#: Types Layer 2 cannot test at all.  A Layer 1 hint for one of these can
#: never be refuted by values, so it stays reportable as `unverified`.
NAME_ONLY_TYPES = frozenset(
    {
        PIIType.PERSON_NAME,
        PIIType.ADDRESS,
        PIIType.HEALTH_DATA,
        PIIType.SALARY,
        PIIType.RELIGION,
        PIIType.CASTE,
        PIIType.GEO_LOCATION,
    }
)
