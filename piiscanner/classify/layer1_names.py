"""Layer 1 -- column name heuristics.

A field name is a hint, never a verdict.  This layer tops out at 0.5
confidence and exists to corroborate (or, via the merge logic, to be refuted
by) what Layer 2 finds in the values.

The critical correctness property is **whole-token matching**.  Substring
matching is the classic bug in name-based classifiers: ``pan`` fires on
``company_panel`` and ``japan_office``, ``dob`` fires on ``dobbs_lane``, and
the customer loses trust in the tool on their first scan.  Everything here
tokenises first and compares whole tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from piiscanner.models import PIIType

#: Layer 1 can never exceed this on its own.
MAX_NAME_CONFIDENCE = 0.5

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_CHARS = re.compile(r"[^A-Za-z0-9]+")
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")


def tokenise(name: str) -> list[str]:
    """Break a field name into lowercase whole tokens.

    Splits on separators, camelCase boundaries, and letter/digit boundaries::

        customer_email   -> ['customer', 'email']
        customerEmail    -> ['customer', 'email']
        IPAddress        -> ['ip', 'address']
        extra_field_3    -> ['extra', 'field', '3']
        company_panel    -> ['company', 'panel']      (no 'pan' token)
        dobbs_lane       -> ['dobbs', 'lane']         (no 'dob' token)
        user.contact.phone -> ['user', 'contact', 'phone']
    """
    if not name:
        return []
    spaced = _CAMEL_BOUNDARY.sub(" ", name)
    spaced = _LETTER_DIGIT.sub(" ", spaced)
    parts = _SPLIT_CHARS.split(spaced)
    return [p.lower() for p in parts if p]


#: token -> PII type.  Every entry is matched as a whole token.
KEYWORDS: dict[str, PIIType] = {
    # -- contact ---------------------------------------------------------
    "email": PIIType.EMAIL,
    "emailid": PIIType.EMAIL,
    "mail": PIIType.EMAIL,
    "eid": PIIType.EMAIL,
    "phone": PIIType.INDIAN_MOBILE,
    "mobile": PIIType.INDIAN_MOBILE,
    "mob": PIIType.INDIAN_MOBILE,
    "msisdn": PIIType.INDIAN_MOBILE,
    "cell": PIIType.INDIAN_MOBILE,
    "contact": PIIType.INDIAN_MOBILE,
    "telephone": PIIType.INDIAN_MOBILE,
    "tel": PIIType.INDIAN_MOBILE,
    "whatsapp": PIIType.INDIAN_MOBILE,
    # -- name ------------------------------------------------------------
    "name": PIIType.PERSON_NAME,
    "fname": PIIType.PERSON_NAME,
    "lname": PIIType.PERSON_NAME,
    "firstname": PIIType.PERSON_NAME,
    "lastname": PIIType.PERSON_NAME,
    "surname": PIIType.PERSON_NAME,
    "fullname": PIIType.PERSON_NAME,
    "middlename": PIIType.PERSON_NAME,
    "nickname": PIIType.PERSON_NAME,
    "guardian": PIIType.PERSON_NAME,
    # -- address ---------------------------------------------------------
    "address": PIIType.ADDRESS,
    "addr": PIIType.ADDRESS,
    "street": PIIType.ADDRESS,
    "locality": PIIType.ADDRESS,
    "landmark": PIIType.ADDRESS,
    "city": PIIType.ADDRESS,
    "town": PIIType.ADDRESS,
    "village": PIIType.ADDRESS,
    "district": PIIType.ADDRESS,
    "state": PIIType.ADDRESS,
    "residence": PIIType.ADDRESS,
    "pincode": PIIType.PINCODE,
    "pin": PIIType.PINCODE,
    "zip": PIIType.PINCODE,
    "zipcode": PIIType.PINCODE,
    "postal": PIIType.PINCODE,
    "postcode": PIIType.PINCODE,
    # -- dates and demographics -------------------------------------------
    "dob": PIIType.DATE_OF_BIRTH,
    "birth": PIIType.DATE_OF_BIRTH,
    "birthdate": PIIType.DATE_OF_BIRTH,
    "birthday": PIIType.DATE_OF_BIRTH,
    "dateofbirth": PIIType.DATE_OF_BIRTH,
    "age": PIIType.AGE,
    "gender": PIIType.GENDER,
    "sex": PIIType.GENDER,
    "religion": PIIType.RELIGION,
    "caste": PIIType.CASTE,
    "community": PIIType.CASTE,
    # -- government identifiers -------------------------------------------
    "aadhaar": PIIType.AADHAAR,
    "aadhar": PIIType.AADHAAR,
    "adhaar": PIIType.AADHAAR,
    "adhar": PIIType.AADHAAR,
    "uid": PIIType.AADHAAR,
    "uidai": PIIType.AADHAAR,
    "pan": PIIType.PAN,
    "pannumber": PIIType.PAN,
    "gstin": PIIType.GSTIN,
    "gst": PIIType.GSTIN,
    "passport": PIIType.INDIAN_PASSPORT,
    "voter": PIIType.VOTER_ID,
    "epic": PIIType.VOTER_ID,
    "vehicle": PIIType.VEHICLE_REG,
    "rego": PIIType.VEHICLE_REG,
    "registration": PIIType.VEHICLE_REG,
    # -- financial ---------------------------------------------------------
    "ifsc": PIIType.IFSC,
    "account": PIIType.BANK_ACCOUNT,
    "acct": PIIType.BANK_ACCOUNT,
    "accno": PIIType.BANK_ACCOUNT,
    "bank": PIIType.BANK_ACCOUNT,
    "iban": PIIType.BANK_ACCOUNT,
    # Bare `card` is deliberately absent.  In payments schemas the token
    # appears mostly in card *attribute* columns (`card_<variant|network|type>`)
    # and in cardholder-name columns, none of which is a card number.  The
    # precise forms below plus the ("card","number") bigram cover the real cases.
    "cardnumber": PIIType.CARD_NUMBER,
    "cardno": PIIType.CARD_NUMBER,
    "creditcard": PIIType.CARD_NUMBER,
    "debitcard": PIIType.CARD_NUMBER,
    "ccnum": PIIType.CARD_NUMBER,
    "upi": PIIType.UPI_VPA,
    "vpa": PIIType.UPI_VPA,
    "salary": PIIType.SALARY,
    "ctc": PIIType.SALARY,
    "income": PIIType.SALARY,
    "compensation": PIIType.SALARY,
    "payslip": PIIType.SALARY,
    # -- technical identifiers ---------------------------------------------
    "ip": PIIType.IP_ADDRESS,
    "ipaddr": PIIType.IP_ADDRESS,
    "ipaddress": PIIType.IP_ADDRESS,
    "device": PIIType.DEVICE_ID,
    "deviceid": PIIType.DEVICE_ID,
    "imei": PIIType.DEVICE_ID,
    "udid": PIIType.DEVICE_ID,
    "advertisingid": PIIType.DEVICE_ID,
    "idfa": PIIType.DEVICE_ID,
    "gaid": PIIType.DEVICE_ID,
    "lat": PIIType.GEO_LOCATION,
    "latitude": PIIType.GEO_LOCATION,
    "lng": PIIType.GEO_LOCATION,
    "lon": PIIType.GEO_LOCATION,
    "longitude": PIIType.GEO_LOCATION,
    "geo": PIIType.GEO_LOCATION,
    "location": PIIType.GEO_LOCATION,
    "coordinates": PIIType.GEO_LOCATION,
    # -- health -------------------------------------------------------------
    "health": PIIType.HEALTH_DATA,
    "medical": PIIType.HEALTH_DATA,
    "diagnosis": PIIType.HEALTH_DATA,
    "disease": PIIType.HEALTH_DATA,
    "ailment": PIIType.HEALTH_DATA,
    "prescription": PIIType.HEALTH_DATA,
    "treatment": PIIType.HEALTH_DATA,
    "blood": PIIType.HEALTH_DATA,
    "bloodgroup": PIIType.HEALTH_DATA,
    "disability": PIIType.HEALTH_DATA,
    # `patient` is deliberately absent: `patient_name` says the *table* is
    # health-related, not that the *column* holds a diagnosis.  Mapping it to
    # HEALTH_DATA produced two findings for one column.
}

#: Bigrams that mean something the individual tokens do not.  Checked against
#: adjacent token pairs before the single-token map.
BIGRAMS: dict[tuple[str, str], PIIType] = {
    ("account", "no"): PIIType.BANK_ACCOUNT,
    ("account", "number"): PIIType.BANK_ACCOUNT,
    ("acc", "no"): PIIType.BANK_ACCOUNT,
    ("date", "of"): PIIType.DATE_OF_BIRTH,
    ("birth", "date"): PIIType.DATE_OF_BIRTH,
    ("customer", "name"): PIIType.PERSON_NAME,
    ("first", "name"): PIIType.PERSON_NAME,
    ("last", "name"): PIIType.PERSON_NAME,
    ("full", "name"): PIIType.PERSON_NAME,
    ("middle", "name"): PIIType.PERSON_NAME,
    ("maiden", "name"): PIIType.PERSON_NAME,
    ("father", "name"): PIIType.PERSON_NAME,
    ("mother", "name"): PIIType.PERSON_NAME,
    ("postal", "code"): PIIType.PINCODE,
    ("zip", "code"): PIIType.PINCODE,
    ("pin", "code"): PIIType.PINCODE,
    ("ip", "address"): PIIType.IP_ADDRESS,
    ("device", "id"): PIIType.DEVICE_ID,
    ("blood", "group"): PIIType.HEALTH_DATA,
    ("voter", "id"): PIIType.VOTER_ID,
    ("pan", "card"): PIIType.PAN,
    ("card", "number"): PIIType.CARD_NUMBER,
    ("card", "no"): PIIType.CARD_NUMBER,
    ("cc", "number"): PIIType.CARD_NUMBER,
    ("aadhaar", "number"): PIIType.AADHAAR,
    ("mobile", "number"): PIIType.INDIAN_MOBILE,
    ("phone", "number"): PIIType.INDIAN_MOBILE,
    ("email", "id"): PIIType.EMAIL,
    ("email", "address"): PIIType.EMAIL,
}

#: Tokens that cancel a match.  ``company_name`` and ``product_name`` are not
#: personal data, and flagging them is how a scanner earns a reputation for
#: noise.
NEGATING_TOKENS: dict[PIIType, frozenset[str]] = {
    PIIType.PERSON_NAME: frozenset(
        {
            "company", "org", "organisation", "organization", "product",
            "brand", "file", "table", "column", "db", "database", "schema",
            "index", "host", "server", "service", "app", "domain", "bank",
            "merchant", "vendor", "supplier", "category", "tag", "role",
            "group", "team", "project", "event", "template", "class",
            "status", "type", "code", "field", "job", "queue", "topic",
            # Payments-domain organisations, not people.  Every one of these
            # produced a false positive on a real fintech schema, where
            # `<org>_name` columns hold card issuers, payment gateways,
            # programme partners and product names rather than people.
            "issuer", "acquirer", "pg", "gateway", "network", "scheme",
            "partner", "program", "offer", "coupon", "reward", "plan",
        }
    ),
    PIIType.ADDRESS: frozenset(
        {
            "ip", "mac", "email", "server", "host", "url", "web", "contract",
            # `sub` covers workflow sub-status columns (`sub_state`), which are
            # not geographic.  The bare `state` token is handled by a
            # value-level lexicon check in the engine instead, since the name
            # alone genuinely cannot tell you which sense is meant.
            "sub",
        }
    ),
    PIIType.DATE_OF_BIRTH: frozenset(
        {"created", "updated", "modified", "deleted", "expiry", "expires",
         "start", "end", "issue", "issued", "valid", "login", "order",
         "delivery", "payment", "invoice", "joined", "joining"}
    ),
    PIIType.BANK_ACCOUNT: frozenset({"count", "counter", "accounting"}),
    PIIType.GEO_LOCATION: frozenset({"warehouse", "store", "office", "branch"}),
}


@dataclass(frozen=True)
class NameHint:
    """One type suggested by a field name."""

    pii_type: PIIType
    confidence: float
    matched_token: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.pii_type.value}<-{self.matched_token}"


def classify_name(field_name: str) -> list[NameHint]:
    """Return every PII type suggested by ``field_name``.

    More than one hint is normal and useful: ``ip_addr`` suggests both
    IP_ADDRESS and ADDRESS, and the merge logic uses Layer 2 to decide which
    one the values support.
    """
    tokens = tokenise(field_name)
    if not tokens:
        return []
    token_set = set(tokens)
    hits: dict[PIIType, str] = {}

    for left, right in zip(tokens, tokens[1:]):
        found = BIGRAMS.get((left, right))
        if found is not None:
            hits.setdefault(found, f"{left}_{right}")

    for token in tokens:
        found = KEYWORDS.get(token)
        if found is not None:
            hits.setdefault(found, token)

    hints: list[NameHint] = []
    for pii_type, token in hits.items():
        negators = NEGATING_TOKENS.get(pii_type, frozenset())
        # A negating token only cancels a match it did not itself create,
        # so `ip_address` keeps its IP_ADDRESS hint while losing ADDRESS.
        if token_set & negators and token not in negators:
            continue
        hints.append(
            NameHint(
                pii_type=pii_type,
                confidence=MAX_NAME_CONFIDENCE,
                matched_token=token,
            )
        )
    return sorted(hints, key=lambda h: h.pii_type.value)


def suggests(field_name: str, pii_type: PIIType) -> bool:
    """True when the field name corroborates ``pii_type``."""
    return any(h.pii_type is pii_type for h in classify_name(field_name))
