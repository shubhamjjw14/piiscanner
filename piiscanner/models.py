"""Core data model for the PII discovery scanner.

Security invariant enforced here: a ``Finding`` can never carry a raw sampled
value.  Masking is applied by a pydantic validator at construction time, so it
holds regardless of how careless a caller is.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The detection *rules*: patterns, keywords, checksums, lexicons.  Bump when a
#: rule is added, removed, or retuned.
RULESET_VERSION = "1.0.0"

#: The *classifier*: merge logic, confidence ladder, gates, thresholds.  Bump
#: when a decision path changes, even if no rule did.
#:
#: Recorded on every scan and every audit, and load-bearing for ``piiscanner diff``:
#: a findings-level change between two runs on different classifier versions
#: cannot be attributed to the customer's data.
CLASSIFIER_VERSION = "1.1.0"

#: Values shorter than this are masked in full -- with only a couple of
#: characters, "first 2 and last 2" would leak the entire value.
MIN_LENGTH_FOR_PARTIAL_MASK = 6


def mask_value(value: Any) -> str:
    """Return a masked rendering of ``value`` safe to persist.

    Keeps the first two and last two characters and replaces the middle with
    ``X`` (``9876543210`` -> ``98XXXXXX21``).  Anything shorter than
    :data:`MIN_LENGTH_FOR_PARTIAL_MASK` is masked entirely.

    The function is idempotent: masking an already-masked value is a no-op,
    which is what lets us apply it defensively in a validator.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if len(text) < MIN_LENGTH_FOR_PARTIAL_MASK:
        return "X" * len(text)
    return text[:2] + "X" * (len(text) - 4) + text[-2:]


class PIIType(str, Enum):
    """Categories of personal data the scanner can report."""

    AADHAAR = "AADHAAR"
    PAN = "PAN"
    GSTIN = "GSTIN"
    INDIAN_MOBILE = "INDIAN_MOBILE"
    EMAIL = "EMAIL"
    IFSC = "IFSC"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    CARD_NUMBER = "CARD_NUMBER"
    INDIAN_PASSPORT = "INDIAN_PASSPORT"
    VOTER_ID = "VOTER_ID"
    VEHICLE_REG = "VEHICLE_REG"
    UPI_VPA = "UPI_VPA"
    PINCODE = "PINCODE"
    IP_ADDRESS = "IP_ADDRESS"
    IP_ADDRESS_PRIVATE = "IP_ADDRESS_PRIVATE"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PERSON_NAME = "PERSON_NAME"
    ADDRESS = "ADDRESS"
    AGE = "AGE"
    GENDER = "GENDER"
    RELIGION = "RELIGION"
    CASTE = "CASTE"
    HEALTH_DATA = "HEALTH_DATA"
    SALARY = "SALARY"
    DEVICE_ID = "DEVICE_ID"
    GEO_LOCATION = "GEO_LOCATION"


#: Types with no checksum or strong structure of their own.  Layer 2 alone can
#: never push these past 0.5 -- the field name has to corroborate.
WEAK_TYPES: frozenset[PIIType] = frozenset(
    {
        PIIType.INDIAN_PASSPORT,
        PIIType.VOTER_ID,
        PIIType.PINCODE,
        PIIType.AGE,
        PIIType.GENDER,
        PIIType.DATE_OF_BIRTH,
        PIIType.PERSON_NAME,
        PIIType.ADDRESS,
    }
)

#: Types whose detection includes a checksum or check-digit validation.
CHECKSUM_TYPES: frozenset[PIIType] = frozenset(
    {PIIType.AADHAAR, PIIType.GSTIN, PIIType.CARD_NUMBER}
)


class DetectionLayer(str, Enum):
    """Which analysis layer contributed to a finding."""

    NAME = "layer1_names"
    VALUE = "layer2_values"


class MaskedEvidenceModel(BaseModel):
    """Base for any model carrying sampled examples.

    Subclasses declare an ``evidence`` field and inherit the masking validator,
    so the no-raw-values invariant is one implementation rather than one per
    model.  Added when ``FieldRecord`` needed the same guarantee ``Finding``
    had: two copies of a security invariant is one copy too many.

    ``validate_assignment`` is on deliberately.  Without it pydantic runs
    validators at construction only, so ``finding.evidence = [raw]`` stored a
    raw value and every downstream artefact inherited it -- the invariant held
    for the code paths we wrote and not for the next one.  An audit test writing
    to the attribute directly is what surfaced this.
    """

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("evidence", mode="before", check_fields=False)
    @classmethod
    def _mask_evidence(cls, value: Any) -> list[str]:
        """Mask every evidence item, whatever the caller passed in.

        Applied defensively rather than trusting call sites: ``mask_value`` is
        idempotent, so pre-masked input survives unchanged while a raw value
        that slipped through gets masked here.
        """
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            value = [value]
        return [mask_value(item) for item in value][:3]


class FieldSchema(BaseModel):
    """A single scannable field, as reported by a connector.

    For MongoDB, ``name`` is a dotted path (``user.contact.phone``) and
    ``native_type`` is inferred rather than declared.
    """

    dataset: str
    collection: str
    name: str
    native_type: str
    nullable: bool = True
    #: Rows/documents in the parent table or collection, when cheaply known.
    approx_row_count: int | None = None
    #: MongoDB: documents in the sample containing this path.  Sparse fields
    #: are meaningful, so we keep the presence count separate from the sample.
    presence_count: int | None = None
    #: True when the field cannot hold PII (boolean, interval, ...) and should
    #: be skipped before any sampling cost is incurred.
    skip: bool = False
    skip_reason: str | None = None


class FieldRecord(MaskedEvidenceModel):
    """One field the scanner looked at, flagged or not.

    The findings list answers "what did you find". This answers "what did you
    look at", which is what an audit needs: you cannot measure recall from a
    list of hits alone, and you cannot sample unflagged fields you never
    recorded.

    Deliberately carries no confidence, no ``pii_type`` and no verdict beyond
    the ``flagged`` boolean -- see :mod:`piiscanner.audit.worksheet` for why the
    labelling worksheet must stay blind.
    """

    dataset: str
    collection: str
    field: str
    native_type: str

    #: False when the field was excluded before any sampling cost was incurred.
    examined: bool = True
    skip_reason: str | None = None
    #: True when classification produced at least one finding above threshold.
    flagged: bool = False

    approx_row_count: int | None = None
    presence_count: int | None = None

    sample_count: int = 0
    null_rate: float = 0.0
    blank_rate: float = 0.0
    distinct_ratio: float = 0.0

    #: Up to three masked examples, drawn from the sample regardless of whether
    #: any rule matched.  Empty for skipped fields, which are never sampled.
    evidence: list[str] = Field(default_factory=list, max_length=3)

    @property
    def path(self) -> str:
        """Stable identity for this field within one source."""
        return f"{self.dataset}.{self.collection}.{self.field}"


class Finding(MaskedEvidenceModel):
    """One PII verdict for one field."""

    source: str
    dataset: str
    collection: str
    field: str

    pii_type: PIIType
    confidence: float = Field(ge=0.0, le=1.0)

    #: Fraction of sampled non-blank values matching the type's rules.
    match_rate: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)

    #: Share of sampled rows that were strictly ``NULL``.  Narrow on purpose --
    #: read ``blank_rate`` for "how empty is this column".
    null_rate: float = Field(ge=0.0, le=1.0)
    #: Share of sampled rows carrying no content at all: ``NULL`` *or* ``""``.
    #: Always >= ``null_rate``.  A column that is 62% empty string has
    #: ``null_rate == 0.0`` and ``blank_rate == 0.62``; reporting only the
    #: former made such a column look fully populated.
    blank_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Distinct values over *non-blank* values -- a different denominator from
    #: the two rates above.  See ``FieldSample.distinct_ratio``.
    distinct_ratio: float = Field(ge=0.0, le=1.0)

    #: Up to three masked examples.  Never raw values -- see the validator.
    evidence: list[str] = Field(default_factory=list, max_length=3)
    detected_by: list[DetectionLayer] = Field(default_factory=list)

    #: Set when the verdict rests on the field name alone.
    unverified: bool = False
    #: Row count of the parent table, used as the secondary sort key so a
    #: finding on a 40M-row table outranks the same finding on a lookup table.
    approx_row_count: int | None = None
    notes: str | None = None

    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Masking is inherited from MaskedEvidenceModel.

    @property
    def sort_key(self) -> tuple[float, int]:
        """Descending confidence, then descending row count."""
        return (-self.confidence, -(self.approx_row_count or 0))

    @property
    def path(self) -> str:
        """Stable identity for this field within one source."""
        return f"{self.dataset}.{self.collection}.{self.field}"


class ConnectorInfo(BaseModel):
    """Provenance for one scanned source."""

    alias: str
    kind: str
    driver_version: str
    server_version: str | None = None


class ScanResult(BaseModel):
    """Everything one scan run produced."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    ruleset_version: str = RULESET_VERSION
    classifier_version: str = CLASSIFIER_VERSION
    connectors: list[ConnectorInfo] = Field(default_factory=list)

    datasets_scanned: int = 0
    collections_scanned: int = 0
    fields_examined: int = 0
    fields_skipped: int = 0

    findings: list[Finding] = Field(default_factory=list)

    #: Every field looked at, flagged or not.  Populated only when the scan was
    #: run with an inventory requested: it is ~10x the size of the findings list
    #: and only the audit harness needs it.  Empty means "not recorded", which
    #: is why ``fields_examined`` remains an independent count.
    fields: list[FieldRecord] = Field(default_factory=list)

    @property
    def has_inventory(self) -> bool:
        return bool(self.fields)

    def sorted_findings(self) -> list[Finding]:
        """Findings ranked by confidence, then by parent table size."""
        return sorted(self.findings, key=lambda f: f.sort_key)

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.pii_type.value] = counts.get(finding.pii_type.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
