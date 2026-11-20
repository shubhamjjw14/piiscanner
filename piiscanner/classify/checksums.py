"""Check-digit algorithms.

Regex alone gives an unusable false-positive rate on Indian identifiers -- a
12-digit order reference looks exactly like an Aadhaar number.  These
functions are what let the scanner distinguish them.

Every function here is total: it returns ``False`` for malformed input rather
than raising, so callers can hand it arbitrary sampled values.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Verhoeff (Aadhaar)
# --------------------------------------------------------------------------

#: Multiplication table for the dihedral group D5.
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

#: Permutation table, applied cyclically by digit position.
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

#: Multiplicative inverse in D5.
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _digits(value: str) -> list[int] | None:
    """Parse ``value`` into digits, or ``None`` if it is not all digits."""
    if not value or not value.isdigit():
        return None
    return [int(ch) for ch in value]


def verhoeff_validate(value: str) -> bool:
    """True if ``value`` is a digit string whose trailing Verhoeff digit checks out.

    Iterates the digits in reverse, folding each through the permutation and
    multiplication tables; the number is valid when the accumulator lands
    back on zero.
    """
    digits = _digits(value)
    if digits is None:
        return False
    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][digit]]
    return c == 0


def verhoeff_checksum(base: str) -> str:
    """Return the Verhoeff check digit for ``base`` (the number without it).

    Raises :class:`ValueError` on non-numeric input -- unlike the validators,
    this is a generator used by test-data code, where bad input is a bug.
    """
    digits = _digits(base)
    if digits is None:
        raise ValueError("Verhoeff check digit requires a digit string")
    c = 0
    # Position offset by one: the check digit will occupy index 0 once appended.
    for i, digit in enumerate(reversed(digits)):
        c = _D[c][_P[(i + 1) % 8][digit]]
    return str(_INV[c])


def aadhaar_validate(value: str) -> bool:
    """True if ``value`` is a structurally valid Aadhaar number.

    Accepts the common ``1234 5678 9012`` spacing.  Requires 12 digits, a
    leading digit in 2-9 (UIDAI never issues numbers starting 0 or 1), and a
    passing Verhoeff digit.
    """
    normalised = value.replace(" ", "").replace("-", "") if value else ""
    if len(normalised) != 12 or not normalised.isdigit():
        return False
    if normalised[0] in "01":
        return False
    return verhoeff_validate(normalised)


# --------------------------------------------------------------------------
# Luhn (payment cards)
# --------------------------------------------------------------------------


def luhn_validate(value: str) -> bool:
    """Standard mod-10 check.  Spaces and dashes are ignored."""
    normalised = value.replace(" ", "").replace("-", "") if value else ""
    digits = _digits(normalised)
    if digits is None or len(digits) < 2:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def luhn_checksum(base: str) -> str:
    """Return the Luhn check digit that makes ``base`` valid."""
    digits = _digits(base)
    if digits is None:
        raise ValueError("Luhn check digit requires a digit string")
    total = 0
    # The check digit lands at index 0 of the reversed number, shifting the
    # doubling parity of everything before it.
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((10 - total % 10) % 10)


# --------------------------------------------------------------------------
# GSTIN
# --------------------------------------------------------------------------

_GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_VALUES = {ch: i for i, ch in enumerate(_GSTIN_ALPHABET)}

#: State/UT codes in use for GST registration.
VALID_GST_STATE_CODES = frozenset(f"{n:02d}" for n in range(1, 39))

#: Fourth character of a PAN encodes the holder type.
PAN_ENTITY_TYPES = frozenset("PCHFATBLJG")


def gstin_checksum(base: str) -> str:
    """Return the GSTIN check character for the first 14 characters.

    Each character's value is multiplied by an alternating weight of 1 and 2
    starting at 1; the quotient and remainder of each product against 36 are
    summed, and the check digit completes the total to a multiple of 36.
    """
    if not base or len(base) != 14:
        raise ValueError("GSTIN check digit requires the first 14 characters")
    total = 0
    for i, ch in enumerate(base.upper()):
        value = _GSTIN_VALUES.get(ch)
        if value is None:
            raise ValueError(f"invalid GSTIN character: {ch!r}")
        product = value * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return _GSTIN_ALPHABET[(36 - total % 36) % 36]


def gstin_validate(value: str) -> bool:
    """True if ``value`` is a 15-character GSTIN with a passing check digit.

    Structure only goes so far, so this also checks the state code, the fixed
    ``Z`` at position 13, and that characters 2-11 form a structurally valid
    PAN.
    """
    if not value or len(value) != 15:
        return False
    candidate = value.upper()
    if any(ch not in _GSTIN_VALUES for ch in candidate):
        return False
    if candidate[:2] not in VALID_GST_STATE_CODES:
        return False
    if candidate[13] != "Z":
        return False
    if not pan_validate(candidate[2:12]):
        return False
    try:
        expected = gstin_checksum(candidate[:14])
    except ValueError:
        return False
    return expected == candidate[14]


# --------------------------------------------------------------------------
# PAN (structure only -- no public checksum exists)
# --------------------------------------------------------------------------


def pan_validate(value: str) -> bool:
    """True if ``value`` has valid PAN structure: ``AAAAA9999A``.

    There is no published check digit for PAN, so this is structural only:
    five letters, four digits, one letter, with the fourth character being a
    recognised entity-type code.
    """
    if not value or len(value) != 10:
        return False
    candidate = value.upper()
    if not candidate[:5].isalpha() or not candidate[:5].isascii():
        return False
    if not candidate[5:9].isdigit():
        return False
    if not candidate[9].isalpha() or not candidate[9].isascii():
        return False
    return candidate[3] in PAN_ENTITY_TYPES
