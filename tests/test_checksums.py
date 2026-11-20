"""Unit tests for the check-digit algorithms.

These are the tests that matter most in Phase 1: the difference between a
credible scanner and a noise generator is whether AADHAAR/GSTIN/CARD_NUMBER
findings survive a checksum.  Each algorithm is tested three ways -- against
published vectors, by round-tripping generated check digits, and by mutating
single characters to confirm the check actually catches errors.
"""

from __future__ import annotations

import random

import pytest

from piiscanner.classify.checksums import (
    PAN_ENTITY_TYPES,
    VALID_GST_STATE_CODES,
    aadhaar_validate,
    gstin_checksum,
    gstin_validate,
    luhn_checksum,
    luhn_validate,
    pan_validate,
    verhoeff_checksum,
    verhoeff_validate,
)

SEED = 20260802


# --------------------------------------------------------------------------
# Verhoeff
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,expected",
    [
        ("236", "3"),  # classic published vector
        ("12345", "1"),
        ("0", "4"),
        ("142857", "0"),
    ],
)
def test_verhoeff_known_check_digits(base: str, expected: str) -> None:
    assert verhoeff_checksum(base) == expected
    assert verhoeff_validate(base + expected)


def test_verhoeff_roundtrip_on_random_bases() -> None:
    """Generated check digits must validate, for 11-digit Aadhaar-length bases."""
    rng = random.Random(SEED)
    for _ in range(2000):
        base = "".join(rng.choice("0123456789") for _ in range(11))
        number = base + verhoeff_checksum(base)
        assert verhoeff_validate(number), number


def test_verhoeff_detects_every_single_digit_mutation() -> None:
    """Changing any one digit must break validation -- the whole point of D5."""
    rng = random.Random(SEED + 1)
    for _ in range(300):
        base = "".join(rng.choice("0123456789") for _ in range(11))
        number = base + verhoeff_checksum(base)
        for position in range(len(number)):
            for replacement in "0123456789":
                if replacement == number[position]:
                    continue
                mutated = number[:position] + replacement + number[position + 1 :]
                assert not verhoeff_validate(mutated), mutated


def test_verhoeff_detects_adjacent_transpositions() -> None:
    """D5 catches transpositions, which plain mod-10 schemes miss."""
    rng = random.Random(SEED + 2)
    for _ in range(500):
        base = "".join(rng.choice("0123456789") for _ in range(11))
        number = base + verhoeff_checksum(base)
        for i in range(len(number) - 1):
            if number[i] == number[i + 1]:
                continue
            swapped = (
                number[:i] + number[i + 1] + number[i] + number[i + 2 :]
            )
            assert not verhoeff_validate(swapped), swapped


@pytest.mark.parametrize(
    "bad", ["", "abcd", "1234-5678", "12 34", "12345678901a", None]
)
def test_verhoeff_validate_rejects_non_digits(bad: object) -> None:
    assert verhoeff_validate(bad) is False  # type: ignore[arg-type]


def test_verhoeff_checksum_raises_on_bad_input() -> None:
    with pytest.raises(ValueError):
        verhoeff_checksum("12a4")


# --------------------------------------------------------------------------
# Aadhaar (Verhoeff plus structural rules)
# --------------------------------------------------------------------------


def _make_aadhaar(rng: random.Random) -> str:
    base = rng.choice("23456789") + "".join(
        rng.choice("0123456789") for _ in range(10)
    )
    return base + verhoeff_checksum(base)


def test_aadhaar_accepts_generated_numbers_and_common_spacing() -> None:
    rng = random.Random(SEED + 3)
    for _ in range(500):
        number = _make_aadhaar(rng)
        assert aadhaar_validate(number), number
        spaced = f"{number[:4]} {number[4:8]} {number[8:]}"
        assert aadhaar_validate(spaced), spaced


def test_aadhaar_rejects_leading_zero_or_one() -> None:
    """UIDAI never issues numbers starting 0 or 1, even if Verhoeff passes."""
    rng = random.Random(SEED + 4)
    checked = 0
    for _ in range(2000):
        base = rng.choice("01") + "".join(
            rng.choice("0123456789") for _ in range(10)
        )
        number = base + verhoeff_checksum(base)
        assert verhoeff_validate(number)  # the checksum itself is fine...
        assert not aadhaar_validate(number)  # ...but the number is not issuable
        checked += 1
    assert checked == 2000


@pytest.mark.parametrize(
    "bad",
    [
        "234567890123",  # 12 digits, wrong check digit
        "23456789012",  # too short
        "2345678901234",  # too long
        "2345 6789 012A",  # non-digit
        "",
    ],
)
def test_aadhaar_rejects_known_bad(bad: str) -> None:
    assert aadhaar_validate(bad) is False


def test_aadhaar_rejects_random_twelve_digit_numbers_at_expected_rate() -> None:
    """A 12-digit order reference must almost never look like an Aadhaar.

    Verhoeff leaves roughly a 1-in-10 pass rate for arbitrary numbers, which
    is exactly why the decoy suite also relies on field-name context.
    """
    rng = random.Random(SEED + 5)
    trials = 5000
    passes = sum(
        aadhaar_validate(
            rng.choice("23456789")
            + "".join(rng.choice("0123456789") for _ in range(11))
        )
        for _ in range(trials)
    )
    assert 0.05 < passes / trials < 0.15


# --------------------------------------------------------------------------
# Luhn
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    [
        "79927398713",  # classic published vector
        "4539578763621486",
        "4111111111111111",
        "5500 0000 0000 0004",
        "3400-0000-0000-009",
    ],
)
def test_luhn_accepts_known_valid(number: str) -> None:
    assert luhn_validate(number)


@pytest.mark.parametrize(
    "number",
    ["79927398710", "79927398714", "4539578763621487", "4111111111111112"],
)
def test_luhn_rejects_known_invalid(number: str) -> None:
    assert luhn_validate(number) is False


def test_luhn_roundtrip_and_single_digit_mutations() -> None:
    rng = random.Random(SEED + 6)
    for _ in range(500):
        length = rng.choice([12, 15, 18])
        base = "".join(rng.choice("0123456789") for _ in range(length))
        number = base + luhn_checksum(base)
        assert luhn_validate(number), number
        for position in range(len(number)):
            for replacement in "0123456789":
                if replacement == number[position]:
                    continue
                mutated = number[:position] + replacement + number[position + 1 :]
                assert not luhn_validate(mutated), mutated


@pytest.mark.parametrize("bad", ["", "a", "4", "abcd", "12 34 xy"])
def test_luhn_rejects_malformed(bad: str) -> None:
    assert luhn_validate(bad) is False


# --------------------------------------------------------------------------
# PAN
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pan", ["AAPFU0939F", "ABCPD1234E", "AAACT2727Q", "ZZZJZ0000Z"]
)
def test_pan_accepts_valid_structure(pan: str) -> None:
    assert pan_validate(pan)


@pytest.mark.parametrize(
    "pan",
    [
        "AAPFU0939",  # too short
        "AAPFU09399F",  # too long
        "AAPF10939F",  # digit in the letter block
        "AAPFU093AF",  # letter in the digit block
        "AAPFU09394",  # digit in the final position
        "AAAXA1234A",  # X is not an entity type
        "AAA1A1234A",
        "",
    ],
)
def test_pan_rejects_invalid_structure(pan: str) -> None:
    assert pan_validate(pan) is False


def test_pan_entity_type_position_is_enforced() -> None:
    """The fourth character is the holder type; everything else must fail."""
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = f"AAA{ch}A1234A"
        assert pan_validate(candidate) is (ch in PAN_ENTITY_TYPES), candidate


def test_pan_is_case_insensitive() -> None:
    assert pan_validate("aapfu0939f")


# --------------------------------------------------------------------------
# GSTIN
# --------------------------------------------------------------------------


def test_gstin_known_valid() -> None:
    assert gstin_checksum("27AAPFU0939F1Z") == "V"
    assert gstin_validate("27AAPFU0939F1ZV")


def _make_gstin(rng: random.Random) -> str:
    state = f"{rng.randint(1, 38):02d}"
    pan = (
        "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
        + rng.choice(sorted(PAN_ENTITY_TYPES))
        + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + "".join(rng.choice("0123456789") for _ in range(4))
        + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    entity = rng.choice("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    base = f"{state}{pan}{entity}Z"
    return base + gstin_checksum(base)


def test_gstin_roundtrip_on_generated_numbers() -> None:
    rng = random.Random(SEED + 7)
    for _ in range(1000):
        gstin = _make_gstin(rng)
        assert gstin_validate(gstin), gstin


def test_gstin_detects_single_character_mutations() -> None:
    rng = random.Random(SEED + 8)
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for _ in range(200):
        gstin = _make_gstin(rng)
        for position in range(15):
            for replacement in alphabet:
                if replacement == gstin[position]:
                    continue
                mutated = gstin[:position] + replacement + gstin[position + 1 :]
                assert not gstin_validate(mutated), mutated


def test_gstin_rejects_invalid_state_codes() -> None:
    rng = random.Random(SEED + 9)
    gstin = _make_gstin(rng)
    for state in ("00", "39", "99"):
        assert state not in VALID_GST_STATE_CODES
        candidate = state + gstin[2:14]
        candidate += gstin_checksum(candidate)
        # The check digit is recomputed, so only the state rule can reject it.
        assert gstin_validate(candidate) is False, candidate


def test_gstin_rejects_embedded_invalid_pan() -> None:
    """Characters 2-11 must be a structurally valid PAN."""
    base = "27AAPXU0939F1Z"  # X is not a valid entity type
    candidate = base + gstin_checksum(base)
    assert gstin_validate(candidate) is False


def test_gstin_requires_z_at_position_13() -> None:
    rng = random.Random(SEED + 10)
    gstin = _make_gstin(rng)
    base = gstin[:13] + "Y"
    candidate = base + gstin_checksum(base)
    assert gstin_validate(candidate) is False


@pytest.mark.parametrize(
    "bad",
    [
        "27AAPFU0939F1Z",  # 14 chars
        "27AAPFU0939F1ZVV",  # 16 chars
        "27AAPFU0939F1Z!",  # invalid character
        "",
    ],
)
def test_gstin_rejects_malformed(bad: str) -> None:
    assert gstin_validate(bad) is False


def test_gstin_checksum_raises_on_wrong_length() -> None:
    with pytest.raises(ValueError):
        gstin_checksum("27AAPFU0939F1ZV")
    with pytest.raises(ValueError):
        gstin_checksum("27AAPFU")
