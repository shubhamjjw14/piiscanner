"""Interpolating between two key values, for bounded key-range sampling.

Both connectors sample a large collection the same way: read the lowest and
highest key, pick random points between them, and seek to each.  That needs a
value *between* two keys, which is arithmetic for a number and a date but not
for a string -- so a string is treated as a big integer in base ``base``, one
digit per code point.

This lives apart from either connector because it is the same problem in both:
a varchar primary key in MySQL and a string ``_id`` in MongoDB interpolate
identically, and one implementation means one set of edge cases rather than
two that drift.  It also keeps the MongoDB connector importable without
``pymysql`` installed.
"""

from __future__ import annotations

import random

#: One past the highest Unicode code point -- the default base, wide enough for
#: any string a driver can hand back.
CHAR_BASE = 0x110000


def string_to_int(value: str, length: int, base: int = CHAR_BASE) -> int:
    """Treat a string as a base-``base`` integer, right-padded to length."""
    total = 0
    for ch in value.ljust(length, "\x00"):
        total = total * base + ord(ch)
    return total


def int_to_string(value: int, length: int, base: int = CHAR_BASE) -> str:
    """Inverse of :func:`string_to_int`.

    Unpaired surrogate code points can't be handed back to a driver, so one
    landing mid-range is replaced with a plain space -- the result only needs
    to be a valid, in-range seek value, not a realistic one.
    """
    chars = []
    for _ in range(length):
        value, rem = divmod(value, base)
        if 0xD800 <= rem <= 0xDFFF:
            rem = 0x20
        chars.append(chr(rem))
    return "".join(reversed(chars)).rstrip("\x00")


def interpolate_string(
    lo: str, hi: str, rng: random.Random, base: int = CHAR_BASE
) -> str:
    """A random string that lexically sits between ``lo`` and ``hi``.

    This is what makes a varchar/UUID key seekable the same way an integer one
    is: key-range sampling never needs a real value to probe from, only one
    that falls somewhere in the observed range.

    Characters are drawn per position from what real keys use there; see
    :class:`CharSpan`.  With only the two bounds to go on, that is what those
    two strings use.
    """
    span = CharSpan(base)
    span.learn(lo)
    span.learn(hi)
    return span.between(lo, hi, rng)


class CharSpan:
    """The lowest and highest character real keys use at each position.

    Random seek points are built from these, one position at a time.  Drawing
    from the whole of a charset spread nothing.  For keys like
    ``aardsda01``..``zychto01``, almost every random second character sorted
    after ``z``, so every probe landed at the start of a letter's run, and a
    91,000-row table sampled 175 distinct players.  Keys built from a fixed
    prefix, a timestamp and a random tail are worse still, since each part uses
    different characters.

    A span starts from the lowest and highest key and widens as probes read
    real keys back, so later probes spread better than the first.  ``base``
    caps every span, so a probe never holds a character the column's charset
    cannot store.
    """

    def __init__(self, base: int = CHAR_BASE) -> None:
        self._cap = base - 1
        self.low: list[int] = []
        self.high: list[int] = []

    def learn(self, value: str) -> None:
        for i, ch in enumerate(value):
            code = min(ord(ch), self._cap)
            if i == len(self.low):
                self.low.append(code)
                self.high.append(code)
            else:
                self.low[i] = min(self.low[i], code)
                self.high[i] = max(self.high[i], code)

    def between(self, lo: str, hi: str, rng: random.Random) -> str:
        """A random string that sorts between ``lo`` and ``hi``.

        Each position is a digit in its own base: 0 means "the string has
        ended", so a shorter string sorts before a longer one sharing its
        prefix, as it does in the database.
        """
        length = max(len(lo), len(hi), len(self.low))
        if not length:
            return lo
        low = self.low + [0] * (length - len(self.low))
        high = self.high + [0] * (length - len(self.high))
        radix = [h - l + 2 for l, h in zip(low, high)]

        def to_int(value: str) -> int:
            total = 0
            for i in range(length):
                digit = 0
                if i < len(value):
                    digit = min(max(ord(value[i]), low[i]), high[i]) - low[i] + 1
                total = total * radix[i] + digit
            return total

        lo_n, hi_n = to_int(lo), to_int(hi)
        if hi_n <= lo_n:
            return lo
        point = rng.randint(lo_n, hi_n)
        digits = []
        for i in reversed(range(length)):
            point, digit = divmod(point, radix[i])
            digits.append(digit)
        digits.reverse()
        while digits and digits[-1] == 0:
            digits.pop()
        chars = []
        for i, digit in enumerate(digits):
            code = low[i] + max(digit, 1) - 1
            chars.append(" " if 0xD800 <= code <= 0xDFFF else chr(code))
        return "".join(chars)
