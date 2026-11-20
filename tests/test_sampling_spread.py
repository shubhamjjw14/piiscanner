"""A sample must be as large as asked for and spread across the table.

Two ways it once collapsed, both measured on a real baseball database:

* **Removing repeats by a column that repeats.**  The sampler seeked along
  the first column of any index and kept one row per value.  A salaries table
  keyed by year + team + league + player came back as 24 rows of 23,000.
* **Random text positions that sort after every real key.**  Seek points for
  a text key were drawn from all of Unicode, so after the first letter almost
  every random character sorted after ``z``.  Every probe landed at the start
  of a letter's run.

The live tests build their own scratch tables and drop them afterwards.  They
skip when no local MySQL root is available.
"""

from __future__ import annotations

import random
from urllib.parse import urlparse

import pytest

from piiscanner.connectors.keyspace import CharSpan

DATABASE = "piiscanner_sampling_test"
SAMPLE = 2000


# --------------------------------------------------------------------------
# CharSpan: no database needed
# --------------------------------------------------------------------------


def _real_keys(rng: random.Random, n: int) -> list[str]:
    """Keys like a ledger's: fixed prefix, timestamp digits, random letters."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return sorted(
        f"PLG{1700000000 + rng.randrange(10_000_000)}"
        + "".join(rng.choice(letters) for _ in range(5))
        for _ in range(n)
    )


def test_learned_spans_land_points_among_real_keys() -> None:
    """Points drawn from learned spans fall between many different keys, not
    at the start of a handful of runs."""
    import bisect

    rng = random.Random(5)
    keys = _real_keys(rng, 20_000)
    span = CharSpan()
    for key in keys[::50]:
        span.learn(key)
    landed = {bisect.bisect_left(keys, span.between(keys[0], keys[-1], rng))
              for _ in range(500)}
    assert len(landed) > 450


def test_points_stay_between_the_bounds() -> None:
    rng = random.Random(9)
    span = CharSpan()
    span.learn("aardsda01")
    span.learn("zychto01")
    for _ in range(500):
        assert "aardsda01" <= span.between("aardsda01", "zychto01", rng) <= "zychto01"


def test_a_span_never_holds_a_character_the_charset_cannot_store() -> None:
    span = CharSpan(base=0x100)
    span.learn("a中")
    rng = random.Random(1)
    for _ in range(100):
        assert all(ord(ch) <= 0xFF for ch in span.between("a", "a中", rng))


# --------------------------------------------------------------------------
# Live: scratch tables shaped like the ones that collapsed
# --------------------------------------------------------------------------


def _live():
    from tests._mysql import _available

    return _available()


@pytest.fixture(scope="module")
def connector():
    if not _live():
        pytest.skip("needs a local MySQL root can write to")
    from tests._mysql import ADMIN_DSN, _admin

    from piiscanner.connectors.mysql import MySQLConnector

    conn = _admin()
    rng = random.Random(3)
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {DATABASE}")
        cur.execute(f"CREATE DATABASE {DATABASE}")
        cur.execute(f"USE {DATABASE}")
        # Few years lead the key; each year holds thousands of rows.
        cur.execute("CREATE TABLE salaries (yearID SMALLINT NOT NULL, "
                    "teamID CHAR(3) NOT NULL, playerID VARCHAR(9) NOT NULL, "
                    "salary INT, PRIMARY KEY (yearID, teamID, playerID), "
                    "KEY (teamID))")
        rows = [(1985 + y, f"T{t:02d}", f"p{p:05d}", 1000)
                for y in range(5) for t in range(30) for p in range(200)]
        cur.executemany("INSERT INTO salaries VALUES (%s, %s, %s, %s)", rows)
        # A text key with a fixed prefix, timestamp digits and random letters.
        cur.execute("CREATE TABLE ledger (id VARCHAR(20) NOT NULL PRIMARY KEY, "
                    "amount INT)")
        cur.executemany("INSERT INTO ledger VALUES (%s, 1)",
                        [(k,) for k in set(_real_keys(rng, 30_000))])
        # No key the database guarantees unique.
        cur.execute("CREATE TABLE events (user_ref INT NOT NULL, at DATE, "
                    "KEY (user_ref))")
        cur.executemany("INSERT INTO events VALUES (%s, '2020-01-01')",
                        [(u % 40,) for u in range(25_000)])
        cur.execute("ANALYZE TABLE salaries, ledger, events")
        cur.fetchall()

    parsed = urlparse(ADMIN_DSN)
    dsn = (f"mysql://{parsed.username or 'root'}:{parsed.password or ''}"
           f"@{parsed.hostname}:{parsed.port or 3306}/{DATABASE}")
    live = MySQLConnector(dsn, seed=11)
    with live:
        yield live
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {DATABASE}")
    conn.close()


def _sample(connector, table, columns):
    connector._table_row_counts = lambda dataset: {
        "salaries": 30_000, "ledger": 30_000, "events": 25_000}
    return connector._fetch_table_sample(DATABASE, table, columns, SAMPLE)


def test_a_key_led_by_a_few_years_still_gives_a_full_sample(connector) -> None:
    rows = _sample(connector, "salaries", ["yearID", "teamID", "playerID"])
    assert len(rows) == SAMPLE
    assert len({(r["yearID"], r["teamID"], r["playerID"]) for r in rows}) == SAMPLE
    # Spread across the inside of each year.  Seeking on the year alone read
    # the first rows of each year, which all belong to one team.
    assert len({r["yearID"] for r in rows}) == 5
    assert len({r["teamID"] for r in rows}) >= 20


def test_a_prefixed_text_key_gives_a_full_spread_sample(connector) -> None:
    import bisect

    rows = _sample(connector, "ledger", ["id"])
    assert len(rows) == SAMPLE
    # Every tenth of the table is reached, and none holds a third of the sample.
    keys = [r["id"] for r in connector._query(
        f"SELECT id FROM {DATABASE}.ledger ORDER BY id")]
    tenths = [bisect.bisect_left(keys, r["id"]) * 10 // len(keys) for r in rows]
    assert set(tenths) == set(range(10))
    assert max(tenths.count(t) for t in range(10)) < SAMPLE / 3


def test_a_table_with_no_unique_key_is_not_sampled_by_seeking(connector) -> None:
    assert connector._seekable_key(DATABASE, "events") is None
    strategy, _ = connector._choose_strategy(DATABASE, "events", 25_000)
    # No index to seek on, so the table is scanned once with a random filter
    # rather than read from the top.  See plans/spread-sampling.md.
    assert strategy == "spread"
