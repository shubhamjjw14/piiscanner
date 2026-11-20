"""Sampling-strategy tests for the MySQL connector.

Three things changed here and all get direct coverage:

*seekable keys aren't just integers*
    ``key_range`` sampling used to require an integer primary key, so any
    table keyed by a UUID/varchar or a date fell back to biased ``head``
    sampling.  ``_seekable_key`` / ``_random_seek_point`` generalise that to
    any column with a total order.

*a string key isn't always in the connection's charset*
    The connection is utf8mb4, so a bound probe parameter arrives as utf8mb4.
    Against a ``latin1`` key column MySQL refuses the comparison outright
    (error 1267) instead of coercing, which aborted a real scan partway
    through.  ``_charset_base`` and the ``CONVERT(... USING ...)`` operand fix
    both halves of that: the comparison resolves, and the generated probe stays
    inside a repertoire the column can actually represent.

*the RNG is no longer pinned to a fixed default*
    Every ``key_range`` probe used to come from ``random.Random(0xC0FFEE)``
    unless a caller passed an explicit seed -- which no real caller does.
    That meant a scheduled re-scan of a large table probed the exact same
    rows forever.  These tests pin the new behaviour: no seed means a fresh
    sequence per instance, an explicit seed stays reproducible.

None of this needs a live database -- ``_query`` is monkeypatched with a
small dispatcher keyed on a distinctive substring of the SQL.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from piiscanner.connectors.mysql import (
    MAX_HEAD_SWEEP,
    MySQLConnector,
    SeekKey,
    _charset_base,
    _int_to_string,
    _interpolate_datetime,
    _interpolate_string,
    _key_kind,
    _random_seek_point,
    _string_to_int,
)

DSN = "mysql://user:pw@127.0.0.1:3306/db"


def _connector(**kwargs) -> MySQLConnector:
    """A connector that never touches the network -- ``_query`` is faked."""
    return MySQLConnector(DSN, **kwargs)


# --------------------------------------------------------------------------
# _key_kind
# --------------------------------------------------------------------------


@pytest.mark.parametrize("data_type,expected", [
    ("int", "int"), ("bigint", "int"), ("smallint", "int"), ("mediumint", "int"),
    ("date", "date"), ("datetime", "date"), ("timestamp", "date"),
    ("varchar", "str"), ("char", "str"),
    ("text", None), ("blob", None), ("json", None), ("float", None),
])
def test_key_kind_classification(data_type: str, expected: str | None) -> None:
    assert _key_kind(data_type) == expected


# --------------------------------------------------------------------------
# String interpolation
# --------------------------------------------------------------------------


def test_string_round_trip() -> None:
    for s in ("RTABCD1234", "cc_9f8e7d", "0", "zzzzzzzz"):
        length = len(s)
        assert _int_to_string(_string_to_int(s, length), length) == s


def test_interpolate_string_lands_in_range() -> None:
    import random

    rng = random.Random(1)
    for _ in range(200):
        mid = _interpolate_string("RT0000000000AA", "RTzzzzzzzzzzZZ", rng)
        assert "RT0000000000AA" <= mid <= "RTzzzzzzzzzzZZ"


def test_interpolate_string_degenerate_range_returns_lo() -> None:
    import random

    rng = random.Random(1)
    assert _interpolate_string("same", "same", rng) == "same"
    assert _interpolate_string("zzz", "aaa", rng) == "zzz"  # lo > hi: hi_n <= lo_n


def test_interpolate_string_is_deterministic_given_a_seed() -> None:
    import random

    a = _interpolate_string("aaa", "zzz", random.Random(42))
    b = _interpolate_string("aaa", "zzz", random.Random(42))
    assert a == b


# --------------------------------------------------------------------------
# Datetime interpolation
# --------------------------------------------------------------------------


def test_interpolate_datetime_lands_in_range() -> None:
    import random

    rng = random.Random(2)
    lo, hi = datetime(2020, 1, 1), datetime(2024, 1, 1)
    for _ in range(50):
        mid = _interpolate_datetime(lo, hi, rng)
        assert lo <= mid <= hi


def test_interpolate_datetime_accepts_plain_dates() -> None:
    import random

    mid = _interpolate_datetime(date(2020, 1, 1), date(2020, 1, 2), random.Random(1))
    assert datetime(2020, 1, 1) <= mid <= datetime(2020, 1, 2)


def test_interpolate_datetime_degenerate_range_returns_lo() -> None:
    import random

    lo = datetime(2020, 1, 1)
    assert _interpolate_datetime(lo, lo, random.Random(1)) == lo


def test_random_seek_point_dispatches_on_kind() -> None:
    import random

    rng = random.Random(3)
    assert isinstance(_random_seek_point(1, 100, "int", rng), int)
    assert isinstance(
        _random_seek_point(datetime(2020, 1, 1), datetime(2021, 1, 1), "date", rng),
        datetime,
    )
    assert isinstance(_random_seek_point("aa", "zz", "str", rng), str)


# --------------------------------------------------------------------------
# _seekable_key -- prefers PRIMARY, skips nullable and unsupported types
# --------------------------------------------------------------------------


def _index_row(index, column, data_type, seq=1, unique=True, nullable=False,
               charset=None, collation=None):
    return {"index_name": index, "non_unique": 0 if unique else 1, "seq": seq,
            "column_name": column, "data_type": data_type,
            "is_nullable": "YES" if nullable else "NO",
            "charset": charset, "collation": collation}


def _seek(monkeypatch, rows):
    conn = _connector()
    monkeypatch.setattr(conn, "_query", lambda sql, params=(): rows)
    return conn._seekable_key("db", "t")


def test_seekable_key_prefers_primary_over_another_unique_index(monkeypatch) -> None:
    assert _seek(monkeypatch, [
        _index_row("PRIMARY", "id", "varchar", charset="utf8mb4",
                   collation="utf8mb4_0900_ai_ci"),
        _index_row("uq_created", "created_at", "datetime"),
    ]) == SeekKey("id", "str", "utf8mb4", "utf8mb4_0900_ai_ci")


def test_seekable_key_skips_unsupported_types_to_find_next_candidate(monkeypatch) -> None:
    assert _seek(monkeypatch, [
        _index_row("PRIMARY", "blob_col", "json"),
        _index_row("uq_seq", "seq", "bigint"),
    ]) == SeekKey("seq", "int")


def test_seekable_key_reports_a_non_utf8_charset(monkeypatch) -> None:
    """A schema that predates a utf8mb4 migration keeps latin1 tables, and the
    probe has to be converted into the key column's charset to compare at all.
    """
    assert _seek(monkeypatch, [
        _index_row("PRIMARY", "id", "varchar", charset="latin1",
                   collation="latin1_swedish_ci"),
    ]) == SeekKey("id", "str", "latin1", "latin1_swedish_ci")


def test_seekable_key_ignores_charset_for_non_string_keys(monkeypatch) -> None:
    assert _seek(monkeypatch, [
        _index_row("PRIMARY", "id", "bigint", charset="latin1",
                   collation="latin1_swedish_ci"),
    ]) == SeekKey("id", "int")


def test_seekable_key_returns_none_when_nothing_usable(monkeypatch) -> None:
    assert _seek(monkeypatch, [_index_row("PRIMARY", "blob_col", "json")]) is None


def test_an_index_that_allows_repeats_is_never_a_sampling_key(monkeypatch) -> None:
    """Sampling removes repeats by key.  Along a column that only *has* an
    index, that threw away all but one row per value: a salaries table came
    back as 24 rows of 23,000."""
    assert _seek(monkeypatch, [
        _index_row("salaries_teamID", "teamID", "varchar", unique=False),
    ]) is None


def test_a_unique_index_with_a_nullable_column_is_not_a_key(monkeypatch) -> None:
    """A UNIQUE index admits any number of NULLs, so it does not name one row."""
    assert _seek(monkeypatch, [
        _index_row("uq_email", "email", "varchar", nullable=True),
    ]) is None


def test_a_key_unique_only_in_combination_brings_all_its_columns(monkeypatch) -> None:
    key = _seek(monkeypatch, [
        _index_row("PRIMARY", "yearID", "smallint", seq=1),
        _index_row("PRIMARY", "teamID", "char", seq=2),
        _index_row("PRIMARY", "playerID", "varchar", seq=3),
        _index_row("salaries_teamID", "teamID", "char", unique=False),
    ])
    assert [k.column for k in key.columns] == ["yearID", "teamID", "playerID"]


def test_a_key_unique_on_its_own_beats_a_combined_primary_key(monkeypatch) -> None:
    key = _seek(monkeypatch, [
        _index_row("PRIMARY", "id", "varchar", seq=1),
        _index_row("PRIMARY", "created_at", "datetime", seq=2),
        _index_row("uq_ref", "ref", "bigint"),
    ])
    assert key == SeekKey("ref", "int")


def test_a_combined_key_with_an_unseekable_column_is_skipped(monkeypatch) -> None:
    """Every column must be seekable for rows to be told apart and reached."""
    assert _seek(monkeypatch, [
        _index_row("PRIMARY", "id", "int", seq=1),
        _index_row("PRIMARY", "amount", "decimal", seq=2),
    ]) is None


# --------------------------------------------------------------------------
# _choose_strategy
# --------------------------------------------------------------------------


def test_small_table_uses_order_by_rand(monkeypatch) -> None:
    conn = _connector()
    monkeypatch.setattr(conn, "_seekable_key", lambda d, t: SeekKey("id", "int"))
    assert conn._choose_strategy("db", "t", row_count=100) == ("order_by_rand", None)


def test_large_table_with_varchar_pk_uses_key_range_not_head(monkeypatch) -> None:
    """The actual regression this block fixes: a UUID/varchar PK on a large
    table used to be indistinguishable from "no key at all"."""
    conn = _connector()
    monkeypatch.setattr(
        conn, "_seekable_key",
        lambda d, t: SeekKey("id", "str", "latin1", "latin1_swedish_ci"),
    )
    assert conn._choose_strategy("db", "t", row_count=1_000_000) == (
        "key_range", SeekKey("id", "str", "latin1", "latin1_swedish_ci"),
    )


def test_large_table_with_date_pk_uses_key_range(monkeypatch) -> None:
    conn = _connector()
    monkeypatch.setattr(conn, "_seekable_key", lambda d, t: SeekKey("created_at", "date"))
    assert conn._choose_strategy("db", "t", row_count=1_000_000) == (
        "key_range", SeekKey("created_at", "date"),
    )


def test_large_table_with_no_seekable_key_is_scanned_for_a_spread_sample(monkeypatch) -> None:
    """Reading the first rows lost MIMIC's chartevents: sorted by patient,
    its first rows were one patient.  See plans/spread-sampling.md."""
    conn = _connector()
    monkeypatch.setattr(conn, "_seekable_key", lambda d, t: None)
    assert conn._choose_strategy("db", "t", row_count=1_000_000) == ("spread", None)


def test_a_keyless_table_too_large_to_scan_reads_its_first_rows(monkeypatch) -> None:
    from piiscanner.connectors.mysql import SPREAD_SAMPLE_MAX_ROWS

    conn = _connector()
    monkeypatch.setattr(conn, "_seekable_key", lambda d, t: None)
    assert conn._choose_strategy(
        "db", "t", row_count=SPREAD_SAMPLE_MAX_ROWS + 1) == ("head", None)


# --------------------------------------------------------------------------
# _key_range_sample -- works the same regardless of key type
# --------------------------------------------------------------------------


def test_key_range_sample_dedupes_across_overlapping_probes(monkeypatch) -> None:
    conn = _connector(seed=7)
    calls = {"n": 0}

    def fake_query(sql, params=()):
        if "MIN(" in sql:
            return [{"lo": "a0000000", "hi": "zzzzzzzz"}]
        calls["n"] += 1
        # Every probe returns the same two rows -- proves de-duplication.
        return [{"id": "row-1", "v": 1}, {"id": "row-2", "v": 2}]

    monkeypatch.setattr(conn, "_query", fake_query)
    rows = conn._key_range_sample(
        "`db`.`t`", "`id`, `v`", SeekKey("id", "str"), sample_size=25
    )
    assert {r["id"] for r in rows} == {"row-1", "row-2"}
    assert calls["n"] > 0


def test_key_range_sample_returns_empty_when_table_is_empty(monkeypatch) -> None:
    conn = _connector()
    monkeypatch.setattr(conn, "_query", lambda sql, params=(): [{"lo": None, "hi": None}])
    assert conn._key_range_sample(
        "`db`.`t`", "`id`", SeekKey("id", "int"), sample_size=25
    ) == []


# --------------------------------------------------------------------------
# _head_sample -- the stateful sweep
# --------------------------------------------------------------------------


def test_head_sample_without_cursor_file_always_reads_from_zero(monkeypatch) -> None:
    conn = _connector()  # no cursor_path
    seen_offsets = []

    def fake_query(sql, params=()):
        seen_offsets.append(sql)
        return []

    monkeypatch.setattr(conn, "_query", fake_query)
    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=10_000_000)
    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=10_000_000)
    assert all("OFFSET 0" in sql for sql in seen_offsets)


def test_head_sample_with_cursor_file_advances_between_runs(tmp_path, monkeypatch) -> None:
    cursor_file = tmp_path / "cursor.json"
    conn = _connector(cursor_path=str(cursor_file))
    seen = []
    monkeypatch.setattr(conn, "_query", lambda sql, params=(): seen.append(sql) or [])

    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=10_000_000)
    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=10_000_000)

    assert "OFFSET 0" in seen[0]
    assert "OFFSET 500" in seen[1]


def test_head_sample_wraps_around_at_max_sweep(tmp_path, monkeypatch) -> None:
    cursor_file = tmp_path / "cursor.json"
    conn = _connector(cursor_path=str(cursor_file))
    seen = []
    monkeypatch.setattr(conn, "_query", lambda sql, params=(): seen.append(sql) or [])

    from piiscanner.connectors.mysql import _save_cursor

    _save_cursor(cursor_file, "db.t", MAX_HEAD_SWEEP)
    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=10_000_000)
    assert "OFFSET 0" in seen[0]


def test_head_sample_does_not_wrap_before_reaching_the_end(tmp_path, monkeypatch) -> None:
    """A partial final batch (rows 900-999 of a 1,000-row table) is still
    useful data -- wrapping early would throw it away for nothing."""
    cursor_file = tmp_path / "cursor.json"
    conn = _connector(cursor_path=str(cursor_file))
    seen = []
    monkeypatch.setattr(conn, "_query", lambda sql, params=(): seen.append(sql) or [])

    from piiscanner.connectors.mysql import _save_cursor

    _save_cursor(cursor_file, "db.t", 900)
    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=1_000)
    assert "OFFSET 900" in seen[0]


def test_head_sample_wraps_at_row_count_if_smaller_than_max_sweep(tmp_path, monkeypatch) -> None:
    """A keyless table with fewer rows than MAX_HEAD_SWEEP shouldn't wait
    for the much larger cap before looping back to the start."""
    cursor_file = tmp_path / "cursor.json"
    conn = _connector(cursor_path=str(cursor_file))
    seen = []
    monkeypatch.setattr(conn, "_query", lambda sql, params=(): seen.append(sql) or [])

    from piiscanner.connectors.mysql import _save_cursor

    _save_cursor(cursor_file, "db.t", 1_000)
    conn._head_sample("db", "t", "`db`.`t`", "`id`", sample_size=500, row_count=1_000)
    assert "OFFSET 0" in seen[0]


def test_cursor_state_is_isolated_per_table(tmp_path) -> None:
    from piiscanner.connectors.mysql import _load_cursor, _save_cursor

    path = tmp_path / "cursor.json"
    _save_cursor(path, "db.a", 500)
    _save_cursor(path, "db.b", 1000)
    state = _load_cursor(path)
    assert state == {"db.a": 500, "db.b": 1000}


# --------------------------------------------------------------------------
# RNG seeding -- the fixed-default-seed bug
# --------------------------------------------------------------------------


def test_no_seed_gives_a_different_sequence_per_instance() -> None:
    a = _connector()._rng.random()
    b = _connector()._rng.random()
    assert a != b


def test_explicit_seed_is_reproducible() -> None:
    a = _connector(seed=99)._rng.random()
    b = _connector(seed=99)._rng.random()
    assert a == b


# --------------------------------------------------------------------------
# charset handling -- the crash that stopped a real scan at table 76 of 265
# --------------------------------------------------------------------------


def test_charset_base_caps_a_single_byte_charset() -> None:
    """A latin1 column can only hold code points up to 0xFF; a probe above
    that would be folded to '?' and drag every probe toward the key space's
    start."""
    assert _charset_base("latin1", "aaa", "zzz") == 0x100


def test_charset_base_allows_all_of_unicode_for_utf8mb4() -> None:
    assert _charset_base("utf8mb4", "aaa", "zzz") == 0x110000


def test_charset_base_caps_utf8mb3_at_the_bmp() -> None:
    assert _charset_base("utf8mb3", "aaa", "zzz") == 0x10000


def test_charset_base_is_floored_by_the_bounds_themselves() -> None:
    """An unrecognised multi-byte charset is assumed single-byte.  Without a
    floor at the largest code point actually observed, the bounds would sit
    outside the base and the interpolation's ordering would break."""
    assert _charset_base("gbk", "a", "中") == 0x4E2E


def test_charset_base_handles_empty_bounds() -> None:
    assert _charset_base("latin1", "", "") == 0x100


def test_interpolated_probe_stays_inside_a_latin1_repertoire() -> None:
    import random

    base = _charset_base("latin1", "a0000000", "zfffffff")
    rng = random.Random(7)
    for _ in range(200):
        probe = _interpolate_string("a0000000", "zfffffff", rng, base)
        assert all(ord(ch) <= 0xFF for ch in probe), probe


def _key_range_conn(monkeypatch, key, calls):
    conn = _connector(seed=1)

    def fake_query(sql, params=()):
        calls.append((sql, params))
        if "MIN(" in sql:
            return [{"lo": "aaaa", "hi": "zzzz"}]
        return [{"id": f"k{len(calls)}"}]

    monkeypatch.setattr(conn, "_query", fake_query)
    conn._key_range_sample("`db`.`t`", "`id`", key, 50)
    return calls


def _probes(calls):
    return [sql for sql, _ in calls if "WHERE" in sql]


def test_key_range_converts_the_probe_into_a_latin1_key_column(monkeypatch) -> None:
    calls: list = []
    _key_range_conn(
        monkeypatch, SeekKey("id", "str", "latin1", "latin1_swedish_ci"), calls
    )
    probes = _probes(calls)
    assert probes, "expected at least one probe"
    assert all("CONVERT(%s USING latin1)" in sql for sql in probes)
    # The column stays bare -- wrapping it instead would kill the index seek.
    assert all("CONVERT(`id`" not in sql for sql in probes)


def test_key_range_collates_the_probe_to_the_key_column(monkeypatch) -> None:
    """The case CONVERT alone does not fix: same charset, different collation.
    Both operands are IMPLICIT so neither yields, and MySQL raises 1267 --
    naming the column's collation makes this side EXPLICIT, which outranks it.
    """
    calls: list = []
    _key_range_conn(
        monkeypatch, SeekKey("id", "str", "utf8mb4", "utf8mb4_unicode_ci"), calls
    )
    probes = _probes(calls)
    assert probes
    assert all(
        "CONVERT(%s USING utf8mb4) COLLATE utf8mb4_unicode_ci" in sql
        for sql in probes
    )


def test_key_range_leaves_an_unnamed_charset_alone(monkeypatch) -> None:
    calls: list = []
    _key_range_conn(monkeypatch, SeekKey("id", "str"), calls)
    probes = _probes(calls)
    assert probes
    assert all("CONVERT" not in sql for sql in probes)
    assert all("`id` >= %s" in sql for sql in probes)


def test_key_range_converts_even_without_a_known_collation(monkeypatch) -> None:
    calls: list = []
    _key_range_conn(monkeypatch, SeekKey("id", "str", "latin1", None), calls)
    probes = _probes(calls)
    assert probes
    assert all("CONVERT(%s USING latin1)" in sql for sql in probes)
    assert all("COLLATE" not in sql for sql in probes)


def test_key_range_refuses_names_that_are_not_bare_identifiers(monkeypatch) -> None:
    """Charset and collation are interpolated into SQL because a placeholder is
    not allowed after USING or COLLATE, so anything unexpected is dropped."""
    calls: list = []
    _key_range_conn(
        monkeypatch, SeekKey("id", "str", "latin1; DROP TABLE x", None), calls
    )
    probes = _probes(calls)
    assert probes
    assert all("DROP TABLE" not in sql for sql in probes)
    assert all("CONVERT" not in sql for sql in probes)

    calls = []
    _key_range_conn(
        monkeypatch, SeekKey("id", "str", "latin1", "x; DROP TABLE y"), calls
    )
    probes = _probes(calls)
    assert probes
    assert all("DROP TABLE" not in sql for sql in probes)
    assert all("COLLATE" not in sql for sql in probes)


def test_int_key_never_gets_a_convert(monkeypatch) -> None:
    conn = _connector(seed=1)
    calls: list = []

    def fake_query(sql, params=()):
        calls.append(sql)
        if "MIN(" in sql:
            return [{"lo": 1, "hi": 1000}]
        return [{"id": len(calls)}]

    monkeypatch.setattr(conn, "_query", fake_query)
    conn._key_range_sample(
        "`db`.`t`", "`id`", SeekKey("id", "int", "latin1", "latin1_swedish_ci"), 50
    )
    assert all("CONVERT" not in sql for sql in calls)


# --------------------------------------------------------------------------
# one unreadable table must not cost the rest of the scan its results
# --------------------------------------------------------------------------


def test_a_failing_table_is_recorded_and_skipped_not_fatal(monkeypatch) -> None:
    from piiscanner.models import FieldSchema

    conn = _connector()
    schemas = [
        FieldSchema(dataset="db", collection="bad", name="a", native_type="varchar"),
        FieldSchema(dataset="db", collection="good", name="b", native_type="varchar"),
    ]
    monkeypatch.setattr(conn, "get_schema", lambda dataset: schemas)

    def fake_rows(dataset, table, columns, sample_size):
        if table == "bad":
            raise RuntimeError("(1267, 'Illegal mix of collations')")
        return [{"b": "value"}]

    monkeypatch.setattr(conn, "_cached_rows", fake_rows)

    samples = list(conn.iter_samples("db", 10))

    # The good table still produced its sample ...
    assert [s.schema.collection for s in samples] == ["good"]
    # ... and the bad one is on the record, not silently absent.
    assert len(conn.sampling_errors) == 1
    dataset, collection, reason = conn.sampling_errors[0]
    assert (dataset, collection) == ("db", "bad")
    assert "1267" in reason


def test_sampling_errors_starts_empty() -> None:
    assert _connector().sampling_errors == []


def test_key_range_selects_the_key_column_even_if_not_asked_for(monkeypatch) -> None:
    """Probes are de-duplicated by key, so the key must come back with each
    row.  A caller asking only for other columns must not break that."""
    conn = _connector(seed=1)
    calls: list = []

    def fake_query(sql, params=()):
        calls.append(sql)
        if "MIN(" in sql:
            return [{"lo": "aaaa", "hi": "zzzz"}]
        return [{"id": f"k{len(calls)}", "v": "x"}]

    monkeypatch.setattr(conn, "_query", fake_query)
    monkeypatch.setattr(conn, "_table_row_counts", lambda d: {"t": 1_000_000})
    monkeypatch.setattr(conn, "_seekable_key", lambda d, t: SeekKey("id", "str"))

    rows = conn._fetch_table_sample("db", "t", ["v"], 25)

    probes = [sql for sql in calls if "WHERE" in sql]
    assert probes
    assert all(sql.startswith("SELECT `id`, `v` FROM") for sql in probes)
    assert rows  # dedup found the key and did not raise


# --------------------------------------------------------------------------
# a dropped connection is not 79 unreadable tables
# --------------------------------------------------------------------------


def _schema_pair():
    from piiscanner.models import FieldSchema

    return [
        FieldSchema(dataset="db", collection="one", name="a", native_type="varchar"),
        FieldSchema(dataset="db", collection="two", name="b", native_type="varchar"),
    ]


def test_connection_loss_reconnects_and_retries_the_table(monkeypatch) -> None:
    import pymysql

    from piiscanner.connectors import base as base_mod

    slept: list = []
    monkeypatch.setattr(base_mod.time, "sleep", slept.append)
    conn = _connector()
    monkeypatch.setattr(conn, "get_schema", lambda dataset: _schema_pair())
    reconnects = []
    monkeypatch.setattr(conn, "close", lambda: None)
    monkeypatch.setattr(conn, "connect", lambda: reconnects.append(1))

    attempts: list = []

    def fake_rows(dataset, table, columns, sample_size):
        attempts.append(table)
        # The socket dies on the first table, once.
        if table == "one" and attempts.count("one") == 1:
            raise pymysql.err.OperationalError(2013, "Lost connection during query")
        return [{"a": "v"}]

    monkeypatch.setattr(conn, "_cached_rows", fake_rows)
    samples = list(conn.iter_samples("db", 10))

    assert reconnects == [1]
    assert slept == [base_mod.RECONNECT_BACKOFF[0]]  # backed off before retrying
    # Retried, and no table was written off as unreadable.
    assert attempts == ["one", "one", "two"]
    assert conn.sampling_errors == []
    assert {s.schema.collection for s in samples} == {"one", "two"}


def test_a_dead_network_ends_the_scan_instead_of_skipping_every_table(
    monkeypatch,
) -> None:
    """The failure mode this guards: reporting one skip per remaining table
    reads as "these tables hold no personal data" when none were read."""
    import pymysql

    from piiscanner.connectors import base as base_mod

    slept: list = []
    monkeypatch.setattr(base_mod.time, "sleep", slept.append)
    conn = _connector()
    monkeypatch.setattr(conn, "get_schema", lambda dataset: _schema_pair())
    monkeypatch.setattr(conn, "close", lambda: None)
    monkeypatch.setattr(conn, "connect", lambda: None)

    attempts: list = []

    def always_dead(dataset, table, columns, sample_size):
        attempts.append(table)
        raise pymysql.err.InterfaceError(0, "")

    monkeypatch.setattr(conn, "_cached_rows", always_dead)

    with pytest.raises(pymysql.err.InterfaceError):
        list(conn.iter_samples("db", 10))

    # Every backoff step was used, then it gave up -- on the FIRST table,
    # without touching the second.
    assert slept == list(base_mod.RECONNECT_BACKOFF)
    assert attempts == ["one"] * (len(base_mod.RECONNECT_BACKOFF) + 1)


def test_a_genuinely_bad_table_is_not_retried(monkeypatch) -> None:
    import pymysql

    conn = _connector()
    monkeypatch.setattr(conn, "get_schema", lambda dataset: _schema_pair())
    reconnects = []
    monkeypatch.setattr(conn, "connect", lambda: reconnects.append(1))

    attempts: list = []

    def fake_rows(dataset, table, columns, sample_size):
        attempts.append(table)
        if table == "one":
            raise pymysql.err.OperationalError(1267, "Illegal mix of collations")
        return [{"b": "v"}]

    monkeypatch.setattr(conn, "_cached_rows", fake_rows)
    samples = list(conn.iter_samples("db", 10))

    assert reconnects == []            # a collation error is not a dead socket
    assert attempts == ["one", "two"]  # tried once, not retried
    assert len(conn.sampling_errors) == 1
    assert {s.schema.collection for s in samples} == {"two"}
