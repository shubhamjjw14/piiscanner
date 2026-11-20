"""MySQL connector.

Read-only by three independent mechanisms, because one is not enough:

1.  The session is put into ``TRANSACTION READ ONLY`` immediately after connect,
    so the server rejects writes.
2.  :meth:`MySQLConnector._query` refuses to execute any statement that is not
    a ``SELECT`` or ``SHOW``, raising :class:`ReadOnlyViolation`.  This catches
    a bug in our own code before it reaches the wire.
3.  The scanner is documented to run as a ``SELECT``-only database user.

Sampling never issues an unbounded scan.  See :meth:`_choose_strategy` for how
the strategy is picked and what each one costs.
"""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, NamedTuple
from urllib.parse import unquote, urlparse

import pymysql
from pymysql.cursors import Cursor

from piiscanner.connectors.base import (
    DEFAULT_SAMPLE_SIZE,
    SAMPLED_FIRST_ROWS_TOO_BIG,
    SAMPLED_FIRST_ROWS_TOO_SLOW,
    SAMPLED_SPREAD,
    Connector,
    FieldSample,
    ReadOnlyViolation,
)
from piiscanner.connectors.keyspace import (
    CHAR_BASE as _CHAR_BASE,
)
from piiscanner.connectors.keyspace import CharSpan
from piiscanner.connectors.keyspace import (
    int_to_string as _int_to_string,
)
from piiscanner.connectors.keyspace import (
    interpolate_string as _interpolate_string,
)
from piiscanner.connectors.keyspace import (
    string_to_int as _string_to_int,
)
from piiscanner.models import ConnectorInfo, FieldSchema

log = logging.getLogger("piiscanner.connectors.mysql")

#: Schemas that belong to the server, not the customer.
SYSTEM_SCHEMAS = frozenset(
    {"information_schema", "performance_schema", "mysql", "sys"}
)

#: Column types that cannot hold personal data.  Numeric types are absent from
#: this list on purpose: Aadhaar and mobile numbers are routinely stored as
#: BIGINT, and skipping numerics is the fastest way to miss them.
UNSCANNABLE_TYPES = frozenset(
    {
        "bit", "binary", "varbinary", "tinyblob", "blob", "mediumblob",
        "longblob", "geometry", "point", "linestring", "polygon",
        "multipoint", "multilinestring", "multipolygon", "geomcollection",
    }
)

#: Below this row count, ``ORDER BY RAND()`` is affordable and gives a true
#: uniform sample.  Above it, we switch to index-driven key sampling.
SMALL_TABLE_THRESHOLD = 20_000

#: Largest table, by the database's own estimate, that a keyless table is
#: scanned to sample.  A scan with a random filter runs at about ten million
#: rows a second on a laptop (11.7M rows: 1.0 s), so this is a few seconds.
#: Above it the first rows are read, and the map says so.
SPREAD_SAMPLE_MAX_ROWS = 50_000_000

#: Scans the spread sample may take when the row estimate proves too high.
SPREAD_SAMPLE_PASSES = 3

#: Longest the spread scan may run.  The same deadline the linkage reads use.
SPREAD_SAMPLE_TIMEOUT_MS = 20_000


#: Rows pulled per probe when key-range sampling.  Small enough that a probe
#: is an index seek plus a short read, large enough to keep round trips down.
PROBE_CHUNK = 25

#: Column types usable as a seek key for ``key_range`` sampling, and how a
#: random point between MIN and MAX is generated for each.  Not just
#: integers: any column with a total order works, because seeking only needs
#: *some* value in range, not a numerically meaningful one.
_INT_KEY_TYPES = frozenset({"int", "bigint", "smallint", "mediumint", "integer"})
_DATE_KEY_TYPES = frozenset({"date", "datetime", "timestamp"})
_STR_KEY_TYPES = frozenset({"varchar", "char"})

#: Character sets whose repertoire is all of Unicode, and those limited to the
#: Basic Multilingual Plane.  Anything else is assumed single-byte.  A key
#: column is not always in the connection's charset -- a schema that predates a
#: utf8mb4 migration keeps latin1 tables -- and a seek value has to stay inside
#: the *column's* repertoire, not the connection's.  A code point the column's
#: charset cannot represent is folded to ``?`` (0x3F) on conversion, which
#: would quietly drag every probe on that table toward the start of the key
#: space instead of spreading them across it.
_UNICODE_CHARSETS = frozenset({"utf8mb4", "utf16", "utf16le", "utf32"})
_BMP_CHARSETS = frozenset({"utf8", "utf8mb3", "ucs2"})

#: Charset and collation names are interpolated into SQL (a placeholder is not
#: allowed after ``USING`` or ``COLLATE``), so they are checked against this
#: first -- they come from ``information_schema``, but nothing unvalidated
#: reaches a query string.
_CHARSET_NAME = re.compile(r"^[a-z0-9_]+$")


class SeekKey(NamedTuple):
    """The column ``key_range`` sampling probes against, and how to compare it.

    ``charset``/``collation`` are ``None`` for non-string keys, and carry the
    key column's own values otherwise -- which are not necessarily the
    connection's.  Comparing across either mismatch is an error in MySQL, not a
    silent coercion, so both have to travel with the column.
    """

    column: str
    kind: str
    charset: str | None = None
    collation: str | None = None
    #: The key's remaining columns, when it is unique only as a combination
    #: (``salaries`` is unique on year + team + league + player, not on year).
    #: Empty for a key that is unique on its own.
    rest: tuple["SeekKey", ...] = ()

    @property
    def columns(self) -> tuple["SeekKey", ...]:
        """Every column of the key, leading one first."""
        return (self, *self.rest)

    def operand(self) -> str:
        """The placeholder a probe value is compared through for this column.

        A bare ``%s`` unless the column has a charset and collation of its own
        to be matched against; see :meth:`MySQLConnector._key_range_sample`.
        """
        if self.kind == "str" and self.charset and _CHARSET_NAME.match(self.charset):
            operand = f"CONVERT(%s USING {self.charset})"
            if self.collation and _CHARSET_NAME.match(self.collation):
                operand += f" COLLATE {self.collation}"
            return operand
        return "%s"


_EPOCH = datetime(1970, 1, 1)

#: How far a keyless ``head`` sample is allowed to sweep before wrapping back
#: to the start. On a table with no seekable index, ``OFFSET`` has nothing to
#: jump through -- it costs a scan of ``offset`` rows -- so letting it grow
#: without bound would eventually recreate the full-scan cost this strategy
#: exists to avoid.  This buys rotating coverage of the table's first
#: ``MAX_HEAD_SWEEP`` rows across repeated runs, not full coverage.
MAX_HEAD_SWEEP = 50_000

#: Characters no identifier may contain: NUL and other control characters.
#: Everything else is legal inside backticks, and real schemas use it --
#: AdventureWorks has a column called `Database Version`, and a Czech
#: accounting schema has tables called `issued-invoices`.
_UNSAFE_IN_IDENTIFIER = re.compile(r"[\x00-\x1f\x7f]")

#: MySQL's own limit on an identifier's length.
_MAX_IDENTIFIER = 64


#: MySQL error codes that mean the socket is gone rather than the statement
#: being wrong.  2006/2013 are the server's own "gone away"/"lost during query";
#: pymysql raises ``InterfaceError(0, '')`` for a write to an already-closed
#: connection, which is how every table after the first failure presents.
_CONNECTION_LOST_CODES = frozenset({0, 2002, 2003, 2006, 2013})


def _key_kind(data_type: str) -> str | None:
    """Classify a column's type for seeking, or ``None`` if it can't be used."""
    if data_type in _INT_KEY_TYPES:
        return "int"
    if data_type in _DATE_KEY_TYPES:
        return "date"
    if data_type in _STR_KEY_TYPES:
        return "str"
    return None


def _charset_base(charset: str | None, lo: str, hi: str) -> int:
    """The base to treat key strings as integers in, for one key column.

    One past the highest code point ``charset`` can hold, so an interpolated
    seek point is always a value that charset can actually represent.  Floored
    by the largest code point seen in the bounds themselves: an unrecognised
    multi-byte charset is assumed single-byte, and without this floor that
    assumption would put ``lo``/``hi`` out of range and break the ordering the
    interpolation depends on.
    """
    if charset in _UNICODE_CHARSETS or charset is None:
        cap = _CHAR_BASE - 1
    elif charset in _BMP_CHARSETS:
        cap = 0xFFFF
    else:
        cap = 0xFF
    observed = max((ord(ch) for ch in lo + hi), default=0)
    return max(cap, observed) + 1


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime(value.year, value.month, value.day)  # a plain date


def _interpolate_datetime(lo: Any, hi: Any, rng: random.Random) -> datetime:
    """A random point between two DATE/DATETIME/TIMESTAMP bounds."""
    lo_dt, hi_dt = _to_datetime(lo), _to_datetime(hi)
    lo_s, hi_s = (lo_dt - _EPOCH).total_seconds(), (hi_dt - _EPOCH).total_seconds()
    if hi_s <= lo_s:
        return lo_dt
    return _EPOCH + timedelta(seconds=rng.uniform(lo_s, hi_s))


def _random_seek_point(
    lo: Any,
    hi: Any,
    kind: str,
    rng: random.Random,
    base: int = _CHAR_BASE,
    span: CharSpan | None = None,
) -> Any:
    """A random value between ``lo`` and ``hi``, in whatever type ``kind`` is.

    For a string key, ``span`` carries the characters real keys were seen to
    use at each position, so points spread the way the keys do.
    """
    if kind == "int":
        return rng.randint(lo, hi)
    if kind == "date":
        return _interpolate_datetime(lo, hi, rng)
    if span is not None:
        span.learn(lo)
        span.learn(hi)
        return span.between(lo, hi, rng)
    return _interpolate_string(lo, hi, rng, base)


def _load_cursor(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cursor(path: Path, key: str, offset: int) -> None:
    state = _load_cursor(path)
    state[key] = offset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def quote_identifier(name: str) -> str:
    """Backtick-quote a schema/table/column name the way MySQL documents.

    Inside backticks any character is literal except the backtick itself,
    which is written twice.  That is the whole escaping rule, and it is what
    keeps a name from ever becoming SQL: ``a`b`` becomes ```a``b```, one
    identifier, however it was meant.

    This used to refuse anything but letters, digits, ``_`` and ``$``.  Safe,
    but real schemas do not stay inside that: one column called `Database
    Version` stopped a whole map of a 71-table database.  Control characters
    and over-long names are still refused -- no schema needs them, and they
    are what an attempt to smuggle something through would look like.
    """
    if (not name or len(name) > _MAX_IDENTIFIER
            or _UNSAFE_IN_IDENTIFIER.search(name)):
        raise ValueError(f"refusing to quote unusual identifier: {name!r}")
    return "`" + name.replace("`", "``") + "`"


def parse_mysql_dsn(dsn: str) -> dict[str, Any]:
    """Parse ``mysql://user:pw@host:port/db`` into pymysql connect kwargs."""
    parsed = urlparse(dsn)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"expected a mysql:// DSN, got {parsed.scheme!r}")
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username) if parsed.username else "root",
        "password": unquote(parsed.password) if parsed.password else "",
        "database": parsed.path.lstrip("/") or None,
    }


class MySQLConnector(Connector):
    """Read-only sampling access to a MySQL server."""

    kind = "mysql"

    def __init__(
        self,
        dsn: str,
        alias: str | None = None,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        seed: int | None = None,
        cursor_path: str | None = None,
    ) -> None:
        super().__init__(dsn, alias)
        self._params = parse_mysql_dsn(dsn)
        self.default_dataset = self._params.get("database")
        self.sample_size = sample_size
        self._conn: pymysql.connections.Connection | None = None
        # No seed given -> random.Random(None) seeds itself from OS entropy, a
        # fresh sequence every run.  A fixed default here was the bug: every
        # key_range probe -- and so every scheduled re-scan of a large table --
        # would land on exactly the same rows, forever.  Pass an explicit seed
        # to get that determinism back (tests do this).
        self._rng = random.Random(seed)
        #: Where a keyless table's ``head`` sweep position is persisted across
        #: runs.  ``None`` (the default) means every run reads the same first
        #: rows -- unchanged behaviour for anyone not opting in.
        self._cursor_path = Path(cursor_path) if cursor_path else None
        #: One table's sampled rows, cached so N columns cost one query.
        self._row_cache: tuple[str, str, int, list[dict[str, Any]]] | None = None
        #: ``{"db.table": how it was sampled}`` for tables with no index, the
        #: only ones where the answer varies.  Read by the linkage map, which
        #: tells the reader when a table was sampled from its first rows.
        self.sample_methods: dict[str, str] = {}

    # -- connection ------------------------------------------------------

    def connect(self) -> None:
        self._conn = pymysql.connect(
            **self._params,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            local_infile=False,
            charset="utf8mb4",
        )
        with self._conn.cursor() as cur:
            # Server-side enforcement: any write in this session now errors.
            cur.execute("SET SESSION TRANSACTION READ ONLY")
        log.info("connected to mysql at %s:%s", self._params["host"], self._params["port"])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._row_cache = None

    def _cursor(self) -> Cursor:
        if self._conn is None:
            raise RuntimeError("connector is not connected; call connect() first")
        return self._conn.cursor()

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only statement.

        Anything that is not a SELECT or SHOW raises
        :class:`ReadOnlyViolation` before it reaches the server.  This is a
        guard against our own bugs, not against a hostile caller.
        """
        first_word = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        if first_word not in {"SELECT", "SHOW"}:
            raise ReadOnlyViolation(
                f"refusing to execute a non-read statement: {first_word or '<empty>'}"
            )
        with self._cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def stream_column(
        self, dataset: str, table: str, column: str, limit: int | None = None
    ) -> Iterator[Any]:
        """Yield every value of one column without holding the table in memory.

        Used by linkage stage 4 to build a parent key's full membership set.
        Deliberately narrow: it takes identifiers rather than SQL, so there is
        no statement for a caller to get wrong, and it builds the only shape it
        will ever need.

        Rows arrive one at a time from an unbuffered cursor.  ``_query`` calls
        ``fetchall``, which for a key column of several million rows would pull
        the whole thing into a Python list before anything could use it -- the
        exact cost this stage is designed to avoid.

        This is a full pass over one column, so it must run against a replica or
        a snapshot.  Stage 3 exists partly to keep the number of columns that
        reach here down to one per value group.

        ``limit`` bounds the read in the query itself.  Abandoning an
        unbuffered cursor part-way does not save anything: closing it drains
        every remaining row from the server first.
        """
        if self._conn is None:
            raise RuntimeError("connector is not connected; call connect() first")
        sql = (
            f"SELECT {quote_identifier(column)} AS v "
            f"FROM {quote_identifier(dataset)}.{quote_identifier(table)}"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        # Unbuffered: rows are read from the server as consumed.  Uses its own
        # cursor class rather than the connector's default DictCursor.
        cursor = self._conn.cursor(pymysql.cursors.SSCursor)
        try:
            cursor.execute(sql)
            for row in cursor:
                yield row[0]
        finally:
            cursor.close()

    def info(self) -> ConnectorInfo:
        version = None
        if self._conn is not None:
            rows = self._query("SELECT VERSION() AS v")
            version = rows[0]["v"] if rows else None
        return ConnectorInfo(
            alias=self.alias,
            kind=self.kind,
            driver_version=f"pymysql {pymysql.__version__}",
            server_version=version,
        )

    # -- schema ----------------------------------------------------------

    def list_datasets(self) -> list[str]:
        rows = self._query(
            "SELECT schema_name AS schema_name FROM information_schema.schemata "
            "ORDER BY schema_name"
        )
        names = [r["schema_name"] for r in rows]
        return [n for n in names if n.lower() not in SYSTEM_SCHEMAS]

    def _table_row_counts(self, dataset: str) -> dict[str, int]:
        """Approximate row counts from the optimiser's statistics.

        These are estimates, not exact counts -- deliberately.  ``SELECT
        COUNT(*)`` on a 40M-row InnoDB table is exactly the full scan we
        promised never to do.
        """
        rows = self._query(
            "SELECT table_name AS table_name, table_rows AS table_rows "
            "FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
            (dataset,),
        )
        return {r["table_name"]: int(r["table_rows"] or 0) for r in rows}

    def get_schema(self, dataset: str) -> list[FieldSchema]:
        counts = self._table_row_counts(dataset)
        rows = self._query(
            "SELECT table_name AS table_name, column_name AS column_name, "
            "data_type AS data_type, column_type AS column_type, "
            "is_nullable AS is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = %s "
            "ORDER BY table_name, ordinal_position",
            (dataset,),
        )
        schemas: list[FieldSchema] = []
        for row in rows:
            table = row["table_name"]
            if table not in counts:  # a view, not a base table
                continue
            data_type = (row["data_type"] or "").lower()
            column_type = (row["column_type"] or "").lower()
            skip, reason = self._skip_reason(data_type, column_type)
            schemas.append(
                FieldSchema(
                    dataset=dataset,
                    collection=table,
                    name=row["column_name"],
                    native_type=column_type or data_type,
                    nullable=(row["is_nullable"] == "YES"),
                    approx_row_count=counts.get(table),
                    skip=skip,
                    skip_reason=reason,
                )
            )
        return schemas

    @staticmethod
    def _skip_reason(data_type: str, column_type: str) -> tuple[bool, str | None]:
        if data_type in UNSCANNABLE_TYPES:
            return True, f"type {data_type} cannot hold PII"
        # MySQL has no real BOOLEAN; it is tinyint(1).  The whole tinyint
        # range is too narrow to hold any identifier we care about.
        if data_type == "tinyint":
            return True, "tinyint range is too narrow to hold PII"
        return False, None

    # -- sampling --------------------------------------------------------

    def _seekable_key(self, dataset: str, table: str) -> SeekKey | None:
        """Return a key the database guarantees is unique, to sample along.

        Only a PRIMARY or UNIQUE index whose columns are all NOT NULL is
        eligible.  Sampling removes overlapping probes by key, and that is only
        sound when the key names exactly one row: removing repeats of a column
        that merely *has* an index once threw away 21 of every 22 rows of a
        salaries table keyed by player and year, leaving 24 rows of 23,000.
        A NULL breaks uniqueness too, since a UNIQUE index admits any number of
        them.

        A key unique only as a combination comes back with its other columns
        in ``rest``.  A key unique on its own is preferred when there is one,
        since a probe along it is a single seek; after that, the primary key.

        Not restricted to integers: any column with a total order works, since
        seeking only needs *some* point between MIN and MAX, not a numerically
        meaningful one.  ``kind`` (``int`` / ``date`` / ``str``) tells
        :func:`_random_seek_point` how to generate that point.  A combined key
        is used up to its first column of a type that cannot be seeked on; the
        leading column must be seekable or the index is skipped.
        """
        rows = self._query(
            "SELECT k.index_name AS index_name, k.non_unique AS non_unique, "
            "       k.seq_in_index AS seq, k.column_name AS column_name, "
            "       c.data_type AS data_type, c.is_nullable AS is_nullable, "
            "       c.character_set_name AS charset, "
            "       c.collation_name AS collation "
            "FROM information_schema.statistics k "
            "JOIN information_schema.columns c "
            "  ON c.table_schema = k.table_schema "
            " AND c.table_name = k.table_name "
            " AND c.column_name = k.column_name "
            "WHERE k.table_schema = %s AND k.table_name = %s "
            "ORDER BY (k.index_name = 'PRIMARY') DESC, k.index_name, k.seq_in_index",
            (dataset, table),
        )
        indexes: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            indexes.setdefault(row["index_name"], []).append(row)

        candidates: list[tuple[int, int, SeekKey]] = []
        for order, (name, columns) in enumerate(indexes.items()):
            if any(int(c["non_unique"]) for c in columns):
                continue
            if any(c["is_nullable"] != "NO" for c in columns):
                continue
            seekable: list[SeekKey] = []
            for c in sorted(columns, key=lambda c: int(c["seq"])):
                kind = _key_kind(c["data_type"])
                if kind is None:
                    break
                seekable.append(
                    SeekKey(c["column_name"], kind)
                    if kind != "str"
                    else SeekKey(c["column_name"], kind, c["charset"], c["collation"])
                )
            if len(seekable) < len(columns):
                continue
            key = seekable[0]._replace(rest=tuple(seekable[1:]))
            candidates.append((len(columns), order, key))
        if not candidates:
            return None
        return min(candidates, key=lambda c: (c[0], c[1]))[2]

    def _choose_strategy(
        self, dataset: str, table: str, row_count: int
    ) -> tuple[str, SeekKey | None]:
        """Pick a sampling strategy and explain the cost.

        ``order_by_rand``  small tables only; a genuine uniform sample, and a
                           scan of a small table is cheap.
        ``key_range``      large tables with a seekable leading index column
                           -- integer, date, or string/UUID.  Seeks to random
                           positions and reads a short run from each.  Cost is
                           bounded by the sample size, never by table size or
                           key type.
        ``spread``         large tables with no index of a usable type.  One
                           scan, keeping each row with a small probability,
                           so the sample comes from the whole table.
        ``head``           last resort: no index and too large to scan.
                           Biased toward storage order -- recorded in
                           :attr:`sample_methods` so the map can say so.
        """
        if row_count <= SMALL_TABLE_THRESHOLD:
            return "order_by_rand", None
        key = self._seekable_key(dataset, table)
        if key is not None:
            return "key_range", key
        if row_count <= SPREAD_SAMPLE_MAX_ROWS:
            return "spread", None
        return "head", None

    def _fetch_table_sample(
        self, dataset: str, table: str, columns: list[str], sample_size: int
    ) -> list[dict[str, Any]]:
        """Sample rows once for the whole table, not once per column."""
        row_count = self._table_row_counts(dataset).get(table, 0)
        strategy, key = self._choose_strategy(dataset, table, row_count)
        qualified = f"{quote_identifier(dataset)}.{quote_identifier(table)}"
        select_list = ", ".join(quote_identifier(c) for c in columns)

        if strategy == "order_by_rand":
            return self._query(
                f"SELECT {select_list} FROM {qualified} "
                f"ORDER BY RAND() LIMIT {int(sample_size)}"
            )

        if strategy == "key_range":
            assert key is not None
            # Probes overlap and are de-duplicated by key, so the key column has
            # to come back with every row.  It normally would -- a seekable type
            # is never one the schema pass skips -- but selecting it explicitly
            # means dedup can't silently depend on that staying true.
            missing = [k.column for k in key.columns if k.column not in columns]
            key_select = ", ".join([*map(quote_identifier, missing), select_list])
            return self._key_range_sample(qualified, key_select, key, sample_size)

        cache_key = f"{dataset}.{table}"
        if strategy == "spread":
            rows = self._spread_sample(qualified, select_list, sample_size, row_count)
            if rows is not None:
                self.sample_methods[cache_key] = SAMPLED_SPREAD
                return rows
            self.sample_methods[cache_key] = SAMPLED_FIRST_ROWS_TOO_SLOW
        else:
            self.sample_methods[cache_key] = SAMPLED_FIRST_ROWS_TOO_BIG

        log.warning(
            "table %s.%s has no seekable key and %d rows; sampling in storage "
            "order (biased), sweeping %s if a cursor file is configured",
            dataset, table, row_count,
            "across runs" if self._cursor_path else "not enabled",
        )
        return self._head_sample(
            dataset, table, qualified, select_list, sample_size, row_count
        )

    def _spread_sample(
        self, qualified: str, select_list: str, sample_size: int, row_count: int
    ) -> list[dict[str, Any]] | None:
        """A sample from the whole of a table that has no index to seek on.

        One scan, keeping each row with probability ``p``, so every part of
        the table is equally likely to be in the sample.  Reading the first
        rows instead is what lost MIMIC's ``chartevents``: the file was
        sorted by patient, its first 2,000 rows were one patient, and the
        patient column looked like a constant.  Measured there, this takes
        0.15 s and sees 98 of the 100 patients.

        ``p`` aims a little over the sample size, and the result is cut to
        size at random rather than by ``LIMIT`` -- a ``LIMIT`` keeps the
        earliest rows and brings the bias back.  The ``LIMIT`` that is there
        is a guard against a row estimate far below the truth.

        ``None`` when the scan did not finish in time; the caller reads the
        first rows instead and records that it did.
        """
        # The row count is the database's estimate, and an estimate can be
        # far too high -- stale statistics on a table that has shrunk.  Then
        # the scan keeps almost nothing.  What came back says how big the
        # table really is, so aim again from that, a few times at most.
        p = min(1.0, 1.2 * sample_size / max(row_count, 1))
        rows: list[dict[str, Any]] = []
        for _ in range(SPREAD_SAMPLE_PASSES):
            try:
                rows = list(self._query(
                    f"SELECT /*+ MAX_EXECUTION_TIME({SPREAD_SAMPLE_TIMEOUT_MS}) */ "
                    f"{select_list} FROM {qualified} WHERE RAND() < {p:.10f} "
                    f"LIMIT {int(sample_size) * 4}"
                ))
            except pymysql.err.MySQLError as exc:
                if self.is_connection_lost(exc):
                    raise
                log.warning("spread sample of %s failed (%s); reading first rows",
                            qualified, str(exc)[:80])
                return None
            if p >= 1.0 or len(rows) * 2 >= sample_size:
                break
            # About len(rows) / p rows in the table; at least one row's worth
            # of chance when nothing came back.
            actual = max(len(rows), 1) / p
            p = min(1.0, 1.2 * sample_size / actual)
        if len(rows) > sample_size:
            rows = self._rng.sample(list(rows), sample_size)
        return list(rows)

    def _head_sample(
        self,
        dataset: str,
        table: str,
        qualified: str,
        select_list: str,
        sample_size: int,
        row_count: int,
    ) -> list[dict[str, Any]]:
        """Last resort for a table with no usable seek key at all.

        With no cursor file configured, this is a plain ``LIMIT`` from the
        start -- every run reads the same rows, unchanged from before.  With
        one configured, each run advances an ``OFFSET`` instead, so a
        scheduled re-scan sweeps further into the table rather than reading
        row 0 forever.  Capped at :data:`MAX_HEAD_SWEEP`: without an index,
        ``OFFSET`` has nothing to jump through, so letting it grow without
        bound would eventually cost as much as the full scan this strategy
        exists to avoid.
        """
        offset = 0
        cache_key = f"{dataset}.{table}"
        if self._cursor_path is not None:
            offset = _load_cursor(self._cursor_path).get(cache_key, 0)
            if offset >= min(row_count, MAX_HEAD_SWEEP):
                offset = 0
        rows = self._query(
            f"SELECT {select_list} FROM {qualified} "
            f"LIMIT {int(sample_size)} OFFSET {int(offset)}"
        )
        if self._cursor_path is not None:
            _save_cursor(self._cursor_path, cache_key, offset + sample_size)
        return rows

    def _key_range_sample(
        self,
        qualified: str,
        select_list: str,
        key: SeekKey,
        sample_size: int,
    ) -> list[dict[str, Any]]:
        """Seek to random key positions and read a short run from each.

        Handles gaps in the key space: each probe reads forward from a random
        key value, so deleted ranges cost a seek rather than an empty result.
        Rows are de-duplicated by key because probes can overlap.  ``key.kind``
        decides how the random seek value is generated (see
        :func:`_random_seek_point`) but nothing else here changes by type --
        an index seek costs the same whether the key is a number or a UUID.

        A string key is compared through ``CONVERT(%s USING <charset>) COLLATE
        <collation>``, because MySQL rejects a mismatched comparison rather than
        coercing one side (error 1267, "illegal mix of collations").  Both
        halves are needed and neither is redundant:

        ``CONVERT`` settles the *charset*.  The connection is utf8mb4, so a
        bound parameter arrives as utf8mb4; against a latin1 key column MySQL
        refuses, since narrowing utf8mb4 could drop characters.

        ``COLLATE`` settles the *collation*, which ``CONVERT`` leaves at the
        charset's default.  Two utf8mb4 operands still fail if one is
        ``utf8mb4_unicode_ci`` and the other ``utf8mb4_0900_ai_ci`` -- both are
        IMPLICIT, so neither yields.  Naming the column's own collation makes
        this side EXPLICIT, which outranks IMPLICIT and settles it.

        Both wrap the *parameter*, never the column.  Wrapping the column would
        make it unindexable and turn every probe into the full scan this
        strategy exists to avoid.

        A key unique only as a combination is probed one column at a time; see
        :meth:`_combined_key_sample`.
        """
        if key.rest:
            return self._combined_key_sample(qualified, select_list, key, sample_size)
        quoted_key = quote_identifier(key.column)
        operand = key.operand()
        bounds = self._query(
            f"SELECT MIN({quoted_key}) AS lo, MAX({quoted_key}) AS hi FROM {qualified}"
        )
        lo, hi = (bounds[0]["lo"], bounds[0]["hi"]) if bounds else (None, None)
        if lo is None or hi is None:
            return []
        if hi <= lo:
            return self._query(
                f"SELECT {select_list} FROM {qualified} LIMIT {int(sample_size)}"
            )

        base = _charset_base(key.charset, lo, hi) if key.kind == "str" else _CHAR_BASE

        span = CharSpan(base) if key.kind == "str" else None
        collected: dict[Any, dict[str, Any]] = {}
        probes_needed = max(1, sample_size // PROBE_CHUNK)
        # Cap the probe count so a sparse key space cannot loop for long.
        max_probes = probes_needed * 4
        for _ in range(max_probes):
            if len(collected) >= sample_size:
                break
            start = _random_seek_point(lo, hi, key.kind, self._rng, base, span)
            rows = self._query(
                f"SELECT {select_list} FROM {qualified} "
                f"WHERE {quoted_key} >= {operand} "
                f"ORDER BY {quoted_key} LIMIT {PROBE_CHUNK}",
                (start,),
            )
            for row in rows:
                collected[row[key.column]] = row
                if span is not None and isinstance(row[key.column], str):
                    span.learn(row[key.column])
        return list(collected.values())[:sample_size]

    def _combined_key_sample(
        self,
        qualified: str,
        select_list: str,
        key: SeekKey,
        sample_size: int,
    ) -> list[dict[str, Any]]:
        """Sample along a key that is unique only as a combination of columns.

        Seeking on the leading column alone does not spread: when it holds few
        values (a year, a tenant), every probe lands on the start of one of
        them and reads the same first rows again.  So a probe walks down the
        key one column at a time.  It picks a random point in the first
        column's range and takes the value at or after it, then does the same
        for the second column among rows with that first value, and so on.
        The last column gets a random point, not a value.

        From that position it reads forward in key order.  The read is split
        into one query per column, each an equality on the columns before it
        and a range on its own, because MySQL scans a whole index for a
        ``(a, b) >= (x, y)`` comparison instead of seeking.  Measured on a
        91,000-row table: the tuple form read every row to return 25, the
        split form read 35.

        Rows are de-duplicated on every column of the key together, which is
        what the database guarantees unique.

        Values are reached in proportion to their *range*, not their row count,
        so a value holding many rows is under-represented next to one holding
        few.  That is the price of never scanning.  It is the same bias the
        single-column probe has across gaps in its key.
        """
        columns = key.columns
        quoted = [quote_identifier(c.column) for c in columns]

        def where(depth: int, extra: str = "") -> str:
            """Fix the first ``depth`` key columns, plus one extra condition."""
            clauses = [f"{q} = {c.operand()}"
                       for q, c in zip(quoted[:depth], columns[:depth])]
            if extra:
                clauses.append(extra)
            return " WHERE " + " AND ".join(clauses) if clauses else ""

        leading_bounds: tuple[Any, Any] | None = None
        # One per string column, made on first use with that column's charset.
        spans: dict[int, CharSpan] = {}

        def descend() -> list[Any] | None:
            nonlocal leading_bounds
            position: list[Any] = []
            for depth, (column, q) in enumerate(zip(columns, quoted)):
                if depth == 0 and leading_bounds is not None:
                    lo, hi = leading_bounds
                else:
                    bounds = self._query(
                        f"SELECT MIN({q}) AS lo, MAX({q}) AS hi FROM {qualified}"
                        + where(depth),
                        tuple(position),
                    )
                    lo, hi = (bounds[0]["lo"], bounds[0]["hi"]) if bounds else (None, None)
                    if depth == 0:
                        leading_bounds = (lo, hi)
                if lo is None or hi is None:
                    return None
                last = depth == len(columns) - 1
                if hi <= lo:
                    position.append(lo)
                    continue
                base = (_charset_base(column.charset, lo, hi)
                        if column.kind == "str" else _CHAR_BASE)
                span = (spans.setdefault(depth, CharSpan(base))
                        if column.kind == "str" else None)
                point = _random_seek_point(lo, hi, column.kind, self._rng, base, span)
                if last:
                    position.append(point)
                    continue
                found = self._query(
                    f"SELECT {q} AS v FROM {qualified}"
                    + where(depth, f"{q} >= {column.operand()}")
                    + f" ORDER BY {q} LIMIT 1",
                    (*position, point),
                )
                position.append(found[0]["v"] if found else hi)
            return position

        def read_from(position: list[Any]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for depth in range(len(columns) - 1, -1, -1):
                need = PROBE_CHUNK - len(rows)
                if need <= 0:
                    break
                compare = ">=" if depth == len(columns) - 1 else ">"
                q, column = quoted[depth], columns[depth]
                rows += self._query(
                    f"SELECT {select_list} FROM {qualified}"
                    + where(depth, f"{q} {compare} {column.operand()}")
                    + f" ORDER BY {', '.join(quoted[depth:])} LIMIT {int(need)}",
                    (*position[:depth], position[depth]),
                )
            return rows

        collected: dict[tuple[Any, ...], dict[str, Any]] = {}
        max_probes = max(1, sample_size // PROBE_CHUNK) * 4
        for _ in range(max_probes):
            if len(collected) >= sample_size:
                break
            position = descend()
            if position is None:
                return []
            for row in read_from(position):
                collected[tuple(row[c.column] for c in columns)] = row
                for depth, span in spans.items():
                    if isinstance(row[columns[depth].column], str):
                        span.learn(row[columns[depth].column])
        return list(collected.values())[:sample_size]

    def _read_table(
        self, dataset: str, table: str, columns: list[str], sample_size: int
    ) -> list[dict[str, Any]]:
        """Sample one table, surviving a connection that drops underneath us."""
        return self.with_reconnect(
            f"{dataset}.{table}",
            lambda: self._cached_rows(dataset, table, columns, sample_size),
        )

    def is_connection_lost(self, exc: BaseException) -> bool:
        if not isinstance(
            exc, (pymysql.err.OperationalError, pymysql.err.InterfaceError)
        ):
            return False
        code = exc.args[0] if exc.args else None
        return code in _CONNECTION_LOST_CODES

    def _cached_rows(
        self, dataset: str, table: str, columns: list[str], sample_size: int
    ) -> list[dict[str, Any]]:
        cached = self._row_cache
        if cached and cached[0] == dataset and cached[1] == table and cached[2] == sample_size:
            return cached[3]
        rows = self._fetch_table_sample(dataset, table, columns, sample_size)
        self._row_cache = (dataset, table, sample_size, rows)
        return rows

    def _scannable_columns(self, dataset: str, table: str) -> list[FieldSchema]:
        return [
            s
            for s in self.get_schema(dataset)
            if s.collection == table and not s.skip
        ]

    def sample_field(
        self, schema: FieldSchema, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> FieldSample:
        columns = [s.name for s in self._scannable_columns(schema.dataset, schema.collection)]
        if schema.name not in columns:
            columns = columns + [schema.name]
        rows = self._cached_rows(schema.dataset, schema.collection, columns, sample_size)
        return self._sample_from_rows(schema, rows)

    @staticmethod
    def _sample_from_rows(
        schema: FieldSchema, rows: list[dict[str, Any]]
    ) -> FieldSample:
        # Counts are derived by FieldSample from `values`, so there is nothing
        # here that can disagree with them.
        return FieldSample(
            schema=schema,
            values=[row.get(schema.name) for row in rows],
        )

    def iter_samples(
        self, dataset: str, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Iterator[FieldSample]:
        """Sample every scannable field, one query per table.

        The default implementation in the ABC would issue one query per
        column; on a 60-column table that is 60 scans instead of one.
        """
        schemas = self.get_schema(dataset)
        by_table: dict[str, list[FieldSchema]] = {}
        for schema in schemas:
            if schema.skip:
                continue
            by_table.setdefault(schema.collection, []).append(schema)

        for table, fields in by_table.items():
            columns = [f.name for f in fields]
            try:
                rows = self._read_table(dataset, table, columns, sample_size)
            except Exception as exc:  # noqa: BLE001 - recorded, then reported
                # A connection error here already survived one reconnect and
                # retry in _read_table, so the network is gone rather than the
                # table being bad.  Let it out: every remaining table would
                # fail too, and a skip list is read as "nothing found here",
                # not as "never looked".
                if self.is_connection_lost(exc):
                    raise
                # An unreadable table, on the other hand, must not cost the
                # other 264 their results.  This is a generator, so letting
                # that out would end the whole dataset, not just this table.
                self.record_skip(dataset, table, exc)
                continue
            log.info(
                "sampled %s.%s: %d rows across %d fields",
                dataset, table, len(rows), len(fields),
            )
            for schema in fields:
                yield self._sample_from_rows(schema, rows)
