"""Read-only sampling access to a PostgreSQL server.

The same contract as :mod:`piiscanner.connectors.mysql`, so the scanner and the
linkage rules work unchanged.  Three things differ, and each is decided here:

* **A dataset is the whole database, and a table is named ``schema.table``.**
  Postgres puts many schemas in one database and applications use them: a
  hospital keeps a patient in ``mimiciv_hosp`` and their ICU stay in
  ``mimiciv_icu``, joined by foreign keys across the two.  Treating one
  schema as the whole world lost every row in the other, silently.  So with
  no schema in the DSN, every non-system schema is read and each table
  carries its schema in its name.  ``?schema=billing`` restricts a run to
  one schema, and then tables are plain names, as before:
  ``postgresql://ro@host:5432/appdb`` or ``...appdb?schema=billing``.
* **Spread sampling is built in.**  ``TABLESAMPLE BERNOULLI`` keeps each row
  with a probability, across the whole table -- what the MySQL connector has
  to build by hand with ``RAND()``.  Above
  :data:`SPREAD_SAMPLE_MAX_ROWS` it samples whole pages instead
  (``TABLESAMPLE SYSTEM``), which reads only the pages it keeps.
* **Query deadlines are a setting, not a hint.**  The dialect marks a query
  with a leading ``/*piiscanner:timeout=N*/``; this connector sets
  ``statement_timeout`` for that one statement, inside a read-only
  transaction.

Read-only three ways: the session defaults every transaction to read-only on
the server, :meth:`PostgresConnector._query` refuses anything that is not a
``SELECT``/``WITH``/``SHOW`` before it is sent, and nothing here ever builds
a write.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlparse

from piiscanner.connectors.base import (
    DEFAULT_SAMPLE_SIZE,
    SAMPLED_FIRST_ROWS_TOO_SLOW,
    SAMPLED_SPREAD,
    Connector,
    FieldSample,
    ReadOnlyViolation,
)
from piiscanner.models import ConnectorInfo, FieldSchema

log = logging.getLogger(__name__)

#: Schemas belonging to the server, never scanned.
SYSTEM_SCHEMAS = frozenset({"information_schema", "pg_catalog", "pg_toast"})

#: Column types that cannot hold personal data.  As on MySQL, numbers are
#: scanned: phone and Aadhaar numbers are stored as BIGINT.
UNSCANNABLE_TYPES = frozenset({
    "bytea", "boolean", "bit", "bit varying", "point", "line", "lseg", "box",
    "path", "polygon", "circle", "tsvector", "tsquery", "geometry", "geography",
})

#: Below this many rows (by estimate), ``ORDER BY random()`` reads the table
#: and is a true uniform sample.  Same threshold as MySQL.
SMALL_TABLE_THRESHOLD = 20_000

#: Above this many rows, sample whole pages (``TABLESAMPLE SYSTEM``) rather
#: than every row (``BERNOULLI``): it reads only the pages it keeps.
SPREAD_SAMPLE_MAX_ROWS = 50_000_000

#: Scans a spread sample may take when the row estimate proves too high.
SPREAD_SAMPLE_PASSES = 3

#: Longest the sampling scan may run.
SPREAD_SAMPLE_TIMEOUT_MS = 20_000

#: How the dialect marks a query that must finish within a deadline.
_DEADLINE = re.compile(r"^\s*/\*piiscanner:timeout=(\d+)\*/\s*")

_UNSAFE_IN_IDENTIFIER = re.compile(r"[\x00-\x1f\x7f]")
_MAX_IDENTIFIER = 63


def quote_identifier(name: str) -> str:
    """Double-quote a name the way Postgres documents.

    Inside double quotes everything is literal except the double quote,
    written twice.  Quoting also keeps a name's case: ``"Person"`` and
    ``person`` are different tables in Postgres.  Control characters and
    over-long names are refused, as on MySQL.
    """
    if (not name or len(name) > _MAX_IDENTIFIER
            or _UNSAFE_IN_IDENTIFIER.search(name)):
        raise ValueError(f"refusing an unusual identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def parse_postgres_dsn(dsn: str) -> dict[str, Any]:
    """``postgresql://user:pw@host:port/db?schema=s`` into connect arguments."""
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError(f"expected a postgresql:// DSN, got {parsed.scheme!r}")
    query = parse_qs(parsed.query)
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 5432,
        "user": unquote(parsed.username) if parsed.username else "postgres",
        "password": unquote(parsed.password) if parsed.password else "",
        "dbname": parsed.path.lstrip("/") or "postgres",
        "schema": (query.get("schema") or [None])[0],
    }


class PostgresConnector(Connector):
    """Read-only sampling access to a PostgreSQL database."""

    kind = "postgres"

    def __init__(
        self,
        dsn: str,
        alias: str | None = None,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        seed: int | None = None,
        cursor_path: str | None = None,
    ) -> None:
        super().__init__(dsn, alias)
        params = parse_postgres_dsn(dsn)
        schema = params.pop("schema")
        self._params = params
        #: Every schema, tables named ``schema.table`` -- or only ``schema``.
        self.whole_database = schema is None
        self.default_dataset = schema or params["dbname"]
        self.sample_size = sample_size
        self._conn = None
        self._rng = random.Random(seed)
        self._row_cache: tuple[str, str, int, list[dict[str, Any]]] | None = None
        #: ``{"schema.table": how it was sampled}`` -- see the MySQL connector.
        self.sample_methods: dict[str, str] = {}

    # -- names -----------------------------------------------------------

    def split_name(self, dataset: str, table: str) -> tuple[str, str]:
        """``(schema, table)`` for a table of ``dataset``."""
        if self.whole_database:
            schema, _, name = table.partition(".")
            return schema, name
        return dataset, table

    def qualified(self, dataset: str, table: str) -> str:
        schema, name = self.split_name(dataset, table)
        return f"{quote_identifier(schema)}.{quote_identifier(name)}"

    def _schema_filter(self, column: str, dataset: str) -> tuple[str, tuple]:
        """SQL limiting ``column`` to this run's schemas."""
        if self.whole_database:
            return (f"{column} NOT IN ('information_schema', 'pg_catalog') "
                    f"AND left({column}, 3) <> 'pg_'", ())
        return f"{column} = %s", (dataset,)

    def _name(self, schema_column: str, table_column: str) -> str:
        if self.whole_database:
            return f"{schema_column} || '.' || {table_column}"
        return table_column

    # -- connection ------------------------------------------------------

    def connect(self) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self._conn = psycopg.connect(
            **self._params, autocommit=True, row_factory=dict_row,
            # Server-side enforcement: every transaction in this session is
            # read-only, so any write errors whatever sent it.
            options="-c default_transaction_read_only=on",
        )
        log.info("connected to postgres at %s:%s/%s", self._params["host"],
                 self._params["port"], self._params["dbname"])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._row_cache = None

    def is_connection_lost(self, exc: BaseException) -> bool:
        import psycopg

        if not isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
            return False
        return self._conn is None or self._conn.closed or self._conn.broken

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only statement, with its deadline if it carries one.

        Anything that is not a ``SELECT``, ``WITH`` or ``SHOW`` raises
        :class:`ReadOnlyViolation` before it reaches the server -- a guard
        against our own bugs.
        """
        if self._conn is None:
            raise RuntimeError("connector is not connected; call connect() first")
        deadline = _DEADLINE.match(sql)
        body = sql[deadline.end():] if deadline else sql
        first_word = body.strip().split(None, 1)[0].upper() if body.strip() else ""
        if first_word not in {"SELECT", "WITH", "SHOW"}:
            raise ReadOnlyViolation(
                f"refusing to execute a non-read statement: {first_word or '<empty>'}")
        if deadline is None:
            with self._conn.cursor() as cur:
                cur.execute(body, params or None)
                return list(cur.fetchall())
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {int(deadline.group(1))}")
            cur.execute(body, params or None)
            return list(cur.fetchall())

    def stream_column(
        self, dataset: str, table: str, column: str, limit: int | None = None
    ) -> Iterator[Any]:
        """Yield every value of one column without holding the table in memory.

        A server-side cursor, so rows arrive as they are consumed.  See the
        MySQL connector for why this exists and why ``limit`` is in the query.
        """
        if self._conn is None:
            raise RuntimeError("connector is not connected; call connect() first")
        sql = (f"SELECT {quote_identifier(column)} AS v "
               f"FROM {self.qualified(dataset, table)}")
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._conn.transaction():
            with self._conn.cursor(name="piiscanner_stream") as cur:
                cur.itersize = 10_000
                cur.execute(sql)
                for row in cur:
                    yield row["v"]

    def info(self) -> ConnectorInfo:
        import psycopg

        version = None
        if self._conn is not None:
            rows = self._query("SHOW server_version")
            version = rows[0]["server_version"] if rows else None
        return ConnectorInfo(alias=self.alias, kind=self.kind,
                             driver_version=f"psycopg {psycopg.__version__}",
                             server_version=version)

    # -- schema ----------------------------------------------------------

    def list_datasets(self) -> list[str]:
        if self.whole_database:
            return [self._params["dbname"]]
        rows = self._query("SELECT nspname AS s FROM pg_catalog.pg_namespace "
                           "ORDER BY nspname")
        return [r["s"] for r in rows
                if r["s"] not in SYSTEM_SCHEMAS and not r["s"].startswith("pg_")]

    def _table_row_counts(self, dataset: str) -> dict[str, int]:
        """The planner's row estimates (``reltuples``): never a ``COUNT(*)``.

        A table never analysed reports -1; it is read as 0, which sends it
        down the small-table path where it is read once and measured.
        """
        # A partitioned table holds no rows itself: its estimate is the sum
        # of its partitions'.  Partitions are not listed -- they are part of
        # the parent, and listing both would count every row twice.
        where, params = self._schema_filter("s.nspname", dataset)
        rows = self._query(
            f"SELECT {self._name('s.nspname', 'c.relname')} AS t, "
            "CASE WHEN c.relkind = 'p' THEN "
            "  (SELECT COALESCE(SUM(GREATEST(p.reltuples, 0)), 0) "
            "   FROM pg_catalog.pg_inherits i "
            "   JOIN pg_catalog.pg_class p ON p.oid = i.inhrelid "
            "   WHERE i.inhparent = c.oid) "
            "  ELSE c.reltuples END AS n "
            "FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace s ON s.oid = c.relnamespace "
            f"WHERE {where} AND c.relkind IN ('r', 'p') "
            "  AND NOT c.relispartition", params)
        return {r["t"]: max(0, int(r["n"] or 0)) for r in rows}

    def get_schema(self, dataset: str) -> list[FieldSchema]:
        counts = self._table_row_counts(dataset)
        where, params = self._schema_filter("table_schema", dataset)
        rows = self._query(
            f"SELECT {self._name('table_schema', 'table_name')} AS table_name, "
            "column_name AS column_name, "
            "data_type AS data_type, udt_name AS udt_name, is_nullable AS is_nullable "
            f"FROM information_schema.columns WHERE {where} "
            "ORDER BY 1, ordinal_position", params)
        schemas: list[FieldSchema] = []
        for row in rows:
            table = row["table_name"]
            if table not in counts:  # a view, not a base table
                continue
            data_type = (row["data_type"] or "").lower()
            if data_type == "user-defined":
                data_type = (row["udt_name"] or "").lower()
            skip = data_type in UNSCANNABLE_TYPES
            schemas.append(FieldSchema(
                dataset=dataset, collection=table, name=row["column_name"],
                native_type=data_type, nullable=(row["is_nullable"] == "YES"),
                approx_row_count=counts.get(table), skip=skip,
                skip_reason=f"type {data_type} cannot hold PII" if skip else None))
        return schemas

    # -- sampling --------------------------------------------------------

    def _fetch_table_sample(
        self, dataset: str, table: str, columns: list[str], sample_size: int
    ) -> list[dict[str, Any]]:
        """About ``sample_size`` rows, from all of the table.

        Small tables: ``ORDER BY random()``.  Larger ones: each row kept with
        a probability (``BERNOULLI``), aimed a little over the sample size and
        cut to size at random -- a ``LIMIT`` would keep the earliest rows and
        bring back the first-rows bias that lost MIMIC's ``chartevents``.  If
        the scan does not finish in time, the first rows are read and the
        table is recorded as sampled that way.
        """
        row_count = self._table_row_counts(dataset).get(table, 0)
        qualified = self.qualified(dataset, table)
        select_list = ", ".join(quote_identifier(c) for c in columns)
        key = f"{dataset}.{table}"

        if row_count <= SMALL_TABLE_THRESHOLD:
            return self._query(f"SELECT {select_list} FROM {qualified} "
                               f"ORDER BY random() LIMIT {int(sample_size)}")

        method = "BERNOULLI" if row_count <= SPREAD_SAMPLE_MAX_ROWS else "SYSTEM"
        pct = min(100.0, 120.0 * sample_size / max(row_count, 1))
        rows: list[dict[str, Any]] = []
        try:
            for _ in range(SPREAD_SAMPLE_PASSES):
                rows = self._query(
                    f"/*piiscanner:timeout={SPREAD_SAMPLE_TIMEOUT_MS}*/ "
                    f"SELECT {select_list} FROM {qualified} "
                    f"TABLESAMPLE {method} ({pct:.10f}) "
                    f"LIMIT {int(sample_size) * 4}")
                if pct >= 100.0 or len(rows) * 2 >= sample_size:
                    break
                # The estimate was too high; aim again from what came back.
                actual = max(len(rows), 1) * 100.0 / pct
                pct = min(100.0, 120.0 * sample_size / actual)
        except Exception as exc:  # noqa: BLE001 - fall back, and say so
            if self.is_connection_lost(exc):
                raise
            log.warning("spread sample of %s failed (%s); reading first rows",
                        qualified, str(exc)[:80])
            self.sample_methods[key] = SAMPLED_FIRST_ROWS_TOO_SLOW
            return self._query(f"SELECT {select_list} FROM {qualified} "
                               f"LIMIT {int(sample_size)}")
        self.sample_methods[key] = SAMPLED_SPREAD
        if len(rows) > sample_size:
            rows = self._rng.sample(list(rows), sample_size)
        return list(rows)

    def _cached_rows(
        self, dataset: str, table: str, columns: list[str], sample_size: int
    ) -> list[dict[str, Any]]:
        cached = self._row_cache
        if cached and cached[0] == dataset and cached[1] == table and cached[2] == sample_size:
            return cached[3]
        rows = self._fetch_table_sample(dataset, table, columns, sample_size)
        self._row_cache = (dataset, table, sample_size, rows)
        return rows

    def _read_table(self, dataset: str, table: str, columns: list[str],
                    sample_size: int) -> list[dict[str, Any]]:
        return self.with_reconnect(
            f"{dataset}.{table}",
            lambda: self._cached_rows(dataset, table, columns, sample_size))

    def sample_field(self, schema: FieldSchema,
                     sample_size: int = DEFAULT_SAMPLE_SIZE) -> FieldSample:
        columns = [s.name for s in self.get_schema(schema.dataset)
                   if s.collection == schema.collection and not s.skip]
        if schema.name not in columns:
            columns = columns + [schema.name]
        rows = self._cached_rows(schema.dataset, schema.collection, columns, sample_size)
        return FieldSample(schema=schema, values=[r.get(schema.name) for r in rows])

    def iter_samples(self, dataset: str,
                     sample_size: int = DEFAULT_SAMPLE_SIZE) -> Iterator[FieldSample]:
        """Sample every scannable field, one query per table."""
        by_table: dict[str, list[FieldSchema]] = {}
        for schema in self.get_schema(dataset):
            if not schema.skip:
                by_table.setdefault(schema.collection, []).append(schema)
        for table, fields in by_table.items():
            try:
                rows = self._read_table(dataset, table, [f.name for f in fields],
                                        sample_size)
            except Exception as exc:  # noqa: BLE001 - recorded, then reported
                if self.is_connection_lost(exc):
                    raise
                self.record_skip(dataset, table, exc)
                continue
            log.info("sampled %s.%s: %d rows across %d fields",
                     dataset, table, len(rows), len(fields))
            for schema in fields:
                yield FieldSample(schema=schema, values=[r.get(schema.name) for r in rows])
