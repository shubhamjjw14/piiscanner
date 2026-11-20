"""Load the synthetic dataset into MySQL.

This is the one place in the codebase that writes to a database, and it is a
development tool, not part of the scanner.  Nothing under ``piiscanner/connectors``
or ``piiscanner/classify`` may import it.

Usage::

    python -m piiscanner.testdata.load_mysql --admin-url mysql://root@127.0.0.1/ \\
        --create-readonly-user
"""

from __future__ import annotations

import sys
import time
from typing import Any
from urllib.parse import unquote, urlparse

import click
import pymysql

from piiscanner.testdata.generate import DATABASE_NAME, SEED, TABLES, generate_rows

#: The scanner is meant to run as this user, which holds SELECT and nothing else.
READONLY_USER = "piiscanner_ro"
READONLY_PASSWORD = "piiscanner_ro_pw"


def parse_mysql_url(url: str) -> dict[str, Any]:
    """Turn ``mysql://user:pw@host:port/db`` into pymysql connect kwargs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"expected a mysql:// URL, got {parsed.scheme!r}")
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username) if parsed.username else "root",
        "password": unquote(parsed.password) if parsed.password else "",
        "database": parsed.path.lstrip("/") or None,
    }


def load(admin_url: str, seed: int = SEED, quiet: bool = False) -> None:
    """Drop and rebuild the synthetic database from scratch."""
    params = parse_mysql_url(admin_url)
    params["database"] = None
    conn = pymysql.connect(**params, autocommit=True, local_infile=False)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {DATABASE_NAME}")
            cur.execute(
                f"CREATE DATABASE {DATABASE_NAME} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cur.execute(f"USE {DATABASE_NAME}")

            for table in TABLES:
                started = time.monotonic()
                cur.execute(table.ddl)
                placeholders = ", ".join(["%s"] * len(table.columns))
                insert = (
                    f"INSERT INTO {table.name} "
                    f"({', '.join(table.columns)}) VALUES ({placeholders})"
                )
                batch: list[tuple[Any, ...]] = []
                written = 0
                for row in generate_rows(table, seed):
                    batch.append(row)
                    if len(batch) >= table.batch_size:
                        cur.executemany(insert, batch)
                        written += len(batch)
                        batch.clear()
                if batch:
                    cur.executemany(insert, batch)
                    written += len(batch)
                if not quiet:
                    elapsed = time.monotonic() - started
                    click.echo(
                        f"  {table.name:<18} {written:>7,} rows  "
                        f"({elapsed:.1f}s)  {table.comment}"
                    )
            # Keep the optimiser's row estimates fresh; the connector reads
            # them from information_schema to decide its sampling strategy.
            for table in TABLES:
                cur.execute(f"ANALYZE TABLE {table.name}")
    finally:
        conn.close()


def create_readonly_user(admin_url: str, quiet: bool = False) -> str:
    """Create a SELECT-only user and return the DSN the scanner should use.

    Running the scanner as this user is defence in depth: the code never
    issues a write, and the grants make a write impossible even if it did.
    """
    params = parse_mysql_url(admin_url)
    params["database"] = None
    conn = pymysql.connect(**params, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE USER IF NOT EXISTS '{READONLY_USER}'@'%' "
                f"IDENTIFIED BY '{READONLY_PASSWORD}'"
            )
            cur.execute(
                f"ALTER USER '{READONLY_USER}'@'%' "
                f"IDENTIFIED BY '{READONLY_PASSWORD}'"
            )
            # Revoke first so re-running cannot silently accumulate grants.
            try:
                cur.execute(f"REVOKE ALL PRIVILEGES ON *.* FROM '{READONLY_USER}'@'%'")
            except pymysql.err.MySQLError:
                pass
            cur.execute(
                f"GRANT SELECT ON {DATABASE_NAME}.* TO '{READONLY_USER}'@'%'"
            )
            cur.execute("FLUSH PRIVILEGES")
    finally:
        conn.close()

    host = params["host"]
    port = params["port"]
    dsn = (
        f"mysql://{READONLY_USER}:{READONLY_PASSWORD}@{host}:{port}/{DATABASE_NAME}"
    )
    if not quiet:
        click.echo(f"  read-only user ready: {dsn}")
    return dsn


@click.command()
@click.option(
    "--admin-url",
    default="mysql://root@127.0.0.1:3306/",
    show_default=True,
    help="Admin DSN used to create the database. Never used by the scanner.",
)
@click.option("--seed", default=SEED, show_default=True, help="Generation seed.")
@click.option(
    "--create-readonly-user/--no-create-readonly-user",
    "with_readonly_user",
    default=True,
    show_default=True,
    help="Also create a SELECT-only user for the scanner to use.",
)
def main(admin_url: str, seed: int, with_readonly_user: bool) -> None:
    """Rebuild the synthetic PII dataset in MySQL."""
    click.echo(f"Building {DATABASE_NAME} (seed={seed})")
    started = time.monotonic()
    load(admin_url, seed)
    if with_readonly_user:
        create_readonly_user(admin_url)
    click.echo(f"Done in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
