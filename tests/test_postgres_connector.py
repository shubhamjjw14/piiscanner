"""The PostgreSQL connector: finds what it should, as an account that can only read.

A scratch schema of synthetic people is scanned by a role holding SELECT and
nothing else -- the account the README tells people to create.  Needs a
local PostgreSQL; skips without one.  Point ``PIISCANNER_TEST_PG_DSN`` at a
server you may create schemas and roles in, never one holding real data.
"""

from __future__ import annotations

import os
import random

import pytest

from piiscanner.testdata.generate import (
    gen_aadhaar,
    gen_email,
    gen_gstin,
    gen_mobile,
    gen_name,
    gen_order_reference,
)

PG_ADMIN = os.environ.get("PIISCANNER_TEST_PG_DSN",
                          "postgresql://postgres:postgres@127.0.0.1:5432/postgres")
SCHEMA = "piiscanner_pg_connector_test"
ROLE = "piiscanner_pg_connector_ro"
ROWS = 600


@pytest.fixture(scope="module")
def readonly_dsn():
    psycopg = pytest.importorskip("psycopg")
    try:
        admin = psycopg.connect(PG_ADMIN, autocommit=True, connect_timeout=3)
    except Exception:  # noqa: BLE001
        pytest.skip("needs a local PostgreSQL (set PIISCANNER_TEST_PG_DSN)")
    rng = random.Random(11)
    rows = []
    for i in range(ROWS):
        first, last, _ = gen_name(rng)
        rows.append((i + 1, f"{first} {last}", gen_email(rng, first, last),
                     gen_mobile(rng), gen_aadhaar(rng), gen_gstin(rng),
                     gen_order_reference(rng)))
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        cur.execute(f"DROP ROLE IF EXISTS {ROLE}")
        cur.execute(f"CREATE SCHEMA {SCHEMA}")
        cur.execute(f"CREATE TABLE {SCHEMA}.customers (id int PRIMARY KEY, full_name text, "
                    "email text, mobile text, aadhaar_no text, gstin text, order_ref text)")
        cur.executemany(f"INSERT INTO {SCHEMA}.customers VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        rows)
        cur.execute(f"ANALYZE {SCHEMA}.customers")
        cur.execute(f"CREATE ROLE {ROLE} LOGIN PASSWORD 'readonly'")
        cur.execute(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {ROLE}")
        cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO {ROLE}")
    scheme, rest = PG_ADMIN.split("://", 1)
    host_and_db = rest.split("@", 1)[1]
    yield f"{scheme}://{ROLE}:readonly@{host_and_db}?schema={SCHEMA}"
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        cur.execute(f"DROP ROLE IF EXISTS {ROLE}")
    admin.close()


@pytest.fixture(scope="module")
def findings(readonly_dsn):
    from piiscanner.classify.engine import run_scan
    from piiscanner.config import ScanConfig, build_connector

    config = ScanConfig(source=readonly_dsn, sample_size=500)
    with build_connector(config) as connector:
        result = run_scan(connector, config)
    return {(f.collection, f.field): f.pii_type.value for f in result.findings}


def test_the_personal_columns_are_found(findings) -> None:
    assert findings[("customers", "email")] == "EMAIL"
    assert findings[("customers", "mobile")] == "INDIAN_MOBILE"
    assert findings[("customers", "aadhaar_no")] == "AADHAAR"
    assert findings[("customers", "gstin")] == "GSTIN"
    assert findings[("customers", "full_name")] == "PERSON_NAME"


def test_an_order_reference_is_not_aadhaar(findings) -> None:
    """12 digits, and about 1 in 10 pass the Aadhaar checksum by chance."""
    assert ("customers", "order_ref") not in findings


def test_a_write_is_refused_before_it_is_sent(readonly_dsn) -> None:
    from piiscanner.config import ScanConfig, build_connector
    from piiscanner.connectors.base import ReadOnlyViolation

    with build_connector(ScanConfig(source=readonly_dsn)) as connector:
        with pytest.raises(ReadOnlyViolation):
            connector._query(f"DELETE FROM {SCHEMA}.customers")


def test_the_session_itself_is_read_only(readonly_dsn) -> None:
    """Below the guard: the server refuses a write on piiscanner's session
    even for an account that could write."""
    import psycopg

    from piiscanner.config import ScanConfig, build_connector

    admin_dsn = f"{PG_ADMIN}{'&' if '?' in PG_ADMIN else '?'}schema={SCHEMA}"
    with build_connector(ScanConfig(source=admin_dsn)) as connector:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            with connector._conn.cursor() as cur:
                cur.execute(f"DELETE FROM {SCHEMA}.customers")
