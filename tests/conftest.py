"""Shared fixtures.

Database-backed tests skip cleanly when the synthetic dataset has not been
loaded, so ``pytest`` still works on a fresh checkout.  Load it with::

    python -m piiscanner.testdata.load_mysql
"""

from __future__ import annotations

import pytest

from piiscanner.classify.engine import run_scan
from piiscanner.config import ScanConfig, build_connector
from piiscanner.models import ScanResult
from piiscanner.testdata.generate import DATABASE_NAME

TEST_DSN = f"mysql://piiscanner_ro:piiscanner_ro_pw@127.0.0.1:3306/{DATABASE_NAME}"
MONGO_DSN = f"mongodb://127.0.0.1:27017/{DATABASE_NAME}"


def _dataset_available() -> bool:
    try:
        import pymysql

        conn = pymysql.connect(
            host="127.0.0.1",
            port=3306,
            user="piiscanner_ro",
            password="piiscanner_ro_pw",
            database=DATABASE_NAME,
            connect_timeout=3,
        )
    except Exception:
        return False
    conn.close()
    return True


DATASET_AVAILABLE = _dataset_available()

requires_dataset = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=(
        "synthetic MySQL dataset not loaded; "
        "run `python -m piiscanner.testdata.load_mysql`"
    ),
)


@pytest.fixture(scope="session")
def scan_result() -> ScanResult:
    """One scan of the synthetic dataset, shared across the suite."""
    config = ScanConfig(source=TEST_DSN, alias="test", sample_size=500)
    connector = build_connector(config)
    with connector:
        return run_scan(connector, config)


def _mongo_available() -> bool:
    try:
        import pymongo

        client = pymongo.MongoClient(MONGO_DSN, serverSelectionTimeoutMS=2000)
        names = client[DATABASE_NAME].list_collection_names()
        client.close()
    except Exception:
        return False
    return bool(names)


MONGO_AVAILABLE = _mongo_available()

requires_mongo = pytest.mark.skipif(
    not MONGO_AVAILABLE,
    reason=(
        "synthetic MongoDB dataset not loaded; "
        "run `python -m piiscanner.testdata.load_mongodb`"
    ),
)


@pytest.fixture(scope="session")
def mongo_scan_result() -> ScanResult:
    """One scan of the synthetic MongoDB dataset, shared across the suite."""
    config = ScanConfig(
        source=MONGO_DSN, alias="test-mongo", sample_size=500,
        datasets=[DATABASE_NAME],
    )
    connector = build_connector(config)
    with connector:
        return run_scan(connector, config)
