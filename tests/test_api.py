"""``import piiscanner``: the README's example, run as written.

Needs the synthetic dataset in a local MySQL; skips without it (see README,
Development).
"""

from __future__ import annotations

import pytest

import piiscanner
from tests.conftest import DATASET_AVAILABLE, TEST_DSN

pytestmark = pytest.mark.skipif(not DATASET_AVAILABLE, reason="synthetic dataset not loaded")


@pytest.fixture(scope="module")
def result() -> piiscanner.ScanResult:
    return piiscanner.scan(TEST_DSN)


def test_the_readme_example(result) -> None:
    found = {(f.collection, f.field): f.pii_type for f in result.findings}
    assert found[("kyc", "aadhaar_no")] is piiscanner.PIIType.AADHAAR
    assert found[("users", "email")] is piiscanner.PIIType.EMAIL
    # A column with a meaningless name, found by its values alone.
    assert found[("legacy_import", "col_17")] is piiscanner.PIIType.AADHAAR


def test_evidence_comes_back_masked(result) -> None:
    [aadhaar] = [f for f in result.findings if (f.collection, f.field) == ("kyc", "aadhaar_no")]
    assert aadhaar.evidence
    # First two and last two characters kept, everything between masked --
    # whatever the stored form (some are spaced: 14 characters, not 12).
    import re
    for e in aadhaar.evidence:
        assert re.fullmatch(r"\d\dX+\d\d", e), e


def test_it_scans_the_database_the_connection_string_names(result) -> None:
    assert result.datasets_scanned == 1
    assert {f.dataset for f in result.findings} == {TEST_DSN.rsplit("/", 1)[1]}


def test_the_names_are_importable_from_the_top() -> None:
    assert set(piiscanner.__all__) == {"scan", "ScanResult", "Finding", "PIIType", "__version__"}
    assert piiscanner.__version__[0].isdigit()


def test_include_narrows_the_scan() -> None:
    narrow = piiscanner.scan(TEST_DSN, include=["kyc"])
    assert narrow.findings and {f.collection for f in narrow.findings} == {"kyc"}


def test_a_table_that_could_not_be_read_is_named(monkeypatch) -> None:
    """Not covered is not the same as clean, and a Python caller must be
    able to tell -- the command line says so on screen."""
    from piiscanner.connectors.mysql import MySQLConnector

    real = MySQLConnector._read_table

    def failing(self, dataset, table, *args, **kwargs):
        if table == "kyc":
            raise RuntimeError("permission denied for table kyc")
        return real(self, dataset, table, *args, **kwargs)

    monkeypatch.setattr(MySQLConnector, "_read_table", failing)
    partial = piiscanner.scan(TEST_DSN)
    assert partial.tables_not_read == [f"{TEST_DSN.rsplit('/', 1)[1]}.kyc"]
    assert not any(f.collection == "kyc" for f in partial.findings)
    assert "permission denied" not in partial.model_dump_json()   # names only


def test_a_clean_scan_reads_every_table(result) -> None:
    assert result.tables_not_read == []
