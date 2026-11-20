"""End-to-end quality suites against the loaded synthetic dataset.

Three separable claims:

*recall*      every deliberately planted PII column is found
*precision*   no decoy column is flagged
*read-only*   the scanner cannot write, by code and by grant
"""

from __future__ import annotations

import pytest

from piiscanner.benchmark import format_score, score_scan
from piiscanner.connectors.base import ReadOnlyViolation
from piiscanner.config import ScanConfig, build_connector
from piiscanner.testdata.generate import (
    DATABASE_NAME,
    EXPECTED,
    FORBIDDEN,
    GROUND_TRUTH,
)
from tests.conftest import TEST_DSN, requires_dataset

pytestmark = requires_dataset


# --------------------------------------------------------------------------
# Recall -- every planted field is found
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expectation", EXPECTED, ids=[f"{e.collection}.{e.field}" for e in EXPECTED]
)
def test_planted_pii_is_found(scan_result, expectation) -> None:
    found = [
        f for f in scan_result.findings
        if (f.collection, f.field) == (expectation.collection, expectation.field)
    ]
    assert found, (
        f"{expectation.collection}.{expectation.field} was missed "
        f"(expected {expectation.pii_type.value})"
    )
    types = {f.pii_type for f in found}
    assert expectation.pii_type in types, (
        f"{expectation.collection}.{expectation.field} found, but labelled "
        f"{[t.value for t in types]} instead of {expectation.pii_type.value}"
    )


def test_layer2_only_cases_are_found_without_name_help(scan_result) -> None:
    """The badly named columns -- these justify value scanning existing."""
    layer2_cases = [e for e in EXPECTED if e.layer2_only]
    assert layer2_cases
    for expectation in layer2_cases:
        found = [
            f for f in scan_result.findings
            if (f.collection, f.field) == (expectation.collection, expectation.field)
            and f.pii_type is expectation.pii_type
        ]
        assert found, f"Layer 2 missed {expectation.collection}.{expectation.field}"
        assert not found[0].unverified


# --------------------------------------------------------------------------
# Precision -- the decoys stay silent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection,field,why",
    FORBIDDEN,
    ids=[f"{c}.{f}" for c, f, _ in FORBIDDEN],
)
def test_decoy_columns_produce_no_findings(scan_result, collection, field, why) -> None:
    hits = [
        f for f in scan_result.findings
        if (f.collection, f.field) == (collection, field)
    ]
    assert not hits, (
        f"false positive on {collection}.{field} ({why}): "
        f"flagged as {[h.pii_type.value for h in hits]}"
    )


# --------------------------------------------------------------------------
# The headline metric
# --------------------------------------------------------------------------


def test_precision_and_recall_are_perfect_on_synthetic_data(scan_result) -> None:
    """The synthetic dataset is the floor, not the ceiling.

    Perfect scores here mean the scanner handles the cases we thought of.
    They are not evidence about real customer data, which is why the
    benchmark output also prints the unscored count.
    """
    score = score_scan(scan_result)
    assert score.recall == 1.0, "\n" + format_score(score)
    assert score.precision == 1.0, "\n" + format_score(score)
    assert score.type_accuracy == 1.0, "\n" + format_score(score)


def test_findings_are_sorted_by_confidence_then_row_count(scan_result) -> None:
    ordered = scan_result.sorted_findings()
    keys = [(-f.confidence, -(f.approx_row_count or 0)) for f in ordered]
    assert keys == sorted(keys)


def test_every_finding_records_its_sample_size(scan_result) -> None:
    """A verdict from 3 values must be distinguishable from one from 500."""
    for finding in scan_result.findings:
        assert finding.sample_count > 0
        assert 0.0 <= finding.null_rate <= 1.0
        assert 0.0 <= finding.distinct_ratio <= 1.0


# --------------------------------------------------------------------------
# Read-only enforcement
# --------------------------------------------------------------------------


def test_connector_refuses_to_execute_writes() -> None:
    """The code-level guard fires before anything reaches the server."""
    connector = build_connector(ScanConfig(source=TEST_DSN))
    with connector:
        for statement in (
            "UPDATE users SET city = 'x'",
            "DELETE FROM users",
            "INSERT INTO users VALUES (1)",
            "DROP TABLE users",
            "CREATE TABLE zz (a INT)",
            "TRUNCATE users",
        ):
            with pytest.raises(ReadOnlyViolation):
                connector._query(statement)


def test_session_is_read_only_at_the_server() -> None:
    """Independently of our guard, the session itself rejects writes."""
    import pymysql

    connector = build_connector(ScanConfig(source=TEST_DSN))
    with connector:
        with connector._cursor() as cur:
            with pytest.raises(pymysql.err.MySQLError):
                cur.execute("CREATE TABLE piiscanner_should_not_exist (a INT)")


def test_scan_does_not_change_the_data(scan_result) -> None:
    """Checksum the tables before and after a scan."""
    import pymysql

    def snapshot() -> dict[str, int]:
        conn = pymysql.connect(
            host="127.0.0.1", user="piiscanner_ro", password="piiscanner_ro_pw",
            database="piiscanner_testdata",
        )
        try:
            out = {}
            with conn.cursor() as cur:
                for table in ("users", "kyc", "legacy_import", "decoys"):
                    cur.execute(f"CHECKSUM TABLE `{table}`")
                    out[table] = cur.fetchone()[1]
            return out
        finally:
            conn.close()

    before = snapshot()
    config = ScanConfig(source=TEST_DSN, sample_size=200)
    connector = build_connector(config)
    from piiscanner.classify.engine import run_scan

    with connector:
        run_scan(connector, config)
    assert snapshot() == before


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------


def test_include_and_exclude_globs_are_honoured() -> None:
    from piiscanner.classify.engine import run_scan

    # Scoped to its own database on purpose.  Without this the scan walks
    # every schema the read-only account can see, so adding an unrelated test
    # database elsewhere on the server silently changes this test's answer.
    config = ScanConfig(
        source=TEST_DSN, datasets=[DATABASE_NAME],
        include=["kyc", "legacy_*"], exclude=["*_import"],
    )
    connector = build_connector(config)
    with connector:
        result = run_scan(connector, config)
    collections = {f.collection for f in result.findings}
    assert collections == {"kyc"}


def test_sampling_a_large_table_is_bounded(scan_result) -> None:
    """The 120k-row table must be sampled, not scanned."""
    events = [f for f in scan_result.findings if f.collection == "events"]
    assert events
    for finding in events:
        assert finding.sample_count <= 500
        assert finding.approx_row_count and finding.approx_row_count > 100_000
