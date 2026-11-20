"""No raw value may reach disk, logs, or the findings output.

This suite is the enforcement of constraint 2 and constraint 4.  It works by
reading real values straight out of the synthetic database, then asserting
that none of those exact strings appears anywhere in what the scanner emits.
That is a stronger check than testing the masking function in isolation: it
catches a leak through any path, including ones added later.
"""

from __future__ import annotations

import io
import json
import logging
import random
import re

import pytest

from piiscanner.classify.engine import classify_field, run_scan
from piiscanner.config import (
    NoValuesFilter,
    ScanConfig,
    build_connector,
    configure_logging,
)
from piiscanner.models import mask_value
from piiscanner.report.json_out import to_dict, write_json
from piiscanner.report.text_out import print_report
from piiscanner.testdata.generate import gen_aadhaar, gen_card, gen_email, gen_mobile
from tests.conftest import TEST_DSN, requires_dataset
from tests.test_layer2_values import make_sample

#: Columns whose values are unambiguous PII, used as the leak canaries.
CANARY_COLUMNS = [
    ("users", "email"),
    ("users", "mobile"),
    ("kyc", "aadhaar_no"),
    ("kyc", "pan_number"),
    ("kyc", "gstin"),
    ("legacy_import", "col_17"),
    ("legacy_import", "field_x"),
    ("numeric_pii", "uid_number"),
]


def _fetch_canaries(limit: int = 40) -> list[str]:
    """Pull real values out of the synthetic database to hunt for."""
    import pymysql

    conn = pymysql.connect(
        host="127.0.0.1", port=3306, user="piiscanner_ro",
        password="piiscanner_ro_pw", database="piiscanner_testdata",
    )
    values: list[str] = []
    try:
        with conn.cursor() as cur:
            for table, column in CANARY_COLUMNS:
                cur.execute(
                    f"SELECT `{column}` FROM `{table}` "
                    f"WHERE `{column}` IS NOT NULL LIMIT {limit}"
                )
                values.extend(str(row[0]) for row in cur.fetchall())
    finally:
        conn.close()
    return [v for v in values if len(v) >= 6]


# --------------------------------------------------------------------------
# Unit-level: the masking contract
# --------------------------------------------------------------------------


def test_evidence_never_equals_a_raw_value() -> None:
    rng = random.Random(99)
    for generator in (gen_aadhaar, gen_mobile, gen_card):
        values = [generator(rng) for _ in range(100)]
        findings = classify_field(make_sample("col_1", values), "test")
        for finding in findings:
            for item in finding.evidence:
                assert item not in values
                assert "X" in item


def test_masked_evidence_reveals_at_most_four_characters() -> None:
    rng = random.Random(7)
    for _ in range(200):
        raw = gen_aadhaar(rng)
        masked = mask_value(raw)
        revealed = sum(1 for a, b in zip(raw, masked) if a == b)
        assert revealed <= 4, f"{masked} reveals too much of {raw}"


def test_email_masking_hides_the_local_part() -> None:
    rng = random.Random(11)
    for _ in range(100):
        raw = gen_email(rng, "priyadarshini", "bhattacharya")
        masked = mask_value(raw)
        local = raw.split("@")[0]
        assert local not in masked
        assert "@" not in masked


# --------------------------------------------------------------------------
# End to end: nothing leaks through the real pipeline
# --------------------------------------------------------------------------


@requires_dataset
def test_no_raw_value_in_json_output(scan_result, tmp_path) -> None:
    canaries = _fetch_canaries()
    assert canaries, "expected to find canary values in the synthetic dataset"

    path = write_json(scan_result, tmp_path / "findings.json")
    written = path.read_text(encoding="utf-8")
    for value in canaries:
        assert value not in written, f"raw value leaked into findings JSON: {value[:2]}..."


@requires_dataset
def test_no_raw_value_in_serialised_result(scan_result) -> None:
    canaries = _fetch_canaries()
    blob = json.dumps(to_dict(scan_result))
    for value in canaries:
        assert value not in blob


@requires_dataset
def test_no_raw_value_in_captured_logs() -> None:
    """Run a whole scan at DEBUG and grep everything the logger emits.

    The capture is wired into the real ``piiscanner`` logger rather than pytest's
    ``caplog``, because :func:`configure_logging` deliberately sets
    ``propagate = False`` -- so caplog would see an empty string and the test
    would pass without checking anything.  This way the assertion runs against
    the same handler pipeline, and the same ``NoValuesFilter``, that ships.
    """
    canaries = _fetch_canaries()
    assert canaries

    configure_logging(verbose=True)
    logger = logging.getLogger("piiscanner")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    handler.addFilter(NoValuesFilter())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        config = ScanConfig(source=TEST_DSN, alias="leak-test", sample_size=200)
        connector = build_connector(config)
        with connector:
            run_scan(connector, config)
    finally:
        logger.removeHandler(handler)

    captured = buffer.getvalue()
    assert captured.strip(), "expected the scan to log something"
    assert "scan complete" in captured
    for value in canaries:
        assert value not in captured, f"raw value leaked into logs: {value[:2]}..."


@requires_dataset
def test_no_raw_value_in_terminal_report(scan_result) -> None:
    from rich.console import Console

    console = Console(file=open("/dev/null", "w"), record=True, width=200)
    print_report(scan_result, console, top=100)
    rendered = console.export_text()
    for value in _fetch_canaries():
        assert value not in rendered


@requires_dataset
def test_findings_json_contains_no_long_digit_runs(scan_result, tmp_path) -> None:
    """A defence-in-depth check that does not depend on knowing the values.

    Any run of 9 or more consecutive digits in the output would mean an
    unmasked identifier, whatever its source.
    """
    written = write_json(scan_result, tmp_path / "f.json").read_text()
    # Timestamps and run ids are allowed; strip them before scanning.
    stripped = re.sub(r'"(scanned_at|started_at|finished_at|run_id)":\s*"[^"]*"', "", written)
    offenders = re.findall(r"\d{9,}", stripped)
    assert not offenders, f"unmasked digit runs in output: {offenders[:3]}"


@requires_dataset
def test_sample_repr_is_redacted_on_real_data() -> None:
    config = ScanConfig(source=TEST_DSN, sample_size=50)
    connector = build_connector(config)
    with connector:
        sample = next(
            s for s in connector.iter_samples("piiscanner_testdata", 50)
            if s.schema.name == "aadhaar_no"
        )
        raw_values = [str(v) for v in sample.non_null]
        rendering = f"{sample!r} {sample!s}"
    assert raw_values
    for value in raw_values:
        assert value not in rendering
