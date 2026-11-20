"""End-to-end quality suites against the loaded synthetic MongoDB dataset.

The MySQL suites cover shared behaviour.  These cover what only a document
store exercises: inferred schemas, deep nesting, PII inside arrays, and sparse
paths that appear in a fraction of documents.
"""

from __future__ import annotations

import pytest

from piiscanner.benchmark import format_score, score_scan
from piiscanner.classify.engine import run_scan
from piiscanner.config import ScanConfig, build_connector
from piiscanner.connectors.base import ReadOnlyViolation
from piiscanner.models import PIIType
from piiscanner.testdata.generate import (
    DATABASE_NAME,
    MONGO_EXPECTED,
    MONGO_FORBIDDEN,
    MONGO_GROUND_TRUTH,
)
from tests.conftest import MONGO_DSN, requires_mongo

pytestmark = requires_mongo


# --------------------------------------------------------------------------
# Recall
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expectation",
    MONGO_EXPECTED,
    ids=[f"{e.collection}.{e.field}" for e in MONGO_EXPECTED],
)
def test_planted_pii_is_found(mongo_scan_result, expectation) -> None:
    found = [
        f for f in mongo_scan_result.findings
        if (f.collection, f.field) == (expectation.collection, expectation.field)
    ]
    assert found, (
        f"{expectation.collection}.{expectation.field} was missed "
        f"(expected {expectation.pii_type.value})"
    )
    assert expectation.pii_type in {f.pii_type for f in found}


def test_pii_four_levels_deep_is_found(mongo_scan_result) -> None:
    """`account.owner.identity.aadhaar` -- nothing shallow about it."""
    finding = next(
        f for f in mongo_scan_result.findings
        if f.field == "account.owner.identity.aadhaar"
    )
    assert finding.pii_type is PIIType.AADHAAR
    assert finding.confidence >= 0.95


def test_pii_inside_an_array_of_objects_is_found(mongo_scan_result) -> None:
    """Array indices merge, so this is one field and not five hundred."""
    findings = [
        f for f in mongo_scan_result.findings
        if f.field == "recipients[].contact.mobile"
    ]
    assert len(findings) == 1
    assert findings[0].pii_type is PIIType.INDIAN_MOBILE


def test_no_array_index_leaks_into_a_field_path(mongo_scan_result) -> None:
    """A path like `recipients.3.contact.mobile` means the merge failed."""
    import re

    for finding in mongo_scan_result.findings:
        assert not re.search(r"\.\d+\.", f".{finding.field}."), finding.field


def test_sparse_path_is_found_and_its_thin_sample_is_recorded(mongo_scan_result) -> None:
    """Present in ~8% of documents: still a finding, with an honest count."""
    finding = next(
        f for f in mongo_scan_result.findings
        if f.field == "account.owner.identity.passport"
    )
    dense = next(
        f for f in mongo_scan_result.findings
        if f.field == "account.owner.identity.aadhaar"
    )
    assert finding.sample_count < dense.sample_count / 5
    assert 0 < finding.sample_count < 120


# --------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection,field,why",
    MONGO_FORBIDDEN,
    ids=[f"{c}.{f}" for c, f, _ in MONGO_FORBIDDEN],
)
def test_decoy_paths_produce_no_findings(
    mongo_scan_result, collection, field, why
) -> None:
    hits = [
        f for f in mongo_scan_result.findings
        if (f.collection, f.field) == (collection, field)
    ]
    assert not hits, (
        f"false positive on {collection}.{field} ({why}): "
        f"flagged as {[h.pii_type.value for h in hits]}"
    )


def test_precision_and_recall_on_mongo(mongo_scan_result) -> None:
    score = score_scan(mongo_scan_result, MONGO_GROUND_TRUTH)
    assert score.recall == 1.0, "\n" + format_score(score)
    assert score.precision == 1.0, "\n" + format_score(score)
    assert score.type_accuracy == 1.0, "\n" + format_score(score)


# --------------------------------------------------------------------------
# Schema inference and cost
# --------------------------------------------------------------------------


def test_object_ids_are_skipped() -> None:
    config = ScanConfig(source=MONGO_DSN, sample_size=100, datasets=[DATABASE_NAME])
    connector = build_connector(config)
    with connector:
        schemas = connector.get_schema(DATABASE_NAME)
    ids = [s for s in schemas if s.name == "_id"]
    assert ids
    # These are integers in the synthetic data, so they are scannable but must
    # still produce no findings; the skip path is exercised by ObjectId data.
    assert all(s.native_type in {"int", "ObjectId"} for s in ids)


def test_booleans_are_skipped() -> None:
    config = ScanConfig(source=MONGO_DSN, sample_size=100, datasets=[DATABASE_NAME])
    connector = build_connector(config)
    with connector:
        schemas = connector.get_schema(DATABASE_NAME)
    is_active = next(s for s in schemas if s.name == "is_active")
    assert is_active.skip
    assert "bool" in (is_active.skip_reason or "")


def test_presence_count_is_recorded_for_sparse_paths() -> None:
    config = ScanConfig(source=MONGO_DSN, sample_size=500, datasets=[DATABASE_NAME])
    connector = build_connector(config)
    with connector:
        schemas = connector.get_schema(DATABASE_NAME)
    sparse = next(
        s for s in schemas if s.name == "account.owner.identity.passport"
    )
    dense = next(s for s in schemas if s.name == "account.owner.identity.aadhaar")
    assert sparse.presence_count and dense.presence_count
    assert sparse.presence_count < dense.presence_count / 5


def test_each_collection_is_sampled_once_per_scan(monkeypatch) -> None:
    """One ``$sample`` per collection, not three.

    This is a cost guarantee and a consistency one: ``$sample`` returns
    different documents each call, so re-sampling would mean the inferred
    schema and the classified values came from different sets of documents.
    """
    from piiscanner.connectors.mongodb import MongoDBConnector

    config = ScanConfig(source=MONGO_DSN, sample_size=200, datasets=[DATABASE_NAME])
    connector = build_connector(config)

    round_trips: list[tuple[str, str]] = []
    original = MongoDBConnector._sample_documents

    def counting(self, dataset, collection, sample_size):
        key = (dataset, collection, sample_size)
        if key not in self._cache:
            round_trips.append((dataset, collection))
        return original(self, dataset, collection, sample_size)

    monkeypatch.setattr(MongoDBConnector, "_sample_documents", counting)

    with connector:
        collections = connector.client[DATABASE_NAME].list_collection_names()
        run_scan(connector, config)

    assert sorted(c for _, c in round_trips) == sorted(collections)


# --------------------------------------------------------------------------
# Read-only enforcement
# --------------------------------------------------------------------------


def test_write_stages_are_rejected_in_practice() -> None:
    from piiscanner.connectors.mongodb import assert_read_only_pipeline

    with pytest.raises(ReadOnlyViolation):
        assert_read_only_pipeline([{"$out": "stolen_copy"}])


def test_scan_does_not_change_the_data(mongo_scan_result) -> None:
    import pymongo

    client = pymongo.MongoClient(MONGO_DSN)
    try:
        database = client[DATABASE_NAME]
        before = {
            name: database[name].estimated_document_count()
            for name in database.list_collection_names()
        }
        config = ScanConfig(source=MONGO_DSN, sample_size=200, datasets=[DATABASE_NAME])
        connector = build_connector(config)
        with connector:
            run_scan(connector, config)
        after = {
            name: database[name].estimated_document_count()
            for name in database.list_collection_names()
        }
    finally:
        client.close()
    assert before == after
