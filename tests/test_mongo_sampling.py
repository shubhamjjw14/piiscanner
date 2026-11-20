"""Bounded-sampling tests for the MongoDB connector.

The MySQL connector learned to sample a huge table without scanning it; this
covers the same guarantee for MongoDB, where the shape of the problem differs
in one important way.

*``$sample`` is not always cheap*
    It uses a pseudo-random cursor only when the requested size is under 5% of
    the collection.  Above that the server silently falls back to a full
    collection scan feeding a top-k sort -- which at the default sample size of
    500 is *every* collection under 10,000 documents, and which runs against a
    100MB memory ceiling it can exceed outright on large documents.
    ``_choose_strategy`` decides on that basis instead of assuming.

*``_id`` is the universal seek key*
    Every collection has one and it is always indexed, so ``id_range`` probing
    is the direct analogue of the MySQL connector's ``key_range``.  ``_id`` can
    be an ObjectId, an int, or a string, and each interpolates differently.

*a dead connection is not a hundred empty collections*
    Same reasoning as MySQL: an unreadable collection is skipped and recorded,
    a dropped connection is retried and then allowed to end the scan.

None of this needs a live server -- the client is monkeypatched.
"""

from __future__ import annotations

import datetime
import random

import pymongo
import pytest
from bson import ObjectId

from piiscanner.connectors.mongodb import (
    MAX_CACHE_BYTES,
    MAX_CACHED_COLLECTIONS,
    MAX_SORT_BYTES,
    SMALL_COLLECTION_THRESHOLD,
    MongoDBConnector,
    _approx_size,
    _random_id_point,
)

DSN = "mongodb://127.0.0.1:27017/db"


def _connector(**kwargs) -> MongoDBConnector:
    return MongoDBConnector(DSN, **kwargs)


# --------------------------------------------------------------------------
# _choose_strategy
# --------------------------------------------------------------------------


def test_large_collection_uses_the_random_cursor() -> None:
    """500 of 10,000,000 is far under 5%, so $sample will not scan.  This is
    the best case and it is also the big-collection case."""
    assert _connector()._choose_strategy(10_000_000, 300, 500) == "sample"


def test_small_collection_accepts_the_scan_fallback() -> None:
    """500 of 5,000 is 10% -- the server will scan and sort.  That is fine
    here: reading 5,000 small documents is cheap and gives a true uniform
    sample, which is better than anything seeking can offer."""
    assert _connector()._choose_strategy(5_000, 200, 500) == "scan"


def test_just_under_five_percent_is_the_random_cursor() -> None:
    assert _connector()._choose_strategy(10_001, 200, 500) == "sample"


def test_exactly_five_percent_is_not(pytestconfig) -> None:
    """The server's rule is a strict inequality; predicting it wrongly is the
    only thing this constant can get wrong."""
    assert _connector()._choose_strategy(10_000, 200, 500) == "scan"


def test_many_documents_but_a_large_sample_seeks_instead_of_sorting() -> None:
    """The awkward middle: too many documents for a blocking sort, but the
    sample is too big a fraction for $sample to take the cheap path."""
    count = SMALL_COLLECTION_THRESHOLD + 1
    assert _connector()._choose_strategy(count, 200, count) == "id_range"


def test_big_documents_seek_rather_than_risk_the_sort_memory_limit() -> None:
    """MongoDB aborts a blocking sort at 100MB without allowDiskUse, and a
    read-only scanner should not make the customer's server spill to disk."""
    count = 5_000
    avg = (MAX_SORT_BYTES // count) + 1
    assert _connector()._choose_strategy(count, avg, 500) == "id_range"
    # Same document count, small documents -> the scan is affordable again.
    assert _connector()._choose_strategy(count, 10, 500) == "scan"


def test_unknown_count_falls_back_to_scanning() -> None:
    """A view or restricted namespace reports nothing; $sample is still the
    safe choice, and the server decides for itself what it costs."""
    assert _connector()._choose_strategy(0, 0, 500) == "scan"


# --------------------------------------------------------------------------
# _random_id_point -- ObjectId / int / str, and the refusal case
# --------------------------------------------------------------------------


def test_objectid_points_land_inside_the_range() -> None:
    lo = ObjectId.from_datetime(datetime.datetime(2024, 1, 1))
    hi = ObjectId.from_datetime(datetime.datetime(2025, 1, 1))
    rng = random.Random(3)
    points = [_random_id_point(lo, hi, rng) for _ in range(300)]
    assert all(isinstance(p, ObjectId) for p in points)
    assert all(lo <= p <= hi for p in points)
    # Genuinely spread, not one value repeated.
    assert len({str(p) for p in points}) > 250


def test_int_ids_land_inside_the_range() -> None:
    rng = random.Random(4)
    points = [_random_id_point(1, 5000, rng) for _ in range(300)]
    assert all(isinstance(p, int) for p in points)
    assert all(1 <= p <= 5000 for p in points)


def test_string_ids_land_inside_the_range() -> None:
    rng = random.Random(5)
    lo, hi = "00000000-0000", "ffffffff-ffff"
    for _ in range(200):
        assert lo <= _random_id_point(lo, hi, rng) <= hi


def test_a_bool_id_is_not_treated_as_an_int() -> None:
    """bool subclasses int, so an unguarded isinstance check would interpolate
    True/False into a meaningless 0/1 seek."""
    assert _random_id_point(False, True, random.Random(1)) is None


def test_an_uninterpolatable_id_returns_none_so_the_caller_can_fall_back() -> None:
    assert _random_id_point(1.5, 9.5, random.Random(1)) is None
    assert _random_id_point(b"aa", b"zz", random.Random(1)) is None


def test_degenerate_ranges_return_the_low_bound() -> None:
    rng = random.Random(1)
    only = ObjectId()
    assert _random_id_point(only, only, rng) == only
    assert _random_id_point(7, 7, rng) == 7


# --------------------------------------------------------------------------
# id_range sampling
# --------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction=1):
        # Must be real: the connector reads the collection's bounds as
        # sort(_id, 1).limit(1) and sort(_id, -1).limit(1), and a no-op sort
        # would hand back the low bound for both.
        self._docs = sorted(
            self._docs, key=lambda d: d[key], reverse=direction < 0
        )
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def max_time_ms(self, ms):
        self.max_time = ms
        return self

    def __iter__(self):
        return iter(self._docs)


class _FakeCollection:
    """An int-keyed collection, indexed by _id, that answers range queries.

    Documents are generated on demand rather than materialised: these tests
    stand in for collections of a hundred million documents, and building that
    list would cost far more than the code under test ever does.
    """

    def __init__(self, lo=1, hi=0, ids=None):
        if ids is not None:  # an explicit, small set of documents
            self._explicit = [{"_id": i, "v": f"row{i}"} for i in ids]
            self._lo, self._hi = 0, -1
        else:
            self._explicit = None
            self._lo, self._hi = lo, hi
        self.queries: list = []

    @property
    def _empty(self):
        return self._explicit == [] or (self._explicit is None and self._hi < self._lo)

    def _range(self, start, count):
        if self._explicit is not None:
            docs = [d for d in self._explicit if d["_id"] >= start]
            return docs[:count]
        first = max(start, self._lo)
        return [
            {"_id": i, "v": f"row{i}"}
            for i in range(first, min(first + count, self._hi + 1))
        ]

    def find(self, filt=None, projection=None):
        self.queries.append(filt)
        if self._empty:
            return _FakeCursor([])
        if filt and "_id" in filt:
            # A probe: a short forward run from the seek point.
            return _FakeCursor(self._range(filt["_id"]["$gte"], PROBE_WINDOW))
        # A bounds read: sort() decides which end, so hand back both.
        if self._explicit is not None:
            return _FakeCursor(list(self._explicit))
        return _FakeCursor([{"_id": self._lo}, {"_id": self._hi}])

    def aggregate(self, pipeline, **kwargs):
        self.queries.append(pipeline)
        size = pipeline[0]["$sample"]["size"]
        if self._explicit is not None:
            return iter(self._explicit[:size])
        return iter(self._range(self._lo, size))

    def estimated_document_count(self):
        if self._explicit is not None:
            return len(self._explicit)
        return max(0, self._hi - self._lo + 1)


#: Documents a probe can read forward; comfortably over PROBE_CHUNK so the
#: connector's own limit() is what truncates, not the fake.
PROBE_WINDOW = 200


def _wire(conn, collection, monkeypatch):
    database = {"c": collection}
    monkeypatch.setattr(
        type(conn), "client",
        property(lambda self: {"db": database}),
    )


def test_id_range_returns_distinct_documents(monkeypatch) -> None:
    conn = _connector(seed=1)
    coll = _FakeCollection(1, 100_000)
    _wire(conn, coll, monkeypatch)

    docs = conn._id_range_documents("db", "c", 100)

    assert len(docs) == 100
    assert len({d["_id"] for d in docs}) == 100  # de-duplicated across probes


def test_id_range_is_bounded_by_sample_size_not_collection_size(monkeypatch) -> None:
    """The whole point: a collection a thousand times bigger costs the same."""
    counts = []
    for size in (100_000, 100_000_000):
        conn = _connector(seed=1)
        coll = _FakeCollection(1, size)
        _wire(conn, coll, monkeypatch)
        conn._id_range_documents("db", "c", 100)
        counts.append(len([q for q in coll.queries if isinstance(q, dict)]))
    assert counts[0] == counts[1]


def test_id_range_sets_a_time_limit_on_every_probe(monkeypatch) -> None:
    conn = _connector(seed=1)
    coll = _FakeCollection(1, 10_000)
    cursors: list = []
    original = coll.find

    def spy(filt=None, projection=None):
        cur = original(filt, projection)
        cursors.append(cur)
        return cur

    coll.find = spy
    _wire(conn, coll, monkeypatch)
    conn._id_range_documents("db", "c", 50)

    probes = [c for c in cursors if hasattr(c, "max_time")]
    assert probes, "expected probes to carry a time limit"


def test_id_range_falls_back_to_sample_for_an_unusable_id_type(monkeypatch) -> None:
    conn = _connector(seed=1)
    coll = _FakeCollection(ids=[1.5, 9.5])
    _wire(conn, coll, monkeypatch)

    docs = conn._id_range_documents("db", "c", 10)

    assert docs  # got a sample anyway
    assert any(isinstance(q, list) for q in coll.queries)  # via $sample


def test_id_range_on_an_empty_collection_returns_nothing(monkeypatch) -> None:
    conn = _connector(seed=1)
    _wire(conn, _FakeCollection(ids=[]), monkeypatch)
    assert conn._id_range_documents("db", "c", 10) == []


# --------------------------------------------------------------------------
# cache is bounded by bytes, not just by collection count
# --------------------------------------------------------------------------


def test_cache_evicts_on_byte_budget_before_hitting_the_count_limit() -> None:
    conn = _connector()
    big = [{"path": ["x" * 4_000_000]} for _ in range(20)]  # ~80MB each
    for i in range(6):
        conn._cache[("db", f"c{i}", 500)] = big
        conn._evict()
    assert len(conn._cache) < MAX_CACHED_COLLECTIONS
    assert conn._cached_bytes() <= MAX_CACHE_BYTES or len(conn._cache) == 1


def test_cache_never_evicts_the_only_entry() -> None:
    """Evicting what was just sampled would loop forever re-fetching it."""
    conn = _connector()
    huge = [{"p": ["x" * (MAX_CACHE_BYTES * 2)]}]
    conn._cache[("db", "c", 500)] = huge
    conn._evict()
    assert len(conn._cache) == 1


def test_approx_size_counts_string_payloads() -> None:
    small = _approx_size([{"p": ["a"]}])
    large = _approx_size([{"p": ["a" * 10_000]}])
    assert large > small + 9_000


# --------------------------------------------------------------------------
# one unreadable collection vs a dead connection
# --------------------------------------------------------------------------


def test_a_failing_collection_is_recorded_and_skipped(monkeypatch) -> None:
    from piiscanner.models import FieldSchema

    conn = _connector()
    schemas = [
        FieldSchema(dataset="db", collection="bad", name="a", native_type="str"),
        FieldSchema(dataset="db", collection="good", name="b", native_type="str"),
    ]
    monkeypatch.setattr(conn, "get_schema", lambda dataset, size=None: schemas)

    def fake_sample(dataset, collection, sample_size):
        if collection == "bad":
            raise pymongo.errors.OperationFailure("not authorized on bad")
        return [{"b": ["value"]}]

    monkeypatch.setattr(conn, "_sample_documents", fake_sample)
    monkeypatch.setattr(conn, "_paths", lambda d, c, s: {"b": ["value"]})

    samples = list(conn.iter_samples("db", 10))

    assert [s.schema.collection for s in samples] == ["good"]
    assert len(conn.sampling_errors) == 1
    dataset, collection, reason = conn.sampling_errors[0]
    assert (dataset, collection) == ("db", "bad")
    assert "not authorized" in reason


def test_an_auth_failure_is_not_retried_as_a_connection_drop() -> None:
    """OperationFailure is the collection failing, not the link.  Retrying it
    three times with backoff would only slow the scan down."""
    conn = _connector()
    assert not conn.is_connection_lost(pymongo.errors.OperationFailure("nope"))
    assert not conn.is_connection_lost(
        pymongo.errors.ExecutionTimeout("maxTimeMS expired")
    )


def test_network_errors_are_recognised_as_connection_loss() -> None:
    conn = _connector()
    for exc in (
        pymongo.errors.AutoReconnect("primary stepped down"),
        pymongo.errors.ConnectionFailure("socket closed"),
        pymongo.errors.ServerSelectionTimeoutError("no replica reachable"),
        pymongo.errors.NetworkTimeout("timed out"),
    ):
        assert conn.is_connection_lost(exc), exc


def test_a_dead_network_ends_the_scan_instead_of_skipping_every_collection(
    monkeypatch,
) -> None:
    from piiscanner.connectors import base as base_mod
    from piiscanner.models import FieldSchema

    slept: list = []
    monkeypatch.setattr(base_mod.time, "sleep", slept.append)

    conn = _connector()
    schemas = [
        FieldSchema(dataset="db", collection="one", name="a", native_type="str"),
        FieldSchema(dataset="db", collection="two", name="b", native_type="str"),
    ]
    monkeypatch.setattr(conn, "get_schema", lambda dataset, size=None: schemas)
    monkeypatch.setattr(conn, "close", lambda: None)
    monkeypatch.setattr(conn, "connect", lambda: None)

    attempts: list = []

    def always_dead(dataset, collection, sample_size):
        attempts.append(collection)
        raise pymongo.errors.AutoReconnect("connection pool paused")

    monkeypatch.setattr(conn, "_sample_documents", always_dead)

    with pytest.raises(pymongo.errors.AutoReconnect):
        list(conn.iter_samples("db", 10))

    assert slept == list(base_mod.RECONNECT_BACKOFF)
    # Gave up on the first collection without writing off the second.
    assert set(attempts) == {"one"}


def test_seedless_connectors_probe_different_documents() -> None:
    """A fixed default seed would make every scheduled re-scan of a large
    collection land on exactly the same documents forever."""
    a = [_connector()._rng.random() for _ in range(5)]
    b = [_connector()._rng.random() for _ in range(5)]
    assert a != b


def test_an_explicit_seed_is_reproducible() -> None:
    a = [_connector(seed=7)._rng.random() for _ in range(5)]
    b = [_connector(seed=7)._rng.random() for _ in range(5)]
    assert a == b
