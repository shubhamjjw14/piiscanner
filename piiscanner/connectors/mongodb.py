"""MongoDB connector.

MongoDB has no fixed schema, so the schema is inferred: ``$sample`` pulls a
set of documents, and each one is flattened into dotted paths.  Two properties
matter for correctness here.

**Array indices are merged.**  ``orders.0.pincode`` and ``orders.7.pincode``
are the same field, and treating them separately would give one real field
hundreds of one-document "schemas".  Both flatten to ``orders[].pincode``.

**Presence is tracked separately from the sample size.**  A path found in 12
of 500 documents is a sparse field, not a field with 488 nulls, and the
distinction changes what the match rate means.

Read-only: the connector issues only ``find``, ``aggregate``, and metadata
reads.  :func:`assert_read_only_pipeline` rejects ``$out`` and ``$merge``,
which are the two aggregation stages that can write.

**Sampling is bounded, and ``$sample`` alone does not bound it.**  ``$sample``
uses a genuinely cheap pseudo-random cursor only when it is the first stage and
the requested size is under 5% of the collection -- otherwise the server
silently falls back to a full collection scan feeding a top-k sort.  At the
default sample size of 500 that fallback covers every collection under 10,000
documents, and the sort runs against a 100MB memory ceiling it can exceed
outright on a collection of large documents.  :meth:`MongoDBConnector.
_choose_strategy` picks between ``$sample`` and ``_id``-range seeking on that
basis rather than assuming ``$sample`` is always cheap.
"""

from __future__ import annotations

import logging
import random
from collections import OrderedDict, defaultdict
from typing import Any, Iterator

import pymongo
from bson import ObjectId
from pymongo import MongoClient

from piiscanner.connectors.base import (
    DEFAULT_SAMPLE_SIZE,
    Connector,
    FieldSample,
    ReadOnlyViolation,
)
# Shared with the MySQL connector: interpolating between two string keys is the
# same problem whether the key is a varchar primary key or a string _id.
from piiscanner.connectors.keyspace import interpolate_string
from piiscanner.models import ConnectorInfo, FieldSchema

log = logging.getLogger("piiscanner.connectors.mongodb")

#: Databases that belong to the server, not the customer.
SYSTEM_DATABASES = frozenset({"admin", "local", "config"})

#: Aggregation stages that write.  Never allowed.
WRITE_STAGES = frozenset({"$out", "$merge"})

#: Sampled collections held in memory at once.  Bounded so that scanning a
#: database with hundreds of collections cannot exhaust memory; eviction only
#: costs a re-sample if a collection is revisited.
MAX_CACHED_COLLECTIONS = 16

#: Guard against pathological documents blowing up the flattener.
MAX_DEPTH = 12
#: Only this many elements of any one array are walked.  Beyond it the field's
#: shape is established and further elements add cost, not information.
MAX_ARRAY_ELEMENTS = 20

#: BSON types that cannot hold PII.
UNSCANNABLE_BSON = (bool, bytes, type(None))

#: The fraction of a collection below which ``$sample`` uses its pseudo-random
#: cursor instead of scanning.  This is the server's own documented threshold,
#: not a tuning knob -- changing it here does not change what the server does,
#: it only changes whether we predict the server correctly.
SAMPLE_RANDOM_CURSOR_FRACTION = 0.05

#: Documents ``$sample``'s scan-and-sort fallback may touch before we stop
#: trusting it.  Below this a full scan is genuinely cheap and gives a true
#: uniform sample, which is better than anything seeking can offer.
SMALL_COLLECTION_THRESHOLD = 20_000

#: Bytes the fallback's top-k sort may be estimated to hold.  MongoDB aborts a
#: blocking sort at 100MB unless allowDiskUse is set, and spilling to disk on
#: the customer's server is not something a read-only scanner should cause.
#: Estimated as ``count * avgObjSize``; over this we seek instead.
MAX_SORT_BYTES = 32 * 1024 * 1024

#: Documents read per probe when ``_id``-range sampling, mirroring the MySQL
#: connector's PROBE_CHUNK: an index seek plus a short forward read.
PROBE_CHUNK = 25

#: Ceiling on how long any single server operation may run.  Without it one
#: pathological collection can hang a scan indefinitely -- and the server keeps
#: working on a query whose client has gone away.
MAX_TIME_MS = 30_000

#: Total bytes of sampled documents the cache may hold.  ``MAX_CACHED_COLLECTIONS``
#: bounds how many collections are cached but says nothing about how big they
#: are, and sixteen collections of large documents is gigabytes.
MAX_CACHE_BYTES = 256 * 1024 * 1024


def _random_id_point(lo: Any, hi: Any, rng: random.Random) -> Any:
    """A random ``_id`` value between ``lo`` and ``hi``, or ``None``.

    ``_id`` is whatever the application made it.  Three shapes cover
    essentially every real collection, and anything else returns ``None`` so
    the caller can fall back rather than guess:

    ``ObjectId``  the default.  Its leading four bytes are a Unix timestamp
                  and the rest is counter/random, so the raw 12 bytes read as
                  a big-endian integer are ordered the same way the index is
                  -- interpolating over that integer lands in range.
    ``int``       an application-assigned sequence.
    ``str``       a UUID or similar; interpolated by the same base-N trick the
                  MySQL connector uses for varchar keys.
    """
    if isinstance(lo, ObjectId) and isinstance(hi, ObjectId):
        lo_n = int.from_bytes(lo.binary, "big")
        hi_n = int.from_bytes(hi.binary, "big")
        if hi_n <= lo_n:
            return lo
        return ObjectId(rng.randint(lo_n, hi_n).to_bytes(12, "big"))
    # bool is an int subclass and would interpolate to nonsense; exclude it.
    if isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool):
        return rng.randint(lo, hi) if hi > lo else lo
    if isinstance(lo, str) and isinstance(hi, str):
        return interpolate_string(lo, hi, rng)
    return None


def _approx_size(flattened: list[dict[str, list[Any]]]) -> int:
    """Rough byte size of one collection's flattened sample.

    Deliberately cheap and approximate: this decides whether to evict, and
    paying an exact accounting per entry would cost more than the eviction
    saves.  Path names and value lengths dominate real documents.
    """
    total = 0
    for document in flattened:
        for path, values in document.items():
            total += len(path) + 16
            for value in values:
                total += len(value) if isinstance(value, (str, bytes)) else 16
    return total


def assert_read_only_pipeline(pipeline: list[dict[str, Any]]) -> None:
    """Reject any aggregation pipeline containing a write stage."""
    for stage in pipeline:
        for key in stage:
            if key in WRITE_STAGES:
                raise ReadOnlyViolation(
                    f"refusing to run an aggregation containing {key}"
                )


def flatten_document(
    document: dict[str, Any], max_depth: int = MAX_DEPTH
) -> dict[str, list[Any]]:
    """Flatten a document into ``{dotted_path: [values]}``.

    Nested objects become dotted paths and arrays become ``[]`` segments, so
    array indices collapse into a single field::

        {"user": {"contact": {"phone": "9876543210"}}}
            -> {"user.contact.phone": ["9876543210"]}

        {"orders": [{"shipping": {"pincode": "400001"}},
                    {"shipping": {"pincode": "560001"}}]}
            -> {"orders[].shipping.pincode": ["400001", "560001"]}

    Empty objects and empty arrays yield no paths: there is nothing to
    classify, and inventing a path for them would inflate the field count.
    """
    out: dict[str, list[Any]] = defaultdict(list)

    def walk(value: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(value, (list, tuple)):
            array_path = f"{path}[]" if path else "[]"
            for item in list(value)[:MAX_ARRAY_ELEMENTS]:
                walk(item, array_path, depth + 1)
        elif path:
            out[path].append(value)

    walk(document, "", 0)
    return dict(out)


def infer_bson_type(values: list[Any]) -> str:
    """Name the dominant BSON type in a list of sampled values."""
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value is None:
            continue
        counts[type(value).__name__] += 1
    if not counts:
        return "null"
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1:
        return f"mixed({', '.join(name for name, _ in ranked[:3])})"
    return ranked[0][0]


def _is_scannable(values: list[Any]) -> tuple[bool, str | None]:
    """False when no value in the list could possibly be PII."""
    meaningful = [v for v in values if v is not None]
    if not meaningful:
        return False, "no non-null values sampled"
    if all(isinstance(v, UNSCANNABLE_BSON) for v in meaningful):
        return False, f"type {type(meaningful[0]).__name__} cannot hold PII"
    if all(isinstance(v, ObjectId) for v in meaningful):
        return False, "ObjectId is a server-generated key, not personal data"
    return True, None


class MongoDBConnector(Connector):
    """Read-only sampling access to a MongoDB deployment."""

    kind = "mongodb"

    def __init__(
        self,
        dsn: str,
        alias: str | None = None,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        seed: int | None = None,
    ) -> None:
        super().__init__(dsn, alias)
        self.sample_size = sample_size
        self._client: MongoClient | None = None
        self._seed = seed
        # No seed means random.Random(None), which seeds itself from OS
        # entropy -- a fresh sequence every run.  A fixed default here would
        # make every _id probe, and so every scheduled re-scan of a large
        # collection, land on exactly the same documents forever.
        self._rng = random.Random(seed)
        #: (database, collection, sample_size) -> flattened documents.
        #:
        #: A dict rather than a single slot, and it matters for correctness as
        #: well as cost: ``$sample`` is random, so re-sampling a collection
        #: returns *different* documents.  With a one-entry cache, schema
        #: inference and value sampling saw different data and every
        #: collection was hit three times per scan.
        self._cache: OrderedDict[
            tuple[str, str, int], list[dict[str, list[Any]]]
        ] = OrderedDict()

    # -- connection ------------------------------------------------------

    def connect(self) -> None:
        self._client = MongoClient(
            self.dsn,
            serverSelectionTimeoutMS=5000,
            connect=True,
            # Prefer a secondary when the deployment has one: scanning should
            # not compete with production traffic on the primary.
            readPreference="secondaryPreferred",
        )
        self._client.admin.command("ping")
        log.info("connected to mongodb")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._cache.clear()

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            raise RuntimeError("connector is not connected; call connect() first")
        return self._client

    def info(self) -> ConnectorInfo:
        version = None
        if self._client is not None:
            try:
                version = self.client.server_info().get("version")
            except Exception:  # pragma: no cover - server may restrict buildInfo
                version = None
        return ConnectorInfo(
            alias=self.alias,
            kind=self.kind,
            driver_version=f"pymongo {pymongo.__version__}",
            server_version=version,
        )

    @property
    def default_dataset(self) -> str | None:
        name = self.client.get_default_database().name if self._client else None
        return None if name == "test" else name

    # -- schema ----------------------------------------------------------

    def list_datasets(self) -> list[str]:
        names = self.client.list_database_names()
        return sorted(n for n in names if n not in SYSTEM_DATABASES)

    def _collection_stats(self, dataset: str, collection: str) -> tuple[int, int]:
        """``(document count, average document size in bytes)``.

        Both come from collection metadata, not from reading documents, so
        this costs one cheap command regardless of collection size.  Returns
        zeros when the server will not answer -- for a view, a restricted
        namespace, or a driver that has dropped ``collStats``.
        """
        try:
            count = self.client[dataset][collection].estimated_document_count()
        except Exception:  # pragma: no cover - view or restricted collection
            return 0, 0
        try:
            stats = self.client[dataset].command("collstats", collection)
            avg = int(stats.get("avgObjSize") or 0)
        except Exception:  # pragma: no cover - not available on all deployments
            avg = 0
        return int(count or 0), avg

    def _choose_strategy(self, count: int, avg_size: int, sample_size: int) -> str:
        """Pick how to draw ``sample_size`` documents, and why.

        ``sample``    ``$sample`` with its pseudo-random cursor: the request is
                      a small enough fraction of the collection that the server
                      will not scan.  Cost is bounded by the sample size, not
                      the collection size, so this is what large collections
                      get -- and it is a genuinely uniform sample.

        ``scan``      ``$sample``'s scan-and-sort fallback, accepted on purpose.
                      The collection is small enough that reading all of it is
                      cheap, and a full scan yields a better sample than seeking
                      can.  This is the analogue of the MySQL connector's
                      ``order_by_rand``.

        ``id_range``  Seek to random ``_id`` positions and read a short run from
                      each.  For the awkward middle: too many documents or too
                      many bytes for the fallback's blocking sort, but not few
                      enough of them relative to the sample for ``$sample`` to
                      take the cheap path.  Not uniform -- ``_id`` order is
                      roughly insertion order -- but bounded, which the sort
                      is not.
        """
        if count and sample_size < count * SAMPLE_RANDOM_CURSOR_FRACTION:
            return "sample"
        # Below here the server would scan and sort.  That is fine while the
        # collection is small in both documents and bytes, and not otherwise.
        if count and count > SMALL_COLLECTION_THRESHOLD:
            return "id_range"
        if avg_size and count * avg_size > MAX_SORT_BYTES:
            return "id_range"
        return "scan"

    def _sample_documents(
        self, dataset: str, collection: str, sample_size: int
    ) -> list[dict[str, list[Any]]]:
        """Draw documents by the strategy this collection's size calls for."""
        key = (dataset, collection, sample_size)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        count, avg_size = self._collection_stats(dataset, collection)
        strategy = self._choose_strategy(count, avg_size, sample_size)
        if strategy == "id_range":
            documents = self._id_range_documents(dataset, collection, sample_size)
        else:
            documents = self._aggregate_sample(dataset, collection, sample_size)

        flattened = [flatten_document(doc) for doc in documents]
        self._cache[key] = flattened
        self._evict()
        log.info(
            "sampled %s.%s: %d documents (%s, ~%d docs)",
            dataset, collection, len(flattened), strategy, count,
        )
        return flattened

    def _aggregate_sample(
        self, dataset: str, collection: str, sample_size: int
    ) -> list[dict[str, Any]]:
        pipeline: list[dict[str, Any]] = [{"$sample": {"size": int(sample_size)}}]
        assert_read_only_pipeline(pipeline)
        cursor = self.client[dataset][collection].aggregate(
            pipeline, maxTimeMS=MAX_TIME_MS
        )
        return list(cursor)

    def _id_range_documents(
        self, dataset: str, collection: str, sample_size: int
    ) -> list[dict[str, Any]]:
        """Seek to random ``_id`` positions and read a short run from each.

        The direct analogue of the MySQL connector's ``key_range``: ``_id`` is
        indexed on every collection without exception, so each probe is an
        index seek plus a short forward read, and the cost depends on the
        sample size rather than on how many documents exist.  Probes can
        overlap, so documents are de-duplicated by ``_id``.

        Falls back to ``$sample`` if the bounds cannot be read or ``_id`` turns
        out not to be interpolatable -- a collection keyed by an arbitrary
        BSON type is unusual but not an error, and a scan of it is better than
        no sample at all.
        """
        handle = self.client[dataset][collection]
        low = list(handle.find({}, {"_id": 1}).sort("_id", 1).limit(1))
        high = list(handle.find({}, {"_id": 1}).sort("_id", -1).limit(1))
        if not low or not high:
            return []
        lo, hi = low[0]["_id"], high[0]["_id"]
        point = _random_id_point(lo, hi, self._rng)
        if point is None:
            return self._aggregate_sample(dataset, collection, sample_size)

        collected: dict[Any, dict[str, Any]] = {}
        probes_needed = max(1, sample_size // PROBE_CHUNK)
        # Cap the probe count so a sparse key space cannot loop for long.
        for _ in range(probes_needed * 4):
            if len(collected) >= sample_size:
                break
            start = _random_id_point(lo, hi, self._rng)
            if start is None:  # pragma: no cover - bounds proven usable above
                break
            cursor = (
                handle.find({"_id": {"$gte": start}})
                .sort("_id", 1)
                .limit(PROBE_CHUNK)
                .max_time_ms(MAX_TIME_MS)
            )
            for document in cursor:
                collected[document["_id"]] = document
        return list(collected.values())[:sample_size]

    def _evict(self) -> None:
        """Hold the cache to both a collection count and a byte budget.

        Counting collections alone does not bound memory: sixteen collections
        of large documents is gigabytes.  Size is measured with the BSON
        encoding rather than ``sys.getsizeof`` because that is what the values
        actually are, and it does not need to be exact -- only to stop a wide
        collection from sitting in the cache alongside fifteen others.
        """
        while len(self._cache) > MAX_CACHED_COLLECTIONS or (
            len(self._cache) > 1 and self._cached_bytes() > MAX_CACHE_BYTES
        ):
            evicted, _ = self._cache.popitem(last=False)
            log.debug("evicted sample cache for %s.%s", evicted[0], evicted[1])

    def _cached_bytes(self) -> int:
        return sum(_approx_size(entry) for entry in self._cache.values())

    def _read_collection(
        self, dataset: str, collection: str, sample_size: int
    ) -> list[dict[str, list[Any]]]:
        """Sample one collection, surviving a connection that drops."""
        return self.with_reconnect(
            f"{dataset}.{collection}",
            lambda: self._sample_documents(dataset, collection, sample_size),
        )

    def is_connection_lost(self, exc: BaseException) -> bool:
        """True for pymongo's network-level failures only.

        ``OperationFailure`` is deliberately absent: an unauthorised namespace
        or a rejected pipeline is the *collection* failing, not the link, and
        retrying it three times with backoff would only slow the scan down.
        ``ExecutionTimeout`` likewise -- that is MAX_TIME_MS doing its job.
        """
        return isinstance(
            exc,
            (
                pymongo.errors.AutoReconnect,
                pymongo.errors.ConnectionFailure,
                pymongo.errors.ServerSelectionTimeoutError,
                pymongo.errors.NetworkTimeout,
            ),
        )

    def _paths(
        self, dataset: str, collection: str, sample_size: int
    ) -> dict[str, list[Any]]:
        """Merge per-document paths into ``{path: all sampled values}``."""
        merged: dict[str, list[Any]] = defaultdict(list)
        for document in self._sample_documents(dataset, collection, sample_size):
            for path, values in document.items():
                merged[path].extend(values)
        return dict(merged)

    def _presence(
        self, dataset: str, collection: str, sample_size: int
    ) -> dict[str, int]:
        """How many sampled documents contain each path."""
        counts: dict[str, int] = defaultdict(int)
        for document in self._sample_documents(dataset, collection, sample_size):
            for path in document:
                counts[path] += 1
        return dict(counts)

    def get_schema(self, dataset: str, sample_size: int | None = None) -> list[FieldSchema]:
        size = sample_size or self.sample_size
        database = self.client[dataset]
        schemas: list[FieldSchema] = []
        for collection in sorted(database.list_collection_names()):
            try:
                # One read per collection, behind the shared reconnect policy.
                # Everything below reads from the cache it fills.
                self._read_collection(dataset, collection, size)
            except Exception as exc:  # noqa: BLE001 - recorded, then reported
                if self.is_connection_lost(exc):
                    raise
                self.record_skip(dataset, collection, exc)
                continue
            row_count = self._collection_stats(dataset, collection)[0] or None
            paths = self._paths(dataset, collection, size)
            presence = self._presence(dataset, collection, size)
            sampled_docs = len(
                self._sample_documents(dataset, collection, size)
            )
            for path, values in sorted(paths.items()):
                scannable, reason = _is_scannable(values)
                schemas.append(
                    FieldSchema(
                        dataset=dataset,
                        collection=collection,
                        name=path,
                        native_type=infer_bson_type(values),
                        nullable=presence.get(path, 0) < sampled_docs,
                        approx_row_count=row_count,
                        presence_count=presence.get(path, 0),
                        skip=not scannable,
                        skip_reason=reason,
                    )
                )
        return schemas

    # -- sampling --------------------------------------------------------

    def sample_field(
        self, schema: FieldSchema, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> FieldSample:
        paths = self._paths(schema.dataset, schema.collection, sample_size)
        values = paths.get(schema.name, [])
        return self._build_sample(schema, values, sample_size)

    @staticmethod
    def _build_sample(
        schema: FieldSchema, values: list[Any], sample_size: int
    ) -> FieldSample:
        """Wrap raw values; every count is derived from them by FieldSample.

        ``total_sampled`` is therefore the number of values found for this
        path, not the requested sample size -- a path present in 12 of 500
        documents yields a sample of 12, and the engine's thin-sample penalty
        applies.
        """
        return FieldSample(schema=schema, values=list(values))

    def iter_samples(
        self, dataset: str, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Iterator[FieldSample]:
        """Sample every inferred path, one ``$sample`` per collection."""
        schemas = self.get_schema(dataset, sample_size)
        by_collection: dict[str, list[FieldSchema]] = defaultdict(list)
        for schema in schemas:
            if not schema.skip:
                by_collection[schema.collection].append(schema)

        for collection, fields in by_collection.items():
            try:
                self._read_collection(dataset, collection, sample_size)
            except Exception as exc:  # noqa: BLE001 - recorded, then reported
                if self.is_connection_lost(exc):
                    raise
                self.record_skip(dataset, collection, exc)
                continue
            paths = self._paths(dataset, collection, sample_size)
            for schema in fields:
                yield self._build_sample(
                    schema, paths.get(schema.name, []), sample_size
                )
