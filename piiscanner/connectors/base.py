"""Connector interface.

Adding a data source should mean writing one new module here that subclasses
:class:`Connector` -- nothing in the classification or reporting layers should
need to change.

Every connector is read-only.  Implementations must open their connection in
whatever read-only mode the driver offers and must never issue a statement
that writes.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Callable, Iterator, TypeVar

from piiscanner.models import ConnectorInfo, FieldSchema

log = logging.getLogger("piiscanner.connectors")

DEFAULT_SAMPLE_SIZE = 500

#: Seconds to wait before each reconnect attempt after a connection drops
#: mid-scan.  Short and finite: this exists to ride out a flapping link or a
#: server recycling an idle connection, not to wait out an outage.
RECONNECT_BACKOFF = (1, 5, 15)

T = TypeVar("T")


#: How a table was sampled, as a connector records it in ``sample_methods``
#: and the linkage map reports it.  Shared by every connector.
SAMPLED_SPREAD = "spread"
SAMPLED_FIRST_ROWS_TOO_BIG = "first rows only: table too large to scan for a sample"
SAMPLED_FIRST_ROWS_TOO_SLOW = "first rows only: the sampling scan did not finish in time"


class ReadOnlyViolation(RuntimeError):
    """Raised when scanner code attempts something that could write.

    A bug that reaches this exception is a security defect, not a runtime
    hiccup -- connectors raise it rather than trying to recover.
    """


@dataclass
class FieldSample:
    """Sampled values for a single field.

    ``values`` holds raw data and exists only in memory for the lifetime of
    classification.  It must never be serialised, logged, or written to disk;
    only masked evidence derived from it may leave the process.  ``__repr__``
    is overridden so an accidental ``log.debug(sample)`` cannot leak.

    **Every count here is derived from ``values``, never stored.**  They used to
    be a mix of both: ``null_count`` was passed in by each connector counting
    ``None``, while ``non_null`` was computed here and excluded ``""`` as well.
    The two disagreed, and a column that was 62% empty string reported
    ``null_rate == 0.0``.  Deriving all of them from one list means they cannot
    drift apart again.

    Sampled values fall into exactly three disjoint buckets::

        null_count      +  empty_count  +  non_null_count  ==  total_sampled
        (v is None)        (v == "")       (everything else)
    """

    schema: FieldSchema
    values: list[Any] = field(default_factory=list)

    @property
    def total_sampled(self) -> int:
        """Values drawn for this field, including blanks.

        For MongoDB this is the number of values found for the path, which is
        smaller than the requested sample size when the path is sparse.
        """
        return len(self.values)

    @property
    def non_null(self) -> list[Any]:
        """The values classification actually looks at.

        Excludes both ``None`` and ``""``.  An empty string is *present* in the
        row but carries no content to classify, so it is no more useful than a
        NULL -- see :attr:`empty_count` for the count that keeps them apart.
        """
        return [v for v in self.values if v is not None and v != ""]

    @property
    def null_count(self) -> int:
        """Values that were SQL ``NULL`` / BSON ``null``.  Strictly ``None``."""
        return sum(1 for v in self.values if v is None)

    @property
    def empty_count(self) -> int:
        """Values present in the row but holding the empty string.

        Kept separate from :attr:`null_count` because the distinction is real:
        ``NULL`` usually means "never set", ``""`` usually means "set to
        nothing", and on a real schema they arise from different code paths.
        """
        return sum(1 for v in self.values if v is not None and v == "")

    @property
    def non_null_count(self) -> int:
        """Values with content -- the denominator classification uses."""
        return len(self.non_null)

    @property
    def null_rate(self) -> float:
        """``null_count / total_sampled``.  Strictly ``None``, not blanks.

        Deliberately narrow.  If you want "how much of this column is empty",
        use :attr:`blank_rate`; this one answers "how much of it is NULL".
        """
        if self.total_sampled == 0:
            return 0.0
        return self.null_count / self.total_sampled

    @property
    def blank_rate(self) -> float:
        """``(null_count + empty_count) / total_sampled``.

        The number a reviewer actually wants: the share of sampled rows that
        carry no content, however it was recorded.
        """
        if self.total_sampled == 0:
            return 0.0
        return (self.null_count + self.empty_count) / self.total_sampled

    @property
    def distinct_ratio(self) -> float:
        """Distinct values over :attr:`non_null_count`.

        The denominator is ``non_null_count`` -- the same set :attr:`non_null`
        returns and the same one the classifier gates on -- *not*
        ``total_sampled``.  So a column of 1,000 rows where 900 are blank and
        the remaining 100 are unique reports ``1.0``, not ``0.1``: of the values
        that exist, all are distinct.  Read it alongside :attr:`blank_rate`,
        which is over ``total_sampled``; the two use different denominators on
        purpose and neither is meaningful without the other.
        """
        non_null = self.non_null
        if not non_null:
            return 0.0
        return len({str(v) for v in non_null}) / len(non_null)

    def __repr__(self) -> str:  # pragma: no cover - trivial, but load-bearing
        return (
            f"FieldSample(field={self.schema.name!r}, "
            f"sampled={self.total_sampled}, nulls={self.null_count}, "
            f"empty={self.empty_count}, values=<redacted>)"
        )

    __str__ = __repr__


class Connector(ABC):
    """Read-only access to one data source."""

    #: Short identifier used in findings and reports, e.g. ``"mysql"``.
    kind: str = "unknown"

    def __init__(self, dsn: str, alias: str | None = None) -> None:
        self.dsn = dsn
        self.alias = alias or self.kind
        #: ``(dataset, collection, reason)`` for every table sampling could not
        #: read.  A scan covering hundreds of tables will eventually meet one
        #: the server refuses -- an odd collation, a permission gap, a corrupt
        #: page -- and losing every other table's results to it is worse than
        #: skipping it.  Never silently: the caller reports these, so an
        #: unscanned table is never mistaken for a clean one.
        self.sampling_errors: list[tuple[str, str, str]] = []

    # -- resilience ------------------------------------------------------
    #
    # A scan over hundreds of collections runs for minutes.  Two things go
    # wrong at that length that never show up in a unit test, and they need
    # opposite handling -- which is the whole point of splitting them:
    #
    #   * One object is unreadable (odd collation, permission gap, corrupt
    #     page).  Skip it, record it, keep the other 264 tables' results.
    #   * The connection dies (VPN drop, idle timeout, failover).  Every
    #     remaining object fails too.  Recorded per-object that reads as
    #     "these hold no personal data" when in fact none were looked at.
    #
    # So connectors route their per-object read through :meth:`with_reconnect`
    # and classify driver errors in :meth:`is_connection_lost`.

    def is_connection_lost(self, exc: BaseException) -> bool:
        """True when ``exc`` means the connection died, not that the request
        was bad.  Driver-specific, so subclasses override it; the default is
        conservative and treats nothing as recoverable."""
        return False

    def with_reconnect(self, what: str, read: Callable[[], T]) -> T:
        """Run ``read``, reconnecting and retrying if the connection drops.

        Backs off between attempts -- a link that just dropped is rarely back
        the millisecond after, so retrying instantly only proves the network is
        still down.  Once :data:`RECONNECT_BACKOFF` is exhausted the connection
        is gone rather than flapping, and the error propagates so the caller
        can end the scan honestly instead of manufacturing one skip per
        remaining object.
        """
        for pause in (*RECONNECT_BACKOFF, None):
            try:
                return read()
            except Exception as exc:  # noqa: BLE001 - re-raised unless recoverable
                if pause is None or not self.is_connection_lost(exc):
                    raise
                log.warning(
                    "connection lost while reading %s; reconnecting in %ss",
                    what, pause,
                )
                self.close()
                time.sleep(pause)
                self.connect()
        raise AssertionError("unreachable")  # pragma: no cover

    def record_skip(self, dataset: str, collection: str, exc: BaseException) -> None:
        """Note that ``collection`` could not be read, so the report can say so.

        Callers must re-raise first when :meth:`is_connection_lost` is true --
        see the class comment above for why the two cases must not merge.
        """
        self.sampling_errors.append((dataset, collection, str(exc)))
        log.warning("skipping %s.%s: %s", dataset, collection, exc)

    @abstractmethod
    def connect(self) -> None:
        """Open the connection in read-only mode."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection."""

    @abstractmethod
    def info(self) -> ConnectorInfo:
        """Driver and server versions, for scan provenance."""

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """Databases or schemas visible to these credentials."""

    @abstractmethod
    def get_schema(self, dataset: str) -> list[FieldSchema]:
        """Every scannable field in ``dataset``.

        Fields that cannot hold PII should be returned with ``skip=True`` and
        a reason rather than omitted, so the report can account for them.
        """

    @abstractmethod
    def sample_field(
        self, schema: FieldSchema, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> FieldSample:
        """Draw at most ``sample_size`` values for one field.

        Must never full-scan the underlying table.
        """

    def iter_samples(
        self, dataset: str, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Iterator[FieldSample]:
        """Sample every non-skipped field in ``dataset``.

        Connectors that can sample a whole table in one round trip should
        override this; the default is correct but chatty.
        """
        for schema in self.get_schema(dataset):
            if schema.skip:
                continue
            yield self.sample_field(schema, sample_size)

    def __enter__(self) -> "Connector":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
