"""Scan configuration and connector selection."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

from piiscanner.connectors.base import DEFAULT_SAMPLE_SIZE, Connector


@dataclass
class ScanConfig:
    """Everything one scan run needs to know."""

    source: str
    alias: str | None = None
    sample_size: int = DEFAULT_SAMPLE_SIZE
    #: Glob patterns matched against ``table`` and ``dataset.table``.
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    #: Restrict to specific databases; empty means every non-system database.
    datasets: list[str] = field(default_factory=list)
    seed: int | None = None
    #: Record every field examined, not just the findings.  Off by default: the
    #: inventory is roughly ten times the size of the findings list, and only
    #: ``piiscanner audit`` needs it.
    record_inventory: bool = False
    #: Where a keyless MySQL table's ``head`` sweep position persists across
    #: runs (see ``MySQLConnector._head_sample``).  ``None`` disables it: every
    #: run reads the same first rows, as before.  Ignored by MongoDB, whose
    #: ``$sample`` is already randomised per call.
    cursor_path: str | None = None

    def matches(self, dataset: str, collection: str) -> bool:
        """True when a table should be scanned, per the include/exclude globs.

        Exclude always wins.  With no include patterns, everything not
        excluded is in scope.
        """
        targets = (collection, f"{dataset}.{collection}")
        if any(fnmatch.fnmatch(t, p) for t in targets for p in self.exclude):
            return False
        if not self.include:
            return True
        return any(fnmatch.fnmatch(t, p) for t in targets for p in self.include)


def build_connector(config: ScanConfig) -> Connector:
    """Instantiate the right connector for the DSN's scheme.

    Adding a source means adding a branch here and one new connector module --
    nothing else in the codebase changes.
    """
    scheme = urlparse(config.source).scheme.lower()
    if scheme in {"mysql", "mysql+pymysql"}:
        from piiscanner.connectors.mysql import MySQLConnector

        return MySQLConnector(
            config.source,
            alias=config.alias,
            sample_size=config.sample_size,
            seed=config.seed,
            cursor_path=config.cursor_path,
        )
    if scheme in {"postgresql", "postgres"}:
        from piiscanner.connectors.postgres import PostgresConnector

        return PostgresConnector(
            config.source,
            alias=config.alias,
            sample_size=config.sample_size,
            seed=config.seed,
            cursor_path=config.cursor_path,
        )
    if scheme in {"mongodb", "mongodb+srv"}:
        from piiscanner.connectors.mongodb import MongoDBConnector

        return MongoDBConnector(
            config.source,
            alias=config.alias,
            sample_size=config.sample_size,
            seed=config.seed,
        )
    raise ValueError(
        f"unsupported source scheme {scheme!r}; supported: mysql://, postgresql:// and mongodb://"
    )


class NoValuesFilter(logging.Filter):
    """Belt-and-braces guard against a log record carrying a sampled value.

    The code paths that log are written to emit field paths and counts only.
    This filter is the backstop: it drops any record whose message somehow
    contains a raw ``FieldSample`` payload.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "values=[" not in record.getMessage()


def configure_logging(verbose: bool = False) -> None:
    """Set up logging that records what was scanned, never what was found.

    Quiet by default: warnings and errors only.  Progress is shown by the
    commands themselves; the log is for working out what went wrong.  Left at
    ``INFO`` it put 197 lines of connector chatter into a 400-line demo, half
    of everything a first-time viewer saw.  ``verbose`` turns everything on.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(NoValuesFilter())
    root = logging.getLogger("piiscanner")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.propagate = False
