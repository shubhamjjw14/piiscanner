"""Find the personal data in a database, without writing to it or keeping any of it.

    import piiscanner

    result = piiscanner.scan("mysql://readonly:PASSWORD@host:3306/shop")
    for f in result.findings:
        print(f.collection, f.field, f.pii_type.value, f.confidence, f.evidence)

Works on MySQL, PostgreSQL and MongoDB.  Every session is read-only, every
statement is checked to be a read before it is sent, and evidence is masked
when a finding is created (``9876543210`` becomes ``98XXXXXX10``): no raw
value is ever returned, stored or logged.
"""

from __future__ import annotations

from collections.abc import Iterable

from piiscanner.models import Finding, PIIType, ScanResult

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("piiscanner")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["scan", "ScanResult", "Finding", "PIIType", "__version__"]


def scan(
    source: str,
    *,
    sample_size: int = 500,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
) -> ScanResult:
    """Scan a database and return the personal data found in it.

    ``source`` is a connection string: ``mysql://``, ``postgresql://`` or
    ``mongodb://``.  Use an account that can only read.

    ``sample_size`` is how many values are read per column.  ``include`` and
    ``exclude`` are table-name patterns (``"prod_*"``).  ``datasets`` names
    the databases or schemas to scan; left out, it is the one named in
    ``source``, or every one the account can see if ``source`` names none.

    The result's ``findings`` are what was found; ``tables_not_read`` lists
    tables that could not be read -- not covered, rather than clean.
    """
    from piiscanner.classify.engine import run_scan
    from piiscanner.config import ScanConfig, build_connector

    config = ScanConfig(
        source=source,
        sample_size=sample_size,
        include=list(include or []),
        exclude=list(exclude or []),
        datasets=list(datasets or []),
    )
    connector = build_connector(config)
    with connector:
        # As on the command line: a connection string that names a database
        # means that database, not every schema the account can see.
        named = getattr(connector, "default_dataset", None)
        if not config.datasets and named:
            config.datasets = [named]
        return run_scan(connector, config)
