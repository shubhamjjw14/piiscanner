"""Terminal report.

The audience is a compliance officer or a DBA, not a developer, so the summary
leads with what was found and where, and the confidence column is explained
rather than assumed.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from piiscanner.models import DetectionLayer, Finding, ScanResult

#: How many findings the summary table shows before truncating.
TOP_N = 10


def _confidence_style(confidence: float) -> str:
    if confidence >= 0.9:
        return "bold red"
    if confidence >= 0.7:
        return "yellow"
    return "cyan"


def _detected_by(finding: Finding) -> str:
    labels = {DetectionLayer.NAME: "name", DetectionLayer.VALUE: "value"}
    return "+".join(labels[d] for d in finding.detected_by)


def _row_count(finding: Finding) -> str:
    count = finding.approx_row_count
    if not count:
        return "-"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}k"
    return str(count)


def findings_table(findings: list[Finding], title: str) -> Table:
    """Build the findings table.

    Long field paths and masked evidence are truncated rather than wrapped:
    a table where every row folds across five lines is unreadable, and the
    full detail is in the JSON output anyway.
    """
    table = Table(title=title, title_justify="left", header_style="bold", expand=True)
    table.add_column("Conf", justify="right", width=4)
    table.add_column("Location", no_wrap=True, overflow="ellipsis", ratio=3)
    table.add_column("PII type", no_wrap=True)
    table.add_column("Match", justify="right", width=5)
    table.add_column("Rows", justify="right", width=5)
    table.add_column("Via", no_wrap=True, width=10)
    table.add_column("Evidence (masked)", no_wrap=True, overflow="ellipsis", ratio=4)

    for finding in findings:
        evidence = ", ".join(finding.evidence[:2]) if finding.evidence else "-"
        table.add_row(
            Text(f"{finding.confidence:.2f}", style=_confidence_style(finding.confidence)),
            f"{finding.collection}.{finding.field}",
            finding.pii_type.value,
            f"{finding.match_rate:.0%}",
            _row_count(finding),
            _detected_by(finding),
            evidence,
        )
    return table


def print_report(result: ScanResult, console: Console | None = None, top: int = TOP_N) -> None:
    """Print the full terminal report for a completed scan."""
    console = console or Console()
    findings = result.sorted_findings()

    console.print()
    console.rule("[bold]PII discovery scan[/bold]")
    for connector in result.connectors:
        console.print(
            f"  source     [bold]{connector.alias}[/bold] "
            f"({connector.kind}, server {connector.server_version or 'unknown'})"
        )
    console.print(f"  run id     {result.run_id}")
    console.print(f"  ruleset    {result.ruleset_version}")
    console.print(
        f"  scanned    {result.datasets_scanned} dataset(s), "
        f"{result.collections_scanned} collection(s), "
        f"{result.fields_examined} field(s) examined, "
        f"{result.fields_skipped} skipped"
    )
    console.print(f"  findings   [bold]{len(findings)}[/bold]")
    console.print()

    if not findings:
        console.print("[green]No personal data found above the reporting threshold.[/green]")
        return

    counts = Table(title="Findings by PII type", title_justify="left", header_style="bold")
    counts.add_column("PII type")
    counts.add_column("Fields", justify="right")
    for pii_type, count in result.counts_by_type().items():
        counts.add_row(pii_type, str(count))
    console.print(counts)
    console.print()

    console.print(findings_table(findings[:top], f"Top {min(top, len(findings))} highest-risk findings"))

    unverified = [f for f in findings if f.unverified]
    if unverified:
        console.print()
        console.print(
            f"[dim]{len(unverified)} finding(s) rest on the field name alone and "
            "were not confirmed by values. Review these before acting.[/dim]"
        )
    console.print(
        "\n[dim]Evidence is masked: first and last two characters only. "
        "No raw values are stored or transmitted.[/dim]"
    )


def print_full_table(result: ScanResult, console: Console | None = None) -> None:
    """Print every finding, not just the top slice."""
    console = console or Console()
    console.print(findings_table(result.sorted_findings(), "All findings"))
