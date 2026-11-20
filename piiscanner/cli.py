"""Command line interface."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from piiscanner.benchmark import format_score, score_scan
from piiscanner.classify.engine import run_scan
from piiscanner.config import ScanConfig, build_connector, configure_logging
from piiscanner.models import RULESET_VERSION
from piiscanner.report.json_out import read_json, write_json
from piiscanner.report.text_out import print_full_table, print_report

#: Where to ask about the full version (finding one person's rows and
#: writing their deletion plan).  Filled in before publishing; the public
#: build (make_public.sh) refuses to run while it is still a placeholder.
CONTACT = "shubhamjjw14@gmail.com"


@click.group()
@click.version_option(version=RULESET_VERSION, prog_name="piiscanner")
def cli() -> None:
    """Read-only PII discovery for DPDP Act compliance.

    piiscanner never writes to the databases it scans and never stores a raw
    sampled value. Evidence in every report is masked.
    """


@cli.command()
@click.option("--source", required=True, help="DSN, e.g. mysql://readonly@host/db")
@click.option("--alias", default=None, help="Name for this source in the report.")
@click.option("--sample-size", default=500, show_default=True, help="Values sampled per field.")
@click.option("--include", multiple=True, help="Glob of tables to scan. Repeatable.")
@click.option("--exclude", multiple=True, help="Glob of tables to skip. Repeatable.")
@click.option("--dataset", "datasets", multiple=True, help="Limit to these databases.")
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="Write findings JSON here.")
@click.option("--all", "show_all", is_flag=True, help="Print every finding, not the top 10.")
@click.option(
    "--inventory",
    is_flag=True,
    help="Record every field examined, not just findings. Required by `piiscanner audit`.",
)
@click.option(
    "--cursor-file",
    type=click.Path(dir_okay=False),
    default=None,
    help=(
        "MySQL only: persist keyless large-table sweep position here across "
        "runs, so a scheduled re-scan advances instead of re-reading the "
        "same first rows every time."
    ),
)
@click.option("--verbose", is_flag=True, help="Debug logging (never includes values).")
def scan(
    source: str,
    alias: str | None,
    sample_size: int,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    datasets: tuple[str, ...],
    out: str | None,
    show_all: bool,
    inventory: bool,
    cursor_file: str | None,
    verbose: bool,
) -> None:
    """Scan a database and report the personal data it holds."""
    configure_logging(verbose)
    console = Console()
    config = ScanConfig(
        source=source,
        alias=alias,
        sample_size=sample_size,
        cursor_path=cursor_file,
        include=list(include),
        exclude=list(exclude),
        datasets=list(datasets),
        record_inventory=inventory,
    )

    connector = build_connector(config)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed} fields"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("connecting", total=None)

        def on_field(dataset: str, collection: str, field_name: str) -> None:
            # Field paths and counts only -- never a sampled value.
            progress.update(
                task,
                advance=1,
                description=f"scanning {dataset}.{collection}",
            )

        try:
            with connector:
                # A connection string that names a database means that
                # database.  Left unset, the scan walked every schema the
                # account could see: pointed at `sakila`, it reported Aadhaar
                # numbers from a different database on the same server.
                # `--dataset` still chooses explicitly.
                named = getattr(connector, "default_dataset", None)
                if not config.datasets and named:
                    config.datasets = [named]
                progress.update(task, description="reading schema")
                result = run_scan(connector, config, on_field=on_field)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            progress.stop()
            console.print(f"[red]Scan failed:[/red] {exc}")
            raise SystemExit(2) from exc

    print_report(result, console)
    if show_all:
        console.print()
        print_full_table(result, console)

    # Loud, and after the report so it is the last thing on screen: a table the
    # scanner could not read is not a table without personal data in it.
    if connector.sampling_errors:
        console.print(
            f"\n[yellow]{len(connector.sampling_errors)} table(s) could not be "
            f"sampled and are NOT covered by this report:[/yellow]"
        )
        for dataset, collection, reason in connector.sampling_errors:
            console.print(f"  [yellow]·[/yellow] {dataset}.{collection}: {reason}")

    if out:
        path = write_json(result, out)
        console.print(f"\nFindings written to [bold]{path}[/bold]")

    tables = {(f.dataset, f.collection) for f in result.findings}
    if tables and "linkage" not in cli.commands:
        console.print(
            f"\n[dim]{len(tables)} table(s) hold personal data. Finding one "
            f"person's rows across them and writing a dry-run deletion plan is "
            f"in piiscanner's full version: {CONTACT}[/dim]")


@cli.command()
@click.argument("findings", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
@click.option("--all", "show_all", is_flag=True, help="Print every finding.")
def report(findings: str, output_format: str, show_all: bool) -> None:
    """Re-render a findings file."""
    result = read_json(findings)
    if output_format == "json":
        import json

        from piiscanner.report.json_out import to_dict

        click.echo(json.dumps(to_dict(result), indent=2))
        return
    console = Console()
    print_report(result, console)
    if show_all:
        console.print()
        print_full_table(result, console)


@cli.command()
@click.option(
    "--source",
    default="mysql://piiscanner_ro:piiscanner_ro_pw@127.0.0.1:3306/piiscanner_testdata",
    show_default=True,
    help="DSN of the loaded synthetic dataset.",
)
@click.option("--sample-size", default=500, show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), default=None)
def benchmark(source: str, sample_size: int, out: str | None) -> None:
    """Score the scanner against the synthetic dataset.

    Exits non-zero if anything was missed or any decoy was flagged, so this
    can be wired straight into CI.
    """
    configure_logging(False)
    from piiscanner.testdata.generate import DATABASE_NAME

    config = ScanConfig(
        source=source, alias="benchmark", sample_size=sample_size,
        datasets=[DATABASE_NAME],
    )
    connector = build_connector(config)
    with connector:
        result = run_scan(connector, config)

    from urllib.parse import urlparse

    from piiscanner.testdata.generate import GROUND_TRUTH, MONGO_GROUND_TRUTH

    # The two datasets have different answer keys: the Mongo one is written in
    # flattened path notation and includes the nested and array cases.
    is_mongo = urlparse(source).scheme.startswith("mongodb")
    score = score_scan(result, MONGO_GROUND_TRUTH if is_mongo else GROUND_TRUTH)
    click.echo(format_score(score))
    if out:
        write_json(result, out)
        click.echo(f"\nFindings written to {out}")

    if score.false_negatives or score.false_positives:
        raise SystemExit(1)


@cli.group()
def audit() -> None:
    """Measure precision and recall on a real schema, by hand-labelling.

    The synthetic benchmark scores 1.000 because its answer key was written
    alongside the detectors. These commands measure the classifier against
    something it was not fitted to.

        piiscanner scan --source ... --inventory --out findings.json
        piiscanner audit sample findings.json --name prod-mysql
        # ... fill in worksheet.csv by hand ...
        piiscanner audit score audit/prod-mysql-<date>
    """


@audit.command("sample")
@click.argument("findings", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", required=True, help="Short name for this audit, e.g. prod-mysql.")
@click.option("--root", default="audit", show_default=True, type=click.Path(file_okay=False))
@click.option("--seed", default=0, show_default=True, help="Recorded, so the draw is reproducible.")
@click.option("--unflagged", default=None, type=int, help="Unflagged fields to draw.")
@click.option("--skipped", default=None, type=int, help="Skipped fields to draw.")
def audit_sample(
    findings: str,
    name: str,
    root: str,
    seed: int,
    unflagged: int | None,
    skipped: int | None,
) -> None:
    """Draw a stratified sample from a scan and emit a blind worksheet."""
    from piiscanner.audit.sampling import STRATA, Stratum, draw_sample
    from piiscanner.audit.storage import WORKSHEET_FILE, audit_dir, save_audit
    from piiscanner.audit.worksheet import write_worksheet

    console = Console()
    result = read_json(findings)

    sizes = dict(STRATA)
    if unflagged is not None:
        sizes[Stratum.UNFLAGGED] = unflagged
    if skipped is not None:
        sizes[Stratum.SKIPPED] = skipped

    try:
        sample = draw_sample(result, name=name, seed=seed, sizes=sizes)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2) from exc

    directory = audit_dir(root, name)
    save_audit(sample, directory)
    worksheet = write_worksheet(sample, directory / WORKSHEET_FILE)

    console.print()
    console.print(f"[bold]Audit sample: {name}[/bold]")
    console.print(f"  source     {sample.source_alias}")
    console.print(f"  classifier {sample.classifier_version} · ruleset {sample.ruleset_version}")
    console.print(f"  seed       {sample.seed}")
    console.print()

    table = Table(header_style="bold")
    table.add_column("Stratum")
    table.add_column("Population", justify="right")
    table.add_column("Drawn", justify="right")
    table.add_column("Basis")
    for stratum in Stratum:
        population = sample.population.get(stratum, 0)
        drawn = sample.drawn.get(stratum, 0)
        table.add_row(
            stratum.value,
            str(population),
            str(drawn),
            "census (precision is exact)" if drawn >= population and population
            else "sample (recall is estimated)",
        )
    console.print(table)

    console.print()
    console.print(f"  worksheet  [bold]{worksheet}[/bold]   ({len(sample.fields)} rows to label)")
    console.print(
        "  [dim]The worksheet carries no verdict, no confidence and no stratum, "
        "and the rows are shuffled.[/dim]"
    )
    console.print(f"\n  Then run:  piiscanner audit score {directory}")


@audit.command("score")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--labels", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Filled worksheet. Defaults to worksheet.csv in DIRECTORY.")
def audit_score(directory: str, labels: str | None) -> None:
    """Score a labelled worksheet and write scores.json plus misses.md."""
    from pathlib import Path

    from piiscanner.audit.misses import write_misses
    from piiscanner.audit.report import format_score, write_scores
    from piiscanner.audit.scoring import score_audit
    from piiscanner.audit.storage import WORKSHEET_FILE, load_audit
    from piiscanner.audit.worksheet import read_labels

    console = Console()
    sample = load_audit(directory)
    worksheet = Path(labels) if labels else Path(directory) / WORKSHEET_FILE
    if not worksheet.exists():
        console.print(f"[red]no worksheet at {worksheet}[/red]")
        raise SystemExit(2)

    try:
        parsed, unlabelled = read_labels(worksheet)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2) from exc

    if not parsed:
        console.print(
            f"[red]{worksheet} has no labels yet. Fill in the is_pii column.[/red]"
        )
        raise SystemExit(2)

    score = score_audit(sample, parsed, unlabelled)
    click.echo(format_score(score))

    scores_path = write_scores(score, directory)
    misses_path = write_misses(score, directory)
    console.print(f"  scores     {scores_path}")
    console.print(f"  work queue {misses_path}")


def _add_plugins() -> None:
    """Command groups other packages register under ``piiscanner.commands``.

    This package names none of them.  One that fails to load is reported and
    skipped: the scanner works without it.
    """
    from importlib.metadata import entry_points

    for entry in entry_points(group="piiscanner.commands"):
        if entry.name in cli.commands:
            continue
        try:
            cli.add_command(entry.load(), entry.name)
        except Exception as exc:  # noqa: BLE001 - a broken add-on must not stop a scan
            click.echo(f"warning: could not load the '{entry.name}' commands: {exc}",
                       err=True)


_add_plugins()


def main() -> None:
    sys.exit(cli())


if __name__ == "__main__":
    main()
