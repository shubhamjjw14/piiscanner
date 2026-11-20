"""``misses.md`` -- the work queue an audit produces.

A scorecard tells you how good the classifier is.  This tells you what to do
next, which is the more useful artefact.  Disagreements are grouped by *apparent
cause* rather than listed flat, because the grouping is the finding: twelve
misses sharing one cause is one fix, while twelve misses with twelve causes is a
detector-coverage problem.

Skip misses are kept in their own section throughout.  A field that holds PII but
was excluded before classification is a bug in the skip rules, not a classifier
miss, and mixing the two would point the work at the wrong module.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from piiscanner.audit.scoring import AuditScore, Disagreement
from piiscanner.audit.storage import MISSES_FILE

KIND_TITLES = {
    "false_negative": "False negatives -- PII the scanner did not report",
    "false_positive": "False positives -- reported but not personal data",
    "wrong_type": "Wrong type -- field found, label incorrect",
    "skip_miss": "Skip-rule misses -- PII excluded before classification",
}

KIND_NOTES = {
    "false_negative": (
        "Each of these is a recall failure. Grouped by apparent cause: a cause "
        "mentioning a *name signal* means Layer 1 fired and something downstream "
        "discarded it, which is a merge-logic fix. A cause mentioning *no name "
        "signal* means nothing detected it at all, which needs a detector."
    ),
    "false_positive": (
        "Each of these costs credibility directly -- a reviewer who chases one "
        "down for nothing trusts the next finding less."
    ),
    "wrong_type": (
        "The column was correctly identified as personal data but mislabelled. "
        "Lower severity than a miss: the field still lands in the inventory."
    ),
    "skip_miss": (
        "These are bugs in the skip rules, not the classifier. The field was "
        "never sampled, so no detector had a chance."
    ),
}


def _row_block(item: Disagreement) -> list[str]:
    row = item.row
    lines = [f"#### `{row.path}`", ""]

    facts = [
        f"- type `{row.native_type}`",
        f"- sampled {row.sample_count}"
        + (f" of ~{row.approx_row_count:,} rows" if row.approx_row_count else ""),
        f"- blank {row.blank_rate:.0%}, distinct {row.distinct_ratio:.0%}",
    ]
    if row.evidence:
        facts.append("- examples (masked): " + ", ".join(f"`{e}`" for e in row.evidence))
    else:
        facts.append("- no examples: field was not sampled")

    if row.scanner_flagged:
        verdict = row.scanner_pii_type.value if row.scanner_pii_type else "?"
        facts.append(
            f"- scanner said **{verdict}** at {row.scanner_confidence:.2f}"
            + (" (unverified, name only)" if row.scanner_unverified else "")
        )
    elif row.skip_reason:
        facts.append(f"- skipped because: {row.skip_reason}")
    else:
        facts.append("- scanner reported nothing")

    if item.label.pii_type:
        facts.append(f"- labelled as **{item.label.pii_type}**")
    if item.label.notes:
        facts.append(f"- auditor note: {item.label.notes}")

    lines += facts + [""]
    return lines


def render_misses(score: AuditScore) -> str:
    """Build the markdown work queue."""
    sample = score.sample
    lines = [
        f"# Audit work queue -- {sample.name}",
        "",
        f"Source `{sample.source_alias}` · scan `{sample.scan_run_id}` · "
        f"classifier {sample.classifier_version} · ruleset {sample.ruleset_version}",
        f"Sample seed `{sample.seed}`.",
        "",
        "Precision is exact (the flagged stratum is a census). Recall is an "
        "extrapolation from a sample and carries the interval shown in the "
        "scorecard -- do not quote it without one.",
        "",
        "## Summary",
        "",
        "| Class | Count |",
        "|---|---|",
        f"| False negatives (observed) | {len(score.false_negatives)} |",
        f"| False positives | {len(score.false_positives)} |",
        f"| Wrong type | {len(score.type_wrong)} |",
        f"| Skip-rule misses | {len(score.skip_misses)} |",
        f"| Unsure (excluded from all scores) | {len(score.unsure)} |",
        f"| Unlabelled (excluded) | {len(score.unlabelled)} |",
        "",
    ]

    if not score.disagreements:
        lines += [
            "No disagreements. Every label matched the scanner's verdict.",
            "",
            "Read that carefully: with a small unflagged sample, agreement is "
            "consistent with good recall *and* with a sample that happened to "
            "miss the failures. The scorecard's interval is the honest summary.",
            "",
        ]
        return "\n".join(lines)

    by_kind: dict[str, list[Disagreement]] = defaultdict(list)
    for item in score.disagreements:
        by_kind[item.kind].append(item)

    for kind in ("false_negative", "false_positive", "wrong_type", "skip_miss"):
        items = by_kind.get(kind)
        if not items:
            continue
        lines += [f"## {KIND_TITLES[kind]}", "", KIND_NOTES[kind], ""]

        by_cause: dict[str, list[Disagreement]] = defaultdict(list)
        for item in items:
            by_cause[item.cause].append(item)

        # Biggest groups first: the largest shared cause is the best next fix.
        for cause, group in sorted(by_cause.items(), key=lambda kv: -len(kv[1])):
            lines += [f"### {cause}  ({len(group)} field(s))", ""]
            for item in sorted(group, key=lambda i: i.row.path):
                lines += _row_block(item)

    return "\n".join(lines)


def write_misses(score: AuditScore, directory: str | Path) -> Path:
    target = Path(directory) / MISSES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_misses(score), encoding="utf-8")
    return target
