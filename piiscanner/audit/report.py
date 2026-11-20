"""Terminal scorecard for a scored audit.

Written as plain text rather than a rich table so it can be pasted into a ticket
or a CI log unchanged.  The layout deliberately puts precision and recall in
different blocks with different wording: one is measured, the other is
estimated, and a two-column table implies they are the same kind of number.
"""

from __future__ import annotations

import json
from pathlib import Path

from piiscanner.audit.scoring import AuditScore
from piiscanner.audit.sampling import Stratum
from piiscanner.audit.storage import SCORES_FILE

BAR = "=" * 72


def format_score(score: AuditScore) -> str:
    sample = score.sample
    tp, fp = len(score.true_positives), len(score.false_positives)
    fn, tn = len(score.false_negatives), len(score.true_negatives)

    lines = [
        BAR,
        f"AUDIT  {sample.name}",
        f"       source {sample.source_alias} · classifier "
        f"{sample.classifier_version} · ruleset {sample.ruleset_version}",
        f"       seed {sample.seed} · scan {sample.scan_run_id}",
        BAR,
        "",
        "STRATA",
        "-" * 72,
    ]
    for stratum in Stratum:
        population = sample.population.get(stratum, 0)
        drawn = sample.drawn.get(stratum, 0)
        kind = "census" if drawn >= population and population else "sample"
        lines.append(
            f"  {stratum.value:<10} {drawn:>4} labelled of {population:>4} "
            f"in population   ({kind})"
        )

    lines += [
        "",
        "PRECISION  (exact -- the flagged stratum is a census, not a sample)",
        "-" * 72,
        f"  precision      {score.precision:.3f}   ({tp} TP / {fp} FP)",
        "",
        "RECALL  (estimated -- extrapolated from the unflagged sample)",
        "-" * 72,
        f"  recall       ~ {score.recall.describe()}",
        "",
        f"  observed:      {fn} miss(es) in {fn + tn} labelled unflagged field(s)",
        "",
        "TYPE ACCURACY  (among true positives)",
        "-" * 72,
        f"  accuracy       {score.type_accuracy:.3f}   "
        f"({score.type_correct} correct / {len(score.type_wrong)} wrong)",
    ]
    if score.type_unspecified:
        lines.append(
            f"  {score.type_unspecified} true positive(s) had no readable type "
            "label and are excluded"
        )

    lines += [
        "",
        "SKIP AUDIT  (a bug in the skip rules, not a classifier miss)",
        "-" * 72,
    ]
    if not score.skipped_labelled:
        lines.append("  no skipped fields were labelled")
    else:
        lines.append(
            f"  {len(score.skip_misses)} of {score.skipped_labelled} labelled "
            f"skipped field(s) actually hold PII"
        )
        if score.skip_misses:
            lines.append(
                f"  extrapolates to ~{score.estimated_skip_misses:.0f} across "
                f"{sample.population.get(Stratum.SKIPPED, 0)} skipped field(s)"
            )
            lines.append(
                "  these are NOT in the recall figure above -- they never "
                "reached the classifier"
            )

    lines += [
        "",
        "EXCLUSIONS",
        "-" * 72,
        f"  unsure         {len(score.unsure)} row(s) -- excluded from every "
        "denominator",
        f"  unlabelled     {len(score.unlabelled)} row(s) -- left blank, also "
        "excluded",
    ]
    if score.unmatched_row_ids:
        lines.append(
            f"  unmatched      {len(score.unmatched_row_ids)} label(s) had a "
            "row_id not in the sample"
        )

    lines += [
        "",
        f"F1 (on point estimates, inherits recall's uncertainty)   {score.f1:.3f}",
        "",
    ]
    return "\n".join(lines)


def _serialisable(score: AuditScore) -> dict:
    recall = score.recall
    lo, hi = recall.interval
    rate_lo, rate_hi = recall.fn_rate_interval
    return {
        "audit": score.sample.name,
        "source_alias": score.sample.source_alias,
        "scan_run_id": score.sample.scan_run_id,
        "classifier_version": score.sample.classifier_version,
        "ruleset_version": score.sample.ruleset_version,
        "seed": score.sample.seed,
        "population": {s.value: n for s, n in score.sample.population.items()},
        "drawn": {s.value: n for s, n in score.sample.drawn.items()},
        "precision": {
            "value": round(score.precision, 4),
            "exact": True,
            "true_positives": len(score.true_positives),
            "false_positives": len(score.false_positives),
        },
        "recall": {
            "estimate": round(recall.point, 4),
            "exact": False,
            "ci95": [round(lo, 4), round(hi, 4)],
            "weak": recall.weak,
            "observed_false_negatives": recall.observed_fn,
            "labelled_unflagged": recall.sample_size,
            "unflagged_population": recall.population,
            "fn_rate": round(recall.fn_rate, 4),
            "fn_rate_ci95": [round(rate_lo, 4), round(rate_hi, 4)],
            "estimated_missed_fields": round(recall.estimated_fn, 1),
            "method": (
                "exact one-sided Clopper-Pearson bound (observed count too low "
                "for the normal approximation)"
                if recall.weak
                else "normal approximation with finite-population correction"
            ),
        },
        "type_accuracy": {
            "value": round(score.type_accuracy, 4),
            "correct": score.type_correct,
            "wrong": len(score.type_wrong),
            "unspecified": score.type_unspecified,
        },
        "skip_audit": {
            "labelled": score.skipped_labelled,
            "misses": len(score.skip_misses),
            "estimated_misses": round(score.estimated_skip_misses, 1),
            "fields": [r.path for r in score.skip_misses],
        },
        "exclusions": {
            "unsure": len(score.unsure),
            "unlabelled": len(score.unlabelled),
            "unmatched_row_ids": score.unmatched_row_ids,
        },
        "f1_point": round(score.f1, 4),
        "false_negatives": [
            {"field": d.row.path, "cause": d.cause}
            for d in score.disagreements
            if d.kind == "false_negative"
        ],
        "false_positives": [
            {
                "field": d.row.path,
                "reported": d.row.scanner_pii_type.value if d.row.scanner_pii_type else None,
                "confidence": d.row.scanner_confidence,
                "cause": d.cause,
            }
            for d in score.disagreements
            if d.kind == "false_positive"
        ],
    }


def write_scores(score: AuditScore, directory: str | Path) -> Path:
    target = Path(directory) / SCORES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_serialisable(score), indent=2) + "\n", encoding="utf-8"
    )
    return target
