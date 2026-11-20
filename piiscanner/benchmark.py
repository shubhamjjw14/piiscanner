"""Precision and recall against the synthetic dataset.

This is the product quality metric.  It is deliberately a single number that
is easy to re-run, because a detection tool without a measured false-positive
rate is a tool nobody can justify buying.

Scoring, and its honest limits:

*recall*    over the columns in ``EXPECTED`` -- every deliberately planted
            piece of PII.  A miss is a recall failure.
*precision* over the columns in ``EXPECTED`` plus ``FORBIDDEN``.  A finding on
            a decoy column is a false positive.
*unscored*  findings on columns in neither list.  These are neither rewarded
            nor punished, so they are reported separately and prominently --
            otherwise a scanner could inflate precision by flagging columns
            the answer key never considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from piiscanner.models import Finding, PIIType, ScanResult
from piiscanner.testdata.generate import GROUND_TRUTH, Expectation


@dataclass
class Score:
    """The outcome of scoring one scan against the answer key."""

    true_positives: list[tuple[str, str, PIIType]] = field(default_factory=list)
    false_negatives: list[Expectation] = field(default_factory=list)
    false_positives: list[Finding] = field(default_factory=list)
    #: Found the column, but called it the wrong type.
    type_mismatches: list[tuple[Expectation, PIIType]] = field(default_factory=list)
    #: Findings on columns the answer key says nothing about.
    unscored: list[Finding] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp, fp = len(self.true_positives), len(self.false_positives)
        return tp / (tp + fp) if (tp + fp) else 1.0

    @property
    def recall(self) -> float:
        tp, fn = len(self.true_positives), len(self.false_negatives)
        return tp / (tp + fn) if (tp + fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def type_accuracy(self) -> float:
        """Share of found columns that were also labelled with the right type."""
        found = len(self.true_positives) + len(self.type_mismatches)
        return len(self.true_positives) / found if found else 1.0


def score_scan(result: ScanResult, ground_truth=GROUND_TRUTH) -> Score:
    """Compare a scan result against the synthetic dataset's answer key."""
    score = Score()

    # Best finding per column, so a column with several candidate types is
    # judged on its strongest verdict.
    best: dict[tuple[str, str], Finding] = {}
    all_by_column: dict[tuple[str, str], list[Finding]] = {}
    for finding in result.findings:
        key = (finding.collection, finding.field)
        all_by_column.setdefault(key, []).append(finding)
        if key not in best or finding.confidence > best[key].confidence:
            best[key] = finding

    expected_pairs = ground_truth.expected_pairs()
    forbidden_pairs = ground_truth.forbidden_pairs()

    for expectation in ground_truth.expected:
        key = (expectation.collection, expectation.field)
        findings = all_by_column.get(key)
        if not findings:
            score.false_negatives.append(expectation)
            continue
        types = {f.pii_type for f in findings}
        if expectation.pii_type in types:
            score.true_positives.append((*key, expectation.pii_type))
        else:
            score.type_mismatches.append((expectation, best[key].pii_type))

    for key, findings in all_by_column.items():
        if key in forbidden_pairs:
            score.false_positives.extend(findings)
        elif key not in expected_pairs:
            score.unscored.extend(findings)

    return score


def format_score(score: Score) -> str:
    """A plain-text scorecard, suitable for CI output."""
    lines = [
        "Precision/recall against the synthetic dataset",
        "=" * 60,
        f"  precision      {score.precision:.3f}   "
        f"({len(score.true_positives)} TP / {len(score.false_positives)} FP)",
        f"  recall         {score.recall:.3f}   "
        f"({len(score.true_positives)} TP / {len(score.false_negatives)} FN)",
        f"  F1             {score.f1:.3f}",
        f"  type accuracy  {score.type_accuracy:.3f}",
        f"  unscored       {len(score.unscored)} finding(s) on columns "
        "outside the answer key",
    ]

    if score.false_negatives:
        lines.append("")
        lines.append("MISSED (recall failures):")
        for expectation in score.false_negatives:
            suffix = f"  [{expectation.note}]" if expectation.note else ""
            lines.append(
                f"  - {expectation.collection}.{expectation.field} "
                f"expected {expectation.pii_type.value}{suffix}"
            )

    if score.type_mismatches:
        lines.append("")
        lines.append("WRONG TYPE (column found, label wrong):")
        for expectation, actual in score.type_mismatches:
            lines.append(
                f"  - {expectation.collection}.{expectation.field} "
                f"expected {expectation.pii_type.value}, got {actual.value}"
            )

    if score.false_positives:
        lines.append("")
        lines.append("FALSE POSITIVES (decoy columns that were flagged):")
        for finding in score.false_positives:
            lines.append(
                f"  - {finding.collection}.{finding.field} "
                f"flagged as {finding.pii_type.value} at {finding.confidence:.2f}"
            )

    if score.unscored:
        lines.append("")
        lines.append("UNSCORED (not in the answer key; neither rewarded nor punished):")
        for finding in score.unscored:
            lines.append(
                f"  - {finding.collection}.{finding.field} -> "
                f"{finding.pii_type.value} at {finding.confidence:.2f}"
            )

    return "\n".join(lines)
