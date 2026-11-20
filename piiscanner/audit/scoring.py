"""Scoring an audit: exact precision, estimated recall, honest intervals.

The asymmetry between the two headline numbers is the whole point of this
module, and it is not a limitation to be smoothed over:

*precision is exact*
    Every flagged field is labelled, so the flagged stratum is a census.
    ``TP / (TP + FP)`` is the answer, not an estimate of it.

*recall is extrapolated*
    Only 60 of several hundred unflagged fields are labelled.  The observed
    false-negative rate is scaled up to the whole unflagged population, which
    means recall carries sampling error and must always be reported with an
    interval.

Two failure modes are guarded explicitly.

Reporting a bare recall point estimate invites false confidence, so
:class:`RecallEstimate` has no formatter that omits the interval.

Observing zero or one false negative in the sample makes the normal
approximation invalid -- at ``FN == 0`` the standard error collapses to zero and
the interval degenerates to the single flattering point ``recall == 1.000``.
When that happens the estimate is marked weak and an *exact* one-sided binomial
bound is computed instead, which is wide and says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from piiscanner.audit.sampling import AuditSample, SampledField, Stratum
from piiscanner.audit.worksheet import Label
from piiscanner.models import PIIType

#: z for a two-sided 95% interval.
Z_95 = 1.959963985

#: Below this many observed false negatives the normal approximation to the
#: binomial does not hold.  The usual rule of thumb wants n*p >= 5; at 0 or 1
#: the interval is either degenerate or badly skewed.
WEAK_FN_THRESHOLD = 2

#: Free-text labels a human plausibly writes, mapped to the type the scanner
#: would use.  Matched after upper-casing and collapsing separators.
TYPE_ALIASES: dict[str, PIIType] = {
    "PHONE": PIIType.INDIAN_MOBILE,
    "MOBILE": PIIType.INDIAN_MOBILE,
    "PHONE_NUMBER": PIIType.INDIAN_MOBILE,
    "MOBILE_NUMBER": PIIType.INDIAN_MOBILE,
    "NAME": PIIType.PERSON_NAME,
    "PERSON": PIIType.PERSON_NAME,
    "FULL_NAME": PIIType.PERSON_NAME,
    "AADHAR": PIIType.AADHAAR,
    "ADHAAR": PIIType.AADHAAR,
    "UID": PIIType.AADHAAR,
    "PAN_CARD": PIIType.PAN,
    "DOB": PIIType.DATE_OF_BIRTH,
    "DATE_OF_BIRTH": PIIType.DATE_OF_BIRTH,
    "BIRTH_DATE": PIIType.DATE_OF_BIRTH,
    "IP": PIIType.IP_ADDRESS,
    "IP_ADDR": PIIType.IP_ADDRESS,
    "EMAIL_ADDRESS": PIIType.EMAIL,
    "ACCOUNT": PIIType.BANK_ACCOUNT,
    "ACCOUNT_NUMBER": PIIType.BANK_ACCOUNT,
    "BANK": PIIType.BANK_ACCOUNT,
    "CARD": PIIType.CARD_NUMBER,
    "PINCODE": PIIType.PINCODE,
    "PIN": PIIType.PINCODE,
    "POSTCODE": PIIType.PINCODE,
    "UPI": PIIType.UPI_VPA,
    "VPA": PIIType.UPI_VPA,
    "DEVICE": PIIType.DEVICE_ID,
    "GEO": PIIType.GEO_LOCATION,
    "LOCATION": PIIType.GEO_LOCATION,
    "HEALTH": PIIType.HEALTH_DATA,
    "ADDR": PIIType.ADDRESS,
}


def normalise_type_label(raw: str) -> PIIType | None:
    """Best-effort read of a free-text type label.  ``None`` if unrecognised."""
    token = (raw or "").strip().upper().replace(" ", "_").replace("-", "_")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        return None
    try:
        return PIIType(token)
    except ValueError:
        pass
    return TYPE_ALIASES.get(token)


def binomial_upper_bound(hits: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper confidence bound on a binomial proportion.

    The Clopper-Pearson bound: the largest ``p`` for which observing ``hits`` or
    fewer successes in ``n`` trials still has probability ``alpha``.  Found by
    bisection on the binomial CDF, which needs no third-party maths library and
    is exact enough at these sample sizes.

    Used instead of the normal approximation when the observed count is 0 or 1.
    At ``hits == 0`` this reduces to the familiar ``1 - alpha**(1/n)``.
    """
    if n <= 0:
        return 1.0
    if hits >= n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(
            math.comb(n, k) * p**k * (1.0 - p) ** (n - k) for k in range(hits + 1)
        )

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class RecallEstimate:
    """An extrapolated recall figure that refuses to be quoted alone."""

    true_positives: int
    #: False negatives actually observed in the labelled unflagged sample.
    observed_fn: int
    #: Unflagged fields labelled (unsure excluded).
    sample_size: int
    #: Unflagged fields in the whole population.
    population: int

    @property
    def fn_rate(self) -> float:
        """Observed false-negative rate among unflagged fields."""
        return self.observed_fn / self.sample_size if self.sample_size else 0.0

    @property
    def estimated_fn(self) -> float:
        """Extrapolated count of missed PII across the unflagged population."""
        return self.fn_rate * self.population

    @property
    def weak(self) -> bool:
        """True when the normal approximation cannot be trusted."""
        return self.observed_fn < WEAK_FN_THRESHOLD or self.sample_size == 0

    @property
    def fn_rate_interval(self) -> tuple[float, float]:
        """95% interval on the false-negative rate.

        Normal approximation with a finite-population correction: the sample is
        a tenth of the population here, well past the 5% where ignoring the
        correction overstates the error.

        When the observed count is 0 or 1 the normal interval is degenerate or
        badly skewed, so an exact one-sided Clopper-Pearson upper bound is used
        and the lower bound is pinned at 0.
        """
        n, N = self.sample_size, self.population
        if n == 0:
            return (0.0, 1.0)
        if self.weak:
            return (0.0, binomial_upper_bound(self.observed_fn, n))

        p = self.fn_rate
        se = math.sqrt(p * (1.0 - p) / n)
        if N > n:
            se *= math.sqrt((N - n) / (N - 1))
        return (max(0.0, p - Z_95 * se), min(1.0, p + Z_95 * se))

    def _recall(self, fn_rate: float) -> float:
        estimated_fn = fn_rate * self.population
        denominator = self.true_positives + estimated_fn
        return self.true_positives / denominator if denominator else 1.0

    @property
    def point(self) -> float:
        return self._recall(self.fn_rate)

    @property
    def interval(self) -> tuple[float, float]:
        """95% interval on recall.

        Note the inversion: a *higher* false-negative rate means *lower* recall,
        so the upper bound on the rate produces the lower bound on recall.
        """
        lo_rate, hi_rate = self.fn_rate_interval
        return (self._recall(hi_rate), self._recall(lo_rate))

    def describe(self) -> str:
        """The only rendering.  There is no point-estimate-only formatter."""
        lo, hi = self.interval
        head = f"{self.point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
        if self.sample_size == 0:
            return "unmeasurable -- no unflagged fields were labelled"
        if self.weak:
            _, rate_hi = self.fn_rate_interval
            return (
                f"{head}\n"
                f"      WEAK: only {self.observed_fn} false negative(s) observed in "
                f"{self.sample_size} labelled fields.\n"
                f"      The normal approximation does not hold at this count, so the "
                f"bound above is an\n"
                f"      exact one-sided binomial one: the miss rate could be as high "
                f"as {rate_hi:.1%},\n"
                f"      which is ~{rate_hi * self.population:.0f} missed field(s) across "
                f"{self.population} unflagged.\n"
                f"      Treat the point estimate as unproven, not as evidence of good "
                f"recall."
            )
        return (
            f"{head}\n"
            f"      from {self.observed_fn}/{self.sample_size} observed misses, "
            f"extrapolated to ~{self.estimated_fn:.0f} across "
            f"{self.population} unflagged fields"
        )


@dataclass
class Disagreement:
    """One field where the scanner and the label differ."""

    row: SampledField
    label: Label
    kind: str          # "false_negative" | "false_positive" | "wrong_type" | "skip_miss"
    cause: str         # grouping key for misses.md


@dataclass
class AuditScore:
    """Everything one scored audit produced."""

    sample: AuditSample

    # -- precision: a census, therefore exact --------------------------------
    true_positives: list[SampledField] = field(default_factory=list)
    false_positives: list[SampledField] = field(default_factory=list)

    # -- recall: estimated ---------------------------------------------------
    false_negatives: list[SampledField] = field(default_factory=list)
    true_negatives: list[SampledField] = field(default_factory=list)

    # -- type accuracy -------------------------------------------------------
    type_correct: int = 0
    type_wrong: list[Disagreement] = field(default_factory=list)
    #: Labelled PII but no type given, or a type the harness could not read.
    type_unspecified: int = 0

    # -- skip audit: a separate question -------------------------------------
    skip_misses: list[SampledField] = field(default_factory=list)
    skipped_labelled: int = 0

    # -- data quality --------------------------------------------------------
    unsure: list[SampledField] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)
    unmatched_row_ids: list[str] = field(default_factory=list)

    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp, fp = len(self.true_positives), len(self.false_positives)
        return tp / (tp + fp) if (tp + fp) else 1.0

    @property
    def recall(self) -> RecallEstimate:
        unflagged_labelled = len(self.false_negatives) + len(self.true_negatives)
        return RecallEstimate(
            true_positives=len(self.true_positives),
            observed_fn=len(self.false_negatives),
            sample_size=unflagged_labelled,
            population=self.sample.population.get(Stratum.UNFLAGGED, 0),
        )

    @property
    def type_accuracy(self) -> float:
        total = self.type_correct + len(self.type_wrong)
        return self.type_correct / total if total else 1.0

    @property
    def estimated_skip_misses(self) -> float:
        """Extrapolated count of PII hidden behind the skip rules."""
        if not self.skipped_labelled:
            return 0.0
        rate = len(self.skip_misses) / self.skipped_labelled
        return rate * self.sample.population.get(Stratum.SKIPPED, 0)

    @property
    def f1(self) -> float:
        """F1 on the point estimates.  Inherits recall's uncertainty."""
        p, r = self.precision, self.recall.point
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _fn_cause(row: SampledField) -> str:
    """Why, apparently, did the scanner stay quiet on a field that holds PII?

    Layer 1 is a pure function of the field name, so it can be replayed here
    without the values.  That splits misses into the two classes that need
    different fixes: the name gave a signal and something downstream discarded
    it (merge-logic problem), or there was no signal at all (detector-coverage
    problem).
    """
    from piiscanner.classify.layer1_names import classify_name

    hints = classify_name(row.field)
    if row.sample_count == 0:
        return "no values sampled - nothing to classify"
    if row.blank_rate >= 0.95:
        return "column almost entirely blank"
    if hints:
        names = ", ".join(sorted({h.pii_type.value for h in hints}))
        return f"name suggested {names} but the verdict was dropped (refuted or gated)"
    if row.distinct_ratio >= 0.99 and row.sample_count >= 20:
        return "no name signal; values are high-cardinality opaque strings"
    if row.sample_count < 20:
        return "no name signal; sample too thin to verify by value"
    return "no name signal; no value pattern matched"


def _fp_cause(row: SampledField) -> str:
    """Why, apparently, did the scanner flag a field that holds no PII?"""
    if row.scanner_unverified:
        return "name-only verdict; values never corroborated it"
    if row.scanner_match_rate is not None and row.scanner_match_rate < 0.5:
        return (
            f"value pattern matched only "
            f"{row.scanner_match_rate:.0%} of values yet still reported"
        )
    if row.blank_rate >= 0.5:
        return "verdict rests on a mostly-blank column"
    if row.sample_count < 20:
        return "verdict rests on a thin sample"
    return "value pattern matched but the field is not personal data"


def score_audit(
    sample: AuditSample, labels: list[Label], unlabelled: list[str] | None = None
) -> AuditScore:
    """Join labels back to verdicts and compute the scores.

    ``unsure`` labels are excluded from every denominator, and their count is
    carried on the result so the exclusion is visible rather than silent.
    """
    score = AuditScore(sample=sample, unlabelled=list(unlabelled or []))
    rows = sample.by_row_id()

    for label in labels:
        row = rows.get(label.row_id)
        if row is None:
            score.unmatched_row_ids.append(label.row_id)
            continue

        if label.unsure:
            score.unsure.append(row)
            continue

        # -- skipped fields answer a different question ---------------------
        if row.stratum is Stratum.SKIPPED:
            score.skipped_labelled += 1
            if label.is_pii:
                score.skip_misses.append(row)
                score.disagreements.append(
                    Disagreement(
                        row=row,
                        label=label,
                        kind="skip_miss",
                        cause=f"skipped as: {row.skip_reason or 'unknown reason'}",
                    )
                )
            continue

        if row.stratum is Stratum.FLAGGED:
            if label.is_pii:
                score.true_positives.append(row)
                expected = normalise_type_label(label.pii_type)
                if expected is None:
                    score.type_unspecified += 1
                elif expected is row.scanner_pii_type:
                    score.type_correct += 1
                else:
                    disagreement = Disagreement(
                        row=row,
                        label=label,
                        kind="wrong_type",
                        cause=(
                            f"labelled {expected.value}, reported "
                            f"{row.scanner_pii_type.value if row.scanner_pii_type else '?'}"
                        ),
                    )
                    score.type_wrong.append(disagreement)
                    score.disagreements.append(disagreement)
            else:
                score.false_positives.append(row)
                score.disagreements.append(
                    Disagreement(
                        row=row, label=label, kind="false_positive",
                        cause=_fp_cause(row),
                    )
                )
            continue

        # -- unflagged ------------------------------------------------------
        if label.is_pii:
            score.false_negatives.append(row)
            score.disagreements.append(
                Disagreement(
                    row=row, label=label, kind="false_negative", cause=_fn_cause(row),
                )
            )
        else:
            score.true_negatives.append(row)

    return score
