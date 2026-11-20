"""Tests for the real-schema audit harness.

Two properties carry most of the weight:

*the worksheet is blind*
    If a labeller can infer the scanner's verdict, they anchor on it and the
    audit measures their trust in the tool instead of the tool. Several tests
    here assert the absence of things, which is unusual but correct: the risk is
    a well-meaning future change that adds a "confidence" column for
    convenience.

*recall never gets quoted as a bare number*
    At zero observed false negatives the normal approximation degenerates to
    recall == 1.000 with a zero-width interval, which is the most flattering
    and least honest output the harness could produce. That path is tested
    explicitly.
"""

from __future__ import annotations

import json
import math

import pytest

from piiscanner.audit.misses import render_misses
from piiscanner.audit.report import _serialisable, format_score
from piiscanner.audit.sampling import STRATA, Stratum, draw_sample
from piiscanner.audit.scoring import (
    RecallEstimate,
    binomial_upper_bound,
    normalise_type_label,
    score_audit,
)
from piiscanner.audit.storage import audit_dir, load_audit, save_audit
from piiscanner.audit.worksheet import (
    GIVEN_COLUMNS,
    Label,
    read_labels,
    write_worksheet,
)
from piiscanner.models import (
    ConnectorInfo,
    DetectionLayer,
    FieldRecord,
    Finding,
    PIIType,
    ScanResult,
)

SEED = 777


# --------------------------------------------------------------------------
# Fixtures: a synthetic scan with a known inventory
# --------------------------------------------------------------------------


def _record(field: str, *, flagged: bool = False, examined: bool = True,
            evidence: list[str] | None = None, **kwargs) -> FieldRecord:
    return FieldRecord(
        dataset="db",
        collection="t",
        field=field,
        native_type=kwargs.pop("native_type", "varchar(64)"),
        examined=examined,
        flagged=flagged,
        sample_count=kwargs.pop("sample_count", 500),
        blank_rate=kwargs.pop("blank_rate", 0.0),
        distinct_ratio=kwargs.pop("distinct_ratio", 0.9),
        approx_row_count=kwargs.pop("approx_row_count", 10_000),
        evidence=evidence or [],
        **kwargs,
    )


def _finding(field: str, pii_type: PIIType, confidence: float,
             unverified: bool = False) -> Finding:
    return Finding(
        source="src", dataset="db", collection="t", field=field,
        pii_type=pii_type, confidence=confidence, match_rate=0.9,
        sample_count=500, null_rate=0.0, blank_rate=0.0, distinct_ratio=0.9,
        evidence=["98XXXXXX21"],
        detected_by=[DetectionLayer.NAME] if unverified else [DetectionLayer.VALUE],
        unverified=unverified,
    )


@pytest.fixture
def scan() -> ScanResult:
    """3 flagged, 40 unflagged, 10 skipped."""
    result = ScanResult(connectors=[
        ConnectorInfo(alias="prod", kind="mysql", driver_version="x")
    ])
    for name, pii_type, confidence in [
        ("mobile", PIIType.INDIAN_MOBILE, 0.85),
        ("aadhaar_no", PIIType.AADHAAR, 0.99),
        ("customer_name", PIIType.PERSON_NAME, 0.50),
    ]:
        result.fields.append(_record(name, flagged=True, evidence=["98XXXXXX21"]))
        result.findings.append(
            _finding(name, pii_type, confidence, unverified=confidence == 0.50)
        )
    for i in range(40):
        result.fields.append(_record(f"plain_col_{i:02d}", evidence=["abXXef"]))
    for i in range(10):
        result.fields.append(
            _record(f"blob_col_{i:02d}", examined=False, native_type="blob",
                    skip_reason="type blob cannot hold PII", sample_count=0)
        )
    result.fields_examined = 43
    result.fields_skipped = 10
    return result


@pytest.fixture
def sample(scan: ScanResult):
    return draw_sample(scan, name="unit", seed=SEED,
                       sizes={**STRATA, Stratum.UNFLAGGED: 20, Stratum.SKIPPED: 5})


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_flagged_stratum_is_a_census(sample) -> None:
    """Precision is only exact if every flagged field is labelled."""
    assert sample.population[Stratum.FLAGGED] == 3
    assert sample.drawn[Stratum.FLAGGED] == 3
    assert len(sample.by_stratum(Stratum.FLAGGED)) == 3


def test_other_strata_are_sampled_and_populations_recorded(sample) -> None:
    assert sample.population[Stratum.UNFLAGGED] == 40
    assert sample.drawn[Stratum.UNFLAGGED] == 20
    assert sample.population[Stratum.SKIPPED] == 10
    assert sample.drawn[Stratum.SKIPPED] == 5


def test_population_is_the_whole_population_not_the_draw(sample) -> None:
    """The recall extrapolation multiplies by this; confusing the two would
    understate missed PII by the sampling ratio."""
    assert sample.population[Stratum.UNFLAGGED] > sample.drawn[Stratum.UNFLAGGED]


def test_requesting_more_than_exists_is_a_census_not_an_error(scan) -> None:
    drawn = draw_sample(scan, name="u", seed=SEED,
                        sizes={**STRATA, Stratum.UNFLAGGED: 10_000})
    assert drawn.drawn[Stratum.UNFLAGGED] == 40


def test_sample_is_reproducible_from_the_seed(scan) -> None:
    a = draw_sample(scan, name="u", seed=42)
    b = draw_sample(scan, name="u", seed=42)
    assert [f.path for f in a.fields] == [f.path for f in b.fields]
    assert [f.row_id for f in a.fields] == [f.row_id for f in b.fields]


def test_different_seeds_draw_differently(scan) -> None:
    a = draw_sample(scan, name="u", seed=1, sizes={**STRATA, Stratum.UNFLAGGED: 20})
    b = draw_sample(scan, name="u", seed=2, sizes={**STRATA, Stratum.UNFLAGGED: 20})
    assert {f.path for f in a.fields} != {f.path for f in b.fields}


def test_scan_without_inventory_is_refused(scan) -> None:
    """A findings-only scan would yield a flagged-census-shaped result that
    looks like an audit but cannot measure recall at all."""
    scan.fields = []
    with pytest.raises(ValueError, match="no field inventory"):
        draw_sample(scan, name="u", seed=SEED)


def test_row_ids_are_unique_and_not_derived_from_the_path(sample) -> None:
    ids = [f.row_id for f in sample.fields]
    assert len(set(ids)) == len(ids)
    for row in sample.fields:
        assert row.field not in row.row_id
        assert row.row_id not in row.path


def test_rows_are_shuffled_across_strata(sample) -> None:
    """Consecutive runs of one stratum would let a labeller infer the grouping."""
    strata = [f.stratum for f in sample.fields]
    assert len(set(strata)) == 3
    # Not simply flagged-then-unflagged-then-skipped.
    assert strata != sorted(strata, key=lambda s: list(Stratum).index(s))


# --------------------------------------------------------------------------
# Blindness of the worksheet
# --------------------------------------------------------------------------


def test_worksheet_reveals_no_verdict(sample, tmp_path) -> None:
    path = write_worksheet(sample, tmp_path / "worksheet.csv")
    text = path.read_text()

    header = next(
        line for line in text.splitlines() if line.startswith("row_id")
    ).split(",")
    for forbidden in (
        "stratum", "flagged", "confidence", "pii_type_detected", "verdict",
        "scanner_pii_type", "scanner_confidence", "detected_by", "match_rate",
        "unverified",
    ):
        assert forbidden not in header, forbidden
    # The only pii_type column is the empty one the human fills in.
    assert header.count("pii_type") == 1


def test_worksheet_body_contains_no_pii_type_values(sample, tmp_path) -> None:
    """Even in the notes: a leaked type name would give the verdict away."""
    path = write_worksheet(sample, tmp_path / "w.csv")
    body = "\n".join(
        line for line in path.read_text().splitlines() if not line.startswith("#")
    )
    for pii_type in (PIIType.AADHAAR, PIIType.INDIAN_MOBILE, PIIType.PERSON_NAME):
        assert pii_type.value not in body


def test_worksheet_has_a_row_per_sampled_field(sample, tmp_path) -> None:
    path = write_worksheet(sample, tmp_path / "w.csv")
    rows = [
        line for line in path.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("row_id")
    ]
    assert len(rows) == len(sample.fields)


def test_worksheet_shows_the_stats_a_labeller_needs(sample, tmp_path) -> None:
    path = write_worksheet(sample, tmp_path / "w.csv")
    header = next(l for l in path.read_text().splitlines() if l.startswith("row_id"))
    for needed in ("field_path", "data_type", "sampled", "blank_rate",
                   "distinct_ratio", "example_1", "example_2", "example_3"):
        assert needed in header
    assert set(GIVEN_COLUMNS) <= set(header.split(","))


def test_stratum_mapping_lives_outside_the_worksheet(sample, tmp_path) -> None:
    save_audit(sample, tmp_path)
    stored = json.loads((tmp_path / "sample.json").read_text())
    by_id = {f["row_id"]: f for f in stored["fields"]}
    assert all("stratum" in f for f in by_id.values())
    assert "stratum" not in write_worksheet(sample, tmp_path / "w.csv").read_text().splitlines()[-1]


# --------------------------------------------------------------------------
# Label parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("token,expected", [
    ("y", True), ("Y", True), ("yes", True), ("1", True), ("true", True),
    ("n", False), ("no", False), ("0", False),
    ("u", None), ("unsure", None), ("?", None),
])
def test_is_pii_tokens(sample, tmp_path, token, expected) -> None:
    path = write_worksheet(sample, tmp_path / "w.csv")
    _fill(path, {sample.fields[0].row_id: token})
    labels, _ = read_labels(path)
    assert labels[0].is_pii is expected


def test_unreadable_label_is_an_error_not_a_default(sample, tmp_path) -> None:
    path = write_worksheet(sample, tmp_path / "w.csv")
    _fill(path, {sample.fields[0].row_id: "probably"})
    with pytest.raises(ValueError, match="cannot read is_pii"):
        read_labels(path)


def test_blank_rows_are_reported_not_treated_as_no(sample, tmp_path) -> None:
    """Counting an unlabelled row as `not PII` would inflate recall."""
    path = write_worksheet(sample, tmp_path / "w.csv")
    _fill(path, {sample.fields[0].row_id: "y"})
    labels, unlabelled = read_labels(path)
    assert len(labels) == 1
    assert len(unlabelled) == len(sample.fields) - 1


def _fill(path, answers: dict[str, str], types: dict[str, str] | None = None) -> None:
    """Rewrite a worksheet with the given answers, preserving everything else."""
    import csv

    lines = path.read_text().splitlines(keepends=True)
    comments = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    reader = list(csv.DictReader(body))
    for record in reader:
        rid = record["row_id"]
        if rid in answers:
            record["is_pii"] = answers[rid]
            record["pii_type"] = (types or {}).get(rid, "")
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.writelines(comments)
        writer = csv.DictWriter(handle, fieldnames=reader[0].keys())
        writer.writeheader()
        writer.writerows(reader)


# --------------------------------------------------------------------------
# The binomial bound
# --------------------------------------------------------------------------


def test_zero_hits_bound_matches_the_closed_form() -> None:
    """At 0 observed hits the Clopper-Pearson bound is 1 - alpha**(1/n)."""
    for n in (10, 60, 200):
        assert binomial_upper_bound(0, n) == pytest.approx(
            1 - 0.05 ** (1 / n), abs=1e-6
        )


def test_bound_is_monotone_in_hits_and_in_n() -> None:
    assert binomial_upper_bound(0, 60) < binomial_upper_bound(1, 60)
    assert binomial_upper_bound(1, 200) < binomial_upper_bound(1, 60)


def test_bound_covers_the_true_rate() -> None:
    """Sanity: the bound should be above the point estimate."""
    assert binomial_upper_bound(3, 60) > 3 / 60


# --------------------------------------------------------------------------
# Recall estimation
# --------------------------------------------------------------------------


def test_recall_extrapolates_to_the_population() -> None:
    est = RecallEstimate(true_positives=6, observed_fn=3, sample_size=60,
                         population=585)
    assert est.fn_rate == pytest.approx(0.05)
    assert est.estimated_fn == pytest.approx(29.25)
    assert est.point == pytest.approx(6 / (6 + 29.25))
    assert not est.weak


def test_recall_interval_is_inverted_relative_to_the_fn_rate() -> None:
    """A higher miss rate means lower recall, so the bounds swap."""
    est = RecallEstimate(true_positives=6, observed_fn=6, sample_size=60,
                         population=585)
    lo, hi = est.interval
    assert lo < est.point < hi
    rate_lo, rate_hi = est.fn_rate_interval
    assert est._recall(rate_hi) == pytest.approx(lo)
    assert est._recall(rate_lo) == pytest.approx(hi)


def test_finite_population_correction_narrows_the_interval() -> None:
    """The sample is a tenth of the population; ignoring that overstates error."""
    est = RecallEstimate(true_positives=6, observed_fn=6, sample_size=60,
                         population=585)
    lo, hi = est.fn_rate_interval
    p = est.fn_rate
    uncorrected = 1.959963985 * math.sqrt(p * (1 - p) / 60)
    assert (hi - p) < uncorrected


@pytest.mark.parametrize("observed", [0, 1])
def test_zero_or_one_miss_is_reported_as_weak(observed: int) -> None:
    """The flattering-number guard. At FN=0 the normal interval collapses to
    [1.000, 1.000], which would read as proven perfect recall."""
    est = RecallEstimate(true_positives=6, observed_fn=observed, sample_size=60,
                         population=585)
    assert est.weak
    text = est.describe()
    assert "WEAK" in text
    assert "unproven" in text
    # The interval must be genuinely wide, not degenerate.
    lo, hi = est.interval
    assert hi - lo > 0.3, (lo, hi)


def test_zero_miss_recall_lower_bound_is_not_one() -> None:
    est = RecallEstimate(true_positives=6, observed_fn=0, sample_size=60,
                         population=585)
    assert est.point == 1.0          # the naive answer...
    lo, _ = est.interval
    assert lo < 0.3                  # ...and the honest one
    assert "could be as high as" in est.describe()


def test_describe_always_carries_the_interval() -> None:
    est = RecallEstimate(true_positives=6, observed_fn=4, sample_size=60,
                         population=585)
    assert "95% CI" in est.describe()


def test_no_labelled_unflagged_fields_is_unmeasurable() -> None:
    est = RecallEstimate(true_positives=6, observed_fn=0, sample_size=0,
                         population=585)
    assert "unmeasurable" in est.describe()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _label(sample, path: str, answer: str, pii_type: str = "") -> Label:
    row = next(f for f in sample.fields if f.field == path)
    return Label(row_id=row.row_id, is_pii=_bool(answer), pii_type=pii_type, notes="")


def _bool(answer: str) -> bool | None:
    return {"y": True, "n": False, "u": None}[answer]


def test_precision_is_exact_over_the_flagged_census(sample) -> None:
    labels = [
        _label(sample, "mobile", "y", "INDIAN_MOBILE"),
        _label(sample, "aadhaar_no", "y", "AADHAAR"),
        _label(sample, "customer_name", "n"),
    ]
    score = score_audit(sample, labels)
    assert len(score.true_positives) == 2
    assert len(score.false_positives) == 1
    assert score.precision == pytest.approx(2 / 3)


def test_unflagged_pii_is_a_false_negative(sample) -> None:
    labels = [_label(sample, f.field, "y")
              for f in sample.by_stratum(Stratum.UNFLAGGED)[:2]]
    labels += [_label(sample, f.field, "n")
               for f in sample.by_stratum(Stratum.UNFLAGGED)[2:]]
    score = score_audit(sample, labels)
    assert len(score.false_negatives) == 2
    assert score.recall.observed_fn == 2
    assert score.recall.sample_size == 20


def test_unsure_is_excluded_from_every_denominator(sample) -> None:
    labels = [
        _label(sample, "mobile", "u"),
        _label(sample, "aadhaar_no", "y", "AADHAAR"),
        _label(sample, "customer_name", "n"),
    ]
    labels += [_label(sample, f.field, "u")
               for f in sample.by_stratum(Stratum.UNFLAGGED)]
    score = score_audit(sample, labels)
    assert len(score.unsure) == 21
    # `mobile` is neither a TP nor an FP.
    assert len(score.true_positives) == 1
    assert len(score.false_positives) == 1
    assert score.precision == pytest.approx(0.5)
    # No unflagged field was labelled, so recall is unmeasurable, not 1.0.
    assert score.recall.sample_size == 0
    assert "unmeasurable" in score.recall.describe()


def test_skip_misses_are_separate_from_recall(sample) -> None:
    """A field excluded before classification is a skip bug, not a miss."""
    labels = [_label(sample, "mobile", "y", "INDIAN_MOBILE")]
    labels += [_label(sample, f.field, "y")
               for f in sample.by_stratum(Stratum.SKIPPED)[:2]]
    labels += [_label(sample, f.field, "n")
               for f in sample.by_stratum(Stratum.SKIPPED)[2:]]
    labels += [_label(sample, f.field, "n")
               for f in sample.by_stratum(Stratum.UNFLAGGED)]
    score = score_audit(sample, labels)

    assert len(score.skip_misses) == 2
    assert score.skipped_labelled == 5
    assert len(score.false_negatives) == 0        # not counted as classifier misses
    assert score.recall.observed_fn == 0
    # 2 of 5 labelled, extrapolated over 10 skipped fields.
    assert score.estimated_skip_misses == pytest.approx(4.0)


def test_type_accuracy_among_true_positives(sample) -> None:
    labels = [
        _label(sample, "mobile", "y", "INDIAN_MOBILE"),
        _label(sample, "aadhaar_no", "y", "EMAIL"),        # wrong type
        _label(sample, "customer_name", "y", ""),          # no type given
    ]
    score = score_audit(sample, labels)
    assert len(score.true_positives) == 3
    assert score.type_correct == 1
    assert len(score.type_wrong) == 1
    assert score.type_unspecified == 1
    assert score.type_accuracy == pytest.approx(0.5)


def test_unknown_row_id_is_reported(sample) -> None:
    score = score_audit(sample, [Label(row_id="zzzzzzzz", is_pii=True,
                                       pii_type="", notes="")])
    assert score.unmatched_row_ids == ["zzzzzzzz"]


@pytest.mark.parametrize("raw,expected", [
    ("AADHAAR", PIIType.AADHAAR),
    ("aadhar", PIIType.AADHAAR),
    ("phone", PIIType.INDIAN_MOBILE),
    ("Mobile Number", PIIType.INDIAN_MOBILE),
    ("date of birth", PIIType.DATE_OF_BIRTH),
    ("person_name", PIIType.PERSON_NAME),
    ("name", PIIType.PERSON_NAME),
    ("", None),
    ("something else", None),
])
def test_type_label_normalisation(raw, expected) -> None:
    assert normalise_type_label(raw) is expected


# --------------------------------------------------------------------------
# Cause attribution and misses.md
# --------------------------------------------------------------------------


def test_fn_cause_distinguishes_merge_logic_from_missing_detector(scan) -> None:
    """The grouping is the finding: a name signal that was dropped needs a
    merge-logic fix; no signal at all needs a detector."""
    scan.fields.append(_record("pan_number", evidence=["ABXXXXXX1Z"]))
    scan.fields.append(_record("zzz_opaque", evidence=["abXXef"]))
    drawn = draw_sample(scan, name="u", seed=SEED,
                        sizes={**STRATA, Stratum.UNFLAGGED: 42, Stratum.SKIPPED: 0})
    labels = [_label(drawn, "pan_number", "y"), _label(drawn, "zzz_opaque", "y")]
    score = score_audit(drawn, labels)

    causes = {d.row.field: d.cause for d in score.disagreements}
    assert "name suggested PAN" in causes["pan_number"]
    assert "no name signal" in causes["zzz_opaque"]


def test_misses_md_groups_by_cause_and_leads_with_the_biggest(sample) -> None:
    labels = [_label(sample, f.field, "y")
              for f in sample.by_stratum(Stratum.UNFLAGGED)]
    labels.append(_label(sample, "customer_name", "n"))
    score = score_audit(sample, labels)
    text = render_misses(score)

    assert "False negatives" in text
    assert "False positives" in text
    assert "name-only verdict" in text          # the FP's cause
    assert f"({len(score.false_negatives)} field(s))" in text


def test_misses_md_on_full_agreement_does_not_claim_success(sample) -> None:
    labels = [_label(sample, f.field, "y")
              for f in sample.by_stratum(Stratum.FLAGGED)]
    labels += [_label(sample, f.field, "n")
               for f in sample.by_stratum(Stratum.UNFLAGGED)]
    text = render_misses(score_audit(sample, labels))
    assert "No disagreements" in text
    assert "consistent with good recall *and*" in text


# --------------------------------------------------------------------------
# Reproducibility and storage
# --------------------------------------------------------------------------


def test_audit_round_trips_through_disk(sample, tmp_path) -> None:
    save_audit(sample, tmp_path)
    restored = load_audit(tmp_path)
    assert restored.seed == sample.seed
    assert restored.population == sample.population
    assert restored.classifier_version == sample.classifier_version
    assert [f.row_id for f in restored.fields] == [f.row_id for f in sample.fields]


def test_audit_dir_is_named_for_source_and_date() -> None:
    from datetime import date

    path = audit_dir("audit", "prod-mysql", on=date(2026, 8, 5))
    assert path.name == "prod-mysql-20260805"


def test_scores_json_labels_recall_as_an_estimate(sample) -> None:
    labels = [_label(sample, "mobile", "y", "INDIAN_MOBILE")]
    labels += [_label(sample, f.field, "n")
               for f in sample.by_stratum(Stratum.UNFLAGGED)]
    payload = _serialisable(score_audit(sample, labels))
    assert payload["precision"]["exact"] is True
    assert payload["recall"]["exact"] is False
    assert "ci95" in payload["recall"]
    assert payload["recall"]["weak"] is True
    assert "Clopper-Pearson" in payload["recall"]["method"]


def test_terminal_scorecard_separates_exact_from_estimated(sample) -> None:
    labels = [_label(sample, "mobile", "y", "INDIAN_MOBILE")]
    labels += [_label(sample, f.field, "n")
               for f in sample.by_stratum(Stratum.UNFLAGGED)]
    text = format_score(score_audit(sample, labels))
    assert "PRECISION  (exact" in text
    assert "RECALL  (estimated" in text
    assert "95% CI" in text
    assert "census" in text


# --------------------------------------------------------------------------
# The no-raw-values invariant, extended to audit artefacts
# --------------------------------------------------------------------------


def test_sampled_field_masks_evidence_like_a_finding() -> None:
    from piiscanner.audit.sampling import SampledField

    row = SampledField(
        row_id="abcdefgh", dataset="db", collection="t", field="mobile",
        native_type="varchar", stratum=Stratum.FLAGGED,
        evidence=["9812345621", "priya@example.com"],
    )
    assert row.evidence == ["98XXXXXX21", "prXXXXXXXXXXXXXom"]


def test_no_raw_values_reach_any_audit_artefact(sample, tmp_path) -> None:
    """The same invariant Finding has, asserted across every file we write."""
    raw = "9812345621"
    sample.fields[0].evidence = [raw]          # simulate a leak upstream
    save_audit(sample, tmp_path)
    worksheet = write_worksheet(sample, tmp_path / "worksheet.csv")

    for path in (tmp_path / "sample.json", worksheet):
        assert raw not in path.read_text(), path
