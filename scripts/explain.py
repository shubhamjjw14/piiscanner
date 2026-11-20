"""Dry-run tracer: walk one field through the pipeline, step by step.

Answers "why did you flag this?" -- the first question a compliance reviewer
asks. Prints every decision with the numbers behind it.

    python scripts/explain.py <dsn> <table> <column> [sample_size]

Never prints a raw value: examples are masked exactly as findings are.
"""

from __future__ import annotations

import sys

from piiscanner.classify.engine import (
    AGREEMENT_BONUS,
    CHECKSUM_NOISE_FLOOR,
    CONF_NAME_ONLY,
    MIN_FREE_TEXT_CONSIDERED,
    MIN_FREE_TEXT_MATCHES,
    REPORT_THRESHOLD,
    SMALL_SAMPLE_PENALTY,
    _score_values,
    classify_field,
)
from piiscanner.classify.layer1_names import classify_name, tokenise
from piiscanner.classify.layer2_values import (
    MIN_CONFIDENT_SAMPLE,
    classify_values,
    looks_like_free_text,
    refutes,
)
from piiscanner.classify.patterns import NAME_ONLY_TYPES, RULES_BY_TYPE, normalise
from piiscanner.config import ScanConfig, build_connector
from piiscanner.models import mask_value

BAR = "=" * 78


def step(n: int, title: str) -> None:
    print(f"\nSTEP {n} -- {title}")
    print("-" * 78)


def explain(dsn: str, table: str, column: str, sample_size: int = 500) -> None:
    config = ScanConfig(source=dsn, sample_size=sample_size)
    connector = build_connector(config)
    with connector:
        dataset = getattr(connector, "default_dataset", None) or connector.list_datasets()[0]
        schema = next(
            (s for s in connector.get_schema(dataset)
             if s.collection == table and s.name == column),
            None,
        )
        if schema is None:
            sys.exit(f"no such field: {table}.{column}")
        if schema.skip:
            print(f"{table}.{column} SKIPPED before sampling: {schema.skip_reason}")
            return
        sample = connector.sample_field(schema, sample_size)

    print(BAR)
    print(f"DRY RUN  {table}.{column}")
    print(f"         type={schema.native_type}  table rows~{schema.approx_row_count:,}")
    print(BAR)

    step(1, "TOKENISE THE COLUMN NAME")
    print(f"  {column!r}  ->  {tokenise(column)}")
    print("  whole tokens only, so 'pan' can never match inside 'japan'")

    step(2, "LAYER 1: look each token up in the keyword map")
    hints = classify_name(column)
    if not hints:
        print("  no token matched a keyword  ->  Layer 1 has NO opinion")
        print("  anything found below is Layer 2 working alone")
    for h in hints:
        print(f"  token {h.matched_token!r} suggests {h.pii_type.value}"
              f"   (fixed at {h.confidence}; Layer 1 never exceeds this)")
    hinted = {h.pii_type for h in hints}

    step(3, "SAMPLE THE VALUES")
    non_null = sample.non_null
    print(f"  drew {sample.total_sampled} values: "
          f"{sample.null_count} null + {sample.empty_count} empty string "
          f"= {sample.blank_rate:.0%} blank")
    print(f"  {len(non_null)} usable, {sample.distinct_ratio:.0%} distinct "
          "(of the usable ones)")
    for v in non_null[:3]:
        print(f"    example (masked): {mask_value(normalise(v))}")
    if not non_null:
        print("\n  GATE: no non-null values  ->  NO FINDING")
        print("  an empty column holds no personal data, whatever it is called")
        return
    texts = [normalise(v) for v in non_null if normalise(v)]
    prose = looks_like_free_text(texts)
    print(f"\n  reads like prose? {prose}  ->  mode = "
          f"{'FREE TEXT (search inside)' if prose else 'WHOLE VALUE (must match entirely)'}")

    step(4, "LAYER 2: run every value rule")
    verdicts = classify_values(sample)
    if not verdicts:
        print("  no rule matched anything  ->  Layer 2 has NO opinion")
    for v in verdicts:
        mode = "free-text" if v.free_text else "whole-value"
        print(f"  {v.pii_type.value:<16} matched {v.matched_count:>4}/{v.considered:<4}"
              f" = {v.match_rate:>6.1%}  [{mode}]")
        if "candidate_pass_rate" in v.stats:
            print(f"                   of right-shaped substrings, "
                  f"{v.stats['candidate_pass_rate']:.0%} passed the checksum")

    step(5, "SCORE EACH LAYER-2 CANDIDATE")
    if not verdicts:
        print("  nothing to score")
    for v in verdicts:
        rule = RULES_BY_TYPE[v.pii_type]
        supported = v.pii_type in hinted
        print(f"  {v.pii_type.value}")
        print(f"    flags: checksum={rule.checksum} weak={rule.weak} "
              f"needs_name={rule.requires_name_support} | name agrees={supported}")
        if v.free_text and not (v.matched_count >= MIN_FREE_TEXT_MATCHES
                                and v.considered >= MIN_FREE_TEXT_CONSIDERED):
            print(f"    GATE free-text evidence too thin (needs >={MIN_FREE_TEXT_MATCHES}"
                  f" matches over >={MIN_FREE_TEXT_CONSIDERED} values)  ->  DROPPED")
            continue
        base = _score_values(v, supported)
        if base is None:
            if rule.checksum:
                why = f"checksum rate below the {CHECKSUM_NOISE_FLOOR:.0%} noise floor"
            elif rule.requires_name_support and not supported:
                why = "pattern too generic and the name does not corroborate"
            elif rule.weak:
                why = "weak type below the 50% bar"
            else:
                why = "match rate below 10%"
            print(f"    {why}  ->  DROPPED")
            continue
        print(f"    ladder gives {base:.2f}")
        if supported:
            bumped = min(0.99, max(base, CONF_NAME_ONLY) + AGREEMENT_BONUS)
            print(f"    both layers agree: max({base:.2f}, {CONF_NAME_ONLY})"
                  f" + {AGREEMENT_BONUS} = {bumped:.2f}")

    step(6, "LAYER 1 HINTS THE VALUES DID NOT CONFIRM")
    leftover = [h for h in hints if h.pii_type not in {v.pii_type for v in verdicts}]
    if not leftover:
        print("  none -- the values addressed every name hint")
    for h in leftover:
        if h.pii_type in NAME_ONLY_TYPES:
            print(f"  {h.pii_type.value}: no value-level test exists for this type,")
            print(f"    so nothing can confirm OR refute it  ->  {CONF_NAME_ONLY} UNVERIFIED")
        elif refutes(sample, h.pii_type, verdicts):
            print(f"  {h.pii_type.value}: testable type, {len(non_null)} values, "
                  "almost none matched")
            print("    ->  REFUTED, dropped entirely")
            print("    (this is what stops a column named `pan` full of panel SKUs)")
        else:
            print(f"  {h.pii_type.value}: inconclusive  ->  {CONF_NAME_ONLY} UNVERIFIED")

    step(7, "PENALTIES AND THRESHOLD")
    thin = 0 < len(non_null) < MIN_CONFIDENT_SAMPLE
    print(f"  thin sample? {len(non_null)} non-null -> {'YES' if thin else 'no'}"
          f" (floor {MIN_CONFIDENT_SAMPLE})")
    if thin:
        print(f"    -{SMALL_SAMPLE_PENALTY} applies to VALUE-verified findings only.")
        print("    A name-only finding keeps its score: a column's name is just as")
        print("    informative at 11 rows as at 11 million.")
    print(f"  report threshold: {REPORT_THRESHOLD}")

    step(8, "FINAL")
    findings = classify_field(sample, source="dry-run")
    if not findings:
        print("  NO FINDING REPORTED")
        return
    for f in findings:
        layers = "+".join(
            d.value.replace("layer1_names", "name").replace("layer2_values", "value")
            for d in f.detected_by
        )
        print(f"  {f.pii_type.value}  confidence={f.confidence}  via {layers}"
              f"  unverified={f.unverified}")
        print(f"    evidence: {f.evidence or '(none - name-only finding)'}")
        if f.notes:
            print(f"    notes: {f.notes}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    explain(sys.argv[1], sys.argv[2], sys.argv[3],
            int(sys.argv[4]) if len(sys.argv) > 4 else 500)
