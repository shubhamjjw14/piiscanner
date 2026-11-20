"""The labelling worksheet, and the reason it looks the way it does.

The worksheet is the instrument.  If it is contaminated, every number this
package produces is decoration.

**It must not reveal the scanner's verdict.**  Not the confidence, not the PII
type, not whether the field was flagged, not which stratum the row came from,
not the row order.  A labeller who can tell what the scanner decided will
anchor on it -- agreeing with a false positive because the tool sounded
confident, or dismissing a real miss because the tool stayed quiet.  The result
would be a measurement of the labeller's trust in the tool rather than of the
tool.

Concretely, the columns below are the *only* ones written, and:

- rows are shuffled across strata, so position leaks nothing;
- ``row_id`` is random, not derived from the field path, so it cannot be
  reversed and does not sort into stratum order;
- there is no stratum column -- that mapping lives in ``sample.json``;
- ``evidence`` is masked, so labelling stays possible without reading raw PII.

One honest limitation, stated rather than papered over: skipped fields have no
evidence, because skipping them means never reading them.  A labeller may infer
that an evidence-less row with type ``blob`` was skipped.  That is acceptable --
the question being asked of those rows is "could this hold personal data?", and
the name and declared type are exactly the information the skip rule itself
used.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from piiscanner.audit.sampling import AuditSample

#: Columns the harness writes.  Nothing here is derived from a verdict.
GIVEN_COLUMNS = (
    "row_id",
    "field_path",
    "data_type",
    "sampled",
    "blank_rate",
    "distinct_ratio",
    "approx_rows",
    "example_1",
    "example_2",
    "example_3",
)

#: Columns the human fills in.
LABEL_COLUMNS = ("is_pii", "pii_type", "notes")

COLUMNS = GIVEN_COLUMNS + LABEL_COLUMNS

#: Accepted values for ``is_pii``.  Anything else is a parse error rather than a
#: silent default -- a mislabelled row is worse than a missing one.
YES = frozenset({"y", "yes", "true", "1", "pii"})
NO = frozenset({"n", "no", "false", "0"})
UNSURE = frozenset({"u", "unsure", "?", "maybe", "unknown"})

INSTRUCTIONS = """\
# piiscanner labelling worksheet
#
# Fill in is_pii, pii_type and notes for every row, then run:
#     piiscanner audit score <this directory>
#
# is_pii    y   this field holds personal data about an identifiable person
#           n   it does not
#           u   you cannot tell from what is shown  (excluded from all scores)
#
# pii_type  free text, e.g. AADHAAR, PERSON_NAME, EMAIL. Leave blank if is_pii=n.
#           Used only for type accuracy; a blank on a y row still counts as a
#           correct detection.
#
# notes     anything that would help explain a disagreement later.
#
# Example values are MASKED: first two and last two characters only. Some rows
# have no examples at all -- those fields were excluded before sampling, so
# judge them on name and declared type.
#
# Rows are deliberately shuffled and carry no verdict. Do not try to work out
# what the scanner decided; if you can, the labels are contaminated.
"""


@dataclass(frozen=True)
class Label:
    """One human judgement."""

    row_id: str
    is_pii: bool | None  # None == unsure
    pii_type: str
    notes: str

    @property
    def unsure(self) -> bool:
        return self.is_pii is None


def _fmt_rate(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def write_worksheet(sample: AuditSample, path: str | Path) -> Path:
    """Write the blind worksheet.

    Deliberately writes no verdict column.  Adding one "for convenience" is the
    single change that would invalidate every audit produced afterwards.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as handle:
        handle.write(INSTRUCTIONS)
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in sample.fields:
            evidence = list(row.evidence) + ["", "", ""]
            writer.writerow(
                {
                    "row_id": row.row_id,
                    "field_path": row.path,
                    "data_type": row.native_type,
                    "sampled": row.sample_count,
                    "blank_rate": _fmt_rate(row.blank_rate),
                    "distinct_ratio": _fmt_rate(row.distinct_ratio),
                    "approx_rows": (
                        "" if row.approx_row_count is None else row.approx_row_count
                    ),
                    "example_1": evidence[0],
                    "example_2": evidence[1],
                    "example_3": evidence[2],
                    "is_pii": "",
                    "pii_type": "",
                    "notes": "",
                }
            )
    return target


def _parse_is_pii(raw: str, row_id: str) -> bool | None:
    token = (raw or "").strip().lower()
    if token in YES:
        return True
    if token in NO:
        return False
    if token in UNSURE or token == "":
        return None
    raise ValueError(
        f"row {row_id}: cannot read is_pii={raw!r}. Use y, n or u."
    )


def read_labels(path: str | Path) -> tuple[list[Label], list[str]]:
    """Read a filled worksheet.

    Returns the labels and the row ids left blank.  Blank rows are reported to
    the caller rather than treated as "no": an unlabelled field is missing data,
    and counting it as a negative would inflate recall.
    """
    labels: list[Label] = []
    unlabelled: list[str] = []

    with Path(path).open(newline="", encoding="utf-8") as handle:
        # Skip the leading comment block.
        rows = (line for line in handle if not line.startswith("#"))
        for record in csv.DictReader(rows):
            row_id = (record.get("row_id") or "").strip()
            if not row_id:
                continue
            raw = (record.get("is_pii") or "").strip()
            if not raw:
                unlabelled.append(row_id)
                continue
            labels.append(
                Label(
                    row_id=row_id,
                    is_pii=_parse_is_pii(raw, row_id),
                    pii_type=(record.get("pii_type") or "").strip(),
                    notes=(record.get("notes") or "").strip(),
                )
            )
    return labels, unlabelled
