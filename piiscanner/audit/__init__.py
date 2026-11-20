"""Real-schema audit harness.

The synthetic benchmark scores 1.000 on both sources, which means it has stopped
discriminating: the detectors and the answer key were written together, so it
measures fit to fixtures rather than fitness for real data.  Nothing in the
repository measures recall on a real schema.

This package closes that gap the only way it can be closed -- a human labels a
sample of real fields, blind to what the scanner decided, and the labels are
scored against the verdicts afterwards.

Three properties the design turns on:

*stratified, not random*
    Uniform sampling over ~600 fields where 6 are flagged would return roughly
    zero flagged fields, and precision would be unmeasurable.  The flagged
    stratum is taken as a census; the others are sampled.

*blind by construction*
    The worksheet carries no confidence, no PII type, no verdict, and no
    stratum, and the rows are shuffled.  A labeller who can infer what the
    scanner decided produces labels that agree with it, and the exercise
    measures nothing.

*recall is an estimate and says so*
    Precision comes from a census and is exact.  Recall is extrapolated from a
    60-field sample and is reported with a confidence interval, every time.
"""

from piiscanner.audit.sampling import (
    STRATA,
    AuditSample,
    SampledField,
    Stratum,
    draw_sample,
)
from piiscanner.audit.scoring import AuditScore, RecallEstimate, score_audit
from piiscanner.audit.storage import load_audit, save_audit
from piiscanner.audit.worksheet import Label, read_labels, write_worksheet

__all__ = [
    "STRATA",
    "AuditSample",
    "AuditScore",
    "Label",
    "RecallEstimate",
    "SampledField",
    "Stratum",
    "draw_sample",
    "load_audit",
    "read_labels",
    "save_audit",
    "score_audit",
    "write_worksheet",
]
