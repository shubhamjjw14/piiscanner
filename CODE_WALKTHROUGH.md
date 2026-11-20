# piiscanner — code walkthrough

How the whole thing works, in order, with real traces.

---

## 0. The one-line version

```
DSN → connect (read-only) → list tables → read schema → sample 500 values per column
    → Layer 1 guesses from the NAME → Layer 2 checks the VALUES
    → merge the two into one confidence score → mask evidence → report
```

Two layers, and the whole design rests on one idea: **the name is a hint, the
values are the evidence.** A name alone can never score above 0.5. Values can
reach 0.99. When both agree you get a bonus. When they disagree, the values win.

## 1. File map

```
piiscanner/
  models.py            the vocabulary: PIIType, Finding, ScanResult, mask_value
  config.py            ScanConfig, build_connector, log filter
  cli.py               3 commands: scan / report / benchmark

  connectors/
    base.py            the Connector interface, FieldSample, reconnect policy
    keyspace.py        string-as-big-integer interpolation, shared by both
    mysql.py           MySQL: schema reading + seekable-key sampling
    mongodb.py         MongoDB: strategy choice + flatten nested docs into paths

  classify/
    checksums.py       Verhoeff, Luhn, GSTIN, PAN — the maths
    patterns.py        the rule registry: regex + validator + 3 flags per type
    layer1_names.py    column-name heuristics (tokenise → keyword lookup)
    layer2_values.py   run every rule over the sampled values
    engine.py          merge Layer 1 + Layer 2 → Finding.  The brain.


  audit/               hand-labelled precision/recall on a real schema
  report/              json_out.py, text_out.py
  benchmark.py         precision / recall against the answer key
  testdata/            synthetic Indian dataset + its ground truth

scripts/
  scan_local.sh        grant → prove-cannot-write → scan → revoke
  explain.py           the dry-run tracer used throughout this document
```

Finding one person's rows across a database, and writing their deletion plan,
is piiscanner's full version and is not part of this package or described here.

Dependency direction is strictly one-way:

```
cli → engine → {layer1, layer2} → patterns → checksums
       ↓
   connectors → models
```

`classify/` never imports a database driver. `connectors/` never imports a
classifier. That is what makes "adding a source is a new file, not a refactor"
true: a new connector implements six methods and nothing else changes.

---

## 2. The flow, stage by stage

### Stage 0 — the CLI reads your intent

`cli.py :: scan()` builds a `ScanConfig` and calls `build_connector(config)`,
which switches on the DSN scheme (`mysql://` vs `mongodb://`) and returns the
right connector. That switch is the *only* place the rest of the code knows
which database it's talking to.

`configure_logging()` attaches `NoValuesFilter` — a `logging.Filter` that drops
any log record containing a raw sample payload — and sets `propagate = False`
so nothing escapes to the root logger unfiltered.

### Stage 1 — connect, read-only

`MySQLConnector.connect()`:

```python
self._conn = pymysql.connect(..., local_infile=False)
cur.execute("SET SESSION TRANSACTION READ ONLY")
```

Three independent guards, because one is not enough:

1. `SET SESSION TRANSACTION READ ONLY` — the **server** now rejects writes.
2. `_query()` refuses anything that isn't `SELECT` or `SHOW`, raising
   `ReadOnlyViolation` before it reaches the wire. This catches *our* bugs.
3. The account itself is `GRANT SELECT` only, and `scan_local.sh` proves it by
   attempting a `CREATE TABLE` and aborting if it succeeds.

Mongo's equivalent is `assert_read_only_pipeline()`, which rejects `$out` and
`$merge` — the only two aggregation stages that can write.

### Stage 2 — what columns exist

`get_schema(dataset)` returns a `list[FieldSchema]`, one per column:

```python
FieldSchema(dataset, collection, name, native_type, nullable,
            approx_row_count, skip, skip_reason)
```

Two things worth knowing:

- **Row counts are estimates on purpose.** `_table_row_counts()` reads
  `information_schema.tables.table_rows`. `SELECT COUNT(*)` on a 40M-row InnoDB
  table is exactly the full scan we promised never to do.
- **Unscannable columns are returned with `skip=True`, not dropped.** The report
  can then say "27 fields skipped" instead of silently pretending they don't
  exist. `_skip_reason()` skips blobs, geometry, and `tinyint` (MySQL's fake
  boolean — too narrow to hold any identifier).

Numeric types are deliberately **not** skipped. Aadhaar and mobile numbers are
routinely stored as `BIGINT`, and skipping numerics is the fastest way to miss
them.

For MongoDB there is no schema, so it's inferred. `flatten_document()` turns

```python
{"orders": [{"shipping": {"pincode": "400001"}},
            {"shipping": {"pincode": "560001"}}]}
```

into `{"orders[].shipping.pincode": ["400001", "560001"]}`. The `[]` matters:
without merging array indices, one real field becomes hundreds of
one-document "schemas".

### Stage 3 — sampling, bounded three ways

`_choose_strategy()` picks per table:

| Strategy | When | Cost | Quality |
|---|---|---|---|
| `order_by_rand` | ≤ 20,000 rows | reads that table | true uniform sample |
| `key_range` | larger, has a **seekable** key | index seeks, bounded by sample size | good |
| `head` | larger, no seekable key at all | `LIMIT 500`, sweeping across runs | **biased**, logged as a warning |

"Seekable" is not "integer". Any column with a total order works, because seeking
needs *some* value between MIN and MAX, not a numerically meaningful one — so
`_seekable_key()` accepts integers, dates, and strings, and returns a `SeekKey`
carrying the column's charset and collation alongside its kind.

That generalisation is not cosmetic. On a real schema of 265 tables, 121 were
over the threshold and only **17** had integer primary keys — so 104 large tables,
including one of 660M rows, were being sampled from their first 500 rows. After
the change, 120 of the 121 use index seeks.

`_key_range_sample()` picks random values in the key range, seeks to each, and
reads 25 rows forward. Probes are de-duplicated by key because they can overlap,
and the probe count is capped at 4× so a sparse key space cannot loop for long.

For a string key the comparison is written `WHERE k >= CONVERT(%s USING cs)
COLLATE c`. Both halves are needed. The connection is `utf8mb4`, so a bound
parameter arrives as `utf8mb4`; against a `latin1` key column MySQL **refuses the
comparison** (error 1267) rather than coercing, because narrowing could lose
characters — `CONVERT` settles that. But two `utf8mb4` operands still fail when
one is `utf8mb4_unicode_ci` and the other `utf8mb4_0900_ai_ci`: both are IMPLICIT,
so neither yields — `COLLATE` settles that by making our side EXPLICIT. Both wrap
the **parameter**, never the column; wrapping the column would make it unindexable
and turn every probe into the full scan this strategy exists to avoid.

The generated probe value is also kept inside the key column's own character
repertoire (`_charset_base`). A code point the column cannot represent is folded
to `?` on conversion, which would quietly drag every probe toward the start of the
key space.

**Resilience lives in the base class**, so both connectors get it. A table that
cannot be read is recorded in `connector.sampling_errors` and skipped — one bad
table must not cost the other 264 their results, since `iter_samples` is a
generator and a raise ends the whole dataset. A *connection* failure is handled
oppositely: `with_reconnect()` reconnects with backoff and retries, and if the
link is genuinely gone the error propagates to end the scan. Merging those two
cases is how a dropped VPN once got reported as 79 tables containing no personal
data.

The output is a `FieldSample`. This object holds the only raw data in the
system, and it's guarded:

```python
def __repr__(self) -> str:
    return (f"FieldSample(field={self.schema.name!r}, "
            f"sampled={self.total_sampled}, nulls={self.null_count}, "
            "values=<redacted>)")
__str__ = __repr__
```

So an accidental `log.debug(sample)` or a traceback cannot leak values.

`iter_samples()` samples **one query per table, not per column** — on a
60-column table that's 1 query instead of 60.

#### Mongo's version of the same problem, which is not the same shape

`$sample` looks like it removes the question — it's the server's own random
sampling stage. It doesn't. `$sample` uses a cheap pseudo-random cursor **only
when the requested size is under 5% of the collection**; above that, the server
silently falls back to a full collection scan feeding a top-k sort.

At the default sample size of 500, that fallback covers **every collection under
10,000 documents**. Measured, not assumed — `explain` on a 5,000-document
collection shows the random cursor at size 250 and `COLLSCAN + SORT` at 500.

So `MongoDBConnector._choose_strategy()` decides explicitly:

| Strategy | When | Why |
|---|---|---|
| `sample` | request is under 5% of the collection | server uses its random cursor. This is the *large*-collection case, and it is a true uniform sample |
| `scan` | small in both documents and bytes | the fallback scan is genuinely cheap here, and a full read beats seeking |
| `id_range` | too many documents, or too many bytes for a blocking sort | seek on `_id` — bounded, though not uniform |

The byte test uses `collStats.avgObjSize`: MongoDB aborts a blocking sort at 100MB
without `allowDiskUse`, and a read-only scanner should not make a customer's
server spill to disk.

`_id` is the universal seek key — every collection has one and it is always
indexed — so `id_range` is the direct analogue of MySQL's `key_range`.
`_random_id_point()` handles the three shapes `_id` actually takes: `ObjectId`
(its 12 bytes read as a big-endian integer are ordered the same way the index
is), `int`, and `str`. Anything else returns `None` and the caller falls back to
`$sample` rather than guessing. `bool` is excluded explicitly — it subclasses
`int`, and an unguarded check would interpolate `True`/`False` into a meaningless
0/1 seek.

Every operation carries `maxTimeMS`, so one pathological collection cannot hang a
scan while the server keeps working on a query whose client has gone.

### Stage 4 — Layer 1, the column name

`layer1_names.py :: tokenise()` first. This is the load-bearing detail:

```
'aadhaar_no'   -> ['aadhaar', 'no']
'customerEmail'-> ['customer', 'email']     (camelCase split)
'IPAddress'    -> ['ip', 'address']
'japan_office' -> ['japan', 'office']       ← no 'pan' token
'company_panel'-> ['company', 'panel']      ← no 'pan' token
'dobbs_lane'   -> ['dobbs', 'lane']         ← no 'dob' token
```

**Whole-token matching, never substring.** Substring matching is the classic bug
in name-based classifiers, and it's how a tool loses the customer's trust on
their first scan.

Then `classify_name()` does three passes:

1. **Bigrams first** — `("card", "number") → CARD_NUMBER`, because the pair
   means something the individual tokens don't.
2. **Single tokens** — the `KEYWORDS` map, ~130 entries.
3. **Negators** — `NEGATING_TOKENS` cancels a match. `company_name`,
   `product_name`, `issuer_name` are not people.

The negator rule has a subtlety worth understanding:

```python
if token_set & negators and token not in negators:
    continue
```

A negating token only cancels a match **it did not itself create**. So
`ip_address` keeps its IP_ADDRESS hint (created by `ip_address`) while losing
its ADDRESS hint (`ip` is an ADDRESS negator). One column name, two candidate
types, and the right one survives.

Every hint comes out at exactly `MAX_NAME_CONFIDENCE = 0.5`. Layer 1 has no
gradations — it either recognises a token or it doesn't.

### Stage 5 — Layer 2, the values

`patterns.py` holds the registry. Each `PatternRule` is a regex, an optional
validator, and three flags:

| Flag | Meaning | Example |
|---|---|---|
| `checksum` | carries a real check digit | AADHAAR, GSTIN, CARD_NUMBER |
| `weak` | too common for values alone to be confident | PINCODE, AGE, PASSPORT |
| `requires_name_support` | so generic it's noise without the name | BANK_ACCOUNT, DEVICE_ID |

`layer2_values.py :: classify_values()` then runs every rule over the sample and
returns one `ValueVerdict` per type that fired at all. It picks a mode per
column, not per value:

- **whole-value** (default): `regex.fullmatch(text)`. A 16-digit account number
  is therefore *not* a 10-digit mobile that happens to sit inside it.
- **free-text**: used when `looks_like_free_text()` says the column is prose
  (median length ≥ 40, or most values have 3+ spaces). Then the rule is searched
  *inside* the value.

Free-text regexes use lookarounds, not `\b`:

```python
free_text_regex=re.compile(r"(?<!\d)(?:\+?91[\-\s]?)?[6-9]\d{9}(?!\d)")
```

`\b` doesn't fire next to a `+`, and it would let a 10-digit run inside a
16-digit number match. `(?<!\d)...(?!\d)` says "no digit either side".

### Stage 6 — the merge (engine.py). This is the brain.

`classify_field(sample, source)` in ~120 lines. In order:

**Gate 0 — empty column.**
```python
if not sample.non_null:
    return []
```
An entirely empty column holds no personal data, whatever it's called.

**Build Layer 2 candidates.** For each verdict, apply the gates, then score:

```python
def _score_values(verdict, name_supports):
    if verdict.checksum:
        effective = verdict.stats.get("candidate_pass_rate", rate) if verdict.free_text else rate
        if effective >= 0.8:            return 0.95 if not verdict.free_text else 0.70
        if effective >= 0.30:           return 0.70
        return None                     # noise floor
    if verdict.requires_name_support and not name_supports:  return None
    if verdict.weak:                    return 0.50 if rate >= 0.5 else None
    if rate >= 0.9:                     return 0.75
    if rate >= 0.3:                     return 0.60
    if rate >= 0.1:                     return 0.45
    return None
```

**The agreement bonus:**
```python
score = min(0.99, max(score, 0.50) + 0.10)
```

**Add unconfirmed Layer 1 hints** — but only after two gates:

- `_state_token_gate()` — an ADDRESS hint resting only on the token `state` must
  see a real Indian state name among the values. Otherwise every workflow
  `state` enum is an address.
- `refutes()` — if the type is value-testable, we have ≥20 values, and
  essentially none match, **drop the hint entirely.** This is what stops a
  column named `pan` full of solar-panel SKUs.

**One verdict per field:**
```python
if any(c.verified for c in candidates.values()):
    candidates = {t: c for t, c in candidates.items() if c.verified}
```
Value evidence displaces name-only guesses. Without this, `ip_addr` yields both
IP_ADDRESS (right) and ADDRESS (noise).

**Thin-sample penalty**, and note carefully what it applies to:
```python
if 0 < non_null < 20:
    if candidate.verified:            # value evidence only
        confidence = max(0.0, confidence - 0.15)
    notes.append(f"only {non_null} non-null values sampled")
```
A cardholder-name column is named that whether it holds 11 rows or 11 million.
Penalising the *name* because the table is small dropped genuine PII below the
reporting threshold on a real schema.

**Threshold:** below `REPORT_THRESHOLD = 0.4`, drop it.

### Stage 7 — report

`Finding` is constructed, and this is where constraint 2 becomes structural:

```python
@field_validator("evidence", mode="before")
def _mask_evidence(cls, value):
    return [mask_value(item) for item in value][:3]
```

Every evidence item is masked at construction time. `mask_value` is idempotent,
so pre-masked input passes through unchanged and a raw value that slipped
through gets masked here. **There is no code path that produces a `Finding`
holding a raw value** — not a convention, a type-level invariant.

---

## 3. Dry runs

### 3a. Both layers agree — `kyc.aadhaar_no` → **0.99**

```
STEP 1  'aadhaar_no'  ->  ['aadhaar', 'no']
STEP 2  token 'aadhaar' suggests AADHAAR   (fixed at 0.5)
STEP 3  500 values, 0 null, 100% distinct   mode = WHOLE VALUE
        examples (masked): 26XXXXXXXX85  20XXXXXXXX02  42XXXXXXXX84
STEP 4  AADHAAR       500/500 = 100.0%
        BANK_ACCOUNT  381/500 =  76.2%     ← 12-digit numbers look like accounts
        INDIAN_MOBILE   2/500 =   0.4%
STEP 5  AADHAAR: checksum=True | name agrees=True
          ladder gives 0.95
          both agree: max(0.95, 0.5) + 0.1 = 0.99
        BANK_ACCOUNT: needs_name and the name says AADHAAR, not account -> DROPPED
        INDIAN_MOBILE: below 10% -> DROPPED
STEP 8  AADHAAR confidence=0.99 via name+value
```

Note step 4: **76% of these Aadhaar numbers also match BANK_ACCOUNT**, because
`\d{9,18}` matches any 12-digit number. `requires_name_support` is the only
thing that stops this column producing two findings.

### 3b. Layer 2 alone — `legacy_import.col_17` → **0.95**

The point of the whole exercise: a column whose name tells you nothing.

```
STEP 2  no token matched  ->  Layer 1 has NO opinion
STEP 4  AADHAAR       500/500 = 100.0%
STEP 5  AADHAAR: ladder gives 0.95     (no agreement bonus — nothing to agree with)
STEP 8  AADHAAR confidence=0.95 via value
```

0.95 not 0.99: no name support, so no bonus. A name-based scanner finds nothing
here at all.

### 3c. Free text — `support_tickets.notes` → **0.45**

```
STEP 3  500 values, 18% distinct
        examples: CuXXXXXXX...XXXk.   (long — this is prose)
        reads like prose? True  ->  mode = FREE TEXT (search inside)
STEP 4  INDIAN_MOBILE  80/500 = 16.0%  [free-text]
                       of right-shaped substrings, 100% passed the check
STEP 5  ladder gives 0.45              (rate 0.10–0.30 band)
STEP 8  INDIAN_MOBILE confidence=0.45 via value
        evidence: ['83XXXXXX09', '+9XXXXXXXXXX64', '+9XXXXXXXXXX71']
        notes: found inside free text in 16% of sampled values
```

Two things to see. First, the evidence is the **extracted substring**, not the
sentence — masking a 90-character note would leak most of it. Second, 16% is a
real finding but a weaker one, and 0.45 says exactly that.

### 3d. A decoy — `decoys.order_reference` → **nothing**

12-digit order references. Structurally identical to Aadhaar.

```
STEP 4  BANK_ACCOUNT   500/500 = 100.0%
        INDIAN_MOBILE    3/500 =   0.6%
        (AADHAAR does not appear at all — the Verhoeff digit fails)
STEP 5  BANK_ACCOUNT: needs_name, name does not corroborate -> DROPPED
STEP 8  NO FINDING REPORTED
```

The checksum did the work. Without it these 500 values are indistinguishable
from Aadhaar numbers.

### 3e. Refutation — `decoys.pan`

```
STEP 2  token 'pan' suggests PAN  (0.5)
STEP 4  no rule matched anything  ->  Layer 2 has NO opinion
STEP 6  PAN: testable type, 500 values, almost none matched
          ->  REFUTED, dropped entirely
STEP 8  NO FINDING REPORTED
```

Without refutation, *any* column named `pan` is a permanent 0.5 finding.

---

## 4. The checksum, by hand

Verhoeff folds each digit through a permutation table and a dihedral-group
multiplication table. Valid means the accumulator lands on zero.

```
23456789012 + check digit 4  ->  23XXXXXXXX24
  accumulator ends at 0   ->  VALID
flip one digit               ->  INVALID
```

And a number I guessed from memory:

```
23XXXXXXXX26
  pos  0 digit 6  P[0][6]=6  D[0][6]=6
  pos  1 digit 2  P[1][2]=7  D[6][7]=4
  pos  2 digit 1  P[2][1]=8  D[4][8]=7
  ...
  pos 11 digit 2  P[3][2]=1  D[6][1]=5
  accumulator ends at 5   ->  INVALID
```

**Why the noise floor is 0.30.** Verhoeff has 10 possible check digits, so a
random 12-digit number passes about 1 time in 10. Measured over 100,000 random
numbers:

```
random 12-digit numbers passing Verhoeff: 9955/100000 = 10.0%
```

So a column at 10% Aadhaar is *pure chance*, not a mixed column.
`CHECKSUM_NOISE_FLOOR = 0.30` sits comfortably above it. This is not a tuned
number — it comes from the algorithm.

This showed up live: a partner account-id column on a real schema had **14.4% of
values passing Luhn**. The floor is the only thing between you and a phantom
CARD_NUMBER finding on real customer data.

---


## 5. The critical parts

Ranked by what breaks if it's wrong.

| Rank | What | Where | If broken |
|---|---|---|---|
| 1 | `mask_value` + the evidence validator | `models.py:24,154` | raw PII on disk. Product is unsellable |
| 2 | `_query` read-only guard + `SET SESSION TRANSACTION READ ONLY` | `mysql.py:137,123` | a write to a customer's production database |
| 3 | `tokenise()` whole-token matching | `layer1_names.py:29` | `japan_office` flagged as PAN; trust gone on scan 1 |
| 4 | The checksum validators | `checksums.py` | every 12-digit id becomes Aadhaar |
| 5 | `CHECKSUM_NOISE_FLOOR` | `engine.py:64` | ~10% chance passes reported as findings |
| 6 | `refutes()` | `layer2_values.py:169` | any column named `pan`/`dob` is always a finding |
| 7 | `requires_name_support` | `patterns.py:251` | every transaction id is a bank account |
| 8 | `_choose_strategy` bounds | `mysql.py`, `mongodb.py` | a scan that takes down production |
| 9 | `FieldSample.__repr__` redaction | `base.py:66` | one stray log line leaks a sample |
| 10 | `is_connection_lost` vs. a bad object | `base.py` | a dead link is reported as N clean tables |

Items 1, 2 and 9 are *security*. Items 3–7 are *credibility* — they don't
leak anything, they just make the report useless. Items 8 and 10 are
*operational*.


---

## 6. Code issues found

### Real bugs

**6.1 `null_rate` ignored empty strings — fixed.** `null_count` used to count
only `None` while `non_null` filtered both `None` and `""`, so a real
customer-name column that was 62% empty strings reported `"null_rate": 0.0` —
read by a reviewer as fully populated, and computed over a different
denominator than `distinct_ratio`. `FieldSample` now tracks `null_count`,
`empty_count`, and `non_null_count` separately and exposes `blank_rate =
(null + empty) / total_sampled` alongside a `null_rate` that stays strictly
"how much is `None`" (see `connectors/base.py`). Classification was and remains
unaffected — this was a reporting-only fix, proven by an unchanged 1.000/1.000
score on both synthetic benchmarks.

**6.2 `sample_field()` re-reads the entire schema on every call.**

```python
def sample_field(self, schema, sample_size):
    columns = [s.name for s in self._scannable_columns(...)]   # → get_schema() → 2 queries
```

Measured: 5 `sample_field` calls on *the same table* cost **12
`information_schema` queries**. `run_scan` uses `iter_samples`, which calls
`get_schema` once, so production scans are fine — but `sample_field` is a public
interface method, `explain.py` uses it, and the next caller will pay for it. No
schema caching in the MySQL connector.

**6.3 MySQL's row cache is a single slot — the exact bug already fixed in Mongo.**

```python
self._row_cache: tuple[str, str, int, list[dict]] | None = None    # mysql.py:109
```

Mongo has a bounded LRU precisely because `$sample` is random, so re-sampling
returns *different* documents. `ORDER BY RAND()` is random too. Interleaving
`sample_field` across two tables re-samples both, and the second sample of a
table is a *different* 500 rows than the first. `iter_samples` happens to
process tables consecutively, so it's safe today. It's a latent trap and an
asymmetry between the two connectors.

**6.4 `notes` has no masking validator.** `evidence` is protected structurally;
`notes` is a free-form string with no validator. Today every note is built from
rates and counts, so nothing leaks. But it is the one field on `Finding` where a
future code path *could* embed a value, and constraint 2 deserves the same
type-level guarantee there.

### Found by running against real databases — all fixed

None of these appear on a synthetic dataset. Every one needed real data of real
size to surface, and each is pinned by a test now.

**6.5 ~~Every scheduled re-scan probed identical rows, forever.~~**
`random.Random(seed if seed is not None else 0xC0FFEE)` — and no real caller
passes a seed. A fixed default meant every `key_range` probe on a large table
landed on exactly the same rows on every run, so re-scanning a 660M-row table
could never see anything new. `random.Random(None)` seeds from OS entropy.

**6.6 ~~A collation mismatch aborted a 265-table scan at table 76.~~** The probe
parameter binds as `utf8mb4`; 15 tables were `latin1`, 6 of them with varchar
primary keys. MySQL rejects that comparison outright (error 1267) rather than
coercing. Fixed with `CONVERT(...) COLLATE ...` on the parameter — see Stage 3.
The *second* half was found the same way: `CONVERT` alone still failed on a
`utf8mb4_unicode_ci` column, because both operands were then IMPLICIT and neither
yields.

**6.7 ~~One unreadable table cost the other 264 their results.~~** `iter_samples`
is a generator, so an exception ends the whole dataset rather than one table.
Now recorded in `sampling_errors` and reported loudly by the CLI — an unread table
must never be mistaken for a clean one.

**6.8 ~~A dropped VPN was reported as 79 tables containing no personal data.~~**
This one was caused *by* 6.7: once the socket died, every remaining table raised
`InterfaceError(0, '')` and each was dutifully recorded as its own skip. Read as
a report, that says "these 79 tables are clean" about tables nobody looked at.
A connection-level failure now reconnects with backoff and retries; if the link
is genuinely gone the error propagates and the scan ends honestly.

**6.9 ~~Mid-size Mongo collections were fully scanned and sorted.~~** `$sample`
was assumed cheap. It isn't above 5% of a collection — which at the default sample
size is every collection under 10,000 documents. See Stage 3.


### Dead code

- `PIIType.IP_ADDRESS_PRIVATE` — declared, never used. The private/public split
  is expressed as a `stats["private_share"]` note instead.
- `WEAK_TYPES`, `CHECKSUM_TYPES` in `models.py` — declared, never read.
  `patterns.py` carries the same information as per-rule flags, which is the
  better design. The frozensets are a leftover that can now drift out of sync
  with the rules without anything failing.
- `presence_count` — populated by Mongo and asserted in a test, but the engine
  never reads it. A path in 12 of 500 documents is meaningfully different from
  one in 500 of 500, and nothing acts on that yet.

### Honest limits, not bugs

- **`order_by_rand` does read the whole table** (up to 20,000 rows). It is
  bounded and cheap, but "never full-scan a table" is stated more absolutely in
  the spec than the code delivers. Worth saying plainly rather than being caught
  on it.
- **`_key_range_sample` is not uniform.** Probing `WHERE key >= start LIMIT 25`
  over-samples rows that follow a large gap in the key space. Good enough for
  detection, not a statistically clean sample.
- **Large keyless tables get the biased `head` strategy** — one real table has
  ~30k rows and no integer key. Logged as a warning, but PII appearing only in
  later rows is missed.
- **No detector for encrypted columns.** The single biggest gap; see
  [STATUS.md](STATUS.md) §5A.
