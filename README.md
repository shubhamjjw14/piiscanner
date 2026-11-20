# piiscanner

Find the personal data in your database, without writing to it and without
keeping any of it.

piiscanner connects to MySQL, PostgreSQL or MongoDB with a read-only account,
reads a small sample of every column, and tells you which ones hold personal
data: names, phone numbers, email addresses, and the Indian identifiers that
India's Digital Personal Data Protection Act (DPDP) cares about -- Aadhaar,
PAN, GSTIN, voter ID, passport, UPI, IFSC, bank and card numbers.

```
$ piiscanner scan --source mysql://piiscanner_ro:PASSWORD@127.0.0.1:3306/piiscanner_testdata

  scanned    1 dataset(s), 8 collection(s), 63 field(s) examined, 1 skipped
  findings   36

  Conf  Location               PII type       Match   Rows  Via         Evidence (masked)
  0.99  kyc.aadhaar_no         AADHAAR         100%     3k  name+value  54XXXXXXXX62, 71XXXXXXXX51
  0.99  kyc.gstin              GSTIN           100%     3k  name+value  21XXXXXXXXXXXZM, ...
  0.95  legacy_import.col_17   AADHAAR         100%     2k  value       41XXXXXXXX34, 69XXXXXXXX37
  0.95  legacy_import.field_x  CARD_NUMBER     100%     2k  value       35XXXXXXXXXXX12, ...
  0.85  users.mobile           INDIAN_MOBILE   100%     5k  name+value  +9XXXXXXXXXX68, 63XXXXXX88
  ...
  10 finding(s) rest on the field name alone and were not confirmed by values.
```

(That is the synthetic test dataset that ships with it; `legacy_import.col_17`
is a column with a meaningless name that holds Aadhaar numbers, found by the
values alone.)

Every value it shows is masked. Nothing it reads is kept.

## Install

```bash
pip install piiscanner
```

Python 3.11 or newer. The database drivers come with it.

## Use

Give it an account that can only read. On MySQL:

```sql
CREATE USER 'piiscanner_ro'@'%' IDENTIFIED BY '...';
GRANT SELECT, SHOW VIEW ON shop.* TO 'piiscanner_ro'@'%';
```

On PostgreSQL:

```sql
CREATE ROLE piiscanner_ro LOGIN PASSWORD '...';
GRANT USAGE ON SCHEMA public TO piiscanner_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO piiscanner_ro;
```

Then:

```bash
piiscanner scan --source mysql://piiscanner_ro:PASSWORD@host:3306/shop --out findings.json
piiscanner scan --source postgresql://piiscanner_ro:PASSWORD@host:5432/shop
piiscanner scan --source mongodb://host:27017/shop --include 'prod_*' --exclude '*_archive'

piiscanner report findings.json --format text --all
```

The last line reads a saved scan again.

| Option | |
|---|---|
| `--sample-size` | values read per column (default 500) |
| `--include`, `--exclude` | table name patterns to scan or skip |
| `--dataset` | which databases or schemas to scan, when the connection can see several |
| `--out` | write the findings as JSON |
| `--all` | print every finding, not the top 10 |
| `--inventory` | record every column examined, not just findings |
| `--verbose` | the full log; it never contains values |

On PostgreSQL every schema is scanned unless you add `?schema=name` to the
connection string or pass `--dataset`.

## Use from Python

```python
import piiscanner

result = piiscanner.scan("mysql://piiscanner_ro:PASSWORD@host:3306/shop")

for f in result.findings:
    print(f.collection, f.field, f.pii_type.value, f.confidence, f.evidence)
# kyc aadhaar_no AADHAAR 0.99 ['54XXXXXXXX62', '71XXXXXXXX51']

if result.tables_not_read:
    print("not covered:", result.tables_not_read)
```

`scan()` takes the same choices as the command: `sample_size`, `include`,
`exclude` and `datasets`. It connects, scans and closes, with the same
guarantees -- read-only, evidence masked, nothing kept. The result is a
`ScanResult`; `result.model_dump_json()` gives the JSON `--out` writes.

`tables_not_read` names tables that could not be read (a permission, a
timeout). They are not covered by the result -- which is not the same as
holding no personal data.

## What it finds

26 types, in three groups:

| | |
|---|---|
| **Confirmed from the values** (checksum, structure or format) | Aadhaar (Verhoeff), card number (Luhn), GSTIN (mod-36), PAN, IFSC, Indian passport, vehicle registration, UPI ID, Indian mobile, email, date of birth, IP address (public and private) |
| **Confirmed from the values, only when the column name agrees** | bank account, PIN code, voter ID, device ID, age, gender |
| **Recognised by the column name only** | person name, address, location, health data, salary, religion, caste |

The first group is reported when enough of a column's sampled values pass
the check, whatever the column is called. The second group's patterns are
too generic on their own -- every 16-digit transaction id looks like a bank
account -- so the name must agree too. The third cannot be confirmed from
values at all (there is no checksum for a name), so these are reported at
confidence 0.5, marked *unverified*, for a person to look at.

### Why checksums matter

Roughly 1 in 10 random 12-digit numbers passes the Aadhaar checksum by chance,
because a single check digit has ten possible values. So a 12-digit
`order_reference` column will have about 10% of its values look like valid
Aadhaar numbers. Match *rate*, not the checksum alone, keeps it out: a real
Aadhaar column matches at 95-100%, an order reference at about 10%, and the
threshold sits between them. A test pins that 1-in-10 rate.

Three more rules keep false alarms down:

- **Refutation.** A column named `pan` whose values contradict it (solar-panel
  SKUs) is dropped, not reported on its name.
- **Name support.** The second group above, as described.
- **One verdict per column.** `ip_addr` is an IP address, not also an address.

## What it guarantees

Enforced in code and checked by tests, not promised in prose. The code that
touches your database is in `piiscanner/connectors/`; read it.

| | How |
|---|---|
| **Read-only** | MySQL sessions are set `TRANSACTION READ ONLY`, PostgreSQL sessions `default_transaction_read_only`. On top of that, any statement that is not a read raises `ReadOnlyViolation` before it is sent. MongoDB pipelines containing `$out` or `$merge` are refused. Run it as a SELECT-only account anyway. |
| **No values kept** | A finding's evidence is masked when it is created -- `9876543210` becomes `98XXXXXX10` -- by a validator on the model. No code path builds a finding holding a raw value. |
| **No values in logs** | Logs carry table and column names and counts only; a filter strips anything else; samples print as `<redacted>`. A test reads real values from the database and checks they appear nowhere in the JSON, the report or a full debug log. |
| **No network but your database** | No HTTP client is imported anywhere. |
| **Bounded cost** | Never a full table scan. Large MySQL tables are sampled by index seeks, PostgreSQL by `TABLESAMPLE`, MongoDB by `$sample` or `_id` ranges. |

## Accuracy

`piiscanner benchmark` scans a synthetic Indian dataset with a known answer
key, and exits non-zero on any miss or false alarm, so it drops into CI.

```
                 MySQL    MongoDB
precision        1.000      1.000
recall           1.000      1.000
type accuracy    1.000      1.000
```

Read that honestly: the detectors and the answer key were written together,
so these numbers say it handles the cases we thought of. They are a floor,
not a measurement of your data. `piiscanner audit` is how to measure it on
yours: it draws a sample of columns for a person to label, and scores the
scan against the labels.

## Known limits

- **Personal data leaked into free text is mostly missed.** A sample shows
  whether a *column* is personal data. One Aadhaar number in ten thousand chat
  messages is below the reporting floor, and lowering the floor naively
  floods the report with transaction references that happen to pass the
  checksum.
- **Name-only findings can be noisy.** `card_name` (a card *product*) is
  reported as a person's name until the stop-list learns it. They are marked
  unverified so they can be sorted out.
- **Encrypted columns are found by name only.** Ciphertext matches no
  pattern.
- **No language understanding.** Names in prose are not found.
- **MongoDB's `local` database is skipped** as a system database; an
  application storing data there needs `--dataset local`.
- MySQL row counts come from the optimiser's statistics and are approximate.

## Finding one person's data

A scan answers "which tables hold personal data". A DPDP request asks the
next question: *where is this one person's data, in every table, and how do
I delete it or give them a copy?* That needs every route from every table to
a person, including tables with no name in them at all (their orders, their
payments).

piiscanner's full version does that: it traces one person's rows across the
database and writes a dry-run deletion plan, a review file for rows shared
with somebody else, and the export owed to the person -- and, like this
package, never runs any of it. It is not open source. Write to
shubhamjjw14@gmail.com.

## Development

```bash
git clone https://github.com/shubhamjjw14/piiscanner.git && cd piiscanner
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Tests that need a database skip without one. To run them, load the synthetic
data into a local MySQL (this also creates the read-only account):

```bash
python -m piiscanner.testdata.load_mysql --admin-url mysql://root@127.0.0.1:3306/
docker run -d --name piiscanner-mongo -p 27017:27017 mongo:7 && python -m piiscanner.testdata.load_mongodb
```

[CODE_WALKTHROUGH.md](https://github.com/shubhamjjw14/piiscanner/blob/main/CODE_WALKTHROUGH.md) explains the internals, stage by
stage. `scripts/explain.py` shows why a column was or was not flagged.

## Licence

Apache 2.0. See [LICENSE](https://github.com/shubhamjjw14/piiscanner/blob/main/LICENSE) and [NOTICE](https://github.com/shubhamjjw14/piiscanner/blob/main/NOTICE).
