#!/usr/bin/env bash
#
# End-to-end pipeline against a real database: grant read access, prove the
# account cannot write, scan, then tell you how to revoke.
#
# Configuration comes from the environment, never from defaults baked into this
# file -- a committed connection string is a leak waiting to happen.  Copy
# .env.example to .env, fill it in, and source it:
#
#   cp .env.example .env && $EDITOR .env
#   set -a && . ./.env && set +a
#   scripts/scan_local.sh
#
# Required:
#   PIISCANNER_DB          database to scan
#   PIISCANNER_RO_USER     read-only account to create/reuse
#   PIISCANNER_RO_PASS     its password
#
# One of, unless PIISCANNER_SKIP_GRANT is set:
#   PIISCANNER_CONTAINER   docker container running MySQL (admin via `docker exec`)
#   PIISCANNER_ADMIN_DSN   mysql://user:pw@host:port/  for a non-docker server
#
# Optional:
#   PIISCANNER_HOST        default 127.0.0.1
#   PIISCANNER_PORT        default 3306
#   PIISCANNER_SAMPLE      values sampled per field, default 500
#   PIISCANNER_OUT_DIR     where findings land, default ./audit/raw
#   PIISCANNER_CURSOR      file holding each keyless large table's sweep position,
#                       so a repeat run advances instead of re-reading the same
#                       first rows.  Set this for anything scheduled.
#   PIISCANNER_INVENTORY   any non-empty value records every field examined, not
#                       just the findings.  Needed by `piiscanner audit sample`.
#   PIISCANNER_SKIP_GRANT  any non-empty value skips step 1 and reuses an account
#                       that already exists.  Managed servers (Cloud SQL, RDS)
#                       reject the REVOKE in step 1 even for their admin user,
#                       so grant once by hand there and set this thereafter.
#                       Step 2 still proves the account cannot write.
#
# The scanner never writes to the database it scans.  The only writes this
# script performs are CREATE USER / GRANT SELECT, via the admin account.

set -euo pipefail

cd "$(dirname "$0")/.."
PIISCANNER=".venv/bin/piiscanner"
PYTHON=".venv/bin/python"

require() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "error: $name is not set." >&2
    echo "       Copy .env.example to .env, fill it in, then:" >&2
    echo "         set -a && . ./.env && set +a" >&2
    exit 64
  fi
}

require PIISCANNER_DB
require PIISCANNER_RO_USER
require PIISCANNER_RO_PASS

if [ -z "${PIISCANNER_SKIP_GRANT:-}" ] \
   && [ -z "${PIISCANNER_CONTAINER:-}" ] && [ -z "${PIISCANNER_ADMIN_DSN:-}" ]; then
  echo "error: set either PIISCANNER_CONTAINER or PIISCANNER_ADMIN_DSN," >&2
  echo "       or PIISCANNER_SKIP_GRANT if the account already exists." >&2
  exit 64
fi

DB="$PIISCANNER_DB"
HOST="${PIISCANNER_HOST:-127.0.0.1}"
PORT="${PIISCANNER_PORT:-3306}"
SAMPLE="${PIISCANNER_SAMPLE:-500}"
OUT_DIR="${PIISCANNER_OUT_DIR:-audit/raw}"

mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/findings_$(date +%Y%m%d_%H%M%S).json"

# ---------------------------------------------------------------------------
# 1/4  grant
# ---------------------------------------------------------------------------
if [ -n "${PIISCANNER_SKIP_GRANT:-}" ]; then
  echo "==> 1/4  skipping grant (PIISCANNER_SKIP_GRANT set); step 2 still checks"
else
echo "==> 1/4  granting read-only access"

# REVOKE first so re-running cannot silently accumulate privileges.
GRANT_SQL="
  CREATE USER IF NOT EXISTS '${PIISCANNER_RO_USER}'@'%' IDENTIFIED BY '${PIISCANNER_RO_PASS}';
  ALTER  USER '${PIISCANNER_RO_USER}'@'%' IDENTIFIED BY '${PIISCANNER_RO_PASS}';
  REVOKE ALL PRIVILEGES, GRANT OPTION FROM '${PIISCANNER_RO_USER}'@'%';
  GRANT SELECT ON \`${DB}\`.* TO '${PIISCANNER_RO_USER}'@'%';
  FLUSH PRIVILEGES;"

if [ -n "${PIISCANNER_CONTAINER:-}" ]; then
  require PIISCANNER_ADMIN_USER
  require PIISCANNER_ADMIN_PASS
  docker exec -i "$PIISCANNER_CONTAINER" \
    mysql -u "$PIISCANNER_ADMIN_USER" -p"$PIISCANNER_ADMIN_PASS" \
    -e "$GRANT_SQL" 2>&1 | grep -v '^mysql: \[Warning\]' || true
else
  "$PYTHON" - "$PIISCANNER_ADMIN_DSN" "$GRANT_SQL" <<'PY'
import sys
from urllib.parse import unquote, urlparse

import pymysql

parsed = urlparse(sys.argv[1])
conn = pymysql.connect(
    host=parsed.hostname or "127.0.0.1",
    port=parsed.port or 3306,
    user=unquote(parsed.username) if parsed.username else "root",
    password=unquote(parsed.password) if parsed.password else "",
    autocommit=True,
)
with conn.cursor() as cur:
    for statement in filter(None, (s.strip() for s in sys.argv[2].split(";"))):
        cur.execute(statement)
PY
fi
fi

# ---------------------------------------------------------------------------
# 2/4  prove the account cannot write
# ---------------------------------------------------------------------------
echo "==> 2/4  verifying the account cannot write"
"$PYTHON" - "$HOST" "$PORT" "$PIISCANNER_RO_USER" "$PIISCANNER_RO_PASS" "$DB" <<'PY'
import sys

import pymysql

host, port, user, password, database = sys.argv[1:6]
conn = pymysql.connect(host=host, port=int(port), user=user,
                       password=password, database=database)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = %s", (database,))
print(f"    SELECT ok: {cur.fetchone()[0]} tables visible")

blocked = 0
for statement in ("CREATE TABLE piiscanner_probe (a INT)",
                  "DROP TABLE IF EXISTS piiscanner_probe"):
    try:
        cur.execute(statement)
        print(f"    !! WRITE ALLOWED: {statement}")
    except pymysql.err.MySQLError:
        blocked += 1
if blocked != 2:
    sys.exit("    refusing to scan: the account can write")
print("    writes blocked: confirmed read-only")
PY

# ---------------------------------------------------------------------------
# 3/4  scan
# ---------------------------------------------------------------------------
echo "==> 3/4  scanning (sample size ${SAMPLE} per field)"

# Built as an array, not a string: a DSN password can contain characters the
# shell would otherwise re-split on.
# `if`, not `test && ...`: a false test returns 1, and under `set -e` that
# would abort the script instead of just skipping the flag.
EXTRA=()
if [ -n "${PIISCANNER_CURSOR:-}" ]; then
  mkdir -p "$(dirname "$PIISCANNER_CURSOR")"
  EXTRA+=(--cursor-file "$PIISCANNER_CURSOR")
fi
if [ -n "${PIISCANNER_INVENTORY:-}" ]; then
  EXTRA+=(--inventory)
fi

$PIISCANNER scan \
  --source "mysql://${PIISCANNER_RO_USER}:${PIISCANNER_RO_PASS}@${HOST}:${PORT}/${DB}" \
  --alias "$DB" \
  --sample-size "$SAMPLE" \
  --out "$OUT" \
  "${EXTRA[@]+"${EXTRA[@]}"}" \
  --all

# ---------------------------------------------------------------------------
# 4/4  done
# ---------------------------------------------------------------------------
echo
echo "==> 4/4  done"
echo "    findings: ${OUT}   (under audit/, which is gitignored)"
echo
echo "    re-read it any time:   ${PIISCANNER} report ${OUT} --all"
if [ -n "${PIISCANNER_CONTAINER:-}" ]; then
  echo "    revoke access:         docker exec ${PIISCANNER_CONTAINER} mysql -u <admin> -p \\"
  echo "                             -e \"DROP USER '${PIISCANNER_RO_USER}'@'%';\""
else
  echo "    revoke access:         DROP USER '${PIISCANNER_RO_USER}'@'%';"
fi
