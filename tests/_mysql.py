"""A local MySQL the tests can create scratch databases in.

Tests that need one skip when it is not there.  Point
``PIISCANNER_TEST_ADMIN_DSN`` at a server you may write to; never at one
holding real data.
"""

from __future__ import annotations

import os
from urllib.parse import unquote, urlparse

ADMIN_DSN = os.environ.get("PIISCANNER_TEST_ADMIN_DSN", "mysql://root@127.0.0.1:3306/")


def _admin():
    import pymysql

    parsed = urlparse(ADMIN_DSN)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1", port=parsed.port or 3306,
        user=unquote(parsed.username or "root"),
        password=unquote(parsed.password or ""),
        autocommit=True, connect_timeout=3)


def _available() -> bool:
    try:
        _admin().close()
    except Exception:
        return False
    return True
