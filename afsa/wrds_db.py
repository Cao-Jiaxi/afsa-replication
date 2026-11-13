"""Connection to the WRDS PostgreSQL server.

The official ``wrds`` package pins pandas < 2.3, which would break this project
(pandas 3), so this module talks to the same server through psycopg2 and SQLAlchemy.

Credentials live only in the standard Postgres password file of the user's profile
(``%APPDATA%\\postgresql\\pgpass.conf`` on Windows, ``~/.pgpass`` elsewhere), never in
the repo. Each new connection may trigger a Duo push that must be approved on the phone.

Setup, once, in your own terminal (it asks for the password without echoing it):

    .venv\\Scripts\\python.exe -m afsa.wrds_db setup

Then:

    .venv\\Scripts\\python.exe -m afsa.wrds_db test     # check the login and library access

    from afsa.wrds_db import query
    df = query("select permno, dlycaldt, dlyret from crsp.dsf_v2 where dlycaldt = %(d)s", {"d": "2025-12-31"})

Anything downloaded is licensed WRDS data: save it under Data\\ (git-ignored), never in the repo.
"""

from __future__ import annotations

import getpass
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

HOST = "wrds-pgdata.wharton.upenn.edu"
PORT = 9737
DBNAME = "wrds"


def pgpass_path() -> Path:
    if os.name == "nt":
        return Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def _escape(field: str) -> str:
    return field.replace("\\", "\\\\").replace(":", "\\:")


def _wrds_user() -> str | None:
    """WRDS username: $WRDS_USER, else the first WRDS line of the password file."""
    if os.environ.get("WRDS_USER"):
        return os.environ["WRDS_USER"]
    path = pgpass_path()
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith(f"{HOST}:{PORT}:"):
            return line.split(":")[3]
    return None


def setup() -> None:
    """Ask for the WRDS login and store it in the Postgres password file."""
    user = input("WRDS username: ").strip()
    password = getpass.getpass("WRDS password (not shown): ")
    path = pgpass_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{HOST}:{PORT}:{DBNAME}:"
    lines = [l for l in path.read_text().splitlines() if not l.startswith(prefix)] if path.exists() else []
    lines.append(prefix + _escape(user) + ":" + _escape(password))
    path.write_text("\n".join(lines) + "\n")
    if os.name != "nt":
        path.chmod(0o600)
    print(f"Saved to {path}")


@lru_cache(maxsize=1)
def engine():
    """SQLAlchemy engine for WRDS (one per process; approve the Duo push on first use)."""
    from sqlalchemy import create_engine

    user = _wrds_user()
    if user is None:
        raise RuntimeError("No WRDS login found: run  python -m afsa.wrds_db setup  first.")
    return create_engine(
        f"postgresql+psycopg2://{user}@{HOST}:{PORT}/{DBNAME}",
        connect_args={"sslmode": "require", "application_name": "afsa-replication"},
        pool_pre_ping=True,
    )


def query(sql: str, params: dict | None = None, **read_sql_kwargs) -> pd.DataFrame:
    """Run a SELECT on WRDS and return a DataFrame. Use %(name)s placeholders for params."""
    with engine().connect() as conn:
        return pd.read_sql_query(sql, conn, params=params, **read_sql_kwargs)


def copy_query(sql: str, params: dict | None = None, strings: tuple[str, ...] = ()) -> pd.DataFrame:
    """Run a large SELECT through COPY ... TO STDOUT (CSV) and parse it with pyarrow: several times faster than
    `query` and exact for numerics. Columns named in `strings` stay text (identifiers with leading zeros, flags
    that could be read as numbers); dates come back as datetime64."""
    import io

    import pyarrow as pa
    import pyarrow.csv as pacsv

    raw = engine().raw_connection()
    try:
        cur = raw.cursor()
        stmt = cur.mogrify(sql, params).decode() if params else sql
        buf = io.BytesIO()
        cur.copy_expert(f"COPY ({stmt}) TO STDOUT WITH (FORMAT CSV, HEADER)", buf)
        cur.close()
    finally:
        raw.close()
    buf.seek(0)
    table = pacsv.read_csv(buf, convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in strings}))
    return table.to_pandas()


def test() -> None:
    print(f"Connecting to {HOST} as {_wrds_user()} (approve the Duo push if one arrives) ...")
    print(query("select current_user as user, now() as server_time").to_string(index=False))
    libs = query(
        "select table_schema as library, count(*) as tables from information_schema.tables "
        "where table_schema in ('crsp', 'crsp_a_stock', 'comp', 'comp_na_daily_all', 'crsp_a_ccm', 'ff') "
        "group by 1 order by 1"
    )
    print("\nLibraries visible to this account:")
    print(libs.to_string(index=False) if len(libs) else "  (none of crsp / comp / ff: check the subscription)")
    last = query("select (select max(date) from crsp.dsf) as legacy, (select max(dlycaldt) from crsp.dsf_v2) as ciz")
    print(f"\nLast CRSP day: legacy crsp.dsf {last.legacy[0]}, CIZ crsp.dsf_v2 {last.ciz[0]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"setup": setup, "test": test}.get(cmd, lambda: sys.exit(f"unknown command {cmd!r}: use setup or test"))()
