"""Raw WRDS downloads for the 39 characteristics and rf, their loaders, and Pelger's panel as a parquet file.

    .venv\\Scripts\\python.exe -m afsa.wrds_data download          # every table below that is not on disk yet
    .venv\\Scripts\\python.exe -m afsa.wrds_data download --force  # download again
    .venv\\Scripts\\python.exe -m afsa.wrds_data pelger            # Data\\PelgerOldData\\*.csv -> Data\\WRDS\\pelger\\*.parquet
    .venv\\Scripts\\python.exe -m afsa.wrds_data status            # what is on disk

Files in Data\\WRDS\\raw\\ (as downloaded, never edited; each has a .json manifest with the query and row count):

    crsp_dsf_v2.parquet        crsp.dsf_v2 (daily CIZ), from 1984-01-01, for every PERMNO that is ever in a top-500
                               list (afsa.wrds_universe, broad or common). The full history of each, not only its
                               universe months: momentum, beta and volatility need the days before it enters.
    crsp_index_daily.parquet   crsp.inddlyseriesdata: CRSP NYSE/NYSE MKT/Nasdaq/Arca value-weighted (1000200) and
                               equal-weighted (1000201) market indices
    ff_factors_daily.parquet   ff.factors_daily: mktrf, smb, hml, umd, rf (rf has only 4 decimals: 0 in many years)
    ff_factors_monthly.parquet ff.factors_monthly: the same by month; its rf keeps 4 decimals per MONTH
    ccm_lnkhist.parquet        crsp_a_ccm.ccmxpf_lnkhist: PERMNO-GVKEY links with their date ranges
    comp_funda.parquet         comp.funda (annual), INDL/STD/D/C, every currency, fiscal year-ends from 1978
    comp_company.parquet       comp.company: one row per GVKEY (sic, fic, loc, ...)
    (crsp_msf_v2_*.parquet and crsp_stkdelists.parquet come from afsa.wrds_universe)

Units as in WRDS: CRSP returns in decimals, prices in USD, `shrout` in thousands of shares, `dlycap`/`mthcap` in
thousands of USD, volume in shares; Compustat in millions of the reporting currency (`curcd`).
Everything here derives from CRSP/Compustat: it stays under Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from afsa.config import ROOT, WRDS_DIR

RAW_DIR = WRDS_DIR / "raw"
PELGER_SRC = ROOT / "Data" / "PelgerOldData"
PELGER_DIR = WRDS_DIR / "pelger"
DAILY_START = "1984-01-01"      # 5 years of daily returns (Beta's correlation window) before the first formation date, 1989-10-31
DAILY_END = "2025-12-31"
FUNDA_START = "1978-01-01"

DSF_COLUMNS = [
    "permno", "dlycaldt", "dlyret", "dlyretx", "dlyreti", "dlyretmissflg", "dlyretdurflg", "dlydelflg", "dlydistretflg",
    "dlyprc", "dlyprcflg", "dlyprevprc", "dlyprevdt", "dlycap", "dlycapflg", "dlyvol", "dlyprcvol", "dlynumtrd",
    "dlyopen", "dlyclose", "dlylow", "dlyhigh", "dlybid", "dlyask", "dlyfacprc", "dlycumfacpr", "dlycumfacshr",
    "dlyorddivamt", "dlynonorddivamt", "shrout", "primaryexch", "tradingstatusflg", "conditionaltype",
    "securitytype", "securitysubtype", "sharetype", "issuertype", "usincflg",
]
DSF_STRINGS = ("dlyretmissflg", "dlyretdurflg", "dlydelflg", "dlydistretflg", "dlyprcflg", "dlycapflg", "primaryexch",
               "tradingstatusflg", "conditionaltype", "securitytype", "securitysubtype", "sharetype", "issuertype", "usincflg")

# Compustat annual: identifiers, then every item a characteristic of Chen-Pelger-Zhu / Freyberger-Neuhierl-Weber
# could use (a superset: columns are cheap, a second download is not).
FUNDA_IDS = ["gvkey", "datadate", "fyear", "fyr", "indfmt", "datafmt", "popsrc", "consol", "curcd", "curncd", "costat",
             "conm", "tic", "cusip", "cik", "exchg", "fic", "sich", "naicsh", "apdedate", "fdate", "pdate", "ajex", "ajp"]
FUNDA_ITEMS = [
    # balance sheet
    "at", "act", "che", "ch", "ivst", "rect", "invt", "aco", "ppent", "ppegt", "ivaeq", "ivao", "intan", "gdwl", "ao",
    "lt", "lct", "dlc", "ap", "txp", "lco", "dltt", "txditc", "txdb", "itcb", "lo", "mib", "mibt", "mibn",
    "seq", "ceq", "pstk", "pstkrv", "pstkl", "pstkn", "pstkr", "re", "tstk", "teq", "icapt", "wcap", "dcvt", "dd1",
    # income statement
    "sale", "revt", "cogs", "xsga", "xrd", "xad", "xlr", "xpp", "xrent", "gp", "oibdp", "dp", "am", "oiadp", "ebit", "ebitda",
    "xint", "tie", "nopi", "spi", "pi", "txt", "txdi", "txc", "ib", "ibadj", "ibcom", "ni", "niadj", "xido", "dvp", "dvc",
    "dvt", "epspx", "epsfx", "emp",
    # cash flow
    "oancf", "ivncf", "fincf", "capx", "wcapch", "ibc", "dpc", "xidoc", "txdc", "esubc", "sppiv", "fopo", "fopt",
    "recch", "invch", "apalch", "txach", "aoloch", "sstk", "prstkc", "dv", "dltis", "dltr", "dlcch", "chech", "sppe",
    "aqc", "ivch", "siv", "scf",
    # market
    "csho", "cshpri", "prcc_f", "prcc_c", "mkvalt", "cshtr_f", "adjex_f",
]


# ---------------------------------------------------------------------------------------------------- helpers
def _dates(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def _save(df: pd.DataFrame, name: str, table: str, sql: str, t0: float, note: str = "") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    manifest = {"table": table, "rows": int(len(df)), "columns": list(df.columns), "query": " ".join(sql.split()),
                "downloaded": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0)}
    if note:
        manifest["note"] = note
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"  saved {path.name}: {len(df):,} rows, {len(df.columns)} columns, {path.stat().st_size / 2**20:.0f} MB, "
          f"{time.time() - t0:.0f} s", flush=True)


def universe_permnos() -> list[int]:
    """Every PERMNO that is in a broad or common top-500 list in some month (1989-11 on)."""
    from afsa.wrds_universe import UNIVERSE_DIR

    p = set()
    for kind in ("broad", "common"):
        p |= set(pd.read_parquet(UNIVERSE_DIR / f"{kind}.parquet", columns=["permno"]).permno.tolist())
    return sorted(int(x) for x in p)


# ---------------------------------------------------------------------------------------------------- downloads
def download_dsf() -> None:
    from afsa.wrds_db import copy_query

    permnos = universe_permnos()
    sql = (f"select {', '.join(DSF_COLUMNS)} from crsp.dsf_v2 where permno = any(%(p)s) "
           "and dlycaldt between %(lo)s and %(hi)s order by permno, dlycaldt")
    t0, parts = time.time(), []
    for k in range(0, len(permnos), 400):                                    # blocks of 400 securities
        df = copy_query(sql, {"p": permnos[k:k + 400], "lo": DAILY_START, "hi": DAILY_END}, strings=DSF_STRINGS)
        parts.append(df)
        print(f"    securities {k + 1}-{k + len(permnos[k:k + 400])}: {len(df):,} rows ({time.time() - t0:.0f} s)", flush=True)
    df = _dates(pd.concat(parts, ignore_index=True), ["dlycaldt", "dlyprevdt"])
    df["permno"] = df["permno"].astype("int32")
    for c in DSF_STRINGS:
        df[c] = df[c].astype("category")
    _save(df, "crsp_dsf_v2", "crsp.dsf_v2", sql, t0, f"{len(permnos)} PERMNOs ever in a broad or common top-500 list")


def download_index() -> None:
    from afsa.wrds_db import copy_query

    sql = ("select indno, dlycaldt, dlytotret, dlyprcret, dlyincret, dlytotind, dlyusdcnt, dlyusdval, dlytotcnt, dlytotval "
           f"from crsp.inddlyseriesdata where indno in (1000200, 1000201) and dlycaldt >= '{DAILY_START}' order by indno, dlycaldt")
    t0 = time.time()
    _save(_dates(copy_query(sql), ["dlycaldt"]), "crsp_index_daily", "crsp.inddlyseriesdata", sql, t0,
          "1000200 = value-weighted, 1000201 = equal-weighted NYSE/NYSE MKT/Nasdaq/Arca market index")


def download_ff() -> None:
    from afsa.wrds_db import copy_query

    for name, table in (("ff_factors_daily", "ff.factors_daily"), ("ff_factors_monthly", "ff.factors_monthly")):
        sql = f"select * from {table} where date >= '1980-01-01' order by date"
        t0 = time.time()
        _save(_dates(copy_query(sql), ["date", "dateff"]), name, table, sql, t0)


def download_ccm() -> None:
    from afsa.wrds_db import copy_query

    sql = "select * from crsp_a_ccm.ccmxpf_lnkhist order by gvkey, linkdt"
    t0 = time.time()
    df = _dates(copy_query(sql, strings=("gvkey", "liid", "linkprim", "linktype")), ["linkdt", "linkenddt"])
    _save(df, "ccm_lnkhist", "crsp_a_ccm.ccmxpf_lnkhist", sql, t0, "linkenddt is empty for links that are still valid")


def download_funda() -> None:
    from afsa.wrds_db import copy_query, query

    have = set(query("select column_name from information_schema.columns where table_schema = 'comp' "
                     "and table_name = 'funda'").column_name)
    cols = [c for c in FUNDA_IDS + FUNDA_ITEMS if c in have]
    missing = [c for c in FUNDA_IDS + FUNDA_ITEMS if c not in have]
    if missing:
        print(f"    not in comp.funda, skipped: {missing}")
    sql = (f"select {', '.join(cols)} from comp.funda where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' "
           f"and consol = 'C' and datadate >= '{FUNDA_START}' order by gvkey, datadate")
    t0 = time.time()
    strings = ("gvkey", "indfmt", "datafmt", "popsrc", "consol", "curcd", "curncd", "costat", "conm", "tic", "cusip", "cik", "fic", "scf")
    df = _dates(copy_query(sql, strings=strings), ["datadate", "apdedate", "fdate", "pdate"])
    _save(df, "comp_funda", "comp.funda", sql, t0, "industrial format, standardized, domestic population, consolidated; USD and CAD reporters")

    sql = "select gvkey, conm, sic, naics, fic, loc, stko, idbflag, ipodate, dldte, dlrsn, fyrc, costat, priusa, prican from comp.company order by gvkey"
    t0 = time.time()
    df = _dates(copy_query(sql, strings=("gvkey", "conm", "sic", "naics", "fic", "loc", "idbflag", "dlrsn", "costat", "priusa", "prican")),
                ["ipodate", "dldte"])
    _save(df, "comp_company", "comp.company", sql, t0)


DOWNLOADS = {
    "crsp_dsf_v2": download_dsf, "crsp_index_daily": download_index, "ff_factors_daily": download_ff,
    "ccm_lnkhist": download_ccm, "comp_funda": download_funda,
}


def download(force: bool = False, only: list[str] | None = None) -> None:
    for name, fn in DOWNLOADS.items():
        if only and name not in only:
            continue
        if (RAW_DIR / f"{name}.parquet").exists() and not force:
            print(f"  {name}: already on disk (use --force to download again)")
            continue
        print(f"  {name} ...", flush=True)
        fn()


# ---------------------------------------------------------------------------------------------------- loaders
def load(name: str, columns: list[str] | None = None, **kw) -> pd.DataFrame:
    """A raw file by its name without extension, e.g. load("crsp_dsf_v2", columns=["permno", "dlycaldt", "dlyret"])."""
    return pd.read_parquet(RAW_DIR / f"{name}.parquet", columns=columns, **kw)


def status() -> None:
    for path in sorted(RAW_DIR.glob("*.json")):
        m = json.loads(path.read_text())
        print(f"  {path.stem:<34} {m.get('rows', 0):>12,} rows  {m.get('downloaded', '')}  {m.get('table', '')}")
    for path in sorted(PELGER_DIR.glob("*.parquet")) if PELGER_DIR.exists() else []:
        print(f"  pelger/{path.name}  {path.stat().st_size / 2**20:.0f} MB")


# ---------------------------------------------------------------------------------------------------- Pelger's panel
def pelger(first_year: int = 1984) -> None:
    """The Chen-Pelger-Zhu characteristic panel (monthly, raw values, 46 characteristics; 1963-2016) from `first_year`
    on, as a parquet file. The 1 GB CSV is read in chunks and never held whole. Used only to check our construction."""
    PELGER_DIR.mkdir(parents=True, exist_ok=True)
    t0, parts, n_all = time.time(), [], 0
    for chunk in pd.read_csv(PELGER_SRC / "characteristics-replication.csv", chunksize=200_000):
        n_all += len(chunk)
        parts.append(chunk[chunk.yy >= first_year])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["permno"] = df["permno"].astype("int32")
    df = df.drop(columns=["yy", "mm"])
    out = PELGER_DIR / "characteristics.parquet"
    df.to_parquet(out, index=False)
    print(f"  {out}: {len(df):,} of {n_all:,} stock-months ({first_year} on), {df.permno.nunique():,} PERMNOs, "
          f"{df.date.min().date()} to {df.date.max().date()}, {len(df.columns) - 2} columns, {time.time() - t0:.0f} s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["download", "pelger", "status"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", nargs="*", help="names of the downloads to run: " + ", ".join(DOWNLOADS))
    a = p.parse_args()
    if a.command == "download":
        download(a.force, a.only)
    elif a.command == "pelger":
        pelger()
    status()


if __name__ == "__main__":
    main()
