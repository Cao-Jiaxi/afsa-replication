"""Monthly top-500 universe built directly from WRDS (CRSP CIZ monthly file), and its check against Ander's.

Rule (the paper's section 4.1, "top 500 by market cap of the previous month", with the project's "broad" reading):
the universe of holding month m is the 500 securities with the largest CRSP market cap (`mthcap`) at the end of
month m-1, among all equities (`securitytype` = EQTY, so funds, ETFs and derivatives are out) whose primary
exchange is NYSE, AMEX or NASDAQ (`primaryexch` N, A, Q), that trade regular-way and are still listed on the last
trading day of m-1 (their last trading day in `crsp.stkdelists` is not earlier). Each PERMNO is a separate security;
ADRs, foreign shares,
REITs and units (MLPs) are included. The CIZ classification fields are historical (as of each month), so the
rule uses no later information. The "common" universe keeps US common stocks only (the legacy share codes 10/11).

    .venv\\Scripts\\python.exe -m afsa.wrds_universe download   # crsp.msf_v2 (1985-12 to 2025-12), crsp.stkdelists -> Data\\WRDS\\raw\\
    .venv\\Scripts\\python.exe -m afsa.wrds_universe build      # -> Data\\WRDS\\universe\\{broad,common}.parquet
    .venv\\Scripts\\python.exe -m afsa.wrds_universe compare    # against Ander's universe files

Everything written here derives from CRSP: it stays under Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from afsa.config import SOURCE_DIR, WRDS_DIR

RAW_DIR = WRDS_DIR / "raw"
UNIVERSE_DIR = WRDS_DIR / "universe"
MSF_FILE = RAW_DIR / "crsp_msf_v2_198512_202512.parquet"
DELIST_FILE = RAW_DIR / "crsp_stkdelists.parquet"
FIRST_MONTH, LAST_MONTH = 198512, 202512        # download range (yyyymm)
# Holding months. 1989-11 and 1989-12 are the lookback before the first window: loss days from 1990-01-02 need
# the universe on their 31 previous trading days, i.e. from 1989-11-15 (NOTES.md section 2).
HOLD_START, HOLD_END = pd.Period("1989-11", "M"), pd.Period("2025-12", "M")
N_STOCKS = 500
EXCHANGES = ("N", "A", "Q")                     # NYSE, NYSE American (AMEX), NASDAQ

MSF_COLUMNS = [
    "permno", "permco", "hdrcusip", "cusip", "ticker", "issuernm", "siccd", "yyyymm", "mthcaldt",
    "mthprc", "mthprcflg", "mthprcdt", "mthcap", "mthprevcap", "mthret", "mthretx", "mthdelflg", "mthvol", "shrout",
    "securitytype", "securitysubtype", "sharetype", "issuertype", "usincflg", "primaryexch", "exchangetier",
    "tradingstatusflg", "conditionaltype",
]


# ---------------------------------------------------------------------------------------------------- download
def download() -> pd.DataFrame:
    """All securities of crsp.msf_v2 (every type and exchange, so filters can be varied locally), by 5-year blocks."""
    from afsa.wrds_db import query

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sql = (f"select {', '.join(MSF_COLUMNS)} from crsp.msf_v2 "
           "where yyyymm between %(lo)s and %(hi)s order by yyyymm, permno")
    blocks, lo = [], FIRST_MONTH
    while lo <= LAST_MONTH:
        hi = min((lo // 100 + 4) * 100 + 12, LAST_MONTH)
        t0 = time.time()
        df = query(sql, {"lo": lo, "hi": hi}, parse_dates=["mthcaldt", "mthprcdt"])
        print(f"  {lo}-{hi}: {len(df):,} rows in {time.time() - t0:.0f} s")
        blocks.append(df)
        lo = (hi // 100 + 1) * 100 + 1
    msf = pd.concat(blocks, ignore_index=True)
    for c in ("permno", "permco", "yyyymm", "siccd", "shrout"):
        msf[c] = msf[c].astype("Int64")
    msf.to_parquet(MSF_FILE, index=False)
    manifest = {
        "table": "crsp.msf_v2", "columns": MSF_COLUMNS, "yyyymm": [FIRST_MONTH, LAST_MONTH], "rows": len(msf),
        "permnos": int(msf.permno.nunique()), "downloaded": datetime.now().isoformat(timespec="seconds"),
    }
    MSF_FILE.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {MSF_FILE} ({len(msf):,} rows, {manifest['permnos']:,} PERMNOs)")
    return msf


def download_delists() -> pd.DataFrame:
    """crsp.stkdelists in full (one row per delisting event): the date each security left its exchange."""
    from afsa.wrds_db import query

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    d = query("select * from crsp.stkdelists order by permno, delistingdt",
              parse_dates=["delistingdt", "delnextdt", "delamtdt", "deldlydt"])
    d.to_parquet(DELIST_FILE, index=False)
    DELIST_FILE.with_suffix(".json").write_text(json.dumps(
        {"table": "crsp.stkdelists", "rows": len(d), "downloaded": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"Saved {DELIST_FILE} ({len(d):,} rows, {d.permno.nunique():,} PERMNOs)")
    return d


# ---------------------------------------------------------------------------------------------------- build
def security_type(df: pd.DataFrame) -> pd.Series:
    """common, ADR, foreign (ordinary share of a non-US issuer), REIT, MLP (units) or other; as of each month."""
    t = np.select(
        [df.sharetype.eq("AD"), df.issuertype.eq("REIT"), df.sharetype.eq("UG"), df.usincflg.eq("N"), is_common(df)],
        ["ADR", "REIT", "MLP", "foreign", "common"], default="other")
    return pd.Series(t, index=df.index)


def is_common(df: pd.DataFrame) -> pd.Series:
    """US common stock: the CIZ counterpart of legacy share codes 10/11."""
    return (df.securitytype.eq("EQTY") & df.securitysubtype.eq("COM") & df.sharetype.eq("NS")
            & df.issuertype.isin(["CORP", "ACOR"]) & df.usincflg.eq("Y"))


def build(msf: pd.DataFrame, delists: pd.DataFrame, kind: str = "broad") -> pd.DataFrame:
    """Top 500 of each holding month, formed on the month-end market cap of the previous month."""
    deldt = msf.permno.map(delists.set_index("permno").delistingdt)
    elig = (msf.securitytype.eq("EQTY") & msf.primaryexch.isin(EXCHANGES) & msf.mthcap.gt(0)
            # listed on the formation day: a security whose last trading day (`delistingdt`) came earlier in m-1 keeps
            # a month-end row with the cap of its last trade, but cannot be held in month m. One that delists on the
            # formation day itself stays in: it is held at that close and earns its delisting return (e.g. a merger
            # payout), which CRSP books on the next trading day, the first day of m. A stale month-end price alone
            # (no trade on the last day) does not exclude a security that is still listed.
            & ~(deldt < msf.mthcaldt)
            & msf.conditionaltype.eq("RW"))         # regular-way trading, not when-issued
    if kind == "common":
        elig &= is_common(msf)
    elif kind != "broad":
        raise ValueError(kind)
    f = msf.loc[elig].copy()
    f["month"] = pd.PeriodIndex(f.mthcaldt, freq="M") + 1                       # holding month = formation month + 1
    f = f[(f.month >= HOLD_START) & (f.month <= HOLD_END)]
    assert not f.duplicated(["month", "permno"]).any(), "several rows per PERMNO and month"
    f["me_company"] = f.groupby(["month", "permco"]).mthcap.transform("sum")    # all eligible classes of the firm
    f = f.sort_values(["month", "mthcap", "permno"], ascending=[True, False, True])
    f["rank"] = f.groupby("month").cumcount() + 1
    u = f[f["rank"] <= N_STOCKS].copy()
    u["type"] = security_type(u)
    u = u.rename(columns={"mthcaldt": "formation_date", "mthcap": "me"})
    cols = ["month", "formation_date", "permno", "permco", "ticker", "issuernm", "cusip", "siccd", "primaryexch",
            "securitysubtype", "sharetype", "issuertype", "usincflg", "type", "me", "me_company", "rank"]
    u = u[cols].reset_index(drop=True)
    sizes = u.groupby("month").size()
    assert len(sizes) == (HOLD_END - HOLD_START).n + 1 and (sizes == N_STOCKS).all(), "a month without 500 stocks"
    return u


def build_all() -> dict[str, pd.DataFrame]:
    msf, delists = pd.read_parquet(MSF_FILE), pd.read_parquet(DELIST_FILE)
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ("broad", "common"):
        u = build(msf, delists, kind)
        u.to_parquet(UNIVERSE_DIR / f"{kind}.parquet", index=False)
        print(f"{kind}: {len(u):,} stock-months, {u.permno.nunique():,} PERMNOs, "
              f"{u.month.min()} to {u.month.max()}; types: {u.type.value_counts().to_dict()}")
        out[kind] = u
    return out


# ---------------------------------------------------------------------------------------------------- compare
def compare(kind: str = "broad") -> dict:
    """Month-by-month overlap with Ander's universe (his months only: 1990-01 on), and what explains the differences."""
    ours = pd.read_parquet(UNIVERSE_DIR / f"{kind}.parquet")
    ander = pd.read_parquet(SOURCE_DIR / ("universe.parquet" if kind == "broad" else "universe_comun.parquet"))
    ours = ours[ours.month >= ander.month.min()]
    m = ours.merge(ander[["month", "permno", "me", "rank", "tipo"]], on=["month", "permno"], how="outer",
                   suffixes=("", "_ander"), indicator=True)
    both = m[m._merge == "both"]
    by_month = m.groupby("month")._merge.apply(lambda s: (s == "both").sum())
    res = {
        "kind": kind,
        "stock_months_both": int(len(both)),
        "only_ours": int((m._merge == "left_only").sum()),
        "only_ander": int((m._merge == "right_only").sum()),
        "months_identical": int((by_month == N_STOCKS).sum()),
        "months": int(len(by_month)),
        "min_overlap_in_a_month": int(by_month.min()),
        "me_equal_share": float(np.isclose(both.me, both.me_ander, rtol=1e-6).mean()),
        "rank_equal_share": float((both["rank"] == both.rank_ander).mean()),
    }
    print(json.dumps(res, indent=2))
    if kind == "broad":
        print("\nOur type vs Ander's tipo (stock-months in both):")
        print(pd.crosstab(both.type, both.tipo).to_string())
    diff = m[m._merge != "both"].sort_values(["month", "_merge"])
    if len(diff):
        cols = ["month", "_merge", "permno", "ticker", "type", "me", "rank", "me_ander", "rank_ander", "tipo"]
        print(f"\nStock-months in only one universe (first 40 of {len(diff)}):")
        print(diff[cols].head(40).to_string(index=False))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["download", "build", "compare"])
    a = p.parse_args()
    if a.command == "download":
        download()
        download_delists()
    elif a.command == "build":
        build_all()
    else:
        for kind in ("broad", "common"):
            compare(kind)


if __name__ == "__main__":
    main()
