"""The whole table: the 39 characteristics and rf of every security that is ever on a top-500 list, as raw values.

    python -m afsa.wrds_table                # -> Data/WRDS/table/chars_daily_fill-none.parquet and meta.json
    python -m afsa.wrds_table --check-only   # build and check in memory, write nothing

Rows. One per security (PERMNO) and trading day from 1989-11-01 to 2025-12-31, on every day the security has a row in
the CRSP daily file, for every PERMNO that is on a broad or a common top-500 list in some month of 1989-11 to 2025-12
(afsa.wrds_universe): its days off the lists are rows too. Sorted by date, then PERMNO; one row group per year.

Columns.
    date                 trading day t
    permno, permco       CRSP security and company. The lists rank securities, so a company can have two share
                         classes in the same month's list. A PERMNO has one PERMCO throughout (checked).
    in_broad, in_common  the security is in that universe on day t, as the training arrays define it: on the list
                         of t's month (the 500 largest at the end of the month before) and a row in the daily file
                         that day. The rows with in_broad are the stock-days of wrds_broad_fill-none (checked).
    r2_1 ... Vol         the 39 characteristics in the order of afsa.config.CHARACTERISTICS (the paper's Table 1
                         names), raw, in the units of their definitions (analysis/characteristics/market.md,
                         accounting.md and training_data.md)
    rf                   the risk-free rate of day t-1, a daily rate in decimals (afsa.build_wrds.spread_rf)

Timing. Row t is the paper's X_{t-1}, as in the training arrays: everything in it was known at the close of day t-1.
The 4 daily characteristics come from days t-5 .. t-1 (Ret_D1 is the return of t-1), the 35 monthly ones are those of
t's holding month (computed from data up to the last trading day of the month before), and rf is the rate of day t-1.
The return of day t is not in the table.

Missing values: the user's rule (fill none, CLAUDE.md section 4). A characteristic whose raw inputs are not available
is empty: null in the file, NaN in pandas. Nothing is carried forward and nothing is imputed. The accounting values
are the production build of afsa.chars_accounting (15 secondary items count as 0 where the panel shows it; a missing
XSGA, DP, COGS, CEQ or LCT leaves the characteristic empty), and a daily characteristic needs all five days. The
second half of the rule, the ranking, depends on the universe and is not applied here: the training arrays rank a
day's members that have a value into [-0.5, 0.5] and give an empty value rank 0.

The values come from the code that builds the training arrays (afsa.build_wrds.load_sources). Before anything is
written, every cell is checked by another route (meta.json lists the results):
    - the 35 monthly values against market.parquet and accounting.parquet by (PERMNO, month), empty cells included;
    - the 4 daily values recomputed from the rows of the raw daily file;
    - rf recomputed from the Fama-French monthly file and the trading calendar;
    - in_broad and in_common against the stock-days of the two arrays on disk, and the arrays' ranks, medians and rf
      input recomputed from the table's member rows with the missing-value rule;
    - after writing, every column read back from the file.
Everything here derives from CRSP/Compustat: it stays under Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import build_wrds as bw
from .build_data import Checks
from .build_window import DAILY, MONTHLY, ranks_and_medians
from .config import CHAR_GROUPS, CHARACTERISTICS, N_CHAR, N_SLOTS, TRAIN_DATA_DIR, WRDS_DIR, WRDS_UNIVERSES, data_folder

FILL = "none"
TABLE_DIR = WRDS_DIR / "table"
TABLE_NAME = f"chars_daily_fill-{FILL}.parquet"
FLAG = {"wrds_broad": "in_broad", "wrds_common": "in_common"}
COLUMNS = ("date", "permno", "permco", *FLAG.values(), *CHARACTERISTICS, "rf")
SCHEMA = pa.schema([("date", pa.timestamp("ms")), ("permno", pa.int32()), ("permco", pa.int32()),
                    *((f, pa.bool_()) for f in FLAG.values()), *((c, pa.float64()) for c in CHARACTERISTICS),
                    ("rf", pa.float64())])
TOL_WEEK = 2e-12        # Ret_W1 and STD_W1 are rounded to 12 decimals: another order of operations can flip the last one


# ---------------------------------------------------------------------------------------------------- the rows
def permco_of(permnos: np.ndarray) -> np.ndarray:
    """The PERMCO of each PERMNO, from the top-500 lists of both universes. Raises if a PERMNO has two PERMCOs or is
    on no list."""
    lists = pd.concat([pd.read_parquet(bw.UNIVERSE_DIR / f"{u.removeprefix('wrds_')}.parquet", columns=["permno", "permco"])
                       for u in WRDS_UNIVERSES]).astype("int64").drop_duplicates()
    if lists["permno"].duplicated().any():
        raise ValueError("a PERMNO has two PERMCOs in the top-500 lists")
    out = lists.set_index("permno")["permco"].reindex(np.asarray(permnos, dtype=np.int64))
    if out.isna().any():
        raise ValueError("a PERMNO of the daily file is on no top-500 list")
    return out.to_numpy(dtype=np.int32)


def members(src: dict, days: np.ndarray) -> dict[str, np.ndarray]:
    """member[universe] [T, S]: in the universe on each array day, as afsa.build_wrds.assemble defines it."""
    month = src["month_of_day"][days]
    return {u: bw.universe_lists(u, src)[0][month] & src["has_row"][days] for u in WRDS_UNIVERSES}


def rows(src: dict, days: np.ndarray, member: dict, permco: np.ndarray, ts: np.ndarray) -> dict[str, np.ndarray]:
    """The table rows of the array days `ts` (positions in `days`) as columns, by day, then PERMNO. Two extra
    columns for the checks are not written: `_t` (array day) and `_cal` (calendar row)."""
    cal = days[ts]
    d, c = np.nonzero(src["has_row"][cal])                                      # permnos are sorted: by day, then PERMNO
    i = cal[d]
    out = {"date": src["dates"][i], "permno": src["permnos"][c].astype(np.int32), "permco": permco[c]}
    out.update({FLAG[u]: member[u][ts[d], c] for u in WRDS_UNIVERSES})
    month = src["month_of_day"][i]
    for name in CHARACTERISTICS:
        out[name] = src["daily"][name][i, c] if name in DAILY else src["monthly"][name][month, c]
    out["rf"] = src["rf"][i - 1]                                                # X's rf: the rate of the day before
    out["_t"], out["_cal"] = ts[d], i
    return out


def to_arrow(tab: dict[str, np.ndarray]) -> pa.Table:
    """The written columns; NaN becomes null."""
    cols = [pa.array(tab["date"].astype("datetime64[ms]"), pa.timestamp("ms")), pa.array(tab["permno"], pa.int32()),
            pa.array(tab["permco"], pa.int32()), *(pa.array(tab[f], pa.bool_()) for f in FLAG.values()),
            *(pa.array(tab[c], pa.float64(), from_pandas=True) for c in (*CHARACTERISTICS, "rf"))]
    return pa.Table.from_arrays(cols, schema=SCHEMA)


_MIX = np.uint64(0x9E3779B97F4A7C15)


def fingerprint(x: np.ndarray, first_row: int) -> int:
    """An order-sensitive 64-bit fingerprint of a column chunk whose first row is row `first_row` of the table. NaN
    (null in the file) is one value and -0.0 counts as 0.0. The fingerprints of the chunks add up modulo 2**64."""
    if x.dtype.kind == "f":
        x = np.asarray(x, dtype=np.float64) + 0.0
        bits = np.where(np.isnan(x), np.uint64(0x7FF8000000000000), x.view(np.uint64))
    elif x.dtype.kind == "M":
        bits = x.astype("datetime64[ms]").view(np.int64).view(np.uint64)
    else:
        bits = x.astype(np.int64).view(np.uint64)
    w = np.arange(first_row, first_row + len(x), dtype=np.uint64) * _MIX + np.uint64(1)
    return int(np.sum(bits * w, dtype=np.uint64))


# ---------------------------------------------------------------------------------------------------- other routes
class RawDaily:
    """The rows of the raw daily file under sorted keys (PERMNO, calendar row), with returns and volumes."""

    def __init__(self, cal: np.ndarray):
        from .wrds_data import load

        if len(cal) >= 2 ** 16:
            raise ValueError("the calendar is too long for 16-bit row keys")
        dsf = load("crsp_dsf_v2", columns=["permno", "dlycaldt", "dlyret", "dlyvol"])
        day = dsf["dlycaldt"].to_numpy().astype("datetime64[D]")
        key = (dsf["permno"].to_numpy().astype(np.int64) << 16) | np.searchsorted(cal, day)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.ret = dsf["dlyret"].to_numpy(dtype=np.float64)[order]
        self.vol = dsf["dlyvol"].to_numpy(dtype=np.float64)[order]
        span = (day >= bw.FIRST_DAY) & (day <= bw.LAST_DAY)
        self.rows_in_span = int(span.sum())
        self.permnos_in_span = int(dsf["permno"][span].nunique())

    def get(self, permno: np.ndarray, cal_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return and volume of each (PERMNO, calendar row); NaN where the file has no row."""
        q = (permno.astype(np.int64) << 16) | cal_row
        pos = np.minimum(np.searchsorted(self.key, q), len(self.key) - 1)
        hit = self.key[pos] == q
        return np.where(hit, self.ret[pos], np.nan), np.where(hit, self.vol[pos], np.nan)

    def daily(self, permno: np.ndarray, cal_row: np.ndarray) -> dict[str, np.ndarray]:
        """The 4 daily characteristics of each row, from the five calendar days before it (all five needed)."""
        r, v = np.empty((len(permno), bw.WEEK)), np.empty((len(permno), bw.WEEK))
        for j in range(bw.WEEK):                                                # column j: day t - WEEK + j, oldest first
            r[:, j], v[:, j] = self.get(permno, cal_row - bw.WEEK + j)
        return {"Ret_D1": r[:, -1], "Ret_W1": np.round(np.prod(1.0 + r, axis=1) - 1.0, bw.WEEK_DECIMALS),
                "STD_W1": np.round(np.std(r, axis=1, ddof=1), bw.WEEK_DECIMALS), "Vol": v.sum(axis=1)}


class RawMonthly:
    """market.parquet and accounting.parquet under sorted keys (PERMNO, holding month)."""

    def __init__(self):
        from .chars_accounting import CHARS as ACCOUNTING_CHARS
        from .chars_market import CHARS as MARKET_CHARS

        f = pd.read_parquet(bw.MARKET_FILE, columns=["permno", "month", *MARKET_CHARS]).merge(
            pd.read_parquet(bw.ACCOUNTING_FILE, columns=["permno", "month", *ACCOUNTING_CHARS]), on=["permno", "month"], how="outer")
        m = pd.PeriodIndex(f["month"], freq="M")
        key = (f["permno"].to_numpy().astype(np.int64) << 20) | np.asarray(m.year * 12 + m.month - 1, dtype=np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        if (np.diff(self.key) == 0).any():
            raise ValueError("market.parquet or accounting.parquet has two rows for a (PERMNO, month)")
        self.values = {c: f[c].to_numpy(dtype=np.float64)[order] for c in MONTHLY}

    def get(self, permno: np.ndarray, date: np.ndarray) -> dict[str, np.ndarray]:
        """The 35 values of the holding month that contains each date; NaN where the files have no row."""
        m = pd.DatetimeIndex(date).to_period("M")
        q = (permno.astype(np.int64) << 20) | np.asarray(m.year * 12 + m.month - 1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.key, q), len(self.key) - 1)
        hit = self.key[pos] == q
        return {c: np.where(hit, v[pos], np.nan) for c, v in self.values.items()}


def rf_of_calendar(cal: np.ndarray) -> np.ndarray:
    """The daily rate of each calendar day, recomputed with pandas: the Fama-French monthly rate spread over the
    month's trading days plus its unscheduled closures."""
    from .wrds_data import load

    ffm = load("ff_factors_monthly", columns=["date", "rf"])
    rate = pd.Series(ffm["rf"].to_numpy(dtype=np.float64), index=pd.PeriodIndex(ffm["date"], freq="M"))
    months = pd.DatetimeIndex(cal).to_period("M")
    n_days = pd.Series(1, index=months).groupby(level=0).sum()
    closed = pd.DatetimeIndex(bw.UNSCHEDULED_CLOSURES)
    closed = closed[~closed.isin(pd.DatetimeIndex(cal))]
    extra = pd.Series(1, index=closed.to_period("M")).groupby(level=0).sum().reindex(n_days.index, fill_value=0)
    n = (n_days + extra).reindex(months).to_numpy(dtype=np.float64)
    return (1.0 + rate.reindex(months).to_numpy(dtype=np.float64)) ** (1.0 / n) - 1.0


def read_arrays(universe: str) -> dict | None:
    """The training arrays of a universe on disk (memory-mapped), or None if they are not built."""
    folder = TRAIN_DATA_DIR / data_folder(universe, FILL)
    if not (folder / "meta.json").exists():
        return None
    out = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("dates", "permno", "ranks", "levels")}
    out["built"] = json.loads((folder / "meta.json").read_text())["built"]
    return out


def against_arrays(tab: dict, A: dict, ts: np.ndarray, flag: str) -> dict:
    """The table's member rows of one universe against its arrays, for the contiguous array days `ts`: the same
    stock-days in the same slots; the ranks and medians recomputed from the raw values with the missing-value rule;
    the rf input."""
    m = tab[flag]
    t = tab["_t"][m] - ts[0]
    start = np.r_[0, np.flatnonzero(np.diff(t)) + 1]
    slot = np.arange(len(t)) - np.repeat(start, np.diff(np.r_[start, len(t)]))
    P = np.asarray(A["permno"][ts[0]:ts[-1] + 1])
    n_day = np.bincount(t, minlength=len(ts))
    same = bool(np.array_equal(n_day, (P >= 0).sum(1)) and np.array_equal(P[t, slot], tab["permno"][m]))
    R, L = np.asarray(A["ranks"][ts[0]:ts[-1] + 1]), np.asarray(A["levels"][ts[0]:ts[-1] + 1])
    w_rank = w_med = 0.0
    if same:
        for k, c in enumerate(CHARACTERISTICS):
            V = np.full((len(ts), N_SLOTS), np.nan)
            V[t, slot] = tab[c][m]
            r, med = ranks_and_medians(V)
            w_rank = max(w_rank, float(np.abs(r.astype(np.float32) - R[:, :, k]).max()))
            w_med = max(w_med, float((np.abs(med.astype(np.float32) - L[:, k]) / np.maximum(np.abs(L[:, k]), 1e-30)).max()))
    first = np.r_[0, np.flatnonzero(np.diff(tab["_t"])) + 1]                    # the first row of each day
    rf_same = bool(len(first) == len(ts) and np.array_equal(tab["rf"][first].astype(np.float32), L[:, N_CHAR]))
    return {"same": same, "rows": int(m.sum()), "rank": w_rank, "median": w_med, "rf": rf_same}


# ---------------------------------------------------------------------------------------------------- build
def build(check_only: bool = False, src: dict | None = None, arrays: dict | None = None, out_dir: Path | None = None) -> bool:
    """Builds the table year by year, checks every cell, writes it to `out_dir` (default Data/WRDS/table) and
    reads it back. `arrays` maps a universe to its training arrays; by default they are read from disk."""
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir is not None else TABLE_DIR
    ck = Checks()
    print(f"Building the table of raw characteristics with fill = {FILL}", flush=True)
    src = src if src is not None and src["fill"] == FILL else bw.load_sources(FILL)
    cal = src["dates"]
    days = np.flatnonzero((cal >= bw.FIRST_DAY) & (cal <= bw.LAST_DAY))
    T, S = len(days), len(src["permnos"])
    member = members(src, days)
    permco = permco_of(src["permnos"])
    arrays = arrays if arrays is not None else {u: read_arrays(u) for u in WRDS_UNIVERSES}
    for u, A in arrays.items():
        if A is not None and not np.array_equal(np.asarray(A["dates"]), cal[days]):
            raise ValueError(f"the {u} arrays do not have the table's days: rebuild them with afsa.build_wrds")
    raw, mon, rf_cal = RawDaily(cal), RawMonthly(), rf_of_calendar(cal)
    print(f"  sources loaded in {time.time() - t0:.0f} s: {T:,} days from {cal[days[0]]}, {S:,} securities", flush=True)

    years = cal[days].astype("datetime64[Y]").astype(int) + 1970
    tmp = out_dir / (TABLE_NAME + ".tmp")
    writer = None
    if not check_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    n_rows, prints = 0, {c: 0 for c in COLUMNS}
    bad = {"monthly": 0, "daily": 0, "week": 0.0, "rf": 0.0}
    vs = {u: {"same": True, "rows": 0, "rank": 0.0, "median": 0.0, "rf": True} for u, A in arrays.items() if A is not None}
    empty = {k: np.zeros(N_CHAR, dtype=np.int64) for k in ("all", *FLAG.values())}
    counted = {k: 0 for k in ("all", *FLAG.values())}
    by_year, permnos_seen, sorted_unique = {}, set(), True
    last_key = -1
    try:
        for y in np.unique(years):
            ts = np.flatnonzero(years == y)
            tab = rows(src, days, member, permco, ts)
            n = len(tab["date"])
            # sorted by (day, PERMNO), no duplicate
            key = tab["_t"].astype(np.int64) * 2 ** 31 + tab["permno"]
            sorted_unique &= bool(np.all(np.diff(key) > 0) and key[0] > last_key)
            last_key = int(key[-1])
            # the 35 monthly values against the characteristic files, empty cells included
            ref = mon.get(tab["permno"], tab["date"])
            for c in MONTHLY:
                a, b = tab[c], ref[c]
                bad["monthly"] += int((np.isnan(a) != np.isnan(b)).sum() + (a[~np.isnan(a)] != b[~np.isnan(a)]).sum())
            # the 4 daily values from the rows of the raw daily file
            ref = raw.daily(tab["permno"], tab["_cal"])
            for c in DAILY:
                a, b = tab[c], ref[c]
                diff_nan = int((np.isnan(a) != np.isnan(b)).sum())
                both = ~np.isnan(a) & ~np.isnan(b)
                if c in ("Ret_W1", "STD_W1"):
                    bad["daily"] += diff_nan
                    bad["week"] = max(bad["week"], float(np.abs(a[both] - b[both]).max(initial=0.0)))
                else:
                    bad["daily"] += diff_nan + int((a[both] != b[both]).sum())
            # rf: the rate of the day before
            d_rf = np.abs(tab["rf"] - rf_cal[tab["_cal"] - 1])
            bad["rf"] = max(bad["rf"], float(d_rf.max())) if np.isfinite(d_rf).all() else np.inf
            # the arrays of each universe
            for u, A in arrays.items():
                if A is None:
                    continue
                r = against_arrays(tab, A, ts, FLAG[u])
                v = vs[u]
                v["same"] &= r["same"]
                v["rf"] &= r["rf"]
                v["rows"] += r["rows"]
                v["rank"], v["median"] = max(v["rank"], r["rank"]), max(v["median"], r["median"])
            # what is empty
            for k in empty:
                m = np.ones(n, dtype=bool) if k == "all" else tab[k]
                counted[k] += int(m.sum())
                empty[k] += np.array([int(np.isnan(tab[c][m]).sum()) for c in CHARACTERISTICS])
            by_year[int(y)] = {"days": int(len(ts)), "rows": n, "permnos": int(len(np.unique(tab["permno"]))),
                               **{f: int(tab[f].sum()) for f in FLAG.values()}}
            permnos_seen.update(np.unique(tab["permno"]).tolist())
            for c in COLUMNS:
                prints[c] = (prints[c] + fingerprint(tab[c], n_rows)) % 2 ** 64
            if writer is not None:
                writer.write_table(to_arrow(tab), row_group_size=n)
            n_rows += n
            print(f"  {y}: {n:,} rows, {by_year[int(y)]['permnos']:,} securities", flush=True)
    finally:
        if writer is not None:
            writer.close()

    # ---------------------------------------------------------------- checks
    ck.add("one row per security and trading day with a row in the raw daily file, 1989-11-01 to 2025-12-31",
           n_rows == raw.rows_in_span and len(permnos_seen) == raw.permnos_in_span and sorted_unique,
           f"{n_rows:,} rows (the daily file has {raw.rows_in_span:,} in the span), {len(permnos_seen):,} PERMNOs "
           f"({S:,} in the download), {len(set(permco[np.isin(src['permnos'], list(permnos_seen))].tolist())):,} PERMCOs; "
           f"sorted by date, then PERMNO, no duplicate: {sorted_unique}")
    ck.add("the 35 monthly values equal market.parquet and accounting.parquet for the row's PERMNO and holding month, empty cells included",
           bad["monthly"] == 0, f"{bad['monthly']} cells differ")
    ck.add("the 4 daily values equal a recomputation from the rows of the raw daily file (days t-5 .. t-1, all five needed)",
           bad["daily"] == 0 and bad["week"] <= TOL_WEEK,
           f"{bad['daily']} cells differ in Ret_D1 or Vol or in being empty; largest difference in Ret_W1 or STD_W1 {bad['week']:.1e}")
    ck.add("rf is the rate of the previous trading day, recomputed from the Fama-French monthly file and the calendar",
           bad["rf"] <= 1e-15, f"largest difference {bad['rf']:.1e}")
    for u, A in arrays.items():
        if A is None:
            ck.note(f"{data_folder(u, FILL)} not on disk", f"the rows with {FLAG[u]} were not compared with the training arrays")
            continue
        v = vs[u]
        ck.add(f"the rows with {FLAG[u]} are the stock-days of {data_folder(u, FILL)}, and ranking them with the missing-value rule "
               "gives its ranks, medians and rf input",
               v["same"] and v["rf"] and v["rank"] < 2e-6 and v["median"] < 1e-6,
               f"{v['rows']:,} stock-days in the same slots: {v['same']}; largest difference in a rank {v['rank']:.1e}, "
               f"in a median (relative) {v['median']:.1e}; rf input equal: {v['rf']} (arrays built {A.get('built', 'in memory')})")
    share = {k: {c: float(empty[k][j] / max(counted[k], 1)) for j, c in enumerate(CHARACTERISTICS)} for k in empty}
    ck.note("empty cells, share of all rows (top 6)",
            "; ".join(f"{c} {s:.2%}" for c, s in sorted(share["all"].items(), key=lambda t: -t[1])[:6]))
    ck.note("empty cells, share of in_broad rows (top 6)",
            "; ".join(f"{c} {s:.2%}" for c, s in sorted(share["in_broad"].items(), key=lambda t: -t[1])[:6]))

    if ck.failed or check_only:
        if tmp.exists():
            tmp.unlink()
        print(f"Checks finished in {time.time() - t0:.0f} s: {len(ck.failed)} failed." + (f" FAILED: {'; '.join(ck.failed)}" if ck.failed else ""))
        return not ck.failed

    # ---------------------------------------------------------------- read back
    with pq.ParquetFile(tmp) as f:                                             # closed before the rename (Windows)
        md = f.metadata
        ok_meta = md.num_rows == n_rows and f.schema_arrow.equals(SCHEMA) and md.num_row_groups == len(by_year)
        null_free = all(md.row_group(g).column(j).statistics.null_count == 0 for g in range(md.num_row_groups)
                        for j, c in enumerate(COLUMNS) if c not in CHARACTERISTICS)
        n_back, n_groups = md.num_rows, md.num_row_groups
    differ = [c for c in COLUMNS if fingerprint(pq.read_table(tmp, columns=[c]).column(c).to_numpy(), 0) != prints[c]]
    if not ck.add("read back from the file: every column equals the table in memory, nulls exactly where values are empty",
                  ok_meta and not differ and null_free,
                  f"{n_back:,} rows in {n_groups} row groups; columns that differ: {differ or 'none'}; "
                  f"identifiers and rf without null: {null_free}"):
        tmp.unlink()
        return False
    final = out_dir / TABLE_NAME
    os.replace(tmp, final)

    files = [bw.RAW_DIR / "crsp_index_daily.parquet", bw.RAW_DIR / "crsp_dsf_v2.parquet", bw.RAW_DIR / "ff_factors_monthly.parquet",
             *(bw.UNIVERSE_DIR / f"{u.removeprefix('wrds_')}.parquet" for u in WRDS_UNIVERSES), bw.MARKET_FILE, bw.ACCOUNTING_FILE]
    meta = {
        "built": datetime.now().isoformat(timespec="seconds"), "builder": "afsa.wrds_table", "file": TABLE_NAME, "fill": FILL,
        "sources": {str(p.relative_to(WRDS_DIR)).replace("\\", "/") if p.is_relative_to(WRDS_DIR) else str(p):
                    {"bytes": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
                    for p in files if p.exists()},
        "arrays_compared": {data_folder(u, FILL): (A.get("built", "in memory") if A is not None else None) for u, A in arrays.items()},
        "n_rows": n_rows, "n_permnos": len(permnos_seen), "n_permcos": len(set(permco[np.isin(src["permnos"], list(permnos_seen))].tolist())),
        "first_day": str(cal[days[0]]), "last_day": str(cal[days[-1]]), "n_days": int(T),
        "columns": {
            "date": "trading day t", "permno": "CRSP PERMNO (security)", "permco": "CRSP PERMCO (company)",
            "in_broad": "in the broad universe on day t: on the list of t's month and a row in the daily file that day",
            "in_common": "the same for the common universe (US common stocks only)",
            "Ret_D1": "return of day t-1", "Ret_W1": "compounded return of days t-5 .. t-1, rounded to 12 decimals",
            "STD_W1": "sample standard deviation (divisor 4) of the returns of days t-5 .. t-1, rounded to 12 decimals",
            "Vol": "shares traded (CRSP dlyvol) on days t-5 .. t-1",
            **{c: "monthly: the value of t's holding month (afsa.chars_market or afsa.chars_accounting)" for c in MONTHLY},
            "rf": "risk-free rate of day t-1, daily, in decimals: the Fama-French monthly rate spread over the month's scheduled trading days",
        },
        "groups": {g: list(cs) for g, cs in CHAR_GROUPS.items()},
        "timing": "row t is the paper's X_{t-1}, as in the training arrays: everything in it was known at the close of day t-1",
        "missing_values": "fill none: a characteristic whose raw inputs are not available is null; no carry-forward, no imputation. "
                          "Accounting: the production build of afsa.chars_accounting (15 secondary items count as 0 where the panel "
                          "shows it; a missing XSGA, DP, COGS, CEQ or LCT leaves the characteristic null). Daily: all five days needed. "
                          "Ranking (rank 0 for an empty value) is applied by the training arrays, per universe, not here.",
        "rows_by_year": by_year,
        "empty_share": share,
        "checks": ck.items,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {final} ({final.stat().st_size / 2 ** 20:,.0f} MB) and meta.json in {time.time() - t0:.0f} s.")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check-only", action="store_true", help="build and check in memory, write nothing")
    args = p.parse_args()
    sys.exit(0 if build(check_only=args.check_only) else 1)


if __name__ == "__main__":
    main()
