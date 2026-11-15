"""Month blocks for the rebuilt-history entry rule: Data/TrainingData/history/wrds_<universe>_fill-<rule>/.

    python -m afsa.build_history                         # wrds_broad and wrds_common, fill none (about 1 minute each)
    python -m afsa.build_history --universe wrds_broad   # one universe
    python -m afsa.build_history --check-only            # run the checks, write nothing
    python -m afsa.build_history --universe wrds_broad_buffer600   # a buffered list: 500 to 600 securities, 600 slots

Why. Under the entry rule decided on 2026-11-14 (CLAUDE.md section 7) every member of day t's universe gets a signal
from its 30 lagged residuals, and those are recomputed inside day t's universe: on each lag day s the factors and
residuals are built over day t's members, from their characteristics of s-1 and returns of s, including the days
before a security entered. The day x slot arrays of afsa.build_wrds cannot serve that: there a lag day's ranks are
taken over that day's own universe, and a security has no row before it enters. Day t's universe is the top-500 list
of t's holding month, fixed on the last trading day of the month before, so one block per holding month holds all
that the month's days need, lookback included. Every 8-year window is a run of whole months, so each window has its
lookback, and so does each new member.

The block of holding month m:
    slots          the month's list (the 500 largest at the end of m-1), sorted by PERMNO. A buffered list
                   (config.BUFFERED_UNIVERSES) holds 500 to `keep` securities in `keep` slots, the empty ones last
                   (PERMNO -1, never a member)
    rows           the `lookback` (30) trading days before m's first trading day, then m's trading days (15 to 23),
                   consecutive in the calendar. The weights of a day of m read the residuals of the 30 rows before it,
                   all inside the block. A day's previous weights (for its cost) come from the block of its own month,
                   so the first day of m takes them from m-1's block.
    member         the list security has a row in the CRSP daily file that day. On m's own days these are exactly the
                   members of the day x slot arrays; on the lookback days they include the securities that enter the
                   list in m (their days before entry).
    ranks, levels  row s holds what was known at the close of s-1, as in the day x slot arrays: the 4 daily
                   characteristics from days s-5..s-1 and the 35 monthly ones of s's OWN holding month (on a lookback
                   day in m-1, those of m-1: m's values were formed after it). Each is ranked over the members of that
                   row that have a value, (rank - 1) / (n - 1) - 0.5 with average ranks for ties; a member without a
                   value gets 0 (the missing-value policy of CLAUDE.md section 4). `levels` holds the 39 medians over
                   the same values and the rf of s-1.
    ret            the total return of day s (the excess return is ret - rf); 0 where CRSP has none (as in the
                   arrays) and for non-members.
    listed         trading days with a row in the daily file up to and including s. The file has no gap inside a
                   security's life (the build checks it), so this counts the days since listing (from 1984-01-03,
                   the file's first day, for older securities).
    run            consecutive trading days in the universe of each day (the list of that day's own month) up to and
                   including s: `run` of the day x slot arrays, so the current entry rule can be applied to the blocks.
    accounts       an annual report has entered by m: some holding month up to m uses a fiscal year of the
                   security's firm (accounting.parquet, fiscal rule safe: year-end + 4 months, 6 for ADRs and non-US
                   issuers). Once true, it stays true.
    accounts_post  the same, counting only fiscal years that end on or after the security's first row in the daily
                   file (Compustat backfills pre-listing years: the open question of CLAUDE.md section 10).

What the blocks change. On m's own days they equal the day x slot arrays stock-day by stock-day (the build checks all
of them), so the residuals of those days are the current ones. Only the lookback days before m differ: they are
rebuilt over m's list. The decided rule, applied by the model and not stored: a member gets a signal on day t if
listed > lookback and accounts; it stays in the factors and the hedge either way.

Arrays (C calendar days, M = 434 months 1989-11 .. 2025-12, R = lookback + 23 rows, 500 slots):
    dates.npy          [C]              datetime64[D]  the blocks' calendar, from the first lookback row of 1989-11
    rf.npy             [C]              float64        risk-free rate of each day (afsa.build_wrds.spread_rf)
    months.npy         [M]              datetime64[M]  the holding months
    row.npy            [M, R]           int32          position in `dates` of each block row; -1 = padding
    n_rows.npy         [M]              int16          lookback + the month's trading days
    permno.npy         [M, 500]         int32          the month's list, sorted
    member.npy         [M, R, 500]      bool
    ranks.npy          [M, R, 500, 39]  float32        0 for non-members, padding and missing values
    levels.npy         [M, R, 40]       float32        39 medians + rf of the previous trading day; 0 on padding
    ret.npy            [M, R, 500]      float32
    listed.npy         [M, R, 500]      int16          0 for non-members (capped at 32767)
    run.npy            [M, R, 500]      int16          0 for non-members
    accounts.npy       [M, 500]         bool
    accounts_post.npy  [M, 500]         bool
    meta.json                                          definitions, sources, checks and coverage notes

Everything written here derives from CRSP/Compustat: it stays under Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_wrds as bw
from .build_data import Checks
from .build_window import DAILY, ranks_and_medians
from .config import (CHARACTERISTICS, FILLS, LEVEL_NAMES, N_CHAR, N_LEVEL, N_SLOTS, TRAIN_DATA_DIR, WRDS_BUILDS, WRDS_DIR,
                     WRDS_UNIVERSES, Config, data_folder, n_slots)

HISTORY_DIR = TRAIN_DATA_DIR / "history"
FIRST_MONTH, LAST_MONTH = "1989-11", "2025-12"
INT16_MAX = np.iinfo(np.int16).max
ARRAYS = ("dates", "rf", "months", "row", "n_rows", "permno", "member", "ranks", "levels", "ret", "listed", "run",
          "accounts", "accounts_post")
# Blocks whose lookback rows are recomputed from the whole table (afsa.wrds_table) by the build: the first loss month,
# the months around the closure of September 2001, the first month of the extension, the last month; plus random ones.
TABLE_SAMPLE = ("1990-01", "2001-09", "2001-10", "2022-01", "2025-12")


def history_folder(universe: str, fill: str = "none", lookback: int | None = None) -> Path:
    """Folder of a universe's month blocks. A lookback other than Config's default is part of the name."""
    name = data_folder(universe, fill)
    if lookback is not None and lookback != Config().lookback:
        name += f"_lookback-{lookback}"
    return HISTORY_DIR / name


# ---------------------------------------------------------------------------------------------------- pieces
def consecutive(present: np.ndarray) -> np.ndarray:
    """Consecutive True values along axis 0, up to and including each row (0 where False). int32."""
    run = np.zeros(present.shape, dtype=np.int32)
    run[0] = present[0]
    for i in range(1, len(present)):
        run[i] = np.where(present[i], run[i - 1] + 1, 0)
    return run


def month_rows(src: dict, lookback: int, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For consecutive holding months (indices of src's monthly matrices): the calendar rows of each block
    [M, lookback + longest month] (-1 = padding) and the rows used per block."""
    mod = src["month_of_day"]
    if len(months) == 0 or (np.diff(months) != 1).any():
        raise ValueError("the holding months must be consecutive")
    starts, ends = np.searchsorted(mod, months, "left"), np.searchsorted(mod, months, "right")
    if (ends <= starts).any():
        raise ValueError("a holding month has no trading day in the calendar")
    if (starts - lookback < 1).any():
        raise ValueError(f"the calendar does not hold {lookback} trading days (and the rf of the day before) ahead of the first month")
    n_rows = lookback + ends - starts
    row = np.full((len(months), int(n_rows.max())), -1, dtype=np.int64)
    for b, (s, e) in enumerate(zip(starts, ends)):
        row[b, : n_rows[b]] = np.arange(s - lookback, e)
    return row, n_rows


def accounts_flags(src: dict, path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """[holding months, securities] of src: an annual report has entered by the month (some month up to it uses a
    fiscal year: accounting.parquet has a `datadate` for it), and the same counting only fiscal years that end on or
    after the security's first row in the daily file. Both stay true once true."""
    acc = pd.read_parquet(path or bw.ACCOUNTING_FILE, columns=["permno", "month", "datadate"])
    n_months, S = int(src["month_of_day"].max()) + 1, len(src["permnos"])
    mi = bw._month_index(acc["month"]) - src["month0"]
    ci = np.searchsorted(src["permnos"], acc["permno"].to_numpy())
    if not np.array_equal(src["permnos"][np.minimum(ci, S - 1)], acc["permno"].to_numpy()):
        raise ValueError("accounting.parquet has a PERMNO that is not in crsp_dsf_v2")
    keep = (mi >= 0) & (mi < n_months)
    entered = acc["datadate"].notna().to_numpy()
    listing = src["dates"][src["has_row"].argmax(axis=0)]                       # first row of each security
    after = entered & (acc["datadate"].to_numpy().astype("datetime64[D]") >= listing[np.minimum(ci, S - 1)])
    a, p = np.zeros((n_months, S), dtype=bool), np.zeros((n_months, S), dtype=bool)
    a[mi[keep], ci[keep]] = entered[keep]
    p[mi[keep], ci[keep]] = after[keep]
    return np.maximum.accumulate(a, axis=0), np.maximum.accumulate(p, axis=0)


def assemble(src: dict, universe: str, lookback: int, flags: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """The arrays of one universe (module docstring), one block per month of its list file, plus what the checks need:
    `cols` [M, 500] (columns of src), `ret_missing` [M, R, 500] (a member row without a return), `n_value` [M, R, 39]
    and the list file `u`."""
    on_list, u = bw.universe_lists(universe, src)
    months = np.flatnonzero(on_list.any(axis=1))                               # the months that have a list
    row, n_rows = month_rows(src, lookback, months)
    M, R = row.shape
    width = n_slots(universe)
    cols = np.zeros((M, width), dtype=np.int64)
    used = np.zeros((M, width), dtype=bool)                                    # a shorter list leaves the last slots empty
    for b, m in enumerate(months):
        c = np.flatnonzero(on_list[m])                                         # src permnos are sorted: so is the list
        if len(c) > width:
            raise ValueError(f"the list of month {m} has {len(c)} securities, more than {width}")
        cols[b, :len(c)], used[b, :len(c)] = c, True
    pad = row < 0
    r = np.where(pad, 0, row)                                                  # a valid index for the padding rows
    R3, C3 = r[:, :, None], cols[:, None, :]
    member = src["has_row"][R3, C3] & ~pad[:, :, None] & used[:, None, :]
    raw_ret = src["ret"][R3, C3]
    ret = np.where(member, np.nan_to_num(raw_ret, nan=0.0), 0.0).astype(np.float32)
    listed = np.where(member, np.minimum(consecutive(src["has_row"])[R3, C3], INT16_MAX), 0).astype(np.int16)
    in_universe = on_list[src["month_of_day"]] & src["has_row"]                # [calendar, securities]: the universe of each day
    run = np.where(member, np.minimum(consecutive(in_universe)[R3, C3], INT16_MAX), 0).astype(np.int16)

    ranks = np.zeros((M, R, width, N_CHAR), dtype=np.float32)
    levels = np.zeros((M, R, N_LEVEL), dtype=np.float32)
    n_value = np.zeros((M, R, N_CHAR), dtype=np.int32)
    own_month = src["month_of_day"][r][:, :, None]                             # the holding month of each ROW, not of the block
    for k, c in enumerate(CHARACTERISTICS):
        raw = src["daily"][c][R3, C3] if c in DAILY else src["monthly"][c][own_month, C3]
        v = np.where(member, raw, np.nan).reshape(M * R, width)
        rk, med = ranks_and_medians(v)
        ranks[:, :, :, k] = rk.reshape(M, R, width)
        levels[:, :, k] = med.reshape(M, R)
        n_value[:, :, k] = (~np.isnan(v)).sum(axis=1).reshape(M, R)
    levels[:, :, N_CHAR] = np.where(pad, 0.0, src["rf"][r - 1])               # rf of the previous trading day
    acc, post = flags if flags is not None else accounts_flags(src)
    c0, c1 = int(row[0, 0]), int(row.max())                                    # the blocks' calendar
    arrays = {
        "dates": src["dates"][c0:c1 + 1], "rf": src["rf"][c0:c1 + 1],
        "months": (months + src["month0"] - 1970 * 12).astype("datetime64[M]"),
        "row": np.where(pad, -1, row - c0).astype(np.int32), "n_rows": n_rows.astype(np.int16),
        "permno": np.where(used, src["permnos"][cols], -1).astype(np.int32), "member": member, "ranks": ranks, "levels": levels,
        "ret": ret, "listed": listed, "run": run,
        "accounts": acc[months[:, None], cols] & used, "accounts_post": post[months[:, None], cols] & used,
    }
    return {"arrays": arrays, "cols": cols, "ret_missing": member & np.isnan(raw_ret), "n_value": n_value, "u": u,
            "lookback": lookback, "c0": c0}


# ---------------------------------------------------------------------------------------------------- checks
def own_rows(arr: dict, lookback: int) -> np.ndarray:
    """[M, R]: the row is one of the month's own trading days (not lookback, not padding)."""
    M, R = arr["row"].shape
    return (np.arange(R)[None, :] >= lookback) & (np.arange(R)[None, :] < arr["n_rows"][:, None])


def against_day_arrays(arr: dict, lookback: int, day: dict) -> dict:
    """The month's own rows of every block against the day x slot arrays `day` (dates, permno, ranks, levels, ret,
    run; memory-mapped is fine): each array day must be the own row of exactly one block, with the same members in
    the same order and the same ranks, medians, rf input, returns and run, bit for bit."""
    own = own_rows(arr, lookback)
    b_idx, r_idx = np.nonzero(own)                                             # by block, then row: calendar order
    when = arr["dates"][arr["row"][b_idx, r_idx]]
    out = {"days": len(when), "same_days": bool(np.array_equal(when, np.asarray(day["dates"]))), "members": True,
           "rank": 0.0, "level": 0.0, "ret": 0.0, "run": True, "stock_days": 0}
    if not out["same_days"]:
        return out
    for t, (b, r) in enumerate(zip(b_idx, r_idx)):
        mem = arr["member"][b, r]
        n = int(mem.sum())
        P = np.asarray(day["permno"][t])
        if not (np.array_equal(P[:n], arr["permno"][b][mem]) and (P[n:] == -1).all()):
            out["members"] = False
            continue
        out["stock_days"] += n
        out["rank"] = max(out["rank"], float(np.abs(np.asarray(day["ranks"][t, :n]) - arr["ranks"][b, r][mem]).max()))
        out["level"] = max(out["level"], float(np.abs(np.asarray(day["levels"][t]) - arr["levels"][b, r]).max()))
        out["ret"] = max(out["ret"], float(np.abs(np.asarray(day["ret"][t, :n]) - arr["ret"][b, r][mem]).max()))
        out["run"] &= bool(np.array_equal(np.asarray(day["run"][t, :n]), arr["run"][b, r][mem]))
    return out


def read_day_arrays(universe: str, fill: str) -> dict | None:
    """The day x slot arrays of afsa.build_wrds on disk (memory-mapped), or None if they are not built."""
    folder = TRAIN_DATA_DIR / data_folder(universe, fill)
    if not (folder / "meta.json").exists():
        return None
    out = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("dates", "permno", "ranks", "levels", "ret", "run")}
    out["built"] = json.loads((folder / "meta.json").read_text())["built"]
    return out


def against_table(arr: dict, lookback: int, blocks: np.ndarray, table: Path, dsf: Path) -> dict:
    """Every row of the blocks in `blocks` that the whole table covers (from 1989-11-01), recomputed with pandas from
    the table's raw values (afsa.wrds_table, itself checked cell by cell against other routes) and from the raw daily
    file: the members, the 39 ranks with the missing-value rule, the medians, the rf input and the returns. The
    lookback rows are the point: there the cross-section is the block's list, not the day's own universe."""
    import pyarrow.parquet as pq

    out = {"rows": 0, "lookback_rows": 0, "stock_rows": 0, "members": True, "rank": 0.0, "median": 0.0, "rf": True, "ret": 0.0}
    for b in blocks:
        rows = arr["row"][b][arr["row"][b] >= 0]
        days = arr["dates"][rows]
        days = days[days >= bw.FIRST_DAY]
        P = arr["permno"][b].astype(np.int64)
        lo, hi = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
        tab = pq.read_table(table, filters=[("date", ">=", lo), ("date", "<=", hi)],
                            columns=["date", "permno", *CHARACTERISTICS, "rf"]).to_pandas()
        tab["date"] = tab["date"].to_numpy().astype("datetime64[D]")
        rf_day = tab.groupby("date")["rf"].first().reindex(days)
        tab = tab[tab["permno"].isin(P)]
        ret = pd.read_parquet(dsf, columns=["permno", "dlycaldt", "dlyret"], filters=[("dlycaldt", ">=", lo), ("dlycaldt", "<=", hi)])
        ret = ret[ret["permno"].isin(P)]
        ret["day"] = ret["dlycaldt"].to_numpy().astype("datetime64[D]")
        R_want = ret.pivot(index="day", columns="permno", values="dlyret").reindex(index=days, columns=P)
        present = tab.assign(x=1.0).pivot(index="date", columns="permno", values="x").reindex(index=days, columns=P).notna().to_numpy()
        pos = np.searchsorted(arr["dates"], days)
        r_of = {int(c): i for i, c in enumerate(arr["row"][b])}
        ridx = np.array([r_of[int(p)] for p in pos])
        mem = arr["member"][b][ridx]
        out["members"] &= bool(np.array_equal(present, mem))
        out["rows"] += len(days)
        out["lookback_rows"] += int((ridx < lookback).sum())
        out["stock_rows"] += int(mem.sum())
        want_ret = np.where(mem, np.nan_to_num(R_want.to_numpy(), nan=0.0), 0.0).astype(np.float32)
        out["ret"] = max(out["ret"], float(np.abs(want_ret - arr["ret"][b][ridx]).max()))
        out["rf"] &= bool(np.array_equal(rf_day.to_numpy().astype(np.float32), arr["levels"][b][ridx, N_CHAR]))
        for k, c in enumerate(CHARACTERISTICS):
            x = tab.pivot(index="date", columns="permno", values=c).reindex(index=days, columns=P)
            x = x.where(mem)
            n = x.notna().sum(axis=1).to_numpy()[:, None]
            want = ((x.rank(axis=1, method="average").to_numpy() - 1) / np.maximum(n - 1, 1) - 0.5)
            want = np.where(np.isnan(want), 0.0, want)
            out["rank"] = max(out["rank"], float(np.abs(want - arr["ranks"][b][ridx, :, k]).max()))
            med = x.median(axis=1, skipna=True).fillna(0.0).to_numpy()
            got = arr["levels"][b][ridx, k].astype(np.float64)
            out["median"] = max(out["median"], float((np.abs(got - med.astype(np.float32)) / np.maximum(np.abs(med), 1e-30)).max()))
    return out


def lag_rank_shift(arr: dict, lookback: int) -> dict:
    """How far the lookback rows' ranks are from the ranks the same security had that day in its own day's universe
    (the own row of the block of that day's month), where it was a member of both. Per characteristic: mean and
    largest absolute difference. Also the lookback stock-rows of securities that were not in that day's universe."""
    M, R = arr["row"].shape
    first_row = {int(arr["row"][b, lookback]): b for b in range(M)}            # calendar row of each month's first day
    starts = np.array(sorted(first_row))
    diff_sum, diff_max, n_both, n_off = np.zeros(N_CHAR), np.zeros(N_CHAR), 0, 0
    for b in range(1, M):
        for r in range(lookback):
            c = int(arr["row"][b, r])
            k = int(np.searchsorted(starts, c, "right")) - 1
            if k < 0:
                continue
            ob = first_row[int(starts[k])]
            orow = lookback + c - int(starts[k])
            n_ob = int((arr["permno"][ob] >= 0).sum())                         # used slots come first, sorted
            pos = np.minimum(np.searchsorted(arr["permno"][ob][:n_ob], arr["permno"][b]), n_ob - 1)
            both = (arr["permno"][ob][pos] == arr["permno"][b]) & arr["member"][b, r] & arr["member"][ob, orow][pos]
            n_off += int((arr["member"][b, r] & ~both).sum())
            d = np.abs(arr["ranks"][b, r][both].astype(np.float64) - arr["ranks"][ob, orow][pos[both]])
            diff_sum += d.sum(0)
            diff_max = np.maximum(diff_max, d.max(0, initial=0.0))
            n_both += int(both.sum())
    return {"stock_rows_in_both": n_both, "stock_rows_not_in_own_universe": n_off,
            "mean": {c: float(diff_sum[k] / max(n_both, 1)) for k, c in enumerate(CHARACTERISTICS)},
            "max": {c: float(diff_max[k]) for k, c in enumerate(CHARACTERISTICS)}}


def coverage(arr: dict, lookback: int, ret_missing: np.ndarray) -> dict:
    """Signals on the month's own days from 1990-01-02 (the loss days) under the current rule (31 consecutive days in
    the universe) and the decided one (listed > lookback and accounts), and what the new signals read."""
    own = own_rows(arr, lookback)
    when = arr["dates"][np.where(arr["row"] >= 0, arr["row"], 0)]
    span = (own & (when >= bw.FIRST_LOSS_DAY))[:, :, None]
    mem = arr["member"] & span
    current = mem & (arr["run"] > lookback)
    listed = mem & (arr["listed"] > lookback)
    acc, post = mem & arr["accounts"][:, None, :], mem & arr["accounts_post"][:, None, :]
    decided = listed & acc
    n = int(mem.sum())
    pm = np.broadcast_to(arr["permno"][:, None, :], mem.shape)
    all_p, all_n = np.unique(pm[mem], return_counts=True)
    no_p, no_n = np.unique(pm[mem & ~arr["accounts"][:, None, :]], return_counts=True)
    never = {int(p): int(k) for p, k in zip(no_p, no_n) if k == all_n[np.searchsorted(all_p, p)]}   # no accounts on any member day
    # new signals: members with a decided signal but fewer than lookback + 1 days in the universe. Their lags reach
    # rows before they entered; how many of those rows lack a return?
    new = decided & ~current
    nb, nr, ns = np.nonzero(new)
    cm = np.concatenate([np.zeros((*ret_missing.shape[:1], 1, ret_missing.shape[2]), dtype=np.int32),
                         np.cumsum(ret_missing, axis=1, dtype=np.int32)], axis=1)   # cm[:, j] = missing among rows < j
    lag_missing = int((cm[nb, nr, ns] - cm[nb, nr - lookback, ns]).sum())
    return {"member_stock_days": n, "current": int(current.sum()), "decided": int(decided.sum()), "decided_post": int((listed & post).sum()),
            "no_signal_listing_only": int((mem & ~listed & acc).sum()), "no_signal_accounts_only": int((listed & ~acc).sum()),
            "no_signal_both": int((mem & ~listed & ~acc).sum()), "gain": int(new.sum()), "loss": int((current & ~decided).sum()),
            "never_accounts": never, "new_signal_lag_rows": int(len(nb) * lookback), "new_signal_lag_rows_without_return": lag_missing}


def build(universe: str = "wrds_broad", fill: str = "none", check_only: bool = False, src: dict | None = None,
          lookback: int | None = None, day_arrays: dict | None = None, table: Path | None = None) -> bool:
    t0 = time.time()
    if universe not in WRDS_BUILDS:
        raise ValueError(f"unknown universe {universe!r}; choose from {WRDS_BUILDS}")
    lookback = Config().lookback if lookback is None else lookback
    width = n_slots(universe)
    ck = Checks()
    print(f"Building the month blocks of {universe!r} with fill = {fill}, lookback {lookback}")
    files = [bw.RAW_DIR / "crsp_index_daily.parquet", bw.RAW_DIR / "crsp_dsf_v2.parquet", bw.RAW_DIR / "ff_factors_monthly.parquet",
             bw.UNIVERSE_DIR / f"{universe.removeprefix('wrds_')}.parquet", bw.MARKET_FILE, bw.ACCOUNTING_FILE]
    if not ck.add("source files exist", all(p.exists() for p in files), ", ".join(p.name for p in files)):
        return False
    src = src if src is not None and src["fill"] == fill else bw.load_sources(fill)
    a = assemble(src, universe, lookback)
    arr = a["arrays"]
    M, R = arr["row"].shape
    own = own_rows(arr, lookback)
    member, ranks, levels = arr["member"], arr["ranks"], arr["levels"]

    # ---------------------------------------------------------------- calendar and blocks
    has = src["has_row"]
    first, last = has.argmax(axis=0), len(has) - 1 - has[::-1].argmax(axis=0)
    inside = (np.arange(len(has))[:, None] >= first) & (np.arange(len(has))[:, None] <= last)
    ck.add("the daily file has no gap inside a security's life, so `listed` counts the trading days since listing",
           not (inside & ~has).any(), f"{len(src['permnos']):,} securities, {int(has.sum()):,} rows from {src['dates'][0]}")
    months = pd.PeriodIndex(arr["months"].astype("datetime64[M]").astype(str), freq="M")
    own_dates = arr["dates"][arr["row"][:, lookback:]]
    first_ok = all(pd.Timestamp(arr["dates"][arr["row"][b, lookback]]).to_period("M") == months[b]
                   and pd.Timestamp(arr["dates"][arr["row"][b, lookback] - 1]).to_period("M") != months[b] for b in range(M))
    contiguous = all(np.array_equal(arr["row"][b, : arr["n_rows"][b]], arr["row"][b, 0] + np.arange(arr["n_rows"][b]))
                     and (arr["row"][b, arr["n_rows"][b]:] == -1).all() for b in range(M))
    whole = all((pd.DatetimeIndex(own_dates[b][: arr["n_rows"][b] - lookback]).to_period("M") == months[b]).all() for b in range(M))
    per_month = pd.Series(1, index=pd.DatetimeIndex(arr["dates"]).to_period("M")).groupby(level=0).sum()
    all_days = bool(np.array_equal(arr["n_rows"] - lookback, per_month.reindex(months).to_numpy()))
    ck.add(f"one block per holding month {FIRST_MONTH} .. {LAST_MONTH}: {lookback} consecutive trading days before the month's first "
           "day, then all of the month's days",
           M == len(pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")) and str(months[0]) == FIRST_MONTH and str(months[-1]) == LAST_MONTH
           and first_ok and contiguous and whole and all_days,
           f"{M} blocks of {lookback} + {int((arr['n_rows'] - lookback).min())} to {int((arr['n_rows'] - lookback).max())} rows "
           f"(padded to {R}); calendar {arr['dates'][0]} .. {arr['dates'][-1]} ({len(arr['dates']):,} days)")
    P = arr["permno"]
    u = a["u"]
    lists = u.groupby(pd.PeriodIndex(u["month"], freq="M"))["permno"].apply(lambda s: np.sort(s.to_numpy(dtype=np.int64)))
    n_list = (P >= 0).sum(axis=1)
    size_ok = bool((n_list == N_SLOTS).all() if width == N_SLOTS else ((n_list >= N_SLOTS) & (n_list <= width)).all())
    ck.add(f"the slots of a block are the month's list ({N_SLOTS if width == N_SLOTS else f'{N_SLOTS} to {width}'} every month), "
           "sorted by PERMNO, empty slots last",
           size_ok and all(np.array_equal(P[b, :n_list[b]], lists[months[b]]) and (P[b, n_list[b]:] == -1).all() for b in range(M)),
           f"{len(np.unique(P[P >= 0])):,} PERMNOs on some list; {width} slots, {int(n_list.min())} to {int(n_list.max())} used")

    # ---------------------------------------------------------------- the month's own days = the day x slot arrays
    day = day_arrays if day_arrays is not None else read_day_arrays(universe, fill)
    if day is None:
        ck.note(f"{data_folder(universe, fill)} not on disk", "the months' own rows were not compared with the day x slot arrays")
    else:
        v = against_day_arrays(arr, lookback, day)
        ck.add(f"the months' own rows equal the day x slot arrays {data_folder(universe, fill)} on every stock-day: members, ranks, "
               "medians and rf input, returns, run",
               v["same_days"] and v["members"] and v["rank"] == 0 and v["level"] == 0 and v["ret"] == 0 and v["run"],
               f"{v['days']:,} days, {v['stock_days']:,} stock-days; largest difference in a rank {v['rank']:.1e}, in a level "
               f"{v['level']:.1e}, in a return {v['ret']:.1e}; run equal: {v['run']} (arrays built {day.get('built', 'in memory')})")

    # ---------------------------------------------------------------- the lookback rows, recomputed from the whole table
    table = table if table is not None else WRDS_DIR / "table" / "chars_daily_fill-none.parquet"
    if fill != "none" or not table.exists():
        ck.note("whole table not used", f"{table.name} holds fill none; the rows were not recomputed from it")
    else:
        rng = np.random.default_rng(0)
        pick = [b for b in range(M) if str(months[b]) in TABLE_SAMPLE]
        pick = np.unique(np.r_[pick, rng.choice(np.arange(2, M), size=7, replace=False)])
        t = against_table(arr, lookback, pick, table, bw.RAW_DIR / "crsp_dsf_v2.parquet")
        ck.add("rows of sample blocks, lookback rows included, equal a recomputation with pandas from the whole table's raw values "
               "(ranks over the block's list) and from the raw daily file (returns)",
               t["members"] and t["rank"] < 2e-6 and t["median"] < 1e-6 and t["rf"] and t["ret"] == 0,
               f"{len(pick)} blocks ({', '.join(str(months[b]) for b in pick)}), {t['rows']:,} rows ({t['lookback_rows']:,} lookback), "
               f"{t['stock_rows']:,} stock-rows: "
               f"members equal: {t['members']}; largest difference in a rank {t['rank']:.1e}, in a median (relative) {t['median']:.1e}, "
               f"in a return {t['ret']:g}; rf input equal: {t['rf']}")

    # ---------------------------------------------------------------- what the signal rule relies on
    lb_cum = np.concatenate([np.zeros((M, 1, width), dtype=np.int32), np.cumsum(member, axis=1, dtype=np.int32)], axis=1)
    rows_ = np.arange(lookback, R)
    prior = lb_cum[:, rows_, :] - lb_cum[:, rows_ - lookback, :]               # members among the `lookback` rows before
    need = (arr["listed"][:, rows_, :] > lookback) & own[:, rows_][:, :, None]
    ck.add(f"a member with listed > {lookback} has a row on each of the {lookback} rows before it in its block (its lags exist)",
           bool((prior[need] == lookback).all()), f"{int(need.sum()):,} stock-days checked")
    ok_run = bool((arr["run"] <= arr["listed"]).all() and ((arr["listed"] >= 1) == member).all() and ((arr["run"] >= 1) <= member).all())
    ck.add("listed >= 1 exactly on member rows, and run <= listed", ok_run)
    acc, post = arr["accounts"], arr["accounts_post"]
    mono = True
    for b in range(1, M):
        pos = np.minimum(np.searchsorted(P[b - 1][:n_list[b - 1]], P[b]), n_list[b - 1] - 1)
        hit = (P[b - 1][pos] == P[b]) & (P[b] >= 0)
        mono &= bool((acc[b][hit] >= acc[b - 1][pos[hit]]).all() and (post[b][hit] >= post[b - 1][pos[hit]]).all())
    ck.add("accounts and accounts_post never turn false again, and accounts_post implies accounts", mono and bool((post <= acc).all()),
           f"accounts on {acc[P >= 0].mean():.2%} of list stock-months, accounts_post on {post[P >= 0].mean():.2%}")

    # ---------------------------------------------------------------- values
    pad = arr["row"] < 0
    ck.add("no missing value; ranks in [-0.5, 0.5]; non-members and padding hold 0 in ranks and returns; padding levels are 0",
           bool(np.isfinite(ranks).all() and np.isfinite(levels).all() and np.isfinite(arr["ret"]).all()
                and ranks.min() >= -0.5 and ranks.max() <= 0.5 and not ranks[~member].any() and not arr["ret"][~member].any()
                and not levels[pad].any()),
           f"min {ranks.min():.4f}, max {ranks.max():.4f}")
    n_mem = member.sum(axis=2)
    centred = float(np.abs(ranks.sum(axis=2, dtype=np.float64) / np.maximum(n_mem, 1)[:, :, None]).max())
    ck.add("ranks are centred on zero on every row", centred < 1e-4, f"largest |mean over the row's members| {centred:.1e}")
    fewest = np.where(pad[:, :, None], np.iinfo(np.int32).max, a["n_value"]).min(axis=(0, 1))
    ck.add("every characteristic has a value for at least 100 members on every row", bool(fewest.min() >= 100),
           "fewest: " + ", ".join(f"{CHARACTERISTICS[k]} {fewest[k]}" for k in np.argsort(fewest)[:4]))
    lag = ~own & ~pad
    ck.note("members per row", f"own rows {int(n_mem[own].min())} to {int(n_mem[own].max())}; lookback rows {int(n_mem[lag].min())} "
            f"to {int(n_mem[lag].max())} (list securities not yet listed, or delisted, are absent)")

    # ---------------------------------------------------------------- no look-ahead: Ret_D1 on the lookback rows is the day before's return
    k_d1 = CHARACTERISTICS.index("Ret_D1")
    corr = []
    for shift in (-1, 0, 1):
        cc = []
        for b in range(0, M, 7):                                               # every 7th block
            for r in range(lookback):
                c = int(arr["row"][b, r]) + a["c0"]
                mem = member[b, r] & ~np.isnan(src["daily"]["Ret_D1"][c, a["cols"][b]])
                y = src["ret"][min(c + shift, len(src["dates"]) - 1), a["cols"][b]][mem]
                x = ranks[b, r, mem, k_d1]
                ok = ~np.isnan(y)
                if ok.sum() > 50:
                    cc.append(np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])
        corr.append(float(np.mean(cc)))
    ck.add("Ret_D1 on the lookback rows ranks the return of the day before (no look-ahead)",
           corr[0] > 0.99 and abs(corr[1]) < 0.1 and abs(corr[2]) < 0.1,
           f"mean rank correlation with the return of s-1: {corr[0]:.4f}, of s: {corr[1]:+.4f}, of s+1: {corr[2]:+.4f}")

    # ---------------------------------------------------------------- coverage (notes)
    cov = coverage(arr, lookback, a["ret_missing"])
    n = cov["member_stock_days"]
    ck.note(f"signals on the member stock-days from {bw.FIRST_LOSS_DAY}",
            f"{n:,} stock-days. Current rule ({lookback + 1} consecutive days in the universe): {cov['current'] / n:.3%}. "
            f"Decided rule (listed > {lookback} and annual accounts entered): {cov['decided'] / n:.3%}; with accounts from a fiscal year "
            f"ending after listing only: {cov['decided_post'] / n:.3%}. No signal under the decided rule: {cov['no_signal_listing_only']:,} "
            f"listed too recently, {cov['no_signal_accounts_only']:,} without accounts, {cov['no_signal_both']:,} both. Against the "
            f"current rule: {cov['gain']:,} stock-days gain a signal, {cov['loss']:,} lose it")
    ck.note("securities without accounts on every one of their member days (never a signal under the decided rule)",
            "; ".join(f"PERMNO {p}: {d} stock-days" for p, d in cov["never_accounts"].items()) or "none")
    ck.note("lags of the stock-days that gain a signal", f"{cov['new_signal_lag_rows']:,} lag rows, {cov['new_signal_lag_rows_without_return']:,} "
            "of them on a day CRSP has no return (counted as 0, as everywhere)")
    shift = lag_rank_shift(arr, lookback)
    worst = sorted(shift["max"].items(), key=lambda kv: -kv[1])[:4]
    ck.note("lookback rows: ranks over the block's list against the ranks of the day's own universe (members of both)",
            f"{shift['stock_rows_in_both']:,} stock-rows; mean absolute difference {np.mean(list(shift['mean'].values())):.5f} over the 39; "
            f"largest: " + ", ".join(f"{c} {v:.4f}" for c, v in worst)
            + f"; {shift['stock_rows_not_in_own_universe']:,} stock-rows of securities outside the day's own universe (entrants)")

    print(f"Checks finished in {time.time() - t0:.0f} s: {len(ck.failed)} failed.")
    if ck.failed:
        print("FAILED:", "; ".join(ck.failed))
        return False
    if check_only:
        return True

    # ---------------------------------------------------------------- write
    out = history_folder(universe, fill, lookback)
    out.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        np.save(out / f"{name}.npy", arr[name])
    meta = {
        "universe": universe, "fill": fill, "lookback": lookback, "built": datetime.now().isoformat(timespec="seconds"),
        "builder": "afsa.build_history",
        "sources": {str(p.relative_to(WRDS_DIR)).replace("\\", "/"): {"bytes": p.stat().st_size,
                    "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} for p in files},
        "day_arrays_compared": None if day is None else {"folder": data_folder(universe, fill), "built": day.get("built")},
        "n_months": int(M), "n_rows": int(R), "n_slots": width, "first_month": FIRST_MONTH, "last_month": LAST_MONTH,
        "first_day": str(arr["dates"][0]), "last_day": str(arr["dates"][-1]), "first_loss_day": str(bw.FIRST_LOSS_DAY),
        "characteristics": list(CHARACTERISTICS), "levels": list(LEVEL_NAMES),
        "definitions": {
            "block": f"holding month m: its top-500 list sorted by PERMNO; the {lookback} trading days before m's first day, then m's days",
            "member": "the list security has a row in the CRSP daily file that day",
            "inputs": "row s holds what was known at the close of s-1: daily characteristics from s-5..s-1, monthly ones of s's own "
                      "holding month; ranked over the row's members with a value, (rank - 1) / (n - 1) - 0.5, average ranks for ties, "
                      "0 without a value; medians over the same values; rf of s-1",
            "ret": "total return of day s, 0 where missing and for non-members",
            "listed": "trading days with a row in the daily file up to and including s (counted from 1984-01-03)",
            "run": "consecutive trading days in the universe of each day (the list of that day's month) up to and including s",
            "accounts": "an annual report has entered by m (some holding month up to m uses a fiscal year; fiscal rule safe)",
            "accounts_post": "the same, counting only fiscal years ending on or after the security's first row in the daily file",
            "decided_signal": f"member, listed > {lookback} and accounts (applied by the model, not stored)",
        },
        "arrays": {k: {"shape": list(arr[k].shape), "dtype": str(arr[k].dtype)} for k in ARRAYS},
        "coverage": cov, "lookback_rank_shift": shift,
        "checks": ck.items,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    back = all(np.array_equal(np.load(out / f"{name}.npy", mmap_mode="r"), arr[name]) for name in ARRAYS)
    print(f"  [{'PASS' if back else 'FAIL'}] read back: the {len(ARRAYS)} files equal the arrays in memory")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"Wrote {len(ARRAYS)} arrays + meta.json to {out} ({size / 2**30:.2f} GB) in {time.time() - t0:.0f} s.")
    return back


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--universe", nargs="*", default=list(WRDS_UNIVERSES), choices=list(WRDS_BUILDS),
                   help="[default: wrds_broad and wrds_common; a buffered list such as wrds_broad_buffer600 only when named]")
    p.add_argument("--fill", default="none", choices=FILLS, help="missing characteristics (as afsa.build_wrds) [default: none]")
    p.add_argument("--lookback", type=int, default=None, help=f"trading days before each month [default: Config's, {Config().lookback}]")
    p.add_argument("--check-only", action="store_true", help="run the checks without writing anything")
    args = p.parse_args()
    t0 = time.time()
    src = bw.load_sources(args.fill)
    print(f"sources loaded in {time.time() - t0:.0f} s: {len(src['dates']):,} days from {src['dates'][0]}, {len(src['permnos']):,} securities")
    ok = [build(u, args.fill, args.check_only, src, args.lookback) for u in args.universe]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
