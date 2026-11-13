"""Training arrays built from our own WRDS downloads: Data/TrainingData/wrds_<universe>[_fill-<rule>]/.

    python -m afsa.build_wrds                         # wrds_broad and wrds_common, fill none (1 minute each, 5.6 GB of memory)
    python -m afsa.build_wrds --universe wrds_broad   # one universe
    python -m afsa.build_wrds --fill last             # the paper's section 4.1 instead -> wrds_broad\\, wrds_common\\
    python -m afsa.build_wrds --check-only            # run the checks, write nothing
    python -m afsa.build_wrds --compare               # our ranks against Ander's arrays, per characteristic (a bug detector, 3 minutes)

Inputs (nothing of Ander's is used to build):
    Data\\WRDS\\raw\\crsp_index_daily.parquet        the trading calendar (the days of the CRSP market index)
    Data\\WRDS\\raw\\crsp_dsf_v2.parquet             daily return `dlyret` (delisting return included) and volume `dlyvol`
    Data\\WRDS\\raw\\ff_factors_monthly.parquet      the one-month Treasury bill rate
    Data\\WRDS\\universe\\<kind>.parquet             the top-500 list of each holding month (afsa.wrds_universe)
    Data\\WRDS\\characteristics\\market.parquet      13 monthly characteristics by holding month (afsa.chars_market)
    Data\\WRDS\\characteristics\\accounting.parquet  22 monthly characteristics by holding month (afsa.chars_accounting)

The arrays have the layout of afsa.build_data (its docstring lists them) and one difference: they start on
1989-11-01, not 1990-01-02. The 41 trading days of November and December 1989 are the lookback of the first
training window, so that its loss days start on 1990-01-02 (afsa.train.train_span starts them at index
lookback + 1 or on the first day of the first training year, whichever is later). 1989 itself is not a training
year: train_span rejects a window that would start in it.

Row t pairs what was known at the close of day t-1 with the return of day t:

    membership   a security is in the universe on day t if it is on the list of t's month (the 500 largest at the
                 end of the month before) and has a row in the daily file that day. The slots hold the day's
                 members sorted by PERMNO.
    ret          `dlyret` of day t. The row CRSP adds after the last trading day carries the delisting return and
                 counts as a day in the universe. A missing return counts as 0 (listed in meta.json).
    rf           the Fama-French monthly rate spread evenly over the month's SCHEDULED trading days,
                 rf_d = (1 + rf_m) ** (1 / n_m) - 1. WRDS's daily series keeps 4 decimals per day (steps of 2.5% a
                 year, 0 in whole years); the monthly one keeps 4 decimals per month. The rate is that of the bill
                 bought at the end of the previous month, so it is known when the month starts, and so is n_m:
                 the month's trading days plus the closures nobody knew of when it began (UNSCHEDULED_CLOSURES).
                 With the realized number of days, the days before such a closure would carry a rate that could
                 not be computed then: +0.39 basis points a day on the 5 days before 2001-09-11, +0.07 in April
                 1994, +0.02 in June 2004, +0.004 in October 2012. The price: those four months compound to less
                 than the monthly rate (by 5.9, 1.4, 0.4 and 0.1 basis points), every other month to exactly it.
    35 monthly characteristics   those of t's holding month: information up to the last trading day of the
                 month before (the formation date), as the two characteristic modules define them.
    4 daily characteristics      from the five trading days before t, all five required, else no value:
                 Ret_D1   return of day t-1
                 Ret_W1   compounded return of days t-5 .. t-1
                 STD_W1   sample standard deviation (divisor 4) of those five returns
                 Vol      shares traded over those five days (CRSP `dlyvol`, as reported)
                 Ret_W1 and STD_W1 are rounded to 12 decimals (WEEK_DECIMALS): two securities with the same five
                 returns then hold the same value and share a rank, whatever the order of the floating-point
                 operations (unrounded, such values can differ in the last bit and are ranked apart).
    ranks       each characteristic is ranked across the day's members that have a value, (rank - 1) / (n - 1) - 0.5
                 with average ranks for ties; a member without a value gets 0, the middle of the scale. The median
                 in `levels` is taken over the same values. All 39 are ranked over the day's members (Ander's X
                 ranks the monthly ones once a month, over the month's list).
    levels       the 39 medians of the raw values and the rf of day t-1.

Missing values (config.FILLS):
    none   a characteristic that cannot be computed stays missing: rank 0, no imputation, no carry-forward.
           The user's instruction and the default here.
    last   first replaced by the security's last observed value, with no age limit and including values from
           before it entered the universe (the paper's section 4.1); rank 0 if it never had one.

What the first author's companion code does instead (companionpaper\\, read only) is in
analysis\\characteristics\\training_data.md. Everything written here derives from CRSP/Compustat: it stays under
Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .build_data import Checks
from .build_window import DAILY, MONTHLY, ffill, ranks_and_medians
from .config import (CHARACTERISTICS, FILLS, LEVEL_NAMES, N_CHAR, N_LEVEL, N_SLOTS, SOURCE_DIR, TRAIN_DATA_DIR, WRDS_DIR,
                     WRDS_UNIVERSES, Config, data_folder)
from .panel import membership

RAW_DIR = WRDS_DIR / "raw"
UNIVERSE_DIR = WRDS_DIR / "universe"
CHAR_DIR = WRDS_DIR / "characteristics"
MARKET_FILE, ACCOUNTING_FILE = CHAR_DIR / "market.parquet", CHAR_DIR / "accounting.parquet"
FIRST_DAY, LAST_DAY = np.datetime64("1989-11-01"), np.datetime64("2025-12-31")
FIRST_LOSS_DAY = np.datetime64("1990-01-02")
WEEK = 5                        # trading days in the window of the daily characteristics
WEEK_DECIMALS = 12              # Ret_W1 and STD_W1 are rounded to this many decimals, so that exact ties stay ties
MARKET_INDEX = 1000200          # CRSP value-weighted market index: its days are the trading calendar
PAPER_MARKET = (0.0861, 0.2037, 0.42)   # Table 2, equal-weighted market 1998-2021: mean, volatility, Sharpe
ANDER_FLAG = {"wrds_broad": "in_universe", "wrds_common": "in_universe_comun"}   # in Ander\data\crsp_daily.parquet
ANDER_FOLDER = {"last": "broad", "none": "broad_fill-none"}                    # arrays built from Ander's files
# Days on which the exchanges stayed closed although the closure was not known when the month began. spread_rf counts
# them as scheduled trading days: the daily rate of the days before them must not depend on a closure still to come.
# Closures announced before the month's first trading day (2007-01-02, 2018-12-05, 2025-01-09: the days of mourning
# for Presidents Ford, G. H. W. Bush and Carter) are not listed: they were off the schedule when the month started.
# None of these days is in the CRSP calendar (the build checks it).
UNSCHEDULED_CLOSURES = np.array([
    "1985-09-27",                                             # hurricane Gloria (before the arrays start; changes nothing written)
    "1994-04-27",                                             # day of mourning for President Nixon, who died on 1994-04-22
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",   # the attacks of 11 September 2001
    "2004-06-11",                                             # day of mourning for President Reagan, who died on 2004-06-05
    "2012-10-29", "2012-10-30",                               # hurricane Sandy, announced on 2012-10-28 and 29
], dtype="datetime64[D]")


# ---------------------------------------------------------------------------------------------------- pieces
def scheduled_days(dates: np.ndarray) -> np.ndarray:
    """For each day of `dates`, the trading days its month was scheduled to have when it began: the month's days
    in `dates` plus the unscheduled closures of that month (those that are not in `dates`). `dates` must hold
    every trading day of each month it touches."""
    dates = np.asarray(dates).astype("datetime64[D]")
    months, codes, n = np.unique(dates.astype("datetime64[M]"), return_inverse=True, return_counts=True)
    closed = UNSCHEDULED_CLOSURES[~np.isin(UNSCHEDULED_CLOSURES, dates)].astype("datetime64[M]")
    pos = np.searchsorted(months, closed)
    hit = months[np.minimum(pos, len(months) - 1)] == closed                # closures in a month of `dates`
    return (n + np.bincount(pos[hit], minlength=len(months)))[codes]


def spread_rf(dates: np.ndarray, rf_month: pd.Series) -> np.ndarray:
    """Daily risk-free rate from monthly rates: rf_d = (1 + rf_m) ** (1 / n_m) - 1, with n_m the scheduled trading
    days of the month (scheduled_days). The days of a month compound to the monthly rate, except in a month with
    an unscheduled closure, which compounds to (1 + rf_m) ** (days / n_m) - 1: the rate of a day never depends
    on a closure that was still to come. `dates` must hold every trading day of each month it touches; `rf_month`
    is indexed by Period[M]."""
    months = pd.DatetimeIndex(dates).to_period("M")
    rate = rf_month.reindex(months).to_numpy(dtype=np.float64)
    if np.isnan(rate).any():
        raise ValueError(f"no monthly risk-free rate for {months[np.isnan(rate)][0]}")
    return (1.0 + rate) ** (1.0 / scheduled_days(dates)) - 1.0


def rf_by_month(dates: np.ndarray, rf: np.ndarray, rf_month: pd.Series) -> pd.DataFrame:
    """What the daily rates `rf` of `dates` add up to, month by month (index Period[M]): `days` in `dates` and
    `scheduled`, the monthly `rate`, `compounded` = prod(1 + rf_d) - 1, and `target` = (1 + rate) ** (days /
    scheduled) - 1, which is the monthly rate itself unless the month had an unscheduled closure."""
    months = pd.DatetimeIndex(dates).to_period("M")
    starts = np.flatnonzero(np.r_[True, months.asi8[1:] != months.asi8[:-1]])
    out = pd.DataFrame({"days": np.diff(np.r_[starts, len(months)]), "scheduled": scheduled_days(dates)[starts],
                        "rate": rf_month.reindex(months[starts]).to_numpy(dtype=np.float64),
                        "compounded": np.exp(np.add.reduceat(np.log1p(rf), starts)) - 1.0}, index=months[starts])
    out["target"] = (1.0 + out["rate"]) ** (out["days"] / out["scheduled"]) - 1.0
    return out


def check_rf_months(dates: np.ndarray, rf: np.ndarray, rf_month: pd.Series, calendar: np.ndarray) -> tuple[bool, str]:
    """The build's check of the daily rates `rf` of `dates` against the monthly rates. Every month compounds to
    (1 + rf_m) ** (days / scheduled days) - 1: the monthly rate, and less than it only in the months of the
    unscheduled closures that fall inside `dates`; none of the listed closures is a day of `calendar` (the whole
    CRSP calendar); no rate is negative. Returns (passed, the numbers)."""
    bm = rf_by_month(dates, rf, rf_month)
    cut = (bm["scheduled"] > bm["days"]).to_numpy()                             # months with an unscheduled closure
    inside = UNSCHEDULED_CLOSURES[(UNSCHEDULED_CLOSURES >= dates[0]) & (UNSCHEDULED_CLOSURES <= dates[-1])]
    exact = np.abs(bm["compounded"] - bm["rate"]).to_numpy()[~cut]
    ok = bool(np.abs(bm["compounded"] - bm["target"]).max() < 1e-12 and exact.max(initial=0.0) < 1e-12
              and (bm["compounded"][cut] <= bm["rate"][cut]).all() and np.min(rf) >= 0
              and not np.isin(UNSCHEDULED_CLOSURES, calendar).any()
              and bm.index[cut].equals(pd.PeriodIndex(inside.astype("datetime64[M]"), freq="M").unique())
              and int((bm["scheduled"] - bm["days"]).sum()) == len(inside))
    return ok, (f"largest difference {exact.max(initial=0.0):.1e} over the {int((~cut).sum())} months without one; "
                + "; ".join(f"{m}: {r.days:.0f} of {r.scheduled:.0f} days, {1e4 * (r.compounded - r.rate):+.2f} bp" for m, r in bm[cut].iterrows()))


def daily_characteristics(ret: np.ndarray, vol: np.ndarray, window: int = WEEK,
                          decimals: int | None = WEEK_DECIMALS) -> dict[str, np.ndarray]:
    """The 4 daily characteristics of each day t from rows t-window .. t-1 of day x security matrices of returns
    and share volume (NaN = no row or no value that day). Row t never uses row t or later. A value needs all
    `window` days; the first `window` rows have none (Ret_D1: the first row).

    Ret_W1 and STD_W1 are rounded to `decimals` (None = as computed). A product or a sum of the same five returns
    taken in another order can differ in the last bit (about 1e-16); rounded, values that are equal in exact
    arithmetic are equal and share a rank. The volume is a sum of share counts and is exact."""
    out = {c: np.full(ret.shape, np.nan) for c in DAILY}
    out["Ret_D1"][1:] = ret[:-1]
    if len(ret) > window:
        win = np.lib.stride_tricks.sliding_window_view(ret, window, axis=0)[:-1]          # window k = rows k .. k+window-1
        out["Ret_W1"][window:] = np.prod(1.0 + win, axis=2) - 1.0
        out["STD_W1"][window:] = win.std(axis=2, ddof=1)
        if decimals is not None:
            for c in ("Ret_W1", "STD_W1"):
                np.round(out[c], decimals, out=out[c])
        out["Vol"][window:] = np.lib.stride_tricks.sliding_window_view(vol, window, axis=0)[:-1].sum(axis=2)
    return out


def _month_index(months) -> np.ndarray:
    """Months as integers (year * 12 + month - 1) from a Period[M] series or index."""
    p = pd.PeriodIndex(months, freq="M")
    return np.asarray(p.year * 12 + p.month - 1, dtype=np.int64)


def load_sources(fill: str = "none") -> dict:
    """The raw files as dense matrices over the whole daily file (calendar from 1984, every security of the
    download), with missing characteristics treated by `fill`."""
    from .chars_accounting import CHARS as ACCOUNTING_CHARS
    from .chars_market import CHARS as MARKET_CHARS
    from .wrds_data import load

    if fill not in FILLS:
        raise ValueError(f"unknown fill rule {fill!r}; choose from {FILLS}")
    if set(MARKET_CHARS) | set(ACCOUNTING_CHARS) != set(MONTHLY) or len(MARKET_CHARS) + len(ACCOUNTING_CHARS) != len(MONTHLY):
        raise ValueError("the market and accounting modules do not hold exactly the 35 monthly characteristics")
    carry = ffill if fill == "last" else (lambda a: a)

    idx = load("crsp_index_daily", columns=["indno", "dlycaldt"])
    dates = np.sort(idx.loc[idx["indno"] == MARKET_INDEX, "dlycaldt"].to_numpy().astype("datetime64[D]"))
    dsf = load("crsp_dsf_v2", columns=["permno", "dlycaldt", "dlyret", "dlyvol", "dlydelflg"])
    day = dsf["dlycaldt"].to_numpy().astype("datetime64[D]")
    permnos = np.unique(dsf["permno"].to_numpy()).astype(np.int32)
    row, col = np.searchsorted(dates, day), np.searchsorted(permnos, dsf["permno"].to_numpy())
    if not np.array_equal(dates[np.minimum(row, len(dates) - 1)], day):
        raise ValueError("crsp_dsf_v2 has a day that is not in the calendar of the CRSP market index")
    shape = (len(dates), len(permnos))
    R, V = np.full(shape, np.nan), np.full(shape, np.nan)
    R[row, col] = dsf["dlyret"].to_numpy(dtype=np.float64)
    V[row, col] = dsf["dlyvol"].to_numpy(dtype=np.float64)
    has_row, delist = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    has_row[row, col] = True
    delist[row, col] = (dsf["dlydelflg"].astype(str) == "Y").to_numpy()
    if int(has_row.sum()) != len(dsf):
        raise ValueError("crsp_dsf_v2 has duplicate (permno, day) rows")
    del dsf, day, row, col

    ffm = load("ff_factors_monthly", columns=["date", "rf"])
    rf_month = pd.Series(ffm["rf"].to_numpy(dtype=np.float64), index=pd.PeriodIndex(ffm["date"], freq="M"))
    daily = {c: carry(v) for c, v in daily_characteristics(R, V).items()}

    # the 35 monthly characteristics as holding month x security matrices
    m0 = int(dates[0].astype("datetime64[Y]").astype(int) + 1970) * 12                   # January of the first calendar year
    month_of_day = _month_index(pd.DatetimeIndex(dates).to_period("M")) - m0
    n_months = int(month_of_day.max()) + 1
    monthly = {}
    for path, names in ((MARKET_FILE, MARKET_CHARS), (ACCOUNTING_FILE, ACCOUNTING_CHARS)):
        df = pd.read_parquet(path, columns=["permno", "month", *names])
        mi, ci = _month_index(df["month"]) - m0, np.searchsorted(permnos, df["permno"].to_numpy())
        if not np.array_equal(permnos[np.minimum(ci, len(permnos) - 1)], df["permno"].to_numpy()):
            raise ValueError(f"{path.name} has a PERMNO that is not in crsp_dsf_v2")
        if mi.min() < 0 or df.duplicated(["permno", "month"]).any():
            raise ValueError(f"{path.name}: months before the daily file, or duplicate (permno, month) rows")
        keep = mi < n_months
        for c in names:
            m = np.full((n_months, len(permnos)), np.nan)
            m[mi[keep], ci[keep]] = df[c].to_numpy(dtype=np.float64)[keep]
            monthly[c] = carry(m)
    return {"fill": fill, "dates": dates, "years": dates.astype("datetime64[Y]").astype(int) + 1970, "permnos": permnos,
            "ret": R, "has_row": has_row, "delist": delist, "rf": spread_rf(dates, rf_month), "daily": daily,
            "monthly": monthly, "month_of_day": month_of_day, "month0": m0}


def universe_lists(universe: str, src: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """on_list [months, securities]: the security is on the top-500 list of that holding month. Also the list file."""
    u = pd.read_parquet(UNIVERSE_DIR / f"{universe.removeprefix('wrds_')}.parquet", columns=["month", "formation_date", "permno"])
    mi, ci = _month_index(u["month"]) - src["month0"], np.searchsorted(src["permnos"], u["permno"].to_numpy(dtype=np.int64))
    if not np.array_equal(src["permnos"][np.minimum(ci, len(src["permnos"]) - 1)], u["permno"].to_numpy(dtype=np.int64)):
        raise ValueError("a PERMNO of the universe lists is not in crsp_dsf_v2")
    on_list = np.zeros((int(src["month_of_day"].max()) + 1, len(src["permnos"])), dtype=bool)
    on_list[mi, ci] = True
    return on_list, u


def assemble(src: dict, universe: str) -> dict:
    """The arrays of one universe (see afsa.build_data), plus what the checks need:
    days (calendar rows of the array days), member [T, S], listed [T, S] (on the list of the day's month, with a
    daily row or not), day, slot, col of every stock-day, n_value [T, 39] (members with a value) and u (the list file)."""
    dates = src["dates"]
    days = np.flatnonzero((dates >= FIRST_DAY) & (dates <= LAST_DAY))
    month = src["month_of_day"][days]
    on_list, u = universe_lists(universe, src)
    member = on_list[month] & src["has_row"][days]
    n_per_day = member.sum(axis=1)
    if n_per_day.max() > N_SLOTS:
        raise ValueError(f"{n_per_day.max()} securities in the universe on one day: more than {N_SLOTS} slots")
    day, col = np.nonzero(member)                                              # by day, then PERMNO
    slot = np.arange(len(day)) - np.concatenate([[0], np.cumsum(n_per_day)[:-1]])[day]
    T = len(days)
    permno = np.full((T, N_SLOTS), -1, dtype=np.int32)
    permno[day, slot] = src["permnos"][col]
    R = src["ret"][days]
    ret = np.zeros((T, N_SLOTS), dtype=np.float32)
    ret[day, slot] = np.nan_to_num(R[day, col], nan=0.0)

    ranks = np.zeros((T, N_SLOTS, N_CHAR), dtype=np.float32)
    levels = np.zeros((T, N_LEVEL), dtype=np.float32)
    n_value = np.zeros((T, N_CHAR), dtype=np.int32)
    for k, c in enumerate(CHARACTERISTICS):
        raw = src["daily"][c][days] if c in DAILY else src["monthly"][c][month]
        v = np.where(member, raw, np.nan)
        r, levels[:, k] = ranks_and_medians(v)
        ranks[day, slot, k] = r[day, col]
        n_value[:, k] = (~np.isnan(v)).sum(axis=1)
    levels[:, N_CHAR] = src["rf"][days - 1]                                    # rf of the previous trading day
    permnos, slot_of, run = membership(permno)
    arrays = {"dates": dates[days], "permnos": permnos, "permno": permno, "slot_of": slot_of, "ranks": ranks, "levels": levels,
              "ret": ret, "rf": src["rf"][days], "run": run}
    return {"arrays": arrays, "days": days, "member": member, "listed": on_list[month], "day": day, "slot": slot, "col": col,
            "n_value": n_value, "u": u}


# ---------------------------------------------------------------------------------------------------- checks
def list_days_without_a_row(src: dict, a: dict) -> dict[str, int]:
    """The list stock-days that are not in the universe because the security has no row in the daily file that
    day, by where the day falls in the security's life in the file: `after` its last row (it was delisted),
    `before` its first row, or `inside` (a gap between two rows)."""
    d, c = np.nonzero(a["listed"] & ~a["member"])
    has = src["has_row"]
    first, last = has.argmax(axis=0), len(has) - 1 - has[::-1].argmax(axis=0)   # first and last calendar row of each security
    row = a["days"][d]
    return {"total": len(d), "after": int((row > last[c]).sum()), "before": int((row < first[c]).sum()),
            "inside": int(((row > first[c]) & (row < last[c])).sum())}


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation of each row of a with the same row of b over the cells where both have a value."""
    ok = ~(np.isnan(a) | np.isnan(b))
    n = np.maximum(ok.sum(axis=1), 1)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    a0, b0 = a0 - np.where(ok, (a0.sum(1) / n)[:, None], 0.0), b0 - np.where(ok, (b0.sum(1) / n)[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a0 * b0).sum(1) / np.sqrt((a0 * a0).sum(1) * (b0 * b0).sum(1))


def direct_inputs(universe: str, fill: str, cal: np.ndarray, sample: np.ndarray) -> dict[np.datetime64, dict]:
    """The inputs of a few days computed another way, with pandas from the parquet files: for each calendar row
    in `sample`, the members, their returns, the 39 ranks and medians. Under fill = last the daily characteristics
    of a day are left out (NaN) when a member has none, because the carried value is not recomputed here."""
    need = sorted({cal[i - j] for i in sample for j in range(WEEK + 1)})
    dsf = pd.read_parquet(RAW_DIR / "crsp_dsf_v2.parquet", columns=["permno", "dlycaldt", "dlyret", "dlyvol"],
                          filters=[("dlycaldt", "in", [pd.Timestamp(d) for d in need])])
    dsf["day"] = dsf["dlycaldt"].to_numpy().astype("datetime64[D]")
    u = pd.read_parquet(UNIVERSE_DIR / f"{universe.removeprefix('wrds_')}.parquet", columns=["month", "permno"])
    u["month"] = pd.PeriodIndex(u["month"], freq="M")
    monthly = pd.read_parquet(MARKET_FILE).drop(columns=["formation_date"]).merge(
        pd.read_parquet(ACCOUNTING_FILE), on=["permno", "month"], how="outer")[["permno", "month", *MONTHLY]]
    out = {}
    for i in sample:
        d, m = cal[i], pd.Timestamp(cal[i]).to_period("M")
        today = dsf[dsf["day"] == d].set_index("permno")
        mem = np.sort(np.intersect1d(today.index.to_numpy(), u.loc[u["month"] == m, "permno"].to_numpy(dtype=np.int64)))
        prev = [cal[i - j] for j in range(WEEK, 0, -1)]
        w = dsf[dsf["day"].isin(prev)]
        wr = w.pivot(index="day", columns="permno", values="dlyret").reindex(index=prev, columns=mem)
        wv = w.pivot(index="day", columns="permno", values="dlyvol").reindex(index=prev, columns=mem)
        x = pd.DataFrame({"Ret_D1": wr.iloc[-1], "Ret_W1": ((1 + wr).prod(skipna=False) - 1).round(WEEK_DECIMALS),
                          "STD_W1": wr.std(ddof=1, skipna=False).round(WEEK_DECIMALS), "Vol": wv.sum(skipna=False)})
        if fill == "last":
            mv = monthly[monthly["month"] <= m].sort_values("month").groupby("permno")[list(MONTHLY)].last()   # last non-missing
            if x.isna().any().any():
                x[:] = np.nan
        else:
            mv = monthly[monthly["month"] == m].set_index("permno")[list(MONTHLY)]
        x = x.join(mv.reindex(mem))[list(CHARACTERISTICS)]
        n = x.notna().sum()
        ranks = ((x.rank(method="average") - 1) / (n - 1).clip(lower=1) - 0.5).fillna(0.0)
        out[d] = {"permno": mem, "ret": today["dlyret"].reindex(mem).fillna(0.0).to_numpy(), "ranks": ranks.to_numpy(),
                  "median": x.median().to_numpy(), "skipped": x.isna().all().to_numpy()}
    return out


def month_constancy(a: dict, src: dict) -> dict:
    """How much a stock's rank of a monthly characteristic moves within a month. The raw value is fixed for the
    month, so the rank can move only because the day's members change: by at most c / (n - c - 1), with c the
    securities of the month's list that are in the universe on some days of the month but not all, and n the
    fewest members with a value on a day of the month."""
    S = len(src["permnos"])
    month_of_day = src["month_of_day"][a["days"]]
    key = month_of_day[a["day"]] * S + a["col"]                                # the stock-month of each stock-day
    order = np.argsort(key, kind="stable")
    starts = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
    group_month = key[order][starts] // S
    m_lo = int(month_of_day.min())
    n_months = int(month_of_day.max()) - m_lo + 1
    partial = np.diff(np.r_[starts, len(order)]) < np.bincount(month_of_day - m_lo, minlength=n_months)[group_month - m_lo]
    c = np.bincount(group_month - m_lo, weights=partial, minlength=n_months)   # securities with a broken month
    day_starts = np.flatnonzero(np.r_[True, month_of_day[1:] != month_of_day[:-1]])
    worst, over, n_big, n_all, quiet_worst = 0.0, 0, 0, 0, 0.0
    ranks = a["arrays"]["ranks"]
    for k, name in enumerate(CHARACTERISTICS):
        if name in DAILY:
            continue
        x = ranks[a["day"], a["slot"], k][order]
        rng = np.maximum.reduceat(x, starts) - np.minimum.reduceat(x, starts)
        by_month = np.zeros(n_months)
        np.maximum.at(by_month, group_month - m_lo, rng)
        n_min = np.minimum.reduceat(a["n_value"][:, k], day_starts)
        bound = c / np.maximum(n_min - c - 1, 1)
        over += int((by_month > bound + 1e-6).sum())
        worst, quiet_worst = max(worst, float(rng.max())), max(quiet_worst, float(by_month[c == 0].max(initial=0.0)))
        n_big, n_all = n_big + int((rng > 0.01).sum()), n_all + len(rng)
    return {"months": n_months, "quiet_months": int((c == 0).sum()), "quiet_worst": quiet_worst, "worst": worst,
            "over_bound": over, "share_over_0.01": n_big / n_all}


def build(universe: str = "wrds_broad", fill: str = "none", check_only: bool = False, src: dict | None = None) -> bool:
    t0 = time.time()
    if universe not in WRDS_UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}; choose from {WRDS_UNIVERSES}")
    ck = Checks()
    files = [RAW_DIR / "crsp_index_daily.parquet", RAW_DIR / "crsp_dsf_v2.parquet", RAW_DIR / "ff_factors_monthly.parquet",
             RAW_DIR / "ff_factors_daily.parquet", UNIVERSE_DIR / f"{universe.removeprefix('wrds_')}.parquet", MARKET_FILE, ACCOUNTING_FILE]
    print(f"Building the {universe!r} arrays with fill = {fill} from {WRDS_DIR}")
    if not ck.add("source files exist", all(p.exists() for p in files), ", ".join(p.name for p in files)):
        return False
    src = src if src is not None and src["fill"] == fill else load_sources(fill)
    a = assemble(src, universe)
    arr, days, member, day, slot, col = a["arrays"], a["days"], a["member"], a["day"], a["slot"], a["col"]
    dates, permno, ranks, levels, ret, rf, run = (arr[k] for k in ("dates", "permno", "ranks", "levels", "ret", "rf", "run"))
    T, n_rows, n_per_day = len(dates), len(day), member.sum(axis=1)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    lookback = Config().lookback

    # ---------------------------------------------------------------- calendar and lookback
    ck.add("calendar runs from 1989-11-01 to 2025-12-31, sorted, without gaps in the index file",
           dates[0] == FIRST_DAY and dates[-1] == LAST_DAY and bool(np.all(np.diff(dates) > np.timedelta64(0, "D"))),
           f"{dates[0]} .. {dates[-1]}, {T:,} trading days, {int((years >= 1990).sum()):,} from 1990")
    start = max(int(np.searchsorted(years, 1990)), lookback + 1)                # afsa.train.train_span for the first window
    ck.add("the lookback days exist: loss days of the first window start on 1990-01-02",
           dates[start] == FIRST_LOSS_DAY and start - lookback - 1 >= 0,
           f"{start} trading days before {dates[start]} (the model needs {lookback + 1}); "
           f"{(run[start][permno[start] >= 0] > lookback).mean():.1%} of that day's members have {lookback} prior days in the universe")

    # ---------------------------------------------------------------- universe lists and membership
    u = a["u"]
    um = pd.PeriodIndex(u["month"], freq="M")
    per_month = u.groupby(um).size()
    cal_months = pd.DatetimeIndex(dates).to_period("M")
    last_day_before = pd.Series(src["dates"][days - 1], index=cal_months).groupby(level=0).first()   # day before each month's first day
    formation = u.groupby(um)["formation_date"].first()
    same_formation = formation.reindex(last_day_before.index).to_numpy().astype("datetime64[D]") == last_day_before.to_numpy().astype("datetime64[D]")
    ck.add("a list of 500 for every month of the calendar, formed on the last trading day of the month before",
           bool((per_month.reindex(cal_months.unique()) == N_SLOTS).all() and same_formation.all()),
           f"{len(per_month)} monthly lists, {um.min()} .. {um.max()}")
    ck.add("at most 500 stocks per day", n_per_day.max() <= N_SLOTS,
           f"min {n_per_day.min()}, mean {n_per_day.mean():.1f}, max {n_per_day.max()}; {n_rows:,} stock-days, "
           f"{int(n_per_day[years >= 1990].sum()):,} from 1990")
    listed_days = int((np.repeat(per_month.reindex(cal_months.unique()).to_numpy(), np.unique(cal_months.asi8, return_counts=True)[1])).sum())
    gone = list_days_without_a_row(src, a)
    ck.note("list securities without a row in the daily file that day (not in the universe)",
            f"{listed_days - n_rows:,} of {listed_days:,} list stock-days: {gone['after']:,} after the security's last row (it was delisted), "
            f"{gone['inside']:,} inside its life, {gone['before']:,} before its first row")
    sorted_slots = bool((np.diff(np.where(permno >= 0, permno, np.iinfo(np.int32).max), axis=1) >= 0).all()
                        and ((permno >= 0).sum(1) == n_per_day).all())
    ck.add("slots hold the day's members sorted by PERMNO, filled from slot 0", sorted_slots, f"{len(arr['permnos']):,} PERMNOs are in the universe at some point")
    flag = ANDER_FLAG[universe]
    if (SOURCE_DIR / "crsp_daily.parquet").exists():
        theirs = pd.read_parquet(SOURCE_DIR / "crsp_daily.parquet", columns=["permno", "date"], filters=[(flag, "==", True)])
        from90 = dates[day] >= FIRST_LOSS_DAY
        ours = pd.MultiIndex.from_arrays([dates[day][from90], src["permnos"][col][from90]])
        theirs = pd.MultiIndex.from_arrays([theirs["date"].to_numpy().astype("datetime64[D]"), theirs["permno"].to_numpy().astype(np.int32)])
        ck.note(f"stock-days 1990-2025 against Ander's crsp_daily ({flag})",
                f"ours {len(ours):,}, his {len(theirs):,}; ours only {len(ours.difference(theirs)):,}, his only {len(theirs.difference(ours)):,}")
    ck.note("stock-days with fewer than 30 prior days in the universe (no signal)",
            f"{(run[day, slot] <= lookback).mean():.2%} of all stock-days; {(run[day, slot][day >= start] <= lookback).mean():.2%} from {dates[start]}")

    # ---------------------------------------------------------------- returns
    R = src["ret"][days]
    r_row = R[day, col]
    nan_ret = np.isnan(r_row)
    ck.note("missing returns, counted as 0 (the stock stays in the universe)", f"{int(nan_ret.sum())} stock-days")
    ck.note("delisting rows in the universe (the day after the last trade; they carry the delisting return)",
            f"{int(src['delist'][days][day, col].sum())} stock-days, {int(np.isnan(r_row[src['delist'][days][day, col]]).sum())} of them without a return")
    ck.add("returns are plausible", bool(np.nanmin(r_row) > -1.0 and np.nanmax(r_row) < 5.0),
           f"min {np.nanmin(r_row):.3f}, max {np.nanmax(r_row):.3f}, {int((np.abs(r_row) > 0.5).sum())} with |ret| > 50%")

    # ---------------------------------------------------------------- risk-free rate
    from .wrds_data import load
    ffd = load("ff_factors_daily", columns=["date", "rf"])
    fd = ffd["date"].to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(fd, dates)
    rf_ff = ffd["rf"].to_numpy(dtype=np.float64)[np.minimum(pos, len(fd) - 1)]
    ffm = load("ff_factors_monthly", columns=["date", "rf"])
    ck.add("rf: the days of each month compound to the Fama-French monthly rate, and to less of it in the months "
           "with an unscheduled closure (the rate is spread over the scheduled days); never negative",
           *check_rf_months(dates, rf, pd.Series(ffm["rf"].to_numpy(dtype=np.float64), index=pd.PeriodIndex(ffm["date"], freq="M")), src["dates"]))
    same_days = bool(np.array_equal(fd[np.minimum(pos, len(fd) - 1)], dates))
    by_year = pd.DataFrame({"ours": rf, "ff": rf_ff}).groupby(years).sum()
    ck.add("rf agrees with the Fama-French daily series up to that series' rounding (4 decimals a day)",
           same_days and bool(np.abs(rf - rf_ff).max() <= 1.0e-4) and float(np.corrcoef(rf, rf_ff)[0, 1]) > 0.95,
           f"largest daily difference {np.abs(rf - rf_ff).max():.1e}; ours rounded to 4 decimals equals it on {(np.round(rf, 4) == np.round(rf_ff, 4)).mean():.1%} of days; "
           f"correlation {np.corrcoef(rf, rf_ff)[0, 1]:.3f}; mean per year {rf.mean() * 252:.2%} against {rf_ff.mean() * 252:.2%}; "
           f"the daily series takes {len(np.unique(rf_ff))} values and is 0 on {(rf_ff == 0).mean():.1%} of days, ours takes {len(np.unique(rf))} and is 0 on {(rf == 0).mean():.1%}; "
           f"largest gap between the two in a calendar year {np.abs(by_year['ours'] - by_year['ff']).max():.2%}")
    ck.add("X's rf is the risk-free rate of the previous trading day",
           bool(np.array_equal(levels[1:, -1], rf[:-1].astype(np.float32)) and levels[0, -1] == np.float32(src["rf"][days[0] - 1])),
           f"first row: the rate of {src['dates'][days[0] - 1]}")

    # ---------------------------------------------------------------- the 79 inputs
    shapes = {"dates": ((T,), "datetime64[D]"), "permno": ((T, N_SLOTS), "int32"), "slot_of": ((T, len(arr["permnos"])), "int16"),
              "ranks": ((T, N_SLOTS, N_CHAR), "float32"), "levels": ((T, N_LEVEL), "float32"), "ret": ((T, N_SLOTS), "float32"),
              "rf": ((T,), "float64"), "run": ((T, N_SLOTS), "int16"), "permnos": ((len(arr["permnos"]),), "int32")}
    ck.add("array shapes and dtypes are those of afsa.build_data", all(arr[k].shape == s and str(arr[k].dtype) == d for k, (s, d) in shapes.items()),
           f"ranks {list(ranks.shape)}, levels {list(levels.shape)}")
    ck.add("no missing value in any array", bool(np.isfinite(ranks).all() and np.isfinite(levels).all() and np.isfinite(ret).all() and np.isfinite(rf).all()))
    ck.add("ranks lie in [-0.5, 0.5]; empty slots hold 0", float(ranks.min()) >= -0.5 and float(ranks.max()) <= 0.5 and not ranks[permno < 0].any(),
           f"min {ranks.min():.4f}, max {ranks.max():.4f}")
    rank_mean = float(np.abs(ranks.sum(axis=1, dtype=np.float64) / n_per_day[:, None]).max())
    ck.add("ranks are centred on zero each day", rank_mean < 1e-4, f"largest |cross-sectional mean| {rank_mean:.1e}")
    n_value = a["n_value"]
    fewest = n_value.min(axis=0)
    ck.add("every characteristic has a value for at least 100 members on every day", bool(fewest.min() >= 100),
           "fewest: " + ", ".join(f"{CHARACTERISTICS[k]} {fewest[k]}" for k in np.argsort(fewest)[:4]))
    rank0 = {c: float(1 - n_value[:, k].sum() / n_rows) for k, c in enumerate(CHARACTERISTICS)}
    zero_exact = {c: float((ranks[day, slot, k] == 0).mean()) for k, c in enumerate(CHARACTERISTICS)}
    ck.note(f"fill = {fill}: stock-days without a value (rank 0)",
            "; ".join(f"{c} {s:.2%}" for c, s in sorted(rank0.items(), key=lambda t: -t[1]) if s >= 0.00005))

    # medians and ranks of sample days, recomputed with pandas from the parquet files
    rng = np.random.default_rng(0)
    sample = np.unique(np.r_[0, start, T - 1, rng.choice(T, size=9, replace=False)])
    direct = direct_inputs(universe, fill, src["dates"], days[sample])
    w_rank = w_med = w_ret = 0.0
    same_members, n_checked, n_skipped = True, 0, 0
    for t in sample:
        d = direct[dates[t]]
        n = len(d["permno"])
        if n != n_per_day[t] or not np.array_equal(permno[t, :n], d["permno"]):
            same_members = False
            continue
        keep = ~d["skipped"]
        w_rank = max(w_rank, float(np.abs(ranks[t, :n][:, keep] - d["ranks"][:, keep]).max()))
        med = d["median"][keep].astype(np.float32)
        w_med = max(w_med, float((np.abs(levels[t, :N_CHAR][keep] - med) / np.maximum(np.abs(med), 1e-30)).max()))
        w_ret = max(w_ret, float(np.abs(ret[t, :n] - d["ret"].astype(np.float32)).max()))
        n_checked, n_skipped = n_checked + n, n_skipped + int((~keep).sum())
    ck.add("members, returns, ranks and medians of sample days equal a direct recomputation from the parquet files",
           same_members and w_rank < 2e-6 and w_med < 1e-6 and w_ret == 0.0,
           f"{len(sample)} days, {n_checked:,} stock-days: largest difference in a rank {w_rank:.1e}, in a median (relative) {w_med:.1e}, in a return {w_ret:g}"
           + (f"; {n_skipped} day-characteristics left out (carried daily values)" if n_skipped else ""))

    # ---------------------------------------------------------------- no look-ahead: Ret_D1 is yesterday's return
    k_d1 = CHARACTERISTICS.index("Ret_D1")
    mine = np.full(member.shape, np.nan)
    has = member & ~np.isnan(src["daily"]["Ret_D1"][days])
    mine[day, col] = ranks[day, slot, k_d1]
    mine[~has] = np.nan
    corr = []
    for shift in (-1, 0, 1):
        other = src["ret"][np.minimum(days + shift, len(src["dates"]) - 1)]
        corr.append(float(np.nanmean(_row_corr(mine, np.where(has, ranks_and_medians(np.where(has, other, np.nan))[0], np.nan)))))
    ck.add("Ret_D1 on day t ranks the return of day t-1 (no look-ahead)", corr[0] > 0.99 and abs(corr[1]) < 0.1 and abs(corr[2]) < 0.1,
           f"mean daily rank correlation with the return of t-1: {corr[0]:.4f}, of t: {corr[1]:+.4f}, of t+1: {corr[2]:+.4f}")
    del mine, has

    # ---------------------------------------------------------------- monthly characteristics within a month
    mc = month_constancy(a, src)
    ck.add("a monthly characteristic's rank is constant within the month, up to re-ranking when the day's members change",
           mc["over_bound"] == 0 and mc["quiet_worst"] <= 1e-6,
           f"{mc['quiet_months']} of {mc['months']} months have the same members every day: largest move within such a month {mc['quiet_worst']:.1e}; "
           f"in the others the largest move is {mc['worst']:.4f}, never above the bound from the members that changed; "
           f"{mc['share_over_0.01']:.3%} of stock-month-characteristics move by more than 0.01")

    # ---------------------------------------------------------------- reference numbers
    exret = np.where(permno >= 0, ret - rf[:, None].astype(np.float32), 0.0)
    ew = exret.sum(axis=1) / n_per_day
    oos = (years >= 1998) & (years <= 2021)
    mu, sd = ew[oos].mean() * 252, ew[oos].std() * np.sqrt(252)
    ck.add("equal-weighted universe, excess return, 1998-2021: near the paper's market row",
           abs(mu - PAPER_MARKET[0]) < 0.03 and abs(sd - PAPER_MARKET[1]) < 0.02,
           f"mean {mu:.2%}, volatility {sd:.2%}, Sharpe {mu / sd:.2f} (paper: {PAPER_MARKET[0]:.2%}, {PAPER_MARKET[1]:.2%}, {PAPER_MARKET[2]:.2f})")
    days_per_year = {int(y): int(c) for y, c in zip(*np.unique(years, return_counts=True))}
    ck.note("trading days in the first training window (1990-1997)", f"{sum(days_per_year[y] for y in range(1990, 1998)):,}, all of them loss days")

    print(f"Checks finished in {time.time() - t0:.0f} s: {len(ck.failed)} failed.")
    if ck.failed:
        print("FAILED:", "; ".join(ck.failed))
        return False
    if check_only:
        return True

    # ---------------------------------------------------------------- write
    out = TRAIN_DATA_DIR / data_folder(universe, fill)
    out.mkdir(parents=True, exist_ok=True)
    for name, x in arr.items():
        np.save(out / f"{name}.npy", x)
    meta = {
        "universe": universe, "fill": fill, "built": datetime.now().isoformat(timespec="seconds"), "builder": "afsa.build_wrds",
        "sources": {str(p.relative_to(WRDS_DIR)).replace("\\", "/"): {"bytes": p.stat().st_size,
                    "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} for p in files},
        "n_days": int(T), "n_slots": N_SLOTS, "n_stocks": int(len(arr["permnos"])), "n_stock_days": int(n_rows),
        "n_stock_days_from_1990": int(n_per_day[years >= 1990].sum()),
        "first_day": str(dates[0]), "last_day": str(dates[-1]), "first_loss_day": str(dates[start]), "lookback_days_before_1990": int(start),
        "characteristics": list(CHARACTERISTICS), "levels": list(LEVEL_NAMES),
        "daily_characteristics": {"Ret_D1": "return of day t-1", "Ret_W1": f"compounded return of days t-{WEEK}..t-1",
                                  "STD_W1": f"sample standard deviation (divisor {WEEK - 1}) of the returns of days t-{WEEK}..t-1",
                                  "Vol": f"shares traded (dlyvol) on days t-{WEEK}..t-1",
                                  "rule": f"all {WEEK} trading days needed, else no value",
                                  "rounding": f"Ret_W1 and STD_W1 to {WEEK_DECIMALS} decimals, so that exact ties share a rank"},
        "rf": "Fama-French monthly rf spread over the month's scheduled trading days (its trading days plus its "
              "unscheduled closures): (1 + rf_m) ** (1 / n_m) - 1",
        "unscheduled_closures": [str(d) for d in UNSCHEDULED_CLOSURES],
        "ranking": "(rank - 1) / (n - 1) - 0.5 over the day's members with a value, average ranks for ties; 0 without a value",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arr.items()},
        "days_per_year": days_per_year,
        "rank0_share": rank0,
        "rank_exactly_zero_share": zero_exact,
        "missing_returns_counted_as_zero": [[str(d), int(p)] for d, p in zip(dates[day][nan_ret], src["permnos"][col][nan_ret])],
        "checks": ck.items,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"Wrote {len(arr)} arrays + meta.json to {out} ({size / 2**20:.0f} MB).")
    ok = verify(universe, fill, arr)
    print(f"Done in {time.time() - t0:.0f} s.")
    return ok


def verify(universe: str = "wrds_broad", fill: str = "none", arrays: dict | None = None) -> bool:
    """Re-reads the written arrays, compares them with those in memory if given, and loads them as a Panel on
    the CPU: the loss days of a window that starts in 1990 must start on 1990-01-02 and a batch must come out."""
    import torch

    from .panel import ARRAYS, Panel
    from .train import Task, train_span

    folder = data_folder(universe, fill)
    ok = True
    if arrays is not None:
        ok = all(np.array_equal(np.load(TRAIN_DATA_DIR / folder / f"{name}.npy", mmap_mode="r"), arrays[name]) for name in ARRAYS)
        print(f"  [{'PASS' if ok else 'FAIL'}] verification: the {len(ARRAYS)} files read back equal the arrays in memory")
    cfg = Config(universe=universe, fill=fill, train_years=1)
    panel = Panel(folder, "cpu", first_year=1990, last_year=1990)
    start, stop = train_span(cfg, panel, Task(1991, 0))
    L = cfg.lookback
    b = panel.batch(torch.arange(start - L - 1, start + 5)[None, :], L)
    good = (panel.dates[start] == FIRST_LOSS_DAY and bool(b.valid[0, L + 1].any()) and bool(torch.isfinite(b.X).all())
            and b.X.shape[-1] == N_CHAR and b.levels.shape[-1] == N_LEVEL)
    print(f"  [{'PASS' if good else 'FAIL'}] verification: Panel('{folder}') loads on the CPU; train_span starts the loss days of 1990 on "
          f"{panel.dates[start]} ({stop - start} days); a batch of {b.X.shape[1]} days x {b.X.shape[2]} stocks has "
          f"{int(b.valid[0, L + 1].sum())} stocks with a signal on that day")
    return ok and good


# ---------------------------------------------------------------------------------------------------- Ander's arrays
def compare(universe: str = "wrds_broad") -> pd.DataFrame:
    """Per characteristic, our ranks against the arrays built from Ander's files, 1990-2025: fill = last against
    Data/TrainingData/broad (his X as is) and fill = none against broad_fill-none. A bug detector, not a target:
    the rank correlation over all stock-days, the share of ranks within 0.01, and the mean absolute difference.
    For the monthly characteristics also the raw values against his characteristics.parquet on universe
    stock-months (share equal as afsa.pelger_check defines it), which separates the definitions from the ranking."""
    from scipy.stats import spearmanr

    if universe != "wrds_broad":
        raise ValueError("Ander's arrays on disk are those of the broad universe")
    table = {}
    for fill in ("last", "none"):
        src = load_sources(fill)
        a = assemble(src, universe)
        arr = a["arrays"]
        theirs = TRAIN_DATA_DIR / ANDER_FOLDER[fill]
        t_dates, t_permno = np.load(theirs / "dates.npy"), np.load(theirs / "permno.npy")
        lo = int(np.searchsorted(arr["dates"], t_dates[0]))
        if not (np.array_equal(arr["dates"][lo:], t_dates) and np.array_equal(arr["permno"][lo:], t_permno)):
            raise ValueError(f"{theirs} does not hold the same stock-days: compare by key instead")
        mem = t_permno >= 0
        t_ranks = np.load(theirs / "ranks.npy", mmap_mode="r")
        for k, c in enumerate(CHARACTERISTICS):
            x, y = arr["ranks"][lo:, :, k][mem].astype(np.float64), np.asarray(t_ranks[:, :, k])[mem].astype(np.float64)
            table.setdefault(c, {}).update({f"{fill}_spearman": float(spearmanr(x, y).statistic), f"{fill}_within_0.01": float((np.abs(x - y) <= 0.01).mean()),
                                            f"{fill}_mean_abs": float(np.abs(x - y).mean())})
        if fill == "none":                                                      # raw monthly values on universe stock-months
            ch = pd.read_parquet(SOURCE_DIR / "characteristics.parquet", columns=["permno", "month", *MONTHLY])
            mi = _month_index(pd.PeriodIndex(pd.to_datetime(ch["month"]) if ch["month"].dtype.kind == "M" else ch["month"], freq="M")) - src["month0"]
            ci = np.searchsorted(src["permnos"], ch["permno"].to_numpy())
            ok = (mi >= 0) & (mi <= src["month_of_day"].max()) & (src["permnos"][np.minimum(ci, len(src["permnos"]) - 1)] == ch["permno"].to_numpy())
            on_list, _ = universe_lists(universe, src)
            first = int(src["month_of_day"][np.searchsorted(src["dates"], t_dates[0])])
            for c in MONTHLY:
                his = np.full(on_list.shape, np.nan)
                his[mi[ok], ci[ok]] = ch[c].to_numpy(dtype=np.float64)[ok]
                x, y = src["monthly"][c][first:][on_list[first:]], his[first:][on_list[first:]]
                both = ~np.isnan(x) & ~np.isnan(y)
                table[c].update({"raw_equal": float((np.abs(x - y)[both] <= 1e-6 + 1e-4 * np.abs(y[both])).mean()),
                                 "raw_only_ours": float((~np.isnan(x) & np.isnan(y)).mean()), "raw_only_his": float((np.isnan(x) & ~np.isnan(y)).mean())})
        del src, a, arr
    return pd.DataFrame(table).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--universe", nargs="*", default=list(WRDS_UNIVERSES), choices=list(WRDS_UNIVERSES), help="[default: both]")
    p.add_argument("--fill", default="none", choices=FILLS,
                   help="missing characteristics: none = left at rank 0, last = the last observed value first (paper section 4.1) [default: none]")
    p.add_argument("--check-only", action="store_true", help="run the checks without writing anything")
    p.add_argument("--compare", action="store_true", help="only compare our ranks with the arrays built from Ander's files")
    args = p.parse_args()
    if args.compare:
        with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
            print(compare().to_string())
        return
    t0 = time.time()
    src = load_sources(args.fill)
    print(f"sources loaded in {time.time() - t0:.0f} s: {len(src['dates']):,} days from {src['dates'][0]}, {len(src['permnos']):,} securities")
    ok = [build(u, args.fill, args.check_only, src) for u in args.universe]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
