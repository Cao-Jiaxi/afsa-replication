"""Model inputs for a universe that is fixed within each training window: Data/TrainingData/<rule>/<n>y/<year>/.

    python -m afsa.build_window --check                        # the recipe below reproduces Ander's X on sample months
    python -m afsa.build_window                                # rule "window": test years 1998-2021 (8-year windows)
                                                               #   and 1996-1997 (6-year windows)
    python -m afsa.build_window --rule window_train            # the variant that does not use the test year
    python -m afsa.build_window --fill none                    # missing characteristics left at rank 0 (no carry-forward)

Why. The paper says "the 500 largest stocks based on the previous month market capitalization", which
afsa.build_data reads as a new top 500 every month. The first author's companion paper (Epstein, Sadhwani and
Giesecke, arXiv 2505.11243, section 5), which uses the same 79 inputs, protocol and hyperparameters, says instead:
"For each model fitting, we filter out stocks where any returns in the train or test period are not available. We
include only the 500 assets with the largest market cap at the last quarter of the training period". AFSA's own
figures fit that rule better than a monthly top 500 (analysis/gap/REPORT.md).

Rules, for a model trained on the n calendar years before test year Y:
    window         securities with a return on every trading day of the n training years AND of Y. This is the
                   companion paper's rule. It looks ahead: a stock that stops trading during Y is never held.
    window_train   securities with a return on every trading day of the n training years. No look-ahead: a stock
                   that stops trading during Y leaves the portfolio on that day.
Of these, the 500 with the largest market capitalization on the last trading day of the training years. The list
is then the same on every day of the window.

Limits. Only the 2,521 securities of Ander's files are candidates (those in a monthly top 500 at some point), so a
stock that the rule would pick but that never reached a monthly top 500 is missing, and ADRs are sized by their
US-listed shares only, as everywhere in this project.

Inputs follow the recipe of Ander's X files (checked by --check): each characteristic is ranked across the
day's stocks that have a value, (rank - 1) / (n - 1) - 0.5 with average ranks for ties, and a stock without one gets
rank 0; the median is taken over the same values. What "has a value" means is the fill rule (config.FILLS):
    last   a missing value is first replaced by the stock's last observed one (no age limit). The paper's section 4.1
           and Ander's X files.
    none   a missing value stays missing (Prof. Polak's suggestion, 2026-11-13).
The 35 monthly characteristics are those of the day's month, ranked over the month's list; the 4 daily ones use the
5 trading days before the day, ranked over the day's list. The arrays have the layout of afsa.build_data, restricted
to the window's own days. `slot_inputs` builds the same inputs for the day x slot panel of afsa.build_data.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import CHARACTERISTICS, FILLS, LEVEL_NAMES, N_CHAR, N_SLOTS, SOURCE_DIR, TRAIN_DATA_DIR, WINDOW_RULES, data_folder
from .panel import membership

DAILY = ("Ret_D1", "Ret_W1", "STD_W1", "Vol")
MONTHLY = tuple(c for c in CHARACTERISTICS if c not in DAILY)
LEAD_DAYS = 40            # days kept before the first training year, for the residual lookback


def ffill(a: np.ndarray) -> np.ndarray:
    """Last observed value along axis 0 (NaN until the first observation)."""
    idx = np.where(~np.isnan(a), np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def load_sources(fill: str = "last") -> dict:
    """Ander's files as dense day x security matrices (calendar from 1986), with missing characteristics filled by `fill`."""
    if fill not in FILLS:
        raise ValueError(f"unknown fill rule {fill!r}; choose from {FILLS}")
    carry = ffill if fill == "last" else (lambda a: a)
    cal = pd.read_parquet(SOURCE_DIR / "factors_daily.parquet", columns=["date", "rf"]).sort_values("date")
    dates = cal["date"].to_numpy().astype("datetime64[D]")
    crsp = pd.read_parquet(SOURCE_DIR / "crsp_daily.parquet", columns=["permno", "date", "ret", "vol", "cap"])
    permnos = np.unique(crsp["permno"].to_numpy()).astype(np.int32)
    row = np.searchsorted(dates, crsp["date"].to_numpy().astype("datetime64[D]"))
    col = np.searchsorted(permnos, crsp["permno"].to_numpy())
    dense = {}
    for name in ("ret", "vol", "cap"):
        dense[name] = np.full((len(dates), len(permnos)), np.nan)
        dense[name][row, col] = crsp[name].to_numpy(dtype=np.float64)
    del crsp
    R, V = dense["ret"], dense["vol"]
    # the 4 daily characteristics of day t, from days t-5..t-1 (NaN unless all five are there)
    daily = {c: np.full_like(R, np.nan) for c in DAILY}
    win = np.lib.stride_tricks.sliding_window_view(R, 5, axis=0)[:-1]          # [T-5, S, 5]: window k is days k..k+4
    daily["Ret_D1"][1:] = R[:-1]
    daily["Ret_W1"][5:] = np.prod(1 + win, axis=2) - 1
    daily["STD_W1"][5:] = win.std(axis=2, ddof=1)
    daily["Vol"][5:] = np.lib.stride_tricks.sliding_window_view(V, 5, axis=0)[:-1].sum(axis=2)
    daily = {c: carry(v) for c, v in daily.items()}
    # the 35 monthly characteristics, by holding month (fill: the last observed month carried forward)
    ch = pd.read_parquet(SOURCE_DIR / "characteristics.parquet")
    month_of_row = (ch["month"].dt.year * 12 + ch["month"].dt.month - 1).to_numpy()
    m0 = int(month_of_row.min())
    month_of_day = pd.DatetimeIndex(dates).year * 12 + pd.DatetimeIndex(dates).month - 1 - m0
    n_months = int(max(month_of_row.max() - m0, month_of_day.max())) + 1
    ccol = np.searchsorted(permnos, ch["permno"].to_numpy())
    monthly = {}
    for c in MONTHLY:
        m = np.full((n_months, len(permnos)), np.nan)
        m[month_of_row - m0, ccol] = ch[c].to_numpy(dtype=np.float64)
        monthly[c] = carry(m)
    return {"fill": fill, "dates": dates, "years": dates.astype("datetime64[Y]").astype(int) + 1970, "rf": cal["rf"].to_numpy(dtype=np.float64),
            "permnos": permnos, "ret": R, "cap": dense["cap"], "daily": daily, "monthly": monthly,
            "month_of_day": np.asarray(month_of_day).clip(min=0)}


def ranks_and_medians(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """values [T, n], NaN = no value. Ranks in [-0.5, 0.5] across each row (0 where NaN) and the row medians."""
    nan = np.isnan(values)
    n = (~nan).sum(axis=1, keepdims=True)
    if np.isinf(values).any():
        r = rankdata(values, axis=1, nan_policy="omit")                       # slow path, never needed so far
    else:
        r = rankdata(np.where(nan, np.inf, values), axis=1)                   # NaN ranked last: the others keep their ranks
    with np.errstate(invalid="ignore", divide="ignore"):
        ranks = np.where(nan, 0.0, (r - 1) / np.maximum(n - 1, 1) - 0.5)
        med = np.where(n[:, 0] > 0, np.nanmedian(np.where(n > 0, values, 0.0), axis=1), 0.0)
    return ranks, med


def inputs(src: dict, days: np.ndarray, cols: np.ndarray, member: np.ndarray,
           monthly_member: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The 79 inputs for calendar rows `days` and securities `cols`: ranks [T, n, 39] and levels [T, 40].

    `member` [T, n] says who is in the cross-section each day. `monthly_member`, if given, is the cross-section
    used for the monthly characteristics (Ander ranks those once a month, over the month's list).
    """
    ranks = np.zeros((len(days), len(cols), N_CHAR), dtype=np.float32)
    levels = np.zeros((len(days), len(LEVEL_NAMES)), dtype=np.float32)
    mm = member if monthly_member is None else monthly_member
    for k, c in enumerate(CHARACTERISTICS):
        if c in DAILY:
            v = np.where(member, src["daily"][c][np.ix_(days, cols)], np.nan)
        else:
            v = np.where(mm, src["monthly"][c][np.ix_(src["month_of_day"][days], cols)], np.nan)
        ranks[:, :, k], levels[:, k] = ranks_and_medians(v)
    levels[:, N_CHAR] = src["rf"][days - 1]                                    # rf of the previous trading day
    return ranks, levels


def slot_inputs(src: dict, dates: np.ndarray, permno: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 79 inputs in the day x slot layout of afsa.build_data, for a universe given as PERMNO per slot.

    dates [T] trading days; permno [T, 500] with -1 = empty slot. Returns ranks [T, 500, 39], levels [T, 40] and
    missing [39]: the stock-days of the universe that have no value for each characteristic (rank 0).
    The monthly characteristics are ranked over the month's list, as in Ander's X (see `check`).
    """
    days = np.searchsorted(src["dates"], dates.astype("datetime64[D]"))
    if not np.array_equal(src["dates"][days], dates.astype("datetime64[D]")):
        raise ValueError("a day of the panel is not in the calendar of the sources")
    filled = permno >= 0
    ids = np.unique(permno[filled])
    cols = np.searchsorted(src["permnos"], ids)
    if not np.array_equal(src["permnos"][cols], ids):
        raise ValueError("a PERMNO of the panel is not in crsp_daily")
    day, slot = np.nonzero(filled)
    col = np.searchsorted(ids, permno[filled])                               # column of each stock-day in the dense block
    member = np.zeros((len(dates), len(ids)), dtype=bool)
    member[day, col] = True
    month = src["month_of_day"][days]
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])            # days are sorted, so each month is one block
    in_month = np.logical_or.reduceat(member, starts, axis=0)                # [months, n] in the universe on some day of it
    monthly_member = np.repeat(in_month, np.diff(np.r_[starts, len(days)]), axis=0)
    ranks = np.zeros((len(dates), permno.shape[1], N_CHAR), dtype=np.float32)
    levels = np.zeros((len(dates), len(LEVEL_NAMES)), dtype=np.float32)
    missing = np.zeros(N_CHAR, dtype=np.int64)
    for k, c in enumerate(CHARACTERISTICS):
        if c in DAILY:
            v = np.where(member, src["daily"][c][np.ix_(days, cols)], np.nan)
        else:
            v = np.where(monthly_member, src["monthly"][c][np.ix_(month, cols)], np.nan)
        r, levels[:, k] = ranks_and_medians(v)
        ranks[day, slot, k] = r[day, col]
        missing[k] = int(np.isnan(v[day, col]).sum())
    levels[:, N_CHAR] = src["rf"][days - 1]
    return ranks, levels, missing


def select(src: dict, year: int, train_years: int, rule: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calendar rows of the window (with lead days) and the columns of its 500 securities."""
    years = src["years"]
    train = np.flatnonzero((years >= year - train_years) & (years < year))
    test = np.flatnonzero(years == year)
    if len(train) == 0 or len(test) == 0 or years[train[0]] != year - train_years:
        raise ValueError(f"the sample does not cover the {train_years} years before {year} and {year} itself")
    need = np.concatenate([train, test]) if rule == "window" else train
    complete = ~np.isnan(src["ret"][need]).any(axis=0)
    cap = src["cap"][train[-1]]
    cand = np.flatnonzero(complete & ~np.isnan(cap))
    cols = np.sort(cand[np.argsort(-cap[cand], kind="stable")[:N_SLOTS]])
    days = np.arange(max(train[0] - LEAD_DAYS, 6), test[-1] + 1)
    info = {"candidates": int(len(cand)), "selected": int(len(cols)), "cap_largest_bn": float(cap[cols].max() / 1e6),
            "cap_smallest_bn": float(cap[cols].min() / 1e6), "ranked_on": str(src["dates"][train[-1]])}
    return days, cols, info


def build(src: dict, year: int, train_years: int, rule: str) -> dict:
    days, cols, info = select(src, year, train_years, rule)
    R = src["ret"][np.ix_(days, cols)]
    member = ~np.isnan(R)
    ranks, levels = inputs(src, days, cols, member)
    permno = np.full((len(days), N_SLOTS), -1, dtype=np.int32)
    permno[:, :len(cols)] = np.where(member, src["permnos"][cols][None, :], -1)
    ret = np.zeros((len(days), N_SLOTS), dtype=np.float32)
    ret[:, :len(cols)] = np.where(member, R, 0.0)
    full = np.zeros((len(days), N_SLOTS, N_CHAR), dtype=np.float32)
    full[:, :len(cols)] = ranks
    permnos, slot_of, run = membership(permno)
    arrays = {"dates": src["dates"][days], "permnos": permnos, "permno": permno, "slot_of": slot_of, "ranks": full,
              "levels": levels, "ret": ret, "rf": src["rf"][days], "run": run}
    out = TRAIN_DATA_DIR / data_folder(rule, src["fill"]) / f"{train_years}y" / str(year)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)
    yr = src["years"][days]
    info.update({"universe": f"{data_folder(rule, src['fill'])}/{train_years}y/{year}", "rule": rule, "fill": src["fill"],
                 "test_year": year, "train_years": train_years,
                 "built": datetime.now().isoformat(timespec="seconds"), "n_days": int(len(days)),
                 "first_day": str(arrays["dates"][0]), "last_day": str(arrays["dates"][-1]),
                 "members_per_day_in_test_year": [int(member[yr == year].sum(1).min()), int(member[yr == year].sum(1).max())],
                 "characteristics": list(CHARACTERISTICS), "levels": list(LEVEL_NAMES), "checks": []})
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    return info


def check(src: dict) -> bool:
    """Rebuilds the broad universe's inputs for a few months with the recipe above and compares them with X_amplio."""
    ok = True
    for month in ("1994-03", "2001-09", "2008-10", "2015-06", "2021-01"):
        p = pd.Period(month, "M")
        X = pd.read_parquet(SOURCE_DIR / "X_amplio.parquet", filters=[("date", ">=", p.start_time), ("date", "<=", p.end_time)])
        d = np.unique(X["date"].to_numpy().astype("datetime64[D]"))
        days = np.searchsorted(src["dates"], d)
        ids = np.unique(X["permno"].to_numpy())
        cols = np.searchsorted(src["permnos"], ids)
        r, c = np.searchsorted(d, X["date"].to_numpy().astype("datetime64[D]")), np.searchsorted(ids, X["permno"].to_numpy())
        member = np.zeros((len(d), len(ids)), dtype=bool)
        member[r, c] = True
        ref = np.zeros((len(d), len(ids), N_CHAR), dtype=np.float32)
        ref[r, c] = X[list(CHARACTERISTICS)].to_numpy()
        ref_levels = X.groupby("date")[list(LEVEL_NAMES)].first().to_numpy()
        line = []
        for label, mm in (("daily list", None), ("month's list", np.broadcast_to(member.any(0), member.shape))):
            ranks, levels = inputs(src, days, cols, member, mm)
            diff = np.abs(ranks - ref)[member]                                   # [stock-days, 39]
            rel = np.abs(levels - ref_levels) / np.maximum(np.abs(ref_levels), 1e-12)
            line.append(f"{label}: ranks off by more than 1e-4 in {100 * (diff > 1e-4).mean():.3f}% of entries (largest {diff.max():.3f}), "
                        f"medians off by more than 1e-5 in {100 * (rel > 1e-5).mean():.2f}% (largest {rel.max():.2e})")
            if mm is not None:
                ok &= bool((diff > 1e-4).mean() < 0.002)
        print(f"  {month}: " + "\n           ".join(line))
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rule", default="window", choices=WINDOW_RULES)
    p.add_argument("--years", type=int, nargs="*", default=None, help="test years [default: 1998-2021 with 8 training years, 1996-1997 with 6]")
    p.add_argument("--train-years", type=int, default=8)
    p.add_argument("--fill", default="last", choices=FILLS, help="missing characteristics: last observed value, or none [default: last]")
    p.add_argument("--check", action="store_true", help="only compare the recipe (fill = last) with Ander's X_amplio")
    args = p.parse_args()
    t0 = time.time()
    src = load_sources("last" if args.check else args.fill)
    print(f"sources loaded in {time.time() - t0:.0f} s: {len(src['dates']):,} days, {len(src['permnos']):,} securities")
    if args.check:
        print("the recipe against X_amplio (broad universe):")
        print("PASS" if check(src) else "FAIL")
        return
    jobs = [(y, args.train_years) for y in args.years] if args.years else [(y, 8) for y in range(1998, 2022)] + [(1996, 6), (1997, 6)]
    for year, n in jobs:
        info = build(src, year, n, args.rule)
        print(f"  {info['universe']}: {info['selected']} of {info['candidates']} candidates, caps {info['cap_smallest_bn']:.2f}-"
              f"{info['cap_largest_bn']:.0f} bn on {info['ranked_on']}, {info['n_days']} days, "
              f"members in the test year {info['members_per_day_in_test_year']}", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
