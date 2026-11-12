"""Checks Ander's processed files and writes the model-ready training arrays to Data/TrainingData/<universe>/.

    python -m afsa.build_data                 # check the sources, build the arrays, verify them against the sources
    python -m afsa.build_data --check-only    # run the checks only; nothing is written
    python -m afsa.build_data --universe common

The arrays are a dense day x slot panel: on each trading day the universe's stocks, sorted by PERMNO, fill the
first N <= 500 slots. Everything the model needs is here, so training never touches the parquet files.

    dates.npy    [T]            datetime64[D]  trading days, 1990-01-02 .. 2025-12-31
    permnos.npy  [S]            int32          every PERMNO ever in the universe, sorted; "stock id" = position
    permno.npy   [T, 500]       int32          PERMNO in each slot, -1 = empty slot
    slot_of.npy  [T, S]         int16          slot of stock id s on day t, -1 = not in the universe that day
    ranks.npy    [T, 500, 39]   float32        characteristic ranks in [-0.5, 0.5] (information up to t-1)
    levels.npy   [T, 40]        float32        39 raw cross-sectional medians + rf of the previous day
    ret.npy      [T, 500]       float32        total return of day t (decimal); missing returns count as 0
    rf.npy       [T]            float64        risk-free rate of day t (decimal)
    run.npy      [T, 500]       int16          consecutive trading days in the universe up to and including t
    meta.json                                  column names, source files, check results
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
import pyarrow.parquet as pq

from .config import CHARACTERISTICS, LEVEL_NAMES, N_CHAR, N_LEVEL, N_SLOTS, SOURCE_DIR, TRAIN_DATA_DIR, UNIVERSES
from .panel import membership

EXPECTED_FIRST_DAY = np.datetime64("1990-01-02")
EXPECTED_LAST_DAY = np.datetime64("2025-12-31")


class Checks:
    """Collects named pass/fail checks and prints them as they are made."""

    def __init__(self):
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
        return bool(ok)

    def note(self, name: str, detail: str) -> None:
        self.items.append({"check": name, "ok": None, "detail": detail})
        print(f"  [info] {name}: {detail}", flush=True)

    @property
    def failed(self) -> list[str]:
        return [c["check"] for c in self.items if c["ok"] is False]


def build(universe: str = "broad", check_only: bool = False) -> bool:
    t0 = time.time()
    xfile, flag = UNIVERSES[universe]
    xpath, cpath, fpath = SOURCE_DIR / xfile, SOURCE_DIR / "crsp_daily.parquet", SOURCE_DIR / "factors_daily.parquet"
    ck = Checks()
    print(f"Checking the source files for the {universe!r} universe in {SOURCE_DIR}")

    # ---------------------------------------------------------------- files and schema
    if not ck.add("source files exist", all(p.exists() for p in (xpath, cpath, fpath)), ", ".join(p.name for p in (xpath, cpath, fpath))):
        return False
    pf = pq.ParquetFile(xpath)
    schema = {n: str(t) for n, t in zip(pf.schema_arrow.names, pf.schema_arrow.types)}
    feature_cols = list(CHARACTERISTICS) + list(LEVEL_NAMES)
    ck.add("X has date, permno and the 79 inputs in Table 1 order", list(schema) == ["date", "permno"] + feature_cols,
           f"{len(schema) - 2} input columns")
    ck.add("the 79 inputs are float32", all(schema.get(c) == "float" for c in feature_cols))

    # ---------------------------------------------------------------- keys of X
    keys = pf.read(columns=["date", "permno"]).to_pandas()
    xdate = keys["date"].to_numpy().astype("datetime64[D]")
    xperm = keys["permno"].to_numpy()
    n_rows = len(keys)
    order_ok = bool(np.all((xdate[1:] > xdate[:-1]) | ((xdate[1:] == xdate[:-1]) & (xperm[1:] > xperm[:-1]))))
    ck.add("X is sorted by (date, permno) with unique keys", order_ok, f"{n_rows:,} stock-days")
    if not order_ok:
        print("X must be sorted by (date, permno): stopping."); return False
    dates, day_of_row, n_per_day = np.unique(xdate, return_inverse=True, return_counts=True)
    T = len(dates)
    first_row = np.concatenate([[0], np.cumsum(n_per_day)[:-1]])
    slot_of_row = (np.arange(n_rows) - first_row[day_of_row]).astype(np.int64)
    ck.add("sample runs from 1990-01-02 to 2025-12-31", dates[0] == EXPECTED_FIRST_DAY and dates[-1] == EXPECTED_LAST_DAY,
           f"{dates[0]} .. {dates[-1]}, {T:,} trading days")
    ck.add("at most 500 stocks per day", n_per_day.max() <= N_SLOTS,
           f"min {n_per_day.min()}, mean {n_per_day.mean():.1f}, max {n_per_day.max()}")

    # ---------------------------------------------------------------- calendar and risk-free rate
    fac = pd.read_parquet(fpath)
    cal = fac["date"].to_numpy().astype("datetime64[D]")
    ck.add("calendar has no missing values and is sorted", bool(fac.notna().all().all() and np.all(cal[1:] > cal[:-1])), f"{len(cal):,} days from {cal[0]}")
    cal_from_1990 = cal[cal >= dates[0]]
    ck.add("X covers every trading day of the calendar", np.array_equal(cal_from_1990, dates))
    cal_idx_of_day = np.searchsorted(cal, dates)
    rf_all = fac["rf"].to_numpy(dtype=np.float64)
    rf = rf_all[cal_idx_of_day]
    rf_prev = rf_all[cal_idx_of_day - 1]          # 1990-01-02 is not the first calendar day, so the index is valid

    # ---------------------------------------------------------------- the 79 inputs, one row group at a time
    ranks = np.zeros((T, N_SLOTS, N_CHAR), dtype=np.float32)
    levels = np.zeros((T, N_LEVEL), dtype=np.float32)
    lev_min = np.full((T, N_LEVEL), np.inf, dtype=np.float32)      # within-day range of the per-day inputs
    lev_max = np.full((T, N_LEVEL), -np.inf, dtype=np.float32)
    n_nan = 0
    lo, hi = np.inf, -np.inf
    row = 0
    for g in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(g, columns=feature_cols)
        mat = np.column_stack([tbl.column(c).to_numpy() for c in feature_cols]).astype(np.float32, copy=False)
        n = len(mat)
        d, s = day_of_row[row:row + n], slot_of_row[row:row + n]
        n_nan += int(np.isnan(mat).sum())
        lo, hi = min(lo, float(mat[:, :N_CHAR].min())), max(hi, float(mat[:, :N_CHAR].max()))
        ranks[d, s] = mat[:, :N_CHAR]
        lev = mat[:, N_CHAR:]
        starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])      # first row of each day inside this row group
        days = d[starts]
        lev_min[days] = np.minimum(lev_min[days], np.minimum.reduceat(lev, starts, axis=0))
        lev_max[days] = np.maximum(lev_max[days], np.maximum.reduceat(lev, starts, axis=0))
        levels[days] = lev[starts]
        row += n
    level_spread = float((lev_max - lev_min).max())
    ck.add("no missing value in the 79 inputs", n_nan == 0, f"{n_nan} NaN")
    ck.add("ranks lie in [-0.5, 0.5]", lo >= -0.5 and hi <= 0.5, f"min {lo:.4f}, max {hi:.4f}")
    ck.add("medians and rf are the same for every stock of a day", level_spread == 0.0, f"largest within-day difference {level_spread:g}")
    rank_mean = np.abs(ranks.sum(axis=1) / n_per_day[:, None]).max()
    ck.add("ranks are centred on zero each day", rank_mean < 0.05, f"largest |cross-sectional mean| {rank_mean:.4f}")
    ck.add("X's rf is the risk-free rate of the previous trading day", bool(np.allclose(levels[:, -1], rf_prev, atol=1e-9)),
           f"largest difference {np.abs(levels[:, -1] - rf_prev).max():.2e}")

    # ---------------------------------------------------------------- returns
    crsp = pd.read_parquet(cpath, columns=["permno", "date", "ret", flag])
    cdate_all = crsp["date"].to_numpy().astype("datetime64[D]")
    uni = crsp[flag].to_numpy()
    cu = crsp.loc[uni, ["date", "permno", "ret"]].sort_values(["date", "permno"], kind="stable")
    same_keys = len(cu) == n_rows and np.array_equal(cu["date"].to_numpy().astype("datetime64[D]"), xdate) and np.array_equal(cu["permno"].to_numpy(), xperm)
    ck.add("X rows are exactly the universe stock-days of crsp_daily", same_keys, f"{len(cu):,} universe stock-days in crsp_daily")
    if not same_keys:
        print("Returns cannot be aligned with X: stopping."); return False
    ret_row = cu["ret"].to_numpy(dtype=np.float64)
    nan_ret = np.isnan(ret_row)
    ck.note("missing returns (trading suspensions), counted as 0", f"{int(nan_ret.sum())} stock-days")
    ck.add("returns are plausible", bool(np.nanmin(ret_row) > -1.0 and np.nanmax(ret_row) < 5.0),
           f"min {np.nanmin(ret_row):.3f}, max {np.nanmax(ret_row):.3f}, {int((np.abs(ret_row) > 0.5).sum())} with |ret| > 50%")
    ret = np.zeros((T, N_SLOTS), dtype=np.float32)
    ret[day_of_row, slot_of_row] = np.where(nan_ret, 0.0, ret_row)

    # ---------------------------------------------------------------- identifiers, membership history
    permno = np.full((T, N_SLOTS), -1, dtype=np.int32)
    permno[day_of_row, slot_of_row] = xperm
    permnos, slot_of, run = membership(permno)                 # run = consecutive days in the universe up to t
    S = len(permnos)
    run_row = run[day_of_row, slot_of_row]
    ck.note("stocks", f"{S:,} PERMNOs are in the universe at some point")
    ck.note("stock-days with fewer than 30 prior days in the universe (no signal)",
            f"{(run_row <= 30).mean():.2%} of all stock-days; {(run_row[day_of_row >= 31] <= 30).mean():.2%} after the first 31 days of the sample")

    # ---------------------------------------------------------------- no look-ahead: Ret_D1 is yesterday's return
    cal_idx_row = np.searchsorted(cal, cdate_all)
    all_perm = np.unique(crsp["permno"].to_numpy())
    dense = np.full((len(cal), len(all_perm)), np.nan, dtype=np.float32)
    dense[cal_idx_row, np.searchsorted(all_perm, crsp["permno"].to_numpy())] = crsp["ret"].to_numpy(dtype=np.float32)
    col = np.searchsorted(all_perm, xperm)
    cal_row = cal_idx_of_day[day_of_row]
    chk = pd.DataFrame({"day": day_of_row, "ret_d1": ranks[day_of_row, slot_of_row, CHARACTERISTICS.index("Ret_D1")],
                        "prev": dense[cal_row - 1, col], "same": dense[cal_row, col], "next": dense[np.minimum(cal_row + 1, len(cal) - 1), col]})
    del dense
    for c in ("prev", "same", "next"):
        chk[c] = chk.groupby("day")[c].rank()
    corr = chk.groupby("day")[["ret_d1", "prev", "same", "next"]].corr()["ret_d1"].unstack()
    c_prev, c_same, c_next = (float(corr[c].mean()) for c in ("prev", "same", "next"))
    ck.add("Ret_D1 on day t ranks the return of day t-1 (no look-ahead)", c_prev > 0.99 and abs(c_same) < 0.1 and abs(c_next) < 0.1,
           f"mean daily rank correlation with the return of t-1: {c_prev:.4f}, of t: {c_same:+.4f}, of t+1: {c_next:+.4f}")

    # ---------------------------------------------------------------- reference numbers
    exret = np.where(permno >= 0, ret - rf[:, None].astype(np.float32), 0.0)
    ew = exret.sum(axis=1) / n_per_day
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    oos = (yr >= 1998) & (yr <= 2021)
    mu, sd = ew[oos].mean() * 252, ew[oos].std() * np.sqrt(252)
    ck.note("equal-weighted universe, excess return, 1998-2021",
            f"mean {mu:.2%}, volatility {sd:.2%}, Sharpe {mu / sd:.2f} (paper's market row: 8.61%, 20.37%, 0.42)")
    days_per_year = {int(y): int(c) for y, c in zip(*np.unique(yr, return_counts=True))}
    ck.note("trading days in the first training window (1990-1997)", f"{sum(days_per_year[y] for y in range(1990, 1998)):,}; first day with 31 prior days: {dates[31]}")

    print(f"Checks finished in {time.time() - t0:.0f} s: {len(ck.failed)} failed.")
    if ck.failed:
        print("FAILED:", "; ".join(ck.failed))
        return False
    if check_only:
        return True

    # ---------------------------------------------------------------- write
    out = TRAIN_DATA_DIR / universe
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"dates": dates, "permnos": permnos, "permno": permno, "slot_of": slot_of, "ranks": ranks, "levels": levels,
              "ret": ret, "rf": rf, "run": run}
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)
    nan_list = [[str(d), int(p)] for d, p in zip(xdate[nan_ret], xperm[nan_ret])]
    meta = {
        "universe": universe,
        "built": datetime.now().isoformat(timespec="seconds"),
        "sources": {p.name: {"bytes": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
                    for p in (xpath, cpath, fpath)},
        "n_days": int(T), "n_slots": N_SLOTS, "n_stocks": int(S), "n_stock_days": int(n_rows),
        "first_day": str(dates[0]), "last_day": str(dates[-1]),
        "characteristics": list(CHARACTERISTICS), "levels": list(LEVEL_NAMES),
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "days_per_year": days_per_year,
        "missing_returns_counted_as_zero": nan_list,
        "checks": ck.items,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"Wrote {len(arrays)} arrays + meta.json to {out} ({size / 2**20:.0f} MB).")

    ok = verify(universe)
    print(f"Done in {time.time() - t0:.0f} s.")
    return ok


def verify(universe: str = "broad", n_sample: int = 200_000, seed: int = 0) -> bool:
    """Re-reads the written arrays and compares a random sample of stock-days with the parquet sources."""
    xfile, flag = UNIVERSES[universe]
    out = TRAIN_DATA_DIR / universe
    a = {f.stem: np.load(f, mmap_mode="r") for f in out.glob("*.npy")}
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(SOURCE_DIR / xfile)
    groups = rng.choice(pf.metadata.num_row_groups, size=min(4, pf.metadata.num_row_groups), replace=False)
    worst = 0.0
    n_checked = 0
    for g in groups:
        df = pf.read_row_group(int(g)).to_pandas()
        df = df.sample(n=min(len(df), n_sample // len(groups)), random_state=int(g))
        t = np.searchsorted(a["dates"], df["date"].to_numpy().astype("datetime64[D]"))
        s = np.searchsorted(a["permnos"], df["permno"].to_numpy())
        slot = np.asarray(a["slot_of"])[t, s].astype(np.int64)
        if (slot < 0).any() or not np.array_equal(np.asarray(a["permno"])[t, slot], df["permno"].to_numpy()):
            print("  [FAIL] verification: a sampled stock-day is missing from the arrays"); return False
        worst = max(worst, float(np.abs(np.asarray(a["ranks"])[t, slot] - df[list(CHARACTERISTICS)].to_numpy()).max()),
                    float(np.abs(np.asarray(a["levels"])[t] - df[list(LEVEL_NAMES)].to_numpy()).max()))
        r = pd.read_parquet(SOURCE_DIR / "crsp_daily.parquet", columns=["permno", "date", "ret"],
                            filters=[(flag, "==", True), ("date", ">=", df["date"].min()), ("date", "<=", df["date"].max())])
        m = df[["date", "permno"]].merge(r, on=["date", "permno"], how="left", validate="one_to_one")
        worst = max(worst, float(np.abs(np.asarray(a["ret"])[t, slot] - m["ret"].fillna(0.0).to_numpy(dtype=np.float32)).max()))
        n_checked += len(df)
    ok = worst == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] verification: {n_checked:,} sampled stock-days re-read from the parquet files match the arrays "
          f"(largest difference {worst:g})")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--universe", default="broad", choices=list(UNIVERSES))
    p.add_argument("--check-only", action="store_true", help="run the checks without writing anything")
    args = p.parse_args()
    sys.exit(0 if build(args.universe, args.check_only) else 1)


if __name__ == "__main__":
    main()
