"""Is a variant's Sharpe ratio different from the baseline's? A paired block bootstrap on the daily returns.

    python analysis/gap/paired_test.py --K 8                          # every variant on disk against the baseline
    python analysis/gap/paired_test.py cost10 cumulative --K 8 30
    python analysis/gap/paired_test.py cost10 --K 8 --period validation

The spread across seeds only measures the noise of the optimizer on one sample path: the three seeds trade the same
24 years. This test measures the sampling noise instead. Both runs are reduced to their pooled series (the average
over seeds of the daily net return, scored at the paper's costs), the same resampled days are drawn for both, and
the difference of the two Sharpe ratios is recomputed on each draw.

- Circular blocks of 21 days, 2,000 draws, a fixed random stream: the numbers are reproducible.
- Reported: the difference, its 95% percentile interval and z = difference / standard deviation of the draws.
- A sub-period is resampled within its own days.

The pooled Sharpe ratio is higher than the seed average of RESULTS.md (averaging cuts the volatility): compare a
pooled figure only with another pooled figure. With some thirty variants tested, an interval that excludes zero by
a little is weak evidence. Test years are diagnostics, not a basis for choosing a variant (CLAUDE.md, section 8).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import BLOCKS, PERIODS, VARIANTS, config, run_dir  # noqa: E402

from afsa.config import TRADING_DAYS, Config  # noqa: E402
from afsa.metrics import stitch  # noqa: E402

BLOCK_DAYS, DRAWS, STREAM = 21, 2000, 20261114


def seed_series(cfg: Config, first: int, last: int) -> list[pd.DataFrame] | None:
    """Daily results of each seed of a run over the years first..last (gross, turnover, short, cost, net), or None if
    the run is missing or incomplete."""
    out = []
    for seed_dir in sorted(run_dir(cfg).glob("seed*"), key=lambda p: int(p.name[4:])):
        if any(not (seed_dir / f"{y}.npz").exists() for y in range(first, last + 1)):
            return None
        daily = stitch(seed_dir)
        out.append(daily[(daily.index.year >= first) & (daily.index.year <= last)])
    return out or None


def pooled(cfg: Config, first: int, last: int) -> pd.DataFrame | None:
    """The run's pooled series: the average over seeds of each daily column."""
    seeds = seed_series(cfg, first, last)
    return None if seeds is None else sum(seeds) / len(seeds)


def sharpe(x: np.ndarray) -> np.ndarray:
    """Annualized Sharpe ratio along the last axis."""
    return x.mean(-1) / x.std(-1, ddof=1) * np.sqrt(TRADING_DAYS)


def block_draws(n: int, block: int = BLOCK_DAYS, draws: int = DRAWS, stream: int = STREAM) -> np.ndarray:
    """Day indices of `draws` circular block resamples of n days. [draws, n]"""
    rng = np.random.default_rng(stream)
    starts = rng.integers(0, n, size=(draws, -(-n // block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, block: int = BLOCK_DAYS, draws: int = DRAWS, stream: int = STREAM) -> dict[str, float]:
    """Sharpe ratio of `a` minus that of `b`, two series on the same days, with its bootstrap interval."""
    if len(a) != len(b):
        raise ValueError("the two series must cover the same days")
    idx = block_draws(len(a), block, draws, stream)
    d = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta, sd = float(sharpe(a) - sharpe(b)), float(d.std(ddof=1))
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "sd": sd, "z": delta / sd if sd > 0 else float("nan")}


def compare(a: pd.DataFrame, b: pd.DataFrame, column: str = "net", eras: tuple[tuple[int, int], ...] = ()) -> dict[str, dict[str, float]]:
    """Paired test of two daily frames on their common days: the whole period ("all") and each era that lies in it."""
    days = a.index.intersection(b.index)
    x, y = a.loc[days, column], b.loc[days, column]
    out = {"all": {**paired_bootstrap(x.to_numpy(), y.to_numpy()), "SR": float(sharpe(x.to_numpy())), "days": len(days)}}
    for lo, hi in eras:
        m = (days.year >= lo) & (days.year <= hi)
        if m.any() and days.year.min() <= lo and hi <= days.year.max():
            out[f"{lo}-{str(hi)[2:]}"] = paired_bootstrap(x.to_numpy()[m], y.to_numpy()[m])
    return out


def cell(r: dict[str, float]) -> str:
    return f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"


def table(variants: list[str], K: int, period: str = "test", extra: dict | None = None) -> pd.DataFrame:
    """One row per variant on disk: pooled net Sharpe, and its difference from the baseline with the interval."""
    first, last, train_years = PERIODS[period]
    base = pooled(config("baseline", K, train_years, extra), first, last)
    if base is None:
        return pd.DataFrame()
    eras = BLOCKS if period == "test" else ()
    rows = {"baseline": {"pooled SR net": float(sharpe(base["net"].to_numpy())), "turnover": base["turnover"].mean()}}
    for v in variants:
        if v == "baseline":
            continue
        run = pooled(config(v, K, train_years, extra), first, last)
        if run is None:
            continue
        res = compare(run, base, eras=eras)
        rows[v] = {"pooled SR net": res["all"]["SR"], "turnover": run["turnover"].mean(), "difference [95% interval]": cell(res["all"]),
                   "z": res["all"]["z"], **{era: cell(r) for era, r in res.items() if era != "all"}}
    return pd.DataFrame(rows).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="*", help="variants to test (default: all on disk)")
    p.add_argument("--K", type=int, nargs="+", default=[8])
    p.add_argument("--period", default="test", choices=list(PERIODS))
    args = p.parse_args()
    pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 500)
    for K in args.K:
        t = table(args.variants or list(VARIANTS), K, args.period)
        first, last, _ = PERIODS[args.period]
        print(f"\nK = {K}, {first}-{last}: variant minus baseline, pooled net Sharpe ratio")
        print(t.to_string(float_format=lambda x: f"{x:.2f}", na_rep="") if len(t) else "  no baseline on disk")


if __name__ == "__main__":
    main()
