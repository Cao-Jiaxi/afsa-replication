"""Market exposure, net position and trading of a run by era, and the market row in two conventions.

    python analysis/gap/exposure.py baseline rf_hurdle cumulative cost10 --K 8 30
    python analysis/gap/exposure.py --market

By era (1998-2005, 2006-2013, 2014-2021) and for the whole period, as seed averages:
- beta:        slope of the daily gross return on the equal-weighted excess return of the run's universe.
- net long:    mean of sum(w). The book has unit gross exposure, so sum(w) = 1 - 2 x short exposure, which every run
               stores by day: no weights are needed.
- |net|:       mean of |sum(w)|.
- turnover:    mean daily turnover.
- SR net:      net Sharpe ratio at the paper's costs, and the same after removing beta x market (the era's own beta).

A run whose net Sharpe ratio falls when the market is removed earned part of it from market exposure.

The market row: the paper's Table 2 reports 8.61% a year, 20.37% volatility, Sharpe ratio 0.42 for its
equal-weighted universe. `--market` prints ours as the simple excess return and as the log excess return of the
same portfolio, for each universe on disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_test import seed_series, sharpe  # noqa: E402
from variants import BLOCKS, PERIODS, config  # noqa: E402

from afsa.config import TRADING_DAYS, TRAIN_DATA_DIR, Config, data_folder  # noqa: E402
from afsa.metrics import market_returns  # noqa: E402

PAPER_MARKET = (8.61, 20.37, 0.42)
MARKET_UNIVERSES = (("broad", "last"), ("common", "last"), ("wrds_broad", "none"), ("wrds_common", "none"))


def by_era(cfg: Config, first: int, last: int, eras: tuple[tuple[int, int], ...] = BLOCKS) -> pd.DataFrame | None:
    """One row per period (the whole span, then each era inside it), seed averages; None if the run is incomplete."""
    seeds = seed_series(cfg, first, last)
    if seeds is None:
        return None
    market = market_returns(cfg.universe, cfg.train_years, cfg.fill)
    periods = [(first, last)] + [(a, b) for a, b in eras if first <= a and b <= last]
    rows = {}
    for a, b in periods:
        per_seed = []
        for daily in seeds:
            d = daily[(daily.index.year >= a) & (daily.index.year <= b)]
            m = market.reindex(d.index).to_numpy()
            beta = float(np.cov(d["gross"], m)[0, 1] / m.var(ddof=1))
            net_long = 1 - 2 * d["short"]
            per_seed.append({"beta": beta, "net long": net_long.mean(), "|net|": net_long.abs().mean(), "turnover": d["turnover"].mean(),
                             "SR net": float(sharpe(d["net"].to_numpy())), "SR net, market removed": float(sharpe(d["net"].to_numpy() - beta * m))})
        rows[f"{a}-{b}"] = pd.DataFrame(per_seed).mean()
    return pd.DataFrame(rows).T


def market_rows(first: int = 1998, last: int = 2021) -> pd.DataFrame:
    """Annualized mean and volatility (%) and their ratio for each universe on disk, simple and log excess returns."""
    rows = {"paper, Table 2": dict(zip(("mu", "sigma", "SR"), PAPER_MARKET))}
    for universe, fill in MARKET_UNIVERSES:
        if not (TRAIN_DATA_DIR / data_folder(universe, fill) / "ret.npy").exists():
            continue
        for label, log in (("simple", False), ("log", True)):
            r = market_returns(universe, 8, fill, log=log)
            r = r[(r.index.year >= first) & (r.index.year <= last)]
            mu, sigma = r.mean() * TRADING_DAYS * 100, r.std() * np.sqrt(TRADING_DAYS) * 100
            rows[f"{data_folder(universe, fill)}, {label}"] = {"mu": mu, "sigma": sigma, "SR": mu / sigma}
    return pd.DataFrame(rows).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="*", default=["baseline"])
    p.add_argument("--K", type=int, nargs="+", default=[8])
    p.add_argument("--market", action="store_true", help="print the market row instead")
    args = p.parse_args()
    pd.set_option("display.width", 240, "display.max_columns", 40)
    if args.market:
        print(market_rows().round(2).to_string())
        return
    first, last, train_years = PERIODS["test"]
    for K in args.K:
        for v in args.variants:
            t = by_era(config(v, K, train_years), first, last)
            print(f"\n{v}, K = {K}" + ("" if t is not None else ": not on disk"))
            if t is not None:
                print(t.round(3).to_string())


if __name__ == "__main__":
    main()
