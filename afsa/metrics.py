"""Results of a run on disk: joining the test years, the metrics of Table 2 and the comparison with the paper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, TRAIN_DATA_DIR, WINDOW_RULES, Config, data_folder

# Attention Factors rows of Table 2 (out-of-sample 1998-2021, annualized, mu and sigma in %).
METRICS = ("SR", "mu", "sigma", "SR_net", "mu_net", "sigma_net", "beta")
PAPER_TABLE2 = {
    1: (3.05, 14.45, 4.74, 1.68, 7.94, 4.72, 0.05),
    3: (3.05, 14.91, 4.89, 1.69, 8.25, 4.87, 0.06),
    5: (2.92, 14.21, 4.87, 1.58, 7.66, 4.85, 0.07),
    8: (3.35, 15.70, 4.68, 1.94, 9.05, 4.66, 0.07),
    15: (3.81, 16.66, 4.37, 2.25, 9.78, 4.35, 0.06),
    30: (3.97, 16.66, 4.20, 2.28, 9.52, 4.18, 0.05),
    100: (4.52, 16.45, 3.64, 2.19, 7.93, 3.62, 0.05),
}


# The paper's costs: per unit of turnover and per unit of short exposure per day. Every run is scored with them,
# whatever costs it was trained with (Config.cost_trade and cost_short belong to the training objective).
COST_TRADE, COST_SHORT = 5e-4, 1e-4


def market_returns(universe: str = "broad", train_years: int = 8, fill: str = "last", log: bool = False) -> pd.Series:
    """Equal-weighted excess return of the universe, by day (the paper's market row), from the arrays the run trained
    on (config.data_folder: the `wrds_*` universes exist only with a fill suffix).

    `log` gives the log excess return of the same portfolio, log(1 + R) - log(1 + rf) with R its simple return. Every
    beta uses the simple return (the default).
    For a universe that is fixed per training window, each test year contributes the days of its own list."""
    def one(folder: Path) -> pd.Series:
        member = np.load(folder / "permno.npy") >= 0
        ret, rf, n = np.load(folder / "ret.npy"), np.load(folder / "rf.npy"), np.maximum(member.sum(1), 1)
        if log:
            ex = np.log1p(np.where(member, ret, 0.0).sum(1) / n) - np.log1p(rf.astype(np.float64))
        else:
            ex = np.where(member, ret - rf[:, None], 0.0).sum(1) / n
        return pd.Series(ex, index=pd.DatetimeIndex(np.load(folder / "dates.npy")), name="market")

    root = TRAIN_DATA_DIR / data_folder(universe, fill)
    if universe not in WINDOW_RULES:
        return one(root)
    parts = []
    for folder in sorted((root / f"{train_years}y").glob("[0-9][0-9][0-9][0-9]")):
        s = one(folder)
        parts.append(s[s.index.year == int(folder.name)])
    return pd.concat(parts).rename("market")


def stitch(seed_dir: Path) -> pd.DataFrame:
    """Daily results of one seed over all its test years, net of the paper's costs (COST_TRADE, COST_SHORT).

    On the first day of each year the portfolio moves from the previous year's model to the new one, and pays
    that turnover; the very first day is bought from cash.
    """
    frames, prev = [], None
    for f in sorted(seed_dir.glob("[0-9][0-9][0-9][0-9].npz")):
        z = np.load(f)
        year, turnover = int(f.stem), z["turnover"].copy()
        first = pd.Series(z["w_first"], index=z["permnos"])
        if prev is not None and prev[0] == year - 1:
            turnover[0] = first.sub(prev[1], fill_value=0.0).abs().sum()
        else:
            turnover[0] = first.abs().sum()
        prev = (year, pd.Series(z["w_last"], index=z["permnos"]))
        frames.append(pd.DataFrame({"gross": z["gross"], "turnover": turnover, "short": z["short"]}, index=pd.DatetimeIndex(z["dates"])))
    if not frames:
        return pd.DataFrame(columns=["gross", "turnover", "short", "cost", "net"])
    daily = pd.concat(frames)
    daily["cost"] = COST_TRADE * daily["turnover"] + COST_SHORT * daily["short"]
    daily["net"] = daily["gross"] - daily["cost"]
    daily.index.name = "date"
    return daily


def performance(daily: pd.DataFrame, market: pd.Series | None = None) -> dict[str, float]:
    """Annualized Sharpe ratio, mean and volatility (in %), gross and net of costs, beta and trading statistics."""
    k, out = TRADING_DAYS, {}
    for label, r in (("", daily["gross"]), ("_net", daily["net"])):
        out["SR" + label] = r.mean() / r.std() * np.sqrt(k)
        out["mu" + label] = r.mean() * k * 100
        out["sigma" + label] = r.std() * np.sqrt(k) * 100
    if market is not None:
        m = market.reindex(daily.index)
        out["beta"] = float(np.cov(daily["gross"], m)[0, 1] / m.var())
    out["turnover"] = daily["turnover"].mean()
    out["short"] = daily["short"].mean()
    out["cost_bp"] = daily["cost"].mean() * 1e4
    out["days"] = len(daily)
    return {key: float(v) for key, v in out.items()}


def summarize(run_dir: Path, first_year: int | None = None, last_year: int | None = None, save: bool = True) -> pd.DataFrame:
    """One row per seed, their mean and (for the Sharpe ratios) their standard deviation across seeds."""
    run_dir = Path(run_dir)
    cfg = Config.load(run_dir / "config.json")
    market = market_returns(cfg.universe, cfg.train_years, cfg.fill)
    rows = {}
    for seed_dir in sorted(run_dir.glob("seed*"), key=lambda p: int(p.name[4:])):
        daily = stitch(seed_dir)
        if first_year is not None:
            daily = daily[daily.index.year >= first_year]
        if last_year is not None:
            daily = daily[daily.index.year <= last_year]
        if len(daily) == 0:
            continue
        if save:
            daily.to_parquet(seed_dir / "daily.parquet")
        rows[seed_dir.name] = {**performance(daily, market), "first": str(daily.index[0].date()), "last": str(daily.index[-1].date())}
    table = pd.DataFrame(rows).T
    if len(table):
        num = table.drop(columns=["first", "last"]).astype(float)
        table.loc["mean"] = num.mean()
        table.loc["sd"] = num.std() if len(num) > 1 else np.nan
        if save:
            (run_dir / "summary.json").write_text(json.dumps({"config": cfg.to_dict(), "rows": json.loads(table.to_json(orient="index"))}, indent=2))
    return table


def compare_with_paper(table: pd.DataFrame, K: int) -> str:
    """Our seed average next to the paper's row for the same K."""
    cols = list(METRICS) + ["turnover", "short", "cost_bp"]
    ours = table.loc["mean", cols].astype(float)
    lines = [f"{'':10s}" + "".join(f"{c:>10s}" for c in cols), f"{'ours':10s}" + "".join(f"{ours[c]:10.2f}" for c in cols)]
    if K in PAPER_TABLE2:
        paper = dict(zip(METRICS, PAPER_TABLE2[K]))
        paper["cost_bp"] = (paper["mu"] - paper["mu_net"]) / TRADING_DAYS * 100      # implied average cost, bp per day
        lines.append(f"{'paper':10s}" + "".join(f"{paper[c]:10.2f}" if c in paper else f"{'':>10s}" for c in cols))
    return "\n".join(lines)
