"""The buffered list against the plain one: the same model trained on each (NOTES.md, section 8).

    python analysis/gap/scan.py train   --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none entry_rule=history
    python analysis/gap/scan.py weights --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none entry_rule=history
    python analysis/gap/buffer.py --K 30                    # -> analysis/gap/BUFFER.md (CPU, about a minute)

The plain list's runs are those of SCAN_HISTORY.md (the same two scan.py commands with universe=wrds_broad). One
setting on both lists: training cost 5 bp, ridge 1e-4, 30 epochs, the rebuilt-history entry rule, missing
characteristics at rank 0, seeds 0-2. Test years 1998-2021, holdout 2022-2025, validation 1996-1997 (6-year
windows). Every result is scored at the paper's costs (5 bp per unit of turnover, 1 bp per unit short).

Each list's portfolio holds only its own members, so the two runs trade different universes: the buffered list is
each month's top 500 plus the securities of last month's list that still rank within 600, so it contains the plain
list. Differences are paired block bootstraps (paired_test.py) of the pooled series, buffered minus plain, on the
same days. The layers (the seed ensemble, and the ensemble through the no-trade band whose width each list's own
validation years choose) are deployment layers, never Table 2 rows (deploy.py, selection.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from deploy import DELTAS, market, seed_weights  # noqa: E402
from paired_test import cell, compare, sharpe  # noqa: E402
from scan import PERIODS, SEEDS, Setting, complete  # noqa: E402
from selection import HOLDOUT, TEST, beta, describe, frame, md, paths, pooled  # noqa: E402
from variants import BLOCKS, stats  # noqa: E402

from afsa.config import TRADING_DAYS, Config  # noqa: E402
from afsa.metrics import PAPER_TABLE2, market_returns, stitch  # noqa: E402

OUT = HERE / "BUFFER.md"
SETTING = Setting(5e-4, 1e-4, 30)                    # the setting of SCAN_HISTORY.md, whose plain-list runs are on disk
PLAIN, BUFFERED = "wrds_broad", "wrds_broad_buffer600"
NAME = {PLAIN: "Plain list", BUFFERED: "Buffered list"}
ERAS = (TEST,) + BLOCKS + (HOLDOUT,)


def extra(universe: str) -> dict:
    return {"universe": universe, "fill": "none", "entry_rule": "history"}


def era_label(lo: int, hi: int) -> str:
    return f"{lo}–{hi}" if (lo, hi) == TEST else f"{lo}–{str(hi)[2:]}"


def seed_frames(universe: str, K: int, validation: bool = False) -> list[pd.DataFrame]:
    """Each seed's daily results as stored: the test and holdout years, or the validation years."""
    for period in (("validation",) if validation else ("test", "holdout")):
        if not complete(SETTING, K, period, extra(universe)):
            raise SystemExit(f"{NAME[universe]}, K = {K}: the {period} years are incomplete; train them with scan.py (see the docstring)")
    folder = SETTING.run_dir(K, PERIODS["validation" if validation else "test"][2], extra(universe))
    return [stitch(folder / f"seed{s}") for s in SEEDS]


def objects(universe: str, K: int) -> dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame], str]]:
    """object -> (daily frames of the test and holdout years, of the validation years, label): the model (one frame per
    seed), the seed ensemble, and the ensemble through the band chosen on the validation years (ties: the narrower)."""
    p, v = paths(SETTING, K, extra=extra(universe)), paths(SETTING, K, validation=True, extra=extra(universe))
    if p is None or v is None:
        raise SystemExit(f"{NAME[universe]}, K = {K}: daily weights are missing; write them with scan.py weights (see the docstring)")
    delta = max(DELTAS, key=lambda d: sharpe(pooled([frame(v, f"band {d:g}")])["net"].to_numpy()))
    return {"model": (seed_frames(universe, K), seed_frames(universe, K, validation=True), "model"),
            "ensemble": ([frame(p, "band 0")], [frame(v, "band 0")], "ensemble"),
            "band": ([frame(p, f"band {delta:g}")], [frame(v, f"band {delta:g}")], f"ensemble + band {delta:g}")}


def levels(frames: list[pd.DataFrame], val: list[pd.DataFrame], own: pd.Series, plain: pd.Series) -> dict[str, float]:
    """Seed-averaged statistics (selection.describe, beta against the list's own market), beta against the plain
    list's market, the seeds' spread of the net Sharpe ratio and the validation net Sharpe ratio."""
    d = describe(frames, own)
    test = [f[(f.index.year >= TEST[0]) & (f.index.year <= TEST[1])] for f in frames]
    d["beta plain"] = float(np.mean([beta(f, plain) for f in test]))
    d["SR_net sd"] = float(np.std([stats(f)["SR_net"] for f in test], ddof=1)) if len(test) > 1 else np.nan
    d["validation SR_net"] = float(np.mean([stats(f)["SR_net"] for f in val]))
    return d


def differences(a: tuple, b: tuple, column: str = "net") -> dict[str, str]:
    """Pooled Sharpe ratio of object a minus that of object b, validation years and each era, paired on the same days."""
    out = {"Validation 1996–97": cell(compare(pooled(a[1]), pooled(b[1]), column)["all"])}
    x, y = pooled(a[0]), pooled(b[0])
    for lo, hi in ERAS:
        m, n = (x.index.year >= lo) & (x.index.year <= hi), (y.index.year >= lo) & (y.index.year <= hi)
        out[era_label(lo, hi)] = cell(compare(x[m], y[n], column)["all"])
    return out


def turnover_parts(universe: str, K: int, first: int = TEST[0], last: int = TEST[1]) -> dict[str, float]:
    """The model's daily turnover split by what trades (seed averages from the saved weights; the first day, bought
    from cash, is left out): positions closed because the stock left the list (or delisted), positions opened in
    entrants, and trades in stocks that are members on both days; on the first trading day of a month and on others."""
    dates, permnos, R, member = market(Config(**extra(universe)), first, last)
    folder = SETTING.run_dir(K, PERIODS["test"][2], extra(universe))
    month = dates.astype("datetime64[M]")
    start = month[1:] != month[:-1]                                          # days 2..T: the month's first trading day
    leave, enter, stay = member[:-1] & ~member[1:], ~member[:-1] & member[1:], member[:-1] & member[1:]
    rows = []
    for s in SEEDS:
        W = seed_weights(folder / f"seed{s}", dates, permnos)
        if np.abs(W[~member]).max(initial=0.0) > 0:
            raise ValueError(f"{folder / f'seed{s}'}: weight on a stock outside the list")
        prev, cur = W[:-1], W[1:]
        parts = {"leave": (np.abs(prev) * leave).sum(1), "enter": (np.abs(cur) * enter).sum(1), "stay": (np.abs(cur - prev) * stay).sum(1)}
        total = parts["leave"] + parts["enter"] + parts["stay"]
        stored = stitch(folder / f"seed{s}").loc[pd.DatetimeIndex(dates[1:]), "turnover"].to_numpy()
        if np.abs(total - stored).max() > 1e-4:
            raise ValueError(f"{folder / f'seed{s}'}: the split does not add up to the stored turnover")
        rows.append({"Turnover, all days": total.mean(), "Month's first day": total[start].mean(), "Other days": total[~start].mean(),
                     "First day: leavers closed": parts["leave"][start].mean(), "First day: entrants opened": parts["enter"][start].mean(),
                     "First day: stocks on both lists": parts["stay"][start].mean(),
                     "List changes, % of turnover": 100 * (parts["leave"].sum() + parts["enter"].sum()) / total.sum()})
    out = pd.DataFrame(rows).mean().to_dict()
    out["Entrants a month"] = float(enter[start].sum(1).mean())
    out["Members a day"] = float(member.sum(1).mean())
    return out


def buffered_names(K: int, plain_sr_net: float, first: int = TEST[0], last: int = TEST[1]) -> dict[str, float]:
    """Where the buffered model's book sits: the securities on the buffered list but not on that day's plain list (held
    through the band of ranks 501-600). Seed averages from the saved weights. The flat 5 bp may undercharge these
    smaller securities, so their trades are also priced at 10 bp, and at the cost per unit of turnover that brings the
    buffered model's net Sharpe ratio down to the plain model's (`plain_sr_net`, seed average)."""
    dates, permnos, R, member = market(Config(**extra(BUFFERED)), first, last)
    pdates, ppermnos, _, pmember = market(Config(**extra(PLAIN)), first, last)
    col = np.minimum(np.searchsorted(permnos, ppermnos), len(permnos) - 1)
    if not (np.array_equal(dates, pdates) and np.array_equal(permnos[col], ppermnos)):
        raise ValueError("the two lists' arrays do not share their days or securities")
    plain = np.zeros_like(member)
    plain[:, col] = pmember
    if (plain & ~member).any():
        raise ValueError("a member of the plain list is missing from the buffered list")
    band = member & ~plain
    folder = SETTING.run_dir(K, PERIODS["test"][2], extra(BUFFERED))
    rows, nets = [], []
    for s in SEEDS:
        W = seed_weights(folder / f"seed{s}", dates, permnos)
        A, G = np.abs(W), W * R
        moved = np.abs(W[1:] - W[:-1])
        theirs = np.r_[(A[0] * band[0]).sum(), (moved * (band[1:] | band[:-1])).sum(1)]   # their trades; day 1 from cash
        nets.append((stitch(folder / f"seed{s}").loc[pd.DatetimeIndex(dates), "net"].to_numpy(), theirs))
        rows.append({"Share of gross exposure, %": 100 * (A * band).sum() / A.sum(),
                     "Net weight (long − short)": (W * band).sum(1).mean(),
                     "Share of turnover, %": 100 * theirs[1:].sum() / moved.sum(),
                     "μ from them": (G * band).sum(1).mean() * TRADING_DAYS * 100,
                     "μ from the top 500": (G * plain).sum(1).mean() * TRADING_DAYS * 100})

    def sr_net(bp: float) -> float:                                          # their trades at bp per unit instead of 5
        return float(np.mean([sharpe(n - (bp - 5) * 1e-4 * t) for n, t in nets]))

    lo, hi = 5.0, 100.0                                                      # break-even by bisection: sr_net falls with bp
    if sr_net(lo) <= plain_sr_net or sr_net(hi) > plain_sr_net:
        even = np.nan
    else:
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sr_net(mid) > plain_sr_net else (lo, mid)
        even = (lo + hi) / 2
    return {"Securities a day": float(band.sum(1).mean()), **pd.DataFrame(rows).mean().to_dict(),
            "SR net, their trades at 10 bp": sr_net(10.0), "Break-even cost of their trades, bp": even}


LEVELS = {"SR": "SR", "mu": "μ", "sigma": "σ", "mu_net": "μ net", "turnover": "Turnover", "cost/sigma": "Cost / σ", "SR_net": "SR net",
          "beta": "Beta, own list's market", "beta plain": "Beta, plain list's market", "SR_net ex market": "SR net, market removed"}
PERIOD_COLS = {"validation SR_net": "Validation SR net", "SR_net 1998-05": "1998–2005", "SR_net 2006-13": "2006–13",
               "SR_net 2014-21": "2014–21", "beta 2014-21": "Beta 2014–21", "SR_net 2022-25": "SR net 2022–25",
               "turnover 2022-25": "Turnover 2022–25", "SR_net sd": "Seed sd of SR net"}


def section(K: int) -> list[str]:
    plain_market = market_returns(PLAIN, 8, "none")
    objs = {u: objects(u, K) for u in (PLAIN, BUFFERED)}
    lev, per = {}, {}
    if K in PAPER_TABLE2:
        sr, mu, sigma, sr_net, mu_net, _, b = PAPER_TABLE2[K]
        lev["Paper, Table 2"] = {"SR": sr, "μ": mu, "σ": sigma, "μ net": mu_net, "Turnover": ((mu - mu_net) / TRADING_DAYS * 100 - 0.5) / 5,
                                 "Cost / σ": (mu - mu_net) / sigma, "SR net": sr_net, "Beta, own list's market": b}
    for key in ("model", "ensemble", "band"):
        for u in (PLAIN, BUFFERED):
            frames, val, label = objs[u][key]
            d = levels(frames, val, market_returns(u, 8, "none"), plain_market)
            name = f"{NAME[u]}: {label}"
            lev[name] = {v: d[k] for k, v in LEVELS.items() if k in d}
            per[name] = {v: d[k] for k, v in PERIOD_COLS.items() if k in d and not (key != "model" and k == "SR_net sd")}
    diff = {"model, gross": differences(objs[BUFFERED]["model"], objs[PLAIN]["model"], "gross")}
    for key in ("model", "ensemble", "band"):
        diff[f"{objs[BUFFERED][key][2]} (buffered) vs {objs[PLAIN][key][2]} (plain)" if key == "band" else key] = \
            differences(objs[BUFFERED][key], objs[PLAIN][key])
    turn = {NAME[u]: turnover_parts(u, K) for u in (PLAIN, BUFFERED)}
    return [f"## K = {K}",
            "**1998–2021.** Model rows are seed averages (the paper's object); the ensemble and the band are deployment layers, not Table 2 rows, "
            "and each list's band width is the one its own validation years choose. μ and σ in % a year, turnover per day. The paper's turnover "
            "is implied by its costs, with a short side of 0.5. Each list's beta is against its own equal-weighted excess return, and also against "
            "the plain list's, the paper's market row.",
            md(pd.DataFrame(lev).T, "Run"),
            "**By period.** Net Sharpe ratios, seed averages for the model; validation is 1996–1997 with 6-year windows.",
            md(pd.DataFrame(per).T, "Run"),
            "**Buffered minus plain** (pooled series, paired block bootstrap, 95% interval). Net Sharpe ratio unless the row says gross.",
            md(pd.DataFrame(diff).T, "Object"),
            "**Turnover of the model, 1998–2021** (seed averages from the saved weights). On the first trading day of a month the list changes: "
            "positions in securities that left are closed and entrants are bought. *List changes* counts these closes and opens on every day "
            "(delistings included).",
            md(pd.DataFrame(turn).T, "List"),
            "**The buffered model's book in the securities held through the band** (on the buffered list, not on that day's plain list), "
            "1998–2021, seed averages. μ in % a year, gross. The flat 5 bp may undercharge these smaller securities: the last two columns "
            "price their trades (opening, closing and rebalancing) at 10 bp, and at the cost that brings the model's net Sharpe ratio down to "
            "the plain list's.",
            md(pd.DataFrame({"Buffered list": buffered_names(K, lev[f"{NAME[PLAIN]}: model"]["SR net"])}).T, "")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, nargs="+", default=[30])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    parts = ["# The buffered list against the plain one",
             "Generated by `analysis/gap/buffer.py`; the list is described in `NOTES.md` §8. One setting on both lists: training cost 5 bp, ridge "
             "1e-4, 30 epochs, the rebuilt-history entry rule (`--entry_rule history`), missing characteristics at rank 0, seeds 0–2. "
             "Every result is scored at the paper's costs (5 bp per unit of turnover, 1 bp per unit short), on stock-by-stock excess returns.",
             f"- **Plain list** (`{PLAIN}`): each month's 500 largest securities, the paper's universe (the runs of SCAN_HISTORY.md).",
             f"- **Buffered list** (`{BUFFERED}`): each month's 500 largest plus the securities of last month's list that still rank within 600 "
             "(500 to 564 a month). It contains the plain list. A robustness variant, not a Table 2 row: it departs from the paper's 500 largest, "
             "and the flat 5 bp cost does not charge more for the smaller securities ranked 501–600."]
    for K in args.K:
        parts += section(K)
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
