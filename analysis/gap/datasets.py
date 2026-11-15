"""The four `broad` datasets built from WRDS compared as a 2x2 (CLAUDE.md section 7, 2026-11-15): the plain or the
buffered list, crossed with the 30-day wait or rebuilt history. One setting on all four.

    python analysis/gap/scan.py train   --K 30 --only 5/1e-4/30 --set universe=<universe> fill=none [entry_rule=history]
    python analysis/gap/scan.py weights --K 30 --only 5/1e-4/30 --set universe=<universe> fill=none [entry_rule=history]
    python analysis/gap/datasets.py --K 30                  # -> analysis/gap/DATASETS.md (CPU, about two minutes)

The datasets (universe wrds_broad or wrds_broad_buffer600, fill none):
    A  plain list,    30-day wait       Data/TrainingData/wrds_broad_fill-none (day x slot arrays)
    B  plain list,    rebuilt history   Data/TrainingData/history/wrds_broad_fill-none (month blocks)
    C  buffered list, 30-day wait       Data/TrainingData/wrds_broad_buffer600_fill-none
    D  buffered list, rebuilt history   Data/TrainingData/history/wrds_broad_buffer600_fill-none

Setting: training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0-2. Test years 1998-2021 and 2022-2025 (8-year windows),
validation 1996-1997 (6-year windows). Every result is scored at the paper's costs (5 bp per unit of turnover, 1 bp
per unit short). Objects: the model (statistics averaged over seeds, the paper's object), the seed ensemble, and the
ensemble through the no-trade band whose width each dataset's own validation years choose (deploy.py); the last two
are deployment layers, never Table 2 rows.

Differences are circular block bootstraps of the pooled seed series (paired_test.py: 21-day blocks, 2,000 draws, the
same stream), every series resampled on the same days. A contrast is a sum of Sharpe ratios with signs, so the
interaction (D - C) - (B - A) is tested like a difference. A comparison, not a selection: CLAUDE.md section 8 reserves
choices for the validation years, and Table 2 stays on the plain list with rebuilt history.
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
from paired_test import block_draws, sharpe  # noqa: E402
from scan import PERIODS, SEEDS, Setting, complete  # noqa: E402
from selection import HOLDOUT, TEST, beta, describe, frame, md, paths, pooled, seed_frames  # noqa: E402
from variants import BLOCKS, stats  # noqa: E402

from afsa.config import TRADING_DAYS, TRAIN_DATA_DIR, Config, data_folder  # noqa: E402
from afsa.metrics import PAPER_TABLE2, market_returns, stitch  # noqa: E402

OUT = HERE / "DATASETS.md"
SETTING = Setting(5e-4, 1e-4, 30)
PLAIN, BUFFERED = "wrds_broad", "wrds_broad_buffer600"
CELLS = {                                                                    # key -> (label, universe, entry rule)
    "A": ("A: plain list, 30-day wait", PLAIN, "run"),
    "B": ("B: plain list, rebuilt history", PLAIN, "history"),
    "C": ("C: buffered list, 30-day wait", BUFFERED, "run"),
    "D": ("D: buffered list, rebuilt history", BUFFERED, "history"),
}
TWIN = {"C": "A", "D": "B"}                                                  # the plain-list dataset under the same entry rule
SPANS = (("Validation 1996–97", None), ("1998–2021", TEST)) + tuple((f"{a}–{str(b)[2:]}", (a, b)) for a, b in BLOCKS) + \
        (("2022–25", HOLDOUT), ("1998–2025", (TEST[0], HOLDOUT[1])))
CONTRASTS = (
    ("Rebuilt history on the plain list (B − A)", ("B",), ("A",)),
    ("Rebuilt history on the buffered list (D − C)", ("D",), ("C",)),
    ("Buffered list under the 30-day wait (C − A)", ("C",), ("A",)),
    ("Buffered list under rebuilt history (D − B)", ("D",), ("B",)),
    ("Interaction: (D − C) − (B − A)", ("D", "A"), ("C", "B")),
    ("Both changes (D − A)", ("D",), ("A",)),
)


def extra(key: str) -> dict:
    _, universe, rule = CELLS[key]
    return {"universe": universe, "fill": "none", **({} if rule == "run" else {"entry_rule": rule})}


def objects(key: str, K: int) -> dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame], str]]:
    """object -> (daily frames of 1998-2025, of the validation years, label): the model (one frame per seed, as
    stored), the seed ensemble, and the ensemble through the band its own validation years choose (ties: narrower)."""
    for period in PERIODS:
        if not complete(SETTING, K, period, extra(key)):
            raise SystemExit(f"{CELLS[key][0]}, K = {K}: the {period} years are incomplete; train them with scan.py (see the docstring)")
    p, v = paths(SETTING, K, extra=extra(key)), paths(SETTING, K, validation=True, extra=extra(key))
    if p is None or v is None:
        raise SystemExit(f"{CELLS[key][0]}, K = {K}: daily weights are missing; write them with scan.py weights (see the docstring)")
    if p.index[-1].year != HOLDOUT[1]:
        raise SystemExit(f"{CELLS[key][0]}, K = {K}: the daily paths stop in {p.index[-1].year}")
    delta = max(DELTAS, key=lambda d: sharpe(pooled([frame(v, f"band {d:g}")])["net"].to_numpy()))
    return {"model": (seed_frames(p), seed_frames(v), "model"),
            "ensemble": ([frame(p, "band 0")], [frame(v, "band 0")], "ensemble"),
            "band": ([frame(p, f"band {delta:g}")], [frame(v, f"band {delta:g}")], f"ensemble + band {delta:g}")}


def levels(frames: list[pd.DataFrame], val: list[pd.DataFrame], own: pd.Series, plain: pd.Series) -> dict[str, float]:
    """Seed-averaged statistics. Beta and the market-removed Sharpe ratio are against the plain list's equal-weighted
    excess return (the paper's market row, the same for all four); beta also against the dataset's own list."""
    d = describe(frames, plain)
    span = [f[(f.index.year >= TEST[0]) & (f.index.year <= TEST[1])] for f in frames]
    d["beta own"] = float(np.mean([beta(f, own) for f in span]))
    d["SR_net 1998-2025"] = float(np.mean([stats(f[f.index.year >= TEST[0]])["SR_net"] for f in frames]))
    d["SR_net sd"] = float(np.std([stats(f)["SR_net"] for f in span], ddof=1)) if len(span) > 1 else np.nan
    d["validation SR_net"] = float(np.mean([stats(f)["SR_net"] for f in val]))
    return d


def contrast(objs: dict, plus: tuple[str, ...], minus: tuple[str, ...], obj: str, column: str = "net") -> dict[str, str]:
    """Sum of the pooled Sharpe ratios of `plus` minus those of `minus`, in each span, with its 95% bootstrap interval;
    all series are resampled on the same days."""
    out = {}
    for name, span in SPANS:
        xs = []
        for k in plus + minus:
            frames, val, _ = objs[k][obj]
            x = pooled(val if span is None else frames)[column]
            xs.append(x if span is None else x[(x.index.year >= span[0]) & (x.index.year <= span[1])])
        days = xs[0].index
        for x in xs[1:]:
            days = days.intersection(x.index)
        if any(len(x) != len(days) for x in xs):
            raise ValueError(f"{name}: the datasets do not cover the same days")
        a = np.stack([x.loc[days].to_numpy() for x in xs])
        sign = np.r_[np.ones(len(plus)), -np.ones(len(minus))]
        idx = block_draws(len(days))
        draws = sum(s * sharpe(row[idx]) for s, row in zip(sign, a))
        point = float(sum(s * sharpe(row) for s, row in zip(sign, a)))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[name] = f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    return out


def signal_masks(universe: str, dates: np.ndarray, permnos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[T, S] on `dates`: membership, a signal under the 30-day wait (in the universe on each of the 30 days before)
    and a signal under rebuilt history (listed for more than 30 trading days, accounts entered), from the own rows of
    the month blocks (afsa.build_history), which hold both rules' counters."""
    folder = TRAIN_DATA_DIR / "history" / data_folder(universe, "none")
    L = Config().lookback
    bdates = np.load(folder / "dates.npy").astype("datetime64[D]")
    row, n_rows, permno = np.load(folder / "row.npy"), np.load(folder / "n_rows.npy"), np.load(folder / "permno.npy")
    member, listed, run = (np.load(folder / f"{n}.npy", mmap_mode="r") for n in ("member", "listed", "run"))
    accounts = np.load(folder / "accounts.npy")
    T, S = len(dates), len(permnos)
    M, wait, hist = (np.zeros((T, S), dtype=bool) for _ in range(3))
    for m in range(len(row)):
        own = np.arange(L, int(n_rows[m]))
        d = bdates[row[m, own]]
        t = np.minimum(np.searchsorted(dates, d), T - 1)
        on = dates[t] == d
        if not on.any():
            continue
        slots = np.nonzero(permno[m] >= 0)[0]
        c = np.minimum(np.searchsorted(permnos, permno[m, slots]), S - 1)
        known = permnos[c] == permno[m, slots]
        r, s = own[on], slots[known]
        mem = np.asarray(member[m])[np.ix_(r, s)]
        if (np.asarray(member[m])[np.ix_(r, slots[~known])]).any():
            raise ValueError(f"block {m}: a member is missing from the day arrays' securities")
        cell = np.ix_(t[on], c[known])
        M[cell] = mem
        wait[cell] = mem & (np.asarray(run[m])[np.ix_(r, s)] > L)
        hist[cell] = mem & (np.asarray(listed[m])[np.ix_(r, s)] > L) & accounts[m, s][None, :]
    return M, wait, hist


def turnover_split(W: np.ndarray, member: np.ndarray, stored: np.ndarray) -> tuple[float, float]:
    """Mean daily turnover and the share (%) from list changes: positions closed because the stock left the list (or
    delisted) and positions opened in entrants. The first day, bought from cash, is left out; `stored` is the stored
    turnover of days 2..T, which the split must reproduce."""
    prev, cur = W[:-1], W[1:]
    leave, enter, stay = member[:-1] & ~member[1:], ~member[:-1] & member[1:], member[:-1] & member[1:]
    parts = (np.abs(prev) * leave).sum(1), (np.abs(cur) * enter).sum(1), (np.abs(cur - prev) * stay).sum(1)
    total = parts[0] + parts[1] + parts[2]
    if np.abs(total - stored).max() > 1e-4:
        raise ValueError("the turnover split does not add up to the stored turnover")
    return float(total.mean()), float(100 * (parts[0].sum() + parts[1].sum()) / total.sum())


def mechanism(key: str, K: int, first: int = TEST[0], last: int = TEST[1]) -> dict[str, float]:
    """Where the model's book and return sit, 1998-2021, seed averages from the saved weights. *Waiting* stock-days:
    members with a signal under rebuilt history but not under the 30-day wait (mostly the first 30 days after
    entering the list). Under the 30-day wait the model holds them only through the hedge."""
    _, universe, rule = CELLS[key]
    dates, permnos, R, member = market(Config(universe=universe, fill="none"), first, last)
    M, wait, hist = signal_masks(universe, dates, permnos)
    if not np.array_equal(M, member):
        raise ValueError(f"{universe}: the month blocks' own rows and the day arrays disagree on membership")
    waiting = member & hist & ~wait
    signal = hist if rule == "history" else wait
    folder = SETTING.run_dir(K, PERIODS["test"][2], extra(key))
    month = dates.astype("datetime64[M]")
    start = month[1:] != month[:-1]
    rows = []
    for s in SEEDS:
        W = seed_weights(folder / f"seed{s}", dates, permnos)
        if np.abs(W[~member]).max(initial=0.0) > 0:
            raise ValueError(f"{folder / f'seed{s}'}: weight on a stock outside the list")
        A, G = np.abs(W), W * R
        stored = stitch(folder / f"seed{s}").loc[pd.DatetimeIndex(dates[1:]), "turnover"].to_numpy()
        turnover, changes = turnover_split(W, member, stored)
        rows.append({"Turnover": turnover, "List changes, % of turnover": changes,
                     "Waiting: % of gross exposure": 100 * (A * waiting).sum() / A.sum(),
                     "Waiting: μ": (G * waiting).sum(1).mean() * TRADING_DAYS * 100,
                     "Other stock-days: μ": (G * (member & ~waiting)).sum(1).mean() * TRADING_DAYS * 100})
    entrants = (~member[:-1] & member[1:])[start].sum(1).mean()
    return {"Members a day": float(member.sum(1).mean()), "Entrants a month": float(entrants),
            "No signal, % of member stock-days": float(100 * (member & ~signal).sum() / member.sum()),
            "Waiting, % of member stock-days": float(100 * waiting.sum() / member.sum()), **pd.DataFrame(rows).mean().to_dict()}


def buffer_book(key: str, K: int, twin_sr_net: float, first: int = TEST[0], last: int = TEST[1]) -> dict[str, float]:
    """A buffered dataset's book in the securities held through the band (on the buffered list, not on that day's plain
    list), 1998-2021, seed averages. Their trades are also priced at 10 bp, and at the cost per unit of turnover that
    brings the model's net Sharpe ratio down to its plain-list twin's under the same entry rule (`twin_sr_net`)."""
    dates, permnos, R, member = market(Config(universe=BUFFERED, fill="none"), first, last)
    pdates, ppermnos, _, pmember = market(Config(universe=PLAIN, fill="none"), first, last)
    col = np.minimum(np.searchsorted(permnos, ppermnos), len(permnos) - 1)
    if not (np.array_equal(dates, pdates) and np.array_equal(permnos[col], ppermnos)):
        raise ValueError("the two lists' arrays do not share their days or securities")
    plain = np.zeros_like(member)
    plain[:, col] = pmember
    if (plain & ~member).any():
        raise ValueError("a member of the plain list is missing from the buffered list")
    band = member & ~plain
    folder = SETTING.run_dir(K, PERIODS["test"][2], extra(key))
    rows, nets = [], []
    for s in SEEDS:
        W = seed_weights(folder / f"seed{s}", dates, permnos)
        A, G = np.abs(W), W * R
        moved = np.abs(W[1:] - W[:-1])
        theirs = np.r_[(A[0] * band[0]).sum(), (moved * (band[1:] | band[:-1])).sum(1)]   # their trades; day 1 from cash
        nets.append((stitch(folder / f"seed{s}").loc[pd.DatetimeIndex(dates), "net"].to_numpy(), theirs))
        rows.append({"Share of gross exposure, %": 100 * (A * band).sum() / A.sum(),
                     "Share of turnover, %": 100 * theirs[1:].sum() / moved.sum(),
                     "μ from them": (G * band).sum(1).mean() * TRADING_DAYS * 100})

    def sr_net(bp: float) -> float:                                          # their trades at bp per unit instead of 5
        return float(np.mean([sharpe(n - (bp - 5) * 1e-4 * t) for n, t in nets]))

    lo, hi = 5.0, 100.0                                                      # break-even by bisection: sr_net falls with bp
    if sr_net(lo) <= twin_sr_net or sr_net(hi) > twin_sr_net:
        even = np.nan
    else:
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sr_net(mid) > twin_sr_net else (lo, mid)
        even = (lo + hi) / 2
    return {"Securities a day": float(band.sum(1).mean()), **pd.DataFrame(rows).mean().to_dict(),
            "SR net (5 bp)": sr_net(5.0), "SR net, their trades at 10 bp": sr_net(10.0),
            "SR net of the plain twin (C: A, D: B)": twin_sr_net, "Break-even cost of their trades, bp": even}


LEVELS = {"SR": "SR", "mu": "μ", "sigma": "σ", "mu_net": "μ net", "turnover": "Turnover", "cost/sigma": "Cost / σ", "SR_net": "SR net",
          "beta": "Beta", "beta own": "Beta, own list's market", "SR_net ex market": "SR net, market removed"}
PERIOD_COLS = {"validation SR_net": "Validation 1996–97", "SR_net 1998-05": "1998–2005", "SR_net 2006-13": "2006–13",
               "SR_net 2014-21": "2014–21", "SR_net 2022-25": "2022–25", "SR_net 1998-2025": "1998–2025",
               "beta 2014-21": "Beta 2014–21", "turnover 2022-25": "Turnover 2022–25", "SR_net sd": "Seed sd, 1998–2021"}
OBJECT_NAMES = {"model": "model", "ensemble": "ensemble", "band": "ensemble + band"}


def section(K: int) -> list[str]:
    plain_market = market_returns(PLAIN, 8, "none")
    objs = {k: objects(k, K) for k in CELLS}
    lev, per = {}, {}
    if K in PAPER_TABLE2:
        sr, mu, sigma, sr_net, mu_net, _, b = PAPER_TABLE2[K]
        lev["Paper, Table 2"] = {"SR": sr, "μ": mu, "σ": sigma, "μ net": mu_net, "Turnover": ((mu - mu_net) / TRADING_DAYS * 100 - 0.5) / 5,
                                 "Cost / σ": (mu - mu_net) / sigma, "SR net": sr_net, "Beta": b}
    for obj in ("model", "ensemble", "band"):
        for k in CELLS:
            frames, val, label = objs[k][obj]
            d = levels(frames, val, market_returns(CELLS[k][1], 8, "none"), plain_market)
            name = f"{CELLS[k][0]}: {label}"
            lev[name] = {v: d[c] for c, v in LEVELS.items() if c in d}
            per[name] = {v: d[c] for c, v in PERIOD_COLS.items() if c in d and not (obj != "model" and c == "SR_net sd")}
    tables = []
    for obj, column, title in (("model", "net", "Model, net"), ("model", "gross", "Model, gross"),
                               ("ensemble", "net", "Ensemble, net"), ("band", "net", "Ensemble + band, net")):
        t = pd.DataFrame({name: contrast(objs, plus, minus, obj, column) for name, plus, minus in CONTRASTS}).T
        tables += [f"*{title}.*" + (" Each dataset's band width is the one its own validation years choose." if obj == "band" else ""),
                   md(t, "Contrast")]
    mech = {CELLS[k][0]: mechanism(k, K) for k in CELLS}
    book = {CELLS[k][0]: buffer_book(k, K, lev[f"{CELLS[TWIN[k]][0]}: model"]["SR net"]) for k in TWIN}
    widths = ", ".join(f"{k} {objs[k]['band'][2].split()[-1]}" for k in CELLS)
    return [f"## K = {K}",
            "**1998–2021.** Model rows are seed averages (the paper's object); the ensemble and the band are deployment layers, not Table 2 rows. "
            f"Band widths chosen on each dataset's own validation years: {widths}. μ and σ in % a year, turnover per day. The paper's turnover "
            "is implied by its costs, with a short side of 0.5. Beta and the market-removed Sharpe ratio are against the plain list's "
            "equal-weighted excess return (the paper's market row) for all four; the buffered datasets' beta against their own list is "
            "shown too.",
            md(pd.DataFrame(lev).T, "Run"),
            "**By period.** Net Sharpe ratios, seed averages for the model. Validation is 1996–1997 with 6-year windows; 2022–2025 is "
            "the extension.",
            md(pd.DataFrame(per).T, "Run"),
            "**Contrasts** (pooled seed series, paired block bootstrap on the same days, 95% interval). A positive contrast favours the "
            "first-named dataset; the interaction is positive when rebuilt history gains more on the buffered list than on the plain one.",
            *tables,
            "**Where the book and the return sit** (the model, 1998–2021, seed averages from the saved weights). *Waiting* stock-days are "
            "members that rebuilt history gives a signal and the 30-day wait does not: mostly their first 30 days on the list. Under the "
            "30-day wait the model holds them only through the hedge. *No signal* is under the dataset's own rule (rebuilt history: "
            "listed for 30 trading days or fewer, or no accounts yet). μ is gross, in % a year; *List changes* counts positions closed "
            "in leavers and opened in entrants (delistings included).",
            md(pd.DataFrame(mech).T, "Dataset"),
            "**The buffered datasets' book in the securities held through the band** (on the buffered list, not on that day's plain "
            "list), 1998–2021, seed averages. The flat 5 bp may undercharge these smaller securities: their trades are also priced at "
            "10 bp, and at the cost that brings the model's net Sharpe ratio down to its plain-list twin's under the same entry rule.",
            md(pd.DataFrame(book).T, "Dataset")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, nargs="+", default=[30])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    parts = ["# The four `broad` datasets from WRDS",
             "Generated by `analysis/gap/datasets.py` (design: CLAUDE.md §7, 2026-11-15). One setting on all four: training cost 5 bp, "
             "ridge 1e-4, 30 epochs, missing characteristics at rank 0, seeds 0–2; test years 1998–2021 and 2022–2025 with 8-year windows, "
             "validation 1996–1997 with 6-year windows. Every result is scored at the paper's costs (5 bp per unit of turnover, 1 bp per "
             "unit short), on stock-by-stock excess returns.",
             "| Dataset | List | Entry rule | Training data |\n| --- | --- | --- | --- |\n"
             "| A | plain: each month's 500 largest (the paper's universe) | 30-day wait: a signal from the 31st consecutive day on the list | `wrds_broad_fill-none` (day × slot arrays) |\n"
             "| B | plain | rebuilt history: lags recomputed over the day's list; a signal once listed 30 days with accounts entered | `history\\wrds_broad_fill-none` (month blocks) |\n"
             "| C | buffered: the 500 largest plus last month's members still ranked within 600 | 30-day wait | `wrds_broad_buffer600_fill-none` |\n"
             "| D | buffered | rebuilt history | `history\\wrds_broad_buffer600_fill-none` |",
             "A comparison, not a selection (CLAUDE.md §8): Table 2 stays on the plain list with rebuilt history (B), and the buffered "
             "list stays a robustness variant. The buffered datasets trade a larger list (about 545 securities a day), and the flat 5 bp "
             "does not charge more for the securities ranked 501–600."]
    for K in args.K:
        parts += section(K)
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
