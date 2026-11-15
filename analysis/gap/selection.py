"""Selection rules on the hyperparameter scan of scan.py: the setting with the best net Sharpe ratio, chosen without
look-ahead (Zhishu Zhang's review, P5 and section 6; CLAUDE.md, section 8).

    python analysis/gap/selection.py --K 8 30                 # -> analysis/gap/SCAN.md
    python analysis/gap/selection.py --K 1 3 5 8 15 30 100    # every K, and the choice of K itself
    python analysis/gap/selection.py --arrays broad wrds_broad wrds_common --K 30   # the chosen setting on other arrays -> SCAN_ARRAYS.md
    python analysis/gap/selection.py --K 30 --only 7.5/1e-5/30 7.5/1e-4/30 --set universe=wrds_broad fill=none --seeds 0 1 2 3 4 \
        --out analysis/gap/SCAN_RIDGE.md                                                 # any settings, arrays and seeds
    python analysis/gap/selection.py --per-k 8=5/1e-4/80 30=5/3e-4/30 --set ... --out ...  # each K at its own setting (pilot.py --best)

Objects, each traded at the paper's costs (5 bp per unit of turnover, 1 bp per unit short):
- model:            each seed's portfolio on its own, statistics averaged over the seeds. The paper's object.
- ensemble:         the seeds' weights averaged stock by stock and rescaled to unit gross exposure. A deployment
                    layer (deploy.py).
- ensemble + band:  the ensemble traded through a no-trade band, whose width is chosen with the setting. A layer.

A setting is a training cost, a ridge and a number of epochs (scan.py); for the last object also a band width. The
rules, each a labelled row:
- validation:   the best net Sharpe ratio on 1996-1997 (6-year windows), kept for every later year. This is how the
                paper chose its parameters (its Appendix B) and the choice CLAUDE.md section 8 allows.
- walk-forward: at the start of each year, the best net Sharpe ratio over the test years before it (1998 takes the
                validation choice). The book is carried over, so a change of setting is traded and paid for. It uses
                nothing from the year it trades or later, but it chooses from test years: the review's rule (P5).
- hindsight:    the best fixed setting over 1998-2021. A diagnostic: an upper bound, inflated by the choice among the
                grid (the reality check says by how much). It is also the walk-forward choice for 2022, so its
                2022-2025 figures are out of sample.
Only walk-forward makes no use of the test years it is scored on; 2022-2025 is out of sample for all three rules.
With more than three seeds, each K also gets the ensemble's net Sharpe ratio against the number of seeds it averages
(seed_count_section).

Criterion: the net Sharpe ratio of the object (for the model, the average over seeds of each seed's). Ties go to the
earlier setting of scan.GRID, then to the narrower band. Paired tests (paired_test.py) compare pooled series: for the
model the average of the seeds' daily returns, for a layer its own series, always against the baseline model pooled.
"""
from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy import DELTAS, band_path, ensemble, market, seed_weights, trade  # noqa: E402
from paired_test import block_draws, cell, compare, sharpe  # noqa: E402
from scan import (BASELINE, BOTH, GRID, PERIODS, REPLAY_TOLERANCE, SEEDS, Setting, complete, has_weights,  # noqa: E402
                  parse_set, parse_setting)
from variants import BLOCKS, stats  # noqa: E402

from afsa.config import Config, data_folder  # noqa: E402
from afsa.metrics import COST_SHORT, COST_TRADE, PAPER_TABLE2, market_returns, stitch  # noqa: E402

OUT = Path(__file__).resolve().parent / "SCAN.md"
TEST, HOLDOUT = (1998, 2021), (2022, 2025)
OBJECTS = ("model", "ensemble", "ensemble + band")
RULES = ("validation", "walk-forward", "hindsight")
FIELDS = ("gross", "turnover", "short")


@dataclass(frozen=True)
class Candidate:
    K: int
    setting: Setting
    delta: float = 0.0                                                       # band width; 0 = no band

    def code(self, with_k: bool = False) -> str:
        s = self.setting
        out = f"{s.cost_trade * 1e4:g} bp, {s.ridge:g}, {s.epochs} ep" + (f", band {self.delta:g}" if self.delta else "")
        return (f"K = {self.K}: " if with_k else "") + out


# ---------------------------------------------------------------------------------------------- daily paths
@functools.lru_cache(maxsize=8)
def _market(first: int, last: int, universe: str = "broad", fill: str = "last"):
    return market(Config(universe=universe, fill=fill), first, last)        # the scan's runs trade scan.UNIVERSE


def paths(setting: Setting, K: int, validation: bool = False, extra: dict | None = None,
          seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame | None:
    """Daily gross return, turnover and short exposure of every object of one run from its first test day, columns
    '<object>|<field>': each seed as stored, and the seeds' ensemble through each band width ('band 0' = no band).
    The test run includes 2022-2025 when they are on disk. `extra`: other arrays (scan.Setting.config). Cached in the
    run folder: scan_paths.parquet for seeds 0-2, scan_paths_seeds-<list>.parquet for others."""
    period = "validation" if validation else "test"
    if not (complete(setting, K, period, extra, seeds) and has_weights(setting, K, period, extra, seeds)):
        return None
    first, last, train_years = PERIODS[period]
    if not validation and complete(setting, K, "holdout", extra, seeds) and has_weights(setting, K, "holdout", extra, seeds):
        last = HOLDOUT[1]
    folder, cfg = setting.run_dir(K, train_years, extra), setting.config(K, train_years, extra)
    cache = folder / ("scan_paths.parquet" if tuple(seeds) == SEEDS else f"scan_paths_seeds-{'-'.join(map(str, seeds))}.parquet")
    if cache.exists():
        p = pd.read_parquet(cache)
        if (p.index[0].year, p.index[-1].year) == (first, last):
            return p
    dates, permnos, R, member = _market(first, last, cfg.universe, cfg.fill)
    cols, Ws = {}, []
    for s in seeds:
        seed_dir = folder / f"seed{s}"
        W = seed_weights(seed_dir, dates, permnos)
        stored, mine = stitch(seed_dir).loc[pd.DatetimeIndex(dates)], trade(W, R, dates)
        gap = max(float(np.abs(mine["gross"] - stored["gross"]).max()), float(np.abs(mine["turnover"] - stored["turnover"]).max()))
        if gap > REPLAY_TOLERANCE:
            raise ValueError(f"{seed_dir}: the saved weights do not reproduce the stored results (largest difference {gap:.1e})")
        for f in FIELDS:
            cols[f"seed{s}|{f}"] = stored[f].to_numpy()
        Ws.append(W)
    E = ensemble(Ws)
    for d in DELTAS:
        daily = trade(E if d == 0 else band_path(E, member, d), R, dates)
        for f in FIELDS:
            cols[f"band {d:g}|{f}"] = daily[f].to_numpy()
    p = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    p.to_parquet(cache)
    return p


def frame(p: pd.DataFrame, obj: str) -> pd.DataFrame:
    """One object's columns of a paths frame, as a daily frame with columns gross, turnover and short."""
    return p[[f"{obj}|{f}" for f in FIELDS]].set_axis(list(FIELDS), axis=1)


def seed_names(p: pd.DataFrame) -> list[str]:
    return sorted({c.split("|")[0] for c in p.columns if c.startswith("seed")}, key=lambda s: int(s[4:]))


def seed_frames(p: pd.DataFrame) -> list[pd.DataFrame]:
    return [frame(p, s) for s in seed_names(p)]


def net(p: pd.DataFrame, obj: str) -> np.ndarray:
    return (p[f"{obj}|gross"] - COST_TRADE * p[f"{obj}|turnover"] - COST_SHORT * p[f"{obj}|short"]).to_numpy()


def object_net(p: pd.DataFrame, obj: str, delta: float = 0.0) -> np.ndarray:
    """Daily net returns [T, columns]: one column per seed for the model, one column for a layer."""
    if obj == "model":
        return np.stack([net(p, s) for s in seed_names(p)], axis=1)
    return net(p, f"band {delta:g}")[:, None]


def frames_net(frames: list[pd.DataFrame], dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily net returns [T, frames] of daily frames, on `dates`."""
    return np.stack([(f["gross"] - COST_TRADE * f["turnover"] - COST_SHORT * f["short"]).reindex(dates).to_numpy() for f in frames], axis=1)


# ---------------------------------------------------------------------------------------------- the rules
def mean_sharpe(x: np.ndarray) -> float:
    """Average over columns of each column's Sharpe ratio (daily; only the ranking matters)."""
    sd = x.std(0, ddof=1)
    return float(np.mean(np.where(sd > 0, x.mean(0) / np.where(sd > 0, sd, 1.0), -np.inf)))


def best(nets: dict[Candidate, np.ndarray]) -> Candidate:
    """The candidate with the highest criterion; the first one in the dict on a tie."""
    return max(nets, key=lambda c: mean_sharpe(nets[c]))


def walk_forward_plan(nets: dict[Candidate, np.ndarray], years: np.ndarray, start: Candidate) -> dict[int, Candidate]:
    """For each year, the candidate with the best criterion over the years before it; the first year takes `start`.
    `nets` maps each candidate to its daily net returns [T, columns] on the days whose years are `years`."""
    cands, uniq = list(nets), np.unique(years)
    mom = np.zeros((len(cands), len(uniq), 3, nets[cands[0]].shape[1]))     # [candidate, year, (n, sum, sum of squares), column]
    for j, y in enumerate(uniq):
        rows = years == y
        for i, c in enumerate(cands):
            x = nets[c][rows]
            mom[i, j, 0], mom[i, j, 1], mom[i, j, 2] = len(x), x.sum(0), (x * x).sum(0)
    plan = {int(uniq[0]): start}
    for j in range(1, len(uniq)):
        n, s1, s2 = mom[:, :j].sum(1).transpose(1, 0, 2)                     # each [candidate, column]
        var = (s2 - s1 * s1 / n) / (n - 1)
        sr = np.where(var > 0, (s1 / n) / np.sqrt(np.where(var > 0, var, 1.0)), -np.inf)
        plan[int(uniq[j])] = cands[int(np.argmax(sr.mean(1)))]
    return plan


def realize(plan: dict[int, Candidate], obj: str, first: int, last: int, extra: dict | None = None,
            seeds: tuple[int, ...] = SEEDS) -> list[pd.DataFrame]:
    """Daily results of following `plan` (a candidate per year) from `first` to `last`: one frame per seed for the
    model, one frame for a layer. Each year's targets come from the chosen run's saved weights, and the book held at
    the end of a year is where the next year starts, so a change of setting pays its trades."""
    cfg = plan[first].setting.config(plan[first].K, 8, extra)
    dates, permnos, R, member = _market(first, last, cfg.universe, cfg.fill)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    books = [np.zeros((len(dates), len(permnos))) for _ in (seeds if obj == "model" else (0,))]
    h = None
    for y in range(first, last + 1):
        rows, c = years == y, plan[y]
        Ws = [seed_weights(c.setting.run_dir(c.K, 8, extra) / f"seed{s}", dates[rows], permnos) for s in seeds]
        if obj == "model":
            for book, W in zip(books, Ws):
                book[rows] = W
            continue
        E = ensemble(Ws)
        books[0][rows] = E if obj == "ensemble" else band_path(E, member[rows], c.delta, h)
        h = books[0][rows][-1]
    return [trade(book, R, dates) for book in books]


@functools.lru_cache(maxsize=4)
def _market_row(universe: str = "broad", fill: str = "last") -> pd.Series:
    return market_returns(universe, 8, fill)                                 # equal-weighted excess return: the paper's beta


def beta(f: pd.DataFrame, market: pd.Series | None = None) -> float:
    m = (_market_row() if market is None else market).reindex(f.index)
    return float(np.cov(f["gross"], m)[0, 1] / m.var())


def sharpe_ex_market(f: pd.DataFrame, market: pd.Series | None = None) -> float:
    """Annualized net Sharpe ratio after removing beta x market (beta of the net return over the same days: a
    diagnostic of market exposure, as in exposure.py, not a tradable hedge)."""
    m = (_market_row() if market is None else market).reindex(f.index).to_numpy()
    n = (f["gross"] - COST_TRADE * f["turnover"] - COST_SHORT * f["short"]).to_numpy()
    x = n - np.cov(n, m)[0, 1] / m.var(ddof=1) * m
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def describe(frames: list[pd.DataFrame], market: pd.Series | None = None) -> dict[str, float]:
    """Seed-averaged statistics over 1998-2021, its eras, and 2022-2025 when present. Beta is against `market`
    (default: the equal-weighted `broad` universe)."""
    rows = []
    for f in frames:
        y = f.index.year
        s = stats(f[(y >= TEST[0]) & (y <= TEST[1])])
        s["beta"], s["beta 2014-21"] = beta(f[(y >= TEST[0]) & (y <= TEST[1])], market), beta(f[(y >= 2014) & (y <= 2021)], market)
        s["SR_net ex market"] = sharpe_ex_market(f[(y >= TEST[0]) & (y <= TEST[1])], market)
        for a, b in BLOCKS:
            s[f"SR_net {a}-{str(b)[2:]}"] = stats(f[(y >= a) & (y <= b)])["SR_net"]
        late = f[(y >= HOLDOUT[0]) & (y <= HOLDOUT[1])]
        if len(late):
            s["SR_net 2022-25"], s["turnover 2022-25"] = stats(late)["SR_net"], late["turnover"].mean()
        rows.append(s)
    return pd.DataFrame(rows).mean().to_dict()


def column_sharpe(x: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray | float:
    """Average over columns of each column's annualized Sharpe ratio [T, columns]; on each draw if `idx` [draws, T]."""
    x = x[:, None] if x.ndim == 1 else x
    if idx is None:
        return float(np.mean(sharpe(x.T)))
    return np.mean([sharpe(x[:, k][idx]) for k in range(x.shape[1])], axis=0)


def reality_check(series: dict[Candidate, np.ndarray], base: np.ndarray) -> dict:
    """White's reality check (Sullivan, Timmermann and White 1999) on the net Sharpe ratio: is the best candidate
    better than the baseline by more than the best of the grid would be by chance? The criterion is the selection
    rules' (the seed average for the model); circular 21-day blocks, the same draws as paired_test.py, the same days
    for every candidate and seed. Returns the best candidate, its difference and the p-value."""
    idx = block_draws(len(base))
    base_star, base_sr = column_sharpe(base, idx), column_sharpe(base)
    delta = {c: column_sharpe(x) - base_sr for c, x in series.items()}
    top = max(delta, key=delta.get)
    worst_case = np.full(idx.shape[0], -np.inf)
    for c, x in series.items():
        worst_case = np.maximum(worst_case, column_sharpe(x, idx) - base_star - delta[c])   # recentred: no candidate better
    return {"best": top, "delta": delta[top], "p": float((worst_case >= delta[top]).mean()), "candidates": len(series)}


# ---------------------------------------------------------------------------------------------- one K (or several)
class Scan:
    """Daily paths of every setting on disk for some K, and the rules applied to them. By default the grid of
    scan.py on `broad` with seeds 0-2; `settings`, `extra` (other arrays) and `seeds` give other scans. `per_k` gives
    each K its own setting and only that one (a K scan whose K were each tuned on their own validation years): each
    K's differences are then measured against its own setting's model, and no setting is shared by every K."""

    def __init__(self, Ks: list[int], settings: tuple[Setting, ...] = GRID, extra: dict | None = None, seeds: tuple[int, ...] = SEEDS,
                 per_k: dict[int, Setting] | None = None):
        self.Ks, self.settings, self.extra, self.seeds, self.per_k = Ks, tuple(settings), extra or {}, tuple(seeds), dict(per_k or {})
        cfg = BASELINE.config(Ks[0], 8, self.extra)
        self.universe, self.market = data_folder(cfg.universe, cfg.fill), _market_row(cfg.universe, cfg.fill)
        self.test: dict[tuple[int, Setting], pd.DataFrame] = {}
        self.val: dict[tuple[int, Setting], pd.DataFrame] = {}
        for K in Ks:
            for s in self.settings:
                if self.per_k and s != self.per_k[K]:
                    continue
                p, v = paths(s, K, extra=self.extra, seeds=self.seeds), paths(s, K, validation=True, extra=self.extra, seeds=self.seeds)
                if p is not None and v is not None:
                    self.test[(K, s)], self.val[(K, s)] = p, v
        if not self.test:
            raise SystemExit("no complete setting with weights on disk for these K")
        self.last = min(p.index[-1].year for p in self.test.values())          # common last year
        for key, p in self.test.items():
            self.test[key] = p[p.index.year <= self.last]
        self.dates = next(iter(self.test.values())).index
        self.years = self.dates.year.to_numpy()

    def realize(self, plan: dict[int, Candidate], obj: str) -> list[pd.DataFrame]:
        return realize(plan, obj, TEST[0], self.last, self.extra, self.seeds)

    def candidates(self, obj: str, Ks: list[int] | None = None) -> list[Candidate]:
        deltas = DELTAS if obj == "ensemble + band" else (0.0,)
        return [Candidate(K, s, d) for (K, s) in self.test if Ks is None or K in Ks for d in deltas]

    def nets(self, obj: str, cands: list[Candidate], validation: bool = False) -> dict[Candidate, np.ndarray]:
        src = self.val if validation else self.test
        return {c: object_net(src[(c.K, c.setting)], obj, c.delta) for c in cands}

    def plans(self, obj: str, Ks: list[int] | None = None) -> dict[str, dict[int, Candidate]]:
        cands = self.candidates(obj, Ks)
        test, val = self.nets(obj, cands), self.nets(obj, cands, validation=True)
        in_test = (self.years >= TEST[0]) & (self.years <= TEST[1])
        v = best(val)
        h = best({c: x[in_test] for c, x in test.items()})
        years = range(TEST[0], self.last + 1)
        return {"validation": {y: v for y in years}, "hindsight": {y: h for y in years},
                "walk-forward": walk_forward_plan(test, self.years, v)}

    def common_plans(self, obj: str) -> dict[str, dict[int, tuple[Setting, float]]]:
        """The rules with one setting (and band width) for every K at once, as the paper uses one Table 4: the
        criterion is the average over K (and seeds) of each K's net Sharpe ratio."""
        deltas = DELTAS if obj == "ensemble + band" else (0.0,)
        keys = [(s, d) for s in self.settings if all((K, s) in self.test for K in self.Ks) for d in deltas]

        def stack(src: dict, s: Setting, d: float) -> np.ndarray:                # every K's columns side by side
            return np.concatenate([object_net(src[(K, s)], obj, d) for K in self.Ks], axis=1)

        test, val = {k: stack(self.test, *k) for k in keys}, {k: stack(self.val, *k) for k in keys}
        in_test = (self.years >= TEST[0]) & (self.years <= TEST[1])
        v, h = best(val), best({k: x[in_test] for k, x in test.items()})
        years = range(TEST[0], self.last + 1)
        return {"validation": {y: v for y in years}, "hindsight": {y: h for y in years}, "walk-forward": walk_forward_plan(test, self.years, v)}

    def baseline_setting(self, K: int) -> Setting:
        """The setting K's differences are measured against: BASELINE, or K's own setting when each K has one."""
        return self.per_k.get(K, BASELINE)

    def baseline(self, K: int) -> list[pd.DataFrame]:
        """The baseline model's seeds: from the scan if it is one of its settings, else as stored (seeds 0-2)."""
        setting = self.baseline_setting(K)
        if (K, setting) in self.test:
            return seed_frames(self.test[(K, setting)])
        frames = stitched(setting, K, self.extra)
        if frames is None:
            raise SystemExit(f"the baseline of K = {K} is not complete")
        return [f[(f.index.year >= TEST[0]) & (f.index.year <= self.last)] for f in frames]


def pooled(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = sum(f[list(FIELDS)] for f in frames) / len(frames)
    out["net"] = out["gross"] - COST_TRADE * out["turnover"] - COST_SHORT * out["short"]
    return out


def versus(frames: list[pd.DataFrame], base: list[pd.DataFrame]) -> dict[str, str]:
    """Paired difference of net Sharpe ratios against the baseline model (pooled series), 1998-2021 and 2022-2025."""
    a, b = pooled(frames), pooled(base)
    y = a.index.year
    test = (y >= TEST[0]) & (y <= TEST[1])
    out = {"Δ vs baseline, 1998–2021": cell(compare(a[test], b[test])["all"])}
    if (y > TEST[1]).any():
        out["Δ vs baseline, 2022–25"] = cell(compare(a[~test], b[~test])["all"])
    return out


def summary_of(plan: dict[int, Candidate], with_k: bool = False) -> str:
    chosen = list(plan.values())
    if len(set(chosen)) == 1:
        return chosen[0].code(with_k)
    held = pd.Series([c.code(with_k) for c in chosen]).value_counts()
    switches = sum(a != b for a, b in zip(chosen, chosen[1:]))
    return f"{len(held)} settings, {switches} switches; most held: {held.index[0]} ({held.iloc[0]} years)"


# ---------------------------------------------------------------------------------------------- tables
def md(t: pd.DataFrame, index: str = "") -> str:
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v))
    cols = list(t.columns)
    lines = ["| " + " | ".join([index] + cols) + " |", "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for i, r in t.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt(r[c]) for c in cols]) + " |")
    return "\n".join(lines)


SHOW = {"SR": "SR", "mu": "μ", "sigma": "σ", "turnover": "Turnover", "cost/sigma": "Cost / σ", "SR_net": "SR net",
        "SR_net 1998-05": "1998–2005", "SR_net 2006-13": "2006–13", "SR_net 2014-21": "2014–21", "beta": "Beta", "beta 2014-21": "Beta 2014–21",
        "SR_net ex market": "SR net, market removed",
        "SR_net 2022-25": "SR net 2022–25", "turnover 2022-25": "Turnover 2022–25"}


def setting_table(scan: Scan, K: int, plans: dict) -> str:
    """Every setting of one K: the model on the test years, validation and 2022-2025, and the two layers."""
    base, rows = scan.baseline(K), {}
    marks = {}
    for obj in OBJECTS:
        for rule in ("validation", "hindsight"):
            c = plans[obj][rule][TEST[0]]
            marks.setdefault(c.setting, []).append(f"{rule[0].upper()}{'' if obj == 'model' else ('e' if obj == 'ensemble' else 'b')}")
    chosen = plans["model"]["validation"][TEST[0]].setting                  # how clearly the validation years choose
    chosen_val = pooled(seed_frames(scan.val[(K, chosen)]))
    for s in scan.settings:
        if (K, s) not in scan.test:
            continue
        p, v = scan.test[(K, s)], scan.val[(K, s)]
        seeds = seed_frames(p)
        d = describe(seeds, scan.market)
        r = {SHOW[k]: d[k] for k in ("SR", "sigma", "turnover", "cost/sigma", "SR_net", "SR_net 1998-05", "SR_net 2006-13", "SR_net 2014-21",
                                     "beta 2014-21", "SR_net ex market") if k in d}
        r["Validation SR net"] = np.mean([stats(f)["SR_net"] for f in seed_frames(v)])
        r["Validation: Δ vs V, pooled"] = "" if s == chosen else cell(compare(pooled(seed_frames(v)), chosen_val)["all"])
        in_test = (p.index.year >= TEST[0]) & (p.index.year <= TEST[1])
        r["Ensemble SR net"] = stats(frame(p[in_test], "band 0"))["SR_net"]
        bands = {dl: stats(frame(p[in_test], f"band {dl:g}"))["SR_net"] for dl in DELTAS}
        vb = {dl: mean_sharpe(net(v, f"band {dl:g}")[:, None]) for dl in DELTAS}
        dv = max(vb, key=vb.get)
        r["Ensemble + band (δ from validation)"] = f"{bands[dv]:.2f} ({dv:g})"
        if "SR_net 2022-25" in d:
            r["SR net 2022–25 (diagnostic)"] = d["SR_net 2022-25"]
        r["Δ vs baseline, pooled"] = "" if s == scan.baseline_setting(K) else versus(seeds, base)["Δ vs baseline, 1998–2021"]
        r["Chosen by"] = " ".join(marks.get(s, []))
        name = s.label + (" (baseline)" if s == scan.baseline_setting(K) else " (both candidates)" if s == BOTH else "")
        rows[name] = r
    return md(pd.DataFrame(rows).T, "Setting")


def rule_rows(scan: Scan, K: int | None, plans: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows of each (object, rule), and the walk-forward choice of each year."""
    Ks = scan.Ks if K is None else [K]
    rows, choices = {}, {}
    if K is not None:
        base = scan.baseline(K)
        if K in PAPER_TABLE2:
            sr, mu, sigma, sr_net, mu_net, _, _ = PAPER_TABLE2[K]
            rows[("Paper", "")] = {"Chosen": "", "SR": sr, "μ": mu, "σ": sigma, "Turnover": ((mu - mu_net) / 252 * 100 - 0.5) / 5,
                                   "Cost / σ": (mu - mu_net) / sigma, "SR net": sr_net}
        d = describe(base, scan.market)
        rows[("Baseline model", "")] = {"Chosen": f"{scan.baseline_setting(K).label}, seeds {len(base)}", **{SHOW[k]: d[k] for k in SHOW if k in d}}
    else:
        base = None
    for obj in OBJECTS:
        for rule in RULES:
            plan = {y: c for y, c in plans[obj][rule].items() if y <= scan.last}
            frames = scan.realize(plan, obj)
            d = describe(frames, scan.market)
            r = {"Chosen": summary_of(plan, with_k=K is None), **{SHOW[k]: d[k] for k in SHOW if k in d}}
            if base is not None:
                r.update(versus(frames, base))
            rows[(obj, rule)] = r
            if rule == "walk-forward":
                choices[obj] = {y: c.code(with_k=K is None) for y, c in plan.items()}
    t = pd.DataFrame(rows).T
    t.index = [f"{a}, {b}" if b else a for a, b in t.index]
    return t, pd.DataFrame(choices)


def write(scan: Scan, out: Path = OUT, title: str = "Hyperparameter scan: the best net Sharpe ratio under each selection rule") -> None:
    seeds = f"seeds {scan.seeds[0]}–{scan.seeds[-1]}" if scan.seeds == tuple(range(scan.seeds[0], scan.seeds[-1] + 1)) else f"seeds {list(scan.seeds)}"
    parts = [f"# {title}",
             "Generated by `analysis/gap/selection.py` from the runs of `analysis/gap/scan.py`; read with `analysis/gap/REPORT.md`, section 10. "
             f"Arrays `{scan.universe}`, {seeds}, every result scored at the paper's costs (5 bp per unit of turnover, 1 bp per unit short), "
             "whatever cost the model was trained with. Sharpe ratios are annualized, on stock-by-stock excess returns. "
             "A setting is *training cost, ridge, epochs* (and a band width for the last object).",
             "- **Objects.** *model*: each seed's portfolio, statistics averaged over seeds (the paper's object). *ensemble*: the seeds' weights averaged "
             "and rescaled. *ensemble + band*: the ensemble through a no-trade band. The last two are deployment layers, never Table 2 rows.",
             "- **Rules.** *validation*: best on 1996–1997, the paper's rule and CLAUDE.md §8. *walk-forward*: each year the best over the test years "
             "before it (1998 takes the validation choice); changes of setting are traded and paid for. *hindsight*: the best fixed setting over "
             "1998–2021, a diagnostic upper bound. 2022–2025 is out of sample for all three.",
             f"- **Differences** are paired block bootstraps (21-day blocks, 2,000 draws) of pooled series against the baseline model pooled "
             + (f"({BASELINE.label}). Settings: {len(scan.settings)}; with the band, {len(scan.settings) * len(DELTAS)} candidates per K."
                if not scan.per_k else
                "of each K's own setting (its baseline). Each K has one setting, chosen beforehand on its own validation years: "
                + "; ".join(f"K = {K}: {scan.per_k[K].label}" for K in scan.Ks)
                + f". With the band, {len(DELTAS)} candidates per K, and {len(scan.Ks) * len(DELTAS)} when K is chosen as well.")]
    sections, tables = [], {}
    in_test = (scan.years >= TEST[0]) & (scan.years <= TEST[1])
    for K in scan.Ks:
        plans = {obj: scan.plans(obj, [K]) for obj in OBJECTS}
        t, choices = rule_rows(scan, K, plans)
        tables[K] = t
        series = {c: object_net(scan.test[(K, c.setting)], "model")[in_test] for c in scan.candidates("model", [K])}
        rc = reality_check(series, frames_net(scan.baseline(K), scan.dates)[in_test])
        p = "p < 0.001" if rc["p"] < 0.001 else f"p = {rc['p']:.3f}"
        sections += [f"## K = {K}", "**Selection rules** (1998–2021 unless the column says otherwise)", md(t),
                     f"**Reality check, model:** over 1998–2021 the best of {rc['candidates']} settings ({rc['best'].code()}) beats the baseline "
                     f"by {rc['delta']:+.2f} (seed average, the hindsight criterion); {p} that the best of the grid would do as well by chance.",
                     "**Every setting** (model columns are seed averages; *Validation: Δ vs V*: paired difference on 1996–1997 from the model "
                     "setting the validation rule chose; *Chosen by*: V = validation, H = hindsight; suffix e = ensemble, b = ensemble + band)",
                     setting_table(scan, K, plans),
                     "**Walk-forward choice of each year**", md(choices, "Year")]
        if len(scan.seeds) > len(SEEDS):
            sections += seed_count_section(scan, K)
    if len(scan.Ks) > 1:
        plans = {obj: scan.plans(obj) for obj in OBJECTS}
        t, choices = rule_rows(scan, None, plans)
        sections += [f"## Choosing K as well (K = {', '.join(map(str, scan.Ks))})", md(t), "**Walk-forward choice of each year**", md(choices, "Year")]
    parts += summary(tables) + (common_section(scan) if len(scan.Ks) > 1 and not scan.per_k else [])   # per_k: no setting shared by every K
    out.write_text("\n\n".join(parts + sections) + "\n", encoding="utf-8")


SEED_GROUPS = (1, 2, 3, 5, 10)


def seed_groups(seeds: tuple[int, ...], n: int) -> list[tuple[int, ...]]:
    """Disjoint groups of n consecutive seeds; a remainder is left out (of ten seeds, n = 3 gives 0-2, 3-5 and 6-8)."""
    return [tuple(seeds[i * n:(i + 1) * n]) for i in range(len(seeds) // n)]


def seed_count_rows(scan: Scan, K: int, setting: Setting) -> dict[str, dict[int, np.ndarray]]:
    """The ensemble of n seeds, alone and through a band whose width is chosen on the group's own validation years, for
    each n of SEED_GROUPS: one row per disjoint group of seeds, columns (net Sharpe ratio, turnover) over 1998-2021 and
    the band width. n = 1 is the model itself, since one seed's weights already have unit gross exposure."""
    cfg = setting.config(K, 8, scan.extra)
    first, last, train_years = PERIODS["validation"]
    dates, permnos, R, member = _market(TEST[0], scan.last, cfg.universe, cfg.fill)
    v_dates, v_permnos, v_R, v_member = _market(first, last, cfg.universe, cfg.fill)
    W = {s: seed_weights(setting.run_dir(K, 8, scan.extra) / f"seed{s}", dates, permnos) for s in scan.seeds}
    V = {s: seed_weights(setting.run_dir(K, train_years, scan.extra) / f"seed{s}", v_dates, v_permnos) for s in scan.seeds}
    in_test = pd.DatetimeIndex(dates).year <= TEST[1]

    def scored(H: np.ndarray) -> tuple[float, float]:
        st = stats(trade(H, R, dates)[in_test])
        return st["SR_net"], st["turnover"]

    out: dict[str, dict[int, np.ndarray]] = {"ensemble": {}, "ensemble + band": {}}
    for n in (n for n in SEED_GROUPS if n <= len(scan.seeds)):
        rows: dict[str, list] = {"ensemble": [], "ensemble + band": []}
        for g in seed_groups(scan.seeds, n):
            Ev = ensemble([V[s] for s in g])
            dv = max(DELTAS, key=lambda d: mean_sharpe(trade(Ev if d == 0 else band_path(Ev, v_member, d), v_R, v_dates)["net"].to_numpy()[:, None]))
            E = ensemble([W[s] for s in g])
            rows["ensemble"].append((*scored(E), 0.0))
            rows["ensemble + band"].append((*scored(E if dv == 0 else band_path(E, member, dv)), dv))
        for obj, r in rows.items():
            out[obj][n] = np.array(r)
    return out


def seed_count_section(scan: Scan, K: int) -> list[str]:
    """Net Sharpe ratio and turnover of the ensemble against the number of seeds it averages, for every setting."""
    sr, turnover = {}, {}
    for s in scan.settings:
        if (K, s) not in scan.test:
            continue
        for obj, by_n in seed_count_rows(scan, K, s).items():
            name = f"{obj}: {s.label}"
            sr[name] = {f"n = {n}": f"{a[:, 0].mean():.2f}" + (f" ({a[:, 0].min():.2f} to {a[:, 0].max():.2f})" if len(a) > 1 else "")
                        for n, a in by_n.items()}
            turnover[name] = {f"n = {n}": a[:, 1].mean() for n, a in by_n.items()}
    return [f"**Seeds in the ensemble** (1998–2021). The ensemble of n seeds, alone and through a band whose width each group chooses on "
            f"its own validation years. Disjoint groups of consecutive seeds (n = 3: seeds 0–2, 3–5 and 6–8); each cell is the mean over the "
            f"groups, with the lowest and highest group in brackets. n = 1 is the model itself (the ensemble of one seed is that seed), or "
            f"the band on one seed. Net Sharpe ratio:", md(pd.DataFrame(sr).T, "Object: setting"),
            "**Turnover a day** of the same objects (means over the groups):", md(pd.DataFrame(turnover).T, "Object: setting")]


def common_section(scan: Scan) -> list[str]:
    """One setting for every K: what each rule chooses, and its net Sharpe ratio at each K."""
    test, late, chosen, diff, diff_late = {}, {}, {}, {}, {}
    base = {K: (scan.baseline(K), describe(scan.baseline(K), scan.market)) for K in scan.Ks}
    for obj in ("model", "ensemble + band"):
        plans = scan.common_plans(obj)
        for rule in RULES:
            name, plan = f"{obj}, {rule}", plans[rule]
            chosen[name] = summary_of({y: Candidate(0, s, d) for y, (s, d) in plan.items()})
            for K in scan.Ks:
                frames = scan.realize({y: Candidate(K, s, dl) for y, (s, dl) in plan.items()}, obj)
                d, vs = describe(frames, scan.market), versus(frames, base[K][0])
                test.setdefault(K, {"Baseline model": base[K][1]["SR_net"]})[name] = d["SR_net"]
                diff.setdefault(K, {})[name] = vs["Δ vs baseline, 1998–2021"]
                if "SR_net 2022-25" in d:
                    late.setdefault(K, {"Baseline model": base[K][1]["SR_net 2022-25"]})[name] = d["SR_net 2022-25"]
                    diff_late.setdefault(K, {})[name] = vs["Δ vs baseline, 2022–25"]
    out = [f"## One setting for every K (K = {', '.join(map(str, scan.Ks))})",
           "As the paper uses one Table 4 for all K: the criterion is each K's net Sharpe ratio averaged over the K (and seeds). Chosen:",
           "\n".join(f"- *{name}*: {c}" for name, c in chosen.items()),
           "**Net Sharpe ratio, 1998–2021.** Hindsight columns are diagnostics.", md(pd.DataFrame(test).T, "K"),
           "**Difference from the baseline model, 1998–2021** (pooled series, paired block bootstrap, 95% interval).", md(pd.DataFrame(diff).T, "K")]
    if late:
        out += ["**Net Sharpe ratio, 2022–2025** (out of sample for every rule).", md(pd.DataFrame(late).T, "K"),
                "**Difference from the baseline model, 2022–2025.**", md(pd.DataFrame(diff_late).T, "K")]
    return out


def summary(tables: dict[int, pd.DataFrame]) -> list[str]:
    """Net Sharpe ratio of each rule, one row per K, for 1998-2021 and 2022-2025."""
    keep = ["Baseline model", "model, validation", "model, walk-forward", "model, hindsight", "ensemble + band, validation",
            "ensemble + band, walk-forward", "ensemble + band, hindsight"]
    out = ["## Summary"]
    for col, title in (("SR net", "Net Sharpe ratio, 1998–2021"), ("SR net 2022–25", "Net Sharpe ratio, 2022–2025 (out of sample for every rule)")):
        rows = {}
        for K, t in tables.items():
            if col not in t.columns:
                continue
            rows[K] = {"Paper": PAPER_TABLE2[K][3] if col == "SR net" and K in PAPER_TABLE2 else np.nan,
                       **{name: t.loc[name, col] for name in keep if name in t.index}}
        if rows:
            out += [f"**{title}.** Hindsight rows are diagnostics on 1998–2021.", md(pd.DataFrame(rows).T, "K")]
    return out


ARRAYS = {"broad": {}, "common": {"universe": "common"},
          "wrds_broad": {"universe": "wrds_broad", "fill": "none"}, "wrds_common": {"universe": "wrds_common", "fill": "none"}}
LAYER = (Setting(5e-4, 1e-4, 30), 8e-4)          # the layer the validation rule picks with one setting for all K (REPORT.md 10.2)
ARRAYS_OUT = Path(__file__).resolve().parent / "SCAN_ARRAYS.md"


def stitched(setting: Setting, K: int, extra: dict, validation: bool = False) -> list[pd.DataFrame] | None:
    """Each seed's daily results of a run as stored (no weights needed), or None if its period is incomplete."""
    period = "validation" if validation else "test"
    if not complete(setting, K, period, extra):
        return None
    return [stitch(setting.run_dir(K, PERIODS[period][2], extra) / f"seed{s}") for s in SEEDS]


def arrays_section(K: int, names: list[str]) -> list[str]:
    """The setting chosen in REPORT.md section 10 on other arrays: each array's baseline, the chosen setting and the
    layer. Beta is against each array's own equal-weighted market; differences are paired against its baseline."""
    rows = {}
    if K in PAPER_TABLE2:
        sr, mu, sigma, sr_net, mu_net, _, _ = PAPER_TABLE2[K]
        rows["Paper"] = {"Run": "Table 2", "SR": sr, "μ": mu, "σ": sigma, "Turnover": ((mu - mu_net) / 252 * 100 - 0.5) / 5,
                         "Cost / σ": (mu - mu_net) / sigma, "SR net": sr_net}
    for name in names:
        extra = ARRAYS[name]
        cfg = BASELINE.config(K, 8, extra)
        mkt, base = _market_row(cfg.universe, cfg.fill), stitched(BASELINE, K, extra)
        if base is None:
            print(f"{name}: no complete baseline at K = {K}", flush=True)
            continue
        for label, s in (("baseline", BASELINE), ("chosen setting", BOTH)):
            frames = base if s == BASELINE else stitched(s, K, extra)
            if frames is None:
                print(f"{name}: {s.label} is not complete", flush=True)
                continue
            d = describe(frames, mkt)
            r = {"Run": f"model: {s.label}", **{SHOW[k]: d[k] for k in SHOW if k in d}}
            twin = stitched(s, K, extra, validation=True)
            if twin is not None:
                r["Validation SR net"] = float(np.mean([stats(f)["SR_net"] for f in twin]))
            if s != BASELINE:
                r.update(versus(frames, base))
            rows[f"{name}, {label}"] = r
        p = paths(LAYER[0], K, extra=extra)
        if p is not None:
            f = frame(p, f"band {LAYER[1]:g}")
            d = describe([f], mkt)
            rows[f"{name}, layer"] = {"Run": f"ensemble + band {LAYER[1]:g}: {LAYER[0].label}", **{SHOW[k]: d[k] for k in SHOW if k in d},
                                      **versus([f], base)}
    return [f"## K = {K}",
            "Model rows are seed averages (the paper's object); the layer is a deployment layer, not a Table 2 row. Beta is against each "
            "array's own equal-weighted market. Differences are paired block bootstraps against the same array's baseline model.",
            md(pd.DataFrame(rows).T, "Arrays")]


def write_arrays(Ks: list[int], names: list[str]) -> None:
    parts = ["# The chosen setting on other arrays",
             "Generated by `analysis/gap/selection.py --arrays`; read with `analysis/gap/REPORT.md`, section 10. The setting chosen in section 10 "
             f"({BOTH.label}) and its layer (the seed ensemble of the {LAYER[0].label} model through a band of {LAYER[1]:g}), on the arrays "
             "named below, against each array's baseline. Seeds 0–2, scored at the paper's costs; test years 1998–2021, holdout 2022–2025, "
             "validation 1996–1997.",
             "- `broad`, `common`: Ander's X (`afsa.build_data`), missing characteristics carried forward.",
             "- `wrds_broad`, `wrds_common`: built only from our WRDS downloads (`afsa.build_wrds`), missing characteristics at rank 0 (`--fill none`).",
             "- `broad` and `wrds_broad` are the top 500 of all NYSE/AMEX/NASDAQ equities; `common` and `wrds_common` of US common stocks only."]
    for K in Ks:
        parts += arrays_section(K, names)
    ARRAYS_OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"written: {ARRAYS_OUT}")


def main() -> None:
    global BASELINE                                                          # --baseline replaces it for every function here
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, nargs="+", default=[8, 30])
    p.add_argument("--arrays", nargs="+", choices=list(ARRAYS), default=None,
                   help="instead: the chosen setting on these arrays -> analysis/gap/SCAN_ARRAYS.md")
    p.add_argument("--only", nargs="+", default=None, metavar="BP/RIDGE/EPOCHS", help="instead of the grid, these settings (as in scan.py)")
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE", help="other arrays, e.g. universe=wrds_broad fill=none")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="seeds [default: 0 1 2]")
    p.add_argument("--baseline", default=None, metavar="BP/RIDGE/EPOCHS",
                   help=f"the setting every difference is measured against [default: {BASELINE.label}]")
    p.add_argument("--per-k", nargs="+", default=None, metavar="K=BP/RIDGE/EPOCHS",
                   help="instead of --K and --only: each K with its own setting, e.g. 8=5/1e-4/80 30=5/3e-4/30 (pilot.py --best prints them)")
    p.add_argument("--out", type=Path, default=OUT, help="markdown file [default: analysis/gap/SCAN.md]")
    args = p.parse_args()
    if args.baseline:                                                        # e.g. a rule whose ridge-1e-5 runs do not exist
        BASELINE = parse_setting(args.baseline)
    if args.arrays:
        write_arrays(args.K, args.arrays)
        return
    if args.per_k:
        per_k = {int(k): parse_setting(v) for k, v in (x.split("=", 1) for x in args.per_k)}
        Ks, settings = sorted(per_k), tuple(dict.fromkeys(per_k[K] for K in sorted(per_k)))
        scan = Scan(Ks, settings, parse_set(args.set), tuple(args.seeds), per_k)
        missing = [K for K in Ks if (K, per_k[K]) not in scan.test]
        if missing:
            raise SystemExit(f"no complete run with weights for K = {missing} at its setting")
        print(f"{len(scan.test)} runs with weights, test years {TEST[0]}-{scan.last}", flush=True)
        title = (f"K scan on `{scan.universe}`" + (f" ({', '.join(args.set)})" if args.set else "")
                 + f" with {len(scan.seeds)} seeds, each K at its own setting: each selection rule")
        write(scan, args.out, title)
        print(f"written: {args.out}")
        return
    settings = tuple(parse_setting(t) for t in args.only) if args.only else GRID
    scan = Scan(args.K, settings, parse_set(args.set), tuple(args.seeds))
    print(f"{len(scan.test)} runs with weights, test years {TEST[0]}-{scan.last}", flush=True)
    title = "Hyperparameter scan: the best net Sharpe ratio under each selection rule"
    if args.only or args.set or tuple(args.seeds) != SEEDS:
        title = (f"Scan of {len(settings)} settings on `{scan.universe}`" + (f" ({', '.join(args.set)})" if args.set else "")
                 + f" with {len(scan.seeds)} seeds: each selection rule")
    write(scan, args.out, title)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
