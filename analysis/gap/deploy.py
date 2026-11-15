"""Deployment layers on a trained run: the average of the seeds' portfolios, and a no-trade band.

    python analysis/gap/reeval_weights.py results/runs/attn_K030     # once: the daily weights of the saved models
    python analysis/gap/deploy.py results/runs/attn_K030             # -> deploy.csv and deploy_delta.csv in that folder

Neither layer changes a model. They change how its daily target weights are traded, so their rows never replace
the model's row (the paper's Table 2 has no such layer).

Objects
- per seed:  each seed's portfolio is scored on its own and the statistics are averaged (the rows of RESULTS.md).
- pooled:    the three seeds' daily returns are averaged first: one third of the capital in each seed's portfolio,
             each paying its own trading costs.
- ensemble:  the three seeds' weights are averaged stock by stock and rescaled to unit gross exposure, and that one
             portfolio is traded. Trades of the seeds that offset each other are not paid for.

No-trade band of width delta: a position moves only by the part of the gap to its target that exceeds delta, then
positions outside the day's universe are closed and the book is rescaled to unit gross exposure. delta = 0 is the
model's own portfolio. Three ways to set delta, each a labelled row:
- fixed:        one delta for all years, for every value of the grid. A diagnostic: its best value is read off the
                test years.
- walk-forward: at the start of test year Y, the delta with the highest net Sharpe ratio over the test years before
                Y; the first year uses 0. It uses nothing from Y or later, but it does choose from test-year
                results, which CLAUDE.md section 8 reserves for the validation years: a deployment diagnostic.
- validation:   the delta with the highest net Sharpe ratio on 1996-1997 (the run's `train_years-6` twin), kept
                for all test years. This is the choice our own rule allows.

Costs are the paper's (5 bp per unit of turnover, 1 bp per unit of short exposure). Turnover is the L1 distance
between consecutive books, the first book is bought from cash, and returns are stock-by-stock excess returns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import BLOCKS, stats  # noqa: E402

from afsa.config import TRAIN_DATA_DIR, WINDOW_RULES, Config, data_folder  # noqa: E402
from afsa.metrics import COST_SHORT, COST_TRADE, stitch  # noqa: E402

DELTAS = (0.0, 1e-4, 2e-4, 4e-4, 8e-4, 1.2e-3, 1.6e-3, 2.4e-3, 3.2e-3)
PAPER = (1998, 2021)                                         # the paper's out-of-sample years


# ---------------------------------------------------------------------------------------------- the layers
def band_step(h_prev: np.ndarray, w_t: np.ndarray, member_t: np.ndarray, delta: float, rescale: str = "all",
              eta: float = 1.0) -> np.ndarray:
    """One day of the no-trade band: from yesterday's book to today's, given today's target.

    rescale "all" (the stored results) divides the whole book by its gross exposure; "moved" resizes only the stocks
    that trade or are closed and leaves the others exactly where they were, as the model's trading layer does
    (`afsa.model.trade_book`, which this reproduces). eta < 1 trades only that share of the gap beyond the band."""
    dw = w_t - h_prev
    trade = eta * np.sign(dw) * np.maximum(np.abs(dw) - delta, 0.0)          # soft threshold = prox of an L1 trading cost
    h = np.where(member_t, h_prev + trade, 0.0)                              # never hold a stock outside the universe
    if rescale == "moved":
        moved = (trade != 0) | (~member_t & (h_prev != 0))
        held, movers = np.abs(h[~moved]).sum(), np.abs(h[moved]).sum()
        if movers > 0 and (1 - held) / movers > 0:
            return np.where(moved, (1 - held) / movers * h, h)
    elif rescale != "all":
        raise ValueError("rescale must be all or moved")
    s = np.abs(h).sum()
    return h / s if s > 0 else w_t                                           # unit gross exposure


def band_path(W: np.ndarray, member: np.ndarray, delta: float, h0: np.ndarray | None = None, rescale: str = "all",
              eta: float = 1.0) -> np.ndarray:
    """Books [T, S] of a band of width delta around the targets W, starting from h0 (cash if None)."""
    H, h = np.empty_like(W), (np.zeros(W.shape[1]) if h0 is None else h0)
    for t in range(len(W)):
        h = H[t] = band_step(h, W[t], member[t], delta, rescale, eta)
    return H


def ensemble(Ws: list[np.ndarray]) -> np.ndarray:
    """Average of the seeds' weights, stock by stock, rescaled to unit gross exposure each day."""
    mean = sum(Ws) / len(Ws)
    return mean / np.maximum(np.abs(mean).sum(1, keepdims=True), 1e-12)


def trade(H: np.ndarray, R: np.ndarray, dates: np.ndarray, h0: np.ndarray | None = None) -> pd.DataFrame:
    """Daily gross return, turnover, short exposure, cost and net return of the books H, bought from h0 (cash if None)."""
    prev = np.vstack([np.zeros((1, H.shape[1])) if h0 is None else h0[None], H[:-1]])
    daily = pd.DataFrame({"gross": (H * R).sum(1), "turnover": np.abs(H - prev).sum(1), "short": np.maximum(-H, 0.0).sum(1)},
                         index=pd.DatetimeIndex(dates, name="date"))
    daily["cost"] = COST_TRADE * daily["turnover"] + COST_SHORT * daily["short"]
    daily["net"] = daily["gross"] - daily["cost"]
    return daily


def net_sharpe(daily: pd.DataFrame) -> float:
    return float(daily["net"].mean() / daily["net"].std()) if len(daily) > 1 and daily["net"].std() > 0 else -np.inf


def band_grid(W: np.ndarray, R: np.ndarray, member: np.ndarray, dates: np.ndarray, deltas=DELTAS) -> dict[float, pd.DataFrame]:
    """Daily results of one full path per delta, each with its own book from the first day on."""
    return {d: trade(band_path(W, member, d), R, dates) for d in deltas}


def walk_forward(W: np.ndarray, R: np.ndarray, member: np.ndarray, dates: np.ndarray, grid: dict[float, pd.DataFrame]) -> tuple[pd.DataFrame, dict[int, float]]:
    """The band whose delta is chosen again each year from the years before. Returns its daily results and the
    delta of each year. The book is carried over, so a change of delta is traded and paid for."""
    years = pd.DatetimeIndex(dates).year.to_numpy()
    H, h, chosen = np.empty_like(W), None, {}
    for y in np.unique(years):
        now, before = years == y, years < y
        chosen[int(y)] = max(grid, key=lambda d: net_sharpe(grid[d][before])) if before.any() else min(grid)   # ties: the smaller delta
        H[now] = band_path(W[now], member[now], chosen[int(y)], h)
        h = H[now][-1]
    return trade(H, R, dates), chosen


# ---------------------------------------------------------------------------------------------- data of a run
def market(cfg: Config, first: int, last: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trading days of first..last, the stocks in the universe on any of them, and their excess returns and
    membership [T, S], from the arrays the run trained on."""
    if cfg.universe in WINDOW_RULES:
        raise SystemExit("deploy.py handles universes with one sample for all years, not the fixed-per-window ones")
    folder = TRAIN_DATA_DIR / data_folder(cfg.universe, cfg.fill)
    dates = np.load(folder / "dates.npy").astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    keep = (years >= first) & (years <= last)
    permno = np.load(folder / "permno.npy")[keep]
    ret, rf = np.load(folder / "ret.npy")[keep].astype(np.float64), np.load(folder / "rf.npy")[keep].astype(np.float64)
    filled = permno >= 0
    permnos = np.unique(permno[filled])
    day, col = np.nonzero(filled)[0], np.searchsorted(permnos, permno[filled])
    R, member = np.zeros((int(keep.sum()), len(permnos))), np.zeros((int(keep.sum()), len(permnos)), dtype=bool)
    R[day, col] = ret[filled] - rf[day]
    member[day, col] = True
    return dates[keep], permnos, R, member


def seed_weights(seed_dir: Path, dates: np.ndarray, permnos: np.ndarray) -> np.ndarray:
    """Target weights [T, S] of one seed on `dates`, from the <year>_w.npz files of reeval_weights.py."""
    W, seen = np.zeros((len(dates), len(permnos))), np.zeros(len(dates), dtype=bool)
    for year in np.unique(dates.astype("datetime64[Y]").astype(int) + 1970):
        f = seed_dir / f"{year}_w.npz"
        if not f.exists():
            raise SystemExit(f"{f} is missing: run analysis/gap/reeval_weights.py {seed_dir.parent} first")
        z = np.load(f)
        rows = np.searchsorted(dates, z["dates"].astype("datetime64[D]"))
        col = np.minimum(np.searchsorted(permnos, z["permnos"]), len(permnos) - 1)
        known = permnos[col] == z["permnos"]                                 # the others were members only in the lead days
        if np.abs(z["weights"][:, ~known]).max(initial=0.0) > 0:
            raise ValueError(f"{f}: weight on a stock that is in the universe on no test day")
        W[np.ix_(rows, col[known])] = z["weights"][:, known]
        seen[rows] = True
    if not seen.all():
        raise ValueError(f"{seed_dir}: {int((~seen).sum())} test days have no weights")
    return W


def load(run: Path, first: int | None = None, last: int | None = None):
    """Dates, excess returns, membership and each seed's target weights for the test years of a run on disk."""
    cfg = Config.load(run / "config.json")
    seed_dirs = sorted(run.glob("seed*"), key=lambda p: int(p.name[4:]))
    years = sorted(set.intersection(*[{int(f.stem) for f in d.glob("[0-9][0-9][0-9][0-9].npz")} for d in seed_dirs]))
    first, last = max(first or years[0], years[0]), min(last or years[-1], years[-1])
    dates, permnos, R, member = market(cfg, first, last)
    Ws = [seed_weights(d, dates, permnos) for d in seed_dirs]
    for d, W in zip(seed_dirs, Ws):                                          # the weights reproduce the stored results
        stored, mine = stitch(d).loc[pd.DatetimeIndex(dates)], trade(W, R, dates)
        gap = max(float(np.abs(mine["gross"] - stored["gross"]).max()), float(np.abs(mine["turnover"] - stored["turnover"])[1:].max()))
        if gap > 2e-5:
            raise ValueError(f"{d}: the saved weights do not reproduce the stored results (largest difference {gap:.1e})")
    return cfg, dates, R, member, Ws


# ---------------------------------------------------------------------------------------------- the table
def score(daily: pd.DataFrame) -> dict[str, float]:
    """Statistics of one daily series: the paper's years, their three eras, and the later years if the run has them."""
    y = daily.index.year
    out = stats(daily[(y >= PAPER[0]) & (y <= PAPER[1])]) if ((y >= PAPER[0]) & (y <= PAPER[1])).any() else stats(daily)
    for a, b in BLOCKS:
        if y.min() <= a and b <= y.max():
            out[f"SR_net {a}-{str(b)[2:]}"] = stats(daily[(y >= a) & (y <= b)])["SR_net"]
    if y.max() > PAPER[1]:
        late = stats(daily[y > PAPER[1]])
        out.update({f"SR_net {PAPER[1] + 1}-{str(y.max())[2:]}": late["SR_net"], f"turnover {PAPER[1] + 1}-{str(y.max())[2:]}": late["turnover"]})
    return out


def average(frames: list[pd.DataFrame]) -> dict[str, float]:
    return pd.DataFrame([score(f) for f in frames]).mean().to_dict()


def validation_delta(cfg: Config, run: Path) -> tuple[list[float], float] | None:
    """The delta with the highest net Sharpe ratio on the validation years, for each seed and for the ensemble, from
    the run's twin with 6-year windows; None if that twin or its weights are missing."""
    twin = run.parent / Config(**{**cfg.to_dict(), "train_years": 6}).name()
    if cfg.train_years != 8 or not (twin / "config.json").exists() or not list(twin.glob("seed*/[0-9][0-9][0-9][0-9]_w.npz")):
        return None
    _, dates, R, member, Ws = load(twin)

    def best(W: np.ndarray) -> float:
        grid = band_grid(W, R, member, dates)
        return max(grid, key=lambda d: net_sharpe(grid[d]))

    return [best(W) for W in Ws], best(ensemble(Ws))


def band_test(run: Path) -> pd.DataFrame:
    """Every width of the grid on one run (meant for a validation twin): net Sharpe ratio and turnover, for the seeds
    (seed average) and for the ensemble, and the paired difference from no band with its 95% interval."""
    from paired_test import cell, paired_bootstrap

    _, dates, R, member, Ws = load(run)
    grids = [band_grid(W, R, member, dates) for W in Ws]
    grid_e = band_grid(ensemble(Ws), R, member, dates)
    rows = {}
    for d in DELTAS:
        seeds = [g[d] for g in grids]
        rows[f"{d:g}"] = {"SR net, seeds": np.mean([net_sharpe(s) for s in seeds]) * np.sqrt(252), "Turnover, seeds": np.mean([s["turnover"].mean() for s in seeds]),
                          "Seeds: difference from no band": "" if d == 0 else cell(paired_bootstrap((sum(seeds) / len(seeds))["net"].to_numpy(),
                                                                                                    (sum(g[0.0] for g in grids) / len(grids))["net"].to_numpy())),
                          "SR net, ensemble": net_sharpe(grid_e[d]) * np.sqrt(252), "Turnover, ensemble": grid_e[d]["turnover"].mean(),
                          "Ensemble: difference from no band": "" if d == 0 else cell(paired_bootstrap(grid_e[d]["net"].to_numpy(), grid_e[0.0]["net"].to_numpy()))}
    t = pd.DataFrame(rows).T
    t.index.name = "Band width"
    return t


def seed_agreement(run: Path) -> dict[str, float]:
    """How alike the seeds' books are: mean correlation between two seeds of the day's weights, of the day's trades
    and of the daily gross returns, and the gross exposure of the average book before rescaling (1 = identical)."""
    _, dates, R, _, Ws = load(run)

    def corr(a: np.ndarray, b: np.ndarray) -> float:                         # weights and trades have mean ~0 across stocks
        return float(((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))).mean())

    pairs = [(i, j) for i in range(len(Ws)) for j in range(i + 1, len(Ws))]
    trades = [np.diff(W, axis=0) for W in Ws]
    gross = [(W * R).sum(1) for W in Ws]
    return {"weights": float(np.mean([corr(Ws[i], Ws[j]) for i, j in pairs])), "trades": float(np.mean([corr(trades[i], trades[j]) for i, j in pairs])),
            "returns": float(np.mean([np.corrcoef(gross[i], gross[j])[0, 1] for i, j in pairs])),
            "gross exposure of the average book": float(np.abs(sum(Ws) / len(Ws)).sum(1).mean())}


def table(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows of one run, and the delta the walk-forward band used in each test year (per seed and ensemble)."""
    cfg, dates, R, member, Ws = load(run)
    E = ensemble(Ws)
    seeds = [trade(W, R, dates) for W in Ws]
    rows = {("model", "per seed", ""): average(seeds),
            ("model", "pooled", ""): score(sum(seeds) / len(seeds)),
            ("ensemble", "ensemble", ""): score(trade(E, R, dates))}
    grids = [band_grid(W, R, member, dates) for W in Ws]
    grid_e = band_grid(E, R, member, dates)
    for d in DELTAS[1:]:
        rows[("band, fixed delta", "per seed", f"{d:g}")] = average([g[d] for g in grids])
        rows[("band, fixed delta", "ensemble", f"{d:g}")] = score(grid_e[d])
    forward = [walk_forward(W, R, member, dates, g) for W, g in zip(Ws, grids)]
    forward_e = walk_forward(E, R, member, dates, grid_e)
    choices = pd.DataFrame({**{f"seed{i}": c for i, (_, c) in enumerate(forward)}, "ensemble": forward_e[1]})
    choices.index.name = "test year"
    rows[("band, walk-forward delta", "per seed", f"median {choices.drop(columns='ensemble').stack().median():g}")] = average([f for f, _ in forward])
    rows[("band, walk-forward delta", "ensemble", f"median {choices['ensemble'].median():g}")] = score(forward_e[0])
    chosen = validation_delta(cfg, run)
    if chosen is not None:
        rows[("band, delta from validation", "per seed", " / ".join(f"{d:g}" for d in chosen[0]))] = average([g[d] for g, d in zip(grids, chosen[0])])
        rows[("band, delta from validation", "ensemble", f"{chosen[1]:g}")] = score(grid_e[chosen[1]])
    t = pd.DataFrame(rows).T
    t.index.names = ["layer", "object", "delta"]
    return t, choices


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run folders (results/runs/<name>) with weights from reeval_weights.py")
    p.add_argument("--band-test", action="store_true", help="instead: every band width on these runs, with paired intervals (for a validation twin)")
    args = p.parse_args()
    pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 500)
    for run in args.runs:
        if args.band_test:
            print(f"\n{run.name}: agreement of the seeds {({k: round(v, 3) for k, v in seed_agreement(run).items()})}")
            print(band_test(run).to_string(float_format=lambda x: f"{x:.2f}"), flush=True)
            continue
        t, choices = table(run)
        t.to_csv(run / "deploy.csv")
        choices.to_csv(run / "deploy_delta.csv")
        cols = [c for c in t.columns if c in ("SR", "mu", "sigma", "turnover", "cost/sigma", "SR_net") or c.startswith(("SR_net ", "turnover "))]
        print(f"\n{run.name}\n{t[cols].astype(float).round(2).to_string()}", flush=True)


if __name__ == "__main__":
    main()
