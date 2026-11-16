"""Trials for the extension years 2022-2025 (analysis/gap/EXTENSION.md): one change to a reference run at a time.

    python analysis/gap/extension.py --list                      # the trials, and what is on disk
    python analysis/gap/extension.py train win16 drop03          # GPU: trains what is missing, then replays the daily weights
    python analysis/gap/extension.py report                      # CPU: every trial on disk against its reference
    python analysis/gap/extension.py report win16 --by-year      # one trial, with net SR by year

Why: on dataset D (buffered list, rebuilt history) the model nets 1.31 on 1998-2021 and -0.34 on 2022-2025, and by
year it has netted about zero since 2003. A trial changes the training strategy or the model and asks whether the
years since 2003 improve.

Protocol (CLAUDE.md section 7, written before any trial ran)
- reference:    D, K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0-2 (a trial may name another reference).
- development:  2003-2021, the years on which a trial is judged.
- holdout:      2022-2025, reported once for each trial and never used to choose or to tune one.
- objects:      model (each seed scored alone; seed average, and the pooled series for the paired test), the seed
                ensemble, and the ensemble through a no-trade band whose width is the best of deploy.DELTAS on the
                development years of that run (so the holdout's band was fixed before it).
- kept:         a trial whose development net SR is above its reference's with a paired 95% interval that excludes 0,
                for the model or for the ensemble + band, and not clearly below for the other, and whose holdout
                difference is not negative. Anything else is "not supported", whatever its holdout says.
Every path is traded from cash on its first day at the paper's costs, books carried across years. A trial whose
window needs years before 1990 starts later (a 16-year window: 2006), and is compared with its reference on the
common days. These are diagnostics on test years, never Table 2 rows (CLAUDE.md section 8).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import deploy  # noqa: E402
from paired_test import cell, paired_bootstrap  # noqa: E402

from afsa.config import RESULTS_DIR, Config  # noqa: E402

D = {"universe": "wrds_broad_buffer600", "fill": "none", "entry_rule": "history", "ridge": 1e-4}   # dataset D, reference setting
SEEDS = (0, 1, 2)
FIRST, LAST = 1998, 2025
DEV, HOLDOUT = (2003, 2021), (2022, 2025)
PERIODS = {"1998-02": (1998, 2002), "dev 2003-21": DEV, "2014-21": (2014, 2021), "holdout 2022-25": HOLDOUT, "1998-2021": (1998, 2021)}
REPLAY_TOLERANCE = 1e-4                                       # scan.py: models replayed in other groups than trained

# name -> Config overrides on top of D with K = 30. "ref": the trial it is compared with (default: reference).
# A longer or shorter window keeps the reference's 480 optimizer steps (16 batches x 30 epochs), so that the trial
# changes the sample and not the amount of fitting.
TRIALS: dict[str, dict] = {
    "reference": {},
    # A. the training window: stale sample (shorter should help) against estimation noise (longer should help)
    "win04": {"train_years": 4, "epochs": 60},
    "win12": {"train_years": 12, "epochs": 20},
    "win16": {"train_years": 16, "epochs": 15},
    # B. less fitting
    "ep20": {"epochs": 20},
    "drop03": {"dropout": 0.3},
    "wd05": {"weight_decay": 0.5},
    # C. a slower signal: the sequence model sees the residual price path
    "cum": {"cumulative": True},
    # D. no standing tilt (EXTENSION.md section 7): the signals centred; the weights made orthogonal to the constant
    # and to sets of characteristic ranks inside the model, so the network is trained on the neutral book
    "center": {"signal_center": True},
    "neut_dollar": {"neutral": "dollar"},
    "neut_risk": {"neutral": "risk"},
    "neut_riskmom": {"neutral": "riskmom"},
    # reference points at other K and settings (trained by the scan on D)
    "cost075": {"cost_trade": 7.5e-4},
    "cost10": {"cost_trade": 1e-3},
    "K008": {"K": 8},
    # the K = 100 block of the scan on D (SCAN_BUFFER_BLOCK.md): training cost 5 / 7.5 / 10 bp x ridge 1e-4 / 3e-4 / 1e-3
    **{f"K100_{c:g}_{r:g}": {"K": 100, "cost_trade": c * 1e-4, "ridge": r} for c in (5, 7.5, 10) for r in (1e-4, 3e-4, 1e-3)},
    # E. a tighter hedge than any run of the scan (EXTENSION.md section 8): the ridge below 1e-4 at K = 100
    "K100_7.5_3e-05": {"K": 100, "cost_trade": 7.5e-4, "ridge": 3e-5},
    "K100_7.5_1e-05": {"K": 100, "cost_trade": 7.5e-4, "ridge": 1e-5},
    # F. no position for a member without a signal (EXTENSION.md section 12): the hedge's large shorts in new listings
    # are gone from the book itself. The K = 100 run is compared with the K = 30 one, both without those positions.
    "nosig0": {"nosignal_hedge": False},
    "K100_nosig0": {"K": 100, "cost_trade": 7.5e-4, "ridge": 1e-4, "nosignal_hedge": False, "ref": "nosig0"},
    # G. batch 5 (EXTENSION.md section 14): K = 200, test years 2014-2025 only ("first"), so its band width is chosen on
    # 2014-2021; extension_k200.py also re-chooses the comparators' widths on 2014-2021, which this harness does not.
    "K200_nosig0": {"K": 200, "cost_trade": 7.5e-4, "ridge": 1e-4, "nosignal_hedge": False, "ref": "K100_nosig0", "first": 2014},
    # H. batch 6 (EXTENSION.md section 16): the weight of the explained-variance term (default 100), without no-signal positions
    "lv10": {"nosignal_hedge": False, "lambda_var": 10.0, "ref": "nosig0"},
    "lv1000": {"nosignal_hedge": False, "lambda_var": 1000.0, "ref": "nosig0"},
}


def config(name: str) -> Config:
    over = {k: v for k, v in TRIALS[name].items() if k not in ("ref", "first")}
    return Config(**{"K": 30, **D, **over})


def reference_of(name: str) -> str:
    return TRIALS[name].get("ref", "reference")


def run_dir(name: str) -> Path:
    return RESULTS_DIR / "runs" / config(name).name()


def first_year(cfg: Config) -> int:
    """First test year of a trial: 1998, or later if its window would start before 1990."""
    return max(FIRST, 1990 + cfg.train_years)


def first_of(name: str) -> int:
    """First test year of a named trial: its "first" key if it has one (a trial run on fewer years), else first_year."""
    return TRIALS[name].get("first", first_year(config(name)))


def on_disk(name: str, seeds=SEEDS, weights: bool = False) -> bool:
    cfg, suffix = config(name), "_w.npz" if weights else ".npz"
    return all((run_dir(name) / f"seed{s}" / f"{y}{suffix}").exists() for s in seeds for y in range(first_of(name), LAST + 1))


# ---------------------------------------------------------------------------------------------- training (GPU)
def train(names: list[str], seeds=SEEDS, max_models: int | None = None) -> None:
    from reeval_weights import recover

    from afsa.panel import load_panel, panel_folder
    from afsa.run import run_config
    panel = None
    for name in names:
        cfg = config(name)
        years = list(range(first_of(name), LAST + 1))
        if panel is None or panel.universe != panel_folder(cfg):
            panel = None                                                     # release the previous sample first
            panel = load_panel(cfg, "cuda")
        t0 = time.time()
        folder = run_config(cfg, list(seeds), years, panel, max_models or (12 if cfg.K >= 100 else None))
        n, worst = recover(folder, panel, years=years, tolerance=REPLAY_TOLERANCE)
        print(f"{name}: {folder.name}, test years {years[0]}-{years[-1]}, {n} weight files (largest replay difference {worst:.1e}), "
              f"{time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------- scoring (CPU)
def sr(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 1 and x.std() > 0 else float("nan")


_NO_SIGNAL: dict[tuple, np.ndarray] = {}


def no_signal(cfg: Config, dates: np.ndarray, permnos: np.ndarray) -> np.ndarray:
    """[T, S] True where a stock is a member that day without a signal under entry_rule history: listed for at most
    `lookback` trading days, or no annual accounts entered (the month blocks of afsa.build_history). Such a stock
    still gets a position from the hedge. EXTENSION.md section 9: in 2022-2025 those positions are large, always
    short and were lucky, so the trials are also read with their P&L set aside."""
    from afsa.config import TRAIN_DATA_DIR
    from afsa.panel import panel_folder

    key = (panel_folder(cfg), len(dates), str(dates[0]))
    if key not in _NO_SIGNAL:
        H = TRAIN_DATA_DIR / panel_folder(cfg)
        bdates = np.load(H / "dates.npy").astype("datetime64[D]")
        row, n_rows, bperm, accounts = np.load(H / "row.npy"), np.load(H / "n_rows.npy"), np.load(H / "permno.npy"), np.load(H / "accounts.npy")
        bmember, listed = np.load(H / "member.npy", mmap_mode="r"), np.load(H / "listed.npy", mmap_mode="r")
        day_of = {d: i for i, d in enumerate(dates)}
        mask = np.zeros((len(dates), len(permnos)), dtype=bool)
        for b in range(len(n_rows)):
            for r in range(cfg.lookback, int(n_rows[b])):
                t = day_of.get(bdates[row[b, r]])
                if t is None:
                    continue
                mem = np.asarray(bmember[b, r])
                ns = mem & ~((np.asarray(listed[b, r]) > cfg.lookback) & accounts[b])
                if ns.any():
                    col = np.searchsorted(permnos, bperm[b][ns])
                    ok = (col < len(permnos)) & (permnos[np.minimum(col, len(permnos) - 1)] == bperm[b][ns])
                    mask[t, col[ok]] = True
        _NO_SIGNAL[key] = mask
    return _NO_SIGNAL[key]


def paths(name: str, ex_nosignal: bool = False) -> dict[str, pd.DataFrame]:
    """Daily results of a trial's objects, each one path from cash: the seeds' pooled series, the seed ensemble, and
    the ensemble through the band whose width is best on the development years. `attrs` carry what was chosen.

    ex_nosignal: the P&L of positions in members without a signal (`no_signal`) is set aside from every object; the
    positions and their costs stay, so only what those positions earned or lost is removed."""
    run = run_dir(name)
    cfg, dates, R, member, Ws = deploy.load(run, first_of(name), LAST)
    aside = R * no_signal(cfg, dates, deploy.market(cfg, first_of(name), LAST)[1]) if ex_nosignal else None

    def frame(H: np.ndarray) -> pd.DataFrame:
        daily = deploy.trade(H, R, dates)
        if aside is not None:
            luck = (H * aside).sum(1)
            daily["gross"], daily["net"] = daily["gross"] - luck, daily["net"] - luck
        return daily

    seeds = [frame(W) for W in Ws]
    out = {"model": sum(seeds) / len(seeds)}
    out["model"].attrs["seed average"] = {p: float(np.mean([sr(s.loc[_mask(s, a, b), "net"]) for s in seeds])) for p, (a, b) in PERIODS.items()}
    E = deploy.ensemble(Ws)
    out["ensemble"] = frame(E)
    grid = {d: frame(deploy.band_path(E, member, d)) for d in deploy.DELTAS}
    dev = {d: sr(g.loc[_mask(g, *DEV), "net"]) for d, g in grid.items()}
    best = max(dev, key=lambda d: (dev[d], -d))                              # ties: the narrower band
    out["ensemble + band"] = grid[best]
    out["ensemble + band"].attrs["delta"] = best
    return out


def _mask(daily: pd.DataFrame, a: int, b: int) -> np.ndarray:
    y = daily.index.year
    return (y >= a) & (y <= b)


def rows(name: str, mine: dict[str, pd.DataFrame], ref: dict[str, pd.DataFrame] | None) -> list[dict]:
    out = []
    for obj, daily in mine.items():
        r = {"trial": name, "object": obj + (f" {daily.attrs['delta']:g}" if "delta" in daily.attrs else "")}
        for p, (a, b) in PERIODS.items():
            x = daily[_mask(daily, a, b)]
            if len(x) < 200 or x.index.year.min() > a:                       # the trial does not cover the period
                r[p] = np.nan
                continue
            r[p] = daily.attrs["seed average"][p] if obj == "model" else sr(x["net"])
            if p in ("dev 2003-21", "holdout 2022-25"):
                r[f"turnover {p[-7:]}"] = float(x["turnover"].mean())
        if ref is not None:
            base = ref[obj]
            days = daily.index.intersection(base.index)
            for p in ("dev 2003-21", "2014-21", "holdout 2022-25"):
                a, b = PERIODS[p]
                m = (days.year >= a) & (days.year <= b)
                if m.sum() >= 200:
                    lo = days[m].year.min()
                    r[f"d {p}"] = cell(paired_bootstrap(daily.loc[days[m], "net"].to_numpy(), base.loc[days[m], "net"].to_numpy())) + ("" if lo == a else f" from {lo}")
        out.append(r)
    return out


SPLIT_ERAS = {"1998-02": (1998, 2002), "2003-13": (2003, 2013), "2014-21": (2014, 2021), "dev 2003-21": DEV, "2022-23": (2022, 2023), "2024-25": (2024, 2025),
              "holdout 2022-25": HOLDOUT}


def split(name: str, which: str = "riskmom") -> pd.DataFrame:
    """The seed ensemble of a trial as a tilt plus a neutral part, by era (EXTENSION.md section 5.7). Each day the
    saved weights are split, over the day's members and with nothing rescaled, into their least-squares projection
    on the constant and the characteristic ranks of NEUTRAL_SETS[which] (the tilt) and the rest. Gross return of each
    in % a year, the tilt's share of gross exposure, and each part's gross Sharpe ratio."""
    from afsa.config import CHARACTERISTICS, NEUTRAL_SETS, TRAIN_DATA_DIR, data_folder

    cfg = config(name)
    first = first_of(name)
    _, dates, R, _, Ws = deploy.load(run_dir(name), first, LAST)
    E, permnos = deploy.ensemble(Ws), deploy.market(cfg, first, LAST)[1]
    folder = TRAIN_DATA_DIR / data_folder(cfg.universe, cfg.fill)
    rows_ = np.searchsorted(np.load(folder / "dates.npy").astype("datetime64[D]"), dates)
    permno, ranks = np.load(folder / "permno.npy", mmap_mode="r"), np.load(folder / "ranks.npy", mmap_mode="r")
    idx = [CHARACTERISTICS.index(c) for c in NEUTRAL_SETS[which]]
    tilt, neutral, size, long_ = (np.zeros(len(dates)) for _ in range(4))
    for t in range(len(dates)):
        pn = np.asarray(permno[rows_[t]])
        on = pn >= 0
        col = np.searchsorted(permnos, pn[on])
        A = np.column_stack([np.ones(len(col)), np.asarray(ranks[rows_[t]][on], dtype=np.float64)[:, idx]])
        w, r = E[t, col], R[t, col]
        part = A @ np.linalg.lstsq(A, w, rcond=None)[0]
        tilt[t], neutral[t], size[t], long_[t] = part @ r, (w - part) @ r, np.abs(part).sum(), w.mean() * r.sum()
    years, out = pd.DatetimeIndex(dates).year.to_numpy(), {}
    for era, (a, b) in SPLIT_ERAS.items():
        m = (years >= a) & (years <= b)
        if m.sum() < 200 or years.min() > a:
            continue
        out[era] = {"tilt %": tilt[m].mean() * 25200, "of which net long %": long_[m].mean() * 25200, "neutral part %": neutral[m].mean() * 25200,
                    "tilt share of exposure": size[m].mean(), "tilt SR": sr(pd.Series(tilt[m])), "neutral part SR": sr(pd.Series(neutral[m]))}
    return pd.DataFrame(out)


def by_year(mine: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({obj: daily["net"].groupby(daily.index.year).apply(sr) for obj, daily in mine.items()})


def report(names: list[str], show_years: bool = False, ex_nosignal: bool = False) -> pd.DataFrame:
    cache: dict[str, dict[str, pd.DataFrame]] = {}

    def get(n: str) -> dict[str, pd.DataFrame]:
        if n not in cache:
            cache[n] = paths(n, ex_nosignal)
        return cache[n]

    table = []
    for name in names:
        if not on_disk(name, weights=True):
            print(f"{name}: not on disk with weights ({run_dir(name).name})", flush=True)
            continue
        ref = reference_of(name)
        table += rows(name, get(name), None if name == ref else get(ref))
        if show_years:
            print(f"\n{name}: net SR by year\n{by_year(get(name)).round(2).T.to_string()}", flush=True)
    return pd.DataFrame(table).set_index(["trial", "object"]) if table else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", choices=("train", "report"))
    p.add_argument("trials", nargs="*", help="trial names [default for report: all on disk]")
    p.add_argument("--list", action="store_true")
    p.add_argument("--by-year", action="store_true", help="report: also the net SR of each object by year")
    p.add_argument("--split", action="store_true", help="report: also each trial's seed ensemble as a tilt plus a neutral part, by era")
    p.add_argument("--ex-nosignal", action="store_true", help="report: set aside the P&L of positions in members without a signal (hedge positions in new listings)")
    p.add_argument("--max-models", type=int, default=None, help="models per GPU pass [default: afsa.run's; 12 from K = 100]")
    p.add_argument("--out", type=Path, default=None, help="report: also write the table to this markdown file")
    args = p.parse_args()
    pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 500)
    unknown = [t for t in args.trials if t not in TRIALS]
    if unknown:
        raise SystemExit(f"unknown trials {unknown}; --list shows them")
    if args.list or args.action is None:
        for name, over in TRIALS.items():
            state = "weights" if on_disk(name, weights=True) else "trained" if on_disk(name) else "-"
            print(f"{name:12s} {state:8s} from {first_of(name)}  {over}")
    elif args.action == "train":
        train(args.trials, max_models=args.max_models)
    else:
        names = args.trials or [n for n in TRIALS if on_disk(n, weights=True)]
        if "reference" not in names:
            names = ["reference"] + names
        t = report(names, args.by_year, args.ex_nosignal)
        if args.ex_nosignal:
            print("\nThe P&L of positions in members without a signal is set aside from every object below.")
        if args.split:
            for n in names:
                if on_disk(n, weights=True):
                    print(f"\n{n}: seed ensemble as tilt + neutral part (gross)\n{split(n).round(2).to_string()}", flush=True)
        num = t.select_dtypes("number").columns
        text = t.to_string(float_format=lambda x: f"{x:.2f}", na_rep="")
        print("\n" + text)
        if args.out:
            t[num] = t[num].round(2)
            args.out.write_text(t.reset_index().to_markdown(index=False, floatfmt=".2f") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
