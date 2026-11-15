"""Pilot on the validation years: the ranges of a hyperparameter scan, set before any test year is looked at
(analysis/gap/REPORT.md, section 10.6; CLAUDE.md, section 8).

    python analysis/gap/pilot.py --list              # the pilot grid, and what is on disk
    python analysis/gap/pilot.py train               # trains what is missing, 1996-1997 only (~10 min of GPU at K = 30)
    python analysis/gap/pilot.py weights             # daily weights of the saved models, for the ensemble and the band
    python analysis/gap/pilot.py                     # the tables -> analysis/gap/PILOT.md (CPU, ~1 min)
    python analysis/gap/pilot.py --best --K 8 30     # each K's best layer, as K=BP/RIDGE/EPOCHS for selection.py --per-k

A wide, coarse grid of the scan's three levers (scan.py) at one K, trained on 6-year windows and scored on 1996-1997
only. Nothing here trains on, or reads, a year after 1997.
- training cost per unit of turnover: 2.5, 5, 7.5, 10, 15 and 20 bp. Every result is scored at the paper's costs
  (5 bp per unit of turnover, 1 bp per unit short), whatever cost the model was trained with.
- ridge: 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 and 3e-3
- every pair of the two at 30 epochs (42 settings); epochs 5, 10, 15, 20 and 50 at 5 bp and at 10 bp, both with
  ridge 1e-4 (10 settings more)
- the no-trade band: deploy.py's 9 widths on the seeds' ensemble of every setting (no training)
Defaults: K = 30, seeds 0-2, the rebuilt-history rule on wrds_broad (universe=wrds_broad fill=none entry_rule=history).
`--only`, `--K`, `--set` and `--out` give the same report for another grid, K or arrays: its tables and edge checks
follow the training costs, ridges and epochs of the settings given (levers), e.g. the 28-setting grid on the buffered
list (REPORT.md, section 8).

Differences are paired block bootstraps (paired_test.py: 21-day blocks, 2,000 draws) of pooled series: the seeds'
average daily net return for the model, the ensemble's own series for the layer. A setting is "clearly worse" than
the best when the 95% interval of its difference from the best lies below 0. The best is the maximum of many noisy
figures, so this flags a little too often: use it to drop ranges, not single points.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy import DELTAS  # noqa: E402
from paired_test import cell, compare  # noqa: E402
from scan import PERIODS, SEEDS, Setting, complete, has_weights, parse_set, parse_setting, train, weights  # noqa: E402
from selection import _market_row, beta, frame, md, paths, pooled, seed_frames  # noqa: E402
from variants import stats  # noqa: E402

from afsa.config import TRADING_DAYS, data_folder  # noqa: E402
from afsa.metrics import PAPER_TABLE2  # noqa: E402

OUT = Path(__file__).resolve().parent / "PILOT.md"
COSTS = (2.5e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3)
RIDGES = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
EPOCHS = (5, 10, 15, 20, 30, 50)
EPOCH_POINTS = ((5e-4, 1e-4), (1e-3, 1e-4))                                 # (training cost, ridge) of the two epoch sweeps
GRID = tuple(Setting(c, r, 30) for c in COSTS for r in RIDGES) + tuple(
    Setting(c, r, e) for c, r in EPOCH_POINTS for e in EPOCHS if e != 30)
REFERENCE = Setting(5e-4, 1e-4, 30)                                          # the rebuilt-history runs (CLAUDE.md, section 9)
EXTRA = ["universe=wrds_broad", "fill=none", "entry_rule=history"]
FIRST, LAST, TRAIN_YEARS = PERIODS["validation"]


def sci(x: float) -> str:
    """1e-05 -> '1e-5', 0.0003 -> '3e-4', 0.00025 -> '2.5e-4'."""
    e = int(np.floor(np.log10(x) + 1e-9))
    return f"{x / 10 ** e:g}e{e}"


def name(s: Setting) -> str:
    return f"{s.cost_trade * 1e4:g} bp, {sci(s.ridge)}, {s.epochs} ep"


# ---------------------------------------------------------------------------------------------- one setting
def training_fit(s: Setting, K: int, extra: dict, seeds: tuple[int, ...]) -> dict:
    """The last epoch on the training windows (batches in training mode, averaged over seeds and both years): the gross
    Sharpe ratio and turnover, which do not depend on the training cost; and whether every stored figure is finite."""
    folder, cfg = s.run_dir(K, TRAIN_YEARS, extra), s.config(K, TRAIN_YEARS, extra)
    rows = []
    for seed in seeds:
        for y in range(FIRST, LAST + 1):
            z = np.load(folder / f"seed{seed}" / f"{y}.npz")
            h = dict(zip([str(c) for c in z["history_columns"]], z["history"][-1]))
            finite = all(np.isfinite(z[k]).all() for k in ("gross", "turnover", "short", "history"))
            rows.append({"fit": h["sharpe_gross"] * np.sqrt(TRADING_DAYS / cfg.sharpe_days), "fit turnover": h["turnover"], "finite": finite})
    t = pd.DataFrame(rows)
    return {"fit": t["fit"].mean(), "fit turnover": t["fit turnover"].mean(), "finite": bool(t["finite"].all())}


def one(s: Setting, K: int, extra: dict, seeds: tuple[int, ...], market: pd.Series) -> dict | None:
    """Validation statistics of one setting, or None if its runs or weights are not all on disk."""
    p = paths(s, K, validation=True, extra=extra, seeds=seeds)
    if p is None:
        return None
    frames = seed_frames(p)
    t = pd.DataFrame([{**stats(f), "beta": beta(f, market)} for f in frames])
    g = np.corrcoef(np.stack([f["gross"].to_numpy() for f in frames]))
    bands = {d: stats(frame(p, f"band {d:g}")) for d in DELTAS}
    d = max(DELTAS, key=lambda x: bands[x]["SR_net"])                      # ties go to the narrower band
    return {"SR": t["SR"].mean(), "σ": t["sigma"].mean(), "SR net": t["SR_net"].mean(), "SR net, seed sd": t["SR_net"].std(ddof=1),
            "Turnover": t["turnover"].mean(), "Short": t["short"].mean(), "Beta": t["beta"].mean(),
            "Seed corr": float(g[np.triu_indices(len(frames), 1)].mean()), **training_fit(s, K, extra, seeds),
            "Ensemble SR net": bands[0.0]["SR_net"], "δ": d, "Layer SR net": bands[d]["SR_net"], "Layer turnover": bands[d]["turnover"],
            "_model": pooled(frames), "_layer": pooled([frame(p, f"band {d:g}")])}


# ---------------------------------------------------------------------------------------------- tables
def levers(settings: tuple[Setting, ...]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[tuple[float, float], ...], tuple[int, ...]]:
    """The ranges of a grid: its training costs and ridges at 30 epochs, the (training cost, ridge) points trained at
    more than one number of epochs, and those epochs. For GRID: COSTS, RIDGES, EPOCH_POINTS and EPOCHS."""
    at30 = [s for s in settings if s.epochs == 30]
    runs: dict[tuple[float, float], set[int]] = {}
    for s in settings:
        runs.setdefault((s.cost_trade, s.ridge), set()).add(s.epochs)
    points = tuple(sorted(pt for pt, es in runs.items() if len(es) > 1))
    return (tuple(sorted({s.cost_trade for s in at30})), tuple(sorted({s.ridge for s in at30})), points,
            tuple(sorted({e for pt in points for e in runs[pt]})))


def grid_table(res: dict[Setting, dict], value, costs: tuple[float, ...], ridges: tuple[float, ...]) -> pd.DataFrame:
    """Training cost (rows) by ridge (columns) at 30 epochs; `value(setting, result)` gives each cell."""
    rows = {}
    for c in costs:
        rows[f"{c * 1e4:g} bp"] = {sci(r): (value(Setting(c, r, 30), res[Setting(c, r, 30)]) if Setting(c, r, 30) in res else "")
                                   for r in ridges}
    return pd.DataFrame(rows).T


def marked(x: float, best: bool, worse: bool) -> str:
    return f"**{x:.2f}**" if best else f"{x:.2f} †" if worse else f"{x:.2f}"


def section(K: int, extra: dict, seeds: tuple[int, ...], settings: tuple[Setting, ...]) -> list[str]:
    cfg = REFERENCE.config(K, TRAIN_YEARS, extra)
    market = _market_row(cfg.universe, cfg.fill)
    res = {s: r for s in settings if (r := one(s, K, extra, seeds, market)) is not None}
    if not res:
        return [f"## K = {K}", "No complete setting with weights on disk."]
    top = max(res, key=lambda s: res[s]["SR net"])                          # the selection rules' criterion (selection.py)
    top_layer = max(res, key=lambda s: res[s]["Layer SR net"])
    for s, r in res.items():
        r["Δ vs best"] = compare(r["_model"], res[top]["_model"])["all"]
        r["Δ layer vs best layer"] = compare(r["_layer"], res[top_layer]["_layer"])["all"]
        r["worse"], r["layer worse"] = r["Δ vs best"]["hi"] < 0, r["Δ layer vs best layer"]["hi"] < 0
        if REFERENCE in res:
            r["Δ vs reference"] = compare(r["_model"], res[REFERENCE]["_model"])["all"]

    out = [f"## K = {K}"]
    costs, ridges, points, epochs = levers(settings)                        # the ranges of the grid being reported
    # ---- summary
    others = [s for s in res if s != top]
    half = np.median([(res[s]["Δ vs best"]["hi"] - res[s]["Δ vs best"]["lo"]) / 2 for s in others]) if others else np.nan
    at30 = [s for s in res if s.epochs == 30]
    keep = {lever: sorted({getattr(s, lever) for s in at30 if not res[s][flag]}) for lever, flag in
            (("cost_trade", "worse"), ("ridge", "worse"))}
    keep_layer = {lever: sorted({getattr(s, lever) for s in at30 if not res[s]["layer worse"]}) for lever in ("cost_trade", "ridge")}
    epochs_ok = {pt: sorted({s.epochs for s in res if (s.cost_trade, s.ridge) == pt and not res[s]["worse"]}) for pt in points}
    epochs_ok_layer = {pt: sorted({s.epochs for s in res if (s.cost_trade, s.ridge) == pt and not res[s]["layer worse"]}) for pt in points}
    paper = PAPER_TABLE2.get(K)
    implied = ((paper[1] - paper[4]) / 252 * 100 - 0.5) / 5 if paper else np.nan
    edges = []
    for label, s in (("model", top), ("layer", top_layer)):
        if len(costs) > 1 and s.cost_trade in (costs[0], costs[-1]):
            edges.append(f"the best {label} setting's training cost is at the edge of the pilot range")
        if len(ridges) > 1 and s.ridge in (ridges[0], ridges[-1]):
            edges.append(f"the best {label} setting's ridge is at the edge of the pilot range")
        if len(epochs) > 1 and s.epochs in (epochs[0], epochs[-1]):
            edges.append(f"the best {label} setting's epochs are at the edge of the pilot range")
    if res[top_layer]["δ"] == DELTAS[-1]:
        edges.append("the best layer's band width is the widest of the grid")
    warnings = [x for x in (                                                # setting names hold commas: semicolons between them
        "non-finite results in: " + "; ".join(name(s) for s, r in res.items() if not r["finite"]),
        "beta above 0.10 in: " + "; ".join(name(s) for s, r in res.items() if r["Beta"] > 0.10),
        "seed sd of the net SR above 0.5 in: " + "; ".join(name(s) for s, r in res.items() if r["SR net, seed sd"] > 0.5)) if not x.endswith(": ")]
    t30 = [res[s]["Turnover"] for s in at30]
    bp = lambda xs: ", ".join(f"{x * 1e4:g}" for x in xs) + " bp"         # noqa: E731
    out += ["**Summary** (validation years 1996–1997 only)",
            f"- **Best model setting:** {name(top)}: net SR {res[top]['SR net']:.2f} (seed sd {res[top]['SR net, seed sd']:.2f}), "
            f"turnover {res[top]['Turnover']:.2f}, beta {res[top]['Beta']:.2f}."
            + (f" The reference ({name(REFERENCE)}) gets {res[REFERENCE]['SR net']:.2f}; best minus reference, pooled: "
               f"{cell(res[top]['Δ vs reference'])}." if REFERENCE in res and top != REFERENCE else ""),
            f"- **Best layer** (seed ensemble + band): {name(top_layer)} with a band of {res[top_layer]['δ']:g}: net SR "
            f"{res[top_layer]['Layer SR net']:.2f}, turnover {res[top_layer]['Layer turnover']:.2f}.",
            f"- **Noise:** the paired 95% intervals against the best model setting have a median half-width of {half:.2f}. "
            f"{sum(res[s]['worse'] for s in others)} of {len(others)} settings are clearly worse than it; for the layer, "
            f"{sum(res[s]['layer worse'] for s in res if s != top_layer)} of {len(res) - 1}.",
            f"- **Not clearly worse than the best, at 30 epochs** (a value is listed if any setting with it is): model, training cost "
            f"{bp(keep['cost_trade'])} and ridge {', '.join(map(sci, keep['ridge']))}; layer, training cost {bp(keep_layer['cost_trade'])} "
            f"and ridge {', '.join(map(sci, keep_layer['ridge']))}.",
            "- **Epochs not clearly worse:** " + "; ".join(
                f"at {c * 1e4:g} bp, ridge {sci(r)}: model {', '.join(map(str, epochs_ok[(c, r)])) or 'none'}, layer "
                f"{', '.join(map(str, epochs_ok_layer[(c, r)])) or 'none'}" for c, r in points) + "." if points else "",
            f"- **Turnover** of the models at 30 epochs: {min(t30):.2f} to {max(t30):.2f} a day. The paper's implied turnover at K = {K} "
            f"is {implied:.2f} (test years, short exposure 0.5)." if t30 else "",
            "- **Edges:** " + ("; ".join(edges) + "." if edges else "none: every best value lies inside the pilot ranges."),
            "- **Warnings:** " + (". ".join(w[0].upper() + w[1:] for w in warnings) + "." if warnings else "none.")]

    # ---- grids at 30 epochs
    grid = lambda value: md(grid_table(res, value, costs, ridges), "Training cost")    # noqa: E731
    out += ["**Model: net SR, seed average** (rows: training cost; columns: ridge; 30 epochs). Bold: the best; † clearly worse than it.",
            grid(lambda s, r: marked(r["SR net"], s == top, r["worse"])),
            "**Layer: seed ensemble + band, net SR (band width chosen on the same years)**. Bold: the best; † clearly worse than it.",
            grid(lambda s, r: f"{marked(r['Layer SR net'], s == top_layer, r['layer worse'])} ({r['δ']:g})"),
            "**Model: turnover a day**", grid(lambda s, r: r["Turnover"]),
            "**Model: beta against the equal-weighted market**", grid(lambda s, r: r["Beta"]),
            "**Model: gross SR on the training windows** (last epoch; the fit)", grid(lambda s, r: r["fit"])]

    # ---- epochs
    rows, one_ridge = {}, len({r for _, r in points}) == 1
    for e in epochs:
        row = {}
        for c, r in points:
            s = Setting(c, r, e)
            if s in res:
                x, tag = res[s], f"{c * 1e4:g} bp" + ("" if one_ridge else f", {sci(r)}")
                row.update({f"{tag}: SR net": marked(x["SR net"], s == top, x["worse"]), f"{tag}: turnover": x["Turnover"],
                            f"{tag}: fit": x["fit"], f"{tag}: layer": f"{marked(x['Layer SR net'], s == top_layer, x['layer worse'])} ({x['δ']:g})"})
        rows[e] = row
    if points:
        out += [f"**Epochs** ({f'ridge {sci(points[0][1])}; ' if one_ridge else ''}*fit*: gross SR on the training windows)",
                md(pd.DataFrame(rows).T, "Epochs")]

    # ---- every setting
    rows = {}
    for s, r in res.items():
        row = {k: r[k] for k in ("SR", "σ", "SR net", "SR net, seed sd", "Turnover", "Short", "Beta", "Seed corr", "fit", "fit turnover",
                                 "Ensemble SR net")}
        row["Layer SR net (δ)"] = f"{r['Layer SR net']:.2f} ({r['δ']:g})"
        row["Layer turnover"] = r["Layer turnover"]
        row["Δ vs best model, pooled"] = "" if s == top else cell(r["Δ vs best"])
        if "Δ vs reference" in r:
            row["Δ vs reference, pooled"] = "" if s == REFERENCE else cell(r["Δ vs reference"])
        rows[name(s) + (" (reference)" if s == REFERENCE else "")] = row
    out += ["**Every setting** (model columns are seed averages; *fit*: the last epoch on the training windows)", md(pd.DataFrame(rows).T, "Setting")]
    return out


def write(Ks: list[int], extra: dict, seeds: tuple[int, ...], settings: tuple[Setting, ...], out: Path) -> None:
    cfg = REFERENCE.config(Ks[0], TRAIN_YEARS, extra)
    parts = ["# Pilot on the validation years: where the scan's levers matter",
             "Generated by `analysis/gap/pilot.py`; read with `analysis/gap/REPORT.md`, section 10.6. Every model is trained on 6-year windows "
             f"and scored on 1996–1997 only (`{data_folder(cfg.universe, cfg.fill)}`, entry rule `{cfg.entry_rule}`, seeds "
             f"{', '.join(map(str, seeds))}); nothing after 1997 is read. Scored at the paper's costs (5 bp per unit of turnover, 1 bp per unit "
             "short), whatever cost the model was trained with. Sharpe ratios are annualized, on stock-by-stock excess returns.",
             "- **Settings** are *training cost, ridge, epochs*. The model is each seed's portfolio, statistics averaged over seeds (the paper's "
             "object). The layer is the seeds' ensemble through a no-trade band, its width the best of "
             f"{len(DELTAS)} on the same two years; a deployment layer, never a Table 2 row.",
             "- **Differences** are paired block bootstraps (21-day blocks, 2,000 draws) of pooled series: the seeds' average daily net return "
             "for the model, the ensemble's own series for the layer. *Clearly worse* (†): the 95% interval of the difference from the best "
             "lies below 0. The best is the maximum of many noisy figures, so † appears a little too often: use it to drop ranges, not points.",
             f"- **Reference:** {name(REFERENCE)}, the setting of the rebuilt-history runs."]
    for K in Ks:
        parts += section(K, extra, seeds, settings)
    out.write_text("\n\n".join(p for p in parts if p) + "\n", encoding="utf-8")


def listing(Ks: list[int], settings: tuple[Setting, ...], extra: dict, seeds: tuple[int, ...]) -> None:
    print(f"{len(settings)} settings, seeds {list(seeds)}, validation years {FIRST}-{LAST}; + = complete, w = with weights")
    for s in settings:
        print(f"{name(s):24s} " + "  ".join(f"K={K}: {'+' if complete(s, K, 'validation', extra, seeds) else '.'}"
                                             f"{'w' if has_weights(s, K, 'validation', extra, seeds) else ' '}" for K in Ks))


def best_layers(Ks: list[int], extra: dict, seeds: tuple[int, ...], settings: tuple[Setting, ...]) -> dict[int, tuple[Setting, float, float]]:
    """Each K's best layer on the validation years, as section() reports it (ties go to the earlier setting): the
    setting, its band width and its net SR. The settings of a K scan in which each K is tuned on its own."""
    out = {}
    for K in Ks:
        cfg = REFERENCE.config(K, TRAIN_YEARS, extra)
        market = _market_row(cfg.universe, cfg.fill)
        res = {s: r for s in settings if (r := one(s, K, extra, seeds, market)) is not None}
        if res:
            top = max(res, key=lambda s: res[s]["Layer SR net"])
            out[K] = (top, res[top]["δ"], res[top]["Layer SR net"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", choices=("train", "weights"), help="train the missing runs, or write their daily weights")
    p.add_argument("--list", action="store_true", help="show the grid and what is on disk")
    p.add_argument("--best", action="store_true", help="print each K's best layer setting, last line as K=BP/RIDGE/EPOCHS for selection.py --per-k")
    p.add_argument("--K", type=int, nargs="+", default=[30])
    p.add_argument("--only", nargs="+", default=None, metavar="BP/RIDGE/EPOCHS", help="only these settings, e.g. 5/1e-4/30")
    p.add_argument("--set", nargs="*", default=EXTRA, metavar="FIELD=VALUE", help=f"Config fields for every run [default: {' '.join(EXTRA)}]")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="random seeds [default: 0 1 2]")
    p.add_argument("--out", type=Path, default=OUT, help="markdown file [default: analysis/gap/PILOT.md]")
    args = p.parse_args()
    settings = tuple(parse_setting(t) for t in args.only) if args.only else GRID
    extra, seeds = parse_set(args.set), tuple(args.seeds)
    if args.list:
        listing(args.K, settings, extra, seeds)
    elif args.best:
        best = best_layers(args.K, extra, seeds, settings)
        for K, (s, d, sr) in best.items():
            print(f"K = {K}: {name(s)}, band {d:g}, validation net SR {sr:.2f}")
        print(" ".join(f"{K}={s.cost_trade * 1e4:g}/{sci(s.ridge)}/{s.epochs}" for K, (s, _, _) in best.items()))
    elif args.action == "train":
        train(args.K, ["validation"], settings, extra, seeds)
    elif args.action == "weights":
        weights(args.K, ["validation"], settings, extra, seeds)
    else:
        write(args.K, extra, seeds, settings, args.out)
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
