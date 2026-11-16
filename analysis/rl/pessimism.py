"""Pessimism from seed disagreement on the seed ensembles (CLAUDE.md section 7, row of 2026-11-16; EXPLORATION.md
section 6.1). CPU only: no training.

    python analysis/rl/pessimism.py      -> analysis/rl/PESSIMISM_RESULTS.md (CPU, a few minutes)

Offline RL plans with a lower bound where the data cannot support a value, r - lambda * u (the deck's p383-p384), with
the disagreement of a model ensemble as u (p368). Here u is s[t, i], the standard deviation across seeds 0-2 of stock
i's target on day t, and the ensemble trades less where its seeds disagree, in one of two forms:
  shrink   the seeds' average target w (before rescaling) times w^2 / (w^2 + kappa s^2), rescaled to unit gross
           exposure, then traded as the object trades its ensemble;
  widen    kappa * s added to the object's band width (for the learned rules, to the seeds' averaged band widths).
kappa = 0 is the object's ensemble as reported in TRADING_RULE_RESULTS.md and TRADING_STATE_RESULTS.md.

Objects (seed ensembles on D, K = 30, 5 bp, ridge 1e-4, 30 epochs): D + band after (4e-4, movers-only), the fixed band
trained inside the model + band, the learned rule (TRADING_RULE.md), and its views S and SM (TRADING_STATE.md). Every
path starts from cash on its period's first day and carries its book across years, at the paper's costs.

kappa is chosen on 1996-1997 for each object and form (highest net SR; ties: the smaller kappa). Pessimism is kept for
an object if its chosen kappa > 0 and the paired 95% interval against kappa = 0 on 1996-1997 lies above 0. The test
years are reported at the chosen kappa, and the whole grid as a diagnostic; they never decide.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "analysis" / "gap", HERE):
    sys.path.insert(0, str(p))
import band_trained as bt  # noqa: E402
from deploy import band_path, market, seed_weights, state_path, trade  # noqa: E402
from trading_rule import DELTA, interval  # noqa: E402
from trading_state import ARMS, view_weights  # noqa: E402
from variants import stats  # noqa: E402

GRID = {"shrink": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0), "widen": (0.0, 0.25, 0.5, 1.0, 2.0)}
OBJECTS = ("D + band after", "fixed band trained + band", "rule", "S", "SM")


def unit(W: np.ndarray) -> np.ndarray:
    return W / np.maximum(np.abs(W).sum(1, keepdims=True), 1e-12)


def shrink(Ws: list[np.ndarray], kappa: float) -> np.ndarray:
    """The seeds' average target times w^2 / (w^2 + kappa s^2), rescaled to unit gross exposure (kappa = 0: the
    ensemble of deploy.ensemble)."""
    w = sum(Ws) / len(Ws)
    if kappa == 0:
        return unit(w)
    s2 = np.var(np.stack(Ws), axis=0)
    w2 = w * w
    return unit(w * np.divide(w2, w2 + kappa * s2, out=np.zeros_like(w2), where=(w2 + kappa * s2) > 0))


def seeds_of(name: str, base, dates, permnos) -> dict:
    """The seeds' targets of one object, and what its trading needs."""
    seeds = sorted(bt.run_dir(base).glob("seed*"), key=lambda p: int(p.name[4:]))
    if name == "D + band after":
        return {"W": [seed_weights(d, dates, permnos) for d in seeds]}
    if name == "fixed band trained + band":
        return {"W": [bt.saved(bt.run_dir(dataclasses.replace(base, band=DELTA)) / d.name, dates, permnos, "weights") for d in seeds]}
    rule = dataclasses.replace(base, trade_head=True)
    if name == "rule":
        dirs = [bt.run_dir(rule) / d.name for d in seeds]
        W, D, E = ([bt.saved(d, dates, permnos, k) for d in dirs] for k in ("weights", "delta", "eta"))
        return {"W": W, "delta": sum(D) / len(D), "eta": sum(E) / len(E)}
    dirs = [bt.run_dir(dataclasses.replace(rule, **ARMS[name])) / d.name for d in seeds]
    out = {"W": [], "rules": [], "has": None}
    for d in dirs:
        W, A, K, G = (bt.saved(d, dates, permnos, k) for k in ("weights", "dial_a", "dial_k", "signal"))
        out["W"].append(W), out["rules"].append((A, K, view_weights(d, dates)))
        out["has"] = G != 0
    return out


def books(name: str, x: dict, W: np.ndarray, member: np.ndarray, widen: np.ndarray | float, const: dict) -> np.ndarray:
    """The object's ensemble trading of the targets W, with `widen` added to its band width."""
    if name in ("D + band after", "fixed band trained + band"):
        return band_path(W, member, DELTA + widen, rescale="moved")
    if name == "rule":
        return band_path(W, member, x["delta"] + widen, rescale="moved", eta=x["eta"])
    return state_path(W, member, x["rules"], x["has"], widen=None if np.isscalar(widen) and widen == 0 else
                      np.broadcast_to(widen, W.shape), **const)[0]


def period(first: int, last: int, train_years: int) -> tuple[dict, dict]:
    """(object, form, kappa) -> daily frame on first..last, and the size of the seeds' disagreement."""
    base = dataclasses.replace(bt.BASE, train_years=train_years)
    rule = dataclasses.replace(base, trade_head=True)
    const = {"band0": rule.head_band0, "band_max": rule.head_band_max, "eta0": rule.head_eta0}
    dates, permnos, R, member = market(base, first, last)
    out, size = {}, {}
    for name in OBJECTS:
        x = seeds_of(name, base, dates, permnos)
        s = np.std(np.stack(x["W"]), axis=0)
        w = np.abs(sum(x["W"]) / len(x["W"]))
        m = member & (w > 0)
        size[name] = {"median s / median |w|": float(np.median(s[m]) / np.median(w[m])),
                      "median s (bp of the book)": float(np.median(s[m]) * 1e4)}
        for form, grid in GRID.items():
            for kappa in grid:
                if kappa == 0 and form == "widen":
                    out[(name, form, kappa)] = out[(name, "shrink", 0.0)]
                    continue
                W = shrink(x["W"], kappa if form == "shrink" else 0.0)
                H = books(name, x, W, member, kappa * s if form == "widen" else 0.0, const)
                out[(name, form, kappa)] = trade(H, R, dates)
        del x, s, w, m
    return out, size


def sr(f: pd.DataFrame, years: tuple[int, int] | None = None) -> float:
    return stats(f if years is None else bt.span(f, years))["SR_net"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    val, size_v = period(*bt.VALIDATION, 6)
    test, size_t = period(bt.TEST[0], bt.HOLDOUT[1], 8)
    choice, grid_rows, test_rows, contrast_rows = {}, {}, {}, {}
    for name in OBJECTS:
        for form, grid in GRID.items():
            best = max(grid, key=lambda k: (round(sr(val[(name, form, k)]), 12), -k))      # ties: the smaller kappa
            d, lo, hi = interval(val[(name, form, best)]["net"], val[(name, form, 0.0)]["net"])
            kept = best > 0 and lo > 0
            choice[(name, form)] = {"κ chosen": best, "validation SR_net, κ = 0": sr(val[(name, form, 0.0)]),
                                    "validation SR_net, chosen": sr(val[(name, form, best)]),
                                    "validation difference": "" if best == 0 else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                                    "kept": "yes" if kept else "no"}
            for k in grid:
                grid_rows[(name, form, k)] = {"validation": sr(val[(name, form, k)]), "1998–2021": sr(test[(name, form, k)], bt.TEST),
                                              "2022–25": sr(test[(name, form, k)], bt.HOLDOUT),
                                              "turnover 1998–2021": float(bt.span(test[(name, form, k)], bt.TEST)["turnover"].mean())}
            if best == 0:
                continue
            f = test[(name, form, best)]
            test_rows[(name, form)] = {"κ": best, **{lab: sr(f, s) for lab, s in bt.SPANS},
                                       "turnover": float(bt.span(f, bt.TEST)["turnover"].mean())}
            cells = {}
            for lab, s in bt.SPANS:
                d, lo, hi = interval(bt.span(f, s)["net"], bt.span(test[(name, form, 0.0)], s)["net"])
                cells[lab] = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            contrast_rows[(name, form)] = cells
    flat = lambda rows: pd.DataFrame(rows).T.set_axis([", ".join(f"{v:g}" if isinstance(v, float) else v for v in k)   # noqa: E731
                                                       for k in rows])
    choice_t, grid_t = flat(choice), flat(grid_rows)
    test_t = flat(test_rows) if test_rows else pd.DataFrame()
    contrast_t = flat(contrast_rows) if contrast_rows else pd.DataFrame()
    size_t = pd.DataFrame({"validation": pd.DataFrame(size_v).T.iloc[:, 0], "1998–2025": pd.DataFrame(size_t).T.iloc[:, 0],
                           "s in bp, 1998–2025": pd.DataFrame(size_t).T.iloc[:, 1]})
    kept = [k for k, v in choice.items() if v["kept"] == "yes"]
    lines = [
        "# Pessimism from seed disagreement: results",
        "",
        "Written by `analysis/rl/pessimism.py`, as pre-registered in CLAUDE.md §7 (row of 2026-11-16; EXPLORATION.md §6.1). "
        "The seed ensembles of D at K = 30 (5 bp, ridge 1e-4, 30 epochs, seeds 0–2) trade less where the seeds disagree. "
        "s is the standard deviation across the seeds of a stock's target on a day. *Shrink*: the average target w times "
        "w²/(w² + κs²), rescaled to unit gross exposure. *Widen*: κs added to the band width. κ = 0 is the ensemble as "
        "reported in TRADING_RULE_RESULTS.md and TRADING_STATE_RESULTS.md. Every path starts from cash on its period's "
        "first day and carries its book across years, at the paper's costs.",
        "",
        "## Decision (pre-registered: κ chosen on 1996–1997; kept if κ > 0 and its interval against κ = 0 lies above 0)",
        "",
        f"**Kept for {len(kept)} of {len(choice)} object–form pairs" + (": " + "; ".join(", ".join(k) for k in kept) if kept else "") + ".**",
        "",
        bt.md(choice_t.astype(object)),
        "The interval is a paired circular block bootstrap (21-day blocks, 2,000 draws) of the chosen κ against κ = 0 on "
        "1996–1997. It is biased upward, because κ was chosen on the same days.",
        "",
        "## The test years at the chosen κ (reported, never used to decide)",
        "",
        "Net SR by period; turnover a day on 1998–2021.",
        "",
        bt.md(test_t) if len(test_t) else "No object chose κ > 0.\n",
        "Against κ = 0 on the same days (paired bootstrap, 95% interval):",
        "",
        bt.md(contrast_t) if len(contrast_t) else "None.\n",
        "## The whole grid (a diagnostic)",
        "",
        bt.md(grid_t),
        "## How much the seeds disagree",
        "",
        "Median over member stock-days of s, against the median absolute average target.",
        "",
        bt.md(size_t),
    ]
    out = HERE / "PESSIMISM_RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(choice_t.to_string())
    print(test_t.round(2).to_string() if len(test_t) else "no kappa > 0 chosen")
    print(contrast_t.to_string() if len(contrast_t) else "")
    print(grid_t.round(2).to_string())
    print(size_t.round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
