"""A firmer test of SM, the learned trading rule with views of the book and of the market, and of pessimism, on fresh
cells (CLAUDE.md section 7, row of 2026-11-16 "a firmer test of SM").

    python analysis/rl/sm_firm.py                     -> analysis/rl/SM_FIRM_RESULTS.md (CPU, about 10 minutes)
    python analysis/rl/sm_firm.py --validation-only   -> the pre-registered decision only (1996-1997)

SM is `--trade_head true --trade_state true --trade_market true` on dataset D (buffered list, rebuilt history; 5 bp,
ridge 1e-4, 30 epochs). Fresh cells, none run before the row was written:
  K = 8              SM seeds 0-2, against D's runs at K = 8 (seeds 0-2)
  K = 30, seeds 3-5  SM and D, both trained for this test (folders tagged seeds3-5)
  K = 100            SM seeds 0-2, against D's runs at K = 100 (seeds 0-2)
The first SM cell (K = 30, seeds 0-2; TRADING_STATE_RESULTS.md) is shown for reference only, and at K = 30 the
6-seed ensembles (seeds 0-5) are reported as well.

Objects in each cell: SM's model (one path per seed, statistics averaged) and its seed ensemble (TRADING_STATE.md),
against D's model and D's seed ensemble with the band added after training (delta = 4e-4, movers-only; the width
D's ensemble chooses on 1996-1997 at K = 8, 30 and 100). Every path starts from cash on its period's first day and
carries its book across years, at the paper's costs.

The pre-registered decision pools the three fresh cells on 1996-1997: the cells' daily net series are averaged (a
third of the capital in each cell) and SM is compared with D + band after by one paired block bootstrap, for the
model and for the ensemble. Pessimism (pessimism.py's two forms and grids) is applied to each cell's two ensembles and
to the 6-seed ensembles, with kappa chosen on 1996-1997. The test years are reported and never decide.
"""
from __future__ import annotations

import argparse
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
from deploy import band_path, ensemble, market, seed_weights, state_path, trade  # noqa: E402
from pessimism import GRID, shrink  # noqa: E402
from trading_rule import DELTA, interval  # noqa: E402
from trading_state import check_state, view_weights  # noqa: E402
from variants import stats  # noqa: E402

from afsa.config import RESULTS_DIR, Config  # noqa: E402
from afsa.metrics import PAPER_TABLE2, stitch  # noqa: E402

SM = {"trade_head": True, "trade_state": True, "trade_market": True}
FRESH = {"K = 8": (8, (0, 1, 2), ""), "K = 30, seeds 3–5": (30, (3, 4, 5), "seeds3-5"), "K = 100": (100, (0, 1, 2), "")}
FIRST = {"K = 30, seeds 0–2": (30, (0, 1, 2), "")}                       # TRADING_STATE_RESULTS.md: reference only


def folder(cfg: Config, tag: str) -> Path:
    return RESULTS_DIR / "runs" / (cfg.name() + (f"_{tag}" if tag else ""))


def targets(seed_dir: Path, dates: np.ndarray, permnos: np.ndarray) -> np.ndarray:
    """A seed's daily targets: from reeval_weights.py's <year>_w.npz if there, else from a run saved with --save-weights."""
    if any(seed_dir.glob("[0-9][0-9][0-9][0-9]_w.npz")):
        return seed_weights(seed_dir, dates, permnos)
    return bt.saved(seed_dir, dates, permnos, "weights")


def cell(K: int, seeds: tuple[int, ...], tag: str, first: int, last: int, train_years: int) -> dict:
    """The seeds' inputs of SM and of D in one cell and period, checked against the stored results."""
    base = Config(universe="wrds_broad_buffer600", fill="none", entry_rule="history", ridge=1e-4, K=K, train_years=train_years)
    sm = dataclasses.replace(base, **SM)
    const = {"band0": sm.head_band0, "band_max": sm.head_band_max, "eta0": sm.head_eta0}
    dates, permnos, R, member = market(base, first, last)
    out = {"dates": dates, "R": R, "member": member, "const": const, "D": [], "SM": [], "check": (0.0, 0, 0)}
    for s in seeds:
        d = folder(base, tag) / f"seed{s}"
        W = targets(d, dates, permnos)
        bt.check(stitch(d), trade(W, R, dates), d)                           # the targets reproduce the stored results
        out["D"].append(W)
        d = folder(sm, tag) / f"seed{s}"
        W, A, Kl, G, B = (bt.saved(d, dates, permnos, k) for k in ("weights", "dial_a", "dial_k", "signal", "book"))
        bt.check(stitch(d), trade(B, R, dates), d)                           # the stored book reproduces the stored results
        x = {"dir": d, "W": W, "A": A, "K": Kl, "has": G != 0, "V": view_weights(d, dates), "book": B}
        worst, ties = check_state(x, dates, member, const)                   # the numpy rule reproduces the model's books
        out["check"] = (max(out["check"][0], worst), out["check"][1] + ties, out["check"][2] + len(dates))
        out["SM"].append({"W": W, "rules": [(A, Kl, x["V"])], "has": x["has"]})
    if any((x["has"] != out["SM"][0]["has"]).any() for x in out["SM"]):
        raise ValueError(f"K = {K}, seeds {seeds}: the seeds disagree on who has a signal")
    return out


def join(a: dict, b: dict) -> dict:
    """Two cells on the same days (K = 30, seeds 0-2 and 3-5) as one with six seeds."""
    return {**a, "D": a["D"] + b["D"], "SM": a["SM"] + b["SM"]}


def sm_ensemble(c: dict, W: np.ndarray, widen: np.ndarray | None = None) -> np.ndarray:
    """SM's ensemble trading of the targets W: each day each seed's dials set on the ensemble's book, then averaged."""
    return state_path(W, c["member"], [r for x in c["SM"] for r in x["rules"]], c["SM"][0]["has"], widen=widen, **c["const"])[0]


def objects(c: dict) -> dict[str, list[pd.DataFrame]]:
    """SM's model and ensemble, and D's model and ensemble with the band added after training, as daily frames."""
    R, dates, member = c["R"], c["dates"], c["member"]
    sm = [trade(state_path(x["W"], member, x["rules"], x["has"], **c["const"])[0], R, dates) for x in c["SM"]]
    return {"SM": sm,
            "SM ensemble": [trade(sm_ensemble(c, ensemble([x["W"] for x in c["SM"]])), R, dates)],
            "D + band after": [trade(band_path(W, member, DELTA, rescale="moved"), R, dates) for W in c["D"]],
            "D ensemble + band after": [trade(band_path(ensemble(c["D"]), member, DELTA, rescale="moved"), R, dates)],
            "D model": [trade(W, R, dates) for W in c["D"]],                 # the model as the paper trades it
            "D ensemble": [trade(ensemble(c["D"]), R, dates)]}


DEPLOY = (("D model", "D model (the paper's object)"), ("D ensemble", "D ensemble"),
          ("D ensemble + band after", "D ensemble + band after"), ("SM", "SM model"),
          ("SM ensemble", "SM ensemble (the deployment variant)"))


def deployment_table(lev: pd.DataFrame) -> pd.DataFrame:
    """One block per K: the paper's model, D's objects and SM's, on validation, 1998-2021 and 2022-2025."""
    rows = {}
    for c in ("K = 8", "K = 30, seeds 0–2", "K = 30, seeds 3–5", "K = 30, seeds 0–5", "K = 100"):
        first = lev.loc[f"{c}: SM"]
        rows[f"{c}: paper, Table 2 (model)"] = {"Validation 1996–97": np.nan, "1998–2021": first["Paper SR_net"],
                                                "2022–25": np.nan, "Turnover": np.nan}
        for key, label in DEPLOY:
            r = lev.loc[f"{c}: {key}"]
            rows[f"{c}: {label}"] = {"Validation 1996–97": r["Validation SR_net"], "1998–2021": r["SR_net"],
                                     "2022–25": r["SR_net 2022–25"], "Turnover": r["Turnover"]}
    return pd.DataFrame(rows).T


def pessimism(c: dict) -> dict[tuple[str, str, float], pd.DataFrame]:
    """(object, form, kappa) -> daily frame, for the cell's two ensembles (pessimism.py's forms and grids)."""
    R, dates, member, out = c["R"], c["dates"], c["member"], {}
    for name, Ws in (("SM ensemble", [x["W"] for x in c["SM"]]), ("D ensemble + band after", c["D"])):
        s = np.std(np.stack(Ws), axis=0)
        for form, grid in GRID.items():
            for kappa in grid:
                if kappa == 0 and form == "widen":
                    out[(name, form, kappa)] = out[(name, "shrink", 0.0)]
                    continue
                W = shrink(Ws, kappa if form == "shrink" else 0.0)
                widen = kappa * s if form == "widen" else None
                H = (sm_ensemble(c, W, widen) if name == "SM ensemble" else
                     band_path(W, member, DELTA + (0.0 if widen is None else widen), rescale="moved"))
                out[(name, form, kappa)] = trade(H, R, dates)
    return out


def verdict(pairs: list[tuple[pd.Series, pd.Series]]) -> tuple[str, list[tuple[float, float, float]]]:
    """The first test's rule: better if an interval lies above 0 and none below, worse the other way round."""
    cells = [interval(x, y) for x, y in pairs]
    signs = [1 if lo > 0 else -1 if hi < 0 else 0 for _, lo, hi in cells]
    v = "better" if max(signs) == 1 and min(signs) >= 0 else "worse" if min(signs) == -1 and max(signs) <= 0 else "inconclusive"
    return v, cells


def fmt(d: tuple[float, float, float]) -> str:
    return f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"


def decide(val: dict[str, dict]) -> tuple[list[str], pd.DataFrame]:
    """The pre-registered decision, pooled over the fresh cells, and each cell's own verdict (1996-1997)."""
    rows, lines = {}, []
    pooled = {k: sum(bt.pooled(val[c][k]) for c in FRESH) / len(FRESH) for k in val[next(iter(FRESH))]}
    for label, src in [("pooled", pooled)] + [(c, {k: bt.pooled(v) for k, v in val[c].items()}) for c in FRESH]:
        v, (m, e) = verdict([(src["SM"], src["D + band after"]), (src["SM ensemble"], src["D ensemble + band after"])])
        rows[label] = {"verdict": v, "SM − D + band after": fmt(m), "SM ensemble − D ensemble + band after": fmt(e)}
        if label == "pooled":
            lines = [f"**{v.capitalize()}.** SM − D + band after: {fmt(m)}; SM ensemble − D ensemble + band after: {fmt(e)}."]
    return lines, pd.DataFrame(rows).T


def levels(test: dict[str, dict], val: dict[str, dict]) -> pd.DataFrame:
    rows = {}
    for c in test:
        K = int(c.split("=")[1].split(",")[0])
        for name, frames in test[c].items():
            m = lambda s, k: float(np.mean([stats(bt.span(f, s))[k] for f in frames]))   # noqa: E731
            rows[f"{c}: {name}"] = {"Validation SR_net": float(np.mean([stats(f)["SR_net"] for f in val[c][name]])),
                                    "SR": m(bt.TEST, "SR"), "SR_net": m(bt.TEST, "SR_net"), "σ": m(bt.TEST, "sigma"),
                                    "Turnover": m(bt.TEST, "turnover"),
                                    **{f"SR_net {n}": m(s, "SR_net") for n, s in bt.SPANS[1:]},
                                    "Paper SR_net": PAPER_TABLE2[K][3] if K in PAPER_TABLE2 else np.nan}   # (SR, μ, σ, SR_net, ...)
    return pd.DataFrame(rows).T


def contrasts(test: dict[str, dict], val: dict[str, dict]) -> pd.DataFrame:
    """SM against D + band after in each cell and pooled over the fresh cells, every period (reported)."""
    rows = {}
    pairs = (("SM", "D + band after"), ("SM ensemble", "D ensemble + band after"))
    for label in [*FRESH, "pooled", *[c for c in test if c not in FRESH]]:
        for a, b in pairs:
            cells = {}
            for name, s in (("Validation 1996–97", None),) + bt.SPANS:
                def series(k, src):
                    if label == "pooled":
                        return sum(bt.pooled(src[c][k]) for c in FRESH) / len(FRESH)
                    return bt.pooled(src[label][k])
                x, y = (series(k, val) if s is None else bt.span(series(k, test), s) for k in (a, b))
                cells[name] = fmt(interval(x, y))
            rows[f"{label}: {a} − {b}"] = cells
    return pd.DataFrame(rows).T


def pessimism_tables(pv: dict[str, dict], pt: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    choice, grid = {}, {}
    for c in pv:
        for name in ("SM ensemble", "D ensemble + band after"):
            for form, ks in GRID.items():
                sr = lambda f, s=None: stats(f if s is None else bt.span(f, s))["SR_net"]   # noqa: E731
                best = max(ks, key=lambda k: (round(sr(pv[c][(name, form, k)]), 12), -k))
                d = interval(pv[c][(name, form, best)]["net"], pv[c][(name, form, 0.0)]["net"])
                t = interval(bt.span(pt[c][(name, form, best)], bt.TEST)["net"], bt.span(pt[c][(name, form, 0.0)], bt.TEST)["net"])
                h = interval(bt.span(pt[c][(name, form, best)], bt.HOLDOUT)["net"], bt.span(pt[c][(name, form, 0.0)], bt.HOLDOUT)["net"])
                choice[f"{c}: {name}, {form}"] = {"κ chosen": f"{best:g}", "validation, chosen − κ = 0": "" if best == 0 else fmt(d),
                                                  "kept": "yes" if best > 0 and d[1] > 0 else "no",
                                                  "1998–2021, chosen − κ = 0": "" if best == 0 else fmt(t),
                                                  "2022–25, chosen − κ = 0": "" if best == 0 else fmt(h)}
                for k in ks:
                    grid[f"{c}: {name}, {form}, {k:g}"] = {"validation": sr(pv[c][(name, form, k)]),
                                                           "1998–2021": sr(pt[c][(name, form, k)], bt.TEST),
                                                           "2022–25": sr(pt[c][(name, form, k)], bt.HOLDOUT)}
    return pd.DataFrame(choice).T, pd.DataFrame(grid).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation-only", action="store_true", help="only the pre-registered decision, on 1996-1997")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_rows", 300)
    all_cells = {**FRESH, **FIRST}
    raw_v = {c: cell(*spec, *bt.VALIDATION, 6) for c, spec in all_cells.items()}
    val = {c: objects(raw_v[c]) for c in FRESH}
    lines, decided = decide(val)
    print(*lines, decided.to_string(), sep="\n")
    print("numpy rule against the stored books (largest daily L1 gap, tie days, seed-days):", {c: raw_v[c]["check"] for c in all_cells})
    if args.validation_only:
        return
    val.update({c: objects(raw_v[c]) for c in FIRST})
    six_v = join(raw_v["K = 30, seeds 0–2"], raw_v["K = 30, seeds 3–5"])
    val["K = 30, seeds 0–5"] = objects(six_v)
    pv = {c: pessimism(raw_v[c]) for c in FRESH}
    pv["K = 30, seeds 0–5"] = pessimism(six_v)
    del raw_v, six_v
    test, pt, checks = {}, {}, {}
    for c, spec in all_cells.items():                                       # one cell at a time: the arrays are large
        raw = cell(*spec, bt.TEST[0], bt.HOLDOUT[1], 8)
        checks[c] = raw["check"]
        test[c] = objects(raw)
        if c in FRESH:
            pt[c] = pessimism(raw)
        if c == "K = 30, seeds 3–5":
            keep = raw
        if c == "K = 30, seeds 0–2":
            six = join(raw, keep)
            test["K = 30, seeds 0–5"] = objects(six)
            pt["K = 30, seeds 0–5"] = pessimism(six)
            del six, keep
        del raw
    lev, con = levels(test, val), contrasts(test, val)
    deploy_t = deployment_table(lev)
    choice, grid = pessimism_tables(pv, pt)
    kept = int((choice["kept"] == "yes").sum())
    out = HERE / "SM_FIRM_RESULTS.md"
    text = [
        "# A firmer test of SM, and of pessimism: results",
        "",
        "Written by `analysis/rl/sm_firm.py`, as pre-registered in CLAUDE.md §7 (row of 2026-11-16, \"a firmer test of SM\"). "
        "SM is the learned trading rule with views of the book and of the market ([TRADING_STATE.md](TRADING_STATE.md)) on "
        "dataset D (buffered list, rebuilt history; 5 bp, ridge 1e-4, 30 epochs). **Fresh cells**, none run before the row "
        "was written: K = 8 and K = 100 (seeds 0–2), and K = 30 with seeds 3–5. The first cell (K = 30, seeds 0–2, "
        "[TRADING_STATE_RESULTS.md](TRADING_STATE_RESULTS.md)) is shown for reference only. The comparator is D's model and "
        "D's seed ensemble with the band added after training (δ = 4e-4, the width D's ensemble chooses on 1996–1997 at "
        "every K; movers-only). Every path starts from cash on its period's first day and carries its book across years, "
        "at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).",
        "",
        "Checks: the targets and books reproduce each run's stored results, and SM's numpy rule, applied one day at a time "
        "to the stored book of the day before, gives the stored book (largest daily L1 gap and float32 tie days, out of "
        "seed-days): " + "; ".join(f"{c} {v[0]:.1e}, {v[1]} of {v[2]:,}" for c, v in checks.items()) + ".",
        "",
        "## Decision (pre-registered: 1996–1997 only, the fresh cells pooled)",
        "",
        *lines,
        "",
        "The pooled series hold a third of the capital in each fresh cell. Paired circular block bootstrap (21-day blocks, "
        "2,000 draws, the same days for both), 95% interval; for the models, each cell's series is its seed mean. Each "
        "cell's own verdict, by the same rule:",
        "",
        bt.md(decided.astype(object)),
        "## Levels",
        "",
        "Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; σ in % a year. "
        "*Paper SR_net*: Table 2's model at that K (1998–2021), which is a seed average, not an ensemble.",
        "",
        bt.md(lev),
        "## The deployment table (DEPLOYMENT.md)",
        "",
        "Net SR on 1996–1997, 1998–2021 and 2022–2025, and turnover a day on 1998–2021. The paper's row is Table 2's "
        "model; every other row is ours, from cash with books carried across years. Not Table 2 rows: an ensemble or a "
        "trading rule changes what is traded, and the paper reports the model alone.",
        "",
        bt.md(deploy_t).replace("nan", "–"),
        "## Contrasts: SM against D + band after (reported, never used to decide except the decision above)",
        "",
        bt.md(con),
        "## Pessimism on the fresh cells (pessimism.py's forms and grids; κ chosen on 1996–1997)",
        "",
        f"**Kept for {kept} of {len(choice)} ensemble–form pairs.** Kept means a chosen κ > 0 whose interval against κ = 0 "
        "on 1996–1997 lies above 0 (biased upward, since κ was chosen on the same days). At K = 30, seeds 0–5, s comes from "
        "six seeds.",
        "",
        bt.md(choice.astype(object)),
        "### The whole grid (net SR; a diagnostic)",
        "",
        bt.md(grid),
    ]
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(lev.round(2).to_string())
    print(con.to_string())
    print(choice.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
