"""λ_Var on D's hindsight choice at K = 30 (CLAUDE.md section 7, row of 2026-11-16) -> analysis/gap/LAMBDA_VAR.md.

    python analysis/gap/scan.py train   --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none entry_rule=history lambda_var=<v>
    python analysis/gap/scan.py weights --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none entry_rule=history lambda_var=<v>
    python analysis/gap/lambda_var.py                    # CPU, about two minutes

For v in 0, 10, 30, 300 and 1000 (about 3.5 minutes of GPU each); 100 is the stored run of D's reference.

D's reference (buffered list, rebuilt history, 5 bp training cost, ridge 1e-4, 30 epochs, K = 30, seeds 0-2) with only
λ_Var changed. In the objective, net Sharpe (annualized) + λ_Var × explained variance, the second term reaches only
W^K and Q: λ_Var sets what the factors are for. Objects, built by selection.paths (each period one path from cash, the
paper's costs):
- model:            each seed on its own, statistics averaged over the seeds (the paper's object);
- ensemble:         the seeds' weights averaged stock by stock and rescaled;
- ensemble + band:  the ensemble through a no-trade band of 1.2e-3, hindsight's width at K = 30 (SCAN_BUFFER_GRID.md).
Paired tests: circular block bootstrap (paired_test.py), each value against λ_Var = 100 on the same days; the model's
series is the seeds' pooled daily return. A description, not a selection: the band width was read off the test years,
and Table 4 fixes λ_Var = 100.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection as sel  # noqa: E402
from deploy import seed_weights  # noqa: E402
from paired_test import cell, paired_bootstrap  # noqa: E402
from scan import Setting  # noqa: E402
from variants import stats  # noqa: E402

from afsa.metrics import COST_SHORT, COST_TRADE  # noqa: E402

OUT = Path(__file__).resolve().parent / "LAMBDA_VAR.md"
K = 30
BASE = Setting(5e-4, 1e-4, 30)
D = {"universe": "wrds_broad_buffer600", "fill": "none", "entry_rule": "history"}
VALUES = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
REF = 100.0
DELTA = 1.2e-3
OBJECTS = {"model": None, "ensemble": "band 0", f"ensemble + band {DELTA:g}": f"band {DELTA:g}"}
SPANS = {"validation 1996–97": (1996, 1997), "1998–2021": (1998, 2021), "1998–2005": (1998, 2005), "2006–13": (2006, 2013),
         "2014–21": (2014, 2021), "2022–25": (2022, 2025), "1998–2025": (1998, 2025)}
TESTED = ("validation 1996–97", "1998–2021", "2022–25")


def extra(v: float) -> dict:
    return {**D, "lambda_var": v}


def label(v: float) -> str:
    return f"λ_Var = {v:g}" + (" (Table 4)" if v == REF else "")


def frames(v: float) -> dict[str, list[pd.DataFrame]]:
    """Each object's daily frames (one per seed for the model, one for a layer): the validation path (1996-1997) and
    the test path (1998-2025), each from cash, joined."""
    parts = []
    for validation in (True, False):
        p = sel.paths(BASE, K, validation, extra(v))
        if p is None:
            raise SystemExit(f"λ_Var {v:g}: run or weights missing ({'validation' if validation else 'test'} years)")
        parts.append(p)
    if parts[1].index[-1].year != 2025:
        raise SystemExit(f"λ_Var {v:g}: the test path stops in {parts[1].index[-1].year}")
    out = {}
    for obj, col in OBJECTS.items():
        names = sel.seed_names(parts[1]) if col is None else [col]
        out[obj] = [pd.concat([sel.frame(p, n) for p in parts]) for n in names]
    return out


def span(f: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    return f[(f.index.year >= a) & (f.index.year <= b)]


def net_sr(frs: list[pd.DataFrame], a: int, b: int) -> float:
    """Net Sharpe ratio over a..b, averaged over the frames (the seeds, for the model)."""
    return float(np.mean([stats(span(f, a, b))["SR_net"] for f in frs]))


def pooled_net(frs: list[pd.DataFrame]) -> pd.Series:
    p = sel.pooled(frs)
    return p["net"]


def ev_rows(v: float) -> dict[str, float]:
    """Explained variance out of sample (mean over seeds and years), and the last training epoch's statistics."""
    out, fit = {}, []
    for train_years, (name, (a, b)) in ((6, ("validation 1996–97", SPANS["validation 1996–97"])), (8, ("1998–2021", SPANS["1998–2021"])),
                                        (8, ("2022–25", SPANS["2022–25"]))):
        run = BASE.run_dir(K, train_years, extra(v))
        ev = []
        for seed_dir in sorted(run.glob("seed*")):
            for y in range(a, b + 1):
                z = np.load(seed_dir / f"{y}.npz")
                ev.append(float(z["explained_var"]))
                if name == "1998–2021":
                    fit.append(dict(zip([str(c) for c in z["history_columns"]], z["history"][-1])))
        out[f"EV {name}"] = float(np.mean(ev))
    f = pd.DataFrame(fit).mean()
    out.update({"fit EV": float(f["explained_var"]), "fit SR net": float(f["sharpe_net"]), "fit SR gross": float(f["sharpe_gross"]),
                "fit turnover": float(f["turnover"]), "λ_Var × fit EV": v * float(f["explained_var"])})
    return out


def agreement(v: float, a: int = 1998, b: int = 2021) -> dict[str, float]:
    """How alike the seeds' books are (deploy.seed_agreement's measures) over a..b: mean correlation between two seeds
    of the day's weights and of the day's trades, and the gross exposure of the average book before rescaling."""
    run = BASE.run_dir(K, 8, extra(v))
    dates, permnos, R, _ = sel._market(a, b, D["universe"], D["fill"])
    Ws = [seed_weights(d, dates, permnos) for d in sorted(run.glob("seed*"))]

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        return float(((x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))).mean())

    pairs = [(i, j) for i in range(len(Ws)) for j in range(i + 1, len(Ws))]
    trades = [np.diff(W, axis=0) for W in Ws]
    out = {"weights": np.mean([corr(Ws[i], Ws[j]) for i, j in pairs]), "trades": np.mean([corr(trades[i], trades[j]) for i, j in pairs]),
           "gross exposure of the average book": float(np.abs(sum(Ws) / len(Ws)).sum(1).mean())}
    return {k: float(x) for k, x in out.items()}


def main() -> None:
    market = sel._market_row(D["universe"], D["fill"])
    data = {v: frames(v) for v in VALUES}
    lines = [
        "# λ_Var on D's hindsight choice (K = 30)",
        "",
        "Generated by `analysis/gap/lambda_var.py` (CLAUDE.md §7, row of 2026-11-16). The reading is in `REPORT.md` §10.9.",
        "",
        "- D's reference: buffered list, rebuilt history, 5 bp training cost, ridge 1e-4, 30 epochs, K = 30, seeds 0–2. Only λ_Var changes; "
        "100 is Table 4's value and the stored run. The objective is net Sharpe (annualized, 126-day batches) + λ_Var × explained variance, "
        "and the second term reaches only W^K and Q.",
        f"- Objects: *model* (each seed on its own, statistics averaged over seeds), *ensemble* (the seeds' weights averaged), and the ensemble "
        f"through a no-trade band of {DELTA:g}, hindsight's width at K = 30 (`SCAN_BUFFER_GRID.md`). Each period is one path from cash: "
        "1996–1997 (6-year windows) and 1998–2025. The paper's costs (5 bp per unit of turnover, 1 bp per unit short).",
        "- Δ: the paired difference of net Sharpe ratios against λ_Var = 100 with its 95% interval (circular 21-day blocks, 2,000 draws); "
        "for the model, the seeds' pooled daily return.",
        "- A description, not a selection: the band width was read off the test years, and Table 4 fixes λ_Var = 100.",
    ]

    lines += ["", "## Net Sharpe ratio by period", ""]
    for obj in OBJECTS:
        t = pd.DataFrame({label(v): {s: net_sr(data[v][obj], a, b) for s, (a, b) in SPANS.items()} for v in VALUES}).T
        lines += [f"**{obj}**" + (" (seed average)" if obj == "model" else ""), "", sel.md(t, "λ_Var"), ""]

    lines += ["## Against λ_Var = 100", ""]
    for obj in OBJECTS:
        ref = pooled_net(data[REF][obj])
        t = {}
        for v in VALUES:
            if v == REF:
                continue
            x = pooled_net(data[v][obj])
            t[label(v)] = {s: cell(paired_bootstrap(span(x.to_frame(), *SPANS[s])["net"].to_numpy(), span(ref.to_frame(), *SPANS[s])["net"].to_numpy()))
                           for s in TESTED}
        lines += [f"**{obj}**", "", sel.md(pd.DataFrame(t).T, "λ_Var"), ""]

    lines += ["## 1998–2021 in detail", "", "μ and σ in % a year, turnover and short exposure per day; beta against the buffered list's "
              "equal-weighted excess return; *market removed*: net SR after taking out beta × market (a diagnostic, not a hedge).", ""]
    keep = {"SR": "SR", "mu": "μ", "sigma": "σ", "SR_net": "SR net", "mu_net": "μ net", "turnover": "Turnover", "short": "Short",
            "beta": "Beta", "beta 2014-21": "Beta 2014–21", "SR_net ex market": "SR net, market removed", "turnover 2022-25": "Turnover 2022–25"}
    for obj in OBJECTS:
        t = pd.DataFrame({label(v): sel.describe(data[v][obj], market) for v in VALUES}).T[list(keep)].rename(columns=keep)
        lines += [f"**{obj}**", "", sel.md(t, "λ_Var"), ""]

    lines += ["## Explained variance, fit and the seeds", "",
              "*EV*: the objective's explained variance, (1/N) Σ (1 − Var(ε_i)/Var(R_i)), out of sample, averaged over seeds and test years. "
              "*Fit*: the last training epoch of the 1998–2021 models (mean over their batches; SR annualized). *Seeds*: mean correlation "
              "between two seeds of the day's weights and of the day's trades, 1998–2021, and the gross exposure of their average book "
              "before rescaling (1 = identical books).", ""]
    ev = pd.DataFrame({label(v): {**ev_rows(v), **{f"seeds: {k}": x for k, x in agreement(v).items()}} for v in VALUES}).T
    for c in ev.columns:
        if "EV" in c or "seeds" in c:
            ev[c] = ev[c].map(lambda x: f"{x:.3f}")
    lines += [sel.md(ev, "λ_Var"), ""]

    obj = f"ensemble + band {DELTA:g}"
    years = range(1996, 2026)
    t = pd.DataFrame({label(v): {y: net_sr(data[v][obj], y, y) for y in years} for v in VALUES})
    lines += [f"## Net Sharpe ratio by year, {obj}", "", "1996–1997 from the validation path, the rest from the 1998–2025 path.", "",
              sel.md(t, "Year"), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
