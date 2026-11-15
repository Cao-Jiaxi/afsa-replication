"""Writes analysis/gap/RESULTS.md: the runs of the gap analysis that are on disk, as tables.

    python analysis/gap/results.py

REPORT.md discusses these numbers. Nothing is trained here: run analysis/gap/variants.py first.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy  # noqa: E402
import exposure  # noqa: E402
import paired_test  # noqa: E402
from entrants import shares  # noqa: E402
from fig2_digitize import OUT as FIG2  # noqa: E402
from fig2_digitize import block_growth  # noqa: E402
from variants import PERIODS, VARIANTS, config, row, run_dir  # noqa: E402
from yearly import BLOCKS, ours  # noqa: E402

from afsa.config import Config  # noqa: E402
from afsa.metrics import PAPER_TABLE2  # noqa: E402

WHAT = {
    "baseline": "the configuration of CLAUDE.md, section 7",
    "epochs03": "3 epochs instead of 30", "epochs05": "5 epochs", "epochs08": "8 epochs", "epochs10": "10 epochs", "epochs15": "15 epochs",
    "epochs20": "20 epochs", "epochs60": "60 epochs",
    "batch035": "batches of 35 days (58 steps an epoch)", "batch049": "batches of 49 days (41 steps)", "batch063": "batches of 63 days (32 steps)",
    "batch252": "batches of 252 days (8 steps an epoch)", "batch252_e60": "batches of 252 days, 60 epochs (480 steps, as the baseline)", "cosine": "learning rate decays to zero (cosine)", "lr001": "learning rate 0.001",
    "daily_sharpe": "Sharpe term not annualized, as printed (explained variance weighs 15.9 times more)",
    "lambda_var0": "no explained-variance term", "lambda_var10": "lambda_Var = 10", "lambda_var1000": "lambda_Var = 1000",
    "kernel002": "kernel initialized at 0.02 of the paper's scale (the public LongConv code)", "dropout0": "no dropout", "dropout03": "dropout 0.3",
    "cumulative": "LongConv sees the running sum of the 30 residuals", "rf_hurdle": "risk-free rate subtracted from the portfolio return in the objective",
    "w_init3": "W^K initialized 3 times larger", "w_init10": "W^K initialized 10 times larger", "q_init3": "Q initialized 3 times larger",
    "ridge1e6": "ridge 1e-6 instead of 1e-5", "ridge1e4": "ridge 1e-4", "ridge1e3": "ridge 1e-3",
    "entry_member": "a signal for every member, also in its first 30 days in the universe",
    "ridge1e4_cost10": "ridge 1e-4 and trained with 10 bp", "ridge1e4_cost075": "ridge 1e-4 and trained with 7.5 bp",
    "no_daily": "factor weights without the 4 daily characteristics", "no_past_returns": "without the 8 past-return characteristics (Table 3)",
    "no_frictions": "without the 10 trading-friction characteristics (Table 3)",
    "common": "universe: US common stocks only (monthly top 500)",
    "window": "universe: 500 fixed per window, complete returns in training and test years",
    "window_train": "universe: 500 fixed per window, complete returns in the training years",
    "cost10": "trained with 10 bp per unit of turnover", "cost075": "trained with 7.5 bp",
    "cum_window": "cumulative input + window universe", "cum_window_train": "cumulative input + window_train universe",
    "cum_rf_hurdle": "cumulative input + risk-free rate in the objective", "cum_batch252": "cumulative input + batches of 252 days",
    "cum_kernel002": "cumulative input + kernel at 0.02", "cum_daily_sharpe": "cumulative input + Sharpe term not annualized",
    "cum_no_daily": "cumulative input, no daily characteristics", "cum_epochs60": "cumulative input, 60 epochs",
    "cum_common": "cumulative input + common-stock universe", "cum_no_past_returns": "cumulative input, no past-return characteristics",
    "cum_window_hurdle": "cumulative input + window universe + risk-free rate in the objective",
    "cum_w_init10": "cumulative input + W^K initialized 10 times larger", "cum_cosine": "cumulative input + cosine learning-rate decay",
}
TEST_COLS = ["SR", "mu", "sigma", "turnover", "cost/sigma", "SR_net", "SR_net 1998-05", "SR_net 2006-13", "SR_net 2014-21"]
HEAD = ("| Run | What changes | SR | μ | σ | Turnover | Cost / σ | SR net | SR net 1998–2005 | 2006–2013 | 2014–2021 | Fit SR net | "
        "Validation SR net | Validation turnover |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def fmt(x, digits=2) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def paper_line(K: int) -> str:
    sr, mu, sigma, sr_net, mu_net, _, _ = PAPER_TABLE2[K]
    cost_bp = (mu - mu_net) / 252 * 100
    return (f"| **paper, K = {K}** | Table 2 | {sr:.2f} | {mu:.2f} | {sigma:.2f} | {(cost_bp - 0.5) / 5:.2f} (implied) | {(mu - mu_net) / sigma:.2f} | "
            f"{sr_net:.2f} | – | – | – | – | – | – |")


def line(variant: str, K: int) -> str | None:
    first, last, n = PERIODS["test"]
    t = row(config(variant, K, n), first, last)
    if t is None:
        return None
    first, last, n = PERIODS["validation"]
    v = row(config(variant, K, n), first, last) or {}
    cells = [fmt(t.get(c)) for c in TEST_COLS] + [fmt(t.get("fit SR_net")), fmt(v.get("SR_net")), fmt(v.get("turnover"))]
    return f"| {variant} | {WHAT.get(variant, '')} | " + " | ".join(cells) + " |"


def table(variants: list[str], K: int) -> str:
    lines = [HEAD, paper_line(K)] if K in PAPER_TABLE2 else [HEAD]
    lines += [x for x in (line(v, K) for v in variants) if x]
    return "\n".join(lines)


def yearly_table(K: int, variants: list[str]) -> str:
    paper = block_growth(pd.read_csv(FIG2, index_col="day"))[f"K={K}"]
    cols = {"paper (Figure 2)": paper}
    for v in variants:
        s = ours(v, K)
        if s is not None:
            cols[v] = s
    t = pd.DataFrame(cols)
    out = ["| Years | " + " | ".join(t.columns) + " |", "| --- |" + " --- |" * t.shape[1]]
    for a, b in BLOCKS:
        out.append(f"| {a}–{b} | " + " | ".join(f"{x:.1f}" for x in t.loc[a:b].mean()) + " |")
    out.append("| correlation of the 24 yearly figures with the paper's | – | " + " | ".join(f"{t[c].corr(paper):.2f}" for c in t.columns[1:]) + " |")
    return "\n".join(out)


def md(t: pd.DataFrame, digits: int = 2, index: str = "") -> str:
    """A DataFrame as a markdown table; numbers with `digits` decimals, text as it is."""
    def one(x) -> str:
        return x if isinstance(x, str) else fmt(float(x), digits)

    names = list(t.index.names) if t.index.nlevels > 1 else [index or t.index.name or ""]
    lines = ["| " + " | ".join(names + [str(c) for c in t.columns]) + " |", "|" + " --- |" * (len(names) + t.shape[1])]
    for key, r in t.iterrows():
        lines.append("| " + " | ".join([str(k) for k in (key if isinstance(key, tuple) else (key,))] + [one(x) for x in r]) + " |")
    return "\n".join(lines)


def paired_table(K: int, extra: dict | None = None) -> str | None:
    """Every variant on disk against the baseline of the same K: test years, and the validation years beside them."""
    names = [v for v in VARIANTS if not set(VARIANTS[v]) & set(extra or {})]     # `extra` would override the variant's own change
    test = paired_test.table(names, K, "test", extra)
    if len(test) < 2:
        return None
    val = paired_test.table(names, K, "validation", extra)
    test = test.rename(columns={"difference [95% interval]": "Difference [95% interval]", "pooled SR net": "Pooled SR net", "turnover": "Turnover",
                                "1998-05": "1998–2005", "2006-13": "2006–2013", "2014-21": "2014–2021"})
    if len(val) > 1:
        test["Validation: pooled SR net"] = val["pooled SR net"]
        test["Validation: difference [95% interval]"] = val["difference [95% interval]"]
    return md(test.fillna("–"), index="Run")


def exposure_table(variants: list[str], K: int) -> str | None:
    first, last, n = PERIODS["test"]
    parts = {v: exposure.by_era(config(v, K, n), first, last) for v in variants}
    parts = {v: t for v, t in parts.items() if t is not None}
    if not parts:
        return None
    t = pd.concat(parts, names=["Run", "Years"])
    return md(t.rename(columns={"beta": "Beta", "net long": "Net long", "|net|": "Mean of the absolute net position", "turnover": "Turnover"}))


DEPLOY_COLS = {"SR": "SR", "mu": "μ", "sigma": "σ", "turnover": "Turnover", "cost/sigma": "Cost / σ", "SR_net": "SR net", "SR_net 1998-05": "SR net 1998–2005",
               "SR_net 2006-13": "2006–2013", "SR_net 2014-21": "2014–2021", "SR_net 2022-25": "SR net 2022–2025", "turnover 2022-25": "Turnover 2022–2025"}


def deploy_table(cfg: Config, full: bool) -> str | None:
    """The rows written by deploy.py for one run; without the grid of fixed band widths unless `full`."""
    f = run_dir(cfg) / "deploy.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f, index_col=[0, 1, 2], keep_default_na=False, na_values=[""])
    t.index = pd.MultiIndex.from_tuples([(a, b, "" if pd.isna(c) else str(c)) for a, b, c in t.index], names=["Layer", "Portfolio", "Band width δ"])
    if not full:
        t = t[t.index.get_level_values(0) != "band, fixed delta"]
    return md(t[[c for c in DEPLOY_COLS if c in t.columns]].rename(columns=DEPLOY_COLS))


WRDS = {"universe": "wrds_broad", "fill": "none"}


def wrds_table(K: int = 30) -> str | None:
    """Runs on the WRDS-built arrays (made by afsa.run to 2025): the paper's years, validation and 2022-2025."""
    lines = [("| Run | What changes | SR | μ | σ | Turnover | Cost / σ | SR net | SR net 1998–2005 | 2006–2013 | 2014–2021 | Validation SR net | "
              "Validation turnover | SR net 2022–2025 | Turnover 2022–2025 |"), "|" + " --- |" * 15, paper_line(K).replace("| – | – | – |", "| – | – | – | – |", 1)]
    for v in VARIANTS:
        t = None if set(VARIANTS[v]) & set(WRDS) else row(config(v, K, 8, WRDS), 1998, 2021)
        if t is None:
            continue
        val = row(config(v, K, 6, WRDS), 1996, 1997) or {}
        late = row(config(v, K, 8, WRDS), 2022, 2025) or {}
        cells = [fmt(t.get(c)) for c in TEST_COLS] + [fmt(val.get("SR_net")), fmt(val.get("turnover")), fmt(late.get("SR_net")), fmt(late.get("turnover"))]
        lines.append(f"| {v} | {WHAT.get(v, '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) if len(lines) > 3 else None


def steps_table(K: int = 8) -> str | None:
    """Batch length against the number of optimizer steps: the four cells of a 2 x 2."""
    cells = {("126 days", "240 steps"): "epochs15", ("252 days", "240 steps"): "batch252", ("126 days", "480 steps"): "baseline", ("252 days", "480 steps"): "batch252_e60"}
    rows = {}
    for (days, steps), v in cells.items():
        t = row(config(v, K, PERIODS["test"][2]), *PERIODS["test"][:2])
        if t is None:
            return None
        val = row(config(v, K, PERIODS["validation"][2]), *PERIODS["validation"][:2]) or {}
        rows[(days, steps, v)] = {"SR": t["SR"], "Turnover": t["turnover"], "SR net": t["SR_net"], "Fit SR net": t["fit SR_net"],
                                  "Validation SR net": val.get("SR_net", np.nan)}
    t = pd.DataFrame(rows).T
    t.index.names = ["Batch", "Steps per window", "Run"]
    return md(t)


def gram_table() -> str | None:
    rows = {}
    for K in (8, 30, 100):
        for v in ("ridge1e6", "baseline", "ridge1e4", "ridge1e3"):
            cfg = config(v, K, 8)
            f = run_dir(cfg) / "gram.csv"
            if f.exists():
                m = pd.read_csv(f, index_col=0).mean()
                t = row(cfg, 1998, 2021) or {}
                rows[(K, f"{cfg.ridge:g}")] = {"Median eigenvalue": f"{m['median eigenvalue']:.1e}", "Smallest eigenvalue": f"{m['smallest eigenvalue']:.1e}",
                                               "Directions hedged < 90%": m["directions hedged < 90%"], "Directions hedged < 50%": m["directions hedged < 50%"],
                                               "Mean hedged share": m["mean hedged share"], "σ": t.get("sigma", np.nan), "SR": t.get("SR", np.nan),
                                               "SR net": t.get("SR_net", np.nan)}
    if not rows:
        return None
    t = pd.DataFrame(rows).T
    t.index.names = ["K", "Ridge"]
    return md(t)


GAP_RUNS = ("baseline", "ridge1e4", "ridge1e3", "entry_member", "rf_hurdle", "cumulative", "cost075", "cost10")
GAP_LAYERS = (("model", "pooled", "pooled"), ("ensemble", "ensemble", "ensemble of the seeds"),
              ("band, delta from validation", "ensemble", "ensemble + band, width from validation"),
              ("band, walk-forward delta", "per seed", "band, walk-forward width"),
              ("band, walk-forward delta", "ensemble", "ensemble + band, walk-forward width"))


def gap_table(K: int) -> str | None:
    """The paper's row, each candidate change of the model, and the deployment layers on some of them."""
    if K not in PAPER_TABLE2 or row(config("baseline", K, 8), 1998, 2021) is None:
        return None
    sr, mu, sigma, sr_net, mu_net, _, _ = PAPER_TABLE2[K]
    rows = {("paper, Table 2", ""): {"SR": sr, "mu": mu, "sigma": sigma, "turnover": ((mu - mu_net) / 252 * 100 - 0.5) / 5,
                                     "cost/sigma": (mu - mu_net) / sigma, "SR_net": sr_net}}
    for v in GAP_RUNS:
        cfg = config(v, K, 8)
        t = row(cfg, 1998, 2021)
        if t is None:
            continue
        val = row(config(v, K, 6), 1996, 1997) or {}
        rows[(v, "per seed (the model)")] = {**t, "validation": val.get("SR_net", np.nan)}
        f = run_dir(cfg) / "deploy.csv"
        if f.exists():
            d = pd.read_csv(f, index_col=[0, 1, 2])
            for layer, obj, label in GAP_LAYERS:
                hit = d[(d.index.get_level_values(0) == layer) & (d.index.get_level_values(1) == obj)]
                if len(hit):
                    rows[(v, label)] = hit.iloc[0].to_dict()
    t = pd.DataFrame(rows).T
    t["gap"] = sr_net - t["SR_net"]
    cols = {"SR": "SR", "mu": "μ", "sigma": "σ", "turnover": "Turnover", "cost/sigma": "Cost / σ", "SR_net": "SR net", "gap": "Paper's SR net minus ours",
            "validation": "Validation SR net"}
    t = t[[c for c in cols if c in t.columns]].rename(columns=cols)
    t.index.names = ["Run", "Portfolio"]
    return md(t)


BY_K = (("baseline", None, "Baseline"), ("ridge1e4", None, "Ridge 1e-4"), ("cost075", None, "Trained at 7.5 bp"), ("ridge1e4_cost075", None, "Both"),
        ("baseline", ("ensemble", "ensemble"), "Baseline: ensemble"),
        ("baseline", ("band, delta from validation", "ensemble"), "Baseline: ensemble + band"),
        ("ridge1e4_cost075", ("ensemble", "ensemble"), "Both: ensemble"),
        ("ridge1e4_cost075", ("band, delta from validation", "ensemble"), "Both: ensemble + band"))
BY_K_METRICS = (("SR_net", "Net Sharpe ratio"), ("turnover", "Turnover a day"), ("SR", "Sharpe ratio before costs"), ("mu", "Return before costs, % a year"),
                ("sigma", "Volatility, % a year"))


def by_k_tables() -> list[str]:
    """One table per metric, one row per K: the paper, the models (seed averages) and two deployment layers."""
    cells: dict[str, dict] = {m: {} for m, _ in BY_K_METRICS}
    for K in PAPER_TABLE2:
        sr, mu, sigma, sr_net, mu_net, _, _ = PAPER_TABLE2[K]
        paper = {"SR": sr, "mu": mu, "sigma": sigma, "SR_net": sr_net, "turnover": ((mu - mu_net) / 252 * 100 - 0.5) / 5}
        for m, _ in BY_K_METRICS:
            cells[m][K] = {"Paper": paper[m]}
        for v, layer, label in BY_K:
            cfg = config(v, K, 8)
            if layer is None:
                r = row(cfg, 1998, 2021)
            else:
                f = run_dir(cfg) / "deploy.csv"
                d = pd.read_csv(f, index_col=[0, 1, 2]) if f.exists() else None
                hit = None if d is None else d[(d.index.get_level_values(0) == layer[0]) & (d.index.get_level_values(1) == layer[1])]
                r = None if hit is None or not len(hit) else hit.iloc[0].to_dict()
            for m, _ in BY_K_METRICS:
                cells[m][K][label] = np.nan if r is None else r[m]
    out = []
    for m, title in BY_K_METRICS:
        t = pd.DataFrame(cells[m]).T
        if t.drop(columns="Paper").notna().sum().sum() > len(t):             # more than the baselines
            out += [f"**{title}**", md(t, index="K")]
    return out


def paired_by_k() -> str | None:
    """One row per K: each candidate's paired difference from the baseline, test years and (in brackets) validation years."""
    names = [v for v, layer, _ in BY_K if layer is None and v != "baseline"]
    labels = {v: label for v, layer, label in BY_K if layer is None}
    rows = {}
    for K in PAPER_TABLE2:
        test, val = (paired_test.table(names, K, p) for p in ("test", "validation"))
        if test.empty:
            continue
        rows[K] = {labels[v]: (test.loc[v, "difference [95% interval]"] if v in test.index else "")
                   + (f" ({val.loc[v, 'difference [95% interval]']})" if not val.empty and v in val.index else "") for v in names}
    return md(pd.DataFrame(rows).T, index="K") if rows else None


def review_sections() -> list[str]:
    """Sections added after the review of 2026-11-13 (REPORT.md, section 9)."""
    parts = []
    tables = by_k_tables()
    if tables:
        parts += ["## Table 2 by number of factors: the paper, the candidate changes and the deployment layers",
                  "1998–2021 on `broad`. *Baseline*, *Ridge 1e-4*, *Trained at 7.5 bp* and *Both* are models (seed averages, the paper's object). "
                  "*Ensemble* trades the average of the three seeds' weights; *+ band* adds a no-trade band whose width is chosen on 1996–1997. "
                  "The paper's turnover is implied by its costs."] + tables
    t = paired_by_k()
    if t:
        parts += ["**Each candidate against the baseline, paired** (pooled net Sharpe ratio; `paired_test.py`). Test years 1998–2021, "
                  "and in brackets the validation years 1996–1997, which alone may choose.", t]
    for K in (8, 30):
        t = gap_table(K)
        if t:
            parts += [f"## The gap to the paper after each candidate, K = {K}",
                      "1998–2021. *Per seed* rows are models, as in the tables above (seed averages). The other rows are the same models traded "
                      "through a deployment layer (`analysis/gap/deploy.py`): they are not Table 2 rows. Implied turnover for the paper.", t]
    for K in (8, 30):
        t = paired_table(K)
        if t:
            parts += [f"## Paired tests against the baseline, K = {K}",
                      "Pooled series (the average over seeds of the daily net return), paired block bootstrap: `analysis/gap/paired_test.py`. "
                      "A pooled Sharpe ratio is higher than the seed average of the tables above. Differences are variant minus baseline. "
                      "The validation interval rests on two years.", t]
    t = steps_table()
    if t:
        parts += ["## Batch length against the number of steps, K = 8", t]
    for K in (8, 30):
        t = exposure_table(["baseline", "rf_hurdle", "cumulative", "cost075", "cost10", "ridge1e4", "ridge1e3", "entry_member"], K)
        if t:
            parts += [f"## Market exposure and trading by era, K = {K}",
                      "Seed averages (`analysis/gap/exposure.py`). Net long is the mean of Σw = 1 − 2 × short exposure. The last column removes "
                      "beta × market with the beta of the same years.", t]
    parts += ["## The market row", "Equal-weighted universe, 1998–2021, % a year (`analysis/gap/exposure.py --market`). *Log* is the log excess "
              "return of the same portfolio.", md(exposure.market_rows().rename(columns={"mu": "μ", "sigma": "σ"}), index="Universe and return")]
    t = gram_table()
    if t:
        parts += ["## Ridge and the factor directions it leaves in the residuals",
                  "Eigenvalues of ω^F′ω^F on 20 days of each test year (`analysis/gap/gram.py`). A direction with eigenvalue e is hedged by e / (e + ridge). "
                  "σ, SR and SR net are the run's own, 1998–2021.", t]
    s = shares("broad")
    paper = s.loc[1998:2021]
    parts += ["## Members without a signal",
              f"Under the default rule a stock gets a signal from its 31st consecutive day in the universe (`analysis/gap/entrants.py`). On `broad`, "
              f"1998–2021: {paper['no signal %'].mean():.2f}% of member stock-days on average, between {paper['no signal %'].min():.2f}% "
              f"({paper['no signal %'].idxmin()}) and {paper['no signal %'].max():.2f}% ({paper['no signal %'].idxmax()}); "
              f"{paper['first entry %'].mean():.2f}% are stocks never seen before and {paper['re-entry %'].mean():.2f}% stocks that come back."]
    for K in (8, 30):
        f = run_dir(config("baseline", K, 8)) / "entrants.csv"
        if f.exists():
            m = pd.read_csv(f, index_col=0).mean()
            parts.append(f"K = {K} baseline, signals withheld by the rule, ranked among the day's members: {m['top decile %']:.1f}% in the top decile and "
                         f"{m['bottom decile %']:.1f}% in the bottom one (10% each if they were like the others); their mean absolute signal is "
                         f"{m['|signal| against the others']:.2f} times that of the others.")
    shown = False
    for label, cfg, full in ([(f"K = {K}, baseline", config("baseline", K, 8), True) for K in (8, 30)]
                             + [(f"K = {K}, {v}", config(v, K, 8), False) for K in (8, 30) for v in ("cost10", "cumulative", "rf_hurdle", "ridge1e4")]
                             + [("K = 30, WRDS-built arrays, baseline (test years to 2025)", config("baseline", 30, 8, WRDS), True)]
                             + [(f"K = 30, WRDS-built arrays, {v}", config(v, 30, 8, WRDS), False) for v in ("ridge1e4", "cost10", "ridge1e4_cost10")]):
        t = deploy_table(cfg, full)
        if t:
            if not shown:
                parts += ["## Deployment layers: seed ensemble and no-trade band",
                          "From `analysis/gap/deploy.py`, which explains the rows. These layers change how a model's target weights are traded, not the model; "
                          "the paper's Table 2 has none. *Per seed* is the seed average as in the tables above, *pooled* the average of the three daily "
                          "returns, *ensemble* one portfolio from the averaged weights. A fixed band width is read off the test years (a diagnostic); "
                          "the walk-forward width uses earlier test years only; the validation width is chosen on 1996–1997."]
                shown = True
            parts += [f"**{label}**", t]
    if shown:
        for K in (8, 30):
            twin, main = run_dir(config("baseline", K, 6)), run_dir(config("baseline", K, 8))
            if list(twin.glob("seed*/[0-9][0-9][0-9][0-9]_w.npz")):
                parts += [f"**K = {K}, baseline: every band width on the validation years 1996–1997.** Differences are paired, against no band "
                          "(`deploy.py --band-test`).", md(deploy.band_test(twin), index="Band width δ")]
                a = deploy.seed_agreement(main)
                parts.append(f"Agreement of the three seeds, K = {K}, 1998–2021 (mean correlation between two seeds): daily weights {a['weights']:.2f}, "
                             f"daily trades {a['trades']:.2f}, daily gross returns {a['returns']:.2f}. The average of the three books has a gross exposure of "
                             f"{a['gross exposure of the average book']:.2f} before it is rescaled (1 if the seeds held the same book).")
    t = wrds_table()
    if t:
        parts += ["## Runs on the WRDS-built arrays, K = 30", "`wrds_broad` with `--fill none` (CLAUDE.md, section 4), made by `afsa.run` for test years 1998–2025.", t]
        t = paired_table(30, WRDS)
        if t:
            parts += ["Paired tests against the WRDS baseline (pooled, 1998–2021):", t]
    return parts


def main() -> None:
    singles = [v for v in VARIANTS if not v.startswith("cum_") and v != "baseline"]
    combos = [v for v in VARIANTS if v.startswith("cum_")]
    parts = [
        "# Gap analysis: results",
        f"Generated by `analysis/gap/results.py` on {datetime.now():%Y-%m-%d %H:%M} from the runs in `results/runs/`. Discussion: [REPORT.md](REPORT.md).",
        "- Seed averages over 3 seeds. Test years 1998–2021 with 8-year windows; validation years 1996–1997 with 6-year windows.\n"
        "- μ and σ in % a year, turnover per day. Every run is scored with the paper's costs: 5 bp per unit of turnover, 1 bp per unit of short exposure.\n"
        "- *Cost / σ* is the Sharpe ratio lost to costs: SR net ≈ SR − cost / σ. *Fit SR net* is the net Sharpe ratio on the training window in the last epoch (mean over batches, annualized).\n"
        "- The test-year columns diagnose the gap to the paper. They are not a basis for choosing a configuration (CLAUDE.md, section 8).",
        "## 1. Baseline against the paper, by number of factors",
    ]
    for K in (1, 3, 5, 8, 15, 30, 100):
        parts.append("\n".join([HEAD, paper_line(K), line("baseline", K) or "| baseline | not run |"]) if K == 1 else
                     "\n".join([paper_line(K), line("baseline", K) or "| baseline | not run |"]))
    parts[-7:] = ["\n".join(parts[-7:])]
    parts += ["## 2. One change at a time, K = 8", table(["baseline"] + singles, 8),
              "## 3. Combinations with the cumulative input, K = 8", table(["baseline", "cumulative"] + combos, 8)]
    for K in (30, 1, 3, 5, 15, 100):
        have = [v for v in VARIANTS if v != "baseline" and line(v, K)]
        if have:
            parts += [f"## Other runs, K = {K}", table(["baseline"] + have, K)]
    if FIG2.exists():
        parts += ["## Gross return by period: the paper's Figure 2 against our runs",
                  "Annualized log growth before costs, % a year (`analysis/gap/yearly.py`). The paper's column is read from the figure's vector data "
                  "(`analysis/gap/fig2_digitize.py`); its years are tick intervals of 245 days.",
                  "**K = 8**", yearly_table(8, ["baseline", "cumulative", "window", "cum_window", "no_daily"]),
                  "**K = 30**", yearly_table(30, ["baseline", "cumulative", "window", "cum_window"])]
    parts += review_sections()
    out = Path(__file__).with_name("RESULTS.md")
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
