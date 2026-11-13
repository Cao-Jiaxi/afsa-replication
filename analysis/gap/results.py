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
from fig2_digitize import OUT as FIG2  # noqa: E402
from fig2_digitize import block_growth  # noqa: E402
from variants import PERIODS, VARIANTS, config, row  # noqa: E402
from yearly import BLOCKS, ours  # noqa: E402

from afsa.metrics import PAPER_TABLE2  # noqa: E402

WHAT = {
    "baseline": "the configuration of CLAUDE.md, section 7",
    "epochs03": "3 epochs instead of 30", "epochs05": "5 epochs", "epochs08": "8 epochs", "epochs10": "10 epochs", "epochs15": "15 epochs",
    "epochs20": "20 epochs", "epochs60": "60 epochs",
    "batch035": "batches of 35 days (58 steps an epoch)", "batch049": "batches of 49 days (41 steps)", "batch063": "batches of 63 days (32 steps)",
    "batch252": "batches of 252 days (8 steps an epoch)", "cosine": "learning rate decays to zero (cosine)", "lr001": "learning rate 0.001",
    "daily_sharpe": "Sharpe term not annualized, as printed (explained variance weighs 15.9 times more)",
    "lambda_var0": "no explained-variance term", "lambda_var10": "lambda_Var = 10", "lambda_var1000": "lambda_Var = 1000",
    "kernel002": "kernel initialized at 0.02 of the paper's scale (the public LongConv code)", "dropout0": "no dropout", "dropout03": "dropout 0.3",
    "cumulative": "LongConv sees the running sum of the 30 residuals", "rf_hurdle": "risk-free rate subtracted from the portfolio return in the objective",
    "w_init3": "W^K initialized 3 times larger", "w_init10": "W^K initialized 10 times larger", "q_init3": "Q initialized 3 times larger",
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
    out = Path(__file__).with_name("RESULTS.md")
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
