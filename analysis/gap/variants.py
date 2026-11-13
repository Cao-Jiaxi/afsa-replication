"""Variant runs for the gap analysis (analysis/gap/REPORT.md): one change to the baseline at a time.

    python analysis/gap/variants.py --list
    python analysis/gap/variants.py epochs10 daily_sharpe --K 8 30     # trains what is missing, then prints the table
    python analysis/gap/variants.py --table --K 8                      # tabulates what is on disk

Each variant is a set of `Config` overrides. It is run on the paper's test years (1998-2021, 8-year window) and on
the validation years (1996-1997, 6-year window, which uses nothing after 1997). Whatever a variant is trained
with, its results are scored with the paper's costs (5 bp per unit of turnover, 1 bp per unit of short exposure).

The test-year columns are diagnostics of the gap to the paper. They are not a basis for choosing a variant:
CLAUDE.md, section 8.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.config import RESULTS_DIR, TRADING_DAYS, Config, data_folder  # noqa: E402
from afsa.metrics import PAPER_TABLE2, stitch  # noqa: E402

VARIANTS: dict[str, dict] = {
    "baseline": {},
    # how long the model is fitted: an E-epoch run is the baseline stopped after E epochs (same random streams)
    "epochs03": {"epochs": 3}, "epochs05": {"epochs": 5}, "epochs08": {"epochs": 8}, "epochs10": {"epochs": 10},
    "epochs15": {"epochs": 15}, "epochs20": {"epochs": 20}, "epochs60": {"epochs": 60},
    # the Sharpe ratio of the objective as the paper prints it (per period), against lambda_var = 100
    "daily_sharpe": {"sharpe_days": 1.0},
    "lambda_var0": {"lambda_var": 0.0}, "lambda_var10": {"lambda_var": 10.0}, "lambda_var1000": {"lambda_var": 1000.0},
    # batches
    # batches: 35 or 49 consecutive days are the lengths that cut every calendar year into exactly 245 days, the
    # length of the paper's test years (fig2_digitize.py)
    "batch035": {"batch_days": 35}, "batch049": {"batch_days": 49}, "batch063": {"batch_days": 63},
    "batch252": {"batch_days": 252}, "batch504": {"batch_days": 504},
    # LongConv
    "kernel002": {"kernel_init_std": 0.02},
    "dropout0": {"dropout": 0.0}, "dropout03": {"dropout": 0.3},
    "lr001": {"lr": 1e-3}, "cosine": {"lr_schedule": "cosine"},
    "cumulative": {"cumulative": True},                      # LongConv sees the residual price path, as in Guijarro-Ordonez et al.
    # the objective as printed: raw returns minus the risk-free rate in the Sharpe term
    "rf_hurdle": {"rf_hurdle": True},
    # the factor model
    "w_init3": {"w_init_scale": 3.0}, "w_init10": {"w_init_scale": 10.0}, "q_init3": {"q_init_std": 3.0},
    "no_daily": {"drop_chars": ("Ret_D1", "Ret_W1", "STD_W1", "Vol")},     # only the characteristics that change monthly
    "no_past_returns": {"drop_groups": ("past_returns",)},                 # a row of the paper's Table 3
    "no_frictions": {"drop_groups": ("trading_frictions",)},
    # data
    "common": {"universe": "common"},                                      # US common stocks only
    # 500 stocks fixed per training window (afsa.build_window): the companion paper's rule, and without its look-ahead
    "window": {"universe": "window"}, "window_train": {"universe": "window_train"},
    # what trading cost the policy is trained with (always scored at 5 bp)
    "cost10": {"cost_trade": 1e-3}, "cost075": {"cost_trade": 7.5e-4},
    # combinations with the cumulative input
    "cum_window": {"cumulative": True, "universe": "window"},
    "cum_window_train": {"cumulative": True, "universe": "window_train"},
    "cum_rf_hurdle": {"cumulative": True, "rf_hurdle": True},
    "cum_batch252": {"cumulative": True, "batch_days": 252},
    "cum_kernel002": {"cumulative": True, "kernel_init_std": 0.02},
    "cum_daily_sharpe": {"cumulative": True, "sharpe_days": 1.0},
    "cum_no_daily": {"cumulative": True, "drop_chars": ("Ret_D1", "Ret_W1", "STD_W1", "Vol")},
    "cum_epochs60": {"cumulative": True, "epochs": 60},
    "cum_common": {"cumulative": True, "universe": "common"},
    "cum_no_past_returns": {"cumulative": True, "drop_groups": ("past_returns",)},
    "cum_window_hurdle": {"cumulative": True, "universe": "window", "rf_hurdle": True},
    "cum_w_init10": {"cumulative": True, "w_init_scale": 10.0},
    "cum_cosine": {"cumulative": True, "lr_schedule": "cosine"},
}
# Table 3 of the paper (K = 30): SR, mu, sigma, SR_net, mu_net
PAPER_TABLE3 = {"no_past_returns": (1.50, 7.82, 5.23, 0.59, 3.09), "no_frictions": (2.90, 13.36, 4.61, 1.34, 6.14),
                "cum_no_past_returns": (1.50, 7.82, 5.23, 0.59, 3.09)}
PERIODS = {"test": (1998, 2021, 8), "validation": (1996, 1997, 6)}        # first year, last year, training years
BLOCKS = ((1998, 2005), (2006, 2013), (2014, 2021))
COST_TRADE, COST_SHORT = 5e-4, 1e-4                                        # the paper's costs


def config(variant: str, K: int, train_years: int, extra: dict | None = None) -> Config:
    return Config(**{"K": K, "train_years": train_years, **VARIANTS[variant], **(extra or {})})


def run_dir(cfg: Config) -> Path:
    return RESULTS_DIR / "runs" / cfg.name()


def stats(daily: pd.DataFrame) -> dict[str, float]:
    k = TRADING_DAYS
    net = daily["gross"] - COST_TRADE * daily["turnover"] - COST_SHORT * daily["short"]
    out = {
        "SR": daily["gross"].mean() / daily["gross"].std() * np.sqrt(k), "mu": daily["gross"].mean() * k * 100,
        "sigma": daily["gross"].std() * np.sqrt(k) * 100, "SR_net": net.mean() / net.std() * np.sqrt(k),
        "mu_net": net.mean() * k * 100, "turnover": daily["turnover"].mean(), "short": daily["short"].mean(),
    }
    out["cost/sigma"] = (out["mu"] - out["mu_net"]) / out["sigma"]        # Sharpe points lost to costs: SR_net ~ SR - cost/sigma
    return out


def row(cfg: Config, first: int, last: int) -> dict[str, float] | None:
    """Seed averages of a run on disk, or None if it is missing or incomplete."""
    folder = run_dir(cfg)
    seeds = sorted(folder.glob("seed*"))
    per_seed, fit = [], []
    for seed_dir in seeds:
        if any(not (seed_dir / f"{y}.npz").exists() for y in range(first, last + 1)):
            return None
        daily = stitch(seed_dir, cfg)
        daily = daily[(daily.index.year >= first) & (daily.index.year <= last)]
        s = stats(daily)
        for a, b in BLOCKS:
            if first <= a and b <= last:
                s[f"SR_net {a}-{str(b)[2:]}"] = stats(daily[(daily.index.year >= a) & (daily.index.year <= b)])["SR_net"]
        per_seed.append(s)
        for y in range(first, last + 1):
            z = np.load(seed_dir / f"{y}.npz")
            h = dict(zip([str(c) for c in z["history_columns"]], z["history"][-1]))
            scale = np.sqrt(TRADING_DAYS / cfg.sharpe_days)                # the history is in the objective's own scale
            fit.append({"fit SR_net": h["sharpe_net"] * scale, "fit turnover": h["turnover"], "fit EV": h["explained_var"],
                        "EV": float(z["explained_var"])})
    if not per_seed:
        return None
    out = pd.DataFrame(per_seed).mean().to_dict()
    out["SR sd"] = pd.DataFrame(per_seed)["SR"].std() if len(per_seed) > 1 else np.nan
    out.update(pd.DataFrame(fit).mean().to_dict())
    out["seeds"] = len(per_seed)
    return out


def table(variants: list[str], Ks: list[int], extra: dict | None = None) -> pd.DataFrame:
    rows = {}
    for K in Ks:
        if K in PAPER_TABLE2 and not extra:
            sr, mu, sigma, sr_net, mu_net, _, _ = PAPER_TABLE2[K]
            cost_bp = (mu - mu_net) / TRADING_DAYS * 100
            rows[("paper", K, "test")] = {"SR": sr, "mu": mu, "sigma": sigma, "SR_net": sr_net, "mu_net": mu_net,
                                          "turnover": (cost_bp - 0.5) / 5,      # implied, if the short exposure is 0.5
                                          "cost/sigma": (mu - mu_net) / sigma}
        for v in variants:
            if v in PAPER_TABLE3 and K == 30 and not extra:
                sr, mu, sigma, sr_net, mu_net = PAPER_TABLE3[v]
                rows[(f"paper {v}", K, "test")] = {"SR": sr, "mu": mu, "sigma": sigma, "SR_net": sr_net, "mu_net": mu_net,
                                                   "turnover": ((mu - mu_net) / TRADING_DAYS * 100 - 0.5) / 5,
                                                   "cost/sigma": (mu - mu_net) / sigma}
            for period, (first, last, train_years) in PERIODS.items():
                r = row(config(v, K, train_years, extra), first, last)
                if r is not None:
                    rows[(v, K, period)] = r
    t = pd.DataFrame(rows).T
    t.index.names = ["variant", "K", "period"]
    return t


def train(variants: list[str], Ks: list[int], seeds: list[int], periods: list[str], extra: dict | None = None) -> None:
    from afsa.config import WINDOW_RULES
    from afsa.panel import Panel
    from afsa.run import run_config
    panels: dict[str, Panel | None] = {}
    for v in variants:
        for K in Ks:
            for period in periods:
                first, last, train_years = PERIODS[period]
                cfg = config(v, K, train_years, extra)
                folder = data_folder(cfg.universe, cfg.fill)
                if folder not in panels:                                   # a per-window universe loads one panel per test year
                    panels[folder] = None if cfg.universe in WINDOW_RULES else Panel(folder, "cuda")
                run_config(cfg, seeds, list(range(first, last + 1)), panels[folder])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="*", help="variants to run (default with --table: all on disk)")
    p.add_argument("--K", type=int, nargs="+", default=[8])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE", help="extra Config overrides applied to every variant")
    p.add_argument("--table", action="store_true", help="do not train: tabulate what is on disk")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for name, overrides in VARIANTS.items():
            print(f"{name:16s} {overrides}")
        return
    types = {f.name: type(f.default) for f in dataclasses.fields(Config)}
    extra = {}
    for item in args.set:
        key, value = item.split("=", 1)
        extra[key] = (value.lower() in ("1", "true", "yes")) if types[key] is bool else types[key](value)
    variants = args.variants or list(VARIANTS)
    if not args.table:
        train(variants, args.K, args.seeds, args.periods, extra)
    t = table(variants, args.K, extra)
    pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 500)
    print(t.astype(float).round(2).to_string())


if __name__ == "__main__":
    main()
