"""How fast the trained policy's alpha decays, and what trading it more slowly would do.

    python analysis/gap/execution.py results/runs/attn_K008 --seeds 0 1 2

1. Stale weights: the gross return when the weights are h days old (h = 0 is the strategy as run). If the return
   is gone after a day or two, the policy cannot trade less without giving the return up.
2. Partial adjustment: each day move only a fraction `speed` of the way from yesterday's holdings to the model's
   target. This is a diagnostic of the gap, not a proposed trading rule: its parameter is read off the test years.

Costs: 5 bp per unit of turnover, 1 bp per unit of short exposure, on a portfolio scaled to unit gross exposure.
Each test year starts from the model's own weights of the day before, so the trade between two yearly models is
left out (one day in 252).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.config import WINDOW_RULES, Config, data_folder  # noqa: E402
from afsa.model import AFSAModel  # noqa: E402
from afsa.panel import Panel  # noqa: E402
from afsa.run import window_folder  # noqa: E402

LAGS = (0, 1, 2, 3, 5, 10)
SPEEDS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15)


def year_weights(run: Path, cfg: Config, year: int, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights [D+1, U] (row 0 = the day before the test year) and returns [D, U] of one model on its test year."""
    L = cfg.lookback
    model = AFSAModel.from_states(cfg, [torch.load(run / f"seed{seed}" / f"{year}.pt")], device)
    if cfg.universe in WINDOW_RULES:
        panel = Panel(window_folder(cfg, year), device)
    else:
        panel = Panel(data_folder(cfg.universe, cfg.fill), device, first_year=year, last_year=year, lead_days=L + 8)
    s, e = panel.year_span(year, year)
    b = panel.batch(torch.arange(s - (L + 1), e, device=device)[None, :], L)
    with torch.no_grad():
        out = model(b)
    return out["w"][0].double(), b.R[0, L + 1:].double()


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--first-year", type=int, default=1998)
    p.add_argument("--last-year", type=int, default=2021)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = Config.load(args.run / "config.json")
    stale = {h: [] for h in LAGS}
    slow = {a: [] for a in SPEEDS}
    for seed in args.seeds:
        lag_parts = {h: [] for h in LAGS}
        slow_parts = {a: [] for a in SPEEDS}
        for year in range(args.first_year, args.last_year + 1):
            w, R = year_weights(args.run, cfg, year, seed, args.device)
            D = R.shape[0]
            for h in LAGS:                                                   # weights of day t-h on the returns of day t
                lag_parts[h].append((w[1:D + 1 - h] * R[h:]).sum(-1).cpu().numpy())
            for a in SPEEDS:
                held = w[0].clone()
                rows = []
                for t in range(D):
                    new = (1 - a) * held + a * w[t + 1]
                    rows.append(torch.stack([(new * R[t]).sum(), (new - held).abs().sum(), torch.relu(-new).sum(), new.abs().sum()]))
                    held = new
                slow_parts[a].append(torch.stack(rows).cpu().numpy())
        for h in LAGS:
            stale[h].append(np.concatenate(lag_parts[h]))
        for a in SPEEDS:
            slow[a].append(np.concatenate(slow_parts[a]))
    pd.set_option("display.width", 200)
    print(f"{args.run.name}, test years {args.first_year}-{args.last_year}, seeds {args.seeds} (seed averages)\n")
    t = pd.DataFrame({h: {"gross % a year": np.mean([x.mean() for x in stale[h]]) * 252 * 100, "Sharpe": np.mean([sharpe(x) for x in stale[h]])}
                      for h in LAGS}).T
    t.index.name = "weights are h days old"
    t["share of the fresh return"] = t["gross % a year"] / t.loc[0, "gross % a year"]
    print("1. Stale weights")
    print(t.round(2).to_string())
    rows = {}
    for a in SPEEDS:
        per = []
        for x in slow[a]:
            scale = 1 / x[:, 3].mean()                                       # back to unit gross exposure on average
            gross, turn, short = x[:, 0] * scale, x[:, 1] * scale, x[:, 2] * scale
            net = gross - 5e-4 * turn - 1e-4 * short
            per.append({"SR": sharpe(gross), "mu": gross.mean() * 252 * 100, "sigma": gross.std(ddof=1) * np.sqrt(252) * 100,
                        "SR_net": sharpe(net), "mu_net": net.mean() * 252 * 100, "turnover": turn.mean(), "gross exposure before scaling": x[:, 3].mean()})
        rows[a] = pd.DataFrame(per).mean()
    t = pd.DataFrame(rows).T
    t.index.name = "speed"
    print("\n2. Partial adjustment towards the model's weights (speed 1 = the strategy as run)")
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
