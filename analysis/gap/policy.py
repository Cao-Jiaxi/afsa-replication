"""What a trained policy does on its test year: where the turnover comes from, how concentrated the portfolio is,
which lagged residuals drive the signal, and which kinds of securities carry the trading and the return.

    python analysis/gap/policy.py results/runs/attn_K008 --years 2000 2005 2010 2015 2020 --seeds 0

Turnover is split with the triangle inequality. With s the residual-space signal and P the projection of the day,
the weights are w_t = norm(P_t s_t). Putting yesterday's signal through today's projection, a_t = norm(P_t s_{t-1}):
  signal part     ||w_t - a_t||_1      what changes because the signal changed
  projection part ||a_t - w_{t-1}||_1  what changes because the factor weights or the universe changed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.config import CHARACTERISTICS, SOURCE_DIR, WINDOW_RULES, Config, data_folder  # noqa: E402
from afsa.model import AFSAModel  # noqa: E402
from afsa.panel import Panel  # noqa: E402
from afsa.run import window_folder  # noqa: E402

DAILY = ("Ret_D1", "Ret_W1", "STD_W1", "Vol")              # the other 35 characteristics change once a month
TYPES = {"comun": "US common", "extranjera": "foreign ordinary", "ADR": "ADR", "REIT": "REIT", "MLP": "MLP"}


def norm1(x: torch.Tensor) -> torch.Tensor:
    return x / x.abs().sum(-1, keepdim=True).clamp_min(1e-12)


def analyze(run: Path, year: int, seed: int, device: str, universe: pd.DataFrame) -> tuple[dict, pd.DataFrame, np.ndarray]:
    cfg = Config.load(run / "config.json")
    L = cfg.lookback
    model = AFSAModel.from_states(cfg, [torch.load(run / f"seed{seed}" / f"{year}.pt")], device)
    if cfg.universe in WINDOW_RULES:                                        # the window's own list of stocks
        panel = Panel(window_folder(cfg, year), device)
    else:
        panel = Panel(data_folder(cfg.universe, cfg.fill), device, first_year=year, last_year=year, lead_days=L + 8)
    s, e = panel.year_span(year, year)
    b = panel.batch(torch.arange(s - (L + 1), e, device=device)[None, :], L)
    with torch.no_grad():
        out = model(b)
        wF = out["wF"]
        G, chol = model.gram(wF)
        sig = out["w_port"]                                                    # [1, D+1, U]; row 0 = the day before the year
        raw = model.project(wF[:, L:], (G[:, L:], chol[:, L:]), sig)           # P s, before the L1 scaling
        w = out["w"][0]
        prev = sig[:, :-1] * b.valid[:, L + 1:]                                # yesterday's signal, today's stocks
        a = norm1(model.project(wF[:, L + 1:], (G[:, L + 1:], chol[:, L + 1:]), prev))[0]
        s_hat = norm1(sig[0])
        R = b.R[0, L + 1:]
        dw = (w[1:] - w[:-1]).abs()
        res = {
            "turnover": dw.sum(-1).mean(),
            "signal part": (w[1:] - a).abs().sum(-1).mean(),
            "projection part": (a - w[:-1]).abs().sum(-1).mean(),
            "turnover of the signal": (s_hat[1:] - s_hat[:-1]).abs().sum(-1).mean(),
            "hedge share of |w|": ((raw[0] - sig[0]).abs().sum(-1) / raw[0].abs().sum(-1)).mean(),
            "effective positions": (1 / (w * w).sum(-1)).mean(),
            "top-10 share of |w|": w.abs().topk(10, dim=-1).values.sum(-1).mean(),
            "top-50 share of |w|": w.abs().topk(50, dim=-1).values.sum(-1).mean(),
            "net exposure": w.sum(-1).mean(),
            "same-sign share": (w[1:].abs() * (torch.sign(w[1:]) == torch.sign(w[:-1]))).sum(-1).mean(),
            "gross % a year": (w[1:] * R).sum(-1).mean() * 252 * 100,
        }
        # factor weights on the first test day: how much of each factor sits in its 10 largest stocks (paper: 10-23%)
        res["factor top-10 share"] = wF[0, L + 1].topk(10, dim=0).values.sum(0).mean()
        res["factor turnover"] = (wF[0, L + 1:] - wF[0, L:-1]).abs().sum(1).mean()          # per factor portfolio, per day
        # scores of eq. (1) = X A with A = W Q' / sqrt(d_x): their spread over stocks, and how much of it comes
        # from the four characteristics that change every day
        A = (model.W_rank[0] * model.char_mask[:, None]) @ model.Q[0].T / cfg.d_x ** 0.5     # [39, K]
        daily = torch.tensor([c in DAILY for c in CHARACTERISTICS], device=A.device)
        mem = b.member[0, L + 1:]
        X = b.X[0, L + 1:]
        var = lambda s: (((s - (s * mem[..., None]).sum(1, keepdim=True) / mem.sum(1)[:, None, None]) ** 2) * mem[..., None]).sum(1) / mem.sum(1)[:, None]
        total, part = var(X @ A), var(X[..., daily] @ A[daily])
        res["score sd over stocks"] = total.mean().sqrt()
        res["daily characteristics' share of score variance"] = (part / total).mean()
        res = {k: float(v) for k, v in res.items()}
        # how the signal relates to the recent residuals
        eps = out["eps"][0] * cfg.resid_scale
        x = eps.unfold(0, L, 1)[: eps.shape[0] - L]                            # [D+1, U, L], oldest lag first
        ok = b.valid[0, L:]
        for name, feat in (("corr(signal, -last residual)", -x[..., -1]), ("corr(signal, -5-day residual)", -x[..., -5:].sum(-1)),
                           ("corr(signal, -30-day residual)", -x.sum(-1))):
            res[name] = float(torch.corrcoef(torch.stack([sig[0][ok], feat[ok]]))[0, 1])
    # sensitivity of the signal to each lag (mean gradient over stock-days), newest lag first
    xg = x[None].clone().requires_grad_(True)
    model.signal(xg)[0][ok].sum().backward()
    lag = xg.grad[0][ok].mean(0).flip(0).cpu().numpy()

    # by kind of security and by size
    permnos = panel.permnos[b.cols[0, : int(b.n_cols[0])].cpu().numpy()]
    dates = pd.DatetimeIndex(panel.dates[s:e])
    month = dates.to_period("M")                                             # the universe file is keyed by holding month
    key = pd.MultiIndex.from_arrays([month.repeat(len(permnos)), np.tile(permnos, len(dates))])
    info = universe.reindex(key)
    kind = info["tipo"].to_numpy().reshape(len(dates), -1)
    rank = info["rank"].to_numpy().reshape(len(dates), -1)
    wn, dwn, pnl = w[1:].abs().cpu().numpy(), dw.cpu().numpy(), (w[1:] * R).cpu().numpy()
    groups = {TYPES[t]: kind == t for t in TYPES}
    groups.update({"size rank 1-100": rank <= 100, "size rank 101-300": (rank > 100) & (rank <= 300), "size rank 301-500": rank > 300})
    rows = {g: {"stocks": m.sum(1).mean(), "|w| share": (wn * m).sum(1).mean(), "turnover": (dwn * m).sum(1).mean(),
                "gross % a year": (pnl * m).sum(1).mean() * 252 * 100} for g, m in groups.items()}
    return res, pd.DataFrame(rows).T, lag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--years", type=int, nargs="+", default=[2000, 2005, 2010, 2015, 2020])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = Config.load(args.run / "config.json")
    name = "universe_comun.parquet" if cfg.universe == "common" else "universe.parquet"   # kinds and size ranks: monthly lists only
    universe = pd.read_parquet(SOURCE_DIR / name, columns=["month", "permno", "tipo", "rank"]).set_index(["month", "permno"])
    pd.set_option("display.width", 250)
    summary, kinds, lags = {}, [], []
    for year in args.years:
        per = [analyze(args.run, year, seed, args.device, universe) for seed in args.seeds]
        summary[year] = pd.DataFrame([r[0] for r in per]).mean()
        kinds.append(sum(r[1] for r in per) / len(per))
        lags.append(np.mean([r[2] for r in per], axis=0))
    t = pd.DataFrame(summary)
    t["mean"] = t.mean(axis=1)
    print(f"{args.run.name}, seeds {args.seeds}: one column per test year")
    print(t.round(3).to_string())
    k = sum(kinds) / len(kinds)
    k["turnover per |w|"] = k["turnover"] / k["|w| share"]
    k["gross per unit turnover, bp"] = k["gross % a year"] / 252 * 100 / k["turnover"]
    print("\nBy kind of security and by size (averages over the years above):")
    print(k.round(3).to_string())
    lag = np.mean(lags, axis=0)
    print("\nSensitivity of the signal to each lagged residual, as a share of the total (newest first, 30 lags):")
    print(np.round(lag / np.abs(lag).sum(), 3))


if __name__ == "__main__":
    main()
