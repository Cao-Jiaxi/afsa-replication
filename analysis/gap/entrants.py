"""Members of the universe that get no signal, and where their signal would rank.

    python analysis/gap/entrants.py                                   # share by test year, `broad` arrays
    python analysis/gap/entrants.py --universe wrds_broad --fill none
    python analysis/gap/entrants.py --run results/runs/attn_K030      # + the withheld signals -> entrants.csv there

Under the default `entry_rule = run`, a stock gets a signal only from its 31st consecutive day in the universe
(Config.lookback = 30). The monthly top 500 changes every month and a stock that leaves for one month starts again
from zero, so on any day some members are held only through the factor hedge.

1. Share of member stock-days without a signal, by test year, split into stocks never seen before and stocks that
   come back. From the arrays alone.
2. With `--run`: the saved models are evaluated with `entry_rule = member` (lags from before the entry are zero
   residuals), and each withheld signal is ranked among the day's members. If these stocks were like the others,
   10% of them would fall in the top decile and 10% in the bottom one.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.config import TRAIN_DATA_DIR, WINDOW_RULES, Config, data_folder  # noqa: E402

FIRST, LAST = 1998, 2025


def shares(folder: str, lookback: int = 30) -> pd.DataFrame:
    """Per year: % of member stock-days without a signal, of which first entries and re-entries, and stocks a day."""
    root = TRAIN_DATA_DIR / folder
    permno, run = np.load(root / "permno.npy"), np.load(root / "run.npy")
    dates = np.load(root / "dates.npy").astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    member = permno >= 0
    blocked = member & (run <= lookback)
    day = np.nonzero(member)[0]                                              # row-major: a stock's first hit is its first day
    permnos, first_hit = np.unique(permno[member], return_index=True)
    first_seen = np.where(member, day[first_hit][np.searchsorted(permnos, np.where(member, permno, permnos[0]))], -1)
    start = np.arange(len(dates))[:, None] - (run.astype(np.int64) - 1)      # first day of the current spell
    new = blocked & (start == first_seen)
    rows = {}
    for y in range(FIRST, min(LAST, int(years.max())) + 1):
        m = years == y
        n = member[m].sum()
        rows[y] = {"no signal %": 100 * blocked[m].sum() / n, "first entry %": 100 * new[m].sum() / n,
                   "re-entry %": 100 * (blocked[m] & ~new[m]).sum() / n, "stocks a day": blocked[m].sum() / m.sum()}
    return pd.DataFrame(rows).T


def withheld(run: Path, device: str = "cuda", chunk: int = 36) -> pd.DataFrame:
    """Per test year (seed averages): where the signals of the members without a signal would rank."""
    import torch

    from afsa.model import AFSAModel
    from afsa.panel import Panel

    cfg = Config.load(run / "config.json")
    if cfg.universe in WINDOW_RULES:
        raise SystemExit("entrants.py handles universes with one sample for all years")
    L, panel = cfg.lookback, Panel(data_folder(cfg.universe, cfg.fill), device)
    every = dataclasses.replace(cfg, entry_rule="member")
    files = sorted(run.glob("seed*/[0-9][0-9][0-9][0-9].pt"))
    rows = []
    for lo in range(0, len(files), chunk):
        part = files[lo:lo + chunk]
        spans = [panel.year_span(int(f.stem), int(f.stem)) for f in part]
        D = max(e - s for s, e in spans)
        first = torch.tensor([s for s, _ in spans], device=panel.device)
        days = (first[:, None] + torch.arange(-(L + 1), D, device=panel.device)).clamp_max(panel.loaded[1] - 1)
        batch = panel.batch(days, L)
        with torch.no_grad():
            sig = AFSAModel.from_states(every, [torch.load(f) for f in part], panel.device)(batch)["w_port"][:, 1:]   # loss days
        member, valid = batch.member[:, L + 1:], batch.valid[:, L + 1:]
        order = sig.masked_fill(~member, float("inf")).argsort(-1).argsort(-1)       # 0 = lowest signal of the day
        pct = order / (member.sum(-1, keepdim=True) - 1).clamp_min(1)
        fresh = member & ~valid
        for j, (f, (s, e)) in enumerate(zip(part, spans)):
            own = slice(0, e - s)
            fr, ok = fresh[j, own], valid[j, own]
            rows.append({"year": int(f.stem), "top decile %": 100 * float((pct[j, own][fr] >= 0.9).float().mean()),
                         "bottom decile %": 100 * float((pct[j, own][fr] <= 0.1).float().mean()),
                         "|signal| against the others": float(sig[j, own][fr].abs().mean() / sig[j, own][ok].abs().mean())})
    return pd.DataFrame(rows).groupby("year").mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default="broad")
    p.add_argument("--fill", default="last")
    p.add_argument("--run", type=Path, default=None, help="a run folder: also rank the withheld signals of its models")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    pd.set_option("display.width", 200)
    if args.run is None:
        t = shares(data_folder(args.universe, args.fill))
        print(t.round(2).to_string())
        paper = t.loc[:2021]
        print(f"\n1998-2021: {paper['no signal %'].mean():.2f}% of member stock-days (mean of the yearly figures)")
        return
    t = withheld(args.run, args.device)
    t.to_csv(args.run / "entrants.csv")
    print(t.round(2).to_string())
    print(f"\nmean over years: {t.mean().round(2).to_dict()}")


if __name__ == "__main__":
    main()
