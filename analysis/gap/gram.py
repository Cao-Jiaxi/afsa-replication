"""How much of each factor direction the ridge leaves in the residuals: eigenvalues of the factor Gram matrix.

    python analysis/gap/gram.py results/runs/attn_K030 results/runs/attn_K100     # -> gram.csv in each folder

The residual of a return vector v is v - wF (wF'wF + ridge I)^-1 wF' v, with wF the N x K matrix of factor weights.
Along an eigenvector of wF'wF with eigenvalue e, the hedge removes the share e / (e + ridge) of v and leaves
ridge / (e + ridge). A direction with e far above the ridge is hedged in full; one with e near or below it is
largely left in the residuals, and so in the portfolio.

Each softmax column sums to 1 over about 500 stocks, so the diagonal of wF'wF is at least 1/N = 0.002. Small
eigenvalues mean nearly collinear factors, which is what many factors on 39 characteristics produce.

For each saved model, on 20 evenly spaced days of its test year (seed averages by year, then over years):
- the median and the smallest eigenvalue of the day,
- the number of the K directions hedged by less than 90% and by less than 50%,
- the mean hedged share over the K directions.
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

DAYS = 20


@torch.no_grad()
def eigenvalues(cfg: Config, panel: Panel, states: list[dict], years: list[int], days: int = DAYS) -> np.ndarray:
    """Eigenvalues of wF'wF, ascending, for each model on `days` evenly spaced days of its test year. [M, days, K]"""
    model = AFSAModel.from_states(cfg, states, panel.device)
    spans = [panel.year_span(y, y) for y in years]
    pick = torch.tensor(np.stack([np.linspace(s, e - 1, days).round().astype(np.int64) for s, e in spans]), device=panel.device)
    wF = model.factor_weights(panel.batch(pick, cfg.lookback)).double()      # [M, days, U, K]
    return torch.linalg.eigvalsh(torch.matmul(wF.transpose(2, 3), wF)).clamp_min(0).cpu().numpy()


def summary(run: Path, panel: Panel, chunk: int = 36) -> pd.DataFrame:
    """One row per test year (seed averages) for the models of a run."""
    cfg = Config.load(run / "config.json")
    files = sorted(run.glob("seed*/[0-9][0-9][0-9][0-9].pt"))
    rows = []
    for lo in range(0, len(files), chunk):
        part = files[lo:lo + chunk]
        e = eigenvalues(cfg, panel, [torch.load(f) for f in part], [int(f.stem) for f in part])
        hedged = e / (e + cfg.ridge)
        for f, ev, h in zip(part, e, hedged):
            rows.append({"year": int(f.stem), "median eigenvalue": np.median(ev), "smallest eigenvalue": np.median(ev[:, 0]),
                         "directions hedged < 90%": (h < 0.9).sum(1).mean(), "directions hedged < 50%": (h < 0.5).sum(1).mean(),
                         "mean hedged share": h.mean()})
    return pd.DataFrame(rows).groupby("year").mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    panel = None
    for run in args.runs:
        cfg = Config.load(run / "config.json")
        if cfg.universe in WINDOW_RULES:
            raise SystemExit("gram.py handles universes with one sample for all years")
        folder = data_folder(cfg.universe, cfg.fill)
        if panel is None or panel.universe != folder:
            panel = Panel(folder, args.device)
        t = summary(run, panel)
        t.to_csv(run / "gram.csv")
        m = t.mean()
        print(f"{run.name}: K = {cfg.K}, ridge {cfg.ridge:g}: median eigenvalue {m['median eigenvalue']:.2e}, smallest {m['smallest eigenvalue']:.2e}; "
              f"{m['directions hedged < 90%']:.1f} of {cfg.K} directions hedged by less than 90%, {m['directions hedged < 50%']:.1f} by less than 50%; "
              f"mean hedged share {m['mean hedged share']:.3f}", flush=True)


if __name__ == "__main__":
    main()
