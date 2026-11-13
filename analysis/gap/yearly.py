"""Gross return year by year: the paper's Figure 2 (read back by fig2_digitize.py) next to our runs.

    python analysis/gap/yearly.py --K 8 baseline window window_train
    python analysis/gap/yearly.py --K 30 baseline

Both columns are annualized log growth before costs, in % a year (252 x the mean daily log return). The paper's
"year" is a tick interval of its figure, 245 of its 5,880 days, so the alignment with calendar years is close
but not exact. Ours is the seed average.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig2_digitize import OUT, block_growth  # noqa: E402
from variants import PERIODS, config, run_dir  # noqa: E402

from afsa.metrics import stitch  # noqa: E402

BLOCKS = ((1998, 2002), (2003, 2007), (2008, 2012), (2013, 2021), (1998, 2021))


def ours(variant: str, K: int) -> pd.Series | None:
    first, last, train_years = PERIODS["test"]
    cfg = config(variant, K, train_years)
    per_seed = []
    for seed_dir in sorted(run_dir(cfg).glob("seed*")):
        if any(not (seed_dir / f"{y}.npz").exists() for y in range(first, last + 1)):
            return None
        g = np.log1p(stitch(seed_dir, cfg)["gross"])
        per_seed.append(g.groupby(g.index.year).mean() * 252 * 100)
    return pd.concat(per_seed, axis=1).mean(axis=1) if per_seed else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="*", default=["baseline"])
    p.add_argument("--K", type=int, default=8)
    args = p.parse_args()
    if not OUT.exists():
        raise SystemExit("run analysis/gap/fig2_digitize.py first")
    paper = block_growth(pd.read_csv(OUT, index_col="day"))
    t = pd.DataFrame({f"paper K={args.K}": paper[f"K={args.K}"]})
    for v in args.variants:
        s = ours(v, args.K)
        if s is not None:
            t[v] = s
    t.index.name = "year"
    rows = {f"{a}-{b}": t.loc[a:b].mean() for a, b in BLOCKS}
    pd.set_option("display.width", 200)
    print(t.round(1).to_string())
    print("\nAverages:")
    print(pd.DataFrame(rows).T.round(1).to_string())
    if t.shape[1] > 1:
        print("\nCorrelation of the yearly figures with the paper's:", t.corr().iloc[0, 1:].round(2).to_dict())


if __name__ == "__main__":
    main()
