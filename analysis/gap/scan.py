"""Hyperparameter scan after Zhishu Zhang's review of 2026-11-13 (analysis/gap/REPORT.md, section 10).

    python analysis/gap/scan.py --list --K 8 30          # the grid, and which runs are complete on disk
    python analysis/gap/scan.py train --K 8 30           # trains what is missing: test, validation and holdout years
    python analysis/gap/scan.py weights --K 8 30         # daily weights of the saved models (reeval_weights.py), for the layers
    python analysis/gap/selection.py --K 8 30            # the selection rules and their tables -> analysis/gap/SCAN.md
    python analysis/gap/scan.py train --K 30 --only 7.5/1e-4/30 --set universe=wrds_common fill=none   # one setting, other arrays

The grid spans the three model levers of the review: the training cost (its P2), the ridge (P6) and the training
length (P1). Every other parameter is the baseline's (CLAUDE.md, section 7), and every result is scored at the
paper's costs (5 bp per unit of turnover, 1 bp per unit short), whatever cost the model was trained with.
- training cost per unit of turnover: 5 bp (the paper's), 7.5, 10 and 15 bp
- ridge: 1e-5 (the baseline), 1e-4 (the reviewer's) and 1e-3
- epochs: 30 (Table 4) and 15

Each setting is trained on three periods, with seeds 0, 1 and 2:
- test:       1998-2021, 8-year windows (the paper's Table 2)
- validation: 1996-1997, 6-year windows (nothing after 1997; the paper's Appendix B chooses its parameters there)
- holdout:    2022-2025, 8-year windows, in the test run's folder. No selection rule looks at these years until it
              has made its choice for them.

A parameter at its baseline value does not appear in the folder name, so the runs already on disk (variants.py,
afsa.run) are reused, not retrained.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.config import RESULTS_DIR, Config  # noqa: E402

COSTS = (5e-4, 7.5e-4, 1e-3, 1.5e-3)
RIDGES = (1e-5, 1e-4, 1e-3)
EPOCHS = (30, 15)
PERIODS = {"test": (1998, 2021, 8), "validation": (1996, 1997, 6), "holdout": (2022, 2025, 8)}   # first, last, training years
SEEDS = (0, 1, 2)
UNIVERSE = "broad"
# Replaying saved models in other groups than they were trained in moves float32 results by about 1e-5 (1.3e-5 seen
# on 2026-11-14); a change of code moves them by far more. Each period is replayed on its own, as it was trained.
REPLAY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Setting:
    cost_trade: float
    ridge: float
    epochs: int

    @property
    def label(self) -> str:
        return f"{self.cost_trade * 1e4:g} bp, ridge {self.ridge:g}, {self.epochs} epochs"

    def config(self, K: int, train_years: int = 8, extra: dict | None = None) -> Config:
        """`extra`: other Config fields, e.g. {"universe": "wrds_common", "fill": "none"} for other arrays."""
        return Config(**{"K": K, "train_years": train_years, "cost_trade": self.cost_trade, "ridge": self.ridge, "epochs": self.epochs,
                         **(extra or {})})

    def run_dir(self, K: int, train_years: int = 8, extra: dict | None = None) -> Path:
        return RESULTS_DIR / "runs" / self.config(K, train_years, extra).name()


GRID = tuple(Setting(c, r, e) for c in COSTS for r in RIDGES for e in EPOCHS)
BASELINE = Setting(5e-4, 1e-5, 30)                                  # CLAUDE.md, section 7
BOTH = Setting(7.5e-4, 1e-4, 30)                                    # the two candidates of REPORT.md, section 9


def parse_setting(text: str) -> Setting:
    """'7.5/1e-4/30' (training cost in bp / ridge / epochs) -> a Setting; the grid's own object when it is in the grid."""
    cost, ridge, epochs = text.split("/")
    for s in GRID:
        if np.isclose(s.cost_trade, float(cost) / 1e4) and np.isclose(s.ridge, float(ridge)) and s.epochs == int(epochs):
            return s
    return Setting(float(cost) / 1e4, float(ridge), int(epochs))


def parse_set(items: list[str]) -> dict:
    """['universe=wrds_broad', 'fill=none'] -> Config overrides with the fields' types."""
    types = {f.name: type(f.default) for f in dataclasses.fields(Config)}
    extra = {}
    for item in items:
        key, value = item.split("=", 1)
        extra[key] = (value.lower() in ("1", "true", "yes")) if types[key] is bool else types[key](value)
    return extra


def complete(setting: Setting, K: int, period: str, extra: dict | None = None, seeds: tuple[int, ...] = SEEDS) -> bool:
    first, last, train_years = PERIODS[period]
    folder = setting.run_dir(K, train_years, extra)
    return all((folder / f"seed{s}" / f"{y}.npz").exists() for s in seeds for y in range(first, last + 1))


def has_weights(setting: Setting, K: int, period: str, extra: dict | None = None, seeds: tuple[int, ...] = SEEDS) -> bool:
    first, last, train_years = PERIODS[period]
    folder = setting.run_dir(K, train_years, extra)
    return all((folder / f"seed{s}" / f"{y}_w.npz").exists() for s in seeds for y in range(first, last + 1))


def _panel(settings: tuple[Setting, ...], extra: dict | None):
    from afsa.panel import load_panel
    cfg = settings[0].config(8, 8, extra)
    return load_panel(cfg, "cuda")                                           # the month blocks for entry_rule history


def train(Ks: list[int], periods: list[str], settings: tuple[Setting, ...] = GRID, extra: dict | None = None,
          seeds: tuple[int, ...] = SEEDS, max_models: int | None = None) -> None:
    """max_models: models per GPU pass (afsa.run's default sizes passes from free memory)."""
    from afsa.run import run_config
    panel = _panel(settings, extra)
    for K in Ks:
        for setting in settings:
            for period in periods:
                first, last, train_years = PERIODS[period]
                run_config(setting.config(K, train_years, extra), list(seeds), list(range(first, last + 1)), panel, max_models)


def weights(Ks: list[int], periods: list[str], settings: tuple[Setting, ...] = GRID, extra: dict | None = None,
            seeds: tuple[int, ...] = SEEDS) -> None:
    from reeval_weights import recover
    panel = _panel(settings, extra)
    for K in Ks:
        for setting in settings:
            for period in periods:
                first, last, train_years = PERIODS[period]
                folder = setting.run_dir(K, train_years, extra)
                if complete(setting, K, period, extra, seeds):
                    n, worst = recover(folder, panel, years=list(range(first, last + 1)), tolerance=REPLAY_TOLERANCE)
                    if n:
                        print(f"{folder.name} {period}: {n} weight files written, largest difference {worst:.1e}", flush=True)


def listing(Ks: list[int], settings: tuple[Setting, ...] = GRID, extra: dict | None = None, seeds: tuple[int, ...] = SEEDS) -> None:
    print(f"{len(settings)} settings, seeds {list(seeds)}; each cell: test / validation / holdout complete (+ = yes), w = weights for all three")
    for setting in settings:
        cells = []
        for K in Ks:
            done = "".join("+" if complete(setting, K, p, extra, seeds) else "." for p in PERIODS)
            cells.append(f"K={K}: {done}{'w' if all(has_weights(setting, K, p, extra, seeds) for p in PERIODS) else ' '}")
        print(f"{setting.label:32s} " + "  ".join(cells))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", choices=("train", "weights"), help="train the missing runs, or write their daily weights")
    p.add_argument("--K", type=int, nargs="+", default=[8, 30])
    p.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    p.add_argument("--list", action="store_true", help="show the grid and what is on disk")
    p.add_argument("--only", nargs="+", default=None, metavar="BP/RIDGE/EPOCHS", help="only these settings of the grid, e.g. 7.5/1e-4/30")
    p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE", help="other Config fields for every run, e.g. universe=wrds_common fill=none")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="random seeds [default: 0 1 2]")
    p.add_argument("--max-models", type=int, default=None,
                   help="models trained together in one pass [default: from free GPU memory]. At K = 100 on the buffered list "
                        "(600 slots) the default of 21 spilled 2.5 GB into shared memory: 9.9 and 17.8 s a model in its first "
                        "two passes, against 5.9 s with 12 (2026-11-15)")
    args = p.parse_args()
    settings = tuple(parse_setting(t) for t in args.only) if args.only else GRID
    extra, seeds = parse_set(args.set), tuple(args.seeds)
    if args.list or args.action is None:
        listing(args.K, settings, extra, seeds)
    elif args.action == "train":
        train(args.K, args.periods, settings, extra, seeds, args.max_models)
    else:
        weights(args.K, args.periods, settings, extra, seeds)


if __name__ == "__main__":
    main()
