"""Rolling out-of-sample experiment: 8-year training window, re-estimated every year, one folder per configuration.

    python -m afsa.run --K 8 30 --seeds 0 1 2                     # Table 2 rows, test years 1998-2021
    python -m afsa.run --K 30 --drop_groups past_returns           # a Table 3 row
    python -m afsa.run --K 8 --first-year 2022 --last-year 2025    # the extension
    python -m afsa.run --K 8 --epochs 5 --lr 0.001 --tag trial     # any model or training parameter can be changed
    python -m afsa.run --help                                      # all parameters

Each (test year, seed) pair is one model. Models are trained `--max-models` at a time in a single GPU pass (they
are independent: see afsa/model.py), and each is saved as soon as its batch finishes, so an interrupted run
resumes where it stopped. Results go to results/runs/<configuration>/seed<s>/<year>.npz (+ .pt for the model).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config import RESULTS_DIR, Config
from .metrics import compare_with_paper, summarize
from .panel import Panel
from .train import STATS, Task, evaluate, fit


def default_max_models(cfg: Config, device: torch.device) -> int:
    """How many models fit in one training pass: free GPU memory over the measured footprint of one model
    (tests/hardware/REPORT.md). On the CPU there is no gain from batching beyond a few models."""
    if device.type != "cuda":
        return 4
    free, _ = torch.cuda.mem_get_info(device)
    per_model = 30e6 + 6.5e6 * max(cfg.K, 8) * cfg.batch_days / 126           # bytes, with a safety margin
    if cfg.n_layers > 1:
        per_model += 1.6e9 * cfg.n_layers * cfg.d_model / 32 * cfg.batch_days / 126
    return int(max(1, min(96, 0.8 * free // per_model)))


def run_config(cfg: Config, seeds: list[int], years: list[int], panel: Panel, max_models: int | None = None,
               tag: str = "", save_weights: bool = False, verbose: bool = True) -> Path:
    run_dir = RESULTS_DIR / "runs" / (cfg.name() + (f"_{tag}" if tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.json").exists():
        if Config.load(run_dir / "config.json") != cfg:
            raise ValueError(f"{run_dir} already holds a run with a different configuration: use another --tag")
    else:
        cfg.save(run_dir / "config.json")
    todo = [Task(y, s) for s in seeds for y in years if not (run_dir / f"seed{s}" / f"{y}.npz").exists()]
    max_models = max_models or default_max_models(cfg, panel.device)
    if verbose:
        print(f"{run_dir.name}: {len(todo)} of {len(seeds) * len(years)} models to train, {max_models} at a time", flush=True)
    for lo in range(0, len(todo), max_models):
        chunk = todo[lo:lo + max_models]
        t0 = time.time()
        model, history = fit(cfg, panel, chunk)
        results = evaluate(cfg, panel, model, chunk, keep_weights=save_weights)
        for m, (task, res) in enumerate(zip(chunk, results)):
            folder = run_dir / f"seed{task.seed}"
            folder.mkdir(exist_ok=True)
            torch.save(model.member_state(m), folder / f"{task.test_year}.pt")
            arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
            np.savez(folder / f"{task.test_year}.npz", history=history[:, m], history_columns=np.array(STATS),
                     explained_var=res["explained_var"], **arrays)                      # written last: marks the task as done
        if verbose:
            last = history[-1].mean(0)
            print(f"  {lo + len(chunk):4d}/{len(todo)} models  {time.time() - t0:6.1f} s  "
                  f"last epoch (in-sample, batch mean): net Sharpe {last[1]:.2f}, explained variance {last[3]:.3f}, turnover {last[4]:.2f}",
                  flush=True)
        del model
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, nargs="+", default=[8], help="number(s) of attention factors; one run per value [default: 8]")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="random seeds [default: 0 1 2]")
    p.add_argument("--first-year", type=int, default=1998, help="first test year [default: 1998]")
    p.add_argument("--last-year", type=int, default=2021, help="last test year [default: 2021]")
    p.add_argument("--max-models", type=int, default=None, help="models trained together in one pass [default: from free GPU memory]")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="", help="suffix of the run folder, to keep trials apart")
    p.add_argument("--save-weights", action="store_true", help="also save the daily portfolio weights")
    Config.add_arguments(p, skip=("K",))
    args = p.parse_args()

    years = list(range(args.first_year, args.last_year + 1))
    panel = None
    for K in args.K:
        cfg = Config.from_args(args, K=K)
        if panel is None or panel.universe != cfg.universe:
            panel = Panel(cfg.universe, args.device)
        t0 = time.time()
        run_dir = run_config(cfg, args.seeds, years, panel, args.max_models, args.tag, args.save_weights)
        table = summarize(run_dir, args.first_year, args.last_year)
        print(f"\n{run_dir.name}: test years {args.first_year}-{args.last_year}, {time.time() - t0:.0f} s")
        print(table.drop(columns=["first", "last"]).astype(float).round(3).to_string())
        print("\nSeed average against the paper's Table 2 (1998-2021):")
        print(compare_with_paper(table, K), "\n", flush=True)


if __name__ == "__main__":
    main()
