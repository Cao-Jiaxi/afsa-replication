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
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .config import RESULTS_DIR, ROOT, WINDOW_RULES, Config, data_folder
from .metrics import compare_with_paper, summarize
from .panel import BlockPanel, Panel, load_panel, panel_folder
from .train import STATS, Task, evaluate, fit


def window_folder(cfg: Config, year: int) -> str:
    """Folder (under Data/TrainingData) of the universe that is fixed for the window of test year `year`."""
    return f"{data_folder(cfg.universe, cfg.fill)}/{cfg.train_years}y/{year}"


def default_max_models(cfg: Config, device: torch.device) -> int:
    """How many models to train in one pass, from the measurements in tests/hardware/REPORT.md.

    One model reserves about 165 + 2.1 K MB of GPU memory during training (d_model = 32, batches of 126 days).
    Beyond 36 models the GPU is saturated and larger batches no longer train faster. On the month blocks (entry_rule
    history) every row of the 7 or 8 blocks a batch touches gets factors: peaks of 127, 168 and 379 MB per model for
    K = 8, 30 and 100 against 127, 148 and 271 on the day x slot arrays (measured 2026-11-14), so 150 + 3.5 K MB.
    """
    if device.type != "cuda":
        return 4
    free, _ = torch.cuda.mem_get_info(device)
    days = (cfg.batch_days + cfg.lookback + 1) / 157
    per_model = (165e6 * cfg.d_model / 32 + 2.1e6 * cfg.K) * days
    if cfg.entry_rule == "history":
        per_model = (150e6 * cfg.d_model / 32 + 3.5e6 * cfg.K) * days
    if cfg.n_layers > 1:                                                      # whole [.., lookback, d_model] sequences are kept
        per_model += 130e6 * cfg.lookback * (cfg.n_layers - 1) * cfg.d_model / 32 * days
    return int(max(1, min(36, 0.8 * free // per_model)))


def provenance() -> dict:
    """The code that is about to train: git commit, whether afsa/ differs from it, and a hash of afsa/*.py (which
    identifies the code also when the tree is not committed)."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    digest = hashlib.sha256()
    for f in sorted((ROOT / "afsa").glob("*.py")):
        digest.update(f.read_bytes().replace(b"\r\n", b"\n"))        # the same code hashes the same on any checkout
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git("rev-parse", "HEAD"),
            "afsa_differs_from_commit": bool(git("status", "--porcelain", "--", "afsa")), "afsa_sha256": digest.hexdigest()[:16]}


def run_config(cfg: Config, seeds: list[int], years: list[int], panel: Panel | BlockPanel | None, max_models: int | None = None,
               tag: str = "", save_weights: bool = False, verbose: bool = True, device: str = "cuda") -> Path:
    """Trains and evaluates the missing (test year, seed) pairs of one configuration. `panel` holds the whole
    sample of cfg.universe (`load_panel(cfg)`: the month blocks for entry_rule history); for a universe that is fixed
    per training window it is None, and each test year loads its own panel on `device` and trains its seeds in one pass."""
    run_dir = RESULTS_DIR / "runs" / (cfg.name() + (f"_{tag}" if tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.json").exists():
        if Config.load(run_dir / "config.json") != cfg:
            raise ValueError(f"{run_dir} already holds a run with a different configuration: use another --tag")
    else:
        cfg.save(run_dir / "config.json")
    todo = [Task(y, s) for s in seeds for y in years if not (run_dir / f"seed{s}" / f"{y}.npz").exists()]
    per_window = cfg.universe in WINDOW_RULES
    max_models = max_models or default_max_models(cfg, torch.device(device) if per_window else panel.device)
    if per_window:
        chunks = [c for c in ([t for t in todo if t.test_year == y] for y in years) if c]
    else:
        chunks = [todo[lo:lo + max_models] for lo in range(0, len(todo), max_models)]
    if todo:                                                                 # one entry per call that trains something
        log = run_dir / "provenance.json"
        entries = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(entries + [{**provenance(), "models": len(todo)}], indent=2))
    if verbose:
        how = "one test year at a time" if per_window else f"{max_models} at a time"
        print(f"{run_dir.name}: {len(todo)} of {len(seeds) * len(years)} models to train, {how}", flush=True)
    done = 0
    for chunk in chunks:
        t0 = time.time()
        if per_window:
            panel = Panel(window_folder(cfg, chunk[0].test_year), device)
        model, history = fit(cfg, panel, chunk)
        results = evaluate(cfg, panel, model, chunk, keep_weights=save_weights)
        for m, (task, res) in enumerate(zip(chunk, results)):
            folder = run_dir / f"seed{task.seed}"
            folder.mkdir(exist_ok=True)
            torch.save(model.member_state(m), folder / f"{task.test_year}.pt")
            arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
            np.savez(folder / f"{task.test_year}.npz", history=history[:, m], history_columns=np.array(STATS),
                     explained_var=res["explained_var"], **arrays)                      # written last: marks the task as done
        done += len(chunk)
        if verbose and (not per_window or done == len(todo) or chunk[0].test_year % 6 == 0):
            last = history[-1].mean(0)
            print(f"  {done:4d}/{len(todo)} models  {time.time() - t0:6.1f} s  "
                  f"last epoch (in-sample, batch mean): net Sharpe {last[1]:.2f}, explained variance {last[3]:.3f}, turnover {last[4]:.2f}",
                  flush=True)
        del model, results
        if panel.device.type == "cuda":
            torch.cuda.empty_cache()                                         # so the next pass starts from the same memory
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
        if cfg.universe in WINDOW_RULES:
            panel = None                                                     # one panel per test year, loaded by run_config
        elif panel is None or panel.universe != panel_folder(cfg):
            panel = None                                                     # release the previous sample first
            panel = load_panel(cfg, args.device)
        t0 = time.time()
        run_dir = run_config(cfg, args.seeds, years, panel, args.max_models, args.tag, args.save_weights, device=args.device)
        table = summarize(run_dir, args.first_year, args.last_year)
        print(f"\n{run_dir.name}: test years {args.first_year}-{args.last_year}, {time.time() - t0:.0f} s")
        print(table.drop(columns=["first", "last"]).astype(float).round(3).to_string())
        if (args.first_year, args.last_year) == (1998, 2021):               # the paper's out-of-sample period
            print("\nSeed average against the paper's Table 2 (1998-2021):")
            print(compare_with_paper(table, K))
        print(flush=True)


if __name__ == "__main__":
    main()
