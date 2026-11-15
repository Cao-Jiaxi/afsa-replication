"""Daily portfolio weights of runs that were saved without them, from their saved models. Nothing is retrained.

    python analysis/gap/reeval_weights.py results/runs/attn_K008 results/runs/attn_K030
    python analysis/gap/reeval_weights.py results/runs/attn_K030 --years 2020 2021 --force

Every seed<s>/<year>.pt is evaluated on its test year exactly as afsa.run does (`afsa.train.evaluate`), and the
weights go to seed<s>/<year>_w.npz: `weights` [days x stocks] (float32; row i earns the return of dates[i]),
`permnos` and `dates`. The name does not match the pattern of the result files, so nothing else reads it by mistake.

Check: the recomputed gross return, turnover and short exposure must equal the stored ones to 1e-5, the agreement
the audit of analysis/gap/REPORT.md found between two float32 passes. If they do not, the script stops.
A run made from now on can save its weights directly: `python -m afsa.run ... --save-weights`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.config import WINDOW_RULES, Config  # noqa: E402
from afsa.model import AFSAModel  # noqa: E402
from afsa.panel import BlockPanel, Panel, load_panel, panel_folder  # noqa: E402
from afsa.run import window_folder  # noqa: E402
from afsa.train import Task, evaluate  # noqa: E402

TOLERANCE = 1e-5


def recover(run: Path, panel: Panel | BlockPanel | None, years: list[int] | None = None, force: bool = False, chunk: int = 36,
            device: str = "cuda", tolerance: float = TOLERANCE) -> tuple[int, float]:
    """Writes the missing <year>_w.npz files of one run. Returns how many were written and the largest difference
    between a recomputed and a stored daily figure. Models replayed in other groups than they were trained in can
    differ from the stored figures by about 1e-5 (float32 batching): callers that cannot match the groups pass a
    larger `tolerance`."""
    cfg = Config.load(run / "config.json")
    todo = []
    for seed_dir in sorted(run.glob("seed*"), key=lambda p: int(p.name[4:])):
        for f in sorted(seed_dir.glob("[0-9][0-9][0-9][0-9].pt")):
            year = int(f.stem)
            if (years is None or year in years) and (force or not (seed_dir / f"{year}_w.npz").exists()):
                todo.append((Task(year, int(seed_dir.name[4:])), seed_dir))
    per_window = cfg.universe in WINDOW_RULES
    if per_window:                                                           # one panel per test year
        groups = [[t for t in todo if t[0].test_year == y] for y in sorted({t[0].test_year for t in todo})]
    else:
        groups = [todo[lo:lo + chunk] for lo in range(0, len(todo), chunk)]
    worst = 0.0
    for group in groups:
        if per_window:
            panel = Panel(window_folder(cfg, group[0][0].test_year), device)
        states = [torch.load(folder / f"{task.test_year}.pt") for task, folder in group]
        model = AFSAModel.from_states(cfg, states, panel.device)
        for (task, folder), res in zip(group, evaluate(cfg, panel, model, [t for t, _ in group], keep_weights=True)):
            stored = np.load(folder / f"{task.test_year}.npz")
            diff = max(float(np.abs(res[k] - stored[k]).max()) for k in ("gross", "turnover", "short"))
            if not diff <= tolerance:
                raise SystemExit(f"{folder / str(task.test_year)}: the replay differs from the stored results by {diff:.2e} (limit {tolerance:.0e})")
            worst = max(worst, diff)
            np.savez_compressed(folder / f"{task.test_year}_w.npz", weights=res["weights"], permnos=res["permnos"], dates=res["dates"])
        del model
    return len(todo), worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run folders (results/runs/<name>)")
    p.add_argument("--years", type=int, nargs="+", default=None, help="test years [default: all on disk]")
    p.add_argument("--force", action="store_true", help="rewrite weights that are already there")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    panel = None
    for run in args.runs:
        cfg = Config.load(run / "config.json")
        folder = panel_folder(cfg)                                           # the month blocks for entry_rule history
        if cfg.universe not in WINDOW_RULES and (panel is None or panel.universe != folder):
            panel = None                                                     # release the previous sample first
            panel = load_panel(cfg, args.device)
        n, worst = recover(run, None if cfg.universe in WINDOW_RULES else panel, args.years, args.force, device=args.device)
        print(f"{run.name}: {n} weight files written" + (f", largest difference from the stored results {worst:.1e}" if n else ""), flush=True)


if __name__ == "__main__":
    main()
