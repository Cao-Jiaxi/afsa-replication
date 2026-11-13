"""One training process of a hardware test: trains models on 8-year windows and reports what it used.

    python tests/hardware/worker.py --K 8 --models 24 --report out.json            # 24 models in one batched pass
    python tests/hardware/worker.py --K 8 --models 6 --one-by-one --report out.json  # 6 models, one after another

Model i is trained on the 8 years before test year first_year + (offset + i) % 24, with seed (offset + i) // 24,
so consecutive offsets give the 24 windows of 1998-2021 and then further seeds. Run by run_tests.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

T_START = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psutil  # noqa: E402
import torch  # noqa: E402

from afsa.config import Config  # noqa: E402
from afsa.panel import Panel  # noqa: E402
from afsa.train import STATS, Task, evaluate, fit  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--models", type=int, default=1)
    p.add_argument("--offset", type=int, default=0, help="index of this process's first model")
    p.add_argument("--one-by-one", action="store_true", help="train the models one after another instead of together")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--first-year", type=int, default=1998)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = its default)")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    proc = psutil.Process()
    cfg = Config(K=args.K, epochs=args.epochs)
    tasks = [Task(args.first_year + (args.offset + i) % 24, (args.offset + i) // 24) for i in range(args.models)]
    cuda = args.device == "cuda"
    t_import = time.perf_counter() - T_START

    t0 = time.perf_counter()
    years = [t.test_year for t in tasks]
    panel = Panel(cfg.universe, args.device, min(years) - cfg.train_years, max(years))      # only the years these models need
    if cuda:
        torch.cuda.synchronize()
    t_panel = time.perf_counter() - t0
    vram_panel = torch.cuda.memory_allocated() / 2**20 if cuda else 0.0

    groups = [[t] for t in tasks] if args.one_by_one else [tasks]
    epoch_s, fit_s, eval_s, last = [], 0.0, 0.0, []
    peak = {"train_allocated": 0.0, "train_reserved": 0.0, "evaluate_allocated": 0.0}       # MiB, data included

    def note_peak(phase: str) -> None:
        if cuda:
            peak[f"{phase}_allocated"] = max(peak[f"{phase}_allocated"], torch.cuda.max_memory_allocated() / 2**20)
            if phase == "train":
                peak["train_reserved"] = max(peak["train_reserved"], torch.cuda.max_memory_reserved() / 2**20)
            torch.cuda.reset_peak_memory_stats()

    clock_start = time.time()                                                # wall-clock, to line up with the monitor
    cpu_start = sum(proc.cpu_times()[:2])
    for group in groups:
        marks = [time.perf_counter()]

        def progress(epoch, stats):
            if cuda:
                torch.cuda.synchronize()
            marks.append(time.perf_counter())

        if cuda:
            torch.cuda.reset_peak_memory_stats()
        model, history = fit(cfg, panel, group, progress)
        fit_s += marks[-1] - marks[0]
        epoch_s.append([b - a for a, b in zip(marks[:-1], marks[1:])])
        note_peak("train")
        t0 = time.perf_counter()
        results = evaluate(cfg, panel, model, group)
        if cuda:
            torch.cuda.synchronize()
        eval_s += time.perf_counter() - t0
        note_peak("evaluate")
        last.append(history[-1].mean(0).tolist())
        assert all(len(r["gross"]) > 200 for r in results)
        del model

    steady = [e for g in epoch_s for e in g[1:]] or [e for g in epoch_s for e in g]     # skip each fit's first epoch (warm-up)
    cpu_train = sum(proc.cpu_times()[:2]) - cpu_start                        # CPU seconds of this process while training
    clock_end = time.time()
    cpu = proc.cpu_times()
    mem = proc.memory_info()
    report = {
        "K": args.K, "models": args.models, "offset": args.offset, "one_by_one": args.one_by_one, "epochs": args.epochs,
        "device": args.device, "pid": os.getpid(), "test_years": years,
        "train_clock": [clock_start, clock_end],
        "cpu_cores_train": cpu_train / max(clock_end - clock_start, 1e-9),
        "seconds": {"import": t_import, "panel": t_panel, "fit": fit_s, "evaluate": eval_s, "total": time.perf_counter() - T_START,
                    "first_epoch": sum(g[0] for g in epoch_s) / len(epoch_s), "steady_epoch": sum(steady) / len(steady)},
        "vram_mb": {"panel": vram_panel, "train_peak_allocated": peak["train_allocated"], "evaluate_peak_allocated": peak["evaluate_allocated"],
                    "peak_reserved": max(peak["train_reserved"], torch.cuda.memory_reserved() / 2**20 if cuda else 0.0)},
        "ram_mb": {"peak_rss": getattr(mem, "peak_wset", mem.rss) / 2**20, "peak_commit": getattr(mem, "peak_pagefile", mem.vms) / 2**20},
        "cpu_seconds": cpu.user + cpu.system,
        "last_epoch": dict(zip(STATS, [sum(x) / len(last) for x in zip(*last)])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
