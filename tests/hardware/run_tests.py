"""Hardware tests: what one 8-year training window costs on this machine, and the fastest way to train many.

    python tests/hardware/run_tests.py                         # every scenario not yet on disk (about 20 minutes in all)
    python tests/hardware/run_tests.py --only one_window cpu   # the groups named
    python tests/hardware/run_tests.py --rerun                 # run again what is already on disk
    python tests/hardware/run_tests.py --summary-only          # rebuild RESULTS.md from the saved JSON files

Each scenario starts one or more worker.py processes and samples the GPU (nvidia-smi) and the system (psutil)
while they run. One JSON file per scenario goes to tests/hardware/results/ (with the workers' logs), and
tests/hardware/RESULTS.md tabulates them. REPORT.md is the written conclusion.

Groups of scenarios (all with the paper's settings: 30 epochs, batches of 126 days):
    one_window  one model on one 8-year window, alone, for K = 1, 8, 30, 100
    cpu         the same for K = 8 without the GPU (2 epochs, extrapolated)
    processes   the 24 windows of 1998-2021 (K = 8) split over 1..12 processes, each training one model at a time
    batched     one process training M models in the same GPU pass, for K = 8, 30, 100
    split       48 models (K = 8) as 1 x 48, 2 x 24 and 3 x 16 batched processes
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORKER = HERE / "worker.py"
GPU_QUERY = "utilization.gpu,memory.used,power.draw,temperature.gpu"


def gpu_now() -> list[float]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY},memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [float(x) for x in out.strip().split(",")]


class Monitor:
    """Samples GPU utilization, GPU memory, power, CPU load and RAM every `interval` seconds in the background."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.gpu: list[tuple[float, float, float, float, float]] = []       # time, util %, memory MiB, W, deg C
        self.cpu: list[tuple[float, float, float]] = []                     # time, system CPU %, RAM used MiB
        self._stop = threading.Event()

    def start(self) -> None:
        self.gpu_base_mb = gpu_now()[1]
        self.ram_base_mb = psutil.virtual_memory().used / 2**20
        psutil.cpu_percent(None)
        self._smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
                                      "-lms", str(int(self.interval * 1000))], stdout=subprocess.PIPE, text=True)
        self._threads = [threading.Thread(target=self._read_gpu, daemon=True), threading.Thread(target=self._read_cpu, daemon=True)]
        for t in self._threads:
            t.start()

    def _read_gpu(self) -> None:
        for line in self._smi.stdout:
            try:
                self.gpu.append((time.time(), *[float(x) for x in line.strip().split(",")]))
            except ValueError:
                pass

    def _read_cpu(self) -> None:
        while not self._stop.wait(self.interval):
            self.cpu.append((time.time(), psutil.cpu_percent(None), psutil.virtual_memory().used / 2**20))

    def stop(self) -> None:
        self._stop.set()
        self._smi.terminate()
        for t in self._threads:
            t.join(timeout=2)
        # Other programs (desktop, browser, editor) also hold GPU memory, and their share drifts by up to 1 GB.
        # Taking the larger of the readings before and after keeps that drift out of what the test "added".
        self.gpu_other_mb = [self.gpu_base_mb, gpu_now()[1]]
        self.gpu_base_mb = max(self.gpu_other_mb)

    def summary(self, start: float, end: float) -> dict:
        """Statistics of the samples taken between two wall-clock times (the training phase)."""
        g = [s for s in self.gpu if start <= s[0] <= end] or self.gpu[-1:]
        c = [s for s in self.cpu if start <= s[0] <= end] or self.cpu[-1:]
        mean = lambda xs: sum(xs) / max(len(xs), 1)                          # noqa: E731
        n = psutil.cpu_count()
        return {
            "samples": len(g),
            "gpu_util_mean": mean([s[1] for s in g]), "gpu_util_max": max([s[1] for s in g], default=0.0),
            "gpu_mem_base_mb": self.gpu_base_mb, "gpu_mem_other_before_after_mb": getattr(self, "gpu_other_mb", None),
            "gpu_mem_peak_mb": max([s[2] for s in self.gpu], default=0.0),   # whole run: includes the other programs
            "gpu_mem_added_mb": max([s[2] for s in self.gpu], default=0.0) - self.gpu_base_mb,
            "gpu_power_mean_w": mean([s[3] for s in g]), "gpu_temp_max_c": max([s[4] for s in g], default=0.0),
            "cpu_percent_mean": mean([s[1] for s in c]), "cpu_threads_busy_mean": mean([s[1] for s in c]) * n / 100,
            "cpu_percent_max": max([s[1] for s in c], default=0.0),
            "ram_added_peak_mb": max([s[2] for s in self.cpu], default=self.ram_base_mb) - self.ram_base_mb,
        }


def scenarios() -> dict[str, dict]:
    S: dict[str, dict] = {}

    def add(name, group, workers, **extra):
        S[name] = {"group": group, "workers": [[str(a) for a in w] for w in workers], **extra}

    for K in (8, 1, 30, 100):
        add(f"one_window_K{K:03d}", "one_window", [["--K", K, "--models", 1]])
    add("cpu_K008", "cpu", [["--K", 8, "--models", 1, "--device", "cpu", "--epochs", 2]])
    for P in (1, 2, 3, 4, 6, 8, 12):
        add(f"processes_{P:02d}", "processes",
            [["--K", 8, "--models", 24 // P, "--offset", i * (24 // P), "--one-by-one"] for i in range(P)])
    for K, sizes in ((8, (2, 4, 8, 12, 24, 36, 48, 72, 96)), (30, (24, 48, 72)), (100, (8, 16, 24, 36))):
        for M in sizes:
            add(f"batched_K{K:03d}_M{M:03d}", "batched", [["--K", K, "--models", M]])
    add("split_K008_1x48", "split", [["--K", 8, "--models", 48]])
    add("split_K008_2x24", "split", [["--K", 8, "--models", 24, "--offset", 24 * i] for i in range(2)])
    add("split_K008_3x16", "split", [["--K", 8, "--models", 16, "--offset", 16 * i] for i in range(3)])
    return S


def headroom() -> tuple[float, float, float]:
    """Free GPU memory, available RAM and available commit (RAM + page file), in MiB."""
    g = gpu_now()
    ram = psutil.virtual_memory().available / 2**20
    commit = ram + psutil.swap_memory().free / 2**20
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                           "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            commit = status.ullAvailPageFile / 2**20
    return g[4] - g[1], ram, commit


def run_scenario(name: str, spec: dict, need: dict | None = None) -> dict:
    out = RESULTS / f"{name}.json"
    free_vram, free_ram, free_commit = headroom()
    if need and (need["vram_mb"] > 0.92 * free_vram or need["ram_mb"] > 0.9 * free_ram or need["commit_mb"] > 0.95 * free_commit):
        result = {"name": name, **spec, "skipped": f"needs about {need['vram_mb'] / 1024:.1f} GB of GPU memory, {need['ram_mb'] / 1024:.1f} GB of RAM "
                                                   f"and {need['commit_mb'] / 1024:.1f} GB of commit; {free_vram / 1024:.1f}, {free_ram / 1024:.1f} "
                                                   f"and {free_commit / 1024:.1f} GB are free"}
        out.write_text(json.dumps(result, indent=2))
        print(f"{name:24s} SKIPPED: {result['skipped']}", flush=True)
        return result
    logs = RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    mon = Monitor()
    mon.start()
    time.sleep(1.0)
    t0 = time.time()
    procs, reports = [], []
    for i, args in enumerate(spec["workers"]):
        rep = logs / f"{name}_{i}.json"
        rep.unlink(missing_ok=True)
        reports.append(rep)
        log = open(logs / f"{name}_{i}.log", "w")
        procs.append((subprocess.Popen([sys.executable, str(WORKER), *args, "--report", str(rep)], stdout=log, stderr=subprocess.STDOUT), log))
    for p, log in procs:
        p.wait()
        log.close()
    wall = time.time() - t0
    time.sleep(0.5)
    mon.stop()
    done = [json.loads(r.read_text()) for r in reports if r.exists()]
    result = {"name": name, **spec, "wall_seconds": wall, "n_workers": len(procs), "failed_workers": len(procs) - len(done)}
    if done:
        start, end = max(w["train_clock"][0] for w in done), min(w["train_clock"][1] for w in done)
        models = sum(w["models"] for w in done)
        result.update({
            "models": models,
            "train_seconds": max(w["train_clock"][1] for w in done) - min(w["train_clock"][0] for w in done),
            "models_per_minute_wall": models / wall * 60,
            "system": mon.summary(start, end),
            "workers_sum": {"peak_rss_mb": sum(w["ram_mb"]["peak_rss"] for w in done), "peak_commit_mb": sum(w["ram_mb"]["peak_commit"] for w in done),
                            "cpu_seconds": sum(w["cpu_seconds"] for w in done), "cpu_cores_train": sum(w["cpu_cores_train"] for w in done),
                            "vram_peak_reserved_mb": sum(w["vram_mb"]["peak_reserved"] for w in done)},
            "worker_reports": done,
        })
        result["models_per_minute_train"] = models / result["train_seconds"] * 60
        s = result["system"]
        print(f"{name:24s} wall {wall:6.1f} s  train {result['train_seconds']:6.1f} s  {result['models_per_minute_train']:6.1f} models/min  "
              f"GPU {s['gpu_util_mean']:3.0f}%  VRAM +{s['gpu_mem_added_mb'] / 1024:4.1f} GB  RAM +{s['ram_added_peak_mb'] / 1024:4.1f} GB  "
              f"CPU {result['workers_sum']['cpu_cores_train']:4.1f} cores" + (f"  ({result['failed_workers']} FAILED)" if result["failed_workers"] else ""), flush=True)
    else:
        print(f"{name:24s} FAILED: see {logs / (name + '_0.log')}", flush=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def projected_need(name: str, spec: dict, results: dict[str, dict]) -> dict | None:
    """Memory a scenario should need, from scenarios already run, so that one that cannot fit is skipped."""
    n = len(spec["workers"])
    if spec["group"] == "processes" and n > 1:
        r = results.get("processes_01")
        if not r or "system" not in r:
            return None
        ws = r["workers_sum"]
        return {"vram_mb": n * r["system"]["gpu_mem_added_mb"], "ram_mb": n * ws["peak_rss_mb"], "commit_mb": n * ws["peak_commit_mb"]}
    if spec["group"] in ("batched", "split"):
        K, M = int(spec["workers"][0][1]), int(spec["workers"][0][3])
        prior = sorted((r for r in results.values() if r.get("group") == "batched" and "system" in r and int(r["workers"][0][1]) == K),
                       key=lambda r: r["models"])
        if not prior:
            return None

        def at(value) -> float:
            """Straight line through the two largest batches already measured, evaluated at M models."""
            (m0, v0), (m1, v1) = [(r["models"], value(r)) for r in prior[-2:]] if len(prior) > 1 else [(0, 0.0), (prior[0]["models"], value(prior[0]))]
            return v1 + (v1 - v0) / (m1 - m0) * (M - m1)

        return {"vram_mb": n * at(lambda r: r["system"]["gpu_mem_added_mb"]), "ram_mb": n * at(lambda r: r["workers_sum"]["peak_rss_mb"]),
                "commit_mb": n * at(lambda r: r["workers_sum"]["peak_commit_mb"])}
    return None


def load_results() -> dict[str, dict]:
    return {f.stem: json.loads(f.read_text()) for f in sorted(RESULTS.glob("*.json"))}


def write_summary(results: dict[str, dict]) -> str:
    """Markdown tables of every scenario on disk."""
    g = gpu_now()
    lines = ["# Hardware test results", "",
             f"Generated by `tests/hardware/run_tests.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
             f"GPU memory in use by other programs when the summary was written: {g[1] / 1024:.1f} of {g[4] / 1024:.1f} GB.", ""]

    def table(title, note, header, rows):
        lines.extend([f"## {title}", "", note, "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"])
        lines.extend("| " + " | ".join(r) + " |" for r in rows)
        lines.append("")

    def ok(group):
        return [r for r in results.values() if r.get("group") == group and "system" in r]

    def skipped(group):
        return [f"`{r['name']}` was skipped: {r['skipped']}." for r in results.values() if r.get("group") == group and "skipped" in r] + \
               [f"`{r['name']}`: {r['failed_workers']} of {r['n_workers']} processes failed." for r in results.values()
                if r.get("group") == group and r.get("failed_workers")]

    rows = []
    for r in sorted(ok("one_window") + ok("cpu"), key=lambda r: (r["worker_reports"][0]["device"], r["worker_reports"][0]["K"])):
        w, s = r["worker_reports"][0], r["system"]
        sec = w["seconds"]
        full = sec["first_epoch"] + sec["steady_epoch"] * 29                 # 30 epochs, also when fewer were run
        rows.append([w["device"], str(w["K"]), f"{sec['import'] + sec['panel']:.1f}", f"{full:.1f}" + ("" if w["epochs"] == 30 else " (est.)"),
                     f"{sec['steady_epoch']:.2f}", f"{sec['evaluate']:.2f}", f"{s['gpu_util_mean']:.0f} / {s['gpu_util_max']:.0f}",
                     f"{max(w['vram_mb']['train_peak_allocated'] - w['vram_mb']['panel'], 0):.0f}",
                     f"{max(w['vram_mb']['evaluate_peak_allocated'] - w['vram_mb']['panel'], 0):.0f}",
                     f"{w['vram_mb']['panel']:.0f}", f"{s['gpu_mem_added_mb']:.0f}",
                     f"{w['ram_mb']['peak_rss'] / 1024:.1f}", f"{w['ram_mb']['peak_commit'] / 1024:.1f}", f"{r['workers_sum']['cpu_cores_train']:.1f}"])
    table("One 8-year window, alone",
          "One model, 30 epochs of 16 batches. *Start-up* is importing PyTorch, creating the CUDA context and loading the data. "
          "*GPU %* is nvidia-smi's utilization during training (share of time the GPU was busy, mean / max). "
          "*Training*, *evaluation* and *data* are PyTorch's peak allocations (the first two without the data); *process total* is what "
          "nvidia-smi saw the process add, CUDA context and PyTorch's cache included. "
          "*Commit* is the virtual memory Windows reserves for the process. *CPU cores* is the CPU time of the training "
          "processes divided by the training time (1.0 = one core fully busy).",
          ["Device", "K", "Start-up s", "Train s", "s / epoch", "Evaluate s", "GPU % mean / max", "VRAM training MB", "VRAM evaluation MB",
           "VRAM data MB", "VRAM process total MB", "RAM GB", "Commit GB", "CPU cores"], rows)

    base = results.get("processes_01", {}).get("train_seconds")
    rows = []
    for r in sorted(ok("processes"), key=lambda r: r["n_workers"]):
        s, ws = r["system"], r["workers_sum"]
        rows.append([str(r["n_workers"]), str(r["models"] // r["n_workers"]), f"{r['wall_seconds']:.0f}", f"{r['train_seconds']:.0f}",
                     f"{r['models_per_minute_train']:.1f}", f"{base / r['train_seconds']:.2f}x" if base else "", f"{s['gpu_util_mean']:.0f}",
                     f"{s['gpu_mem_added_mb'] / 1024:.1f}", f"{ws['peak_rss_mb'] / 1024:.1f}", f"{ws['peak_commit_mb'] / 1024:.1f}",
                     f"{r['workers_sum']['cpu_cores_train']:.1f}"])
    table("24 windows as parallel processes (K = 8)",
          "The 24 training windows of 1998-2021, one seed, split evenly over P processes; each process trains its models one after "
          "another. *Wall* includes start-up, *train* is from the first process starting to train until the last one finishes. "
          + " ".join(skipped("processes")),
          ["Processes", "Models each", "Wall s", "Train s", "Models / min", "Speed-up", "GPU %", "VRAM GB", "RAM GB", "Commit GB", "CPU cores"], rows)

    one = {r["worker_reports"][0]["K"]: r["worker_reports"][0]["seconds"] for r in ok("one_window")}
    rows = []
    for r in sorted(ok("batched"), key=lambda r: (r["worker_reports"][0]["K"], r["models"])):
        w, s = r["worker_reports"][0], r["system"]
        sec, K, M = w["seconds"], w["K"], r["models"]
        alone = one.get(K, {}).get("steady_epoch")
        rows.append([str(K), str(M), f"{sec['fit']:.0f}", f"{sec['steady_epoch']:.2f}", f"{sec['steady_epoch'] / 16 / M * 1000:.2f}",
                     f"{M / sec['fit'] * 60:.0f}", f"{alone * M / sec['steady_epoch']:.1f}x" if alone else "", f"{s['gpu_util_mean']:.0f}",
                     f"{(w['vram_mb']['train_peak_allocated'] - w['vram_mb']['panel']) / M:.0f}", f"{w['vram_mb']['peak_reserved'] / 1024:.1f}",
                     f"{s['gpu_mem_added_mb'] / 1024:.1f}", f"{w['ram_mb']['peak_rss'] / 1024:.1f}", f"{r['workers_sum']['cpu_cores_train']:.1f}"])
    table("M models in one batched GPU pass",
          "One process trains M models (different windows and seeds) in the same passes. *ms / model-step* is the time of one "
          "optimizer step divided by M. *Speed-up* compares the time per epoch with M times that of one model alone. *VRAM per model* is "
          "PyTorch's peak allocation during training, without the data, divided by M; *VRAM PyTorch* is what it reserved in all, data "
          "and cache included. "
          + " ".join(skipped("batched")),
          ["K", "M", "Train s", "s / epoch", "ms / model-step", "Models / min", "Speed-up", "GPU %", "VRAM per model MB",
           "VRAM PyTorch GB", "VRAM process total GB", "RAM GB", "CPU cores"], rows)

    rows = []
    for r in sorted(ok("split"), key=lambda r: r["n_workers"]):
        s, ws = r["system"], r["workers_sum"]
        rows.append([f"{r['n_workers']} x {r['models'] // r['n_workers']}", f"{r['wall_seconds']:.0f}", f"{r['train_seconds']:.0f}",
                     f"{r['models_per_minute_train']:.0f}", f"{s['gpu_util_mean']:.0f}", f"{s['gpu_mem_added_mb'] / 1024:.1f}",
                     f"{ws['peak_rss_mb'] / 1024:.1f}", f"{r['workers_sum']['cpu_cores_train']:.1f}"])
    table("48 models (K = 8): one batched process or several",
          "The same 48 models (24 windows x 2 seeds) as one batch of 48, two processes of 24 or three of 16. " + " ".join(skipped("split")),
          ["Processes x models", "Wall s", "Train s", "Models / min", "GPU %", "VRAM GB", "RAM GB", "CPU cores"], rows)

    text = "\n".join(lines)
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None, help="groups or scenario names to run")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--rerun", action="store_true", help="run scenarios again even if their result file exists")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    results = load_results()
    if not args.summary_only:
        for name, spec in scenarios().items():
            if args.only and spec["group"] not in args.only and name not in args.only:
                continue
            if name in results and not args.rerun and ("system" in results[name]):
                continue
            results[name] = run_scenario(name, spec, projected_need(name, spec, results))
            time.sleep(2.0)                                                  # let the GPU memory settle
    print(write_summary(load_results()))


if __name__ == "__main__":
    main()
