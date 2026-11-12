"""Estimation: batches of consecutive days, the training loop for M models at once, and out-of-sample evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .config import Config
from .model import AFSAModel, explained_variance
from .panel import Panel

STATS = ("objective", "sharpe_net", "sharpe_gross", "explained_var", "turnover", "short")


@dataclass(frozen=True)
class Task:
    """One model to estimate: one seed, trained on the `train_years` calendar years before `test_year`."""
    test_year: int
    seed: int


def streams(task: Task) -> tuple[int, int, int]:
    """Independent seeds for a task's initialisation, batch order and dropout. They depend on the task only, so a
    model is trained the same way whichever other models share its batch."""
    return tuple(int(s) for s in np.random.SeedSequence([task.seed, task.test_year]).generate_state(3))


def train_span(cfg: Config, panel: Panel, task: Task) -> tuple[int, int]:
    """Loss days [start, stop) of a task. Each needs lookback + 1 earlier days, which only binds at the sample start."""
    first_year = task.test_year - cfg.train_years
    start, stop = panel.year_span(first_year, task.test_year - 1)
    if stop - start < cfg.batch_days or panel.years[start] != first_year or panel.years[stop - 1] != task.test_year - 1:
        raise ValueError(f"the sample does not cover the {cfg.train_years} years before {task.test_year}")
    start = max(start, cfg.lookback + 1)
    if not panel.has_days(start - cfg.lookback - 1, stop):
        raise ValueError(f"the panel in memory does not hold the training days of test year {task.test_year}")
    return start, stop


def batch_starts(rng: np.random.Generator, start: int, stop: int, cfg: Config) -> np.ndarray:
    """First loss day of each batch of one epoch: `steps_per_epoch` blocks of `batch_days` consecutive days in
    random order. The blocks tile [start, stop) from a random offset; if the window is a few days too short for
    that, they are spread evenly and overlap slightly."""
    nb, B, n = cfg.steps_per_epoch, cfg.batch_days, stop - start
    slack = n - nb * B
    if slack >= 0:
        first = start + rng.integers(0, slack + 1) + B * np.arange(nb)
    else:
        first = start + np.round(np.linspace(0, n - B, nb)).astype(np.int64)
    return rng.permutation(first)


def fit(cfg: Config, panel: Panel, tasks: list[Task], progress: Callable[[int, np.ndarray], None] | None = None,
        dtype: torch.dtype = torch.float32) -> tuple[AFSAModel, np.ndarray]:
    """Trains one model per task, all in the same passes. Returns the models and the training history
    [epochs, M, len(STATS)] (epoch averages over batches)."""
    device, L, B = panel.device, cfg.lookback, cfg.batch_days
    spans = [train_span(cfg, panel, t) for t in tasks]
    seeds = [streams(t) for t in tasks]
    model = AFSAModel(cfg, [s[0] for s in seeds], device, dtype)

    # The medians and rf are standardised with the statistics of each model's own training window.
    lev = panel.levels.double()
    stat = [(lev[a - panel.offset:b - panel.offset].mean(0), lev[a - panel.offset:b - panel.offset].std(0, correction=0))
            for a, b in (panel.year_span(t.test_year - cfg.train_years, t.test_year - 1) for t in tasks)]
    model.set_level_stats(torch.stack([s[0] for s in stat]), torch.stack([s[1] for s in stat]))

    opt = torch.optim.AdamW(model.param_groups(cfg.weight_decay), lr=cfg.lr, fused=device.type == "cuda")
    rngs = [np.random.default_rng(s[1]) for s in seeds]
    gens = [torch.Generator(device=device).manual_seed(s[2]) for s in seeds]
    offsets = torch.arange(-(L + 1), B, device=device)                      # lookback rows, the row before, the loss rows
    history = np.zeros((cfg.epochs, len(tasks), len(STATS)))
    model.train()
    for epoch in range(cfg.epochs):
        starts = torch.as_tensor(np.stack([batch_starts(r, a, b, cfg) for r, (a, b) in zip(rngs, spans)]), device=device)
        acc = torch.zeros(len(tasks), len(STATS), device=device)
        for i in range(starts.shape[1]):
            out = model(panel.batch(starts[:, i:i + 1] + offsets, L), train=True, gens=gens)
            objective, parts = model.objective(out)
            opt.zero_grad(set_to_none=True)
            (-objective.sum()).backward()                                    # the models share no parameter
            opt.step()
            acc += torch.stack([parts[k].detach() for k in STATS], dim=1)
        history[epoch] = (acc / starts.shape[1]).cpu().numpy()
        if progress is not None:
            progress(epoch, history[epoch])
    return model.eval(), history


@torch.no_grad()
def evaluate(cfg: Config, panel: Panel, model: AFSAModel, tasks: list[Task], keep_weights: bool = False,
             max_cells: float = 4e8) -> list[dict]:
    """Out-of-sample results of each model on its test year. One dict per task with the daily gross return,
    turnover and short exposure, and the weights needed to join consecutive years.

    turnover[0] is measured against the weights the same model would have held the day before; `metrics.stitch`
    replaces it with the trade from the previous year's model.
    """
    L, device = cfg.lookback, panel.device
    spans = [panel.year_span(t.test_year, t.test_year) for t in tasks]
    for (s, e), t in zip(spans, tasks):
        if e <= s or not panel.has_days(s - L - 1, e):
            raise ValueError(f"the panel in memory does not hold test year {t.test_year}")
    D = max(e - s for s, e in spans)
    per_model = (L + 1 + D) * 800 * max(cfg.K, cfg.d_model)                  # rough size of the largest tensor
    step = max(1, int(max_cells // per_model))
    results: list[dict] = []
    for lo in range(0, len(tasks), step):
        idx = list(range(lo, min(lo + step, len(tasks))))
        sub = model if len(idx) == len(tasks) else model.select(idx)
        first = torch.tensor([spans[m][0] for m in idx], device=device)
        days = (first[:, None] + torch.arange(-(L + 1), D, device=device)).clamp_max(panel.loaded[1] - 1)
        batch = panel.batch(days, L)
        out = sub.eval()(batch)
        for j, m in enumerate(idx):
            s, e = spans[m]
            n, u = e - s, int(batch.n_cols[j])
            w = out["w"][j, : n + 1, :u].double().cpu().numpy()             # row 0 = the day before the test year
            own = slice(L + 1, L + 1 + n)                                    # this model's test days (years differ in length)
            ev = explained_variance(out["eps"][j:j + 1, own], batch.R[j:j + 1, own], batch.member[j:j + 1, own], cfg.ev_min_days)
            res = {
                "test_year": tasks[m].test_year, "seed": tasks[m].seed,
                "dates": panel.dates[s:e],
                "gross": out["ret"][j, :n].double().cpu().numpy(),
                "turnover": out["turnover"][j, :n].double().cpu().numpy(),
                "short": out["short"][j, :n].double().cpu().numpy(),
                "explained_var": float(ev),
                "permnos": panel.permnos[batch.cols[j, :u].cpu().numpy()],
                "w_first": w[1], "w_last": w[n],
            }
            if keep_weights:
                res["weights"] = w[1:].astype(np.float32)
            results.append(res)
    return results
