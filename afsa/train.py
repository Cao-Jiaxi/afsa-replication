"""Estimation: batches of consecutive days, the training loop for M models at once, and out-of-sample evaluation.

Every function takes either sample: a `Panel` (day x slot arrays) or a `BlockPanel` (month blocks, entry_rule
history). Both index days by their position in the calendar and serve the same batches of days.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .config import Config
from .model import AFSAModel, explained_variance
from .panel import BlockPanel, Panel

STATS = ("objective", "sharpe_net", "sharpe_gross", "explained_var", "turnover", "short")


@dataclass(frozen=True)
class Task:
    """One model to estimate: one seed, trained on the `train_years` calendar years before `test_year`."""
    test_year: int
    seed: int


def streams(task: Task) -> tuple[int, int, int]:
    """Independent seeds for a task's initialization, batch order and dropout. They depend on the task only, so a
    model is trained the same way whichever other models share its batch."""
    return tuple(int(s) for s in np.random.SeedSequence([task.seed, task.test_year]).generate_state(3))


def train_span(cfg: Config, panel: Panel | BlockPanel, task: Task) -> tuple[int, int]:
    """Loss days [start, stop) of a task. Each needs lookback + 1 earlier days, which only binds at the sample start.

    The sample must hold every training year from its first trading day. A sample that starts inside the first
    training year is accepted only if it starts in the first week of January (no year's first trading day is later
    than 4 January). The arrays of afsa.build_wrds start on 1989-11-01 to give the first loss day, 1990-01-02, its
    lookback: those 41 days of 1989 are not a training year."""
    first_year = task.test_year - cfg.train_years
    start, stop = panel.year_span(first_year, task.test_year - 1)
    if stop - start < cfg.batch_days or panel.years[start] != first_year or panel.years[stop - 1] != task.test_year - 1:
        raise ValueError(f"the sample does not cover the {cfg.train_years} years before {task.test_year}")
    if start == 0 and panel.dates[0] > np.datetime64(f"{first_year}-01-07"):
        raise ValueError(f"the sample does not cover the {cfg.train_years} years before {task.test_year}: "
                         f"it starts on {panel.dates[0]}, inside {first_year}")
    start = max(start, cfg.lookback + 1)
    if not panel.covers(start - 1, stop, cfg.lookback):                       # the day before the first loss day too
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


def fit(cfg: Config, panel: Panel | BlockPanel, tasks: list[Task], progress: Callable[[int, np.ndarray], None] | None = None,
        dtype: torch.dtype = torch.float32) -> tuple[AFSAModel, np.ndarray]:
    """Trains one model per task, all in the same passes. Returns the models and the training history
    [epochs, M, len(STATS)] (epoch averages over batches)."""
    device, L, B = panel.device, cfg.lookback, cfg.batch_days
    spans = [train_span(cfg, panel, t) for t in tasks]
    seeds = [streams(t) for t in tasks]
    model = AFSAModel(cfg, [s[0] for s in seeds], device, dtype)

    # The medians and rf are standardized with the statistics of each model's own training window.
    lev = panel.levels.double()
    stat = [(lev[a - panel.offset:b - panel.offset].mean(0), lev[a - panel.offset:b - panel.offset].std(0, correction=0))
            for a, b in (panel.year_span(t.test_year - cfg.train_years, t.test_year - 1) for t in tasks)]
    model.set_level_stats(torch.stack([s[0] for s in stat]), torch.stack([s[1] for s in stat]))

    opt = torch.optim.AdamW(model.param_groups(cfg.weight_decay), lr=cfg.lr, fused=device.type == "cuda")
    schedule = None
    if cfg.lr_schedule == "cosine":                                          # one half-cosine from lr to 0 over all the steps
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * cfg.steps_per_epoch)
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
            if schedule is not None:
                schedule.step()
            acc += torch.stack([parts[k].detach() for k in STATS], dim=1)
        history[epoch] = (acc / starts.shape[1]).cpu().numpy()
        if progress is not None:
            progress(epoch, history[epoch])
    for task, last in zip(tasks, history[-1]):
        if not np.isfinite(last).all():                                      # the other models of the batch are unaffected
            warnings.warn(f"training diverged for test year {task.test_year}, seed {task.seed}: its results are not finite")
    return model.eval(), history


@torch.no_grad()
def evaluate(cfg: Config, panel: Panel | BlockPanel, model: AFSAModel, tasks: list[Task], keep_weights: bool = False,
             max_bytes: float = 1e9) -> list[dict]:
    """Out-of-sample results of each model on its test year. One dict per task with the daily gross return,
    turnover and short exposure, and the weights needed to join consecutive years.

    turnover[0] is measured against the weights the same model would have held the day before; `metrics.stitch`
    replaces it with the trade from the previous year's model. The models are evaluated a few at a time so that
    the pass stays within `max_bytes` of memory.
    """
    L, device = cfg.lookback, panel.device
    spans = [panel.year_span(t.test_year, t.test_year) for t in tasks]
    for (s, e), t in zip(spans, tasks):
        if e <= s or not panel.covers(s - 1, e, L):
            raise ValueError(f"the panel in memory does not hold test year {t.test_year}")
    D = max(e - s for s, e in spans)
    step = max(1, int(max_bytes // panel.pass_bytes(cfg, D)))
    if device.type == "cuda":
        torch.cuda.empty_cache()                                             # hand back what training had cached
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
            w = out["book"][j, : n + 1, :u].double().cpu().numpy()          # row 0 = the day before the test year
            own = slice(1, 1 + n)                                            # this model's test days (years differ in length)
            ev = explained_variance(out["eps_d"][j:j + 1, own], out["R_d"][j:j + 1, own], out["member_d"][j:j + 1, own], cfg.ev_min_days)
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
            if keep_weights:                                                 # the target; with the trading layer also the book
                res["weights"] = out["w"][j, 1: n + 1, :u].float().cpu().numpy()
                if cfg.band > 0 or cfg.band_eta < 1 or cfg.trade_head:
                    res["book"] = w[1:].astype(np.float32)
                if cfg.trade_head:                                           # and the learned rule's band width and speed
                    res["delta"] = out["delta"][j, 1: n + 1, :u].float().cpu().numpy()
                    res["eta"] = out["eta"][j, 1: n + 1, :u].float().cpu().numpy()
                if cfg.trade_state:                                          # and what a book carried across years needs:
                    res["dial_a"] = out["dial_a"][j, 1: n + 1, :u].float().cpu().numpy()   # the logits before the view
                    res["dial_k"] = out["dial_k"][j, 1: n + 1, :u].float().cpu().numpy()   # of the book,
                    res["signal"] = out["has"][j, 1: n + 1, :u].cpu().numpy()             # who has a signal,
                    res["dial_v"] = torch.stack([sub.delta_v[j], sub.eta_v[j]]).float().cpu().numpy()   # the weights on the view [2, 3]
            results.append(res)
    return results
