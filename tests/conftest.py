"""Shared fixtures: a small synthetic panel (no licensed data needed) and a matching small configuration."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afsa.config import N_CHAR, N_LEVEL, Config  # noqa: E402
from afsa.panel import Panel  # noqa: E402

N_SLOTS, N_POOL = 14, 22
YEAR_DAYS = {2000: 70, 2001: 70, 2002: 40}


def synthetic_arrays(seed: int = 0) -> dict[str, np.ndarray]:
    """180 days in three 'years'; 22 stocks of which at most 14 are in the universe on a day, changing every 10 days."""
    rng = np.random.default_rng(seed)
    dates = np.concatenate([np.datetime64(f"{y}-01-01") + np.arange(n) for y, n in YEAR_DAYS.items()])
    T = len(dates)
    ids = 10_000 + 7 * np.arange(N_POOL)
    permno = np.full((T, N_SLOTS), -1, dtype=np.int32)
    for t in range(T):
        block = t // 10
        members = set(range(8))                                    # always in
        members |= {8 + (block + j) % (N_POOL - 8) for j in range(5)}   # rotating: spells of 50 days
        if t % 10 != 3:
            members.add(8 + (block + 9) % (N_POOL - 8))           # one stock with a one-day gap in every block
        if t in (57, 58):
            members.discard(2)                                    # a stock that drops out for two days
        members = sorted(members)[:N_SLOTS]
        permno[t, :len(members)] = ids[members]
    ranks = rng.uniform(-0.5, 0.5, size=(T, N_SLOTS, N_CHAR))
    levels = rng.normal(size=(T, N_LEVEL)) * np.linspace(0.001, 1000.0, N_LEVEL)   # very different scales, like the medians
    levels[:, 3] = 0.0                                                             # a constant column, like rf in 2009-2017
    rf = np.full(T, 1e-4)
    # returns of the pool: two factors plus mean-reverting idiosyncratic noise
    beta = rng.normal(size=(N_POOL, 2))
    f = rng.normal(scale=0.01, size=(T, 2))
    idio = np.zeros((T, N_POOL))
    shock = rng.normal(scale=0.01, size=(T, N_POOL))
    for t in range(1, T):
        idio[t] = -0.5 * idio[t - 1] + shock[t]
    pool_ret = f @ beta.T + idio
    col = np.searchsorted(ids, np.where(permno >= 0, permno, ids[0]))
    ret = np.where(permno >= 0, np.take_along_axis(pool_ret, col, axis=1), 0.0)
    return {"dates": dates, "permno": permno, "ranks": ranks, "levels": levels, "ret": ret, "rf": rf}


def make_panel(arrays: dict[str, np.ndarray] | None = None, dtype=torch.float64) -> Panel:
    return Panel.from_arrays(**(arrays or synthetic_arrays()), device="cpu", dtype=dtype)


@pytest.fixture(scope="session")
def arrays() -> dict[str, np.ndarray]:
    return synthetic_arrays()


@pytest.fixture()
def panel(arrays) -> Panel:
    return make_panel(arrays)


@pytest.fixture()
def cfg() -> Config:
    return Config(K=3, d_x=4, d_model=5, lookback=6, batch_days=16, train_years=1, epochs=2, ev_min_days=5)
