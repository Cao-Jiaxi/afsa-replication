"""The training arrays in Data/TrainingData and the batches built from them (skipped until the arrays are built)."""
import json

import numpy as np
import pytest
import torch

from afsa.config import N_CHAR, N_LEVEL, N_SLOTS, TRAIN_DATA_DIR, Config
from afsa.panel import Panel, membership
from afsa.train import Task, train_span

FOLDER = TRAIN_DATA_DIR / "broad"
pytestmark = pytest.mark.skipif(not (FOLDER / "meta.json").exists(), reason="run `python -m afsa.build_data` first")


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel("broad", "cpu", first_year=1996, last_year=1997)


def test_build_checks_all_passed():
    meta = json.loads((FOLDER / "meta.json").read_text())
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == []
    assert sum(c["ok"] is True for c in meta["checks"]) == 16
    assert (meta["n_days"], meta["n_slots"], meta["n_stock_days"]) == (9067, 500, 4_527_395)
    assert meta["first_day"] == "1990-01-02" and meta["last_day"] == "2025-12-31"
    assert len(meta["missing_returns_counted_as_zero"]) == 87


def test_array_shapes_and_values():
    a = {name: np.load(FOLDER / f"{name}.npy", mmap_mode="r") for name in ("dates", "permnos", "permno", "slot_of", "ranks", "levels", "ret", "rf", "run")}
    T, S = len(a["dates"]), len(a["permnos"])
    assert a["ranks"].shape == (T, N_SLOTS, N_CHAR) and a["levels"].shape == (T, N_LEVEL)
    assert a["permno"].shape == a["ret"].shape == a["run"].shape == (T, N_SLOTS) and a["slot_of"].shape == (T, S)
    member = np.asarray(a["permno"]) >= 0
    assert member.sum() == 4_527_395 and member.sum(1).min() >= 493
    assert (np.diff(np.where(member, a["permno"], np.iinfo(np.int32).max), axis=1) > 0)[member[:, 1:]].all()   # sorted, filled from slot 0
    assert np.isfinite(a["ranks"][::50]).all() and np.isfinite(a["levels"]).all() and np.isfinite(a["ret"]).all()
    assert (np.asarray(a["ret"])[~member] == 0).all() and (np.asarray(a["run"])[~member] == 0).all()
    assert (np.asarray(a["run"])[0][member[0]] == 1).all()                    # the sample starts with no history


def test_membership_history_against_brute_force():
    permno = np.load(FOLDER / "permno.npy", mmap_mode="r")[1500:1700]
    permnos, slot_of, run = membership(np.array(permno))
    rng = np.random.default_rng(0)
    for t, n in zip(rng.integers(40, 200, 300), rng.integers(0, 493, 300)):
        pid = permno[t, n]
        assert slot_of[t, np.searchsorted(permnos, pid)] == n
        k = 0
        while t - k >= 0 and (permno[t - k] == pid).any():
            k += 1
        assert run[t, n] == k


def test_batch_reproduces_the_arrays(panel):
    lookback = 30
    start, stop = panel.year_span(1997, 1997)
    days = torch.tensor([[start + 5 + i for i in range(40)], [stop - 40 + i for i in range(40)]])
    b = panel.batch(days, lookback)
    ranks, ret, rf, permno = (np.load(FOLDER / f"{n}.npy", mmap_mode="r") for n in ("ranks", "ret", "rf", "permno"))
    for m in range(2):
        u = int(b.n_cols[m])
        pid = panel.permnos[b.cols[m, :u].numpy()]
        assert u >= 500 and len(set(pid)) == u and not b.member[m, :, u:].any()
        for i in (0, 17, 39):
            t = int(days[m, i])
            row = {p: s for s, p in enumerate(np.asarray(permno[t])) if p >= 0}
            mem = b.member[m, i, :u].numpy()
            assert set(pid[mem]) == set(row)
            slots = [row[p] for p in pid[mem]]
            assert np.array_equal(b.X[m, i, :u].numpy()[mem], np.asarray(ranks[t])[slots])
            assert np.allclose(b.R[m, i, :u].numpy()[mem], np.asarray(ret[t])[slots] - rf[t], atol=1e-7)
            assert (b.R[m, i, :u].numpy()[~mem] == 0).all()
            past = [set(np.asarray(permno[s])) for s in range(t - lookback, t)]
            valid = np.array([all(p in day for day in past) for p in pid]) & mem
            assert np.array_equal(b.valid[m, i, :u].numpy(), valid)


def test_training_windows_of_the_paper():
    full = Panel.__new__(Panel)                                               # calendar only: no need to load the arrays
    full.dates = np.load(FOLDER / "dates.npy")
    full.years = full.dates.astype("datetime64[Y]").astype(int) + 1970
    full.n_days, full.loaded = len(full.dates), (0, len(full.dates))
    cfg = Config()
    start, stop = train_span(cfg, full, Task(1998, 0))                        # first window: 1990-1997
    assert (start, stop) == (31, 2024) and str(full.dates[start]) == "1990-02-14" and str(full.dates[stop]) == "1998-01-02"
    for year in range(1999, 2027):
        start, stop = train_span(cfg, full, Task(year, 0))
        assert full.years[start] == year - 8 and full.years[stop - 1] == year - 1 and 2010 <= stop - start <= 2024
    with pytest.raises(ValueError):
        train_span(cfg, full, Task(2027, 0))                                  # 2026 is not in the sample
