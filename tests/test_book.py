"""The trading layer (`afsa.model.trade_book`, `Config.band`): the book the model holds and is charged for, against the
numpy band of analysis/gap/deploy.py with the movers-only rescale."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from afsa.model import AFSAModel, trade_book
from test_model import DT, run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "gap"))
import deploy  # noqa: E402


def random_targets(seed=0, M=2, T=80, U=15):
    """Targets with ||w||_1 = 1 over a membership that changes from day to day (entries and exits)."""
    rng = np.random.default_rng(seed)
    member = rng.random((M, T, U)) < 0.85
    w = rng.normal(size=(M, T, U)) * member
    return w / np.abs(w).sum(-1, keepdims=True), member


@pytest.mark.parametrize("band, eta", [(0.0, 1.0), (0.01, 1.0), (0.04, 1.0), (0.01, 0.5), (0.0, 0.3)])
def test_book_equals_the_numpy_band_with_the_movers_only_rescale(band, eta):
    w, member = random_targets()
    book = trade_book(torch.tensor(w), torch.tensor(member), band, eta).numpy()
    for m in range(w.shape[0]):
        ref = deploy.band_path(w[m, 1:], member[m, 1:], band, h0=w[m, 0], rescale="moved", eta=eta)
        assert np.array_equal(book[m, 0], w[m, 0])                          # day 0: the target
        assert np.allclose(book[m, 1:], ref, atol=1e-12)
    assert np.allclose(np.abs(book).sum(-1), 1.0)                            # unit gross exposure every day
    assert (book[~member] == 0).all()                                        # nothing outside the universe


def test_without_band_the_book_is_the_target(cfg, panel):
    w, member = random_targets()
    assert np.allclose(trade_book(torch.tensor(w), torch.tensor(member), 0.0, 1.0).numpy(), w, atol=1e-12)
    _, _, out = run(cfg, panel)                                              # the defaults skip the layer
    assert out["book"] is out["w"]


def test_stocks_that_do_not_trade_keep_their_position_exactly():
    w, member = random_targets(seed=1, M=1, T=120, U=30)
    member[:] = True                                                         # no closings: every frozen stock is a skip
    band = 0.01
    book = trade_book(torch.tensor(w), torch.tensor(member), band, 1.0).numpy()[0]
    uniform = deploy.band_path(w[0, 1:], member[0, 1:], band, h0=w[0, 0])    # the stored results' rescale
    frozen_days = moved_uniform = 0
    for t in range(1, len(book)):
        gap = w[0, t] - book[t - 1]
        trade = np.sign(gap) * np.maximum(np.abs(gap) - band, 0.0)
        moved = trade != 0
        frozen = ~moved & (book[t - 1] != 0)
        movers = np.abs(book[t - 1] + trade)[moved].sum()
        if frozen.any() and movers > 0 and (1 - np.abs(book[t - 1][frozen]).sum()) / movers > 0:
            assert np.array_equal(book[t][frozen], book[t - 1][frozen])
            frozen_days += 1
        if t >= 2:
            g = w[0, t] - uniform[t - 2]
            f = (np.abs(g) <= band) & (uniform[t - 2] != 0)
            moved_uniform += bool(f.any() and not np.allclose(uniform[t - 1][f], uniform[t - 2][f], rtol=0, atol=1e-15))
    assert frozen_days > 50                                                  # the check did run on most days
    assert moved_uniform > 50                                                # while the uniform rescale moves frozen stocks


def test_the_model_is_charged_for_the_book_and_its_target_is_unchanged(cfg, panel):
    _, _, plain = run(cfg, panel)
    banded = dataclasses.replace(cfg, band=0.01)
    _, _, out = run(banded, panel)
    assert torch.equal(out["w"], plain["w"])                                 # same seed: the layer only changes what is charged
    book, R = out["book"][0].numpy(), out["R_d"][0].numpy()
    assert torch.allclose(out["book"], trade_book(out["w"], out["member_d"], 0.01, 1.0))
    assert np.allclose(out["ret"][0].numpy(), (book[1:] * R[1:]).sum(1))
    assert np.allclose(out["turnover"][0].numpy(), np.abs(book[1:] - book[:-1]).sum(1))
    assert np.allclose(out["short"][0].numpy(), np.maximum(-book[1:], 0).sum(1))
    assert out["turnover"].mean() < plain["turnover"].mean()                 # the book trades less than its target


def test_gradients_flow_through_the_book(cfg, panel):
    banded = dataclasses.replace(cfg, band=0.01)
    model = AFSAModel(banded, [0], "cpu", DT)
    lev = panel.levels[:20]
    model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
    batch = panel.batch((20 + torch.arange(40))[None], banded.lookback)
    objective, _ = model.objective(model(batch))
    objective.sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert {"Q", "W_rank"} <= set(grads)
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert grads["Q"].abs().sum() > 0 and grads["head_w"].abs().sum() > 0


def test_with_the_band_the_blocks_still_train_exactly_as_the_day_arrays(cfg):
    """test_blocks' fixed universe, with the layer on: training and the evaluated book agree on both layouts."""
    from test_blocks import block_panel, day_panel, history, world

    from afsa.train import Task, evaluate, fit
    banded = dataclasses.replace(cfg, band=0.01)
    w = world(fixed=True)
    days, blks, hist = day_panel(w), block_panel(w), history(banded)
    tasks = [Task(2001, 0), Task(2002, 3)]
    a, history_a = fit(banded, days, tasks, dtype=DT)
    b, history_b = fit(hist, blks, tasks, dtype=DT)
    assert np.allclose(history_a, history_b, rtol=1e-9, atol=1e-12)
    for x, y in zip(evaluate(banded, days, a, tasks, keep_weights=True), evaluate(hist, blks, b, tasks, keep_weights=True)):
        for key in ("gross", "turnover", "short", "w_first", "w_last", "weights", "book"):
            assert np.allclose(x[key], y[key], rtol=1e-8, atol=1e-11), key
        assert not np.allclose(x["weights"], x["book"])                      # the layer is on: target and book differ
        assert np.allclose(np.abs(x["book"]).sum(1), 1.0)


def test_config_rejects_a_negative_band_or_an_eta_outside_0_1(cfg):
    for bad in ({"band": -1e-4}, {"band_eta": 0.0}, {"band_eta": 1.5}):
        with pytest.raises(ValueError):
            dataclasses.replace(cfg, **bad)
    assert "_band-0.0004_" in dataclasses.replace(cfg, band=4e-4).name()   # a run with the layer has its own folder
