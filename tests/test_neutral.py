"""Neutrality inside the model (Config.signal_center, Config.neutral; analysis/gap/EXTENSION.md).

Off by default. With `neutral`, each day's weights sum to zero and have no exposure to the chosen characteristic
ranks over the day's members, with unit gross exposure; the month blocks give the day arrays' numbers on a fixed
list; and the weights still use nothing from the same or later days."""
import dataclasses

import numpy as np
import pytest
import torch
from test_blocks import DT, L, block_panel, day_panel, forward, history, world

from afsa.config import CHARACTERISTICS, NEUTRAL_SETS, Config
from afsa.model import AFSAModel
from afsa.train import Task, evaluate, fit


def test_off_by_default_and_named_in_the_folder(cfg):
    assert Config().neutral == "" and Config().signal_center is False
    assert Config().name() == "attn_K008"
    assert Config(neutral="riskmom", signal_center=True).name() == "attn_K008_signal_center-True_neutral-riskmom"
    with pytest.raises(ValueError):
        Config(neutral="everything")
    with pytest.raises(ValueError):                                         # not neutral: positions are dropped after the projection
        Config(neutral="risk", nosignal_hedge=False)
    assert set(NEUTRAL_SETS["riskmom"]) <= set(CHARACTERISTICS) and NEUTRAL_SETS["dollar"] == ()


@pytest.mark.parametrize("which", ["dollar", "risk"])
def test_day_arrays_weights_are_neutral(cfg, panel, which):
    """A changing universe on the day x slot arrays: no net position, no exposure to the set, unit gross exposure.
    (The larger sets have as many ranks as the synthetic universe has stocks, which leaves no book at all.)"""
    cfg = dataclasses.replace(cfg, neutral=which)
    model = AFSAModel(cfg, [0, 1], "cpu", DT).eval()
    days = torch.tensor([[40], [90]]) + torch.arange(-(L + 1), 25)[None]
    b = panel.batch(days, L)
    with torch.no_grad():
        out = model(b)
    w, member, X = out["w"], b.member[:, L:], b.X[:, L:]
    assert (w[~member] == 0).all()
    assert torch.allclose(w.abs().sum(-1), torch.ones_like(w[..., 0]), atol=1e-12)
    assert w.sum(-1).abs().max() < 1e-7
    for c in NEUTRAL_SETS[which][:6]:
        expo = (w * X[..., CHARACTERISTICS.index(c)] * member).sum(-1)
        assert expo.abs().max() < 1e-7, c


def test_a_rank_that_is_the_same_for_every_member_is_skipped(cfg, arrays):
    """All members at rank 0 (a characteristic missing for everyone under fill none): the solve stays finite."""
    from conftest import make_panel
    a = {k: v.copy() for k, v in arrays.items()}
    a["ranks"][:, :, CHARACTERISTICS.index("Beta")] = 0.0
    panel = make_panel(a)
    model = AFSAModel(dataclasses.replace(cfg, neutral="risk"), [0], "cpu", DT).eval()
    with torch.no_grad():
        out = model(panel.batch(torch.tensor([[40]]) + torch.arange(-(L + 1), 20)[None], L))
    assert torch.isfinite(out["w"]).all() and out["w"].sum(-1).abs().max() < 1e-7


def test_centered_signals_have_zero_mean(cfg, panel):
    cfg = dataclasses.replace(cfg, signal_center=True)
    model = AFSAModel(cfg, [0, 1], "cpu", DT).eval()
    b = panel.batch(torch.tensor([[40], [90]]) + torch.arange(-(L + 1), 25)[None], L)
    with torch.no_grad():
        out = model(b)
    valid = b.valid[:, L:]
    assert (out["w_port"][~valid] == 0).all()
    assert out["w_port"].sum(-1).abs().max() < 1e-10
    with torch.no_grad():                                                    # it does change the book
        plain = AFSAModel(dataclasses.replace(cfg, signal_center=False), [0, 1], "cpu", DT).eval()(b)
    assert not torch.allclose(plain["w"], out["w"])


def test_centered_block_signals_on_a_changing_universe(cfg):
    """signal_center on the month blocks of a changing universe: a stock without a signal keeps a signal of 0 (also a
    member without one), and the day's signals sum to 0."""
    _, batch, out = forward(history(cfg, signal_center=True), block_panel(world()), (47, 120), 30)
    M, NB, R, S = batch.member.shape
    member = batch.member.view(M, NB * R, S)[torch.arange(M)[:, None], batch.day_block * R + batch.day_row]   # [M, D+1, S]
    assert (member & ~batch.signal).any()                                    # the case occurs
    assert (out["w_port"][~batch.signal] == 0).all()
    assert out["w_port"].sum(-1).abs().max() < 1e-10


def test_without_the_hedge_position_a_stock_without_a_signal_holds_nothing(cfg, panel):
    """Config.nosignal_hedge = False on the day x slot arrays (no signal = fewer than `lookback` prior days in the
    universe) and on the month blocks of a changing universe (new listing, no accounts)."""
    assert Config().nosignal_hedge is True and Config(nosignal_hedge=False).name() == "attn_K008_nosignal_hedge-False"
    off = dataclasses.replace(cfg, nosignal_hedge=False)
    b = panel.batch(torch.tensor([[40], [90]]) + torch.arange(-(L + 1), 25)[None], L)
    with torch.no_grad():
        plain, out = AFSAModel(cfg, [0, 1], "cpu", DT).eval()(b), AFSAModel(off, [0, 1], "cpu", DT).eval()(b)
    valid = b.valid[:, L:]
    assert (b.member[:, L:] & ~valid).any() and (plain["w"][b.member[:, L:] & ~valid] != 0).any()    # the case occurs
    assert (out["w"][~valid] == 0).all()
    assert torch.allclose(out["w"].abs().sum(-1), torch.ones_like(out["w"][..., 0]), atol=1e-12)
    blocks = block_panel(world())
    _, batch, o = forward(history(off), blocks, (47, 120), 30)
    _, _, p = forward(history(cfg), blocks, (47, 120), 30)
    for m, first in enumerate((47, 120)):
        for d, c in enumerate(range(first - 1, first + 30)):
            blk, row = int(blocks.own_block[c]), int(blocks.own_row[c])
            slot = batch.slot[m, d]
            on = slot >= 0
            sig = batch.signal[m, d][slot.clamp_min(0)] & on
            assert (o["w"][m, d][~sig] == 0).all() and abs(float(o["w"][m, d].abs().sum()) - 1) < 1e-12
    assert not torch.allclose(o["w"], p["w"])                               # stocks 13, 14 and 15 lose their hedge positions


@pytest.mark.parametrize("extra", [{"neutral": "risk"}, {"signal_center": True}, {"neutral": "dollar", "signal_center": True}, {"nosignal_hedge": False}])
def test_on_a_fixed_universe_the_blocks_equal_the_day_arrays(cfg, extra):
    """Training and evaluation with entry_rule history reproduce entry_rule run on a list that never changes."""
    w = world(fixed=True)
    cfg = dataclasses.replace(cfg, **extra)
    days, blks, hist = day_panel(w), block_panel(w), history(cfg)
    tasks = [Task(2001, 0), Task(2002, 3)]
    a, history_a = fit(cfg, days, tasks, dtype=DT)
    b, history_b = fit(hist, blks, tasks, dtype=DT)
    assert np.allclose(history_a, history_b, rtol=1e-8, atol=1e-11)
    for x, y in zip(evaluate(cfg, days, a, tasks, keep_weights=True), evaluate(hist, blks, b, tasks, keep_weights=True)):
        for key in ("gross", "turnover", "short", "weights"):
            assert np.allclose(x[key], y[key], rtol=1e-7, atol=1e-10), key


def test_block_weights_are_neutral_on_a_changing_universe(cfg):
    """Entrants, a new listing, a delisting and missing accounts: each day's weights are neutral over that day's
    members, with the ranks of the day's own row of its block."""
    hist = history(cfg, neutral="risk")
    panel = block_panel(world())
    firsts = (47, 120)
    model, batch, out = forward(hist, panel, firsts, 30)
    cols = [CHARACTERISTICS.index(c) for c in NEUTRAL_SETS["risk"]]
    for m, first in enumerate(firsts):
        for d, c in enumerate(range(first - 1, first + 30)):
            blk, row = int(panel.own_block[c]), int(panel.own_row[c])
            slot = batch.slot[m, d]
            on = slot >= 0
            wd = out["w"][m, d]
            assert (wd[~on] == 0).all() and abs(float(wd.abs().sum()) - 1) < 1e-12
            assert on.sum() == panel.member[blk, row].sum()                  # every member of the day has a column
            X = panel.ranks[blk, row][slot[on]][:, cols]
            assert abs(float(wd[on].sum())) < 1e-7
            assert (wd[on] @ X).abs().max() < 1e-7


def test_neutral_block_weights_use_no_information_from_the_same_or_later_days(cfg):
    hist, t0, first = history(cfg, neutral="risk", signal_center=True), 100, 80
    w = world()
    changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
    rng = np.random.default_rng(1)
    changed["ret"][t0:] = rng.normal(scale=0.02, size=changed["ret"][t0:].shape)
    changed["ranks"][t0 + 1:] = rng.uniform(-0.5, 0.5, size=changed["ranks"][t0 + 1:].shape)
    _, _, a = forward(hist, block_panel(w), (first,), 40, seeds=(0,))
    _, _, b = forward(hist, block_panel(changed), (first,), 40, seeds=(0,))
    row = t0 - (first - 1)
    assert torch.equal(a["w"][:, :row + 1], b["w"][:, :row + 1])
    assert not torch.allclose(a["w"][:, row + 1], b["w"][:, row + 1])
