"""The learned trading rule (Config.trade_head; analysis/rl/TRADING_RULE.md).

The rule is off by default. When it is on, the sequence model's hidden state also gives each stock a band width and a
speed for each day, and the book follows `trade_book` with those values. The tests check that:
- the books match a numpy loop written from the spec's step 5;
- constant values reproduce the fixed band exactly;
- zero rule parameters give the fixed band at (head_band0, head_eta0) and leave the target unchanged;
- members without a signal keep the fixed band;
- the gradient reaches the rule;
- on a fixed list, the month blocks give the same numbers as the day arrays."""
import dataclasses
import math

import numpy as np
import pytest
import torch
from test_book import random_targets
from test_model import DT, run

from afsa.config import Config
from afsa.model import AFSAModel, trade_book
from afsa.train import Task, evaluate, fit

# A band wide enough to skip trades in the synthetic universe, whose positions are about 1/14 of the book.
RULE = {"trade_head": True, "head_band0": 0.01, "head_band_max": 0.05, "head_eta0": 0.9}
NEW = ("delta_w", "delta_b", "eta_w", "eta_b")


def numpy_rule(w, member, delta, eta):
    """Step 5 of TRADING_RULE.md for one model, from the targets w [T, U]: the gap, the trade, the draft (closing
    non-members), and the rescale of the movers to unit gross exposure."""
    h = np.zeros_like(w)
    h[0] = w[0]
    for t in range(1, len(w)):
        gap = w[t] - h[t - 1]
        trade = eta[t] * np.sign(gap) * np.maximum(np.abs(gap) - delta[t], 0.0)
        draft = (h[t - 1] + trade) * member[t]
        moved = (trade != 0) | (~member[t] & (h[t - 1] != 0))
        held, movers = np.abs(draft[~moved]).sum(), np.abs(draft[moved]).sum()
        if movers > 0 and (1 - held) / movers > 0:
            h[t] = np.where(moved, (1 - held) / movers * draft, draft)
        elif held + movers > 0:
            h[t] = draft / (held + movers)
        else:
            h[t] = w[t]
    return h


def test_off_by_default_and_named_in_the_folder(cfg):
    assert Config().trade_head is False and Config().name() == "attn_K008"
    assert Config(trade_head=True).name() == "attn_K008_trade_head-True"
    for bad in ({"trade_head": True, "band": 4e-4}, {"trade_head": True, "band_eta": 0.5},
                {"head_band0": 0.0}, {"head_band0": 0.004}, {"head_eta0": 0.0}, {"head_eta0": 1.0}):
        with pytest.raises(ValueError):
            Config(**bad)
    plain = AFSAModel(cfg, [0], "cpu", DT)
    rule = AFSAModel(dataclasses.replace(cfg, **RULE), [0], "cpu", DT)
    assert not set(NEW) & {n for n, _ in plain.named_parameters()}         # off: the model of every earlier run
    for n, p in rule.named_parameters():                                     # on: the same draws, the rule at zero
        assert (p == 0).all() if n in NEW else torch.equal(p, getattr(plain, n)), n


@pytest.mark.parametrize("band, eta", [(0.01, 1.0), (0.04, 1.0), (0.01, 0.5)])
def test_constant_dials_are_the_fixed_band_exactly(band, eta):
    w, member = (torch.tensor(v) for v in random_targets())
    assert torch.equal(trade_book(w, member, torch.full_like(w, band), torch.full_like(w, eta)), trade_book(w, member, band, eta))


def test_per_stock_dials_follow_the_equations_of_the_spec():
    w, member = random_targets(seed=2, M=2, T=100, U=20)
    rng = np.random.default_rng(3)
    delta = rng.uniform(0.0, 0.06, size=w.shape)                             # each stock and day its own band width
    eta = rng.uniform(0.2, 1.0, size=w.shape)                                # and its own speed
    book = trade_book(*(torch.tensor(v) for v in (w, member, delta, eta))).numpy()
    for m in range(len(w)):
        assert np.allclose(book[m], numpy_rule(w[m], member[m], delta[m], eta[m]), rtol=0, atol=1e-12)
    assert np.allclose(np.abs(book).sum(-1), 1.0)                            # unit gross exposure every day
    assert (book[~member] == 0).all()                                        # nothing outside the universe
    gap = np.abs(w[:, 1:] - book[:, :-1])
    skipped = member[:, 1:] & (gap <= delta[:, 1:])
    assert skipped.mean() > 0.1 and (~skipped & member[:, 1:]).mean() > 0.1  # both kinds of day occur


def test_the_numpy_band_of_the_scoring_takes_per_stock_dials_too():
    """analysis/gap/deploy.band_path, which scores books from cash, gives trade_book's books with per-stock dials."""
    from test_book import deploy
    w, member = random_targets(seed=4, M=1, T=90, U=18)
    rng = np.random.default_rng(5)
    delta, eta = rng.uniform(0.0, 0.06, size=w.shape), rng.uniform(0.2, 1.0, size=w.shape)
    book = trade_book(*(torch.tensor(v) for v in (w, member, delta, eta))).numpy()[0]
    ref = deploy.band_path(w[0, 1:], member[0, 1:], delta[0, 1:], h0=w[0, 0], rescale="moved", eta=eta[0, 1:])
    assert np.allclose(book[1:], ref, rtol=0, atol=1e-12)


def test_zero_rule_parameters_are_the_fixed_band_and_leave_the_target(cfg, panel):
    _, _, plain = run(cfg, panel)
    _, _, out = run(dataclasses.replace(cfg, **RULE), panel)
    assert torch.equal(out["w"], plain["w"])                                 # same seed: the same target
    assert torch.allclose(out["delta"], torch.full_like(out["delta"], 0.01), rtol=1e-12, atol=0)
    assert torch.allclose(out["eta"], torch.full_like(out["eta"], 0.9), rtol=1e-12, atol=0)
    assert torch.allclose(out["book"], trade_book(out["w"], out["member_d"], 0.01, 0.9), rtol=0, atol=1e-12)
    assert out["turnover"].mean() < plain["turnover"].mean()                 # the book trades less than its target


def test_dials_follow_the_formula_and_members_without_a_signal_keep_the_fixed_band(cfg, panel):
    rule = dataclasses.replace(cfg, **RULE)
    model = AFSAModel(rule, [0, 1], "cpu", DT).eval()
    with torch.no_grad():                                                    # make the rule matter
        for n in NEW:
            getattr(model, n).copy_(torch.randn_like(getattr(model, n)))
    z, has = torch.randn(2, 3, 4, rule.d_model, dtype=DT), torch.rand(2, 3, 4) < 0.7
    delta, eta = model.dials(z, has)
    a = torch.einsum("mtud,md->mtu", z, model.delta_w) + model.delta_b[:, None, None]
    k = torch.einsum("mtud,md->mtu", z, model.eta_w) + model.eta_b[:, None, None]
    assert torch.allclose(delta[has], (0.05 / (1 + torch.exp(-(a + math.log(0.01 / 0.04)))))[has])
    assert torch.allclose(eta[has], (1 / (1 + torch.exp(-(k + math.log(0.9 / 0.1)))))[has])
    assert (delta[~has] == 0.01).all() and (eta[~has] == 0.9).all()
    assert ((delta > 0) & (delta < 0.05)).all() and ((eta > 0) & (eta < 1)).all()

    lev = panel.levels[:20]                                                  # and in the forward pass, with the day's own masks
    model.set_level_stats(lev.mean(0).expand(2, -1), lev.std(0).expand(2, -1))
    batch = panel.batch((20 + torch.arange(40))[None].expand(2, -1), rule.lookback)
    with torch.no_grad():
        out = model(batch)
    L = rule.lookback
    has, member = batch.valid[:, L:], batch.member[:, L:]
    assert (member & ~has).any() and has.any()
    assert (out["delta"][member & ~has] == 0.01).all() and (out["eta"][member & ~has] == 0.9).all()
    assert out["delta"][has].std() > 0 and out["eta"][has].std() > 0
    assert torch.allclose(out["book"], trade_book(out["w"], out["member_d"], out["delta"], out["eta"]))


def test_the_rule_learns_and_its_dials_are_kept_with_the_weights(cfg, panel):
    rule = dataclasses.replace(cfg, **RULE)
    model = AFSAModel(rule, [0], "cpu", DT)
    lev = panel.levels[:20]
    model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
    objective, _ = model.objective(model(panel.batch((20 + torch.arange(40))[None], rule.lookback)))
    objective.sum().backward()
    for n in NEW:
        g = getattr(model, n).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
    tasks = [Task(2001, 0)]
    trained, history = fit(rule, panel, tasks, dtype=DT)
    assert np.isfinite(history).all()
    assert trained.delta_w.abs().sum() > 0 and trained.eta_w.abs().sum() > 0
    res = evaluate(rule, panel, trained, tasks, keep_weights=True)[0]
    for key in ("book", "delta", "eta"):
        assert res[key].shape == res["weights"].shape, key
    assert ((res["delta"] > 0) & (res["delta"] < 0.05)).all() and ((res["eta"] > 0) & (res["eta"] < 1)).all()
    assert np.allclose(np.abs(res["book"]).sum(1), 1.0, atol=1e-6)


def test_with_the_rule_the_blocks_still_train_exactly_as_the_day_arrays(cfg):
    """test_blocks' fixed universe with the rule on: training, the evaluated book and the dials agree on both layouts."""
    from test_blocks import block_panel, day_panel, history, world
    rule = dataclasses.replace(cfg, **RULE)
    w = world(fixed=True)
    days, blks, hist = day_panel(w), block_panel(w), history(rule)
    tasks = [Task(2001, 0), Task(2002, 3)]
    a, history_a = fit(rule, days, tasks, dtype=DT)
    b, history_b = fit(hist, blks, tasks, dtype=DT)
    assert np.allclose(history_a, history_b, rtol=1e-9, atol=1e-12)
    for x, y in zip(evaluate(rule, days, a, tasks, keep_weights=True), evaluate(hist, blks, b, tasks, keep_weights=True)):
        for key in ("gross", "turnover", "short", "w_first", "w_last", "weights", "book"):
            assert np.allclose(x[key], y[key], rtol=1e-8, atol=1e-11), key
        for key in ("delta", "eta"):                                         # stored in float32
            assert np.allclose(x[key], y[key], rtol=1e-6, atol=1e-9), key
        assert not np.allclose(x["weights"], x["book"])                      # target and book differ
        assert x["delta"].std() > 0 and x["eta"].std() > 0                   # trained: the dials differ across stocks and days
