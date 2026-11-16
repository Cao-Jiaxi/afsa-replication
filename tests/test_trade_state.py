"""The learned trading rule with a view of the book and of the market (Config.trade_state, Config.trade_market;
analysis/rl/TRADING_STATE.md).

Both are off by default and extend the rule of Config.trade_head. With trade_state, each day each stock's band width
and speed also see the position it holds, its target, and whether trading adds to the position or reduces it, so they
are set inside the book's loop (`trade_book_state`). With trade_market, they also see the day's average hidden state.
The tests check that:
- zero new parameters give the rule without them, exactly;
- the books match a numpy loop written from the spec, and the numpy rule of the scoring (analysis/gap/deploy.py);
- the market term is one shift per day, which members without a signal do not move;
- the gradient reaches the new parameters, and `evaluate` keeps what a book carried across years needs;
- on a fixed list, the month blocks give the same numbers as the day arrays."""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_book import random_targets
from test_model import DT, run
from test_trade_head import RULE

from afsa.config import Config
from afsa.model import STATE_FEATURES, AFSAModel, trade_book, trade_book_state
from afsa.train import Task, evaluate, fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "gap"))
import deploy  # noqa: E402

STATE = {**RULE, "trade_state": True}
MARKET = {**RULE, "trade_market": True}
BOTH = {**RULE, "trade_state": True, "trade_market": True}
CONST = {"band0": 0.01, "band_max": 0.05, "eta0": 0.9}                       # RULE's constants
NEW = ("delta_v", "eta_v", "delta_u", "eta_u")


def tensors(*arrays):
    return [torch.tensor(np.ascontiguousarray(v)) for v in arrays]


def random_rule(seed=3, M=2, T=100, U=20):
    """Targets, membership, logits a and k, who has a signal, and weights v [M, 2, 3] on the view of the book."""
    w, member = random_targets(seed=seed, M=M, T=T, U=U)
    rng = np.random.default_rng(seed + 1)
    a, k = rng.normal(-1.0, 1.0, size=w.shape), rng.normal(1.0, 1.0, size=w.shape)
    has = member & (rng.random(w.shape) < 0.9)
    v = rng.normal(0.0, 1.0, size=(M, 2, len(STATE_FEATURES)))
    return w, member, a, k, has, v


def numpy_state_rule(w, member, a, k, has, v, band0, band_max, eta0):
    """TRADING_STATE.md for one model, from the targets w [T, U]: the view of the book, the dials, then step 5 of
    TRADING_RULE.md (gap, trade, draft, movers-only rescale). Day 0 holds its target."""
    h = np.zeros_like(w)
    h[0] = w[0]
    for t in range(1, len(w)):
        n, prev = member[t].sum(), h[t - 1]
        view = np.stack([np.log1p(n * np.abs(prev)), np.log1p(n * np.abs(w[t])), np.sign(prev) * np.sign(w[t] - prev)], -1)
        delta = np.where(has[t], band_max / (1 + np.exp(-(a[t] + view @ v[0]))), band0)
        eta = np.where(has[t], 1 / (1 + np.exp(-(k[t] + view @ v[1]))), eta0)
        gap = w[t] - prev
        trade = eta * np.sign(gap) * np.maximum(np.abs(gap) - delta, 0.0)
        draft = (prev + trade) * member[t]
        moved = (trade != 0) | (~member[t] & (prev != 0))
        held, movers = np.abs(draft[~moved]).sum(), np.abs(draft[moved]).sum()
        if movers > 0 and (1 - held) / movers > 0:
            h[t] = np.where(moved, (1 - held) / movers * draft, draft)
        elif held + movers > 0:
            h[t] = draft / (held + movers)
        else:
            h[t] = w[t]
    return h


def test_off_by_default_named_in_the_folder_and_refused_without_the_rule(cfg):
    assert Config().trade_state is False and Config().trade_market is False
    assert Config(trade_head=True, trade_state=True).name() == "attn_K008_trade_head-True_trade_state-True"
    assert Config(trade_head=True, trade_state=True, trade_market=True).name() == \
        "attn_K008_trade_head-True_trade_state-True_trade_market-True"
    for bad in ({"trade_state": True}, {"trade_market": True}):
        with pytest.raises(ValueError):
            Config(**bad)
    rule = AFSAModel(dataclasses.replace(cfg, **RULE), [0], "cpu", DT)
    both = AFSAModel(dataclasses.replace(cfg, **BOTH), [0], "cpu", DT)
    assert not set(NEW) & {n for n, _ in rule.named_parameters()}
    for n, p in both.named_parameters():                                     # the same draws; the new parameters at zero
        assert (p == 0).all() if n in NEW else torch.equal(p, getattr(rule, n)), n
    assert both.delta_v.shape == (1, 3) and both.delta_u.shape == (1, cfg.d_model)


def test_zero_weights_on_the_book_give_the_rule_without_them_exactly():
    w, member, a, k, has, _ = random_rule()
    zero = torch.zeros(len(w), 3, dtype=DT)
    W, Mb, A, Kl, G = tensors(w, member, a, k, has)
    book, delta, eta = trade_book_state(W, Mb, A, Kl, G, zero, zero, **CONST)
    d_ref, e_ref = torch.where(G, 0.05 * torch.sigmoid(A), 0.01), torch.where(G, torch.sigmoid(Kl), 0.9)
    assert torch.equal(delta, d_ref) and torch.equal(eta, e_ref)
    assert torch.equal(book, trade_book(W, Mb, d_ref, e_ref))


def test_the_rule_with_a_view_of_the_book_follows_the_spec_and_the_numpy_rule_of_the_scoring():
    w, member, a, k, has, v = random_rule()
    W, Mb, A, Kl, G, V = tensors(w, member, a, k, has, v)
    book, delta, eta = trade_book_state(W, Mb, A, Kl, G, V[:, 0], V[:, 1], **CONST)
    book = book.numpy()
    for m in range(len(w)):
        ref = numpy_state_rule(w[m], member[m], a[m], k[m], has[m], v[m], **CONST)
        assert np.allclose(book[m], ref, rtol=0, atol=1e-12)
        T = len(w[m]) - 1                                                    # deploy: from day 0's target on
        H, D, E = deploy.state_path(w[m, 1:], member[m, 1:], [(a[m, 1:], k[m, 1:], np.broadcast_to(v[m], (T, 2, 3)))],
                                    has[m, 1:], h0=w[m, 0], **CONST)
        assert np.allclose(H, book[m, 1:], rtol=0, atol=1e-12)
        assert np.allclose(D, delta[m, 1:].numpy(), rtol=0, atol=1e-12) and np.allclose(E, eta[m, 1:].numpy(), rtol=0, atol=1e-12)
    assert np.allclose(np.abs(book).sum(-1), 1.0) and (book[~member] == 0).all()
    adds = np.sign(book[:, :-1]) * np.sign(w[:, 1:] - book[:, :-1])         # the view changes the dials: both kinds of trade occur
    assert (adds[member[:, 1:]] == 1).mean() > 0.1 and (adds[member[:, 1:]] == -1).mean() > 0.1
    assert not np.allclose(book, trade_book(W, Mb, torch.where(G, 0.05 * torch.sigmoid(A), 0.01),
                                            torch.where(G, torch.sigmoid(Kl), 0.9)).numpy())


def test_a_seed_ensemble_of_rules_averages_their_dials_on_the_same_book():
    w, member, a, k, has, v = random_rule(seed=5, M=3, T=60, U=12)
    W = deploy.ensemble([w[m] for m in range(3)])
    rules = [(a[m], k[m], np.broadcast_to(v[m], (60, 2, 3))) for m in range(3)]
    H, D, E = deploy.state_path(W, member[0], rules, has[0], **CONST)
    h = np.zeros(W.shape[1])
    for t in range(60):                                                      # one day at a time, by hand
        n = member[0, t].sum()
        view = np.stack([np.log1p(n * np.abs(h)), np.log1p(n * np.abs(W[t])), np.sign(h) * np.sign(W[t] - h)], -1)
        d = np.mean([np.where(has[0, t], 0.05 / (1 + np.exp(-(a[m, t] + view @ v[m, 0]))), 0.01) for m in range(3)], 0)
        e = np.mean([np.where(has[0, t], 1 / (1 + np.exp(-(k[m, t] + view @ v[m, 1]))), 0.9) for m in range(3)], 0)
        h = deploy.band_step(h, W[t], member[0, t], d, "moved", e)
        assert np.allclose(H[t], h, rtol=0, atol=1e-12) and np.allclose(D[t], d) and np.allclose(E[t], e)
    widened, Dw, _ = deploy.state_path(W, member[0], rules, has[0], widen=np.full(W.shape, 0.01), **CONST)
    assert np.abs(np.diff(widened, axis=0)).sum() < np.abs(np.diff(H, axis=0)).sum()   # a wider band trades less


def test_zero_new_parameters_give_the_rule_of_trading_rule_md(cfg, panel):
    _, _, rule = run(dataclasses.replace(cfg, **RULE), panel)
    for extra in (STATE, MARKET, BOTH):
        _, _, out = run(dataclasses.replace(cfg, **extra), panel)
        assert torch.equal(out["w"], rule["w"])
        for key in ("book", "delta", "eta"):
            assert torch.equal(out[key], rule[key]), (extra, key)


def test_the_market_term_is_one_shift_per_day_that_members_without_a_signal_do_not_move(cfg):
    model = AFSAModel(dataclasses.replace(cfg, **MARKET), [0, 1], "cpu", DT)
    with torch.no_grad():
        for n in ("delta_w", "delta_b", "eta_w", "eta_b", "delta_u", "eta_u"):
            getattr(model, n).copy_(torch.randn_like(getattr(model, n)))
    z, has = torch.randn(2, 3, 6, cfg.d_model, dtype=DT), torch.rand(2, 3, 6) < 0.7
    has[0, 0] = False                                                        # a day without any signal
    a, k = model.dial_logits(z, has)
    g = has.to(DT)
    zbar = (z * g[..., None]).sum(2) / g.sum(-1, keepdim=True).clamp_min(1)
    own_a = torch.einsum("mtud,md->mtu", z, model.delta_w) + model.delta_b[:, None, None]
    own_k = torch.einsum("mtud,md->mtu", z, model.eta_w) + model.eta_b[:, None, None]
    shift_a = (zbar * model.delta_u[:, None]).sum(-1)[..., None]
    shift_k = (zbar * model.eta_u[:, None]).sum(-1)[..., None]
    assert torch.allclose(a, own_a + shift_a + math.log(0.01 / 0.04)) and torch.allclose(k, own_k + shift_k + math.log(0.9 / 0.1))
    z2 = z.clone()
    z2[~has] += 5.0                                                          # members without a signal do not enter the mean
    a2, k2 = model.dial_logits(z2, has)
    assert torch.allclose(a2[has], a[has]) and torch.allclose(k2[has], k[has])


def test_the_view_learns_and_evaluate_keeps_what_a_carried_book_needs(cfg, panel):
    both = dataclasses.replace(cfg, **BOTH)
    model = AFSAModel(both, [0], "cpu", DT)
    lev = panel.levels[:20]
    model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
    objective, _ = model.objective(model(panel.batch((20 + torch.arange(40))[None], both.lookback)))
    objective.sum().backward()
    for n in NEW:
        g = getattr(model, n).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
    tasks = [Task(2001, 0)]
    trained, history = fit(both, panel, tasks, dtype=DT)
    assert np.isfinite(history).all()
    assert all(getattr(trained, n).abs().sum() > 0 for n in NEW)
    res = evaluate(both, panel, trained, tasks, keep_weights=True)[0]
    for key in ("book", "delta", "eta", "dial_a", "dial_k", "signal"):
        assert res[key].shape == res["weights"].shape, key
    assert res["signal"].dtype == bool and res["dial_v"].shape == (2, 3)
    assert np.allclose(res["dial_v"], torch.stack([trained.delta_v[0], trained.eta_v[0]]).detach().numpy())
    # The numpy rule of the scoring, started from the book of the year's first day, gives the stored book on the
    # other days. Members are where the target is not 0 (the hedge gives every member a position).
    member = res["weights"] != 0
    assert (res["book"][~member] == 0).all()
    n = len(member) - 1
    H, D, E = deploy.state_path(res["weights"][1:].astype(np.float64), member[1:],
                                [(res["dial_a"][1:], res["dial_k"][1:], np.broadcast_to(res["dial_v"], (n, 2, 3)))],
                                res["signal"][1:], h0=res["book"][0].astype(np.float64), **CONST)
    assert np.abs(H - res["book"][1:]).max() < 1e-5
    assert np.abs(D - res["delta"][1:]).max() < 1e-6 and np.abs(E - res["eta"][1:]).max() < 1e-5


def test_pessimism_shrinks_the_ensemble_where_the_seeds_disagree():
    """analysis/rl/pessimism.py: kappa = 0 is deploy.ensemble; a larger kappa moves weight away from the stocks whose
    seeds disagree, and the book keeps unit gross exposure."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "rl"))
    import pessimism
    rng = np.random.default_rng(7)
    base = rng.normal(size=(30, 10))
    noise = 0.1 + 2.0 * (np.arange(10) >= 5)                                 # the seeds disagree on the last five stocks
    Ws = [pessimism.unit(base + rng.normal(size=(30, 10)) * noise) for _ in range(3)]
    assert np.allclose(pessimism.shrink(Ws, 0.0), deploy.ensemble(Ws))
    shrunk = pessimism.shrink(Ws, 4.0)
    assert np.allclose(np.abs(shrunk).sum(1), 1.0)
    assert np.abs(shrunk[:, 5:]).sum(1).mean() < np.abs(deploy.ensemble(Ws)[:, 5:]).sum(1).mean() - 0.05


def test_with_the_views_the_blocks_still_train_exactly_as_the_day_arrays(cfg):
    """test_blocks' fixed universe with both views on: training, the evaluated book and the dials agree on both layouts."""
    from test_blocks import block_panel, day_panel, history, world
    both = dataclasses.replace(cfg, **BOTH)
    w = world(fixed=True)
    days, blks, hist = day_panel(w), block_panel(w), history(both)
    tasks = [Task(2001, 0), Task(2002, 3)]
    a, history_a = fit(both, days, tasks, dtype=DT)
    b, history_b = fit(hist, blks, tasks, dtype=DT)
    assert np.allclose(history_a, history_b, rtol=1e-9, atol=1e-12)
    for x, y in zip(evaluate(both, days, a, tasks, keep_weights=True), evaluate(hist, blks, b, tasks, keep_weights=True)):
        for key in ("gross", "turnover", "short", "w_first", "w_last", "weights", "book"):
            assert np.allclose(x[key], y[key], rtol=1e-8, atol=1e-11), key
        for key in ("delta", "eta", "dial_a", "dial_k", "dial_v"):          # stored in float32
            assert np.allclose(x[key], y[key], rtol=1e-6, atol=1e-9), key
        assert np.array_equal(x["signal"], y["signal"])
        assert np.abs(x["dial_v"]).sum() > 0                                 # trained: the view of the book matters
