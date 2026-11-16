"""Cross-stock attention after the sequence model (Config.cross_heads; analysis/rl/CROSS_STOCK.md).

Off by default. When on, before the read-out each stock's hidden state adds a message read by attention from the
hidden states of the day's stocks with a signal. The tests check that:
- the layer is off by default, names its folder, and with its output weights at zero (as initialized) the model is
  the model without it, bit for bit;
- the message follows the formula, over the keys of one day and one model only, and stocks without a signal are not
  read;
- it is equivariant to the order of the stocks;
- the weights still use no information from the same or later days, and models in one batch stay independent;
- the gradient reaches the layer, and on a fixed list the month blocks give the same numbers as the day arrays."""
import dataclasses
import math

import numpy as np
import pytest
import torch
from conftest import make_panel
from test_model import DT, run

from afsa.config import Config
from afsa.model import AFSAModel
from afsa.train import Task, evaluate, fit

CROSS = {"cross_heads": 2, "cross_head_dim": 3}
NEW = ("xq_w", "xq_b", "xk_w", "xk_b", "xv_w", "xv_b", "xo_w", "xo_b")


def awake(model: AFSAModel) -> AFSAModel:
    """The model with random output weights, so that the layer matters."""
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for n in ("xo_w", "xo_b"):
            p = getattr(model, n)
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def dense_message(model: AFSAModel, z: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """The layer written out with an explicit softmax over each day's keys, one model and day at a time."""
    H, E = model.cfg.cross_heads, model.cfg.cross_head_dim
    out = z.clone()
    M, T, U, d = z.shape
    for m in range(M):
        for t in range(T):
            k = keys[m, t]
            if not k.any():
                continue
            zt = z[m, t]
            q = (zt @ model.xq_w[m] + model.xq_b[m]).view(U, H, E)
            kk = (zt @ model.xk_w[m] + model.xk_b[m]).view(U, H, E)[k]
            v = (zt @ model.xv_w[m] + model.xv_b[m]).view(U, H, E)[k]
            a = torch.softmax(torch.einsum("uhe,jhe->uhj", q, kk) / math.sqrt(E), dim=-1)
            read = torch.einsum("uhj,jhe->uhe", a, v).reshape(U, H * E)
            out[m, t] = zt + read @ model.xo_w[m] + model.xo_b[m]
    return out


def test_off_by_default_named_in_the_folder_and_inert_at_its_start(cfg, panel):
    assert Config().cross_heads == 0
    assert Config(cross_heads=2).name() == "attn_K008_cross_heads-2"
    for bad in ({"cross_heads": -1}, {"cross_heads": 2, "cross_head_dim": 0}):
        with pytest.raises(ValueError):
            Config(**bad)
    plain = AFSAModel(cfg, [0], "cpu", DT)
    cross = AFSAModel(dataclasses.replace(cfg, **CROSS), [0], "cpu", DT)
    for n, p in cross.named_parameters():                                    # the same draws for every other parameter
        if n not in NEW:
            assert torch.equal(p, getattr(plain, n)), n
    assert (cross.xo_w == 0).all() and (cross.xo_b == 0).all() and cross.xq_w.abs().sum() > 0
    _, _, a = run(cfg, panel)
    _, _, b = run(dataclasses.replace(cfg, **CROSS), panel)
    for key in ("w_port", "w", "ret", "turnover"):
        assert torch.equal(a[key], b[key]), key


def test_the_message_follows_the_formula_and_reads_only_stocks_with_a_signal(cfg):
    model = awake(AFSAModel(dataclasses.replace(cfg, **CROSS), [0, 1], "cpu", DT))
    z = torch.randn(2, 3, 7, cfg.d_model, dtype=DT)
    keys = torch.rand(2, 3, 7) < 0.6
    keys[1, 2] = False                                                       # a day without keys: no message
    with torch.no_grad():
        out = model.cross(z, keys)
        assert torch.allclose(out, dense_message(model, z, keys), atol=1e-12)
        assert torch.equal(out[1, 2], z[1, 2])
        z2 = z.clone()
        z2[~keys] += 3.0                                                     # stocks without a signal are not read
        out2 = model.cross(z2, keys)
        assert torch.allclose(out2[keys], out[keys], atol=1e-12)
        perm = torch.randperm(7)                                             # the order of the stocks does not matter
        assert torch.allclose(model.cross(z[:, :, perm], keys[:, :, perm]), out[:, :, perm], atol=1e-12)


def test_no_information_from_the_same_or_later_days_and_models_stay_independent(cfg, arrays, panel):
    cross = dataclasses.replace(cfg, **CROSS)
    first, n_days, t0 = 20, 40, 45
    changed = {k: v.copy() for k, v in arrays.items()}
    rng = np.random.default_rng(1)
    changed["ret"][t0:] = np.where(arrays["permno"][t0:] >= 0, rng.normal(scale=0.02, size=changed["ret"][t0:].shape), 0.0)
    changed["ranks"][t0 + 1:] = rng.uniform(-0.5, 0.5, size=changed["ranks"][t0 + 1:].shape)

    def weights(a):
        model = awake(AFSAModel(cross, [0], "cpu", DT))
        lev = make_panel(a).levels[:first]
        model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
        with torch.no_grad():
            return model(make_panel(a).batch((first + torch.arange(n_days))[None], cross.lookback))["w"]
    a, b = weights(arrays), weights(changed)
    row = t0 - first - cross.lookback
    assert torch.equal(a[:, :row + 1], b[:, :row + 1])
    assert not torch.allclose(a[:, row + 1], b[:, row + 1])
    model = awake(AFSAModel(cross, [0, 1, 2], "cpu", DT))                     # stacking models changes nothing
    firsts = (20, 55, 100)
    stat = [(panel.levels[f:f + 30].mean(0), panel.levels[f:f + 30].std(0)) for f in firsts]
    model.set_level_stats(torch.stack([s[0] for s in stat]), torch.stack([s[1] for s in stat]))
    days = torch.tensor(firsts)[:, None] + torch.arange(40)[None]
    with torch.no_grad():
        together = model(panel.batch(days, cross.lookback))
        for m in range(3):
            batch = panel.batch(days[m:m + 1], cross.lookback)
            alone = model.select([m])(batch)
            u = int(batch.n_cols[0])
            assert torch.allclose(together["w"][m, :, :u], alone["w"][0], atol=1e-11)
            assert torch.allclose(together["ret"][m], alone["ret"][0], atol=1e-11)


def test_the_layer_learns_and_the_blocks_still_train_exactly_as_the_day_arrays(cfg, panel):
    cross = dataclasses.replace(cfg, **CROSS)
    model = AFSAModel(cross, [0], "cpu", DT)
    lev = panel.levels[:20]
    model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
    objective, _ = model.objective(model(panel.batch((20 + torch.arange(40))[None], cross.lookback), train=True))
    objective.sum().backward()
    assert model.xo_w.grad.abs().sum() > 0 and model.xo_b.grad.abs().sum() > 0   # at the start only the output moves
    trained, history = fit(cross, panel, [Task(2001, 0)], dtype=DT)
    assert np.isfinite(history).all() and trained.xo_w.abs().sum() > 0 and trained.xq_w.grad is not None
    from test_blocks import block_panel, day_panel, history as on_blocks, world
    w = world(fixed=True)
    days, blks, hist = day_panel(w), block_panel(w), on_blocks(cross)
    tasks = [Task(2001, 0), Task(2002, 3)]
    a, history_a = fit(cross, days, tasks, dtype=DT)
    b, history_b = fit(hist, blks, tasks, dtype=DT)
    assert np.allclose(history_a, history_b, rtol=1e-9, atol=1e-12)
    for x, y in zip(evaluate(cross, days, a, tasks), evaluate(hist, blks, b, tasks)):
        for key in ("gross", "turnover", "short", "w_first", "w_last"):
            assert np.allclose(x[key], y[key], rtol=1e-8, atol=1e-11), key
    assert a.xo_w.abs().sum() > 0
