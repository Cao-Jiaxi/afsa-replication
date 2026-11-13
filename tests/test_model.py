"""The model against the paper's formulas, written out independently and densely for small synthetic panels."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from afsa.config import CHAR_GROUPS, CHARACTERISTICS, N_CHAR, Config
from afsa.model import AFSAModel, explained_variance, init_member, squash
from conftest import make_panel, synthetic_arrays

DT = torch.float64


def run(cfg, panel, seeds=(0,), first=20, n_days=40, **kw):
    model = AFSAModel(cfg, list(seeds), "cpu", DT)
    lev = panel.levels[:first]                                               # statistics of earlier days only
    model.set_level_stats(lev.mean(0).expand(len(seeds), -1), lev.std(0).expand(len(seeds), -1))
    days = (first + torch.arange(n_days))[None].expand(len(seeds), -1)
    batch = panel.batch(days, cfg.lookback)
    with torch.no_grad():
        return model, batch, model(batch, **kw)


def test_factor_weights_are_a_softmax_over_the_days_universe(cfg, panel):
    _, batch, out = run(cfg, panel)
    wF = out["wF"]
    assert torch.allclose(wF.sum(2), torch.ones_like(wF.sum(2)))             # each factor's weights sum to 1
    assert (wF >= 0).all()
    assert (wF[~batch.member] == 0).all()                                    # nothing outside the universe
    assert not batch.member.all() and batch.member.any(1).all()              # the panel does have entries and exits


def test_residuals_and_weights_match_the_dense_formulas_of_the_paper(cfg, panel):
    """Sections 3.2-3.3 with the N x N matrices written out, on single days."""
    model, batch, out = run(cfg, panel)
    L, dx = cfg.lookback, cfg.d_x
    p = model.member_state(0)
    WK = torch.cat([p["W_rank"], p["W_level"]])                              # W^K, 79 x d_x
    for t in (L, L + 7, 39):
        mem = batch.member[0, t]
        N = int(mem.sum())
        lev = (batch.levels[0, t] - p["level_mean"]) / p["level_std"]
        X = torch.cat([batch.X[0, t][mem], lev.expand(N, -1)], dim=1)        # X_{t-1}: N x 79
        wF = torch.softmax(p["Q"] @ (X @ WK).T / math.sqrt(dx), dim=1)       # eq. (1): K x N, softmax over stocks
        R = batch.R[0, t][mem]
        beta_T = wF.T @ torch.linalg.inv(wF @ wF.T + cfg.ridge * torch.eye(cfg.K, dtype=DT))   # N x K
        w_eps = torch.eye(N, dtype=DT) - beta_T @ wF                         # omega^eps
        assert torch.allclose(out["wF"][0, t][mem].T, wF, atol=1e-12)
        assert torch.allclose(out["eps"][0, t][mem], R - beta_T @ (wF @ R), atol=1e-12)
        assert torch.allclose(out["eps"][0, t][mem], w_eps @ R, atol=1e-12)
        w = w_eps.T @ out["w_port"][0, t - L][mem]                           # omega = (omega^eps)' omega^port
        assert torch.allclose(out["w"][0, t - L][mem], w / w.abs().sum(), atol=1e-12)
        assert torch.allclose(model.loadings(out["wF"])[0, t][mem], beta_T, atol=1e-9)
        # the residual portfolios carry almost no factor exposure: what is left is of the order of the ridge
        G = wF @ wF.T + cfg.ridge * torch.eye(cfg.K, dtype=DT)
        assert torch.allclose(wF @ w_eps, cfg.ridge * torch.linalg.solve(G, wF), atol=1e-12)
        assert (wF @ w_eps).abs().max() < 0.01 * wF.abs().max()


def test_portfolio_returns_costs_and_objective(cfg, panel):
    model, batch, out = run(cfg, panel)
    L = cfg.lookback
    w, R = out["w"][0].numpy(), batch.R[0].numpy()
    assert np.allclose(np.abs(w).sum(1), 1.0)                                # ||omega||_1 = 1 every day
    assert np.allclose(out["ret"][0].numpy(), (w[1:] * R[L + 1:]).sum(1))
    # R^port = eps' omega^port up to the normalization: the portfolio only holds residuals
    G, chol = model.gram(out["wF"])
    scale = np.abs(model.project(out["wF"][:, L:], (G[:, L:], chol[:, L:]), out["w_port"])[0].numpy()).sum(1)
    assert np.allclose(out["ret"][0].numpy(), (out["eps"][0, L + 1:].numpy() * out["w_port"][0, 1:].numpy()).sum(1) / scale[1:])
    turnover = np.abs(w[1:] - w[:-1]).sum(1)
    short = np.maximum(-w[1:], 0).sum(1)
    assert np.allclose(out["turnover"][0].numpy(), turnover) and np.allclose(out["short"][0].numpy(), short)
    net = out["ret"][0].numpy() - 0.0005 * turnover - 0.0001 * short
    objective, parts = model.objective(out)
    assert np.isclose(parts["sharpe_net"][0].item(), net.mean() / net.std() * np.sqrt(252))
    assert np.isclose(objective[0].item(), net.mean() / net.std() * np.sqrt(252) + 100 * out["ev"][0].item())


def test_explained_variance_term(cfg, panel):
    _, batch, out = run(cfg, panel)
    L = cfg.lookback
    eps, R, mem = (x[0, L + 1:].numpy() for x in (out["eps"], batch.R, batch.member))
    terms = [1 - eps[mem[:, i], i].var() / R[mem[:, i], i].var() for i in range(mem.shape[1]) if mem[:, i].sum() >= cfg.ev_min_days]
    assert len(terms) < mem.shape[1]                                         # some stocks have too few days and are left out
    assert np.isclose(out["ev"][0].item(), np.mean(terms))
    # a perfect factor model explains everything, a useless one nothing
    r = torch.randn(1, 30, 4, dtype=DT)
    m = torch.ones(1, 30, 4, dtype=torch.bool)
    assert torch.allclose(explained_variance(torch.zeros_like(r), r, m, 5), torch.ones(1, dtype=DT))
    assert torch.allclose(explained_variance(r, r, m, 5), torch.zeros(1, dtype=DT), atol=1e-12)


def test_no_signal_without_a_full_lookback_in_the_universe(cfg, arrays, panel):
    _, batch, out = run(cfg, panel, first=20, n_days=60)
    L = cfg.lookback
    permno = arrays["permno"]
    w_port = out["w_port"][0]
    n_blocked = 0
    for r in range(w_port.shape[0]):
        t = 20 + L + r
        for u in range(w_port.shape[1]):
            pid = panel.permnos[batch.cols[0, u]]
            has_history = all((permno[s] == pid).any() for s in range(t - L, t + 1))
            assert bool(batch.valid[0, L + r, u]) == has_history
            assert (w_port[r, u] != 0) == has_history
            n_blocked += (not has_history) and bool(batch.member[0, L + r, u])
    assert n_blocked > 20                                                    # the case does occur in the synthetic panel
    # such a stock can still be held through the factor hedge
    assert (out["w"][0][batch.member[0, L:] & ~batch.valid[0, L:]] != 0).any()


def test_weights_use_no_information_from_the_same_or_later_days(cfg, arrays):
    """Changing returns from day t0 on, and characteristics after t0, must leave the weights up to t0 unchanged."""
    first, n_days, t0 = 20, 40, 45
    changed = {k: v.copy() for k, v in arrays.items()}
    rng = np.random.default_rng(1)
    changed["ret"][t0:] = np.where(arrays["permno"][t0:] >= 0, rng.normal(scale=0.02, size=changed["ret"][t0:].shape), 0.0)
    changed["ranks"][t0 + 1:] = rng.uniform(-0.5, 0.5, size=changed["ranks"][t0 + 1:].shape)
    changed["levels"][t0 + 1:] += 1.0
    _, _, a = run(cfg, make_panel(arrays), first=first, n_days=n_days)
    _, _, b = run(cfg, make_panel(changed), first=first, n_days=n_days)
    row = t0 - first - cfg.lookback                                          # row of out["w"] that holds the weights of day t0
    assert torch.equal(a["w"][:, :row + 1], b["w"][:, :row + 1])
    assert not torch.allclose(a["w"][:, row + 1], b["w"][:, row + 1])       # the next day does react
    assert torch.equal(a["ret"][:, :row - 1], b["ret"][:, :row - 1])
    assert not torch.allclose(a["ret"][:, row - 1], b["ret"][:, row - 1])   # the return of t0 itself changed


def test_medians_and_rf_do_not_move_the_factors(cfg, arrays):
    """Eq. (1) as written: inputs shared by all stocks of a day only shift the scores, and the softmax removes that."""
    other = {k: v.copy() for k, v in arrays.items()}
    other["levels"] = other["levels"] * 3.0 + 5.0
    _, _, a = run(cfg, make_panel(arrays))
    _, _, b = run(cfg, make_panel(other))
    _, _, c = run(Config(**{**cfg.to_dict(), "use_levels": False}), make_panel(arrays))
    assert torch.allclose(a["wF"], b["wF"], atol=1e-12) and torch.allclose(a["w"], b["w"], atol=1e-9)
    assert torch.allclose(a["wF"], c["wF"], atol=1e-12) and torch.allclose(a["w"], c["w"], atol=1e-9)


def test_dropped_characteristic_groups_are_ignored(cfg, arrays):
    drop = Config(**{**cfg.to_dict(), "drop_groups": ("past_returns", "value")})
    cols = [CHARACTERISTICS.index(c) for g in drop.drop_groups for c in CHAR_GROUPS[g]]
    other = {k: v.copy() for k, v in arrays.items()}
    other["ranks"][:, :, cols] = np.random.default_rng(2).uniform(-0.5, 0.5, size=other["ranks"][:, :, cols].shape)
    _, _, a = run(drop, make_panel(arrays))
    _, _, b = run(drop, make_panel(other))
    _, _, full = run(cfg, make_panel(other))
    _, _, base = run(cfg, make_panel(arrays))
    assert torch.equal(a["w"], b["w"]) and sum(drop.char_mask) == N_CHAR - len(cols)
    assert not torch.allclose(full["w"], base["w"])                          # the full model does use those columns


def test_dropped_single_characteristics_are_ignored(cfg, arrays):
    drop = Config(**{**cfg.to_dict(), "drop_chars": ("Ret_D1", "Vol")})
    cols = [CHARACTERISTICS.index(c) for c in drop.drop_chars]
    other = {k: v.copy() for k, v in arrays.items()}
    other["ranks"][:, :, cols] = np.random.default_rng(3).uniform(-0.5, 0.5, size=other["ranks"][:, :, cols].shape)
    _, _, a = run(drop, make_panel(arrays))
    _, _, b = run(drop, make_panel(other))
    assert torch.equal(a["w"], b["w"]) and sum(drop.char_mask) == N_CHAR - 2
    assert "_dropchar-Ret_D1+Vol" in drop.name() and Config.from_dict(drop.to_dict()) == drop


def test_cumulative_input_is_the_running_sum_of_the_lagged_residuals(cfg, panel):
    model, batch, out = run(Config(**{**cfg.to_dict(), "cumulative": True}), panel)
    L = cfg.lookback
    x = out["eps"].unfold(1, L, 1)[:, : out["eps"].shape[1] - L] * cfg.resid_scale     # [1, T-L, U, L], oldest lag first
    path = torch.stack([x[..., : j + 1].sum(-1) for j in range(L)], dim=-1)
    with torch.no_grad():
        assert torch.allclose(out["w_port"], model.signal(path) * batch.valid[:, L:], atol=1e-12)
    _, _, plain = run(cfg, panel)
    assert torch.equal(out["eps"], plain["eps"]) and not torch.allclose(out["w_port"], plain["w_port"])


def test_risk_free_hurdle_of_the_printed_objective(cfg, panel):
    """On raw returns R + rf, with the paper's numerator, the net return falls by rf (1 - sum of the weights)."""
    model, batch, out = run(Config(**{**cfg.to_dict(), "rf_hurdle": True}), panel)
    L = cfg.lookback
    w, rf = out["w"][0, 1:].numpy(), batch.rf[0, L + 1:].numpy()
    assert np.allclose(rf, 1e-4)
    raw = (w * (batch.R[0, L + 1:].numpy() + rf[:, None])).sum(1)           # the portfolio's return on raw returns
    net = raw - 0.0005 * out["turnover"][0].numpy() - 0.0001 * out["short"][0].numpy() - rf
    _, parts = model.objective(out)
    assert np.isclose(parts["sharpe_net"][0].item(), net.mean() / net.std() * np.sqrt(252))
    _, plain = AFSAModel(cfg, [0], "cpu", DT).objective(out)                 # the default objective on the same portfolio
    assert not np.isclose(parts["sharpe_net"][0].item(), plain["sharpe_net"][0].item())


def reference_signal(model: AFSAModel, x: torch.Tensor) -> torch.Tensor:
    """LongConv written as in Appendix A and the reference implementation: whole sequences, FFT convolution."""
    cfg, L, d = model.cfg, model.cfg.lookback, model.cfg.d_model
    out = []
    for m in range(model.n_models):
        p = model.member_state(m)
        u = x[m][..., None] * p["in_w"] + p["in_b"]                          # [T, U, L, d]
        for l in range(cfg.n_layers):
            k = torch.sign(p[f"kernel{l}"]) * torch.clamp(p[f"kernel{l}"].abs() - cfg.squash, min=0)   # [d, L]
            u_f = torch.fft.rfft(u.transpose(-1, -2), n=2 * L)               # along time
            k_f = torch.fft.rfft(k, n=2 * L)
            y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L].transpose(-1, -2) + p[f"skip{l}"] * u    # K * u + D u
            z = F.gelu(y) @ p[f"lin_w{l}"].T + p[f"lin_b{l}"]
            u = F.layer_norm(u + z, (d,), p[f"norm_g{l}"], p[f"norm_b{l}"])
        out.append(u[..., -1, :] @ p["head_w"] + p["head_b"])                # read out the newest position
    return torch.stack(out)


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_longconv_equals_the_fft_convolution_of_the_appendix(n_layers):
    cfg = Config(K=2, d_model=6, lookback=9, n_layers=n_layers, squash=0.05)
    model = AFSAModel(cfg, [3, 4], "cpu", DT).eval()
    with torch.no_grad():                                                    # make every parameter matter
        for n, p in model.named_parameters():
            if n.startswith(("norm_", "lin_b", "head_b")):
                p.add_(torch.randn_like(p) * 0.3)
    x = torch.randn(2, 5, 7, cfg.lookback, dtype=DT)
    assert torch.allclose(model.signal(x), reference_signal(model, x), atol=1e-11)


def test_ridge_solve_gradient(cfg, panel, monkeypatch):
    """The hand-written gradient of the K x K solve against finite differences and against plain autograd."""
    import afsa.model

    def solve(A, B):                                                         # G = A'A + ridge I, as in the model
        G = A.transpose(-1, -2) @ A + 1e-3 * torch.eye(A.shape[-1], dtype=DT)
        return afsa.model._SpdSolve.apply(G, torch.linalg.cholesky(G.detach()), B)

    A = torch.randn(2, 3, 7, 4, dtype=DT, requires_grad=True)
    B = torch.randn(2, 3, 4, 1, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(solve, (A, B), atol=1e-7)

    def gradients():
        model = AFSAModel(cfg, [0, 1], "cpu", DT)
        days = torch.tensor([20, 60])[:, None] + torch.arange(40)[None]
        objective, _ = model.objective(model(panel.batch(days, cfg.lookback), train=False))
        objective.sum().backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()}

    ours = gradients()
    monkeypatch.setattr(afsa.model._SpdSolve, "apply", staticmethod(lambda G, chol, B: torch.linalg.solve(G, B)))
    plain = gradients()
    for name in ours:
        if name != "W_level":                                                # its gradient is rounding noise around zero
            assert torch.allclose(ours[name], plain[name], rtol=1e-6, atol=1e-9), name
    assert ours["Q"].abs().sum() > 0


def test_squash_is_a_soft_threshold():
    k = torch.tensor([-0.5, -0.05, 0.0, 0.02, 0.3])
    assert torch.allclose(squash(k, 0.1), torch.tensor([-0.4, 0.0, 0.0, 0.0, 0.2]))


def test_initialization():
    cfg = Config(K=30)
    p = init_member(cfg, 0)
    d, L = cfg.d_model, cfg.lookback
    t, h = torch.arange(1, L + 1) / L, torch.arange(1, d + 1) / d
    decay = torch.exp(-t[None, :] * ((d / 2) ** h)[:, None])                 # Appendix A, geometric decay
    x = p["kernel0"] / decay
    assert abs(x.mean()) < 0.1 and abs(x.std() - 1) < 0.1                    # x ~ N(0, 1)
    assert p["kernel0"].shape == (d, L) and p["Q"].shape == (30, cfg.d_x) and p["W_rank"].shape == (N_CHAR, cfg.d_x)
    assert decay[-1, -1] < 1e-6 < decay[0, -1]                               # fast and slow filters
    same, other = init_member(cfg, 0), init_member(cfg, 1)
    assert all(torch.equal(p[k], same[k]) for k in p) and not torch.equal(p["Q"], other["Q"])
    wide = init_member(Config(K=30, w_init_scale=3.0), 0)                    # only W^K changes
    assert torch.allclose(wide["W_rank"], 3 * p["W_rank"]) and torch.equal(wide["Q"], p["Q"]) and torch.equal(wide["kernel0"], p["kernel0"])


def test_models_in_one_batch_are_independent(cfg, panel):
    """Stacking models changes nothing: each one's outputs equal those of the model run alone."""
    seeds, firsts = (0, 1, 2), (20, 55, 100)
    model = AFSAModel(cfg, list(seeds), "cpu", DT)
    stat = [(panel.levels[f:f + 30].mean(0), panel.levels[f:f + 30].std(0)) for f in firsts]
    model.set_level_stats(torch.stack([s[0] for s in stat]), torch.stack([s[1] for s in stat]))
    days = torch.tensor(firsts)[:, None] + torch.arange(40)[None]
    together = model(panel.batch(days, cfg.lookback))
    obj_together, _ = model.objective(together)
    for m in range(3):
        single = model.select([m])
        batch = panel.batch(days[m:m + 1], cfg.lookback)
        alone = single(batch)
        u = int(batch.n_cols[0])
        for key in ("ret", "turnover", "short", "ev"):
            assert torch.allclose(together[key][m], alone[key][0], atol=1e-11), key
        assert torch.allclose(together["w"][m, :, :u], alone["w"][0], atol=1e-11)
        assert (together["w"][m, :, u:] == 0).all()                          # padding columns hold nothing
        assert torch.allclose(obj_together[m], single.objective(alone)[0][0], atol=1e-9)


def test_gradients_do_not_leak_between_models(cfg, panel):
    model = AFSAModel(cfg, [0, 1], "cpu", DT)
    days = torch.tensor([20, 60])[:, None] + torch.arange(40)[None]
    objective, _ = model.objective(model(panel.batch(days, cfg.lookback), train=True))
    objective[0].backward()
    for n, p in model.named_parameters():
        assert (p.grad[1] == 0).all(), n                                     # model 1 is untouched by model 0's loss
    assert model.Q.grad[0].abs().sum() > 0 and model.kernel0.grad[0].abs().sum() > 0
    assert model.W_level.grad.abs().max() < 1e-9 * model.W_rank.grad.abs().max()   # the inert inputs get no gradient


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_float32_agrees_with_cpu_float64(cfg, arrays):
    ref = run(cfg, make_panel(arrays, DT))[2]
    panel = type(make_panel(arrays)).from_arrays(**arrays, device="cuda", dtype=torch.float32)
    model = AFSAModel(cfg, [0], "cuda")
    lev = panel.levels[:20].double()
    model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
    with torch.no_grad():
        out = model(panel.batch((20 + torch.arange(40))[None], cfg.lookback))
    assert torch.allclose(out["w"].double().cpu(), ref["w"], atol=2e-4)
    assert torch.allclose(out["ret"].double().cpu(), ref["ret"], atol=1e-5)
