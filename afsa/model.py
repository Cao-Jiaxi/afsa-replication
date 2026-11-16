"""Attention Factors + LongConv (AFSA sections 3.2-3.3 and Appendix A), for M independent models at once.

Every parameter has a leading dimension M and model m only ever touches its own slice, so one pass over a
`Batch` trains or evaluates M separate models (different seeds and training windows). M = 1 is the plain model.

Index convention: row t of a batch carries the return R_t and what was known at the close of t-1, so the
weights computed for row t are the paper's omega_{t-1}.

Two layouts of the sample feed the same model. A `Batch` (day x slot arrays, entry rules run and member) has one
cross-section per day, so each lag is the residual of its own day's universe. A `BlockBatch` (month blocks, entry
rule history) recomputes the lags of day t over day t's universe: factors and residuals are computed on every row of
the block of t's month, over the month's list, and the lags are the rows before t in that block.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from .config import CHARACTERISTICS, N_CHAR, N_LEVEL, NEUTRAL_SETS, Config
from .panel import Batch, BlockBatch

ATTENTION_PARAMS = ("W_rank", "W_level", "Q")       # the factor model; every other parameter is the sequence model
STATE_FEATURES = ("position", "target", "adds")     # what the learned trading rule sees of the book (Config.trade_state)


def squash(kernel: torch.Tensor, lam: float) -> torch.Tensor:
    """Squash operator of Appendix A: sign(K) * max(|K| - lam, 0)."""
    return torch.sign(kernel) * torch.relu(kernel.abs() - lam)


def init_member(cfg: Config, seed: int) -> dict[str, torch.Tensor]:
    """Initial parameters of one model, drawn from its own generator, so they do not depend on the batch it is in."""
    g = torch.Generator().manual_seed(int(seed))
    d, dx, L = cfg.d_model, cfg.d_x, cfg.lookback

    def uniform(shape, fan_in):                      # default of nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        return (torch.rand(shape, generator=g) * 2 - 1) / math.sqrt(fan_in)

    W = uniform((N_CHAR + N_LEVEL, dx), N_CHAR + N_LEVEL) * cfg.w_init_scale  # W^K, 79 x d_x
    p = {
        "W_rank": W[:N_CHAR].clone(),                                        # rows of the 39 ranks
        "W_level": W[N_CHAR:].clone(),                                       # rows of the 39 medians and rf
        "Q": torch.randn((cfg.K, dx), generator=g) * cfg.q_init_std,         # K queries
        "in_w": uniform((d,), 1), "in_b": uniform((d,), 1),                  # Linear 1 -> d on each lagged residual
    }
    # Appendix A: K_t^(h) = x * exp(-(t/T) * (d/2)^(h/d)), x ~ N(0, 1), 1 <= t <= T, 1 <= h <= d.
    t = torch.arange(1, L + 1, dtype=torch.float32) / L
    rate = (d / 2) ** (torch.arange(1, d + 1, dtype=torch.float32) / d)
    decay = torch.exp(-rate[:, None] * t[None, :])                           # [d, L]; column 0 is the newest lag
    for l in range(cfg.n_layers):
        p[f"kernel{l}"] = torch.randn((d, L), generator=g) * cfg.kernel_init_std * decay
        p[f"skip{l}"] = torch.randn((d,), generator=g)                       # D
        p[f"lin_w{l}"] = uniform((d, d), d)
        p[f"lin_b{l}"] = uniform((d,), d)
        p[f"norm_g{l}"] = torch.ones(d)
        p[f"norm_b{l}"] = torch.zeros(d)
    p["head_w"] = uniform((d,), d)
    p["head_b"] = uniform((), d)
    if cfg.trade_head:                  # the learned trading rule (AFSAModel.dials): zero parameters are the fixed band
        p["delta_w"], p["delta_b"] = torch.zeros(d), torch.zeros(())
        p["eta_w"], p["eta_b"] = torch.zeros(d), torch.zeros(())
        if cfg.trade_state:             # what the rule sees of the book (trade_book_state); zero = the rule without it
            p["delta_v"], p["eta_v"] = torch.zeros(len(STATE_FEATURES)), torch.zeros(len(STATE_FEATURES))
        if cfg.trade_market:            # what it sees of the day's market (AFSAModel.dial_logits); zero = without it
            p["delta_u"], p["eta_u"] = torch.zeros(d), torch.zeros(d)
    return p


class _SpdSolve(torch.autograd.Function):
    """X = G^-1 B for a symmetric positive definite G whose Cholesky factor is given.

    The gradient uses dX = G^-1 (dB - dG X) directly. Differentiating through the factorization gives the same
    numbers but costs several K x K x K products per day, which dominates the training time for large K.
    """

    @staticmethod
    def forward(ctx, G, chol, B):
        X = torch.cholesky_solve(B, chol)
        ctx.save_for_backward(chol, X)
        return X

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_X):
        chol, X = ctx.saved_tensors
        grad_B = torch.cholesky_solve(grad_X, chol)
        return -torch.matmul(grad_B, X.transpose(-1, -2)), None, grad_B


def explained_variance(eps: torch.Tensor, R: torch.Tensor, member: torch.Tensor, min_days: int) -> torch.Tensor:
    """(1/N) sum_i (1 - Var(eps_i) / Var(R_i)) over the stocks with at least `min_days` days in the block.

    eps, R, member: [M, T, U], with eps and R equal to 0 where the stock is not a member. Returns [M].
    """
    n = member.to(eps.dtype).sum(1)                                           # [M, U] days in the universe
    n1 = n.clamp_min(1)
    var_e = (eps * eps).sum(1) / n1 - (eps.sum(1) / n1) ** 2
    var_r = (R * R).sum(1) / n1 - (R.sum(1) / n1) ** 2
    ok = (n >= min_days) & (var_r > 1e-12)
    ratio = var_e / torch.where(ok, var_r, torch.ones_like(var_r))
    return ((1 - ratio) * ok).sum(1) / ok.sum(1).clamp_min(1)


def trade_book(w: torch.Tensor, member: torch.Tensor, band: float | torch.Tensor, eta: float | torch.Tensor) -> torch.Tensor:
    """The trading layer (CLAUDE.md section 7): from the targets w [M, D+1, U] (one column per stock) to the books held.

    Day 0's book is day 0's target. Each later day: gap = target - book; trade = eta * sign(gap) * max(|gap| - band, 0);
    draft = book + trade, 0 outside the day's universe. The stocks that trade (or are closed) are resized together so
    that ||book||_1 = 1, and the others keep their position exactly. If only new positions trade (the resize factor would
    not be positive), the whole draft is rescaled; an empty draft takes the target. Differentiable almost everywhere.

    band and eta are numbers (Config.band, Config.band_eta), or tensors shaped like w that give each stock its own value
    on each day (the learned trading rule, Config.trade_head): day t's trade uses day t's values.
    """
    book, books = w[:, 0], [w[:, 0]]
    for t in range(1, w.shape[1]):
        band_t = band[:, t] if torch.is_tensor(band) else band
        eta_t = eta[:, t] if torch.is_tensor(eta) else eta
        book = book_step(book, w[:, t], member[:, t], band_t, eta_t)
        books.append(book)
    return torch.stack(books, 1)


def book_step(book: torch.Tensor, target: torch.Tensor, on: torch.Tensor, band, eta) -> torch.Tensor:
    """One day of `trade_book`: from yesterday's book [M, U] to today's, given today's target and membership [M, U] and
    the day's band width and speed (numbers, or [M, U] tensors)."""
    gap = target - book
    trade = eta * torch.sign(gap) * torch.relu(gap.abs() - band)
    draft = (book + trade) * on
    moved = (trade != 0) | (~on & (book != 0))
    size = draft.abs()
    held = (size * ~moved).sum(-1, keepdim=True)
    movers = (size * moved).sum(-1, keepdim=True)
    total = held + movers
    c = (1 - held) / torch.where(movers > 0, movers, torch.ones_like(movers))
    resized = torch.where(moved, c * draft, draft)
    uniform = draft / torch.where(total > 0, total, torch.ones_like(total))
    return torch.where((movers > 0) & (c > 0), resized, torch.where(total > 0, uniform, target))


def state_features(book: torch.Tensor, target: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """The rule's view of each stock's book on one day: log(1 + n|h|), log(1 + n|w|) and sign(h) sign(w - h), with h the
    position held since yesterday, w today's target and n the day's number of members, so an average position counts
    as 1. The last is +1 if trading towards the target adds to the position, -1 if it reduces or reverses it, and 0 for
    no position. book, target [M, U], n [M, 1]; returns [M, U, 3]."""
    return torch.stack([torch.log1p(n * book.abs()), torch.log1p(n * target.abs()),
                        torch.sign(book) * torch.sign(target - book)], -1)


def trade_book_state(w: torch.Tensor, member: torch.Tensor, a: torch.Tensor, k: torch.Tensor, has_signal: torch.Tensor,
                     v_delta: torch.Tensor, v_eta: torch.Tensor, band0: float, band_max: float,
                     eta0: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`trade_book` with dials that also see the book (Config.trade_state; analysis/rl/TRADING_STATE.md). On day t,
    with f the stock's `state_features` from yesterday's book and today's target:

        delta = band_max * sigmoid(a + v_delta' f),    eta = sigmoid(k + v_eta' f)

    for members with a signal, and (band0, eta0) for the others. a, k [M, D+1, U] are the logits of `AFSAModel.dial_logits`
    (they do not depend on the book) and v_delta, v_eta [M, 3] the rule's weights on f, so v = 0 gives `trade_book` with
    the dials of `AFSAModel.dials`. Day 0 holds its target; its dials are those of holding it (stored, never traded).
    Returns the books and the dials used, [M, D+1, U] each."""
    n = member.sum(-1, keepdim=True).to(w.dtype)
    book, books, deltas, etas = w[:, 0], [], [], []
    for t in range(w.shape[1]):
        f = state_features(book, w[:, t], n[:, t])
        delta = torch.where(has_signal[:, t], band_max * torch.sigmoid(a[:, t] + torch.einsum("muf,mf->mu", f, v_delta)), band0)
        eta = torch.where(has_signal[:, t], torch.sigmoid(k[:, t] + torch.einsum("muf,mf->mu", f, v_eta)), eta0)
        if t > 0:
            book = book_step(book, w[:, t], member[:, t], delta, eta)
        books.append(book), deltas.append(delta), etas.append(eta)
    return torch.stack(books, 1), torch.stack(deltas, 1), torch.stack(etas, 1)


class AFSAModel(nn.Module):
    """M independent Attention Factor + LongConv models (one per seed in `seeds`)."""

    def __init__(self, cfg: Config, seeds, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.n_models = len(seeds)
        members = [init_member(cfg, s) for s in seeds]
        for name in members[0]:
            self.register_parameter(name, nn.Parameter(torch.stack([m[name] for m in members])))
        keep = torch.tensor(cfg.char_mask, dtype=torch.float32)
        self.register_buffer("char_mask", keep)                                               # [39] Table 3
        self.register_buffer("level_mask", torch.cat([keep, torch.ones(1)]))                  # [40]
        self.register_buffer("level_mean", torch.zeros(self.n_models, N_LEVEL))               # training-window statistics
        self.register_buffer("level_std", torch.ones(self.n_models, N_LEVEL))
        self.to(device=device, dtype=dtype)
        self.solve_dtype = torch.float64 if cfg.solve_dtype == "float64" else dtype
        self.neutral_cols = [CHARACTERISTICS.index(c) for c in NEUTRAL_SETS[cfg.neutral]] if cfg.neutral else []

    # ------------------------------------------------------------------ bookkeeping
    def param_groups(self, weight_decay: float) -> list[dict]:
        """AdamW groups: weight decay on the sequence model only (Table 4)."""
        attn = [p for n, p in self.named_parameters() if n in ATTENTION_PARAMS]
        seq = [p for n, p in self.named_parameters() if n not in ATTENTION_PARAMS]
        return [{"params": attn, "weight_decay": 0.0}, {"params": seq, "weight_decay": weight_decay}]

    def set_level_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Mean and standard deviation of the 40 per-day inputs over each model's training window. [M, 40]"""
        std = torch.where(std > 0, std, torch.ones_like(std))                 # constant in the window (rf = 0): leave as is
        self.level_mean.copy_(mean.to(self.level_mean))
        self.level_std.copy_(std.to(self.level_std))

    def member_state(self, m: int) -> dict[str, torch.Tensor]:
        """Parameters and input statistics of model m, on the CPU."""
        state = {n: p.detach()[m].cpu().clone() for n, p in self.named_parameters()}
        state["level_mean"], state["level_std"] = self.level_mean[m].cpu().clone(), self.level_std[m].cpu().clone()
        return state

    @classmethod
    def from_states(cls, cfg: Config, states: list[dict], device="cpu", dtype=torch.float32) -> "AFSAModel":
        model = cls(cfg, seeds=[0] * len(states), device=device, dtype=dtype)
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(torch.stack([s[n] for s in states]).to(p))
            model.level_mean.copy_(torch.stack([s["level_mean"] for s in states]).to(model.level_mean))
            model.level_std.copy_(torch.stack([s["level_std"] for s in states]).to(model.level_std))
        return model.eval()

    def select(self, idx: list[int]) -> "AFSAModel":
        """A copy holding only the models in `idx` (used to evaluate a large batch of models in pieces)."""
        return AFSAModel.from_states(self.cfg, [self.member_state(m) for m in idx], self.Q.device, self.Q.dtype)

    # ------------------------------------------------------------------ factor model (section 3.2)
    def factor_weights(self, b: Batch) -> torch.Tensor:
        """Eq. (1): omega^F = Softmax(Q (X W^K)^T / sqrt(d_x)), softmax over the day's universe. [M, T, U, K]"""
        return self._attention(b.X, b.levels, b.member)

    def _attention(self, X: torch.Tensor, levels: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        """factor_weights on X [M, T, U, 39], levels [M, T, 40] and member [M, T, U]; every row needs a member."""
        cfg = self.cfg
        M, T, U, C = X.shape
        Qt = self.Q.transpose(1, 2) / math.sqrt(cfg.d_x)                                       # [M, d_x, K]
        scores = torch.bmm(X.reshape(M, T * U, C), torch.bmm(self.W_rank * self.char_mask[:, None], Qt)).view(M, T, U, cfg.K)
        if cfg.use_levels:
            # The medians and rf are the same for every stock of a day: they shift each factor's scores by a
            # constant, which the softmax over stocks removes. They are kept because the paper's X has 79 columns.
            lev = (levels - self.level_mean[:, None]) / self.level_std[:, None]               # [M, T, 40]
            scores = scores + torch.bmm(lev, torch.bmm(self.W_level * self.level_mask[:, None], Qt))[:, :, None, :]
        return torch.softmax(scores.masked_fill(~member[..., None], float("-inf")), dim=2)

    def gram(self, wF: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """omega^F omega^F' + ridge I and its Cholesky factor. [M, T, K, K] each."""
        G = torch.matmul(wF.transpose(2, 3), wF).to(self.solve_dtype)
        G = G + self.cfg.ridge * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        return G, torch.linalg.cholesky_ex(G.detach()).L

    @staticmethod
    def project(wF: torch.Tensor, gram: tuple[torch.Tensor, torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        """omega^eps v = v - beta' (omega^F v), with beta' = omega^F' (omega^F omega^F' + ridge I)^-1.

        omega^eps is symmetric, so the same map gives the residuals (v = R) and the asset weights (v = omega^port).
        v: [M, T, U]. The N x N matrix is never built.
        """
        G, chol = gram
        Fv = torch.matmul(v.unsqueeze(-2), wF).transpose(-1, -2)                               # factor exposure [M, T, K, 1]
        coef = _SpdSolve.apply(G, chol, Fv.to(chol.dtype)).to(v.dtype)
        return v - torch.matmul(wF, coef).squeeze(-1)

    def loadings(self, wF: torch.Tensor) -> torch.Tensor:
        """beta' = omega^F' (omega^F omega^F' + ridge I)^-1: loading of each stock on each factor. [M, T, U, K]"""
        chol = self.gram(wF)[1]
        return torch.cholesky_solve(wF.transpose(2, 3).to(chol.dtype), chol).transpose(2, 3).to(wF.dtype)

    # ------------------------------------------------------------------ sequence model (section 3.3, Appendix A)
    def _keep(self, shape: tuple[int, ...], like: torch.Tensor, gens, n_cols: list[int] | None) -> torch.Tensor:
        """Dropout mask for activations of `shape` = [M, T, U, ..., d], divided by the keep probability. [M, n, d]"""
        q = 1 - self.cfg.dropout
        if gens is None:
            keep = torch.empty(shape, device=like.device, dtype=like.dtype).bernoulli_(q)
        else:           # one stream per model, sized by that model's own columns: independent of the rest of the batch
            keep = torch.zeros(shape, device=like.device, dtype=like.dtype)
            for m, g in enumerate(gens):
                own = torch.empty((shape[1], n_cols[m], *shape[3:]), device=like.device, dtype=like.dtype)
                keep[m, :, :n_cols[m]] = own.bernoulli_(q, generator=g)
        return keep.mul_(1 / q).flatten(1, -2)

    def _block(self, l: int, u: torch.Tensor, conv: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
        """GELU, dropout, linear layer, residual connection and LayerNorm around a convolution output. [M, n, d]"""
        z = F.gelu(conv)
        if keep is not None:
            z = z * keep
        z = torch.baddbmm(getattr(self, f"lin_b{l}")[:, None, :], z, getattr(self, f"lin_w{l}").transpose(1, 2))
        h = F.layer_norm(u + z, (u.shape[-1],))
        return torch.addcmul(getattr(self, f"norm_b{l}")[:, None, :], h, getattr(self, f"norm_g{l}")[:, None, :])

    def signal(self, x: torch.Tensor, train: bool = False, gens=None, n_cols: list[int] | None = None,
               hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """omega^port = LongConv(lagged residuals). x: [M, T, U, L], oldest lag first. Returns [M, T, U], and with
        `hidden` also the hidden state that the read-out reads, [M, T, U, d] (the input of `dials`).

        Each lagged residual is embedded as u[l, h] = in_w[h] x[l] + in_b[h]; a block computes y = K * u + D u
        along time (causal: y[i] = sum_j u[j] K[i - j]) followed by GELU, dropout, a linear layer, a residual
        connection and LayerNorm; the newest position of the last block is read out by a linear layer.
        Only that position is needed from the last block, so it is computed directly rather than as a whole
        sequence by FFT; both give the same number (tests/test_model.py).
        """
        cfg = self.cfg
        M, T, U, L = x.shape
        d, last = cfg.d_model, cfg.n_layers - 1
        drop = train and cfg.dropout > 0
        w, b = self.in_w, self.in_b
        xf = x.reshape(M, T * U, L)
        k = squash(getattr(self, f"kernel{last}"), cfg.squash)                                 # [M, d, L]; lag 0 = newest
        if cfg.n_layers == 1:
            # The newest output is linear in the L lagged residuals, so the embedding, the convolution and the
            # skip term collapse into one L x d matrix E and no [.., L, d] tensor is ever built:
            #   sum_lag K[h, lag] u[L-1-lag, h] + D[h] u[L-1, h] = sum_l x[l] E[l, h] + c[h]
            E = k.flip(-1) * w[:, :, None]                                                     # [M, d, L]
            E = E + F.pad((self.skip0 * w)[:, :, None], (L - 1, 0))                            # D acts on the newest lag
            c = b * (k.sum(-1) + self.skip0)
            conv = torch.baddbmm(c[:, None, :], xf, E.transpose(1, 2))                         # [M, T*U, d]
            u_last = torch.addcmul(b[:, None, :], xf[:, :, -1:], w[:, None, :])
        else:
            u = torch.addcmul(b[:, None, None, :], xf[..., None], w[:, None, None, :])         # [M, T*U, L, d]
            lag = torch.arange(L, device=x.device)[:, None] - torch.arange(L, device=x.device)[None, :]   # i - j
            for l in range(last):                                                              # whole sequences
                kl = squash(getattr(self, f"kernel{l}"), cfg.squash)
                causal = kl[:, :, lag.clamp_min(0)] * (lag >= 0)                               # [M, d, L, L]: K[h, i-j], j <= i
                conv = torch.einsum("mnjh,mhij->mnih", u, causal) + getattr(self, f"skip{l}")[:, None, None, :] * u
                keep = self._keep((M, T, U, L, d), x, gens, n_cols) if drop else None
                u = self._block(l, u.reshape(M, -1, d), conv.reshape(M, -1, d), keep).view(M, T * U, L, d)
            u_last = u[:, :, -1, :]
            conv = torch.einsum("mnlh,mhl->mnh", u, k.flip(-1)) + getattr(self, f"skip{last}")[:, None, :] * u_last
        h = self._block(last, u_last, conv, self._keep((M, T, U, d), x, gens, n_cols) if drop else None)
        s = torch.baddbmm(self.head_b[:, None, None], h, self.head_w[:, :, None]).view(M, T, U)
        return (s, h.view(M, T, U, d)) if hidden else s

    # ------------------------------------------------------------------ learned trading rule (analysis/rl/TRADING_RULE.md)
    def dials(self, z: torch.Tensor, has_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each stock's band width and speed on each day (cfg.trade_head; TRADING_RULE.md, steps 3c and 3d), from the
        sequence model's hidden state z [M, T, U, d]:

            delta = head_band_max * sigmoid(logit(head_band0 / head_band_max) + delta_w' z + delta_b)
            eta   = sigmoid(logit(head_eta0) + eta_w' z + eta_b)

        The fixed offsets make zero parameters the fixed band (head_band0, head_eta0), so the weight decay of the
        sequence model shrinks the rule towards it. Members without a signal (has_signal [M, T, U] false) keep the fixed
        band. With cfg.trade_market the logits also hold the market term of `dial_logits`. With cfg.trade_state the
        dials also depend on the book, and `trade_book_state` sets them day by day from `dial_logits` instead.
        Returns delta and eta, [M, T, U] each."""
        cfg = self.cfg
        a, k = self.dial_logits(z, has_signal)
        delta = torch.where(has_signal, cfg.head_band_max * torch.sigmoid(a), cfg.head_band0)
        eta = torch.where(has_signal, torch.sigmoid(k), cfg.head_eta0)
        return delta, eta

    def dial_logits(self, z: torch.Tensor, has_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The logits of the band width and the speed before any view of the book, [M, T, U] each:

            a = logit(head_band0 / head_band_max) + delta_w' z + delta_b  [+ delta_u' zbar_t]
            k = logit(head_eta0) + eta_w' z + eta_b  [+ eta_u' zbar_t]

        The bracketed market term is on with cfg.trade_market (analysis/rl/TRADING_STATE.md): zbar_t [d] is the mean of z
        over the day's stocks with a signal, so it shifts every stock's dials of that day by the same amount."""
        cfg = self.cfg
        a0 = math.log(cfg.head_band0 / (cfg.head_band_max - cfg.head_band0))
        k0 = math.log(cfg.head_eta0 / (1 - cfg.head_eta0))
        a = torch.einsum("mtud,md->mtu", z, self.delta_w) + self.delta_b[:, None, None]
        k = torch.einsum("mtud,md->mtu", z, self.eta_w) + self.eta_b[:, None, None]
        if cfg.trade_market:
            g = has_signal.to(z.dtype)
            zbar = torch.einsum("mtud,mtu->mtd", z, g) / g.sum(-1, keepdim=True).clamp_min(1)    # [M, T, d]
            a = a + torch.einsum("mtd,md->mt", zbar, self.delta_u)[:, :, None]
            k = k + torch.einsum("mtd,md->mt", zbar, self.eta_u)[:, :, None]
        return a + a0, k + k0

    def dial_inputs(self, z: torch.Tensor, has_signal: torch.Tensor) -> dict[str, torch.Tensor]:
        """What `_portfolio` needs to run the learned rule, on the rows of z [M, T, U, d]: the dials themselves
        ("delta", "eta"), or with cfg.trade_state their logits and who has a signal ("a", "k", "has"), since the dials
        are then set day by day from the book. [M, T, U] each."""
        if self.cfg.trade_state:
            a, k = self.dial_logits(z, has_signal)
            return {"a": a, "k": k, "has": has_signal}
        delta, eta = self.dials(z, has_signal)
        return {"delta": delta, "eta": eta}

    # ------------------------------------------------------------------ neutrality (analysis/gap/EXTENSION.md)
    @staticmethod
    def center(w_port: torch.Tensor, has_signal: torch.Tensor) -> torch.Tensor:
        """The signals minus their mean over the day's stocks that have one (cfg.signal_center). [M, D, U]"""
        n = has_signal.sum(-1, keepdim=True).clamp_min(1)
        return (w_port - w_port.sum(-1, keepdim=True) / n) * has_signal

    def neutralize(self, w: torch.Tensor, X: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        """w minus its least-squares projection, over the day's members, on the constant and the ranks X of the
        characteristics of cfg.neutral: afterwards sum(w) = 0 and sum(w * rank) = 0 for each of them.

        w [M, D, U] (0 outside the members, and it stays 0 there), X [M, D, U, p] those ranks, known before the day,
        member [M, D, U]. A rank that is the same for every member that day (all missing: rank 0) is skipped by the
        tiny ridge. The p + 1 normal equations are formed in the precision of w and only solved in that of the ridge
        solve, so a float32 book is neutral to float32 accuracy of the book before the projection: at most 6.2e-7 of
        unit gross exposure in the stored books of the neutral trials (analysis/gap/EXTENSION.md, batch 2)."""
        A = torch.cat([torch.ones_like(w)[..., None], X], dim=-1) * member[..., None]          # [M, D, U, p + 1]
        At = A.transpose(-1, -2)
        G = torch.matmul(At, A).to(self.solve_dtype)
        G = G + 1e-8 * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        coef = torch.linalg.solve(G, torch.matmul(At, w[..., None]).to(G.dtype)).to(w.dtype)
        return w - torch.matmul(A, coef).squeeze(-1)

    # ------------------------------------------------------------------ the whole model
    def forward(self, b: Batch | BlockBatch, train: bool = False, gens=None) -> dict[str, torch.Tensor]:
        """Runs T consecutive days. With L = lookback: rows 0..L-1 only supply residual history, rows L..T-1 get
        weights, and rows L+1..T-1 ("loss days", D = T-L-1 of them) get returns and costs.

        Returns wF [M,T,U,K], eps [M,T,U], w_port and w [M,D+1,U] (row L first), ret/turnover/short [M,D], ev [M],
        and the residuals, excess returns and membership of the rows that get weights (eps_d, R_d, member_d
        [M,D+1,U]); with cfg.trade_head also each stock's band width and speed, delta and eta [M,D+1,U], and with
        cfg.trade_state their logits before the view of the book and who has a signal (dial_a, dial_k, has). A BlockBatch
        gives the same outputs, except wF, eps and w_port (`_forward_blocks`).
        """
        cfg, L = self.cfg, self.cfg.lookback
        if isinstance(b, BlockBatch) != (cfg.entry_rule == "history"):
            raise ValueError("entry_rule history reads the month blocks (BlockPanel), the other rules the day x slot arrays "
                             "(Panel): load the sample with afsa.panel.load_panel")
        if isinstance(b, BlockBatch):
            return self._forward_blocks(b, train, gens)
        wF = self.factor_weights(b)
        G, chol = self.gram(wF)
        eps = self.project(wF, (G, chol), b.R)                                                 # residuals; 0 outside the universe
        T = eps.shape[1]
        x = eps.unfold(1, L, 1)[:, : T - L] * cfg.resid_scale                                  # [M, T-L, U, L]: rows t-L..t-1 for row t
        if cfg.cumulative:
            x = x.cumsum(-1)                                                                   # residual price path over the lookback
        n_cols = b.n_cols.tolist() if (train and gens is not None and cfg.dropout > 0) else None
        has_signal = b.member if cfg.entry_rule == "member" else b.valid                       # "run": no signal without L prior days
        if cfg.trade_head:                                                                     # the learned trading rule
            w_port, z = self.signal(x, train, gens, n_cols, hidden=True)
            dial = self.dial_inputs(z, has_signal[:, L:])
        else:
            w_port, dial = self.signal(x, train, gens, n_cols), None
        w_port = w_port * has_signal[:, L:]
        if cfg.signal_center:
            w_port = self.center(w_port, has_signal[:, L:])
        w = self.project(wF[:, L:], (G[:, L:], chol[:, L:]), w_port)                           # (omega^eps)' omega^port
        if cfg.neutral:
            w = self.neutralize(w, b.X[:, L:][..., self.neutral_cols], b.member[:, L:])
        if not cfg.nosignal_hedge:                                                             # no position without a signal
            w = w * has_signal[:, L:]
        w = w / w.abs().sum(-1, keepdim=True).clamp_min(1e-12)                                 # ||omega||_1 = 1
        return {"wF": wF, "eps": eps, "w_port": w_port, "w": w,
                **self._portfolio(w, b.R[:, L:], eps[:, L:], b.member[:, L:], b.rf[:, L:], dial)}

    def _forward_blocks(self, b: BlockBatch, train: bool, gens) -> dict[str, torch.Tensor]:
        """forward on the month blocks (entry_rule history). Factors and residuals are computed on every row of the NB
        blocks over the block's list, and the lags of a day are the L rows before it in its own block: residuals over
        that day's universe, also on days before a stock entered it. The day's weights are hedged with its own row.

        Returns wF [M,NB*R,S,K] and eps [M,NB*R,S] on the blocks' rows, w_port [M,D+1,S] in the slots of each day's
        block, and the rest as forward, in one column per stock.
        """
        cfg, L = self.cfg, self.cfg.lookback
        M, NB, R, S, C = b.X.shape
        rows = NB * R
        wF = self._attention(b.X.view(M, rows, S, C), b.levels.view(M, rows, -1), b.member.view(M, rows, S))
        G, chol = self.gram(wF)
        R_rows = b.R.view(M, rows, S)
        eps = self.project(wF, (G, chol), R_rows)                                              # residuals over each block's list
        m = torch.arange(M, device=eps.device)[:, None]
        q = b.day_block * R + b.day_row                                                        # [M, D+1] row of each day
        lags = q[:, :, None] + torch.arange(-L, 0, device=eps.device)                          # its L rows before, oldest first
        x = eps[m[:, :, None], lags].transpose(2, 3) * cfg.resid_scale                         # [M, D+1, S, L]
        if cfg.cumulative:
            x = x.cumsum(-1)                                                                   # residual price path over the lookback
        n_cols = [S] * M if (train and gens is not None and cfg.dropout > 0) else None
        if cfg.trade_head:                                                                     # the learned trading rule
            w_port, z = self.signal(x, train, gens, n_cols, hidden=True)
            dial = self.dial_inputs(z, b.signal)
        else:
            w_port, dial = self.signal(x, train, gens, n_cols), None
        w_port = w_port * b.signal
        if cfg.signal_center:
            w_port = self.center(w_port, b.signal)
        w = self.project(wF[m, q], (G[m, q], chol[m, q]), w_port)                              # (omega^eps)' omega^port on the day
        if cfg.neutral:                                                                        # the day's own row of its block
            cols = torch.as_tensor(self.neutral_cols, dtype=torch.long, device=w.device)
            X_day = b.X.view(M, rows, S, C)[m[:, :, None], q[:, :, None], :, cols].transpose(2, 3)   # [M, D+1, S, p]
            w = self.neutralize(w, X_day, b.member.view(M, rows, S)[m, q])
        if not cfg.nosignal_hedge:                                                             # no position without a signal
            w = w * b.signal
        w = w / w.abs().sum(-1, keepdim=True).clamp_min(1e-12)                                 # ||omega||_1 = 1
        on, idx = b.slot >= 0, b.slot.clamp_min(0)

        def per_stock(v: torch.Tensor) -> torch.Tensor:                                        # [M, D+1, S] -> [M, D+1, U]
            return torch.gather(v, 2, idx) & on if v.dtype == torch.bool else torch.gather(v, 2, idx) * on
        w = per_stock(w)
        if dial is not None:                                                                   # each stock's dials in its column
            dial = {key: per_stock(v) for key, v in dial.items()}
        return {"wF": wF, "eps": eps, "w_port": w_port, "w": w,
                **self._portfolio(w, per_stock(R_rows[m, q]), per_stock(eps[m, q]), on, b.rf, dial)}

    def _portfolio(self, w: torch.Tensor, R: torch.Tensor, eps: torch.Tensor, member: torch.Tensor,
                   rf: torch.Tensor, dial: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Return, costs and explained variance of the weights w [M, D+1, U]: day 0 holds the weights before the first
        loss day, days 1..D are the loss days. R, eps, member [M, D+1, U] and rf [M, D+1] are on the same days.
        With the trading layer on (cfg.band > 0 or cfg.band_eta < 1) or the learned trading rule (cfg.trade_head: `dial`
        is `dial_inputs` on the same days), return and costs are the book's (`trade_book`, or `trade_book_state` with
        cfg.trade_state); "book" is w otherwise. The rule's band widths and speeds are returned as "delta" and "eta"
        [M, D+1, U]; with cfg.trade_state also its logits before the view of the book and who has a signal ("dial_a",
        "dial_k", "has"), which a book carried across years needs to be scored (analysis/rl/trading_state.py)."""
        cfg = self.cfg
        if dial is not None and cfg.trade_state:
            book, delta, eta = trade_book_state(w, member, dial["a"], dial["k"], dial["has"], self.delta_v, self.eta_v,
                                                cfg.head_band0, cfg.head_band_max, cfg.head_eta0)
        elif dial is not None:
            delta, eta = dial["delta"], dial["eta"]
            book = trade_book(w, member, delta, eta)
        elif cfg.band > 0 or cfg.band_eta < 1:
            book = trade_book(w, member, cfg.band, cfg.band_eta)
        else:
            book = w
        out = {
            "eps_d": eps, "R_d": R, "member_d": member, "book": book,
            "ret": (book[:, 1:] * R[:, 1:]).sum(-1),
            "turnover": (book[:, 1:] - book[:, :-1]).abs().sum(-1),
            "short": torch.relu(-book[:, 1:]).sum(-1),
            "ev": explained_variance(eps[:, 1:], R[:, 1:], member[:, 1:], cfg.ev_min_days),
            # R is in excess of rf stock by stock, so "ret" already charges rf on the net position sum(w). The
            # paper's formula charges it on one whole unit: the difference is rf * (1 - sum(w)).
            "rf_drag": rf[:, 1:] * (1 - book[:, 1:].sum(-1)),
        }
        if dial is not None:
            out["delta"], out["eta"] = delta, eta
        if dial is not None and cfg.trade_state:
            out["dial_a"], out["dial_k"], out["has"] = dial["a"], dial["k"], dial["has"]
        return out

    def objective(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Net Sharpe ratio + lambda_var * explained variance, per model ([M]); to be maximized."""
        cfg = self.cfg
        net = out["ret"] - cfg.cost_trade * out["turnover"] - cfg.cost_short * out["short"]
        if cfg.rf_hurdle:
            net = net - out["rf_drag"]
        scale = math.sqrt(cfg.sharpe_days)
        sharpe = net.mean(1) / net.std(1, correction=0).clamp_min(1e-12) * scale
        parts = {
            "objective": sharpe + cfg.lambda_var * out["ev"],
            "sharpe_net": sharpe,
            "sharpe_gross": out["ret"].mean(1) / out["ret"].std(1, correction=0).clamp_min(1e-12) * scale,
            "explained_var": out["ev"],
            "turnover": out["turnover"].mean(1),
            "short": out["short"].mean(1),
        }
        return parts["objective"], parts
