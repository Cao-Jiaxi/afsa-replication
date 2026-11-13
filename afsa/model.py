"""Attention Factors + LongConv (AFSA sections 3.2-3.3 and Appendix A), for M independent models at once.

Every parameter has a leading dimension M and model m only ever touches its own slice, so one pass over a
`Batch` trains or evaluates M separate models (different seeds and training windows). M = 1 is the plain model.

Index convention: row t of a batch carries the return R_t and what was known at the close of t-1, so the
weights computed for row t are the paper's omega_{t-1}.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from .config import N_CHAR, N_LEVEL, Config
from .panel import Batch

ATTENTION_PARAMS = ("W_rank", "W_level", "Q")       # the factor model; every other parameter is the sequence model


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
        cfg = self.cfg
        M, T, U, C = b.X.shape
        Qt = self.Q.transpose(1, 2) / math.sqrt(cfg.d_x)                                       # [M, d_x, K]
        scores = torch.bmm(b.X.reshape(M, T * U, C), torch.bmm(self.W_rank * self.char_mask[:, None], Qt)).view(M, T, U, cfg.K)
        if cfg.use_levels:
            # The medians and rf are the same for every stock of a day: they shift each factor's scores by a
            # constant, which the softmax over stocks removes. They are kept because the paper's X has 79 columns.
            lev = (b.levels - self.level_mean[:, None]) / self.level_std[:, None]             # [M, T, 40]
            scores = scores + torch.bmm(lev, torch.bmm(self.W_level * self.level_mask[:, None], Qt))[:, :, None, :]
        return torch.softmax(scores.masked_fill(~b.member[..., None], float("-inf")), dim=2)

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

    def signal(self, x: torch.Tensor, train: bool = False, gens=None, n_cols: list[int] | None = None) -> torch.Tensor:
        """omega^port = LongConv(lagged residuals). x: [M, T, U, L], oldest lag first. Returns [M, T, U].

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
        return torch.baddbmm(self.head_b[:, None, None], h, self.head_w[:, :, None]).view(M, T, U)

    # ------------------------------------------------------------------ the whole model
    def forward(self, b: Batch, train: bool = False, gens=None) -> dict[str, torch.Tensor]:
        """Runs T consecutive days. With L = lookback: rows 0..L-1 only supply residual history, rows L..T-1 get
        weights, and rows L+1..T-1 ("loss days", D = T-L-1 of them) get returns and costs.

        Returns wF [M,T,U,K], eps [M,T,U], w_port and w [M,D+1,U] (row L first), ret/turnover/short [M,D], ev [M].
        """
        cfg, L = self.cfg, self.cfg.lookback
        wF = self.factor_weights(b)
        G, chol = self.gram(wF)
        eps = self.project(wF, (G, chol), b.R)                                                 # residuals; 0 outside the universe
        T = eps.shape[1]
        x = eps.unfold(1, L, 1)[:, : T - L] * cfg.resid_scale                                  # [M, T-L, U, L]: rows t-L..t-1 for row t
        if cfg.cumulative:
            x = x.cumsum(-1)                                                                   # residual price path over the lookback
        n_cols = b.n_cols.tolist() if (train and gens is not None and cfg.dropout > 0) else None
        w_port = self.signal(x, train, gens, n_cols) * b.valid[:, L:]                          # no signal without L prior days
        w = self.project(wF[:, L:], (G[:, L:], chol[:, L:]), w_port)                           # (omega^eps)' omega^port
        w = w / w.abs().sum(-1, keepdim=True).clamp_min(1e-12)                                 # ||omega||_1 = 1
        return {
            "wF": wF, "eps": eps, "w_port": w_port, "w": w,
            "ret": (w[:, 1:] * b.R[:, L + 1:]).sum(-1),
            "turnover": (w[:, 1:] - w[:, :-1]).abs().sum(-1),
            "short": torch.relu(-w[:, 1:]).sum(-1),
            "ev": explained_variance(eps[:, L + 1:], b.R[:, L + 1:], b.member[:, L + 1:], cfg.ev_min_days),
            # R is in excess of rf stock by stock, so "ret" already charges rf on the net position sum(w). The
            # paper's formula charges it on one whole unit: the difference is rf * (1 - sum(w)).
            "rf_drag": b.rf[:, L + 1:] * (1 - w[:, 1:].sum(-1)),
        }

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
