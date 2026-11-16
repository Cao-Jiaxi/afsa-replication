# The learned trading rule: flow and equations

This specifies design 5.1 of [EXPLORATION.md](EXPLORATION.md) for implementation; the plain-language version is in the same section. Steps 1–4 are the current model ([CLAUDE.md](../../CLAUDE.md) §5, [afsa/model.py](../../afsa/model.py)), and **new** marks what changes.

**Status (2026-11-16): built, tested, and trained once on D at K = 30.** The rule sits behind `--trade_head true`, which is off by default, and [tests/test_trade_head.py](../../tests/test_trade_head.py) checks it. Its code is listed at the end of this file.

**First results** ([TRADING_RULE_RESULTS.md](TRADING_RULE_RESULTS.md), written by [trading_rule.py](trading_rule.py)):
- **Decision.** The pre-registered decision (1996–1997 only) is *inconclusive*.
- **1998–2021** (reported, not used to decide):
  - The rule nets 1.89 as a model and 2.33 as a seed ensemble. The fixed band trained inside the model nets 1.77 and 2.17. The differences are +0.26 [−0.07, +0.64] and +0.16 [−0.19, +0.56].
  - The gain comes from 1998–2013. In 2022–2025 the rule is lower: −0.22 for the model and −0.35 for the ensemble, both within the noise.
- **What it learned.** It trades much less: 0.33 a day against 0.54. Its median band is 4–6 times the starting 0.04%, at 0.15–0.23% of the book. Its speed is slower, a median η of 0.59–0.85. The widths differ across stocks only modestly: they are narrower for larger and more active targets.

## Notation

| Symbol | Meaning |
| --- | --- |
| row t | What is known at the close of t−1, together with the return of day t. Weights computed on row t (the paper's $\omega_{t-1}$) earn $R_t$ |
| $U_t$, $N_t$ | Day t's universe (the list members) and its size: 500–564 on D's buffered list |
| $m_{i,t}$ | 1 if stock i is a member on day t, else 0 |
| $g_{i,t}$ | 1 if member i gets a signal on day t (rebuilt history: listed for more than 30 trading days, with annual accounts entered) |
| $X_{t-1}$ | Characteristic ranks in $[-0.5,0.5]^{N_t\times39}$ (rank 0 when empty) |
| $R_t$ | Excess returns of day t, $\in\mathbb R^{N_t}$ |
| $K,\ d_x,\ L,\ d$ | Factors; embedding size 32; lags 30; hidden units 32 |
| $\lambda_{ridge},\ c,\ c_s,\ \lambda_{Var}$ | 1e-4, 5 bp, 1 bp and 100 in D's reference (K = 30) |
| $\sigma(x)$ | The logistic function $1/(1+e^{-x})$ |
| $z_{i,t}$ | LongConv's hidden state for stock i on day t, $\in\mathbb R^d$ |
| $s_{i,t}$ | The signal (the paper's $\omega^{port}_i$) |
| $\omega_t$ | The target portfolio |
| $h_t$ | The book: the positions actually held |
| $\delta_{i,t},\ \eta_{i,t}$ | **New:** stock i's band width and speed on day t |

Parameters:
- $\theta_F=(W^K,Q)$: the factor model.
- $\theta_S$: LongConv (embedding $a,b$; kernel $\mathcal K$; skip $D$; $W_1,c_1$; LayerNorm $\gamma,\beta_{LN}$) and the signal read-out $(w_s,b_s)$.
- **New:** $\theta_T=(w_\delta,b_\delta,w_\eta,b_\eta)$, which is $2(d+1)=66$ numbers per model.

## Flow

```
X_{t-1} ─► (1) attention ω^F_t ─► (2) factors, residuals ─► stock i's 30 lags x_{i,t}
                                                                   │
                                                  (3a) LongConv ─► hidden state z_{i,t}
                                                                   │
                  ┌──────────────────────────┬─────────────────────┴───────────────┐
           (3b) signal s_{i,t}        (3c) new: band δ_{i,t}           (3d) new: speed η_{i,t}
                  │                          │                                     │
      (4) hedge + scale ─► target ω_t        │                                     │
                  └────────────────► (5) new: trading rule ◄───────────────────────┘
                                 book h_t = rule(h_{t-1}, ω_t, δ_t, η_t)
                                                │
                       (6) return and costs of the book ─► (7) net Sharpe + λ_Var·EV
```

## Step 1. Attention factor weights (unchanged)

$$\omega^F_t=\operatorname{Softmax}_{i\in U_t}\Big(\frac{Q\,(X_{t-1}W^K)^\top}{\sqrt{d_x}}\Big)\in\mathbb R^{K\times N_t}.$$

Row k of $\omega^F_t$ is factor k's weights over the day's members, and it sums to 1. The 39 medians and $r_f$ add the same constant to every stock's score for a factor and day, so the softmax cancels them.

## Step 2. Factors, loadings, residuals (unchanged)

$$F_t=\omega^F_tR_t,\qquad G_t=\omega^F_t\omega^{F\top}_t+\lambda_{ridge}I_K,\qquad \beta^\top_t=\omega^{F\top}_tG_t^{-1},$$

$$\varepsilon_t=R_t-\beta^\top_tF_t=\omega^\varepsilon_tR_t,\qquad \omega^\varepsilon_t=I_{N_t}-\omega^{F\top}_tG_t^{-1}\omega^F_t\quad(\text{symmetric; }G_t\text{ is solved in float64}).$$

**The lags under rebuilt history (D).** Day t's 30 lags are recomputed inside day t's universe. For each lag day $s=t-30,\dots,t-1$:

$$\varepsilon^{(t)}_s=R^{(t)}_s-\beta^{(t)\top}_sF^{(t)}_s .$$

Here the factor model runs over day t's members, with their characteristics of s−1 and their returns of s, including days before a stock entered the list. Stock i's input, in per cent:

$$x_{i,t}=100\,\big(\varepsilon^{(t)}_{i,t-30},\dots,\varepsilon^{(t)}_{i,t-1}\big)\in\mathbb R^{L}.$$

## Step 3. The sequence model: one hidden state, three read-outs

**3a. Hidden state (unchanged).** One LongConv block, read at the newest position. Lag positions $l=1,\dots,L$ run from oldest to newest.

- Embedding of each lag (Linear 1 → d): $u_{l,k}=a_k\,x_{i,t,l}+b_k$, for $k=1,\dots,d$.
- Convolution at the newest position, with the squashed kernel $\tilde{\mathcal K}=\operatorname{sign}(\mathcal K)\max(|\mathcal K|-\lambda_{sq},0)$:

$$y_k=\sum_{j=0}^{L-1}\tilde{\mathcal K}_{k,j}\,u_{L-j,k}+D_k\,u_{L,k}.$$

- Block (GELU, dropout during training, linear layer, residual connection, LayerNorm):

$$z_{i,t}=\gamma\odot\operatorname{LN}\big(u_L+W_1\operatorname{drop}(\operatorname{GELU}(y))+c_1\big)+\beta_{LN}\in\mathbb R^d.$$

**3b. Signal (unchanged).**

$$s_{i,t}=g_{i,t}\,\big(w_s^\top z_{i,t}+b_s\big).$$

**3c. Band width (new).**

$$\delta_{i,t}=\begin{cases}\delta_{\max}\,\sigma\big(\alpha_0+w_\delta^\top z_{i,t}+b_\delta\big), & g_{i,t}=1,\\ \delta_0, & g_{i,t}=0,\end{cases}\qquad \alpha_0=\operatorname{logit}(\delta_0/\delta_{\max}).$$

**3d. Speed (new).**

$$\eta_{i,t}=\begin{cases}\sigma\big(\kappa_0+w_\eta^\top z_{i,t}+b_\eta\big), & g_{i,t}=1,\\ \eta_0, & g_{i,t}=0,\end{cases}\qquad \kappa_0=\operatorname{logit}(\eta_0).$$

**Constants.**

| Constant | Value | Why |
| --- | --- | --- |
| $\delta_0$ | 4e-4 | D's band chosen on 1996–1997 |
| $\delta_{\max}$ | 3.2e-3 | The top of the band grid |
| $\eta_0$ | 0.99 | Practically the fixed band's η = 1 |
| $\alpha_0$ | logit(0.125) = −1.946 | Follows from $\delta_0$ and $\delta_{\max}$ |
| $\kappa_0$ | logit(0.99) = 4.595 | Follows from $\eta_0$ |

$\delta$ is measured in units of the book, which has unit gross exposure: 4e-4 is 0.04% of the book, about a fifth of an average position.

**Why the fixed offsets $\alpha_0,\kappa_0$.**
- The new parameters start at zero, so every stock starts at the fixed band.
- AdamW's weight decay acts on every sequence-model parameter, biases included (`AFSAModel.param_groups`). With the offsets it shrinks the dials toward the fixed band. Plain sigmoids would instead pull them toward $\delta_{\max}/2$ and $\eta=1/2$.
- Stocks without a signal keep the fixed band, since their history is incomplete.

## Step 4. Target portfolio (unchanged)

$$\tilde\omega_t=\omega^\varepsilon_t\,s_t=s_t-\omega^{F\top}_tG_t^{-1}\big(\omega^F_ts_t\big),\qquad \omega_t=\frac{\tilde\omega_t}{\|\tilde\omega_t\|_1}.$$

The target hedges the signals against the factors, with unit gross exposure. Members without a signal hold positions through the hedge, as now. On the month blocks, $\omega_t$, $\delta_t$ and $\eta_t$ are computed on day t's own row and then gathered from the block's slots into one column per stock, so a stock keeps its column across month ends.

## Step 5. The trading rule (new; generalizes `trade_book`)

The book $h_t$ has one column per stock and is 0 for a non-member. A batch starts from its target, $h_0=\omega_0$, at no cost. For $t=1,\dots,T$ ($T=126$ in training):

**5a. Gap.**

$$\Delta_{i,t}=\omega_{i,t}-h_{i,t-1}.$$

**5b. Trade:** skip a gap within the band, and trade a share η of the rest.

$$\tau_{i,t}=\eta_{i,t}\,\operatorname{sign}(\Delta_{i,t})\,\max\big(|\Delta_{i,t}|-\delta_{i,t},\,0\big).$$

**5c. Draft:** members only, so a stock that leaves the list is closed.

$$\hat h_{i,t}=m_{i,t}\,\big(h_{i,t-1}+\tau_{i,t}\big).$$

**5d. Movers and sizes.**

$$\mathcal M_t=\{i:\tau_{i,t}\ne0\}\cup\{i:m_{i,t}=0,\ h_{i,t-1}\ne0\},\qquad A_t=\sum_{i\notin\mathcal M_t}|\hat h_{i,t}|,\qquad B_t=\sum_{i\in\mathcal M_t}|\hat h_{i,t}|.$$

**5e. Rescale to unit gross exposure, resizing only the movers:**

$$h_{i,t}=\begin{cases}\rho_t\,\hat h_{i,t}, & i\in\mathcal M_t,\\ \hat h_{i,t}, & i\notin\mathcal M_t,\end{cases}\qquad \rho_t=\frac{1-A_t}{B_t},\qquad\text{if }B_t>0\text{ and }\rho_t>0.$$

In the rare case where the held positions alone already reach unit size, $h_t=\hat h_t/(A_t+B_t)$; if the draft is empty, $h_t=\omega_t$. These are exactly `trade_book`'s rules.

**Special cases.**
- $\delta\equiv0,\ \eta\equiv1$: the book is the target, i.e. the current model.
- $\delta\equiv\delta,\ \eta\equiv\eta$ (constants): today's `trade_book(δ, η)`.
- Zero new parameters: the fixed band at $\delta_0=$ 4e-4, with $\eta_0=0.99$.

## Step 6. Return and costs of the book

These are computed as the trading layer does now. Row t's book earns row t's return:

$$r_t=h_t^\top R_t,\qquad \mathrm{TO}_t=\|h_t-h_{t-1}\|_1,\qquad \mathrm{SH}_t=\|\max(-h_t,0)\|_1,$$

$$n_t=r_t-c\,\mathrm{TO}_t-c_s\,\mathrm{SH}_t .$$

## Step 7. Objective (unchanged form)

Over a batch of $T=126$ loss days:

$$J=\sqrt{252}\;\frac{\bar n}{\operatorname{sd}(n)}+\lambda_{Var}\,\frac1{N'}\sum_{i}\Big(1-\frac{\operatorname{Var}(\varepsilon_i)}{\operatorname{Var}(R_i)}\Big).$$

- $\bar n$ and $\operatorname{sd}(n)$ are taken over the batch's days, with denominator T.
- The explained-variance average runs over the $N'$ stocks with at least 21 days in the batch.
- It uses the residuals, not the book, so the dials do not enter it.

**Training.** Maximize J over $\theta_F,\theta_S,\theta_T$ jointly with AdamW: learning rate 3e-3, weight decay 0.05 on $\theta_S$ and $\theta_T$ and none on $\theta_F$. Each epoch has 16 batches; train 30 epochs and keep the last model.

## How the gradient reaches the dials

Outside the band ($|\Delta_{i,t}|>\delta_{i,t}$), the derivatives of 5b are:

$$\frac{\partial\tau_{i,t}}{\partial\delta_{i,t}}=-\eta_{i,t}\operatorname{sign}(\Delta_{i,t}),\qquad \frac{\partial\tau_{i,t}}{\partial\eta_{i,t}}=\operatorname{sign}(\Delta_{i,t})\big(|\Delta_{i,t}|-\delta_{i,t}\big),\qquad \frac{\partial\tau_{i,t}}{\partial h_{i,t-1}}=-\eta_{i,t}.$$

Inside the band, all three are 0.

**How long a dial matters.** Before the rescale, $\partial\hat h_{i,t}/\partial h_{i,t-1}$ is $1-\eta_{i,t}$ on a day the stock trades and 1 on a day it is skipped. So a dial set on day t keeps acting on the book, and through it on later returns and costs. The effect fades at rate $1-\eta$ while the stock trades and does not fade while it is skipped. Backpropagation through the loop over the batch's days adds up these effects:

$$\frac{\partial J}{\partial\delta_{i,t}}=\sum_{u\ge t}\frac{\partial J}{\partial h_u}\,\frac{\partial h_u}{\partial\delta_{i,t}}.$$

Three consequences:
1. A stock's dials learn only from the days it trades.
2. The rescale couples the stocks: $\rho_t$ depends on every mover, so each dial also nudges the other movers' sizes.
3. The signal and the target also receive gradient through the book, so the network can aim differently once it knows how each stock will be traded. The fixed band trained inside the model already did this ([BAND_TRAINED.md](../gap/BAND_TRAINED.md)).

## Evaluation

**A single model.** Run steps 1–6 over every day of a test year with that year's model. Carry the book over from the previous year's last day, starting from cash in 1996 or 1998, and score it at the paper's costs (5 bp, 1 bp), as in BAND_TRAINED.md.

**The seed ensemble.** Average the M seeds' targets and dials, then run the rule once:

$$\bar\omega_t=\frac{\sum_m\omega^{(m)}_t}{\big\|\sum_m\omega^{(m)}_t\big\|_1},\qquad \bar\delta_{i,t}=\frac1M\sum_m\delta^{(m)}_{i,t},\qquad \bar\eta_{i,t}=\frac1M\sum_m\eta^{(m)}_{i,t}.$$

**Comparisons**, decided on 1996–1997 only and reported on 1998–2021 and 2022–2025 with the paired block bootstrap:
- D's model with the band added after training (δ = 4e-4);
- D's model trained through the fixed band (δ = 4e-4, η = 1).

Write the §7 row before any test year runs.

## The code (built 2026-11-16)

| Where | What |
| --- | --- |
| `afsa/config.py` | `trade_head` (default False) and the constants `head_band0` = 4e-4, `head_band_max` = 3.2e-3, `head_eta0` = 0.99. The configuration is refused if `band` or `band_eta` is also set, or if the constants are out of order. With the flag off, every run folder and `config.json` stays valid (CLAUDE.md §3); a run with it on gets its own folder, `..._trade_head-True` |
| `afsa/model.py`, `init_member` | $\theta_T$ as `delta_w`, `delta_b`, `eta_w`, `eta_b`, created as zeros and only when the flag is on. They draw nothing from the random stream, so all other parameters start exactly as before |
| `AFSAModel.signal(..., hidden=True)` | Also returns the hidden state $z_{i,t}$ |
| `AFSAModel.dials` | Steps 3c–3d: δ and η from z, with the fixed band kept for members without a signal |
| `forward`, `_forward_blocks` | Compute the dials on each day's rows. On the month blocks, gather them per stock the way `w` is |
| `trade_book` | Takes δ and η as numbers, as before, or as [M, D+1, U] tensors indexed at day t inside the loop |
| `_portfolio` | Uses the rule when the flag is on and returns `delta` and `eta` |
| `afsa/train.py`, `evaluate` | With `--save-weights`, also keeps the book, `delta` and `eta` (float32, test days × stocks) |

**Tests** (`tests/test_trade_head.py`, 9 tests). With the flag off, every earlier test passes unchanged (252 in all). The new tests check that:
- the flag is off by default and gets its own folder name, and bad settings are refused;
- the parameters start as before, with the rule at zero;
- constant dials reproduce `trade_book(δ, η)` bit for bit;
- per-stock dials match a numpy loop written from step 5;
- zero rule parameters give the fixed band and leave the target unchanged;
- `dials` follows 3c–3d, and members without a signal keep the fixed band in the forward pass;
- the gradient reaches the rule, and training moves it;
- `evaluate` keeps the book and the dials;
- on a fixed list, the month blocks train and evaluate as the day arrays do, dials included.

**To run it.** Train only after a CLAUDE.md §7 row is written and the user gives the go-ahead. D's validation years, then the test years:

```powershell
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --trade_head true --train_years 6 --first-year 1996 --last-year 1997 --seeds 0 1 2 --save-weights
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --trade_head true --first-year 1998 --last-year 2025 --seeds 0 1 2 --save-weights
```

Comparing it with the band added after training and with the fixed band trained inside the model, from cash with books carried across years, still needs a script like `analysis/gap/band_trained.py`. That script would use the saved `delta` and `eta`.
