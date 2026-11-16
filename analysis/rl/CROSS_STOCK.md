# Cross-stock attention on the signal

This is design 5.4 of [EXPLORATION.md](EXPLORATION.md), "stocks that see each other", in the form the exploration proposed: a cross-stock attention layer after the per-stock sequence model, so each stock's signal can depend on the other stocks of the day. Its lightest form, a mean-field view for the trading rule's dials, was tested first ([TRADING_STATE.md](TRADING_STATE.md), `--trade_market`). This file puts attention on the signal itself.

**Status (2026-11-16): built, tested, and trained once on D at K = 30.** The flag `--cross_heads` is off by default (0), and [tests/test_cross_stock.py](../../tests/test_cross_stock.py) checks the layer. The pre-registered test is a CLAUDE.md §7 row of 2026-11-16.

**Results** ([CROSS_STOCK_RESULTS.md](CROSS_STOCK_RESULTS.md), written by [cross_stock.py](cross_stock.py); every path from cash, at the paper's costs):
- **Decision.** The pre-registered decision (1996–1997 only, against D) is *inconclusive*: the model +0.04 [−0.12, +0.20], the ensemble + band +0.05 [−0.09, +0.23].
- **1998–2021** (reported, never used to decide): slightly below D for every object, within the noise.
  - Model: 1.19 against 1.31 (−0.08 [−0.20, +0.04]).
  - Ensemble + band: 2.05 against 2.10 (−0.06 [−0.18, +0.08]).
  - 2022–2025: −0.20 and −0.12. Over 1998–2025 the model's −0.10 [−0.21, +0.00] sits at the edge of the interval.
- **The layer is used, and it does not help.**
  - From 2006 on, the message is 1.8–2.5 times the size of the hidden state it is added to, and it carries 34–64% of the signal's cross-sectional variance.
  - In 1998–2005 the message carries 3%.
  - So the windows that end after 2005 lean on the other stocks, and what they learn there does not carry to the next year. That fits the risk named above: capacity added where the model already fits its window far better than it trades.

## Why

- **Step 3 is already a team of agents.** One agent per stock shares the parameters, and each acts on its own stock's 30 residual lags alone. The team is trained on the portfolio objective, and step 4's factor hedge mixes the agents' outputs. That is centralized training with decentralized execution (p459).
- **Communication lets an agent's action depend on the others'.** The deck's form is graph communication (p461): $m_i=\mathrm{AGG}_j\,\psi(h_i,h_j)$, then $h_i'=\phi(h_i,m_i)$. Attention is that form with a learned, content-dependent aggregation.
- **What it could learn.** A stock's reversal signal may matter more when similar stocks moved too, or less when the whole cross-section moved. The factor model removes common moves from the residuals, but it is linear and has only K factors.
- **The risk.** The layer adds capacity to a model that already fits its window far better than it trades the next year (in-sample net SR 5–7, out of sample 1.1–1.4; [REPORT.md](../gap/REPORT.md)). The companion paper's set-sequence model is the larger version of this idea.

## The layer

LongConv's hidden state for stock i on day t is $z_{i,t}\in\mathbb R^d$ (d = 32; [TRADING_RULE.md](TRADING_RULE.md), step 3a). **New:** before the read-out, each stock adds a message read from the day's stocks with a signal ($g_{j,t}=1$):

$$q_{i}=W_qz_{i,t}+b_q,\qquad k_{j}=W_kz_{j,t}+b_k,\qquad v_{j}=W_vz_{j,t}+b_v\qquad(\text{each split into }H\text{ heads of size }E),$$

$$a_{h,i}=\sum_{j:\,g_{j,t}=1}\frac{\exp\big(q_{h,i}^\top k_{h,j}/\sqrt E\big)}{\sum_{j':\,g_{j',t}=1}\exp\big(q_{h,i}^\top k_{h,j'}/\sqrt E\big)}\,v_{h,j},\qquad z'_{i,t}=z_{i,t}+W_o\big[a_{1,i};\dots;a_{H,i}\big]+b_o .$$

The signal is then read from $z'$: $s_{i,t}=g_{i,t}(w_s^\top z'_{i,t}+b_s)$. With the learned trading rule on, its dials read $z'$ too.

- **Defaults when on:** H = 2 heads of size E = 8 (`--cross_heads 2`, `--cross_head_dim 8`). That adds $3(dHE+HE)+(HEd+d)$ = 2,128 parameters per model, against about 2,200 in the sequence model and 3,500 in the factor model at K = 30.
- **Start.** $W_o$ and $b_o$ start at zero, so the model starts exactly as without the layer. The projections are drawn from the model's own generator after every other parameter, so the other parameters start as before. All of them are under the sequence model's weight decay.
- **What it reads.** Only the same day and the same model. A day with no stock with a signal reads nothing. Every input is known at the close of the day before, so the layer adds no look-ahead.
- **Compute.** Scaled dot-product attention with the memory-efficient kernel on the GPU, which never builds the N × N score matrix. A pass of 36 models × 127 days × 600 slots takes 0.22 s forward and backward at 3.5 GB.

## The code (built 2026-11-16)

| Where | What |
| --- | --- |
| `afsa/config.py` | `cross_heads` (default 0 = off), `cross_head_dim` (default 8). With the layer off, every run folder and `config.json` stays valid (CLAUDE.md §3) |
| `afsa/model.py`, `init_member` | $W_q,W_k,W_v,b_q,b_k,b_v$ as `xq_w` … `xv_b`, drawn last; $W_o,b_o$ as `xo_w`, `xo_b`, zero |
| `AFSAModel.cross` | The layer; `signal(..., keys=)` applies it before the read-out, with the day's signal mask as the keys |

**Tests** (`tests/test_cross_stock.py`, 4 tests). With the layer off, every earlier test passes unchanged. The new tests check that:
- the layer is off by default and names its folder; the other parameters start as before; with zero output weights the targets and returns equal the model without it, bit for bit;
- the message matches the formula written out with an explicit softmax; members without a signal are not read; a day without keys reads nothing; permuting the stocks permutes the result;
- the weights use no information from the same or later days, and stacked models equal models run alone;
- the gradient reaches the layer and training moves it; on a fixed list, the month blocks train and evaluate as the day arrays do.

**To run it.** D's validation years, then the test years (`--max-models 18` keeps the attention's memory within the card):

```powershell
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --cross_heads 2 --train_years 6 --first-year 1996 --last-year 1997 --seeds 0 1 2 --save-weights --max-models 18
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --cross_heads 2 --first-year 1998 --last-year 2025 --seeds 0 1 2 --save-weights --max-models 18
.venv\Scripts\python.exe analysis\rl\cross_stock.py
```
