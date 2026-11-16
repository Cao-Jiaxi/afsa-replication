# The trading rule that sees what it holds, and the market

This extends the learned trading rule ([TRADING_RULE.md](TRADING_RULE.md)), which is design 5.1 of [EXPLORATION.md](EXPLORATION.md). The extension gives that rule two more inputs:
- **The book** (`--trade_state true`). This is design 5.2, "holdings as an input", placed where the input acts on the position itself.
- **The market** (`--trade_market true`). This is a light form of design 5.4, "stocks that see each other".

The steps and symbols are those of TRADING_RULE.md; **new** marks what changes.

**Status (2026-11-16): built, tested, and trained once on D at K = 30.** Both views are off by default, and [tests/test_trade_state.py](../../tests/test_trade_state.py) checks them. The pre-registered test is a CLAUDE.md §7 row of 2026-11-16.

**First results** ([TRADING_STATE_RESULTS.md](TRADING_STATE_RESULTS.md), written by [trading_state.py](trading_state.py); every path from cash, at the paper's costs):
- **Decision.** The pre-registered decision (1996–1997 only, against the rule without views) is *inconclusive* for both arms:
  - **S**: model +0.06 [−0.18, +0.30], ensemble +0.15 [−0.08, +0.41];
  - **SM**: model +0.01 [−0.28, +0.32], ensemble +0.11 [−0.17, +0.43].
- **1998–2021** (reported, never used to decide). Net SR, model / ensemble:

  | Object | 1998–2021 | 2022–2025 |
  | --- | --- | --- |
  | Rule | 1.89 / 2.33 | −0.42 / −0.48 |
  | S | 1.85 / 2.16 | −0.33 / −0.32 |
  | SM | 1.99 / **2.66** | −0.41 / −0.33 |
  | Fixed band trained | 1.77 / 2.17 | −0.25 / −0.13 |

  - Against the rule, on 1998–2021, S is lower: −0.12 [−0.24, +0.06] as a model and −0.17 [−0.28, +0.02] as an ensemble.
  - SM is higher: +0.23 [−0.19, +0.60] and +0.33 [−0.15, +0.81]. Both are within the noise.
  - Against the fixed band trained inside the model, SM gains +0.49 as a model [+0.19, +0.77] and +0.49 as an ensemble [+0.20, +0.80]; both intervals exclude 0. Its ensemble's 2.66 is above the paper's K = 30 (2.28), but an ensemble is not the paper's object.
  - In 2022–2025 neither view helps: both sit at the rule's level, below D + band after (−0.01 / +0.10).
- **What the book's view learned.** In every window, the band is wider and the speed lower for trades that reduce a position than for trades that add to it. Larger positions get a wider band and a slower speed. So the rule adds readily, reduces reluctantly, and lets large positions ride.
  - In 1998–2005, S's median band is 17.8 bp for reducing trades and 14.1 bp for adding ones. The rule without the view has none of this asymmetry: 15.3 and 15.4 bp.
  - In later windows the band's asymmetry fades while the speed's grows. In 2014–2025, S's median speed is 0.76 for adding trades and 0.60 for reducing ones.
- **What the market's view adds.** SM trades least (0.28 a day; its ensemble 0.26) at the lowest σ (its ensemble 3.39% a year against 3.80%). In 2014–2021 its speed fell to a median of 0.32–0.49, against 0.56–0.59 for the rule. There its ensemble netted 1.36 against 0.91, which is +0.45 but with an interval of [−0.30, +1.06].

**A firmer test of SM** ([SM_FIRM_RESULTS.md](SM_FIRM_RESULTS.md); CLAUDE.md §7 row "a firmer test of SM"). SM was run on fresh cells (K = 8, K = 100, and K = 30 with seeds 3–5) and compared with D + band after.
- **Decision.** The pre-registered decision (1996–1997, the cells pooled) is *better* for the model, +0.79 [+0.02, +1.62], but not for the ensemble, +0.23 [−0.53, +1.03].
- **1998–2021, the fresh cells pooled:** +0.08 for the model and +0.20 for the ensemble, both within the noise.
  - The gain is in 1998–2005 (+0.73 and +0.49).
  - 2022–2025 is worse: −0.86 [−1.63, −0.11] for the model and −0.74 [−1.46, +0.00] for the ensemble.
- **With six seeds at K = 30,** SM's ensemble nets 2.61 against 2.22 on 1998–2021 (+0.39 [+0.08, +0.71]).
- **At K = 8 it trails;** at K = 100 its ensemble leads by +0.26 [−0.03, +0.57].
- **Pessimism,** repeated on these cells, is kept for none.

## Why

- **The deck's portfolio MDP puts the holdings in the state** (p482). Without them the policy has restricted observation (p285): it cannot do "I already hold most of this, so leave it".
  - The learned rule sees the book only through the gap $\Delta=\omega-h$.
  - Its band width and speed are set from the stock's residual history $z$ alone, before the book is known.
  - So the rule cannot treat a position it holds differently from one it does not.
- **Recurrent reinforcement learning feeds back the last position.** Moody and Saffell (2001) train the trader's position $F_t$ as a function of recent returns *and* $F_{t-1}$. That is the classic direct-RL design for trading with costs.
- **Statistical-arbitrage rules are asymmetric.** The s-score rule of the paper's PCA + OU benchmark (Avellaneda and Lee 2010) opens a position at one threshold and closes it at another. A band whose width depends on whether a trade adds to the position or reduces it can learn that asymmetry, in either direction.
- **The market view is communication between the agents.** Step 3 is a team of per-stock agents that share parameters (p459). The simplest message is the mean of the others' states (graph communication with mean aggregation, p461). Every stock's dials then shift together with the day's state. The learned rule's bands widened from one window to the next (0.15% in 1998–2005, 0.23% in 2022–2025); this view lets them move within a year too.

## Flow

```
                 z_{i,t} (LongConv) ──┬──► (3b) signal ─► (4) target ω_t ─────────────┐
                                      │                                              │
  new: z̄_t = mean of z over the day ─┼──► (3c') logits a_{i,t}, k_{i,t}              │
       (with --trade_market)          │            │                                 │
                                      │            ▼                                 ▼
  yesterday's book h_{t-1} ──────────────► new (5'): view f_{i,t} ─► δ_{i,t}, η_{i,t} ─► (5) book h_t
       (with --trade_state)                                                          │
                                                                    (6) return and costs ─► (7) objective
```

## Step 3c′. The logits (new with `--trade_market`)

$$a_{i,t}=\alpha_0+w_\delta^\top z_{i,t}+b_\delta+u_\delta^\top\bar z_t,\qquad k_{i,t}=\kappa_0+w_\eta^\top z_{i,t}+b_\eta+u_\eta^\top\bar z_t,$$

$$\bar z_t=\frac{\sum_i g_{i,t}\,z_{i,t}}{\max\big(1,\sum_i g_{i,t}\big)} .$$

- The mean runs over the day's stocks with a signal ($g_{i,t}=1$), on the day's own row of its month block.
- $u_\delta,u_\eta\in\mathbb R^d$ add 2d = 64 parameters per model.
- Without the flag, the market terms are absent and $a,k$ are the logits of TRADING_RULE.md, steps 3c–3d.

## Step 5′. The view of the book (new with `--trade_state`)

On day t, before the trade, each stock's view is computed from yesterday's book $h_{i,t-1}$ and today's target $\omega_{i,t}$. With $N_t$ members, an average position counts as 1:

$$f_{i,t}=\Big(\log(1+N_t|h_{i,t-1}|),\ \ \log(1+N_t|\omega_{i,t}|),\ \ \operatorname{sign}(h_{i,t-1})\operatorname{sign}(\omega_{i,t}-h_{i,t-1})\Big).$$

- The first two entries are the size of the position held and the size of the target.
- The third is +1 if trading towards the target would add to the position, −1 if it would reduce or reverse it, and 0 if there is no position.

The day's dials are then:

$$\delta_{i,t}=\begin{cases}\delta_{\max}\,\sigma\big(a_{i,t}+v_\delta^\top f_{i,t}\big), & g_{i,t}=1,\\ \delta_0, & g_{i,t}=0,\end{cases}\qquad \eta_{i,t}=\begin{cases}\sigma\big(k_{i,t}+v_\eta^\top f_{i,t}\big), & g_{i,t}=1,\\ \eta_0, & g_{i,t}=0.\end{cases}$$

- $v_\delta,v_\eta\in\mathbb R^3$ add 6 parameters per model.
- The book then follows step 5 of TRADING_RULE.md (gap, trade, draft, movers-only rescale) with these dials.
- Since the dials depend on yesterday's book, they are set inside the loop over the batch's days, one day at a time (`trade_book_state`).

**Special cases.** All new parameters start at zero and are under the sequence model's weight decay, so they shrink toward the rule of TRADING_RULE.md:
- $v=0$ and $u=0$ give that rule exactly (a test checks this bit for bit);
- with every new and old rule parameter at zero, the book is the fixed band at $\delta_0=$ 4e-4, $\eta_0=0.99$.

**How the gradient reaches the view.** The size entries are differentiable in $h_{t-1}$ and $\omega_t$; the sign entry is not, and only selects. A dial set on day t now depends on the book, so its effect on later days passes through two paths: the trade (as before) and the dials of later days.

## Evaluation

**A model.** Run each test year with its model, carry the book over from the previous year, and start from cash in 1996 or 1998, as in TRADING_RULE.md. With `--trade_state` the dials depend on the book, so the stored dials of a test year belong to the book that starts from that year's target. Scoring therefore recomputes them along the carried book.
- `--save-weights` keeps, for each test day and stock: the logits $a,k$ (`dial_a`, `dial_k`), who has a signal (`signal`), and each model's view weights $(v_\delta,v_\eta)$ (`dial_v`).
- `analysis/gap/deploy.state_path` recomputes the dials and the book from those, in numpy.

**The seed ensemble.**
1. Average the seeds' targets and rescale them to unit gross exposure.
2. Each day, set each seed's dials on the ensemble's own book and target, and average the dials.
3. Run the rule once with the averaged dials.

This matches TRADING_RULE.md's ensemble, which averages the seeds' dials.

## The code (built 2026-11-16)

| Where | What |
| --- | --- |
| `afsa/config.py` | `trade_state`, `trade_market` (default False; each needs `trade_head`). With the flags off, every run folder and `config.json` stays valid (CLAUDE.md §3) |
| `afsa/model.py`, `init_member` | $v_\delta,v_\eta$ as `delta_v`, `eta_v` (3 each); $u_\delta,u_\eta$ as `delta_u`, `eta_u` (d each). They start at zero and draw nothing from the random stream |
| `book_step` | One day of `trade_book`, split out unchanged so both loops share it |
| `state_features`, `trade_book_state` | Step 5′ and the loop that sets the dials day by day |
| `AFSAModel.dial_logits` | Step 3c′: the logits, with the market term when `trade_market` is on. `dials` now calls it |
| `AFSAModel.dial_inputs` | The dials (without `trade_state`), or their logits and the signal mask (with it), for `_portfolio` |
| `afsa/train.py`, `evaluate` | With `--save-weights` and `trade_state`, also keeps `dial_a`, `dial_k`, `signal` and `dial_v` |
| `analysis/gap/deploy.py`, `state_path` | The rule in numpy, for one model or a seed ensemble, from cash or from a given book; `widen` adds to the band (`analysis/rl/pessimism.py`) |

**Tests** (`tests/test_trade_state.py`, 9 tests). With the flags off, every earlier test passes unchanged. The new tests check that:
- both flags are off by default, name their folder, and are refused without `trade_head`; the parameters start as before, with the new ones at zero;
- zero view weights give `trade_book` with the rule's dials, bit for bit;
- the rule with a view of the book matches a numpy loop written from this file, and `deploy.state_path` matches it too;
- a seed ensemble's `state_path` averages the seeds' dials on the same book, and a wider band trades less;
- with zero new parameters, the model's target, book and dials are those of `--trade_head` alone, for each view and for both;
- the market term is one shift per day, and members without a signal do not move it;
- the gradient reaches the new parameters and training moves them; `evaluate` keeps what a carried book needs, and `state_path` reproduces the stored book from the year's first day;
- on a fixed list, the month blocks train and evaluate as the day arrays do, with both views on;
- `analysis/rl/pessimism.py`'s shrink: κ = 0 is the plain ensemble, and a larger κ moves weight away from the stocks whose seeds disagree.

**To run it.** D's validation years, then the test years, for the book's view (S) and then both views (SM). Each arm takes about 10 minutes of GPU: the loop that sets the dials day by day costs little beyond the trading layer's own loop.

```powershell
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --trade_head true --trade_state true --train_years 6 --first-year 1996 --last-year 1997 --seeds 0 1 2 --save-weights
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --trade_head true --trade_state true --first-year 1998 --last-year 2025 --seeds 0 1 2 --save-weights
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --trade_head true --trade_state true --trade_market true --train_years 6 --first-year 1996 --last-year 1997 --seeds 0 1 2 --save-weights
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --K 30 --ridge 1e-4 --trade_head true --trade_state true --trade_market true --first-year 1998 --last-year 2025 --seeds 0 1 2 --save-weights
.venv\Scripts\python.exe analysis\rl\trading_state.py
```

## References

- Avellaneda, M. and J.-H. Lee (2010). Statistical arbitrage in the US equities market. *Quantitative Finance* 10(7).
- Moody, J. and M. Saffell (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks* 12(4).
