# Reinforcement learning in AFSA

Written on 2026-11-16 at the user's request, after Prof. Polak's RL lecture ([LECTURE_NOTES.md](LECTURE_NOTES.md); its slides are cited as pN, its sections as "deck §N"). The question: how can the RL framework be used in this model, given that step 3, the LongConv sequence model, can be replaced?

Nothing here was trained. Every number comes from existing runs, and each one cites its file. None of the designs below would be a Table 2 row, because each departs from the paper's model. They would be reported like the deployment layers.

## 1. Summary

- **AFSA already is an RL agent.**
  - It is a deterministic policy (p308): it maps what is known at the close of t−1 to a portfolio, and it is trained by direct policy search on its net Sharpe ratio.
  - AFSA assumes that trading does not move prices. So the market's next state does not depend on the action, and the reward is a known function of the action and the realized returns.
  - Two things follow. Any policy's reward on history can be computed, and the gradient of the batch objective is exact by backpropagation. No exploration, critic or importance weights are needed.
  - The generic RL methods would be a step back: REINFORCE is noisy (p295, p313), a critic adds bias (p303), and PPO's clipping is not a trust region (p329).
- **What the RL view adds is the state.**
  - The deck's portfolio MDP puts the holdings in the state (p482). AFSA's network never sees them: yesterday's book enters only the loss, as on the final deck's slide *The Model Cannot See What It Holds*.
  - With trading costs the optimal policy depends on the holdings. Proportional costs give a no-trade band. Quadratic costs give partial adjustment toward an aim portfolio, which is LQ control (deck §17).
  - `trade_book` (band δ, speed η) is that policy class with two global numbers.
- **The evidence so far.**
  - Adding the band after training raises D's model by +0.45 net SR on 1998–2021 (1.31 → 1.76).
  - Training through the same band ties with adding it afterwards: +0.02 [−0.06, +0.09].
  - So one fixed trading rule already captures what knowing the holdings gives. What is left to test is a rule whose width and speed change by stock and day.
- **Recommended first experiment: the per-stock network also outputs its own band width and trading speed (section 5.1).** The gradient stays exact. It adds 66 parameters per model and contains both the current model and the fixed band as special cases. It costs about 7 minutes of GPU at K = 30. Second, a CPU-only check: shrink each stock's ensemble target where the seeds disagree (offline RL's pessimism, p383–p384; section 6.1).
- **Tested since (2026-11-16, D at K = 30, each pre-registered).** Every decision on 1996–1997 was *inconclusive* or *not kept*. The test years are reported only.
  - **5.1, the learned trading rule** ([TRADING_RULE_RESULTS.md](TRADING_RULE_RESULTS.md)): 1.89 as a model and 2.33 as an ensemble on 1998–2021. It does this by trading far less.
  - **5.2, the rule's view of the book** ([TRADING_STATE.md](TRADING_STATE.md)): slightly below the rule on 1998–2021. It learned to add readily and reduce reluctantly.
  - **5.2 with 5.4's market view:** 1.99 and 2.66 on 1998–2021, +0.49 above the fixed band trained inside the model with an interval excluding 0, but not clearly above the rule.
  - **6.1, pessimism** ([PESSIMISM_RESULTS.md](PESSIMISM_RESULTS.md)): kept for no ensemble.
  - **In 2022–2025** none of them helps.
- **What RL cannot do here.**
  - A trading policy can only spend less of the gross edge on costs. It cannot create the edge.
  - Since 2003, a fresh trade earns 1–3 bp per unit on its first day, against a 5 bp cost. The gap to the paper lies in its gross returns of 2001–2006 ([PAPER_BY_YEAR.md](../gap/PAPER_BY_YEAR.md)).
  - With the windfall of 2022–2025 set aside, even free trading would bring the K = 30 ensemble only to +0.10 on 2022–2025 ([EXTENSION.md](../gap/EXTENSION.md)).
- **What does not transfer.** Each of these solves a problem AFSA does not have (section 7):
  - exploration and bandits for the trades;
  - score-function methods (REINFORCE, PPO, SAC) for the weights;
  - Q-learning and DQN;
  - offline RL's pessimism against unlogged actions.

## 2. AFSA as a decision process

| The deck's portfolio MDP (p480–p486) | AFSA as built |
| --- | --- |
| Information $\mathcal F_t$ at decision time (p482) | The close of t−1: characteristics $X_{t-1}$, returns up to t−1, and day t's universe (set at the previous month-end) |
| Action: post-decision weights $w_t$ | $\omega_{t-1}$ (row t of a batch), with $\|\omega\|_1=1$ |
| Pre-trade portfolio $\tilde w_t$ (after the previous returns) | Not an input to the policy. It appears in the cost as $\omega_{t-1}$, without drift for the day's returns (the paper's convention) |
| State | The policy sees each stock's 30 lagged residuals and, through the factor model, the day's characteristics. It does not see the holdings |
| Transition | The market part is exogenous, since the paper assumes no impact. Only the holdings depend on the action |
| Cost $c(w_t,\tilde w_t,S_t)$ | 5 bp per unit of turnover plus 1 bp per unit short, the same for every stock |
| Reward and mandate (p483) | Net Sharpe ratio over a batch of 126 days, plus λ_Var × explained variance. A ratio of batch moments, not a sum of daily rewards |
| Episode | One batch. Its first day holds its target at no cost: a reset (cf. p14, p334) |
| Timing contract (p486) | CLAUDE.md §4 (row t pairs $X_{t-1}$ with $R_t$) and §8 (no look-ahead; normalization fitted on the training window; choices made on 1996–1997 only) |

Three things follow.

1. **Counterfactuals are identified.** The deck warns about logged data. Logs that contain only action A cannot price action B (p379), and another order could have filled differently (p481). Neither applies here: under the no-impact cost model, any weight vector earns $w^\top R_t$ less a known cost. The backtest is still conditional on that cost model (p481), and the flat 5 bp is the assumption to question first.
2. **The objective is not additive.** A Sharpe ratio over a batch has no Bellman recursion, as the deck notes for static CVaR (p485). Value-based methods (TD, Q-learning, actor-critic) would need a per-step surrogate reward. Pathwise training does not: it differentiates the batch objective as a whole.
3. **The policy does not see its holdings.** It is a restricted-observation policy in the sense of p285. There, a memoryless deterministic policy cannot do "R, then L", but a policy that sees its state can. Here the policy cannot do "I already hold most of this, so leave it."

## 3. What kind of agent AFSA already is

In the deck's taxonomy (p31–p33), AFSA is a policy-based, model-free control method, trained by direct policy search.

- **A deterministic policy (p308).** The same inputs always give the same weights; dropout is off outside training.
- **A pathwise gradient through a known model.**
  - The deck derives the score-function gradient (p290) because the environment's transitions are unknown.
  - AFSA's environment is a known differentiable function: the returns are data and the cost is a formula. Backpropagation therefore gives the exact gradient of the batch objective with respect to every parameter.
  - This is the deterministic policy gradient (p308), with the action gradient $\nabla_aQ$ computed exactly from the realized path instead of by a critic (p309).
- **Exact credit assignment across days within a batch.**
  - Today's weights affect today's return and tomorrow's cost.
  - With the trading layer on, they also affect every later book. `trade_book`'s loop over the batch's 127 days carries that gradient, at about 2.1 times the time of a pass.
- **Full information, so no exploration.** The bandit machinery of deck §2 answers "what would the other arm have paid?". Here that is known.

So the generic RL methods would each replace an exact gradient with an estimate:
- REINFORCE is unbiased but noisy (p295), and its signal-to-noise ratio can fall exponentially with the horizon (p313).
- An actor-critic trades that noise for the critic's bias (p303, p335).
- PPO's clipping does not bound the policy's change (p329).

## 4. What the RL view finds missing: the holdings

- **The deck.** "The state includes holdings and any variables required by the cost model" (p482).
- **Control theory (outside the deck).**
  - *Proportional costs.* The optimal policy has a no-trade region around the target: trade only when a position leaves the region, and then only to its edge (Constantinides 1986; Davis and Norman 1990).
  - *Quadratic costs and mean-variance preferences.* The problem is LQ (deck §17). The state is the holdings and the signals, the control is the trade, and the Riccati solution is a linear feedback: trade a fixed fraction of the gap to an *aim* portfolio, which weights the current and the expected future signals (Gârleanu and Pedersen 2013). Noise changes only the constant (p465–p466), so certainty equivalence holds, and only the expected path of the signals matters.
  - *The deck's two-date execution example* (p487) shows the same effect in small. With quadratic costs, selling 10 units at once costs 100, equal halves cost 75, and the optimal split, 20/3 then 10/3, costs 200/3.
- **AFSA's version.** `trade_book` ([afsa/model.py](../../afsa/model.py)) is a band δ plus partial adjustment η: each day the book skips a stock whose gap to its target is at most δ, and trades η of the rest of the gap less δ. These are two global numbers.
- **The benchmark has it too.** The s-score rule of the paper's PCA + OU benchmark (Avellaneda and Lee 2010) opens a position past one threshold and closes it at another, so its action depends on whether it already holds the stock.

**What the band has given so far** (D, K = 30, 5 bp, ridge 1e-4, seeds 0–2; [BAND_TRAINED.md](../gap/BAND_TRAINED.md)):

| Object | Net SR, 1996–1997 / 1998–2021 / 2022–2025 |
| --- | --- |
| D model, no band | 8.42 / 1.31 / −0.34 |
| D model + band added after training (δ = 4e-4) | 8.96 / 1.76 / −0.01 |
| Model trained through the band (δ = 4e-4, η = 1) | 8.91 / 1.77 / −0.25 |

- The band adds +0.45 on 1998–2021.
- Training through it changes the target. The network aims sharper (its target's gross SR is 4.00 against 3.70) and trades more (0.72 a day against 0.66).
- It does not change the net result: +0.02 [−0.06, +0.09] on 1998–2021, and −0.25 [−0.42, −0.09] on 2022–2025.

In control terms, the network learned a different aim once it knew the rule. With one global δ and η, there was little left to gain.

## 5. Replacing LongConv: four designs

What step 3 must keep:
- one network shared by all stocks;
- an output that feeds $(\omega^\varepsilon)^\top$ in step 4;
- differentiability, so the gradient stays exact.

What it can change is what it sees and what it outputs. A different encoder (a GRU, a state-space model, a transformer over the lags) is a separate question: it changes the architecture, not the decision problem. The RL view says to replace LongConv with a policy that also decides *how to trade, given what it holds*.

### 5.1 A learned trading rule (recommended first)

The full flow and equations are in [TRADING_RULE.md](TRADING_RULE.md).

The per-stock network outputs three numbers per stock and day instead of one: the signal (as now), a band width $\delta_{i,t}\ge0$ and a speed $\eta_{i,t}\in(0,1]$. The book then follows `trade_book` with these per-stock values:

$$h_{i,t}=h_{i,t-1}+\eta_{i,t}\,\mathrm{sign}(\Delta_{i,t})\max(|\Delta_{i,t}|-\delta_{i,t},\,0),\qquad\Delta_{i,t}=\omega_{i,t}-h_{i,t-1},$$

followed by the existing rescale to $\|h_t\|_1=1$. The objective, the batches and the scoring stay as they are.

- **Why this form.**
  - It is the policy class the theory predicts (section 4), so the network does not have to discover "leave small gaps alone" from scratch.
  - It acts on the book after the factor projection, where a stock's weight is its actual position.
  - It adds 2(d + 1) = 66 parameters per model. That matters for a model that already fits its window far better than it trades the next year: in-sample net SR 5–7, out of sample 1.1–1.4 ([REPORT.md](../gap/REPORT.md)).
  - It contains the current model (δ = 0, η = 1) and the fixed band as special cases, so it can start from D's validation band (δ = 4e-4).
- **What it can learn that a fixed band cannot.**
  - In theory the width and the speed depend on the cost, the stock's risk and how long its signal lasts. The last two vary across stocks and days, and the sequence model's hidden state already summarizes the stock's recent residuals.
  - Under the flat 5 bp cost, the rule can use only that variation. With stock-specific costs it could also use the cost differences. A gain under flat costs is therefore a lower bound on what the rule is worth.
- **Inputs.**
  - The sequence model's last hidden state (d = 32), through one linear layer.
  - $\delta=\delta_{max}\,\sigma(\cdot)$ with $\delta_{max}$ = 3.2e-3 (the top of the band grid), and $\eta=\sigma(\cdot)$.
  - Optionally, the stock's Spread and STD_W1 ranks (known at t−1).
- **Code.**
  - A flag, off by default (CLAUDE.md §3), e.g. `--trade_head true`.
  - `trade_book` takes per-stock δ and η, shaped [M, D+1, U]; its formula already broadcasts.
  - `signal` also returns the two extra outputs, gathered per stock on the month blocks the way `w` is.
  - A test that a rule with constant outputs equals the fixed band exactly.
- **Cost.** About 7 minutes of GPU for D at K = 30, seeds 0–2, 1996–1997 and 1998–2025 (the trading layer runs about 2.1 times slower than a normal pass).
- **Draft pre-registration**, to become a CLAUDE.md §7 row before any test year runs:
  - Base: D's reference setting.
  - Compare against D's model + band added after (δ = 4e-4), and against the fixed band trained inside the model, both as models and as seed ensembles.
  - Choose or reject on 1996–1997 only. Report 1998–2021 and 2022–2025 with the paired block bootstrap.
  - If the validation interval includes 0, report the result as inconclusive, because 1996–1997 has rewarded trading that later years punished (the scan on D, [REPORT.md](../gap/REPORT.md) §10.7).
- **Honest expectation: a small gain at most.** The fixed band trained inside the model tied with the band added after, so any gain must come from differences across stocks and days.

### 5.2 Holdings as an input (a recurrent policy)

Feed yesterday's book on stock i, scaled by N, to the per-stock network beside its 30 residual lags. Alternatively, give the network a hidden state that persists across days (a GRU instead of a fixed 30-day window). This is the general "policy that sees its state" (p24–p26, p285).

Two drawbacks:
- The network's output for stock i is not stock i's position, because step 4 hedges it against the factors and mixes the stocks. "Hold what you have" would have to be learned through that map.
- It adds capacity where the model already overfits.

Try it only if 5.1 shows that the state pays.

**Built and tested (2026-11-16) in a form that avoids the first drawback.** The holdings enter the learned rule's band width and speed, which act on the stock's position itself, not the signal: [TRADING_STATE.md](TRADING_STATE.md), `--trade_state`. The pre-registered decision is *inconclusive*. On 1998–2021 it is slightly below the rule without the view (−0.12 as a model, −0.17 as an ensemble, within the noise). What it learned: a wider band and a slower speed for trades that reduce a position than for trades that add to it.

### 5.3 Predict the signal's path, then solve for the trade (model-based)

The sequence model predicts each residual portfolio's expected return over the next H days (an alpha term structure). The trade is then computed by control:
- for quadratic costs, the Gârleanu–Pedersen aim portfolio and trading rate (the LQ solution of deck §17);
- for proportional costs, a band around that aim.

This is planning with an expectation model (p344). Under certainty equivalence the expected path is enough (p466), though not for the tails. Model error grows with the horizon (p361–p362, p368), so H should stay short, 5–10 days.

The design separates prediction from control, and it is the classical finance structure. But it departs from the paper's architecture, so it is a project, not a trial.

### 5.4 Stocks that see each other (the multi-agent view)

- **Step 3 is already a team of agents.** One agent per stock shares the parameters. The team is trained centrally on the portfolio objective, and each agent acts on its own stock's residuals: centralized training with decentralized execution (p459).
- **Step 4 is its mixer.** The factor hedge and the L1 scaling combine the agents' outputs into one portfolio, the role a mixing network plays in QMIX (p460).
- **The change.** A cross-stock attention layer after the per-stock encoder would let each stock's action depend on the others' (graph communication, p461), much like the companion paper's set-sequence model.

This is a larger change that adds capacity: an ablation for later.

**Its lightest form was built and tested (2026-11-16) on the trading rule, not the signal.** Each stock's dials also see the mean hidden state of the day's stocks, one shift per day: [TRADING_STATE.md](TRADING_STATE.md), `--trade_market`, on top of the view of the book.
- The pre-registered decision is *inconclusive*.
- On 1998–2021 it nets 1.99 as a model and 2.66 as an ensemble, +0.23 and +0.33 above the rule without views, within the noise.
- Against the fixed band trained inside the model it gains +0.49 for both the model and the ensemble, and both intervals exclude 0.
- In 2022–2025 it does not help.
- The cross-stock attention on the signal is still untested.

## 6. RL ideas outside the policy network

### 6.1 Pessimism from seed disagreement (CPU only)

- **The idea.** Offline RL guards against values the data cannot support by planning with a lower bound, $\tilde r=\hat r-\lambda u$ (p383–p384). Model disagreement is a usable, though uncalibrated, measure of that uncertainty (p368).
- **Our seeds disagree.** Their daily returns correlate at 0.81–0.88, their books at 0.71–0.79, and their trades at 0.60 ([REPORT.md](../gap/REPORT.md) §9).
- **The rule.** The seed ensemble averages the seeds. A pessimistic ensemble would also shrink each stock's target by its disagreement: $\bar\omega_i\leftarrow\bar\omega_i\cdot\bar\omega_i^2/(\bar\omega_i^2+\kappa s_i^2)$, where $s_i$ is the seeds' standard deviation. Then rescale to unit gross exposure and apply the band. κ is chosen on 1996–1997.
- **Data.** It runs on saved weights: D's runs (seeds 0–2) and the ten-seed ridge scan on `wrds_broad`. Both have validation twins.
- **Expectation.** Little gain, since averaging already cancels the positions the seeds disagree on. But it costs only minutes of CPU.
- **Run (2026-11-16, [PESSIMISM_RESULTS.md](PESSIMISM_RESULTS.md); pre-registered).** Two forms were tried on five of D's ensembles at K = 30: the shrink above, and a band widened by κs.
  - The seeds disagree by about half a typical position: median s is 6–9 bp, against a median average target of about 14 bp.
  - On 1996–1997 the best κ was 0 or small, and no interval against κ = 0 excluded 0, so pessimism was kept for none.
  - At the chosen κ the test years move by +0.03 to +0.07 for the learned rules' ensembles (shrink).
  - They move by +0.14 [+0.11, +0.18] for the fixed band trained inside the model (widen). The test years prefer wider bands anyway (hindsight's 1.2e-3 against 4e-4), so that gain cannot be credited to the disagreement.
  - Widening hurts the learned rules, whose bands are already wide.

### 6.2 λ_Var as an auxiliary task

The explained-variance term is an auxiliary prediction that shapes the factor model's representation. The deck says to judge such terms by ablation at a matched budget (p276–p277). Batch 6 of [EXTENSION.md](../gap/EXTENSION.md) (λ_Var 10 and 1000) is that ablation. It is half trained, and finishing it takes about 13 minutes of GPU.

### 6.3 Other objectives

Pathwise training needs no Bellman structure, so objectives that break it (p485) are fine. One example: mean return less λ × lower-tail CVaR in its variational form (p484), with the threshold learned jointly. The mandate here is the paper's net Sharpe, and the reward must match the mandate (p483). So this would be a robustness study, not a replacement.

### 6.4 Choosing settings as online learning (a diagnostic only)

- **What walk-forward is.** Walk-forward over a grid sees every candidate's return each year. That is prediction with expert advice, not a bandit, and our *all years* rule is follow-the-leader.
- **A soft version.** Exponential weights (Hedge; the multiplicative form of p321) would hold a weighted mixture of the candidates' books instead of switching between them. That is a conservative mixture (p323) and an ensemble across settings, and our gains so far have come from ensembles.
- **Its limits.** It chooses from test years, so under CLAUDE.md §8 it can only be a diagnostic, and its learning rate must be fixed in advance.

### 6.5 Per-step rewards and a critic (not needed now)

The deck's value-based tools come back in two cases: a part of the decision stops being differentiable (discrete lots, a market-impact simulator), or the policy must look beyond the 126-day batch. Then the tools are:
- a per-step reward: mean-variance $r_t-\frac\kappa2r_t^2$, or the differential Sharpe ratio of Moody and Saffell (2001), from outside the deck;
- a critic that values the book held at a batch's end, since a truncation needs a bootstrap (p334);
- GAE (p331).

None of the designs 5.1–5.4 needs them.

## 7. What does not transfer, and why

| Tool | Why not here |
| --- | --- |
| Exploration: ε-greedy, UCB, Thompson sampling (deck §2) | Full information: under no impact, every action's reward can be computed from history |
| REINFORCE, PPO, TRPO, SAC for the weights (deck §8–9) | They replace an exact gradient with a noisy estimate over about 500 continuous actions a day (p295, p313) |
| Q-learning, DQN, fitted Q iteration (deck §6–7, §11) | The actions are continuous and high-dimensional; the Sharpe objective is not additive; the deadly triad (p255) |
| CQL, IQL and coverage (deck §11) | They guard against unlogged actions, but here counterfactuals are identified. Our offline problem is a changing market (the 2001 break, 2022–2025), which coverage does not measure |
| RLHF, DPO, GRPO (deck §13–15) | There are no preferences and no verifier; only loose analogies apply |

## 8. What the project already does that deck §19 asks

- **Chronological training, selection and test, with the selection frozen first** (p494): CLAUDE.md §8's validation rule and the §7 rows written before their test years run.
- **Training randomness separated from evaluation noise** (p490): 3 or 10 seeds, and a paired block bootstrap on the seed-mean series pooled over seeds.
- **Matched budgets and recorded tuning** (p491): the scans' grids and rules are written down.
- **Repeated search changes how the final estimate reads** (p494): White's reality check in the scans; the deployment layers are never Table 2 rows.
- **No exclusion interval is needed** (p494): the label is a one-day return, and the lags use only past residuals.

## References outside the deck

- Avellaneda, M. and J.-H. Lee (2010). Statistical arbitrage in the US equities market. *Quantitative Finance* 10(7).
- Constantinides, G. M. (1986). Capital market equilibrium with transaction costs. *Journal of Political Economy* 94(4).
- Davis, M. H. A. and A. R. Norman (1990). Portfolio selection with transaction costs. *Mathematics of Operations Research* 15(4).
- Freund, Y. and R. E. Schapire (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences* 55(1).
- Gârleanu, N. and L. H. Pedersen (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance* 68(6).
- Moody, J. and M. Saffell (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks* 12(4).
