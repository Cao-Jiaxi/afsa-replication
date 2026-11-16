# SM as a deployment variant

This file reports one deployment variant of our replication. It sits beside the model rows and is never a Table 2 row (CLAUDE.md §7, 2026-11-14). The variant is **SM**: the paper's model trained with a learned trading rule that sees each stock's history, its position and the day's market, traded as a seed ensemble. Every number here comes from [SM_FIRM_RESULTS.md](SM_FIRM_RESULTS.md), written by [sm_firm.py](sm_firm.py).

## What it is

- **The model.** The paper's Attention Factors + LongConv, trained end to end on the net Sharpe ratio, on dataset D (the buffered list with rebuilt history), at D's setting: 5 bp training cost, ridge 1e-4, 30 epochs.
- **The trading rule, inside training.** Each day each stock's book moves toward the model's target only past a band, and only part of the way.
  - The band's width and the speed are set per stock and day from that stock's residual history ([TRADING_RULE.md](TRADING_RULE.md)).
  - They are also set from the stock's position, its target, whether the trade adds to the position or reduces it, and the day's average hidden state ([TRADING_STATE.md](TRADING_STATE.md)).
  - The network is scored on the book it holds, so it learns to aim and to trade together. Flags: `--trade_head true --trade_state true --trade_market true`.
- **The seed ensemble.**
  1. Three seeds are trained, and their targets are averaged and rescaled to unit gross exposure.
  2. Each day, each seed's band and speed are set on the shared book, then averaged.
  3. The averaged rule trades that one book.
- **Why it is not a Table 2 row.** Table 2 reports the model trading its own target, averaged over seeds. SM changes how the target is traded, and the ensemble changes the target itself. Both are deployment layers, like the band added after training.

## How it was chosen and tested

All steps were pre-registered in CLAUDE.md §7 (rows of 2026-11-16). Choices were made on 1996–1997 only; the test years 1998–2021 and 2022–2025 are reported.

1. **First test** (K = 30, seeds 0–2; [TRADING_STATE_RESULTS.md](TRADING_STATE_RESULTS.md)). Against the learned rule without the views of the book and the market, the decision was *inconclusive*.
2. **The choice of SM**, made before its fresh cells ran. On 1996–1997, SM's ensemble was slightly ahead of the rule's (9.90 against 9.79, within the noise), and the user asked to test SM further.
   - The choice was made knowing both rules' 1998–2021 results at K = 30 with seeds 0–2.
   - So that cell is not out-of-sample evidence for SM. The fresh cells below are.
3. **A firmer test on fresh cells:** K = 8 and K = 100 (seeds 0–2), and K = 30 with seeds 3–5, against D's model and ensemble with the band added after training (δ = 4e-4).
   - The pre-registered decision, pooled over the three cells on 1996–1997, is **better** for the model: +0.79 [+0.02, +1.62].
   - For the ensemble the difference is within the noise: +0.23 [−0.53, +1.03].
   - This is the only positive pre-registered decision among the RL designs so far.

## Results

Net Sharpe ratio, annualized, at the paper's costs (5 bp per unit traded, 1 bp per unit short a day). Every path starts from cash and carries its book across years.
- *Model* is the average over seeds, the form of the paper's rows.
- *Ensemble* is one portfolio.
- *+ band after* is D's model or ensemble through the band δ = 4e-4, which is the width D's ensemble chooses on 1996–1997 at every K.
- K = 30 uses six seeds (0–5), since seeds 3–5 were trained for this test.

**1998–2021**

| K | Paper, Table 2 (model) | D model | D model + band after | D ensemble + band after | SM model | SM ensemble |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.94 | 1.52 | 1.88 | 2.13 | 1.72 | 1.95 |
| 30 | 2.28 | 1.32 | 1.78 | 2.22 | 1.99 | **2.61** |
| 100 | 2.19 | 0.77 | 1.33 | 1.83 | 1.46 | 2.09 |

**2022–2025**

| K | D model | D model + band after | D ensemble + band after | SM model | SM ensemble |
| --- | --- | --- | --- | --- | --- |
| 8 | −0.74 | −0.53 | −0.50 | −0.80 | −0.67 |
| 30 | −0.33 | −0.02 | +0.08 | −0.43 | −0.50 |
| 100 | −0.33 | +0.15 | +0.36 | −1.03 | −0.73 |

**Turnover a day, 1998–2021.**

| K | D model | D ensemble + band after | SM model | SM ensemble |
| --- | --- | --- | --- | --- |
| 8 | 0.64 | 0.44 | 0.31 | 0.29 |
| 30 | 0.66 | 0.46 | 0.29 | 0.27 |
| 100 | 0.72 | 0.52 | 0.29 | 0.28 |

**K = 30 by seed group.** On 1998–2021, SM's ensemble nets 2.66 with seeds 0–2 (not fresh) and 2.29 with seeds 3–5 (fresh). D's ensemble + band after nets 2.10 and 2.14.

## What the results show

- **SM trades much less.** At every K, its ensemble trades 35–45% less than D's ensemble with the band added after training, and its model less than half as much as D's model. That is where its gain comes from: it keeps the edge with less trading.
- **On 1998–2021, SM's ensemble is ahead of D's ensemble + band at K = 30 and 100, and behind at K = 8.**
  - K = 30, six seeds: +0.39 [+0.08, +0.71].
  - K = 100: +0.26 [−0.03, +0.57].
  - K = 8: −0.19 [−0.53, +0.24].
  - Pooled over the three fresh cells it is +0.20 [−0.07, +0.52] for the ensemble and +0.08 for the model, so the test years do not confirm the validation decision with any confidence.
- **The gain is early.** Pooled over the fresh cells, SM leads by +0.73 (model) and +0.49 (ensemble) in 1998–2005, and not after 2013.
- **In 2022–2025 SM is worse.** Pooled over the fresh cells it trails by −0.86 [−1.63, −0.11] (model) and −0.74 [−1.46, +0.00] (ensemble). In the extension years the simple band added after training remains the better deployment layer.
- **Against the paper.** SM's ensemble nets 1.95, 2.61 and 2.09 at K = 8, 30 and 100, against the paper's 1.94, 2.28 and 2.19. That is not a like-for-like comparison: the paper's row is a seed-averaged model that trades its own target. Our like-for-like row is D's model, at 1.52, 1.32 and 0.77.

## Limits

- **Costs.** The cost model is the paper's flat 5 bp per unit traded. A rule that trades less would gain more under realistic costs that vary by stock, but we have not measured that.
- **Selection noise.** Every decision rests on two validation years. Their intervals are ±0.8 to ±1.0 for one cell, and those years rewarded trading that later years punished ([REPORT.md](../gap/REPORT.md) §10.7).
- **Seeds.** Three seeds per cell, six at K = 30. Ensembles kept gaining up to ten seeds in the ridge scan ([SCAN_RIDGE.md](../gap/SCAN_RIDGE.md)).
- **Pessimism did not help.** Trading less where the seeds disagree (shrinking those positions or widening their bands) was kept for no ensemble in either test ([PESSIMISM_RESULTS.md](PESSIMISM_RESULTS.md), [SM_FIRM_RESULTS.md](SM_FIRM_RESULTS.md)).

## To reproduce

GPU, about 10 minutes per K at K = 8 and 30, and 40 minutes at K = 100. Then the scoring on CPU, about 10 minutes.

```powershell
$B = "--universe wrds_broad_buffer600 --fill none --entry_rule history --ridge 1e-4 --trade_head true --trade_state true --trade_market true --save-weights".Split(" ")
.venv\Scripts\python.exe -m afsa.run @B --K 8 --seeds 0 1 2 --train_years 6 --first-year 1996 --last-year 1997
.venv\Scripts\python.exe -m afsa.run @B --K 8 --seeds 0 1 2 --first-year 1998 --last-year 2025
.venv\Scripts\python.exe -m afsa.run @B --K 30 --seeds 0 1 2 --train_years 6 --first-year 1996 --last-year 1997
.venv\Scripts\python.exe -m afsa.run @B --K 30 --seeds 0 1 2 --first-year 1998 --last-year 2025
.venv\Scripts\python.exe -m afsa.run @B --K 30 --seeds 3 4 5 --tag seeds3-5 --train_years 6 --first-year 1996 --last-year 1997
.venv\Scripts\python.exe -m afsa.run @B --K 30 --seeds 3 4 5 --tag seeds3-5 --first-year 1998 --last-year 2025
.venv\Scripts\python.exe -m afsa.run @B --K 100 --seeds 0 1 2 --max-models 12 --train_years 6 --first-year 1996 --last-year 1997
.venv\Scripts\python.exe -m afsa.run @B --K 100 --seeds 0 1 2 --max-models 12 --first-year 1998 --last-year 2025
.venv\Scripts\python.exe analysis\rl\sm_firm.py
```

D's comparator runs at seeds 0–2 already exist (REPORT.md §8). D's seeds 3–5 at K = 30 come from the same two commands without the three trading flags, tagged `seeds3-5`.
