# Why our out-of-sample Sharpe ratios differ from the paper's

Written 2026-11-12 for Jesse and Ander. "The paper" is Epstein, Wang, Choi and Pelger (ICAIF '25). "Ours" is the baseline of CLAUDE.md §7: monthly top-500 universe, 3 seeds, test years 1998–2021. Every number below comes from the scripts in this folder; the full tables are in [RESULTS.md](RESULTS.md).

**Update, 2026-11-14.** Section 9 reports the tasks of Zhishu Zhang's review. It changes conclusion 3 and decision 6.1: the input of the sequence model is no longer the main explanation, and the raw input stays the main specification. Section 10 scans the review's three model levers at every K and chooses a setting under its rules; section 10.5 scans the ridge with ten seeds on the WRDS-built arrays and chooses it walk-forward, with the no-trade band. Sections 1–8 are otherwise as written on 2026-11-12.

## Conclusions

1. **The gap is on the cost side.** Before costs we match the paper for K = 8 (Sharpe 3.53 against 3.35, return 15.4% against 15.7%). We trade 0.66 of the portfolio a day; the paper's costs imply about 0.43. That difference alone explains the net Sharpe of 1.36 against 1.94.
2. **Our code is not the cause.** An independent audit rebuilt saved models with dense matrices from the raw arrays and matched the stored returns and turnover to 1e-5. Training and evaluation charge costs the same way.
3. **Main cause: what the sequence model is fed.** We feed LongConv the 30 daily residuals. The predecessor paper, whose cost model and benchmark code this paper reuses, feeds the cumulative residuals, and the authors' slides describe the signal as patterns of cumulative residuals. With cumulative input our turnover is 0.41 (K = 8) and 0.46 (K = 30), against the paper's 0.43 and 0.47. The net Sharpe rises to 1.51 and 1.26.
4. **What is still missing is gross return, not trading speed.** At the paper's turnover we earn 13.1% (K = 8) and 11.2% (K = 30) a year before costs; the paper earns 15.7% and 16.7%. Its extra return falls in 1998–2007. After 2013 the paper earns less than we do.
5. **The universe is the leading suspect for point 4, and we cannot rebuild it.** About a third of the paper's stocks are not in ours. A companion paper by the first author describes the same data as 500 stocks fixed per training window. Our approximation of that rule reproduces the paper's weak returns after 2013 but not its strong early years.
6. **Our models fit their training window far better than they trade the next year**, and the more they are fitted, the more they trade. The choices we inherited were tuned on 1996–1997, a period that rewards fast trading.

Ruled out as causes: the scale of the Sharpe term, kernel and embedding initialization, the risk-free term of the printed objective, and seeds. Training length and batch length do change how much the model trades (conclusion 6), but no setting of either closes the gap (section 9.2, T3 and T5).

## 1. The gap in numbers

Net Sharpe is roughly gross Sharpe minus cost / σ, so the comparison splits into a gross part and a cost part.

| | SR | μ % | σ % | Turnover a day | Cost / σ | SR net |
| --- | --- | --- | --- | --- | --- | --- |
| Paper, K = 8 | 3.35 | 15.70 | 4.68 | 0.43 (implied) | 1.42 | 1.94 |
| Ours, K = 8 | 3.53 | 15.38 | 4.36 | 0.66 | 2.17 | 1.36 |
| Ours, cumulative input, K = 8 | 2.94 | 13.09 | 4.45 | 0.41 | 1.43 | 1.51 |
| Paper, K = 30 | 3.97 | 16.66 | 4.20 | 0.47 (implied) | 1.70 | 2.28 |
| Ours, K = 30 | 3.86 | 13.32 | 3.45 | 0.67 | 2.81 | 1.06 |
| Ours, cumulative input, K = 30 | 3.37 | 11.21 | 3.33 | 0.46 | 2.11 | 1.26 |

- The paper's turnover is not reported. It follows from its costs (μ minus μ net) at 5 bp per unit traded and 1 bp per unit short, with half the portfolio short.
- Across K the paper's net Sharpe rises (1.68 at K = 1 to 2.28 at K = 30). Ours peaks at K = 3 (1.44) and falls to 0.39 at K = 100, because our return and volatility per unit of exposure fall as factors are added (σ 7.35 at K = 1, 2.80 at K = 100) while the paper's stay near 4.2–4.9 up to K = 30.

Gross return by period, % a year, K = 8. The paper's column is read from the vector data of its Figure 2 (`fig2_digitize.py`):

| Years | Paper | Ours | Ours, cumulative input | Ours, window universe |
| --- | --- | --- | --- | --- |
| 1998–2002 | 41.8 | 32.9 | 28.0 | 28.7 |
| 2003–2007 | 14.1 | 12.0 | 10.5 | 8.7 |
| 2008–2012 | 11.6 | 12.9 | 10.8 | 10.0 |
| 2013–2021 | 4.0 | 8.6 | 7.3 | 4.4 |

## 2. What the documents show

- **The paper does not say what LongConv receives beyond "the past 30 daily residual returns".** The predecessor paper (Guijarro-Ordonez, Pelger and Zanotti, `2106.04028v2.pdf`) says: "The input to the signal extraction functions are the last L cumulative residuals". Its public code passes cumulative sums to every policy model. The paper takes its cost model and its OU benchmark "as in" that work, and the authors' slides list "time-series patterns of cumulative residuals".
- **Figure 2 holds 5,880 days, not the 6,040 trading days of 1998–2021.** Table 2's returns follow from its curves only with 5,880 days (to 0.01 for all seven K), so they are gross returns of the same runs, evaluated on 160 fewer days. Which days are missing is unknown. The 25 year labels are evenly spaced 245 days apart, so either each test year has 245 days or the labels are only approximate. (The "OU K = 8" curve does not match Table 2: 5.3% a year against 2.0%.)
- **A companion paper by the first author** (Epstein, Sadhwani and Giesecke, arXiv 2505.11243) uses the same 79 inputs, 8-year windows, learning rate, dropout, epochs, weight decay and cluster, with sequences of 246 days. It describes the data as: "we filter out stocks where any returns in the train or test period are not available. We include only the 500 assets with the largest market cap at the last quarter of the training period". It also states excess returns and a daily Sharpe ratio in the loss. No code of either paper is public; the companion's code is a supplementary file on OpenReview that needs a browser to download.
- **The paper's figures name stocks a monthly top 500 cannot hold.** Of the 45 newest listings labelled in Figure 3, 28 are in our universe on that day, 7 more are picked by the companion rule, 4 are not picked (three just below its cut and one ADR) and 6 are not in Ander's files at all (five ADRs that CRSP sizes by their US-listed shares, and one REIT). Ander found 70 of 109 across Figures 3 and 4.
- **The market row differs:** the paper's equal-weighted universe returns 8.61% a year with 20.37% volatility; ours 9.94% and 20.52%; the common-stock universe 10.50% and 20.70%.
- **The paper's factors are more concentrated.** In Figure 4 the 10 largest stocks hold 10–23% of each factor; in ours 4–5%.

## 3. What our trained model does

- **It fits the past much better than it trades the future.** In the last epoch the net Sharpe on the training windows is 5.2 (K = 8) and 6.8 (K = 30); out of sample it is 1.36 and 1.06.
- **More fitting means more trading.** Turnover on the training window falls until about epoch 10 and then rises again. Stopping at 10, 20, 30 or 60 epochs gives a test turnover of 0.53, 0.58, 0.66 and 0.85, and a net Sharpe of 1.34, 1.44, 1.36 and 0.60.
- **The signal is a five-day reversal.** 76% of its sensitivity sits in the 5 newest residuals. Weights that are one day old keep 56% of the return, and five days old 26%.
- **The factor weights re-sort every day on the four daily characteristics.** Removing those four lowers turnover from 0.66 to 0.53 and the gross Sharpe to 2.99; the net Sharpe barely moves (1.42). That trading costs about what it earns.
- **The policy trades too eagerly for the returns it finds out of sample.** Moving only half-way to the model's weights each day would give a net Sharpe of 1.67. Training with a cost of 10 bp and scoring at 5 bp gives 1.66, with the same validation result as the baseline (7.18 against 7.19).
- **Why raw residuals trade faster.** The skip term, the residual connection and the read-out all act on the newest input. With raw residuals that is yesterday's residual return; with cumulative input it is the 30-day residual price move.

## 4. What each change does

K = 8, test years, one change at a time (`variants.py`). Validation is 1996–1997 with 6-year windows.

| Change | SR | Turnover | SR net | Validation SR net | Reading |
| --- | --- | --- | --- | --- | --- |
| Baseline | 3.53 | 0.66 | 1.36 | 7.19 | |
| Cumulative residuals into LongConv | 2.94 | 0.41 | 1.51 | 6.32 | Matches the paper's turnover |
| Trained with 10 bp, scored at 5 bp | 2.73 | 0.32 | 1.66 | 7.18 | Diagnostic: the baseline over-trades |
| No daily characteristics in the factors | 2.99 | 0.53 | 1.42 | 5.83 | Factor churn costs what it earns |
| Batches of 252 days (half the steps) | 3.48 | 0.60 | 1.43 | 6.76 | Small |
| 60 epochs (twice the steps) | 3.21 | 0.85 | 0.60 | 6.98 | Over-fitting raises turnover |
| Kernel initialized at 0.02 | 3.39 | 0.62 | 1.43 | 7.21 | Small |
| W^K initialized 3 times larger | 3.32 | 0.60 | 1.39 | 6.83 | Small |
| Risk-free rate in the objective, as printed | 3.50 | 0.67 | 1.44 | 7.13 | Reproduces the paper's μ and σ (16.3, 4.66), not its turnover |
| Sharpe term not annualized, as printed | 2.90 | 0.70 | 0.76 | 5.17 | Worse on every count |
| Universe: US common stocks only | 3.14 | 0.70 | 0.99 | 6.96 | The universe moves the net Sharpe by 0.4 |
| Universe: 500 fixed per window (companion rule) | 3.24 | 0.51 | 1.12 | – | Lower return and lower turnover |
| The same without its look-ahead | 3.09 | 0.50 | 1.02 | – | The look-ahead is worth about 0.1 |
| Cumulative input + fixed-per-window universe | 2.78 | 0.29 | 1.42 | – | No gain over cumulative alone |
| Cumulative input + risk-free rate in the objective | 2.90 | 0.42 | 1.56 | 6.49 | Small gain; K = 30: turnover 0.47, net 1.35 |

- The validation years prefer the faster variant almost every time. Their net Sharpe is 7, several times that of any later period, so they reward speed that later stops paying.
- Table 3 points the same way. Without past-return characteristics the paper's K = 30 model has a Sharpe of 1.50 and a turnover near 0.28; ours has 3.14 and 0.62. The paper's sequence model on its own is slow and weak; ours is fast and strong.

## 5. What remains open

- **Gross return at the paper's turnover** (13.1% against 15.7% for K = 8; 11.2% against 16.7% for K = 30), and why the paper gains from more factors while we do not.
- **The paper's universe.** The fixed-per-window rule is built here only from the 2,521 securities of Ander's files, with ADRs sized by their US-listed shares. A stock the rule would pick that never reached a monthly top 500 is missing.
- **Not run:** the cumulative input for K = 1, 3, 5, 15 and 100, and cumulative input without past-return characteristics (the direct comparison with Table 3).
- **What would settle it:** the companion paper's supplementary code on OpenReview, the professor's earlier replication, or the authors.

## 6. Decisions for Jesse and Ander

Nothing in `afsa/config.py` was changed: the defaults are still the baseline.

1. **Sequence-model input.** *Withdrawn on 2026-11-14 (section 9.4): keep the raw input as the main specification and report the cumulative input as an ablation.* The recommendation of 2026-11-12 was: make the cumulative input the main specification and report the raw input beside it. For it: the predecessor paper states it, and it reproduces the paper's turnover for both K. Against it: the validation years prefer the raw input (7.19 against 6.32), and I saw the test-year turnover before recommending it.
2. **Universe.** Keep the monthly top 500 as the main universe, since that is what the paper's sentence says, and report the fixed-per-window universe as a robustness check. Ask the professor which universe his replication used.
3. **Reporting.** State plainly that our net Sharpe is below the paper's, with the split of section 1.

## 7. Checks by independent agents

- **Code audit** (blind to my hypotheses): no defect; dense replay of saved models matches to 1e-5; stitching and metrics recomputed identically. It tested the cumulative input separately, on validation-year runs of its own, and found the same effect (turnover −28%).
- **Source research:** the quotes of section 2, the predecessor's code conventions (consecutive 125-day batches, costs on asset weights, first day of a batch free), and that no authors' code is public.
- **Not done:** a review of this written report by fresh agents. Both agents ran before it existed.

## 8. Reproduce

```powershell
.venv\Scripts\python.exe analysis\gap\fig2_digitize.py                   # the paper's Figure 2 -> paper_fig2.csv
.venv\Scripts\python.exe -m afsa.build_window                            # fixed-per-window universes (add --rule window_train)
.venv\Scripts\python.exe analysis\gap\variants.py cumulative --K 8 30    # train a variant and tabulate it (--list shows all)
.venv\Scripts\python.exe analysis\gap\results.py                         # RESULTS.md from the runs on disk
.venv\Scripts\python.exe analysis\gap\yearly.py --K 8 baseline cumulative window
.venv\Scripts\python.exe analysis\gap\policy.py results\runs\attn_K008   # turnover split, lag sensitivity, concentration
.venv\Scripts\python.exe analysis\gap\execution.py results\runs\attn_K008
.venv\Scripts\python.exe analysis\gap\universe.py                        # Figure 3 names, rule overlap, market row
# section 9 (2026-11-14)
.venv\Scripts\python.exe analysis\gap\variants.py cost075 ridge1e4 ridge1e4_cost075 entry_member --K 8 30
.venv\Scripts\python.exe analysis\gap\paired_test.py --K 8 30            # paired block bootstrap against the baseline (--period validation)
.venv\Scripts\python.exe analysis\gap\exposure.py baseline rf_hurdle --K 8   # beta, net position and turnover by era (--market: the market row)
.venv\Scripts\python.exe analysis\gap\reeval_weights.py results\runs\attn_K030 results\runs\attn_K030_train_years-6   # weights of the saved models
.venv\Scripts\python.exe analysis\gap\deploy.py results\runs\attn_K030   # seed ensemble and no-trade band -> deploy.csv in the run folder
.venv\Scripts\python.exe analysis\gap\gram.py results\runs\attn_K030     # eigenvalues of the factor Gram matrix -> gram.csv
.venv\Scripts\python.exe analysis\gap\entrants.py --run results\runs\attn_K030   # members without a signal -> entrants.csv
# section 10 (2026-11-14): about 3 h of training, then about 7 min of selection
.venv\Scripts\python.exe analysis\gap\scan.py train --K 1 3 5 8 15 30 100     # the 24 settings: test, validation and holdout years
.venv\Scripts\python.exe analysis\gap\scan.py weights --K 1 3 5 8 15 30 100   # their daily weights
.venv\Scripts\python.exe analysis\gap\selection.py --K 1 3 5 8 15 30 100      # the selection rules -> SCAN.md
# section 10.4: train, then `weights`, with the same flags (runs on disk are skipped); the same for universe=wrds_common
.venv\Scripts\python.exe analysis\gap\scan.py train --K 30 --only 5/1e-5/30 5/1e-4/30 7.5/1e-4/30 --set universe=wrds_broad fill=none
.venv\Scripts\python.exe analysis\gap\selection.py --arrays broad wrds_broad wrds_common --K 30   # -> SCAN_ARRAYS.md
# section 10.5: about 28 min of training and weights, then about 3 min of selection; `weights` takes the same flags as `train`
.venv\Scripts\python.exe analysis\gap\scan.py train --K 30 --only 7.5/1e-5/30 7.5/3e-5/30 7.5/1e-4/30 7.5/3e-4/30 7.5/1e-3/30 --set universe=wrds_broad fill=none --seeds 0 1 2 3 4 5 6 7 8 9
.venv\Scripts\python.exe analysis\gap\selection.py --K 30 --only 7.5/1e-5/30 7.5/3e-5/30 7.5/1e-4/30 7.5/3e-4/30 7.5/1e-3/30 --set universe=wrds_broad fill=none --seeds 0 1 2 3 4 5 6 7 8 9 --out analysis\gap\SCAN_RIDGE.md
# the buffered list against the plain one (NOTES.md section 8): about 3 min of training and weights, then about 10 s
.venv\Scripts\python.exe analysis\gap\scan.py train --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none entry_rule=history
.venv\Scripts\python.exe analysis\gap\scan.py weights --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none entry_rule=history
.venv\Scripts\python.exe analysis\gap\buffer.py --K 30                   # -> BUFFER.md (the plain list's runs are SCAN_HISTORY.md's)
# section 10.6, the pilot on 1996-1997 only: about 10 min of training and 3 min of weights, then about 1 min
.venv\Scripts\python.exe analysis\gap\pilot.py train                     # the 52 settings at K = 30 (--list: what is on disk)
.venv\Scripts\python.exe analysis\gap\pilot.py weights
.venv\Scripts\python.exe analysis\gap\pilot.py                           # -> PILOT.md
# the four broad datasets from WRDS as a 2x2 (NOTES.md section 9): the buffered list under the 30-day wait (about 2 min), then about 40 s
.venv\Scripts\python.exe analysis\gap\scan.py train --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none
.venv\Scripts\python.exe analysis\gap\scan.py weights --K 30 --only 5/1e-4/30 --set universe=wrds_broad_buffer600 fill=none
.venv\Scripts\python.exe analysis\gap\datasets.py --K 30                 # -> DATASETS.md (the other three runs: SCAN_HISTORY.md, BUFFER.md, SCAN_ARRAYS.md)
# section 10.7, the scan on D: stage 1 about 1 h of training and weights, stage 2 about 2 h 10 min, stage 3 about 40 min; each report a few seconds to 5 min
$set = "universe=wrds_broad_buffer600", "fill=none", "entry_rule=history"
$grid = foreach ($c in "2.5", "5", "7.5", "10") { foreach ($r in "1e-5", "3e-5", "1e-4", "3e-4", "1e-3") { "$c/$r/30" } }
$grid += foreach ($e in 20, 50) { foreach ($c in "2.5", "5", "7.5", "10") { "$c/1e-4/$e" } }   # the 28 settings
$ext = foreach ($c in "2.5", "5", "7.5", "10") { "$c/1e-4/80" }                                 # stage 1's extension
.venv\Scripts\python.exe analysis\gap\pilot.py train --K 1 3 5 8 15 30 100 --only $grid $ext --set $set   # stage 1: validation years; then `weights` with the same flags
.venv\Scripts\python.exe analysis\gap\pilot.py --K 1 3 5 8 15 30 100 --only $grid $ext --set $set --out analysis\gap\PILOT_BUFFER.md
.venv\Scripts\python.exe analysis\gap\scan.py train --K 8 30 --only $grid --set $set --periods test holdout   # stage 2; then `weights` with the same flags
.venv\Scripts\python.exe analysis\gap\selection.py --K 8 30 --only $grid --set $set --baseline 5/1e-4/30 --out analysis\gap\SCAN_BUFFER_GRID.md
$k100 = foreach ($c in "5", "7.5", "10") { foreach ($r in "1e-4", "3e-4", "1e-3") { "$c/$r/30" } }   # stage 2 at K = 100 (added after stage 3): the 3 x 3 block
.venv\Scripts\python.exe analysis\gap\scan.py train --K 100 --only $k100 --set $set --periods test holdout --max-models 12   # about 1 h; then `weights` (no --max-models)
.venv\Scripts\python.exe analysis\gap\selection.py --K 8 30 100 --only $k100 --set $set --baseline 5/1e-4/30 --out analysis\gap\SCAN_BUFFER_BLOCK.md   # the block at all three K
.venv\Scripts\python.exe analysis\gap\pilot.py --best --K 1 3 5 8 15 30 100 --only $grid $ext --set $set  # stage 3: each K's choice; its last line is $pairs
$pairs = "1=10/1e-4/80", "3=7.5/1e-4/80", "5=5/1e-4/80", "8=5/1e-4/80", "15=2.5/1e-4/50", "30=5/3e-4/30", "100=7.5/3e-4/30"
foreach ($p in $pairs) { $K, $s = $p -split "="; foreach ($a in "train", "weights") { .venv\Scripts\python.exe analysis\gap\scan.py $a --K $K --only $s --set $set --periods test holdout } }
.venv\Scripts\python.exe analysis\gap\selection.py --per-k $pairs --set $set --out analysis\gap\SCAN_BUFFER.md
# section 10.8, look-back rules for walk-forward on stage 2, and on the first scan: no training, about 2 min
.venv\Scripts\python.exe analysis\gap\lookback.py                       # -> LOOKBACK.md (--no-broad: dataset D only)
```

Run one training job at a time: two 36-model passes together fill the GPU's 16 GB and slow both about fifteen times.

## 9. After the review of 2026-11-13

Zhishu Zhang, who has an independent implementation of the paper, reviewed our code and sections 1–8 and proposed ten tasks. They were run on 2026-11-14. The tables are in [RESULTS.md](RESULTS.md), from "Table 2 by number of factors" on.

How to read the numbers of this section:
- Runs are on `broad`, test years 1998–2021, scored at the paper's costs, unless stated.
- A **model** row is a seed average: each seed is scored on its own. This is what the paper reports ("averaged across multiple starting seeds", caption of its Table 3, whose baseline row is Table 2's K = 30 row).
- A **layer** row trades the same models differently (section 9.3). It is not a Table 2 row.
- A difference with an interval is a paired block bootstrap on the pooled series (`paired_test.py`). A pooled Sharpe ratio is 0.04–0.07 above the seed average net of costs, and 0.1–0.4 above it before costs.
- Test years diagnose; only 1996–1997 may choose (CLAUDE.md §8). Each validation result is given beside its test result.

### 9.1 Where the gap stands

| | SR | μ % | σ % | Turnover a day | SR net | Validation SR net |
| --- | --- | --- | --- | --- | --- | --- |
| **Paper, K = 8** | 3.35 | 15.70 | 4.68 | 0.43 (implied) | 1.94 | |
| Baseline | 3.53 | 15.38 | 4.36 | 0.66 | 1.36 | 7.19 |
| Ridge 1e-4 | 3.46 | 16.08 | 4.64 | 0.64 | 1.49 | 7.31 |
| Trained at 7.5 bp | 3.09 | 14.50 | 4.69 | 0.45 | 1.66 | 7.12 |
| Both | 3.11 | 15.41 | 4.96 | 0.45 | 1.77 | 7.07 |
| *Layer:* baseline, ensemble of the seeds | 3.70 | 16.33 | 4.41 | 0.62 | 1.68 | |
| *Layer:* baseline, ensemble + band | 3.61 | 15.80 | 4.38 | 0.48 | 1.98 | |
| **Paper, K = 30** | 3.97 | 16.66 | 4.20 | 0.47 (implied) | 2.28 | |
| Baseline | 3.86 | 13.32 | 3.45 | 0.67 | 1.06 | 7.31 |
| Ridge 1e-4 | 3.77 | 14.28 | 3.79 | 0.66 | 1.25 | 7.47 |
| Trained at 7.5 bp | 3.44 | 12.95 | 3.77 | 0.50 | 1.44 | 7.68 |
| Both | 3.38 | 13.88 | 4.11 | 0.48 | 1.64 | 7.60 |
| *Layer:* baseline, ensemble of the seeds | 4.10 | 14.55 | 3.54 | 0.64 | 1.49 | |
| *Layer:* baseline, ensemble + band | 3.99 | 14.06 | 3.52 | 0.50 | 1.87 | |

1. **Two changes of the model bring its risk and its trading to the paper's level.** A ridge of 1e-4 raises return and volatility per unit of exposure; training with a cost of 7.5 bp cuts turnover. With both, the K = 30 model has the paper's volatility (4.11 against 4.20), turnover (0.48 against 0.47) and cost / σ (1.75 against 1.70).
2. **What is left is return before costs at that risk and turnover.** With both changes the Sharpe ratio before costs is 3.11 against 3.35 (K = 8) and 3.38 against 3.97 (K = 30). At K = 30 the paper earns 2.8 points more a year. Net: 1.77 against 1.94, and 1.64 against 2.28.
3. **One model cannot have both our fast model's return and our slow model's turnover. Trading three of them through a band can.** The ensemble with a band reaches 1.98 (K = 8) and 1.87 (K = 30) with nothing changed in the models. The paper describes no such layer, and its numbers are seed averages.
4. **The seeds disagree more than their returns suggest.** Two seeds' daily returns correlate at 0.81–0.88, but their books at 0.79 (K = 8) and 0.71 (K = 30), and their daily trades at 0.60: a good part of what a seed trades on a day is its own noise. That is why averaging the books helps as much as it does.
5. **By number of factors, the gap left after both changes has two kinds.** Turnover matches the paper at every K (table below). Volatility and return do not:
    - **K = 1 and 3: too much risk.** Return before costs is above the paper's, but volatility is 1.3–1.7 times the paper's. The ridge changes nothing here (K = 1: +0.02).
    - **K = 5 and 8: about the paper's level.** Net 1.74 and 1.77 against 1.58 and 1.94.
    - **K = 15 to 100: too little return.** Volatility is near the paper's, but return before costs falls with K while the paper's stays at 16.5–16.7%.
    - **The shape differs.** The paper's net Sharpe ratio rises with K up to 30. Ours, with both changes, is flat from K = 3 to 30 (1.64–1.77). The ensemble with band of these models gives 1.90–2.21 for K = 3–30, but that is a layer.

| K | SR / SR net: paper | both changes | μ %: paper | both | σ %: paper | both | Turnover: paper | both | Both − baseline: test (validation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.05 / 1.68 | 2.10 / 1.32 | 14.45 | 17.12 | 4.74 | 8.16 | 0.42 | 0.44 | +0.26 [+0.14, +0.38] (+0.38 [−0.15, +0.90]) |
| 3 | 3.05 / 1.69 | 2.64 / 1.69 | 14.91 | 16.28 | 4.89 | 6.16 | 0.43 | 0.39 | +0.27 [+0.12, +0.43] (+0.57 [−0.00, +1.13]) |
| 5 | 2.92 / 1.58 | 2.79 / 1.74 | 14.21 | 15.73 | 4.87 | 5.64 | 0.42 | 0.39 | +0.37 [+0.22, +0.53] (+0.32 [−0.20, +0.80]) |
| 8 | 3.35 / 1.94 | 3.11 / 1.77 | 15.70 | 15.41 | 4.68 | 4.96 | 0.43 | 0.45 | +0.42 [+0.25, +0.61] (−0.16 [−0.58, +0.24]) |
| 15 | 3.81 / 2.25 | 3.18 / 1.64 | 16.66 | 14.52 | 4.37 | 4.57 | 0.45 | 0.47 | +0.49 [+0.32, +0.68] (+0.37 [−0.19, +0.95]) |
| 30 | 3.97 / 2.28 | 3.38 / 1.64 | 16.66 | 13.88 | 4.20 | 4.11 | 0.47 | 0.48 | +0.60 [+0.37, +0.85] (+0.19 [−0.33, +0.78]) |
| 100 | 4.52 / 2.19 | 3.51 / 1.14 | 16.45 | 11.60 | 3.64 | 3.30 | 0.58 | 0.53 | +0.85 [+0.67, +1.07] (+0.94 [+0.24, +1.72]) |

The last column is the paired test on pooled series, so its levels differ from the seed averages; the paired tests of each change alone are in RESULTS.md.

### 9.2 The ten tasks

| Task | The review expected | Found |
| --- | --- | --- |
| T1. Daily weights from the saved models | Agreement with the stored results to 1e-6 | Every replayed run reproduces its stored daily return, turnover and short exposure with a difference of exactly 0. So today's code evaluates the models of 2026-11-12 as it did then. |
| T2. Seed ensemble and no-trade band | At K = 30: ensemble +0.2 to +0.35, band +0.4 to +0.6 | Ensemble +0.43 (1.06 to 1.49). Band, width chosen walk-forward: 1.80 a seed, 2.10 on the ensemble. Band, width chosen on 1996–1997: 1.60 and 1.87. Section 9.3. |
| T3. Paired tests | Training length, batch length, kernel, W^K scale, no daily characteristics: within ±0.1, intervals covering 0. 10 bp training: above 0 | Holds at K = 8 for 10–20 epochs (−0.04 to +0.06), 252-day batches (+0.06), kernel (+0.05), W^K (+0.03), no daily characteristics (+0.05). 60 epochs: −0.77 [−1.00, −0.55]. 10 bp: +0.46 [+0.24, +0.67]. |
| T4. Training cost at K = 30 | Net 1.4–1.7, turnover 0.30–0.40 | 7.5 bp: 1.44 at 0.50. 10 bp: 1.49 at 0.40. Pooled: +0.44 [+0.31, +0.58] and +0.51 [+0.30, +0.73]. Validation: +0.30 [−0.06, +0.72] and +0.10 [−0.50, +0.77]. |
| T5. 252-day batches with the baseline's 480 steps | About 1.36, the baseline: the gain of long batches is a gain of fewer steps | 1.02 at turnover 0.76: −0.34 [−0.50, −0.19]. At equal steps a longer batch fits more (fit SR net 6.29 against 5.23) and trades more. Batch length matters on its own. |
| T6. Ridge | 1e-4 equal or better at K = 30 (0 to +0.25); 1e-6 worse; largest effect at K = 100 | Section 9.4. K = 30: +0.19 a seed, pooled +0.18 [+0.02, +0.35]; 1e-6: −0.16 [−0.26, −0.06]. K = 100: net 0.39, 0.70, 1.15 and σ 2.80, 3.07, 3.94 for 1e-5, 1e-4, 1e-3. |
| T7. Beta and net position by era | The printed objective (`rf_hurdle`) tilts the book long: its gain is market exposure | Not supported. In 1998–2005 its beta is 0.02 against 0.01 and its net long 0.04 against 0.02; the gain is still there with the market removed (K = 8: 1.36 against 1.28). |
| T8. Market row in log returns | Within 0.3 of the paper's 8.61 | `broad`: 7.83, which is 0.78 below; the simple return is 1.33 above. `common`: 8.35. The convention alone does not reconcile our main universe. |
| T9. Members without a signal | 2–5% of member days, small effect | 3.92% (2.68% in 2005 to 8.35% in 2000). Withheld signals are slightly more extreme than the others (11.4% in the top decile, 12.3% in the bottom). With a signal for every member: +0.14 [+0.00, +0.30] at K = 8, +0.15 [−0.00, +0.33] at K = 30; validation +0.53 [−0.15, +1.10] and −0.04 [−0.57, +0.50]. |
| T10. Cumulative input as an ablation | Not significant | K = 8: +0.12 [−0.06, +0.33]. K = 30: +0.24 [+0.04, +0.45], all of it in 2006–2013. Validation: −0.94 [−1.84, −0.16] and −0.71 [−1.55, +0.12]. Decision 6.1 is withdrawn. |

Where the runs contradict the review:
- **T5:** batch length is not only a step count.
- **The validation years can see over-trading.** The review says two atypical years cannot judge a turnover control. They do: on the seeds' own books a band of 1e-4 to 4e-4 raises the validation net Sharpe ratio with intervals above 0 for both K (K = 30, width 4e-4: +0.37 [+0.21, +0.52]; RESULTS.md, "every band width on the validation years"), and they prefer a ridge of 1e-4.
- **T7:** the gain of the printed objective is not market exposure.
- **T6:** the ridge does not work as the review describes (section 9.4).

A separate finding of T7: after 2013 every run carries market exposure, and for K = 8 that is all it earns. The baseline's beta is 0.01, 0.06 and 0.17 in the three eras. Its net Sharpe ratio of 0.45 in 2014–2021 is 0.00 once beta × market is removed; for K = 30 it goes from −0.30 to −0.66.

### 9.3 The deployment layers

`deploy.py` trades the models' daily target weights in two other ways. Neither retrains anything.

- **Ensemble:** the three seeds' weights are averaged stock by stock and rescaled to unit gross exposure. No parameter.
- **No-trade band of width δ:** a position moves only by the part of its gap to the target that exceeds δ. An average position is about 0.002 of the book.

| K = 30, baseline | Turnover | SR | SR net | 1998–2005 | 2006–2013 | 2014–2021 |
| --- | --- | --- | --- | --- | --- | --- |
| Model, seed average | 0.67 | 3.86 | 1.06 | 3.45 | −0.06 | −0.30 |
| Ensemble | 0.64 | 4.10 | 1.49 | 3.97 | 0.35 | 0.05 |
| Band a seed, δ from validation (4e-4, 8e-4, 4e-4) | 0.49 | 3.75 | 1.60 | 3.95 | 0.48 | 0.28 |
| Ensemble + band, δ from validation (4e-4) | 0.50 | 3.99 | 1.87 | 4.29 | 0.75 | 0.48 |
| Band a seed, δ walk-forward (median 1.2e-3) | 0.39 | 3.60 | 1.80 | 3.93 | 0.80 | 0.58 |
| Ensemble + band, δ walk-forward | 0.40 | 3.87 | 2.10 | 4.37 | 1.00 | 0.80 |
| Ensemble + band, best fixed δ (1.2e-3 to 1.6e-3; read off the test years) | 0.29–0.34 | 3.53–3.72 | 2.14 | | | |

- **Three ways to set δ.** *Validation:* the width with the best net Sharpe ratio on 1996–1997, kept for all test years. This is the choice CLAUDE.md §8 allows. *Walk-forward:* each year, the width that was best over the earlier test years. It uses no future data but chooses from test years, so it is a diagnostic. *Fixed:* every width of the grid, read off the test years.
- **From K = 8 the validation years pick a narrow band** (4e-4), because 1996–1997 reward speed; the test years would pick one three times wider. For the baseline at K = 1 and 3 they pick 2.4e-3, and at K = 5, 8e-4. For the models with both changes they pick 1e-4 to 1.2e-3.
- **The layers help most where the model loses.** In 2014–2021 the K = 30 model goes from −0.30 to 0.48 (validation width) or 0.80 (walk-forward).
- **On other models the layers add less,** because training at a higher cost already removes the trades a band would skip: for K = 30 trained at 10 bp, the ensemble with band gives 1.79 (validation width) against the model's 1.49.

### 9.4 The ridge, and what changes in the recommendations

Seed averages; validation in brackets:

| K | | λ = 1e-6 | 1e-5 (baseline) | 1e-4 | 1e-3 | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | SR net | 1.25 (7.20) | 1.36 (7.19) | 1.49 (7.31) | 1.48 (5.79) | 1.94 |
| 8 | σ % | 4.31 | 4.36 | 4.64 | 6.23 | 4.68 |
| 30 | SR net | 0.90 (7.00) | 1.06 (7.31) | 1.25 (7.47) | 1.46 (7.09) | 2.28 |
| 30 | σ % | 3.32 | 3.45 | 3.79 | 4.78 | 4.20 |
| 100 | SR net | 0.13 (5.29) | 0.39 (5.48) | 0.70 (6.14) | 1.15 (6.50) | 2.19 |
| 100 | σ % | 2.66 | 2.80 | 3.07 | 3.94 | 3.64 |

- **A larger ridge raises return and volatility together, at the same turnover** (0.62–0.68 in every cell). Costs then weigh less against the volatility. This is the "lower μ and σ" half of the cost gap of section 1, and it grows with K.
- **The mechanism is a partial hedge.** The review expected the small eigenvalues of ω^F′ω^F to sit near 1e-4 to 1e-5 whatever the ridge. They do not: the trained factors keep them at a few times the ridge (median 1.7e-5, 5.2e-5, 2.7e-4 and 1.4e-3 for K = 30). So the hedge is always partial, and the ridge sets how partial: on average it removes 89%, 79%, 69% and 58% of a factor direction.
- **What is not hedged shows up as market exposure, late.** K = 30, beta in 2014–2021: 0.08, 0.15 and 0.27 for 1e-5, 1e-4 and 1e-3. Net of the market the gain of 1e-4 over 1e-5 remains (1.16 against 0.98).
- **Validation:** 1e-4 is above 1e-5 at six of the seven K and level at K = 1 (−0.04 [−0.17, +0.09]). The interval excludes 0 only at K = 100: +0.11 [−0.14, +0.35], +0.13 [−0.18, +0.46] and +0.74 [+0.17, +1.36] for K = 8, 30, 100. On the WRDS-built arrays, K = 30: +0.52 [+0.12, +0.96]. 1e-3 is rejected at K = 8 (−1.44 [−2.16, −0.81]).
- **The effect grows with K.** On the test years 1e-4 adds +0.02 at K = 1 and +0.33 [+0.14, +0.51] at K = 100. With one factor the hedge has only one direction to leave in the residuals.

Recommendations, for Jesse and Ander to decide. Nothing in `afsa/config.py` was changed.

1. **Ridge 1e-4** is the one change the validation years support. Ander's log records 1e-5 and 1e-4 as tied on his validation, so 1e-5 was a tie-break. Report beta beside it.
2. **Training cost** stays a labelled candidate. It is the largest gain of any model change, on the test years at every K (+0.21 to +0.55, all intervals above 0). Validation favours 7.5 bp at K = 1 and 3 (+0.42 [+0.04, +0.81], +0.69 [+0.18, +1.19]), leans its way at K = 5, 30 and 100, and ties at K = 8 and 15. The idea came from test years.
3. **Sequence-model input:** raw. The cumulative input is one of several ways to cut turnover, and the one the validation years reject at K = 8.
4. **Layers:** report the ensemble and the band with δ from validation as labelled rows under the model rows.
5. **Entry rule:** leave as it is. The gain is at the edge of its interval, validation is split, and on the WRDS-built arrays it is worse in 2022–2025 (−0.94 against −0.43).

### 9.5 What is still unexplained

- **Return before costs at the paper's risk and turnover from K = 15 on, and too much volatility at K = 1 and 3** (points 2 and 5 of section 9.1). The universe remains the first suspect (conclusion 5). The seeds' disagreement is a second one: if the authors' fitted models were less noisy than ours, one of theirs would behave like our ensemble.
- **Nothing here was run on the paper's 5,880 days** (section 2); all figures use the 6,040 trading days.
- **2022–2025** (WRDS-built arrays, K = 30): the model loses (SR net −0.43). Trained at 10 bp it is flat (−0.04), and so is the ensemble with band (+0.09 with the validation width, whose δ was fixed before 2022). No version earns its costs after 2021.
- **About thirty variants were tested.** An interval that excludes zero by little, such as the printed objective's +0.09 [+0.03, +0.14], is weak evidence.
- **Not checkable here:** the reviewer's own numbers. Their runs differ from ours in universe (US common stocks), period (1999–2021), ridge (1e-4), objective (as printed) and batching.

## 10. A hyperparameter scan under the review's rules

Run on 2026-11-14 at Jesse's request: scan the review's three model levers at every K and choose the setting with the best net Sharpe ratio under its rules. Tables: [SCAN.md](SCAN.md), from `scan.py` and `selection.py`.

### 10.1 Design

- **Grid:** 24 settings at every K, seeds 0–2, everything else the baseline. The runs of section 9 are points of the grid and were reused.
    - Training cost: 5, 7.5, 10 and 15 bp per unit of turnover (the review's P2). Every result is still scored at 5 bp.
    - Ridge: 1e-5, 1e-4 and 1e-3 (P6).
    - Epochs: 30 and 15 (P1).
- **Periods:** test years 1998–2021; validation years 1996–1997 (6-year windows); and a holdout, 2022–2025, that no rule looks at before it has chosen.
- **Objects:**
    - *the model*: seed averages, the paper's object;
    - *the seed ensemble*;
    - *the ensemble through a no-trade band*: the band width is chosen with the setting, so there are 216 candidates per K. The last two are deployment layers (section 9.3).
- **Rules:**
    - *Validation*: the best net Sharpe ratio on 1996–1997. This is the paper's own rule ("the last two years of the first training window", its Appendix B) and the one CLAUDE.md §8 allows.
    - *Walk-forward*: each year, the best over the test years before it (the review's P5). A change of setting is traded and paid for.
    - *Hindsight*: the best fixed setting over 1998–2021. A diagnostic; the reality check below discounts it for the choice among 24 settings.
    - Each rule is applied per K, and also with one setting for every K, as the paper uses one Table 4.
- **Statistics:** paired tests as in section 9, eras, beta, and the net Sharpe ratio with beta × market removed. White's reality check asks whether the best of the grid beats the baseline by more than the best of 24 draws would by chance.
- **Caveat:** the three levers were found by looking at test years (sections 4 and 9), so no rule is fully out of sample on 1998–2021. The holdout is cleaner. Even there, some K = 30 runs on the WRDS-built arrays had already been scored on those years (section 9.5).

### 10.2 The best setting

With one setting for every K, the paper's rule picks **7.5 bp, ridge 1e-4, 30 epochs** for the model. That is the pair of candidates of section 9.4. For the layer it picks **5 bp, ridge 1e-4, 30 epochs** traded as the seed ensemble through a band of 8e-4.

| K | Paper | Baseline | Model, chosen setting | Difference, paired | Layer, chosen setting | Baseline, 2022–25 | Model, 2022–25 | Layer, 2022–25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.68 | 1.05 | 1.32 | +0.26 [+0.14, +0.38] | 1.48 | −0.65 | −0.33 | −0.36 |
| 3 | 1.69 | 1.44 | 1.69 | +0.27 [+0.12, +0.43] | 1.99 | −0.66 | −0.44 | −0.32 |
| 5 | 1.58 | 1.35 | 1.74 | +0.37 [+0.22, +0.53] | 2.12 | −0.61 | −0.43 | −0.31 |
| 8 | 1.94 | 1.36 | 1.77 | +0.42 [+0.25, +0.61] | 2.17 | −0.67 | −0.38 | −0.36 |
| 15 | 2.25 | 1.20 | 1.64 | +0.49 [+0.32, +0.68] | 2.17 | −0.48 | −0.32 | −0.09 |
| 30 | 2.28 | 1.06 | 1.64 | +0.60 [+0.37, +0.85] | 2.17 | −0.44 | −0.10 | +0.02 |
| 100 | 2.19 | 0.39 | 1.14 | +0.85 [+0.67, +1.07] | 2.01 | −0.09 | +0.04 | +0.58 |

Net Sharpe ratios. The model columns are seed averages; the layer is a deployment layer, not a Table 2 row.

1. **The validation years and the test years agree.** The setting the paper's rule picks on 1996–1997 is also the best fixed setting over 1998–2021. So the choice costs nothing against hindsight, and it beats the baseline at every K. For the layer, hindsight would pick a band of 1.2e-3 and add at most 0.08.
2. **The review's walk-forward rule does worse than validation.**
    - Averaged over the seven K, the model gets 1.56 with one setting chosen on 1996–1997 and 1.45 walk-forward; the layer gets 2.02 and 1.89.
    - Choosing per K, the model gets 1.50 against 1.43, though walk-forward wins at K = 3 and 100; the layer ties (1.92 each).
    - From K = 5 on the ranking of settings changes from year to year, so walk-forward switches 7 to 14 times in 28 years and buys noise. At K = 1 and 3 it switches 2–3 times.
    - The validation years, which the review says cannot judge a turnover control, chose better here.
3. **Choosing per K is worse on average than one setting for all K** (1.50 against 1.56 for the model). Each K's two validation years are a noisy guide: at K = 8 they pick 5 bp (1.49, against 1.77 for the common setting), at K = 30 7.5 bp with ridge 1e-5 (1.44 against 1.64). Choosing per K does better only at K = 1 and 100 (1.38 and 1.25, against 1.32 and 1.14).
4. **The best of the grid beats the baseline by more than luck** at K = 5 to 100 (reality check p = 0.019, 0.018, 0.004, < 0.001 and < 0.001). At K = 1 and 3 it does not (p = 0.09).
5. **The gains are not market exposure.** With beta × market removed, every chosen row stays within 0.1 of its net Sharpe ratio. The chosen setting does raise beta after 2013, by 0.02–0.09 (K = 8: 0.22 against the baseline's 0.17).
6. **What each lever does** (K = 8 and 30, in full in SCAN.md):
    - Training cost: 7.5–10 bp is best for the model. 15 bp trades 0.2–0.3 a day and is worse than 7.5 and 10 bp at every ridge.
    - Ridge: 1e-4 is the robust value. Hindsight picks 1e-3 at K = 15, 30 and 100. But validation rejects it at K = 8, and at K = 15 and 30 it carries a beta of 0.28–0.31 in 2014–2021 (the paper's beta is 0.05–0.07).
    - Epochs: 15 epochs helps only at 5 bp, and not at every ridge. At 7.5 bp or more, 30 epochs is as good or better: training cost and training length are one lever, as the review says (P1). The exception is K = 100, whose best fixed setting has 15 epochs.
    - With the band, the best models train at 5 bp at every K but K = 1. The band does the training cost's job, and does it better.
7. **2022–2025: the chosen settings lose less, but none earns its costs at K ≤ 15.**
    - Against the baseline, the chosen model gains +0.10 to +0.35, with intervals above 0 only at K = 1 and 8. The layer gains +0.31 to +0.66, with intervals above 0 at every K but 30 (lower bound 0.00).
    - In level, the model is at −0.44 to −0.32 for K = 1–15, and at −0.10 and +0.04 for K = 30 and 100. The layer is positive only at K = 30 (+0.02) and K = 100 (+0.58).

### 10.3 What it changes

- **The setting.** Ridge 1e-4 with training at 7.5 bp, 30 epochs, was found on test years (section 9). It is now also the choice of the paper's own selection rule on 1996–1997, at every K together. It stays a decision for Jesse and Ander (CLAUDE.md §10): it changes two §7 defaults, `ridge` and `cost_trade` (see CLAUDE.md §3 first).
- **The gap.** With this setting the model matches or beats the paper at K = 3 and 5. It is 0.17 short at K = 8, 0.36 at K = 1, and 0.6–1.05 at K ≥ 15 (section 9.5). The layer beats the paper at K = 3–8 but stays 0.08–0.18 below it at K ≥ 15, and the paper describes no layer.
- **The rules.** Keep validation as the rule that sets the specification. Report walk-forward as a diagnostic: here it chose worse.
- **The extension.** On 2022–2025 no version earns its costs at K ≤ 15. That should be reported as found, not tuned away.

### 10.4 The chosen setting on the WRDS-built arrays (K = 30)

Same seeds and years as above, on the arrays built only from our WRDS downloads (missing characteristics at rank 0); tables in [SCAN_ARRAYS.md](SCAN_ARRAYS.md). Net Sharpe ratios:

| Arrays | Baseline | Chosen setting | Difference, paired | Layer | Validation: baseline → chosen | 2022–25: baseline / chosen / layer |
| --- | --- | --- | --- | --- | --- | --- |
| `broad` (Ander's X) | 1.06 | 1.64 | +0.60 [+0.37, +0.85] | 2.17 | 7.31 → 7.60 | −0.44 / −0.10 / +0.02 |
| `wrds_broad` | 0.98 | 1.62 | +0.67 [+0.42, +0.92] | 2.11 | 7.00 → 7.61 | −0.43 / −0.13 / −0.00 |
| `wrds_common` (US common stocks) | 0.54 | 1.28 | +0.86 [+0.64, +1.11] | 1.76 | 7.23 → 7.87 | −1.24 / −0.66 / −0.66 |

- **On `wrds_broad` the setting gives what it gives on Ander's arrays** (1.62 against 1.64), with the paper's risk and trading: σ 4.11 and turnover 0.48, against 4.20 and 0.47. Neither the data source nor the missing-value rule changes the result.
- **US common stocks alone are harder.** The baseline earns 0.54, and loses after 2013 (−1.19 in 2014–2021). The setting helps most there (+0.86), but still ends at 1.28.
- **The validation years prefer the setting on both new arrays,** as on `broad`.
- **2022–2025:** the setting loses less on both arrays, but earns nothing.

### 10.5 The ridge with ten seeds on `wrds_broad`, chosen walk-forward (K = 30)

Run on 2026-11-14 at Jesse's request: scan a few ridges with more seeds on the WRDS-built arrays, choose walk-forward, and include the no-trade band. Tables: [SCAN_RIDGE.md](SCAN_RIDGE.md).

- **Design.** Training cost 7.5 bp and 30 epochs (section 10.2). λ_ridge 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3. Seeds 0–9 on the three periods of section 10.1: 1,320 new models, 27 minutes of training. The ensemble averages all ten seeds. With the band there are 45 candidates (9 widths). The rules are those of section 10.1; walk-forward is the one asked for.
- **Baseline:** section 10.4's, the `wrds_broad` baseline (5 bp, ridge 1e-5, seeds 0–2): net SR 0.98, turnover 0.67, 2022–2025 −0.43. The paper: 2.28 at an implied turnover of 0.47.

| Net SR, 1998–2021 | Validation | Walk-forward | Hindsight (diagnostic) | Walk-forward: turnover | Walk-forward: 2022–25 | Walk-forward − baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Model, seed average | 1.43 | **1.60** | 1.71 | 0.48 | −0.20 | +0.64 [+0.28, +1.01] |
| *Layer:* ensemble of ten seeds | 1.94 | **2.04** | 2.08 | 0.46 | −0.04 | +1.00 [+0.69, +1.32] |
| *Layer:* ensemble + band | 2.10 | **2.25** | 2.31 | 0.35 | +0.01 | +1.20 [+0.90, +1.52] |

What each rule chose:
- *Validation:* ridge 3e-5 for all three objects, with a band of 2e-4.
- *Walk-forward:*
    - the model: mostly 3e-4 in 1999–2012, then 1e-3 from 2013 (6 switches);
    - the ensemble: 3e-4 in 26 of 28 years;
    - the ensemble with band: mostly 3e-4 with bands of 1e-4 to 8e-4 until 2016, then ridge 1e-4 with a band of 8e-4 (11 switches).
- *Hindsight:* 1e-3, 3e-4, and 1e-4 with a band of 8e-4.

1. **The new net Sharpe ratio, chosen walk-forward:** 1.60 for the model, 2.04 for the ensemble of ten seeds and 2.25 for the ensemble with band. The baseline gets 0.98 and the paper 2.28. The paper's figure is a seed average, so only the model row compares with it; the layer rows are not Table 2 rows.
2. **Here walk-forward beats validation for every object,** by 0.17, 0.10 and 0.15, and comes within 0.04–0.11 of hindsight. On `broad` it did worse (section 10.2). The reason is that the two periods rank the ridges in opposite order:
    - validation: 7.57, 7.66, 7.56, 7.47 and 7.17 from 1e-5 to 1e-3. Against its choice, 3e-5, the intervals of 1e-5, 1e-4 and 3e-4 cover 0 (−0.08, −0.11 and −0.19, pooled), and 1e-3 is rejected (−0.54 [−0.98, −0.05]);
    - test years: 1.36, 1.43, 1.57, 1.65 and 1.71.

   So walk-forward leaves the validation choice after one year. The grid itself was informed by test years (sections 9.4 and 10.1), which favours walk-forward and hindsight.
3. **A larger ridge earns more and carries more market risk.**
    - From 1e-5 to 1e-3, σ rises from 3.75% to 5.28% and the beta of 2014–2021 from 0.09 to 0.28. The paper's beta at K = 30 is 0.05 over 1998–2021.
    - With beta × market removed, the order stands (1.28 to 1.70), so on average the gain is not market exposure.
    - The walk-forward model, at 1e-3 from 2013, has a beta of 0.28 in 2014–2021.
4. **Ten seeds against three.** The ensemble gains with every seed added, at a falling rate. At ridge 1e-4, its net SR is 1.57 with one seed, 1.80 with two, 1.93 with three, 1.97 with five and 2.05 with ten (disjoint groups, SCAN_RIDGE.md).
    - From three seeds to ten the ensemble gains 0.09–0.15, and with the band 0.05–0.14. The gain is largest at small ridges.
    - Turnover barely moves (0.48 to 0.47 at 1e-4), so the gain is return per unit of risk: the seeds' noise averages out.
    - Ten seeds also measure the model better. At 1e-4 the average is 1.57, against 1.62 for seeds 0–2 (section 10.4); single seeds range from 1.27 to 1.76.
5. **2022–2025: no version earns its costs.**
    - The walk-forward model loses less than the baseline (−0.20 against −0.43). The layers are flat (−0.04 and +0.01).
    - Every walk-forward interval covers 0. Among the validation rows, only the layers' gains over the baseline exclude it: ensemble +0.39 [+0.02, +0.78], ensemble with band +0.49 [+0.11, +0.87].
6. **What it changes.**
    - Walk-forward chooses from test years, so its rows stay diagnostics, and the specification stays with the validation years (CLAUDE.md §8).
    - At 7.5 bp on these arrays, the validation years no longer single out ridge 1e-4. They cannot tell 1e-5 to 3e-4 apart, and they reject only 1e-3.
    - The evidence on the rule is now mixed: validation chose better on `broad` (section 10.2) and worse here.
    - When a layer is reported, train more than three seeds: the ensemble keeps gaining up to ten.

### 10.6 A pilot on the validation years: the ranges for the next scan (K = 30, rebuilt history)

Run on 2026-11-14 at Jesse's request, before any new scan on test years. It trains a wide, coarse grid and scores it on 1996–1997 only, so that the next scan's ranges owe nothing to 1998 onward (CLAUDE.md §8; compare the caveat of section 10.1). Tables: [PILOT.md](PILOT.md), from `pilot.py`.

- **Design.** K = 30, the rebuilt-history rule on `wrds_broad`, seeds 0–2, 6-year windows.
    - Training cost 2.5, 5, 7.5, 10, 15 and 20 bp × ridge 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 and 3e-3, at 30 epochs.
    - Epochs 5, 10, 15, 20 and 50, at 5 and at 10 bp, both with ridge 1e-4.
    - The 9 band widths on every seed ensemble.
    - 52 settings: 306 new models in 8.5 minutes, plus the rebuilt-history validation run (5 bp, ridge 1e-4, 30 epochs: the *reference*).
- **Noise.** Two years separate little. Against the best model setting, the paired 95% intervals (pooled series) have a median half-width of 0.65. *Clearly worse* (†) means the interval lies below 0. Because the best is the maximum of 52 noisy figures, † appears a little too often.

Model, net SR (seed average), 30 epochs:

| Training cost | 1e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5 bp | 5.39 † | 6.16 † | 6.60 † | 7.09 † | 7.04 † | 7.15 † | 6.31 † |
| 5 bp | 6.59 † | 7.17 † | 7.38 † | 8.00 | 7.83 | 7.84 | 6.71 † |
| 7.5 bp | 7.08 † | 7.74 | 7.95 | 8.13 | **8.20** | 7.72 † | 6.53 † |
| 10 bp | 7.21 † | 7.79 | 7.89 | 7.89 | 7.97 | 7.44 † | 6.62 † |
| 15 bp | 7.00 † | 7.26 † | 7.46 † | 7.34 † | 7.17 † | 6.65 † | 6.04 † |
| 20 bp | 6.77 † | 6.83 † | 6.82 † | 6.39 † | 6.04 † | 5.39 † | 4.12 † |

Layer (seed ensemble + band, width chosen on the same years), net SR, 30 epochs. † is measured against the best layer, which has 50 epochs (9.39):

| Training cost | 1e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5 bp | 7.37 † | 8.20 † | 8.60 † | 9.09 | 8.80 | 8.77 | 7.82 † |
| 5 bp | 7.88 † | 8.39 † | 8.69 † | 9.27 | 9.09 | 8.93 | 7.51 † |
| 7.5 bp | 7.92 † | 8.52 † | 8.78 | 8.89 | 8.97 | 8.44 † | 7.07 † |
| 10 bp | 7.78 † | 8.32 † | 8.47 † | 8.42 † | 8.47 † | 7.88 † | 6.95 † |
| 15 bp | 7.32 † | 7.60 † | 7.80 † | 7.73 † | 7.63 † | 7.04 † | 6.32 † |
| 20 bp | 7.04 † | 7.24 † | 7.58 † | 7.55 † | 7.52 † | 6.48 † | 5.52 † |

1. **The settings already in use sit on the plateau.**
    - The best model setting is 7.5 bp, ridge 3e-4, 30 epochs (8.20). The reference gets 8.00 (best minus reference +0.12 [−0.38, +0.68]), and the scan's choice of section 10.2 (7.5 bp, 1e-4, 30 epochs) 8.13.
    - The best layer is 5 bp, ridge 1e-4, 50 epochs with a band of 8e-4 (9.39); with 30 epochs, 9.27.
    - The validation years cannot tell these apart.
2. **Ridge: an inverted U, flat from about 3e-5 to 1e-3 at 5–10 bp.**
    - 1e-6 is clearly worse at every training cost, by 1.1 to 2.7 (pooled).
    - 3e-3 is clearly worse at every cost. Even on these years it carries a beta of 0.07–0.12 at 2.5–10 bp (0.17–0.22 at 15–20 bp), against 0.02–0.09 for ridges up to 1e-3 at 2.5–10 bp.
    - A larger ridge fits the training windows less (at 5 bp, the training-window gross SR falls from 25.9 at 1e-5 to 16.3 at 3e-3) and raises σ (2.44% to 3.40%).
3. **Training cost: 5–10 bp for the model, 2.5–7.5 bp for the layer.**
    - At 2.5 bp the model trades 0.86–0.97 a day and is clearly worse at every ridge.
    - At 15 and 20 bp it trades 0.32–0.48 a day, and both objects are clearly worse at every ridge. At 20 bp the seeds also disagree: their daily returns correlate at 0.29–0.75 for ridges from 3e-5 up, against 0.78–0.94 elsewhere.
    - The band's chosen width falls as the training cost rises. For ridges up to 1e-3 it is mostly 8e-4 at 2.5 bp, 4e-4 at 5 bp, 2e-4 to 4e-4 at 7.5 bp and 1e-4 to 2e-4 at 10 bp; at 15–20 bp no band is chosen. The band does the training cost's job, as in section 10.2.
4. **Training length and training cost are one lever** (ridge 1e-4):
    - At 5 bp the model peaks at 20 epochs (8.20) and is clearly worse at 50 (7.24). There it overfits: turnover 0.90, training-window gross SR 32.9.
    - At 10 bp it needs 30–50 epochs (7.89 and 7.91).
    - 5 and 10 epochs are clearly worse at both costs.
    - The layer gains a little from longer training (9.17, 9.27 and 9.39 at 20, 30 and 50 epochs), because the band absorbs the extra trading. 50 epochs is the edge of the pilot's range, but the gain is within the noise.
5. **Turnover on these years runs above the later years' at the same setting:** the reference trades 0.83 here and 0.68 in 1998–2021 (CLAUDE.md §9). Here, the paper's implied 0.47 is reached only at 15 bp, which is clearly worse.

**Proposed grid for the next scan** (for Jesse's decision; to be fixed before any test year is run):
- **Training cost:** 2.5, 5, 7.5 and 10 bp. This brackets the model's best (7.5 bp) and the layer's (5 bp). 15 and 20 bp are dropped: both objects are clearly worse there at every ridge.
- **Ridge:** 1e-5 (the §7 baseline), 3e-5, 1e-4, 3e-4 and 1e-3. 1e-6 and 3e-3 are dropped: clearly worse at every cost, and 3e-3 carries market exposure.
- **Epochs:** 30 across the grid, plus 20 and 50 at ridge 1e-4 with each training cost. Training length moves with the training cost, and the layer prefers longer training. 5, 10 and 15 epochs are dropped.
- **Band:** the 9 widths of `deploy.py`; every choice here lies inside them.

That is 28 settings per K. The pilot's runs are the validation years of 24 of them at K = 30.

**Caveats.**
- The pilot covers K = 30 only, and section 10.2 found the best values moving with K.
- The epoch sweeps are at ridge 1e-4 only.
- The validation years are easy (net SR about 8, against 1–2 later), and the best figure (8.20) is optimistic, being the best of 52.

### 10.7 A scan on the buffered list with rebuilt history (dataset D), each K at its own setting

Run on 2026-11-15 at Jesse's request, on D of the 2×2 (`--universe wrds_broad_buffer600 --fill none --entry_rule history`; NOTES.md §9), seeds 0–2. The design was written into CLAUDE.md §7 before any of its test years ran. Tables: [PILOT_BUFFER.md](PILOT_BUFFER.md) (validation years, every K), [SCAN_BUFFER_GRID.md](SCAN_BUFFER_GRID.md) (the grid on the test years, K = 8 and 30), [SCAN_BUFFER_BLOCK.md](SCAN_BUFFER_BLOCK.md) (a 3 × 3 block of costs and ridges on the test years, K = 8, 30 and 100) and [SCAN_BUFFER.md](SCAN_BUFFER.md) (each K at its own setting).

- **Grid:** section 10.6's proposal, 28 settings. Training cost 2.5, 5, 7.5 and 10 bp × ridge 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3 at 30 epochs, plus 20 and 50 epochs at ridge 1e-4 with each cost; the 9 band widths on every seed ensemble.
- **Stage 1, validation years only** (1996–1997). First K = 8, 30 and 100. The best layer at K = 8 sat at 50 epochs, the edge, so 80 epochs at ridge 1e-4 with each cost was added (32 settings). Then K = 1, 3, 5 and 15, for stage 3. In all, 224 runs (1,344 models: 3 seeds × 2 validation years each) in about an hour.
- **Stage 2, test and holdout years** (1998–2025), K = 8 and 30, the 28 settings: 4,704 models in 2 h 9 min. A description of the grid on the test years; it sets nothing.
    - Added after stage 3 at Jesse's request: K = 100 on 3 training costs × 3 ridges at 30 epochs (5, 7.5 and 10 bp × 1e-4, 3e-4 and 1e-3), 672 new models in about 70 minutes; the ninth setting is stage 3's. The block surrounds K = 100's validation choice (7.5 bp, 3e-4) and was chosen on 1996–1997: outside it the model is clearly worse in 10 of the 11 cost × ridge cells of `PILOT_BUFFER.md`, inside it in 3 of 9.
- **Stage 3:** every K trains at its own validation choice for the ensemble + band, among the 32 settings (`pilot.py --best`). Jesse chose this after stage 1 showed that the best training length depends on K: one setting for all K, the paper's convention, would cost some K 0.3–1.0 of validation net SR. Each rule then chooses the band width for each K, and K and band width together (63 candidates).

**Stage 1.** Net SR on the validation years. *Reference*: 5 bp, ridge 1e-4, 30 epochs, the 2×2's setting. † clearly worse than the best layer.

| K | Best model setting | Model | Best layer setting (band) | Layer | Reference: model | Reference: layer |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 bp, 1e-4, 50 ep | 4.78 | 10 bp, 1e-4, 80 ep (1.2e-3) | 5.37 | 3.89 | 4.63 † |
| 3 | 10 bp, 1e-4, 50 ep | 5.97 | 7.5 bp, 1e-4, 80 ep (1.2e-3) | 6.83 | 5.50 | 6.43 |
| 5 | 7.5 bp, 1e-4, 80 ep | 6.99 | 5 bp, 1e-4, 80 ep (8e-4) | 8.37 | 6.76 | 7.37 † |
| 8 | 7.5 bp, 1e-4, 80 ep | 8.33 | 5 bp, 1e-4, 80 ep (4e-4) | 10.10 | 8.01 | 9.15 † |
| 15 | 7.5 bp, 1e-4, 50 ep | 9.13 | 2.5 bp, 1e-4, 50 ep (8e-4) | 10.65 | 9.00 | 10.26 |
| 30 | 7.5 bp, 1e-4, 30 ep | 8.76 | 5 bp, 3e-4, 30 ep (4e-4) | 9.83 | 8.42 | 9.74 |
| 100 | 7.5 bp, 1e-4, 20 ep | 7.88 | 7.5 bp, 3e-4, 30 ep (2e-4) | 8.86 | 7.00 | 8.83 |

1. **The best training length falls with K.** For the layer it is 80 epochs up to K = 8, 50 at K = 15 and 30 at K = 30 and 100; for the model, 50–80 epochs up to K = 15 and 20–30 above. Longer training trades more and makes the seeds disagree more (K = 8, 5 bp: turnover 0.76 → 0.93 and seed correlation 0.82 → 0.70 from 30 to 80 epochs). The ensemble and the band absorb both, so the layer takes longer training than the model.
2. **Edges left open.** As written, stage 3 takes the choice among the 32 settings, although 80 epochs is the edge for the layers of K = 1, 3, 5 and 8, 10 bp for K = 1 and 2.5 bp for K = 15.
3. **Smaller K choose wider bands:** 1.2e-3 at K = 1 and 3, 8e-4 at K = 5 and 15, 4e-4 at K = 8 and 30, 2e-4 at K = 100. With the reference setting the same holds (1.6e-3 at K = 1 down to 4e-4 from K = 8).
4. **Gain of each K's choice over the reference layer:** +0.74 / +0.40 / +1.00 / +0.95 / +0.39 / +0.09 / +0.03 for K = 1 / 3 / 5 / 8 / 15 / 30 / 100, clearly so only at K = 1, 5 and 8. The paired intervals have median half-widths of 0.49–0.65, as in section 10.6.

**Stage 2.** Net SR, 1998–2021 (setting chosen, or band):

| | K = 8 | K = 30 | K = 100 (3 × 3 block, 30 ep) |
| --- | --- | --- | --- |
| Paper (model) | 1.94 | 2.28 | 2.19 |
| Model, reference setting | 1.52 | 1.31 | 0.77 |
| Model, validation | 1.52 (7.5 bp, 1e-4, 50 ep) | 1.72 (7.5 bp, 1e-4, 30 ep) | 1.44 (7.5 bp, 3e-4) |
| Model, one setting for both K, validation | 1.73 (7.5 bp, 1e-4, 30 ep) | 1.72 (the same) | – |
| Model, hindsight | 1.75 (7.5 bp, 3e-4, 30 ep) | 1.82 (10 bp, 3e-4, 30 ep) | 1.57 (10 bp, 3e-4) |
| Ensemble + band, reference setting (band from validation) | 2.12 (4e-4) | 2.08 (4e-4) | 1.81 (4e-4) |
| Ensemble + band, validation | 1.88 (5 bp, 1e-4, 50 ep; 4e-4) | 2.09 (5 bp, 3e-4, 30 ep; 4e-4) | 2.08 (7.5 bp, 3e-4; 2e-4) |
| Ensemble + band, walk-forward (252 candidates; K = 100: 81) | 1.99 (13 switches) | 1.93 (17 switches) | 2.04 (9 switches) |
| Ensemble + band, hindsight | 2.22 (reference setting; 1.2e-3) | 2.28 (reference setting; 1.2e-3) | 2.26 (10 bp, 3e-4; 4e-4) |

The block holds both hindsight choices of K = 8 and 30, so on it their hindsight rows are the same (`SCAN_BUFFER_BLOCK.md`). Their validation choices differ there only at K = 8, where the block has no 50-epoch setting: the layer's becomes 5 bp, 1e-4, 30 epochs with a band of 4e-4, 2.12 against 1.88.

1. **With one setting for both K, validation picks section 10.2's setting** (7.5 bp, ridge 1e-4, 30 epochs): +0.23 [+0.11, +0.36] and +0.42 [+0.28, +0.56] over the reference model. Chosen per K, it gains nothing at K = 8 (50 epochs: +0.03 [−0.09, +0.16]).
2. **The test years do not reward longer training at low training costs.** At 5 bp the model's net SR falls from 30 to 50 epochs at both K (1.52 → 1.12 at K = 8; 1.31 → 0.91 at K = 30), as turnover rises (0.64 → 0.80; 0.66 → 0.75). At K = 8 the layer is best at 20 or 30 epochs at every cost, never at 50. The validation years ranked the other way (layer at K = 8, 5 bp: 9.15 at 30 epochs, 9.88 at 50, 10.10 at 80). Stage 3 trains K = 1–8 at 80 epochs, so it carries this risk.
3. **In hindsight, the best base for the layer is the reference setting at both K**, with a wider band than validation picks (1.2e-3 against 4e-4).
4. **Walk-forward over all 252 candidates switches 13–17 times** and does no better than validation at K = 30. Stage 3 walks forward over the band width (and K) only.
5. **The model's hindsight best is not distinguishable from luck at K = 8** (reality check p = 0.17); at K = 30 it is (p < 0.001).
6. **2022–2025:** every chosen model loses less than the reference (−0.74 / −0.34 at K = 8 / 30); the ensemble + band gets −0.57 / −0.18 by validation and −0.40 / +0.13 walk-forward.
7. **K = 100 needs a higher training cost.**
    - At 5 bp its models trade 0.68–0.72 a day and net 0.77–1.16; at 10 bp they trade 0.42–0.45 and net 1.48–1.57.
    - Both hindsight choices are 10 bp with ridge 3e-4. The layer gets 2.26, above the paper's 2.19, while at K = 8 and 30 its best base was the reference setting.
    - The validation choice (7.5 bp, 3e-4) is 0.13 below hindsight for the model and 0.18 for the layer. On 1996–1997 the 10 bp settings at ridge 1e-4 and 3e-4 tie with it: −0.14 [−0.67, +0.31] and −0.22 [−0.64, +0.24].
    - One setting for K = 8, 30 and 100 on the block: hindsight keeps the reference setting with a band of 1.2e-3, and K = 100 then gets 2.09.
    - From ridge 1e-4 to 1e-3 the model's net SR rises at every cost (5 bp: 0.77 / 0.95 / 1.16; 10 bp: 1.48 / 1.57 / 1.57), and so does its beta: 0.21–0.29 in 2014–2021 at 1e-3, against 0.07–0.11 at 1e-4.
    - 2022–2025: the layer earns its costs (validation +0.25; the hindsight setting, held by walk-forward from 2019, +0.35). The models lose −0.11 to −0.03, the reference model −0.33.

**Stage 3.** Each K at its own validation choice (the stage 1 table): 504 new models in 40 minutes. Net SR; *layer* = seed ensemble + band. The paper's figures are its model (seed average), so only the model column compares like with like.

| K | Setting | Paper | Model | Layer, validation band | Layer, walk-forward band | Layer, walk-forward, 2022–2025 | Turnover: model / layer, walk-forward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 bp, 1e-4, 80 ep | 1.68 | 1.38 | 1.66 | 1.66 | −0.29 | 0.63 / 0.43 |
| 3 | 7.5 bp, 1e-4, 80 ep | 1.69 | 0.96 | 1.54 | 1.53 | −0.12 | 0.74 / 0.48 |
| 5 | 5 bp, 1e-4, 80 ep | 1.58 | 0.46 | 1.46 | 1.47 | −0.83 | 0.94 / 0.55 |
| 8 | 5 bp, 1e-4, 80 ep | 1.94 | 0.41 | 1.29 | 1.44 | −0.92 | 0.95 / 0.58 |
| 15 | 2.5 bp, 1e-4, 50 ep | 2.25 | −0.15 | 1.42 | 1.63 | −0.92 | 1.04 / 0.58 |
| 30 | 5 bp, 3e-4, 30 ep | 2.28 | 1.39 | 2.09 | 2.19 | −0.13 | 0.66 / 0.40 |
| 100 | 7.5 bp, 3e-4, 30 ep | 2.19 | 1.44 | 2.08 | 2.16 | +0.31 | 0.54 / 0.32 |
| K chosen as well | | | 1.39 (K = 30) | 1.42 (K = 15) | 2.11 | −0.13 | 0.66 / 0.35 |

1. **At K = 30 and 100 the layer comes within 0.1 of the paper's model:** 2.19 against 2.28 and 2.16 against 2.19. Both trained 30 epochs at ridge 3e-4. K = 100 is the only K that earns its costs in 2022–2025 (+0.31; +0.25 with the validation band).
2. **At K = 5, 8 and 15 the validation choice overfits.** These settings train long at a low cost (80 epochs at 5 bp; 50 epochs at 2.5 bp). The models trade 0.94–1.04 a day and get 0.46 / 0.41 / −0.15, then −0.62 / −1.20 / −2.03 in 2014–2021 and −1.50 / −1.71 / −2.79 in 2022–2025. The layers recover +0.9 to +1.8 but stay 0.1–0.6 below the paper.
    - At K = 8 the reference setting did better on the same years (stage 2): model 1.52 against 0.41, layer with the validation band 2.12 against 1.29. On the plain list the reference layer gets 2.19 with the validation band (`SCAN_HISTORY.md`), so the shortfall comes from the training setting, not the list.
    - This is stage 2's finding 2 at full strength: the validation years reward trading that the later years punish.
3. **K = 1 and 3 come near the paper in net SR** (1.66 against 1.68; 1.53 against 1.69), at training costs of 10 and 7.5 bp, but not in risk. σ is 9.8% and 7.4% against the paper's 4.7% and 4.9%, and K = 1 has a beta of 0.21 (0.32 in 2014–2021); without the market its net SR is 1.61. Section 9.1 found the same excess volatility at K = 1 and 3.
4. **The walk-forward band adds 0 to 0.21 over the validation band** (most at K = 8, 15 and 30). It holds 8e-4 to 1.2e-3 in most years (2e-4 at K = 1). In 2022–2025 its book trades 0.05–0.06 a day at K = 30 and 100.
5. **Choosing K.**
    - Validation picks K = 15, whose layer had the best validation net SR of all (10.65). It does worst on the test years: 1.42, and −1.45 in 2022–2025.
    - Walk-forward over K and band (63 candidates) holds K = 30 until 2008, K = 100 from 2009 to 2020 and K = 30 again from 2021. It gets 2.11 with 5 switches, and −0.13 in 2022–2025, where K = 100 alone gets +0.31.
6. **The model rows fall short of the paper at every K,** by 0.3 (K = 1) to 2.4 (K = 15). The layers make up much of it, but they are deployment layers (CLAUDE.md §7), not the paper's object.

**What this says about the validation rule.** It is the paper's rule (Appendix B) and §8's, but on this grid its two years misjudge training length and training cost. They are easy years (net SR 5–11), in which a model that trades 0.9 a day still nets well. Section 10.6 (finding 5) and stage 2 had shown turnover on these years running above the later years' at the same setting. A choice made among 30-epoch settings only (Table 4's training length) would have avoided the worst picks. But that restriction is suggested by the test years of stages 2 and 3, so it would be a choice informed by test years, to be labelled as such (§8).

### 10.8 Look-back rules for walk-forward (dataset D, stage 2)

Run on 2026-11-15 at Jesse's request, as written into CLAUDE.md §7 before it ran; no training, about 2 minutes of CPU. Tables: [LOOKBACK.md](LOOKBACK.md). The question: the best training setting and band may drift with the market, so should walk-forward score the candidates on a fixed look-back instead of on every test year so far? The plain 8-year rule had been tried on these candidates before the guards were written, so the guards were designed knowing its result. The first scan, never looked at with these rules, is the fresher check.

- **Candidates:** stage 2's seed ensembles (28 settings) through the 9 band widths: 252 at K = 8, 252 at K = 30, 504 with K chosen. A candidate's record is its daily net return traded alone from 1998.
- **Rules:** *all years* (the current walk-forward); *last 8 years*; *fading memory* (each year's weight halves every 8 years); *recent return, full risk* (mean over the last 8 years divided by the standard deviation since 1998). *Validation* and *hindsight* alongside.
- **Decision, fixed in advance:** a rule beats *all years* only if it is higher on 1998–2021 at K = 8, at K = 30 and with K chosen, no single year reverses that, and it is not lower on 2022–2025.

Net SR on 1998–2021 (2022–2025 in brackets):

| Rule | K = 8 | K = 30 | K chosen |
| --- | --- | --- | --- |
| All years (current) | 1.99 (−0.40) | 1.93 (+0.13) | 1.91 (+0.13) |
| Last 8 years | 1.74 (−0.38) | 1.78 (−0.01) | 1.65 (−0.26) |
| Fading memory | 1.80 (−0.40) | 1.98 (+0.08) | 1.96 (−0.12) |
| Recent return, full risk | 1.74 (−0.41) | 1.85 (+0.11) | 1.81 (−0.07) |
| Validation | 1.88 (−0.57) | 2.09 (−0.18) | 1.88 (−0.57) |
| Hindsight (diagnostic) | 2.22 (−0.40) | 2.28 (+0.13) | 2.28 (+0.13) |

1. **No rule beats *all years*.** The 8-year window loses everywhere on 1998–2021 (−0.25 / −0.15 / −0.26). With K chosen the loss excludes 0 on both periods: −0.26 [−0.46, −0.03], and −0.39 [−0.78, −0.03] in 2022–2025. Fading memory comes closest: +0.05 at K = 30 and with K chosen (intervals [−0.01, +0.11] and [−0.04, +0.12]), but −0.19 [−0.34, −0.06] at K = 8 and lower in 2022–2025 in all three. Recent return, full risk loses 0.25 [−0.48, −0.03] at K = 8.
2. **The losses are spread over many years.** Trading any single year as *all years* reverses none of them, except 2020 for recent return at K = 30 (−0.08 becomes +0.01).
3. **What a short record follows is market exposure.** In the 2010s the 8-year rules moved to ridge 1e-3, which hedges the factors less (section 10.5). At K = 8 the last-8-years rule held it in 2014–2019; at K = 30 in 2011–2015, 2017 and 2020, where *all years* held it only in 2010–2011. With K chosen it held K = 8 from 2010 to 2023.
    - In a rising market that exposure earns, so it tops a record of recent years. Gross return did not rise (15.8% against 16.2% at K = 8; 14.9% against 14.7% with K chosen), but volatility did (5.9% against 4.9%; 5.6% against 4.8%), and so did beta in 2014–2021 (0.33 against 0.20 and 0.23).
    - Scaling by the volatility since 1998 did not stop it. The recent-return rule held ridge 1e-3 in most years of the 2010s at every K, and its beta in 2014–2021 is 0.31–0.32 at every K.
    - Fading memory kept *all years'* choices in most years (at K = 30 the same setting in 2012–2016), and its beta stayed at *all years'* (0.19–0.24).
4. **The first scan agrees** (second check, outside the decision; Ander's arrays with the 30-day wait, 24 settings × 9 band widths at each K). Against *all years*, the 8-year window is higher on 1998–2021 in 1 of 8 cases (7 K and K chosen), fading memory in 4 and recent return in 2, and no gain excludes 0. The clear differences are losses: the 8-year window −0.31 at K = 3; recent return −0.64 and −0.28 at K = 3 and 5 and −0.20 with K chosen; fading memory −0.13 at K = 1.
5. **Reading.** On these data an 8-year record does not track a drift in the best training setting and band; it tracks which settings carried the market. *All years* stays the walk-forward rule, and the validation rule stays the rule for the specification (§8). All of this is deployment diagnostics, never Table 2 rows.
