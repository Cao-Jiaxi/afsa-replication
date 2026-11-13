# Why our out-of-sample Sharpe ratios differ from the paper's

Written 2026-11-12 for Jesse and Ander. "The paper" is Epstein, Wang, Choi and Pelger (ICAIF '25). "Ours" is the baseline of CLAUDE.md §7: monthly top-500 universe, 3 seeds, test years 1998–2021. Every number below comes from the scripts in this folder; the full tables are in [RESULTS.md](RESULTS.md).

## Conclusions

1. **The gap is on the cost side.** Before costs we match the paper for K = 8 (Sharpe 3.53 against 3.35, return 15.4% against 15.7%). We trade 0.66 of the portfolio a day; the paper's costs imply about 0.43. That difference alone explains the net Sharpe of 1.36 against 1.94.
2. **Our code is not the cause.** An independent audit rebuilt saved models with dense matrices from the raw arrays and matched the stored returns and turnover to 1e-5. Training and evaluation charge costs the same way.
3. **Main cause: what the sequence model is fed.** We feed LongConv the 30 daily residuals. The predecessor paper, whose cost model and benchmark code this paper reuses, feeds the cumulative residuals, and the authors' slides describe the signal as patterns of cumulative residuals. With cumulative input our turnover is 0.41 (K = 8) and 0.46 (K = 30), against the paper's 0.43 and 0.47. The net Sharpe rises to 1.51 and 1.26.
4. **What is still missing is gross return, not trading speed.** At the paper's turnover we earn 13.1% (K = 8) and 11.2% (K = 30) a year before costs; the paper earns 15.7% and 16.7%. Its extra return falls in 1998–2007. After 2013 the paper earns less than we do.
5. **The universe is the leading suspect for point 4, and we cannot rebuild it.** About a third of the paper's stocks are not in ours. A companion paper by the first author describes the same data as 500 stocks fixed per training window. Our approximation of that rule reproduces the paper's weak returns after 2013 but not its strong early years.
6. **Our models fit their training window far better than they trade the next year**, and the more they are fitted, the more they trade. The choices we inherited were tuned on 1996–1997, a period that rewards fast trading.

Ruled out as causes: the scale of the Sharpe term, training length, batch length, kernel and embedding initialization, the risk-free term of the printed objective, and seeds.

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

1. **Sequence-model input.** I recommend making the cumulative input the main specification and reporting the raw input beside it. For it: the predecessor paper states it, and it reproduces the paper's turnover for both K. Against it: the validation years prefer the raw input (7.19 against 6.32), and I saw the test-year turnover before recommending it.
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
```

Run one training job at a time: two 36-model passes together fill the GPU's 16 GB and slow both about fifteen times.
