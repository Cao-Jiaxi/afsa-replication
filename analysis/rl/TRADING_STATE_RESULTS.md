# The learned trading rule with views of the book and of the market: results

Written by `analysis/rl/trading_state.py`, as pre-registered in CLAUDE.md §7 (rows of 2026-11-16; the views are specified in [TRADING_STATE.md](TRADING_STATE.md)). Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0–2. **S**: the dials also see the book (`--trade_state true`). **SM**: they also see the day's market (`--trade_market true` as well). *Rule*: the learned rule without views ([TRADING_RULE_RESULTS.md](TRADING_RULE_RESULTS.md)). Test years 1998–2021 and 2022–2025 use 8-year windows; the validation years 1996–1997 use 6-year windows. Every path starts from cash on its period's first day and carries its book across years, scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).

For S and SM the dials depend on the book, so they are recomputed along each carried path (in float64, by `deploy.state_path`). Checked one day at a time against the model's own float32 books: applied to the stored book of the day before, the numpy rule gives the stored book of the day, with a largest daily L1 gap of 1.5e-07 (S), 1.6e-07 (SM). The exceptions are 0 of 22,650 seed-days (S) and 1 of 22,650 seed-days (SM) with a tie: a stock whose gap to its target equals its band width to within 1e-6, so the two precisions decide differently whether it trades. A stock that trades is resized with the other movers, so a tie can move the day's book by up to a few tenths of a per cent.

## Decision (pre-registered: 1996–1997 only, each arm against the rule without views)

- **S: Inconclusive.** S − rule: +0.06 [-0.18, +0.30]; S ensemble − rule ensemble: +0.15 [-0.08, +0.41].
- **SM: Inconclusive.** SM − rule: +0.01 [-0.28, +0.32]; SM ensemble − rule ensemble: +0.11 [-0.17, +0.43].

Pooled seed-mean net series for the models; one series for each ensemble. Paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), 95% interval. The decision was computed from the validation runs before any test-year model of S or SM had trained.

## Levels

Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; μ_net and σ in % a year.

| | Validation SR_net | SR | SR_net | μ_net | σ | Turnover | Short | SR_net 1998–2005 | SR_net 2006–13 | SR_net 2014–21 | SR_net 2022–25 | SR_net 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule | 9.10 | 3.21 | 1.89 | 7.56 | 4.01 | 0.33 | 0.46 | 4.46 | 1.19 | 0.68 | -0.42 | 1.61 |
| S | 9.05 | 3.13 | 1.85 | 7.31 | 3.99 | 0.31 | 0.46 | 4.52 | 1.28 | 0.53 | -0.33 | 1.58 |
| SM | 8.97 | 3.27 | 1.99 | 7.12 | 3.71 | 0.28 | 0.46 | 4.77 | 1.08 | 0.48 | -0.41 | 1.62 |
| fixed band trained | 8.91 | 3.95 | 1.77 | 6.44 | 3.65 | 0.54 | 0.46 | 4.13 | 0.60 | 0.70 | -0.25 | 1.45 |
| D model + band after | 8.96 | 3.63 | 1.76 | 6.90 | 3.94 | 0.50 | 0.46 | 4.02 | 0.75 | 0.74 | -0.01 | 1.49 |
| rule ensemble | 9.79 | 3.65 | 2.33 | 8.82 | 3.80 | 0.31 | 0.47 | 5.13 | 1.41 | 0.91 | -0.48 | 1.98 |
| S ensemble | 9.93 | 3.41 | 2.16 | 8.45 | 3.92 | 0.30 | 0.47 | 5.05 | 1.49 | 0.76 | -0.32 | 1.87 |
| SM ensemble | 9.90 | 3.94 | 2.66 | 8.97 | 3.39 | 0.26 | 0.45 | 5.25 | 1.36 | 1.36 | -0.33 | 2.24 |
| fixed band trained, ensemble + band | 9.81 | 4.17 | 2.17 | 8.09 | 3.75 | 0.51 | 0.46 | 4.69 | 0.99 | 0.96 | -0.13 | 1.81 |
| D ensemble + band after | 9.75 | 3.83 | 2.10 | 8.45 | 4.04 | 0.47 | 0.45 | 4.53 | 1.09 | 0.94 | 0.10 | 1.80 |

## Contrasts

Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws), 95% interval. Only the validation column of the first four rows decides; everything else is reported.

| | Validation 1996–97 | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S − rule | +0.06 [-0.18, +0.30] | -0.12 [-0.24, +0.06] | +0.02 [-0.10, +0.16] | +0.09 [-0.16, +0.32] | -0.18 [-0.35, -0.02] | +0.10 [-0.07, +0.28] | -0.08 [-0.18, +0.08] |
| S ensemble − rule ensemble | +0.15 [-0.08, +0.41] | -0.17 [-0.28, +0.02] | -0.08 [-0.23, +0.05] | +0.07 [-0.14, +0.32] | -0.15 [-0.29, -0.00] | +0.16 [-0.07, +0.39] | -0.11 [-0.21, +0.05] |
| SM − rule | +0.01 [-0.28, +0.32] | +0.23 [-0.19, +0.60] | +0.26 [+0.07, +0.46] | -0.08 [-0.42, +0.27] | -0.10 [-0.73, +0.38] | -0.08 [-0.74, +0.56] | +0.10 [-0.27, +0.43] |
| SM ensemble − rule ensemble | +0.11 [-0.17, +0.43] | +0.33 [-0.15, +0.81] | +0.12 [-0.09, +0.33] | -0.05 [-0.58, +0.50] | +0.45 [-0.30, +1.06] | +0.15 [-0.19, +0.50] | +0.26 [-0.12, +0.67] |
| SM − S | -0.05 [-0.36, +0.26] | +0.34 [-0.16, +0.77] | +0.23 [+0.03, +0.45] | -0.16 [-0.43, +0.11] | +0.08 [-0.57, +0.53] | -0.19 [-0.81, +0.41] | +0.18 [-0.25, +0.56] |
| SM ensemble − S ensemble | -0.04 [-0.35, +0.29] | +0.50 [-0.10, +1.03] | +0.20 [-0.01, +0.40] | -0.12 [-0.57, +0.32] | +0.60 [-0.20, +1.22] | -0.01 [-0.36, +0.34] | +0.37 [-0.10, +0.84] |
| S − fixed band trained | +0.51 [-0.19, +1.20] | +0.15 [-0.24, +0.62] | +0.60 [+0.27, +0.95] | +0.82 [+0.39, +1.27] | -0.17 [-0.60, +0.35] | -0.12 [-0.85, +0.64] | +0.21 [-0.14, +0.60] |
| S ensemble − fixed band trained, ensemble + band | +0.12 [-0.54, +0.79] | -0.00 [-0.41, +0.49] | +0.36 [+0.01, +0.72] | +0.50 [+0.00, +1.03] | -0.20 [-0.62, +0.34] | -0.19 [-0.90, +0.55] | +0.06 [-0.29, +0.48] |
| SM − fixed band trained | +0.45 [-0.32, +1.22] | +0.49 [+0.19, +0.77] | +0.83 [+0.47, +1.21] | +0.65 [+0.23, +1.09] | -0.09 [-0.67, +0.40] | -0.30 [-1.25, +0.62] | +0.39 [+0.09, +0.66] |
| SM ensemble − fixed band trained, ensemble + band | +0.08 [-0.64, +0.84] | +0.49 [+0.20, +0.80] | +0.55 [+0.20, +0.91] | +0.38 [-0.05, +0.83] | +0.40 [-0.21, +0.95] | -0.21 [-0.89, +0.47] | +0.43 [+0.16, +0.71] |

## The dials by kind of trade

Member stock-days with a signal along each seed's own path, pooled over seeds. *Opens*: no position held; *adds*: trading towards the target would add to the position; *reduces*: it would reduce or reverse it. δ is the median band width in basis points of the book (4 = the fixed band); η the median speed; *share* the share of stock-days of each kind; *untouched* the share of held positions (member on both days) that do not trade on a day.

| | δ opens | δ adds | δ reduces | η opens | η adds | η reduces | share opens | share adds | share reduces | untouched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule, 1996–97 | 14.17 | 13.64 | 13.69 | 0.90 | 0.88 | 0.88 | 0.00 | 0.32 | 0.67 | 0.42 |
| S, 1996–97 | 13.45 | 11.88 | 15.98 | 0.94 | 0.94 | 0.91 | 0.00 | 0.31 | 0.68 | 0.44 |
| SM, 1996–97 | 14.32 | 12.73 | 16.11 | 0.91 | 0.89 | 0.86 | 0.00 | 0.31 | 0.69 | 0.45 |
| rule, 1998–2005 | 14.08 | 15.38 | 15.28 | 0.87 | 0.86 | 0.85 | 0.00 | 0.32 | 0.68 | 0.45 |
| S, 1998–2005 | 13.44 | 14.14 | 17.76 | 0.94 | 0.94 | 0.90 | 0.00 | 0.31 | 0.69 | 0.47 |
| SM, 1998–2005 | 14.02 | 14.90 | 17.53 | 0.90 | 0.90 | 0.86 | 0.00 | 0.31 | 0.69 | 0.47 |
| rule, 2006–13 | 20.75 | 21.31 | 21.15 | 0.80 | 0.76 | 0.75 | 0.00 | 0.32 | 0.68 | 0.58 |
| S, 2006–13 | 19.47 | 20.07 | 22.08 | 0.87 | 0.87 | 0.79 | 0.00 | 0.32 | 0.68 | 0.59 |
| SM, 2006–13 | 20.39 | 20.22 | 21.59 | 0.80 | 0.79 | 0.69 | 0.00 | 0.31 | 0.69 | 0.57 |
| rule, 2014–21 | 20.60 | 19.49 | 19.62 | 0.60 | 0.59 | 0.56 | 0.00 | 0.33 | 0.67 | 0.58 |
| S, 2014–21 | 20.11 | 19.58 | 21.04 | 0.72 | 0.76 | 0.60 | 0.00 | 0.32 | 0.68 | 0.61 |
| SM, 2014–21 | 24.58 | 19.77 | 21.11 | 0.32 | 0.49 | 0.32 | 0.00 | 0.31 | 0.68 | 0.56 |
| rule, 2022–25 | 24.17 | 22.70 | 23.08 | 0.60 | 0.60 | 0.58 | 0.00 | 0.34 | 0.65 | 0.68 |
| S, 2022–25 | 23.80 | 23.02 | 23.95 | 0.71 | 0.76 | 0.60 | 0.00 | 0.34 | 0.66 | 0.68 |
| SM, 2022–25 | 26.85 | 25.21 | 26.31 | 0.74 | 0.75 | 0.60 | 0.00 | 0.34 | 0.66 | 0.60 |

## The view weights

v_δ and v_η: each dial's weights on the view of the book (position size, target size, adds), averaged over seeds and the period's windows. A positive v_δ widens the band, a positive v_η speeds up trading. Log sizes are in units of an average position.

| | v_δ position | v_δ target | v_δ adds | v_η position | v_η target | v_η adds |
| --- | --- | --- | --- | --- | --- | --- |
| S, 1996–97 | 0.22 | 0.05 | -0.24 | -0.21 | 0.01 | 0.23 |
| SM, 1996–97 | 0.12 | -0.04 | -0.19 | -0.13 | 0.09 | 0.14 |
| S, 1998–2005 | 0.28 | 0.09 | -0.23 | -0.25 | -0.01 | 0.20 |
| SM, 1998–2005 | 0.17 | -0.01 | -0.16 | -0.19 | 0.09 | 0.16 |
| S, 2006–13 | 0.20 | 0.11 | -0.14 | -0.25 | -0.05 | 0.23 |
| SM, 2006–13 | 0.10 | 0.02 | -0.08 | -0.20 | 0.05 | 0.19 |
| S, 2014–21 | 0.15 | 0.10 | -0.07 | -0.24 | -0.04 | 0.28 |
| SM, 2014–21 | 0.09 | 0.07 | 0.00 | -0.25 | -0.02 | 0.23 |
| S, 2022–25 | 0.16 | 0.11 | -0.02 | -0.28 | -0.02 | 0.34 |
| SM, 2022–25 | 0.09 | 0.06 | 0.01 | -0.25 | 0.04 | 0.22 |

