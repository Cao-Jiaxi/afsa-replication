# Pessimism from seed disagreement: results

Written by `analysis/rl/pessimism.py`, as pre-registered in CLAUDE.md §7 (row of 2026-11-16; EXPLORATION.md §6.1). The seed ensembles of D at K = 30 (5 bp, ridge 1e-4, 30 epochs, seeds 0–2) trade less where the seeds disagree. s is the standard deviation across the seeds of a stock's target on a day. *Shrink*: the average target w times w²/(w² + κs²), rescaled to unit gross exposure. *Widen*: κs added to the band width. κ = 0 is the ensemble as reported in TRADING_RULE_RESULTS.md and TRADING_STATE_RESULTS.md. Every path starts from cash on its period's first day and carries its book across years, at the paper's costs.

## Decision (pre-registered: κ chosen on 1996–1997; kept if κ > 0 and its interval against κ = 0 lies above 0)

**Kept for 0 of 10 object–form pairs.**

| | κ chosen | validation SR_net, κ = 0 | validation SR_net, chosen | validation difference | kept |
| --- | --- | --- | --- | --- | --- |
| D + band after, shrink | 0.00 | 9.75 | 9.75 |  | no |
| D + band after, widen | 0.00 | 9.75 | 9.75 |  | no |
| fixed band trained + band, shrink | 0.00 | 9.81 | 9.81 |  | no |
| fixed band trained + band, widen | 0.25 | 9.81 | 9.84 | +0.02 [-0.13, +0.18] | no |
| rule, shrink | 0.50 | 9.79 | 9.82 | +0.04 [-0.11, +0.20] | no |
| rule, widen | 0.00 | 9.79 | 9.79 |  | no |
| S, shrink | 0.25 | 9.93 | 9.97 | +0.04 [-0.09, +0.17] | no |
| S, widen | 0.00 | 9.93 | 9.93 |  | no |
| SM, shrink | 0.25 | 9.90 | 10.00 | +0.10 [-0.04, +0.24] | no |
| SM, widen | 0.00 | 9.90 | 9.90 |  | no |

The interval is a paired circular block bootstrap (21-day blocks, 2,000 draws) of the chosen κ against κ = 0 on 1996–1997. It is biased upward, because κ was chosen on the same days.

## The test years at the chosen κ (reported, never used to decide)

Net SR by period; turnover a day on 1998–2021.

| | κ | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed band trained + band, widen | 0.25 | 2.31 | 4.81 | 1.16 | 1.11 | -0.07 | 1.94 | 0.44 |
| rule, shrink | 0.50 | 2.40 | 5.21 | 1.50 | 0.90 | -0.39 | 2.04 | 0.32 |
| S, shrink | 0.25 | 2.23 | 5.11 | 1.48 | 0.78 | -0.25 | 1.93 | 0.31 |
| SM, shrink | 0.25 | 2.69 | 5.26 | 1.38 | 1.28 | -0.29 | 2.26 | 0.27 |

Against κ = 0 on the same days (paired bootstrap, 95% interval):

| | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed band trained + band, widen | +0.14 [+0.11, +0.18] | +0.12 [+0.06, +0.19] | +0.17 [+0.12, +0.23] | +0.15 [+0.09, +0.21] | +0.06 [-0.01, +0.12] | +0.13 [+0.10, +0.16] |
| rule, shrink | +0.07 [-0.02, +0.16] | +0.08 [-0.03, +0.19] | +0.09 [-0.13, +0.35] | -0.01 [-0.11, +0.06] | +0.09 [-0.16, +0.33] | +0.07 [-0.02, +0.14] |
| S, shrink | +0.06 [-0.01, +0.11] | +0.06 [-0.01, +0.13] | -0.01 [-0.16, +0.15] | +0.02 [-0.06, +0.08] | +0.07 [-0.07, +0.23] | +0.06 [-0.00, +0.11] |
| SM, shrink | +0.03 [-0.08, +0.14] | +0.02 [-0.05, +0.08] | +0.02 [-0.16, +0.21] | -0.08 [-0.27, +0.09] | +0.04 [-0.12, +0.21] | +0.02 [-0.08, +0.11] |

## The whole grid (a diagnostic)

| | validation | 1998–2021 | 2022–25 | turnover 1998–2021 |
| --- | --- | --- | --- | --- |
| D + band after, shrink, 0 | 9.75 | 2.10 | 0.10 | 0.47 |
| D + band after, shrink, 0.25 | 9.65 | 2.11 | 0.11 | 0.48 |
| D + band after, shrink, 0.5 | 9.58 | 2.10 | 0.12 | 0.49 |
| D + band after, shrink, 1 | 9.49 | 2.09 | 0.14 | 0.50 |
| D + band after, shrink, 2 | 9.37 | 2.05 | 0.15 | 0.52 |
| D + band after, shrink, 4 | 9.20 | 1.99 | 0.17 | 0.54 |
| D + band after, widen, 0 | 9.75 | 2.10 | 0.10 | 0.47 |
| D + band after, widen, 0.25 | 9.73 | 2.22 | 0.15 | 0.40 |
| D + band after, widen, 0.5 | 9.53 | 2.28 | 0.16 | 0.35 |
| D + band after, widen, 1 | 8.99 | 2.27 | 0.16 | 0.28 |
| D + band after, widen, 2 | 8.04 | 2.01 | 0.18 | 0.19 |
| fixed band trained + band, shrink, 0 | 9.81 | 2.17 | -0.13 | 0.51 |
| fixed band trained + band, shrink, 0.25 | 9.73 | 2.18 | -0.10 | 0.52 |
| fixed band trained + band, shrink, 0.5 | 9.65 | 2.18 | -0.09 | 0.53 |
| fixed band trained + band, shrink, 1 | 9.54 | 2.16 | -0.09 | 0.54 |
| fixed band trained + band, shrink, 2 | 9.38 | 2.12 | -0.07 | 0.57 |
| fixed band trained + band, shrink, 4 | 9.16 | 2.04 | -0.05 | 0.59 |
| fixed band trained + band, widen, 0 | 9.81 | 2.17 | -0.13 | 0.51 |
| fixed band trained + band, widen, 0.25 | 9.84 | 2.31 | -0.07 | 0.44 |
| fixed band trained + band, widen, 0.5 | 9.71 | 2.39 | -0.03 | 0.38 |
| fixed band trained + band, widen, 1 | 9.24 | 2.39 | 0.03 | 0.30 |
| fixed band trained + band, widen, 2 | 7.94 | 2.16 | 0.08 | 0.20 |
| rule, shrink, 0 | 9.79 | 2.33 | -0.48 | 0.31 |
| rule, shrink, 0.25 | 9.80 | 2.39 | -0.42 | 0.32 |
| rule, shrink, 0.5 | 9.82 | 2.40 | -0.39 | 0.32 |
| rule, shrink, 1 | 9.72 | 2.39 | -0.36 | 0.33 |
| rule, shrink, 2 | 9.54 | 2.34 | -0.29 | 0.34 |
| rule, shrink, 4 | 9.25 | 2.30 | -0.25 | 0.36 |
| rule, widen, 0 | 9.79 | 2.33 | -0.48 | 0.31 |
| rule, widen, 0.25 | 9.46 | 2.35 | -0.48 | 0.27 |
| rule, widen, 0.5 | 9.12 | 2.30 | -0.49 | 0.24 |
| rule, widen, 1 | 8.15 | 2.08 | -0.40 | 0.19 |
| rule, widen, 2 | 6.61 | 1.72 | -0.25 | 0.13 |
| S, shrink, 0 | 9.93 | 2.16 | -0.32 | 0.30 |
| S, shrink, 0.25 | 9.97 | 2.23 | -0.25 | 0.31 |
| S, shrink, 0.5 | 9.90 | 2.23 | -0.27 | 0.31 |
| S, shrink, 1 | 9.84 | 2.23 | -0.22 | 0.32 |
| S, shrink, 2 | 9.71 | 2.21 | -0.23 | 0.34 |
| S, shrink, 4 | 9.43 | 2.16 | -0.27 | 0.36 |
| S, widen, 0 | 9.93 | 2.16 | -0.32 | 0.30 |
| S, widen, 0.25 | 9.57 | 2.19 | -0.34 | 0.26 |
| S, widen, 0.5 | 9.21 | 2.17 | -0.49 | 0.22 |
| S, widen, 1 | 8.14 | 2.06 | -0.49 | 0.17 |
| S, widen, 2 | 6.69 | 1.82 | -0.65 | 0.12 |
| SM, shrink, 0 | 9.90 | 2.66 | -0.33 | 0.26 |
| SM, shrink, 0.25 | 10.00 | 2.69 | -0.29 | 0.27 |
| SM, shrink, 0.5 | 9.92 | 2.67 | -0.25 | 0.27 |
| SM, shrink, 1 | 9.83 | 2.62 | -0.21 | 0.28 |
| SM, shrink, 2 | 9.62 | 2.52 | -0.22 | 0.30 |
| SM, shrink, 4 | 9.26 | 2.39 | -0.25 | 0.32 |
| SM, widen, 0 | 9.90 | 2.66 | -0.33 | 0.26 |
| SM, widen, 0.25 | 9.76 | 2.63 | -0.19 | 0.23 |
| SM, widen, 0.5 | 9.45 | 2.58 | -0.21 | 0.20 |
| SM, widen, 1 | 8.26 | 2.37 | -0.12 | 0.16 |
| SM, widen, 2 | 6.56 | 1.98 | -0.14 | 0.11 |

## How much the seeds disagree

Median over member stock-days of s, against the median absolute average target.

| | validation | 1998–2025 | s in bp, 1998–2025 |
| --- | --- | --- | --- |
| D + band after | 0.44 | 0.48 | 6.41 |
| fixed band trained + band | 0.46 | 0.50 | 6.70 |
| rule | 0.49 | 0.65 | 8.37 |
| S | 0.49 | 0.65 | 8.41 |
| SM | 0.50 | 0.73 | 8.96 |

