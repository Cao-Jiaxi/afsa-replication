# The learned trading rule: results

Written by `analysis/rl/trading_rule.py`, as pre-registered in CLAUDE.md §7 (row of 2026-11-16; the rule itself is in [TRADING_RULE.md](TRADING_RULE.md)). Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0–2. Test years 1998–2021 and 2022–2025 use 8-year windows; the validation years 1996–1997 use 6-year windows. Every path starts from cash on its period's first day and carries its book across years, scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).

## Decision (pre-registered: 1996–1997 only, against the fixed band trained inside the model)

**Inconclusive.**

- rule − fixed band trained: +0.45 [-0.31, +1.19]
- rule ensemble − fixed band trained, ensemble + band: -0.03 [-0.70, +0.62]

Pooled seed-mean net series for the models; one series for each ensemble. Paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), 95% interval.

## Levels

Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; μ_net and σ in % a year.

| | Validation SR_net | SR | SR_net | μ_net | σ | Turnover | Short | SR_net 1998–2005 | SR_net 2006–13 | SR_net 2014–21 | SR_net 2022–25 | SR_net 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule | 9.10 | 3.21 | 1.89 | 7.56 | 4.01 | 0.33 | 0.46 | 4.46 | 1.19 | 0.68 | -0.42 | 1.61 |
| rule's target | 7.07 | 4.29 | -0.60 | -1.81 | 3.02 | 1.08 | 0.48 | 2.52 | -2.72 | -2.49 | -3.17 | -0.97 |
| rule's targets + fixed band | 7.84 | 4.31 | 0.21 | 0.65 | 3.06 | 0.90 | 0.48 | 3.23 | -1.76 | -1.64 | -2.39 | -0.16 |
| fixed band trained | 8.91 | 3.95 | 1.77 | 6.44 | 3.65 | 0.54 | 0.46 | 4.13 | 0.60 | 0.70 | -0.25 | 1.45 |
| D model + band after | 8.96 | 3.63 | 1.76 | 6.90 | 3.94 | 0.50 | 0.46 | 4.02 | 0.75 | 0.74 | -0.01 | 1.49 |
| rule ensemble | 9.79 | 3.65 | 2.33 | 8.82 | 3.80 | 0.31 | 0.47 | 5.13 | 1.41 | 0.91 | -0.48 | 1.98 |
| fixed band trained, ensemble + band | 9.81 | 4.17 | 2.17 | 8.09 | 3.75 | 0.51 | 0.46 | 4.69 | 0.99 | 0.96 | -0.13 | 1.81 |
| D ensemble + band after | 9.75 | 3.83 | 2.10 | 8.45 | 4.04 | 0.47 | 0.45 | 4.53 | 1.09 | 0.94 | 0.10 | 1.80 |

## Contrasts

Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws), 95% interval. Only the validation column decides; the others are reported.

| | Validation 1996–97 | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rule − fixed band trained | +0.45 [-0.31, +1.19] | +0.26 [-0.07, +0.64] | +0.57 [+0.24, +0.94] | +0.73 [+0.28, +1.21] | +0.01 [-0.39, +0.51] | -0.22 [-0.92, +0.52] | +0.29 [-0.01, +0.62] |
| rule ensemble − fixed band trained, ensemble + band | -0.03 [-0.70, +0.62] | +0.16 [-0.19, +0.56] | +0.44 [+0.11, +0.79] | +0.43 [-0.16, +0.99] | -0.05 [-0.43, +0.45] | -0.35 [-1.01, +0.30] | +0.17 [-0.14, +0.50] |
| rule − rule's targets + fixed band | +1.60 [+0.94, +2.39] | +1.91 [+1.37, +2.48] | +1.53 [+1.17, +1.91] | +3.32 [+2.61, +4.22] | +2.54 [+1.73, +3.50] | +2.11 [+1.42, +2.85] | +1.99 [+1.53, +2.46] |
| rule − D model + band after | +0.46 [-0.33, +1.18] | +0.28 [-0.07, +0.67] | +0.70 [+0.37, +1.07] | +0.57 [+0.14, +1.02] | -0.03 [-0.45, +0.52] | -0.47 [-1.24, +0.32] | +0.26 [-0.05, +0.60] |
| rule ensemble − D ensemble + band after | +0.03 [-0.70, +0.68] | +0.23 [-0.16, +0.64] | +0.60 [+0.28, +0.95] | +0.32 [-0.30, +0.92] | -0.02 [-0.45, +0.52] | -0.58 [-1.27, +0.11] | +0.18 [-0.14, +0.53] |

## The dials

Member stock-days with a signal, pooled over seeds; δ in basis points of the book (4 = the fixed band, 0.04%). ρ: the mean over days of the rank correlation across stocks, with the size of the stock's target (its absolute value), or with its mean absolute daily target change over the last 20 days. *Untouched*: the share of held positions (member on both days) that do not trade on a day, for the rule and for the fixed band trained inside the model.

| | δ p10 | δ median | δ p90 | η p10 | η median | η p90 | δ ρ target size | δ ρ target change | η ρ target change | untouched | untouched, fixed band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1996–97 | 7.98 | 13.68 | 18.01 | 0.68 | 0.88 | 0.97 | 0.02 | 0.05 | -0.01 | 0.42 | 0.17 |
| 1998–2005 | 9.50 | 15.31 | 20.15 | 0.68 | 0.85 | 0.97 | 0.04 | 0.00 | 0.08 | 0.45 | 0.17 |
| 2006–13 | 14.73 | 21.20 | 30.39 | 0.24 | 0.75 | 0.93 | -0.10 | -0.22 | 0.21 | 0.58 | 0.23 |
| 2014–21 | 12.30 | 19.58 | 25.92 | 0.33 | 0.57 | 0.86 | -0.14 | -0.16 | 0.19 | 0.58 | 0.40 |
| 2022–25 | 16.92 | 22.96 | 27.96 | 0.34 | 0.59 | 0.88 | -0.12 | -0.19 | 0.17 | 0.68 | 0.49 |

