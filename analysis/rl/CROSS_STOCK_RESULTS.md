# Cross-stock attention on the signal: results

Written by `analysis/rl/cross_stock.py`, as pre-registered in CLAUDE.md §7 (rows of 2026-11-16; the layer is specified in [CROSS_STOCK.md](CROSS_STOCK.md)). Dataset D (buffered list, rebuilt history), K = 30, 5 bp, ridge 1e-4, 30 epochs, seeds 0–2, with 2 attention heads of 8 after the sequence model, against D's stored runs. Test years 1998–2021 and 2022–2025 use 8-year windows; the validation years 1996–1997 use 6-year windows. Every path starts from cash on its period's first day and carries its book across years, scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day). *+ band after*: δ = 4e-4, movers-only.

## Decision (pre-registered: 1996–1997 only)

**Inconclusive.**

- attention model − D model: +0.04 [-0.12, +0.20]
- attention ensemble + band after − D ensemble + band after: +0.05 [-0.09, +0.23]

Pooled seed-mean net series for the models; one series for each ensemble. Paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), 95% interval. Computed before any test-year model trained.

## Levels

Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; μ_net and σ in % a year.

| | Validation SR_net | SR | SR_net | μ_net | σ | Turnover | Short | SR_net 1998–2005 | SR_net 2006–13 | SR_net 2014–21 | SR_net 2022–25 | SR_net 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention model | 8.45 | 3.44 | 1.19 | 4.85 | 4.09 | 0.64 | 0.45 | 3.54 | 0.17 | 0.20 | -0.53 | 0.93 |
| attention model + band after | 8.97 | 3.36 | 1.63 | 6.66 | 4.10 | 0.47 | 0.45 | 3.99 | 0.66 | 0.59 | -0.16 | 1.36 |
| attention ensemble | 9.40 | 3.71 | 1.64 | 6.76 | 4.16 | 0.60 | 0.45 | 4.15 | 0.59 | 0.49 | -0.32 | 1.34 |
| attention ensemble + band after | 9.81 | 3.65 | 2.05 | 8.41 | 4.13 | 0.44 | 0.45 | 4.57 | 1.02 | 0.86 | -0.02 | 1.73 |
| D model | 8.42 | 3.70 | 1.31 | 5.17 | 3.96 | 0.66 | 0.46 | 3.56 | 0.26 | 0.36 | -0.34 | 1.06 |
| D model + band after | 8.96 | 3.63 | 1.76 | 6.90 | 3.94 | 0.50 | 0.46 | 4.02 | 0.75 | 0.74 | -0.01 | 1.49 |
| D ensemble | 9.35 | 3.89 | 1.70 | 6.89 | 4.08 | 0.62 | 0.45 | 4.12 | 0.67 | 0.57 | -0.18 | 1.41 |
| D ensemble + band after | 9.75 | 3.83 | 2.10 | 8.45 | 4.04 | 0.47 | 0.45 | 4.53 | 1.09 | 0.94 | 0.10 | 1.80 |

## Contrasts

Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws), 95% interval. Only the validation column of the rows in the decision decides; everything else is reported.

| | Validation 1996–97 | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention model − D model | +0.04 [-0.12, +0.20] | -0.08 [-0.20, +0.04] | -0.00 [-0.09, +0.08] | -0.08 [-0.35, +0.22] | -0.15 [-0.37, +0.04] | -0.20 [-0.42, +0.03] | -0.10 [-0.21, +0.00] |
| attention model + band after − D model + band after | +0.03 [-0.13, +0.19] | -0.07 [-0.19, +0.06] | -0.01 [-0.09, +0.08] | -0.05 [-0.33, +0.24] | -0.13 [-0.34, +0.07] | -0.16 [-0.37, +0.07] | -0.09 [-0.20, +0.02] |
| attention ensemble − D ensemble | +0.05 [-0.10, +0.21] | -0.06 [-0.18, +0.07] | +0.03 [-0.06, +0.12] | -0.08 [-0.36, +0.22] | -0.08 [-0.29, +0.10] | -0.14 [-0.35, +0.09] | -0.07 [-0.18, +0.04] |
| attention ensemble + band after − D ensemble + band after | +0.05 [-0.09, +0.23] | -0.06 [-0.18, +0.08] | +0.04 [-0.05, +0.13] | -0.07 [-0.36, +0.24] | -0.08 [-0.30, +0.12] | -0.12 [-0.35, +0.13] | -0.07 [-0.18, +0.05] |

## How much the other stocks matter

Each saved test-year model run on its own year: the mean over stocks with a signal of the message's norm against the hidden state it is added to, and the share of each day's cross-sectional variance of the signal that the message alone carries (its read-out, w_s'm), averaged over the era's models.

| | |message| / |z| | signal variance from the message |
| --- | --- | --- |
| 1998–2005 | 0.203 | 0.034 |
| 2006–13 | 1.836 | 0.344 |
| 2014–21 | 2.004 | 0.602 |
| 2022–25 | 2.468 | 0.637 |

