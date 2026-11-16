# A firmer test of SM, and of pessimism: results

Written by `analysis/rl/sm_firm.py`, as pre-registered in CLAUDE.md §7 (row of 2026-11-16, "a firmer test of SM"). SM is the learned trading rule with views of the book and of the market ([TRADING_STATE.md](TRADING_STATE.md)) on dataset D (buffered list, rebuilt history; 5 bp, ridge 1e-4, 30 epochs). **Fresh cells**, none run before the row was written: K = 8 and K = 100 (seeds 0–2), and K = 30 with seeds 3–5. The first cell (K = 30, seeds 0–2, [TRADING_STATE_RESULTS.md](TRADING_STATE_RESULTS.md)) is shown for reference only. The comparator is D's model and D's seed ensemble with the band added after training (δ = 4e-4, the width D's ensemble chooses on 1996–1997 at every K; movers-only). Every path starts from cash on its period's first day and carries its book across years, at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).

Checks: the targets and books reproduce each run's stored results, and SM's numpy rule, applied one day at a time to the stored book of the day before, gives the stored book (largest daily L1 gap and float32 tie days, out of seed-days): K = 8 1.7e-07, 0 of 21,129; K = 30, seeds 3–5 1.6e-07, 0 of 21,129; K = 100 1.7e-07, 0 of 21,129; K = 30, seeds 0–2 1.6e-07, 1 of 21,129.

## Decision (pre-registered: 1996–1997 only, the fresh cells pooled)

**Better.** SM − D + band after: +0.79 [+0.02, +1.62]; SM ensemble − D ensemble + band after: +0.23 [-0.53, +1.03].

The pooled series hold a third of the capital in each fresh cell. Paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), 95% interval; for the models, each cell's series is its seed mean. Each cell's own verdict, by the same rule:

| | verdict | SM − D + band after | SM ensemble − D ensemble + band after |
| --- | --- | --- | --- |
| pooled | better | +0.79 [+0.02, +1.62] | +0.23 [-0.53, +1.03] |
| K = 8 | inconclusive | +0.66 [-0.14, +1.45] | -0.22 [-1.08, +0.59] |
| K = 30, seeds 3–5 | inconclusive | +0.12 [-0.85, +1.14] | -0.18 [-1.12, +0.81] |
| K = 100 | better | +0.86 [+0.06, +1.69] | +0.33 [-0.48, +1.20] |

## Levels

Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; σ in % a year. *Paper SR_net*: Table 2's model at that K (1998–2021), which is a seed average, not an ensemble.

| | Validation SR_net | SR | SR_net | σ | Turnover | SR_net 1998–2005 | SR_net 2006–13 | SR_net 2014–21 | SR_net 2022–25 | SR_net 1998–2025 | Paper SR_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K = 8: SM | 8.68 | 2.85 | 1.72 | 4.53 | 0.31 | 4.35 | 1.10 | 0.41 | -0.80 | 1.29 | 1.94 |
| K = 8: SM ensemble | 8.93 | 2.93 | 1.95 | 4.80 | 0.29 | 4.74 | 1.43 | 0.61 | -0.67 | 1.63 | 1.94 |
| K = 8: D + band after | 8.28 | 3.32 | 1.88 | 4.89 | 0.48 | 4.06 | 1.12 | 0.79 | -0.53 | 1.52 | 1.94 |
| K = 8: D ensemble + band after | 9.15 | 3.47 | 2.13 | 4.95 | 0.44 | 4.45 | 1.40 | 0.93 | -0.50 | 1.75 | 1.94 |
| K = 8: D model | 8.01 | 3.39 | 1.52 | 4.89 | 0.64 | 3.66 | 0.74 | 0.51 | -0.74 | 1.19 | 1.94 |
| K = 8: D ensemble | 8.96 | 3.56 | 1.84 | 4.98 | 0.60 | 4.12 | 1.10 | 0.69 | -0.64 | 1.48 | 1.94 |
| K = 30, seeds 3–5: SM | 8.71 | 3.39 | 1.99 | 3.55 | 0.30 | 4.67 | 0.97 | 0.70 | -0.46 | 1.53 | 2.28 |
| K = 30, seeds 3–5: SM ensemble | 9.62 | 3.67 | 2.29 | 3.46 | 0.29 | 5.18 | 1.39 | 0.55 | -0.52 | 1.90 | 2.28 |
| K = 30, seeds 3–5: D + band after | 8.90 | 3.68 | 1.79 | 3.88 | 0.49 | 4.16 | 0.76 | 0.72 | -0.03 | 1.51 | 2.28 |
| K = 30, seeds 3–5: D ensemble + band after | 9.79 | 3.89 | 2.14 | 3.98 | 0.47 | 4.70 | 1.13 | 0.90 | 0.02 | 1.81 | 2.28 |
| K = 30, seeds 3–5: D model | 8.43 | 3.75 | 1.33 | 3.90 | 0.66 | 3.66 | 0.27 | 0.33 | -0.31 | 1.07 | 2.28 |
| K = 30, seeds 3–5: D ensemble | 9.39 | 3.94 | 1.72 | 4.03 | 0.62 | 4.24 | 0.70 | 0.55 | -0.17 | 1.42 | 2.28 |
| K = 100: SM | 8.06 | 2.62 | 1.46 | 4.37 | 0.29 | 4.21 | 0.87 | 0.47 | -1.03 | 1.21 | 2.19 |
| K = 100: SM ensemble | 9.17 | 3.52 | 2.09 | 3.23 | 0.28 | 4.73 | 1.01 | 0.61 | -0.73 | 1.57 | 2.19 |
| K = 100: D + band after | 7.62 | 3.71 | 1.33 | 3.38 | 0.55 | 3.64 | 0.33 | 0.26 | 0.15 | 1.14 | 2.19 |
| K = 100: D ensemble + band after | 8.84 | 3.96 | 1.83 | 3.59 | 0.52 | 4.46 | 0.80 | 0.55 | 0.36 | 1.60 | 2.19 |
| K = 100: D model | 7.00 | 3.83 | 0.77 | 3.36 | 0.72 | 3.04 | -0.33 | -0.21 | -0.33 | 0.60 | 2.19 |
| K = 100: D ensemble | 8.37 | 4.09 | 1.36 | 3.58 | 0.68 | 3.93 | 0.27 | 0.17 | 0.00 | 1.15 | 2.19 |
| K = 30, seeds 0–2: SM | 8.97 | 3.27 | 1.99 | 3.71 | 0.28 | 4.77 | 1.08 | 0.48 | -0.41 | 1.62 | 2.28 |
| K = 30, seeds 0–2: SM ensemble | 9.90 | 3.94 | 2.66 | 3.39 | 0.26 | 5.25 | 1.36 | 1.36 | -0.33 | 2.24 | 2.28 |
| K = 30, seeds 0–2: D + band after | 8.96 | 3.63 | 1.76 | 3.94 | 0.50 | 4.02 | 0.75 | 0.74 | -0.01 | 1.49 | 2.28 |
| K = 30, seeds 0–2: D ensemble + band after | 9.75 | 3.83 | 2.10 | 4.04 | 0.47 | 4.53 | 1.09 | 0.94 | 0.10 | 1.80 | 2.28 |
| K = 30, seeds 0–2: D model | 8.42 | 3.70 | 1.31 | 3.96 | 0.66 | 3.56 | 0.26 | 0.36 | -0.34 | 1.06 | 2.28 |
| K = 30, seeds 0–2: D ensemble | 9.35 | 3.89 | 1.70 | 4.08 | 0.62 | 4.12 | 0.67 | 0.57 | -0.18 | 1.41 | 2.28 |
| K = 30, seeds 0–5: SM | 8.84 | 3.33 | 1.99 | 3.63 | 0.29 | 4.72 | 1.03 | 0.59 | -0.43 | 1.57 | 2.28 |
| K = 30, seeds 0–5: SM ensemble | 9.87 | 3.93 | 2.61 | 3.38 | 0.27 | 5.29 | 1.45 | 1.07 | -0.50 | 2.18 | 2.28 |
| K = 30, seeds 0–5: D + band after | 8.93 | 3.66 | 1.78 | 3.91 | 0.49 | 4.09 | 0.76 | 0.73 | -0.02 | 1.50 | 2.28 |
| K = 30, seeds 0–5: D ensemble + band after | 10.02 | 3.91 | 2.22 | 4.05 | 0.46 | 4.77 | 1.23 | 0.96 | 0.08 | 1.89 | 2.28 |
| K = 30, seeds 0–5: D model | 8.43 | 3.72 | 1.32 | 3.93 | 0.66 | 3.61 | 0.27 | 0.34 | -0.33 | 1.06 | 2.28 |
| K = 30, seeds 0–5: D ensemble | 9.65 | 3.97 | 1.82 | 4.10 | 0.61 | 4.36 | 0.82 | 0.62 | -0.12 | 1.52 | 2.28 |

## The deployment table (DEPLOYMENT.md)

Net SR on 1996–1997, 1998–2021 and 2022–2025, and turnover a day on 1998–2021. The paper's row is Table 2's model; every other row is ours, from cash with books carried across years. Not Table 2 rows: an ensemble or a trading rule changes what is traded, and the paper reports the model alone.

| | Validation 1996–97 | 1998–2021 | 2022–25 | Turnover |
| --- | --- | --- | --- | --- |
| K = 8: paper, Table 2 (model) | – | 1.94 | – | – |
| K = 8: D model (the paper's object) | 8.01 | 1.52 | -0.74 | 0.64 |
| K = 8: D ensemble | 8.96 | 1.84 | -0.64 | 0.60 |
| K = 8: D ensemble + band after | 9.15 | 2.13 | -0.50 | 0.44 |
| K = 8: SM model | 8.68 | 1.72 | -0.80 | 0.31 |
| K = 8: SM ensemble (the deployment variant) | 8.93 | 1.95 | -0.67 | 0.29 |
| K = 30, seeds 0–2: paper, Table 2 (model) | – | 2.28 | – | – |
| K = 30, seeds 0–2: D model (the paper's object) | 8.42 | 1.31 | -0.34 | 0.66 |
| K = 30, seeds 0–2: D ensemble | 9.35 | 1.70 | -0.18 | 0.62 |
| K = 30, seeds 0–2: D ensemble + band after | 9.75 | 2.10 | 0.10 | 0.47 |
| K = 30, seeds 0–2: SM model | 8.97 | 1.99 | -0.41 | 0.28 |
| K = 30, seeds 0–2: SM ensemble (the deployment variant) | 9.90 | 2.66 | -0.33 | 0.26 |
| K = 30, seeds 3–5: paper, Table 2 (model) | – | 2.28 | – | – |
| K = 30, seeds 3–5: D model (the paper's object) | 8.43 | 1.33 | -0.31 | 0.66 |
| K = 30, seeds 3–5: D ensemble | 9.39 | 1.72 | -0.17 | 0.62 |
| K = 30, seeds 3–5: D ensemble + band after | 9.79 | 2.14 | 0.02 | 0.47 |
| K = 30, seeds 3–5: SM model | 8.71 | 1.99 | -0.46 | 0.30 |
| K = 30, seeds 3–5: SM ensemble (the deployment variant) | 9.62 | 2.29 | -0.52 | 0.29 |
| K = 30, seeds 0–5: paper, Table 2 (model) | – | 2.28 | – | – |
| K = 30, seeds 0–5: D model (the paper's object) | 8.43 | 1.32 | -0.33 | 0.66 |
| K = 30, seeds 0–5: D ensemble | 9.65 | 1.82 | -0.12 | 0.61 |
| K = 30, seeds 0–5: D ensemble + band after | 10.02 | 2.22 | 0.08 | 0.46 |
| K = 30, seeds 0–5: SM model | 8.84 | 1.99 | -0.43 | 0.29 |
| K = 30, seeds 0–5: SM ensemble (the deployment variant) | 9.87 | 2.61 | -0.50 | 0.27 |
| K = 100: paper, Table 2 (model) | – | 2.19 | – | – |
| K = 100: D model (the paper's object) | 7.00 | 0.77 | -0.33 | 0.72 |
| K = 100: D ensemble | 8.37 | 1.36 | 0.00 | 0.68 |
| K = 100: D ensemble + band after | 8.84 | 1.83 | 0.36 | 0.52 |
| K = 100: SM model | 8.06 | 1.46 | -1.03 | 0.29 |
| K = 100: SM ensemble (the deployment variant) | 9.17 | 2.09 | -0.73 | 0.28 |

## Contrasts: SM against D + band after (reported, never used to decide except the decision above)

| | Validation 1996–97 | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K = 8: SM − D + band after | +0.66 [-0.14, +1.45] | -0.10 [-0.52, +0.44] | +0.42 [+0.06, +0.78] | +0.14 [-0.34, +0.60] | -0.38 [-0.78, +0.17] | -0.39 [-1.01, +0.27] | -0.13 [-0.48, +0.26] |
| K = 8: SM ensemble − D ensemble + band after | -0.22 [-1.08, +0.59] | -0.19 [-0.53, +0.24] | +0.29 [-0.08, +0.65] | +0.03 [-0.38, +0.44] | -0.32 [-0.71, +0.22] | -0.17 [-0.64, +0.33] | -0.13 [-0.43, +0.22] |
| K = 30, seeds 3–5: SM − D + band after | +0.12 [-0.85, +1.14] | +0.42 [+0.16, +0.72] | +0.73 [+0.36, +1.14] | +0.28 [-0.18, +0.74] | +0.10 [-0.34, +0.56] | -0.59 [-1.46, +0.33] | +0.23 [-0.03, +0.50] |
| K = 30, seeds 3–5: SM ensemble − D ensemble + band after | -0.18 [-1.12, +0.81] | +0.15 [-0.17, +0.47] | +0.48 [+0.09, +0.90] | +0.26 [-0.22, +0.72] | -0.36 [-1.06, +0.33] | -0.54 [-1.35, +0.31] | +0.09 [-0.20, +0.38] |
| K = 100: SM − D + band after | +0.86 [+0.06, +1.69] | +0.05 [-0.47, +0.92] | +0.74 [+0.42, +1.08] | +0.63 [+0.16, +1.15] | +0.19 [-0.40, +1.18] | -1.62 [-2.85, -0.36] | +0.02 [-0.44, +0.74] |
| K = 100: SM ensemble − D ensemble + band after | +0.33 [-0.48, +1.20] | +0.26 [-0.03, +0.57] | +0.27 [-0.05, +0.63] | +0.21 [-0.21, +0.72] | +0.06 [-0.47, +0.73] | -1.10 [-2.31, +0.22] | -0.03 [-0.36, +0.28] |
| pooled: SM − D + band after | +0.79 [+0.02, +1.62] | +0.08 [-0.37, +0.70] | +0.73 [+0.43, +1.04] | +0.35 [-0.04, +0.77] | -0.11 [-0.54, +0.56] | -0.86 [-1.63, -0.11] | +0.04 [-0.34, +0.51] |
| pooled: SM ensemble − D ensemble + band after | +0.23 [-0.53, +1.03] | +0.20 [-0.07, +0.52] | +0.49 [+0.18, +0.80] | +0.29 [-0.08, +0.66] | -0.17 [-0.54, +0.32] | -0.74 [-1.46, +0.00] | +0.12 [-0.12, +0.39] |
| K = 30, seeds 0–2: SM − D + band after | +0.47 [-0.33, +1.24] | +0.51 [+0.20, +0.79] | +0.96 [+0.60, +1.35] | +0.49 [+0.05, +0.91] | -0.13 [-0.71, +0.37] | -0.55 [-1.53, +0.39] | +0.36 [+0.05, +0.64] |
| K = 30, seeds 0–2: SM ensemble − D ensemble + band after | +0.15 [-0.60, +0.90] | +0.56 [+0.25, +0.86] | +0.72 [+0.36, +1.10] | +0.28 [-0.16, +0.70] | +0.43 [-0.19, +1.02] | -0.43 [-1.16, +0.29] | +0.44 [+0.17, +0.72] |
| K = 30, seeds 0–5: SM − D + band after | +0.38 [-0.46, +1.24] | +0.57 [+0.33, +0.82] | +0.92 [+0.57, +1.31] | +0.42 [-0.01, +0.84] | +0.05 [-0.38, +0.44] | -0.65 [-1.46, +0.16] | +0.40 [+0.15, +0.64] |
| K = 30, seeds 0–5: SM ensemble − D ensemble + band after | -0.15 [-0.90, +0.60] | +0.39 [+0.08, +0.71] | +0.52 [+0.16, +0.91] | +0.22 [-0.23, +0.64] | +0.10 [-0.57, +0.73] | -0.58 [-1.29, +0.11] | +0.29 [+0.01, +0.57] |

## Pessimism on the fresh cells (pessimism.py's forms and grids; κ chosen on 1996–1997)

**Kept for 0 of 16 ensemble–form pairs.** Kept means a chosen κ > 0 whose interval against κ = 0 on 1996–1997 lies above 0 (biased upward, since κ was chosen on the same days). At K = 30, seeds 0–5, s comes from six seeds.

| | κ chosen | validation, chosen − κ = 0 | kept | 1998–2021, chosen − κ = 0 | 2022–25, chosen − κ = 0 |
| --- | --- | --- | --- | --- | --- |
| K = 8: SM ensemble, shrink | 0 |  | no |  |  |
| K = 8: SM ensemble, widen | 0 |  | no |  |  |
| K = 8: D ensemble + band after, shrink | 0 |  | no |  |  |
| K = 8: D ensemble + band after, widen | 0 |  | no |  |  |
| K = 30, seeds 3–5: SM ensemble, shrink | 0 |  | no |  |  |
| K = 30, seeds 3–5: SM ensemble, widen | 0 |  | no |  |  |
| K = 30, seeds 3–5: D ensemble + band after, shrink | 0 |  | no |  |  |
| K = 30, seeds 3–5: D ensemble + band after, widen | 0 |  | no |  |  |
| K = 100: SM ensemble, shrink | 0 |  | no |  |  |
| K = 100: SM ensemble, widen | 0 |  | no |  |  |
| K = 100: D ensemble + band after, shrink | 0 |  | no |  |  |
| K = 100: D ensemble + band after, widen | 0.25 | +0.10 [-0.08, +0.29] | no | +0.17 [+0.13, +0.22] | +0.18 [+0.11, +0.25] |
| K = 30, seeds 0–5: SM ensemble, shrink | 0 |  | no |  |  |
| K = 30, seeds 0–5: SM ensemble, widen | 0 |  | no |  |  |
| K = 30, seeds 0–5: D ensemble + band after, shrink | 0 |  | no |  |  |
| K = 30, seeds 0–5: D ensemble + band after, widen | 0 |  | no |  |  |

### The whole grid (net SR; a diagnostic)

| | validation | 1998–2021 | 2022–25 |
| --- | --- | --- | --- |
| K = 8: SM ensemble, shrink, 0 | 8.93 | 1.95 | -0.67 |
| K = 8: SM ensemble, shrink, 0.25 | 8.84 | 1.96 | -0.63 |
| K = 8: SM ensemble, shrink, 0.5 | 8.76 | 1.95 | -0.61 |
| K = 8: SM ensemble, shrink, 1 | 8.69 | 1.94 | -0.57 |
| K = 8: SM ensemble, shrink, 2 | 8.52 | 1.90 | -0.52 |
| K = 8: SM ensemble, shrink, 4 | 8.26 | 1.83 | -0.48 |
| K = 8: SM ensemble, widen, 0 | 8.93 | 1.95 | -0.67 |
| K = 8: SM ensemble, widen, 0.25 | 8.59 | 1.90 | -0.59 |
| K = 8: SM ensemble, widen, 0.5 | 8.15 | 1.82 | -0.53 |
| K = 8: SM ensemble, widen, 1 | 7.41 | 1.70 | -0.38 |
| K = 8: SM ensemble, widen, 2 | 5.83 | 1.38 | -0.18 |
| K = 8: D ensemble + band after, shrink, 0 | 9.15 | 2.13 | -0.50 |
| K = 8: D ensemble + band after, shrink, 0.25 | 9.03 | 2.14 | -0.48 |
| K = 8: D ensemble + band after, shrink, 0.5 | 8.96 | 2.14 | -0.47 |
| K = 8: D ensemble + band after, shrink, 1 | 8.85 | 2.12 | -0.45 |
| K = 8: D ensemble + band after, shrink, 2 | 8.68 | 2.09 | -0.42 |
| K = 8: D ensemble + band after, shrink, 4 | 8.44 | 2.01 | -0.40 |
| K = 8: D ensemble + band after, widen, 0 | 9.15 | 2.13 | -0.50 |
| K = 8: D ensemble + band after, widen, 0.25 | 9.08 | 2.22 | -0.45 |
| K = 8: D ensemble + band after, widen, 0.5 | 8.81 | 2.26 | -0.41 |
| K = 8: D ensemble + band after, widen, 1 | 8.20 | 2.27 | -0.37 |
| K = 8: D ensemble + band after, widen, 2 | 7.17 | 2.04 | -0.28 |
| K = 30, seeds 3–5: SM ensemble, shrink, 0 | 9.62 | 2.29 | -0.52 |
| K = 30, seeds 3–5: SM ensemble, shrink, 0.25 | 9.59 | 2.24 | -0.45 |
| K = 30, seeds 3–5: SM ensemble, shrink, 0.5 | 9.55 | 2.17 | -0.40 |
| K = 30, seeds 3–5: SM ensemble, shrink, 1 | 9.47 | 2.09 | -0.38 |
| K = 30, seeds 3–5: SM ensemble, shrink, 2 | 9.34 | 1.98 | -0.30 |
| K = 30, seeds 3–5: SM ensemble, shrink, 4 | 9.18 | 1.83 | -0.27 |
| K = 30, seeds 3–5: SM ensemble, widen, 0 | 9.62 | 2.29 | -0.52 |
| K = 30, seeds 3–5: SM ensemble, widen, 0.25 | 9.54 | 2.24 | -0.43 |
| K = 30, seeds 3–5: SM ensemble, widen, 0.5 | 9.62 | 2.11 | -0.38 |
| K = 30, seeds 3–5: SM ensemble, widen, 1 | 9.01 | 1.90 | -0.27 |
| K = 30, seeds 3–5: SM ensemble, widen, 2 | 7.64 | 1.39 | -0.24 |
| K = 30, seeds 3–5: D ensemble + band after, shrink, 0 | 9.79 | 2.14 | 0.02 |
| K = 30, seeds 3–5: D ensemble + band after, shrink, 0.25 | 9.74 | 2.09 | 0.04 |
| K = 30, seeds 3–5: D ensemble + band after, shrink, 0.5 | 9.68 | 2.06 | 0.05 |
| K = 30, seeds 3–5: D ensemble + band after, shrink, 1 | 9.55 | 2.00 | 0.07 |
| K = 30, seeds 3–5: D ensemble + band after, shrink, 2 | 9.34 | 1.93 | 0.09 |
| K = 30, seeds 3–5: D ensemble + band after, shrink, 4 | 9.02 | 1.83 | 0.10 |
| K = 30, seeds 3–5: D ensemble + band after, widen, 0 | 9.79 | 2.14 | 0.02 |
| K = 30, seeds 3–5: D ensemble + band after, widen, 0.25 | 9.79 | 2.23 | 0.04 |
| K = 30, seeds 3–5: D ensemble + band after, widen, 0.5 | 9.62 | 2.24 | 0.06 |
| K = 30, seeds 3–5: D ensemble + band after, widen, 1 | 9.08 | 2.16 | 0.05 |
| K = 30, seeds 3–5: D ensemble + band after, widen, 2 | 8.14 | 1.84 | 0.16 |
| K = 100: SM ensemble, shrink, 0 | 9.17 | 2.09 | -0.73 |
| K = 100: SM ensemble, shrink, 0.25 | 9.11 | 2.18 | -0.86 |
| K = 100: SM ensemble, shrink, 0.5 | 8.99 | 2.18 | -0.91 |
| K = 100: SM ensemble, shrink, 1 | 8.87 | 2.14 | -0.96 |
| K = 100: SM ensemble, shrink, 2 | 8.68 | 2.05 | -0.99 |
| K = 100: SM ensemble, shrink, 4 | 8.38 | 1.90 | -0.98 |
| K = 100: SM ensemble, widen, 0 | 9.17 | 2.09 | -0.73 |
| K = 100: SM ensemble, widen, 0.25 | 9.00 | 2.07 | -0.61 |
| K = 100: SM ensemble, widen, 0.5 | 8.80 | 1.96 | -0.41 |
| K = 100: SM ensemble, widen, 1 | 7.68 | 1.71 | -0.25 |
| K = 100: SM ensemble, widen, 2 | 5.80 | 1.33 | -0.17 |
| K = 100: D ensemble + band after, shrink, 0 | 8.84 | 1.83 | 0.36 |
| K = 100: D ensemble + band after, shrink, 0.25 | 8.81 | 1.80 | 0.40 |
| K = 100: D ensemble + band after, shrink, 0.5 | 8.77 | 1.78 | 0.43 |
| K = 100: D ensemble + band after, shrink, 1 | 8.70 | 1.74 | 0.45 |
| K = 100: D ensemble + band after, shrink, 2 | 8.57 | 1.68 | 0.46 |
| K = 100: D ensemble + band after, shrink, 4 | 8.35 | 1.59 | 0.47 |
| K = 100: D ensemble + band after, widen, 0 | 8.84 | 1.83 | 0.36 |
| K = 100: D ensemble + band after, widen, 0.25 | 8.94 | 2.00 | 0.54 |
| K = 100: D ensemble + band after, widen, 0.5 | 8.88 | 2.09 | 0.62 |
| K = 100: D ensemble + band after, widen, 1 | 8.71 | 2.15 | 0.66 |
| K = 100: D ensemble + band after, widen, 2 | 7.25 | 1.95 | 0.54 |
| K = 30, seeds 0–5: SM ensemble, shrink, 0 | 9.87 | 2.61 | -0.50 |
| K = 30, seeds 0–5: SM ensemble, shrink, 0.25 | 9.80 | 2.60 | -0.42 |
| K = 30, seeds 0–5: SM ensemble, shrink, 0.5 | 9.83 | 2.54 | -0.34 |
| K = 30, seeds 0–5: SM ensemble, shrink, 1 | 9.75 | 2.44 | -0.24 |
| K = 30, seeds 0–5: SM ensemble, shrink, 2 | 9.57 | 2.31 | -0.20 |
| K = 30, seeds 0–5: SM ensemble, shrink, 4 | 9.29 | 2.15 | -0.17 |
| K = 30, seeds 0–5: SM ensemble, widen, 0 | 9.87 | 2.61 | -0.50 |
| K = 30, seeds 0–5: SM ensemble, widen, 0.25 | 9.68 | 2.50 | -0.49 |
| K = 30, seeds 0–5: SM ensemble, widen, 0.5 | 9.26 | 2.31 | -0.33 |
| K = 30, seeds 0–5: SM ensemble, widen, 1 | 8.38 | 1.97 | 0.01 |
| K = 30, seeds 0–5: SM ensemble, widen, 2 | 5.95 | 1.28 | 0.08 |
| K = 30, seeds 0–5: D ensemble + band after, shrink, 0 | 10.02 | 2.22 | 0.08 |
| K = 30, seeds 0–5: D ensemble + band after, shrink, 0.25 | 9.91 | 2.20 | 0.10 |
| K = 30, seeds 0–5: D ensemble + band after, shrink, 0.5 | 9.83 | 2.19 | 0.11 |
| K = 30, seeds 0–5: D ensemble + band after, shrink, 1 | 9.69 | 2.16 | 0.13 |
| K = 30, seeds 0–5: D ensemble + band after, shrink, 2 | 9.47 | 2.10 | 0.15 |
| K = 30, seeds 0–5: D ensemble + band after, shrink, 4 | 9.15 | 2.02 | 0.19 |
| K = 30, seeds 0–5: D ensemble + band after, widen, 0 | 10.02 | 2.22 | 0.08 |
| K = 30, seeds 0–5: D ensemble + band after, widen, 0.25 | 9.99 | 2.32 | 0.13 |
| K = 30, seeds 0–5: D ensemble + band after, widen, 0.5 | 9.78 | 2.36 | 0.16 |
| K = 30, seeds 0–5: D ensemble + band after, widen, 1 | 9.08 | 2.27 | 0.16 |
| K = 30, seeds 0–5: D ensemble + band after, widen, 2 | 7.41 | 1.79 | 0.29 |

