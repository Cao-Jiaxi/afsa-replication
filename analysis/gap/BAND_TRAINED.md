# The band trained inside the model

Written by `analysis/gap/band_trained.py` (CLAUDE.md §7). Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0–2; test years 1998–2021 and 2022–2025 with 8-year windows, validation 1996–1997 with 6-year windows. Band δ = 0.0004 (D's validation choice for the ensemble band), η = 1, movers-only rescale: only stocks that trade or are closed are resized.

*Trained model*: the network is scored during training on the book that the band holds, so it is charged only for trades that happen. *D model + band after*: D's stored model, which never saw the band, with the same band applied to its targets. Every path starts from cash on the first day of its period (1998 and 1996) and carries the book across years. Scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).

## Levels

Model rows are seed averages; ensemble rows are one portfolio (the seeds' targets averaged, rescaled, then the band). Columns are 1998–2021 unless named; μ_net and σ in % a year.

| | Validation SR_net | SR | SR_net | μ_net | σ | Turnover | Short | SR_net 1998–2005 | SR_net 2006–13 | SR_net 2014–21 | SR_net 2022–25 | SR_net 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D model | 8.42 | 3.70 | 1.31 | 5.17 | 3.96 | 0.66 | 0.46 | 3.56 | 0.26 | 0.36 | -0.34 | 1.06 |
| D model + band after | 8.96 | 3.63 | 1.76 | 6.90 | 3.94 | 0.50 | 0.46 | 4.02 | 0.75 | 0.74 | -0.01 | 1.49 |
| trained model's target | 8.21 | 4.00 | 1.23 | 4.51 | 3.67 | 0.72 | 0.46 | 3.58 | -0.04 | 0.26 | -0.64 | 0.94 |
| trained model | 8.91 | 3.95 | 1.77 | 6.44 | 3.65 | 0.54 | 0.46 | 4.13 | 0.60 | 0.70 | -0.25 | 1.45 |
| D ensemble | 9.35 | 3.89 | 1.70 | 6.89 | 4.08 | 0.62 | 0.45 | 4.12 | 0.67 | 0.57 | -0.18 | 1.41 |
| D ensemble + band after (stored rescale) | 9.74 | 3.82 | 2.08 | 8.38 | 4.04 | 0.47 | 0.45 | 4.52 | 1.07 | 0.92 | 0.08 | 1.78 |
| D ensemble + band after | 9.75 | 3.83 | 2.10 | 8.45 | 4.04 | 0.47 | 0.45 | 4.53 | 1.09 | 0.94 | 0.10 | 1.80 |
| trained ensemble + band | 9.81 | 4.17 | 2.17 | 8.09 | 3.75 | 0.51 | 0.46 | 4.69 | 0.99 | 0.96 | -0.13 | 1.81 |

The run's own stored results restart the book from its target at each year's start; their 1998–2021 SR_net, 1.77, differs from the continuous book's by -0.00.

## Contrasts

Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), 95% interval.

| | Validation 1996–97 | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trained model − D model + band after | +0.01 [-0.27, +0.27] | +0.02 [-0.06, +0.09] | +0.13 [+0.01, +0.25] | -0.16 [-0.34, -0.01] | -0.03 [-0.13, +0.09] | -0.25 [-0.42, -0.09] | -0.03 [-0.10, +0.04] |
| trained model − D model | +0.57 [+0.33, +0.83] | +0.49 [+0.42, +0.56] | +0.62 [+0.49, +0.74] | +0.36 [+0.22, +0.51] | +0.37 [+0.26, +0.48] | +0.09 [-0.08, +0.27] | +0.42 [+0.36, +0.49] |
| D model + band after − D model | +0.56 [+0.36, +0.79] | +0.47 [+0.42, +0.53] | +0.48 [+0.38, +0.59] | +0.52 [+0.40, +0.66] | +0.40 [+0.34, +0.46] | +0.34 [+0.27, +0.43] | +0.45 [+0.41, +0.50] |
| trained ensemble + band − D ensemble + band after | +0.06 [-0.26, +0.34] | +0.06 [-0.01, +0.14] | +0.16 [+0.04, +0.29] | -0.10 [-0.27, +0.04] | +0.02 [-0.08, +0.14] | -0.23 [-0.40, -0.05] | +0.01 [-0.06, +0.08] |

