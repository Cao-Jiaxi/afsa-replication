# The band trained inside the model

Written by `analysis/gap/band_trained.py` (CLAUDE.md §7). Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0–2; test years 1998–2021 and 2022–2025 with 8-year windows, validation 1996–1997 with 6-year windows. Band δ = 0.0012 (the width hindsight chose for D's ensemble + band at K = 30 in the scan's stage 2 (the best fixed band over 1998–2021, `SCAN_BUFFER_GRID.md`). It was chosen on the test years, so this is a diagnostic against hindsight, not a selection (CLAUDE.md §8)), η = 1, movers-only rescale: only stocks that trade or are closed are resized.

*Trained model*: the network is scored during training on the book that the band holds, so it is charged only for trades that happen. *D model + band after*: D's stored model, which never saw the band, with the same band applied to its targets. Every path starts from cash on the first day of its period (1998 and 1996) and carries the book across years. Scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).

## Levels

Model rows are seed averages; ensemble rows are one portfolio (the seeds' targets averaged, rescaled, then the band). Columns are 1998–2021 unless named; μ_net and σ in % a year.

| | Validation SR_net | SR | SR_net | μ_net | σ | Turnover | Short | SR_net 1998–2005 | SR_net 2006–13 | SR_net 2014–21 | SR_net 2022–25 | SR_net 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D model | 8.42 | 3.70 | 1.31 | 5.17 | 3.96 | 0.66 | 0.46 | 3.56 | 0.26 | 0.36 | -0.34 | 1.06 |
| D model + band after | 8.57 | 3.38 | 2.06 | 7.92 | 3.86 | 0.32 | 0.46 | 4.29 | 1.10 | 0.93 | 0.14 | 1.77 |
| trained model's target | 7.53 | 4.35 | 0.49 | 1.58 | 3.25 | 0.90 | 0.48 | 2.88 | -1.20 | -0.50 | -1.55 | 0.17 |
| trained model | 8.93 | 4.14 | 2.00 | 6.39 | 3.20 | 0.45 | 0.48 | 4.36 | 0.66 | 0.78 | -0.59 | 1.62 |
| D ensemble | 9.35 | 3.89 | 1.70 | 6.89 | 4.08 | 0.62 | 0.45 | 4.12 | 0.67 | 0.57 | -0.18 | 1.41 |
| D ensemble + band after (stored rescale) | 9.04 | 3.56 | 2.28 | 8.88 | 3.90 | 0.31 | 0.45 | 4.63 | 1.29 | 1.11 | 0.13 | 1.96 |
| D ensemble + band after | 9.00 | 3.56 | 2.33 | 9.19 | 3.95 | 0.30 | 0.45 | 4.65 | 1.35 | 1.11 | 0.18 | 2.02 |
| trained ensemble + band | 9.76 | 4.38 | 2.40 | 7.84 | 3.28 | 0.42 | 0.47 | 4.93 | 0.94 | 1.09 | -0.46 | 1.98 |

The run's own stored results restart the book from its target at each year's start; their 1998–2021 SR_net, 2.00, differs from the continuous book's by -0.00.

## Contrasts

Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), 95% interval.

| | Validation 1996–97 | 1998–2021 | 1998–2005 | 2006–13 | 2014–21 | 2022–25 | 1998–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trained model − D model + band after | +0.60 [-0.01, +1.26] | -0.05 [-0.26, +0.15] | +0.12 [-0.15, +0.43] | -0.47 [-0.88, -0.12] | -0.15 [-0.53, +0.23] | -0.76 [-1.25, -0.28] | -0.16 [-0.34, +0.04] |
| trained model − D model | +0.75 [+0.25, +1.35] | +0.76 [+0.56, +0.96] | +0.91 [+0.64, +1.19] | +0.43 [+0.12, +0.76] | +0.46 [+0.12, +0.79] | -0.25 [-0.74, +0.25] | +0.62 [+0.44, +0.81] |
| D model + band after − D model | +0.15 [-0.38, +0.73] | +0.81 [+0.61, +1.02] | +0.79 [+0.48, +1.09] | +0.90 [+0.54, +1.29] | +0.62 [+0.39, +0.86] | +0.51 [+0.27, +0.76] | +0.77 [+0.61, +0.95] |
| trained ensemble + band − D ensemble + band after | +0.76 [+0.09, +1.49] | +0.07 [-0.15, +0.27] | +0.28 [-0.02, +0.59] | -0.42 [-0.80, -0.09] | -0.02 [-0.41, +0.38] | -0.64 [-1.17, -0.12] | -0.04 [-0.24, +0.16] |

