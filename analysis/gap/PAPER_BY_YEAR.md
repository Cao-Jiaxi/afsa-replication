# The paper by year, from its Figure 2, against ours (K = 30)

Written by `analysis/gap/paper_by_year.py`; how each number is made is in its docstring. Table 2 gives net Sharpe ratios only for 1998–2021 as a whole; everything by year or era below is rebuilt from Figure 2 (gross) and, for the net figures, an assumption about how the paper's cost moved over time.

## 1. How well the rebuilt figures match Table 2

Gross μ and σ in % a year. σ is from 10-day blocks within each figure year, pooled over the 24 years. μ is exact. σ is within 2% of Table 2's at K = 30 and within 7% at K = 8 and 100, which moves those Sharpe ratios by up to 0.3: read the era and year figures below to about that precision.

| 1998–2021 | μ rebuilt | μ Table 2 | σ rebuilt | σ Table 2 | SR rebuilt | SR Table 2 | net SR rebuilt (constant cost) | net SR Table 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K = 8 | 15.68 | 15.70 | 4.33 | 4.68 | 3.62 | 3.35 | 2.08 | 1.94 |
| K = 30 | 16.65 | 16.66 | 4.11 | 4.20 | 4.05 | 3.97 | 2.31 | 2.28 |
| K = 100 | 16.45 | 16.45 | 3.80 | 3.64 | 4.33 | 4.52 | 2.09 | 2.19 |

σ of K = 30 by block length (Table 2: 4.20%): 1 days 3.75%, 2 days 3.98%, 5 days 4.12%, 10 days 4.11%, 21 days 4.45%. Single days understate it because the digitizer interpolated the points matplotlib dropped; 5 and 10 days agree.

## 2. By era

For the paper, the net columns subtract its average cost (Table 2's μ − μ_net, 7.14% a year at K = 30), either constant or scaled by era as our reference model's cost moves (1998–2005: 12.2%, 2006–2013: 10.0%, 2014–2021: 6.2%, 1998–2021: 9.5% a year). Ours are measured; for them both net columns would be the same, so only the first is filled.

| run, period | gross μ % | σ % | gross SR | net SR, constant cost | net SR, cost moving like ours |
| --- | --- | --- | --- | --- | --- |
| paper K = 8, 1998–2005 | 32.34 | 4.76 | 6.80 | 5.40 | 5.00 |
| paper K = 8, 2006–2013 | 10.36 | 4.45 | 2.33 | 0.83 | 0.76 |
| paper K = 8, 2014–2021 | 4.33 | 3.74 | 1.16 | −0.62 | −0.01 |
| paper K = 8, 1998–2021 | 15.68 | 4.33 | 3.62 | 2.08 | 2.08 |
| paper K = 30, 1998–2005 | 31.82 | 4.28 | 7.44 | 5.77 | 5.28 |
| paper K = 30, 2006–2013 | 12.44 | 4.23 | 2.94 | 1.25 | 1.16 |
| paper K = 30, 2014–2021 | 5.70 | 3.80 | 1.50 | −0.38 | 0.27 |
| paper K = 30, 1998–2021 | 16.65 | 4.11 | 4.05 | 2.31 | 2.31 |
| paper K = 100, 1998–2005 | 31.61 | 4.10 | 7.70 | 5.63 | 5.02 |
| paper K = 100, 2006–2013 | 12.01 | 3.35 | 3.59 | 1.04 | 0.91 |
| paper K = 100, 2014–2021 | 5.74 | 3.91 | 1.47 | −0.71 | 0.04 |
| paper K = 100, 1998–2021 | 16.45 | 3.80 | 4.33 | 2.09 | 2.09 |
| ours: reference model (5 bp, ridge 1e-4), seeds pooled, 1998–2005 | 25.25 | 3.49 | 7.23 | 3.73 |  |
| ours: reference model (5 bp, ridge 1e-4), seeds pooled, 2006–2013 | 10.88 | 3.28 | 3.32 | 0.27 |  |
| ours: reference model (5 bp, ridge 1e-4), seeds pooled, 2014–2021 | 7.84 | 4.37 | 1.79 | 0.37 |  |
| ours: reference model (5 bp, ridge 1e-4), seeds pooled, 1998–2021 | 14.65 | 3.77 | 3.88 | 1.38 |  |
| ours: ensemble + band, hindsight (5 bp, ridge 1e-4, band 1.2e-3), 1998–2005 | 24.02 | 3.72 | 6.45 | 4.63 |  |
| ours: ensemble + band, hindsight (5 bp, ridge 1e-4, band 1.2e-3), 2006–2013 | 10.31 | 3.88 | 2.66 | 1.29 |  |
| ours: ensemble + band, hindsight (5 bp, ridge 1e-4, band 1.2e-3), 2014–2021 | 7.30 | 4.02 | 1.82 | 1.11 |  |
| ours: ensemble + band, hindsight (5 bp, ridge 1e-4, band 1.2e-3), 1998–2021 | 13.87 | 3.90 | 3.56 | 2.28 |  |
| ours: model, hindsight (10 bp, ridge 3e-4), seeds pooled, 1998–2005 | 27.49 | 4.41 | 6.23 | 4.28 |  |
| ours: model, hindsight (10 bp, ridge 3e-4), seeds pooled, 2006–2013 | 9.27 | 4.36 | 2.13 | 1.11 |  |
| ours: model, hindsight (10 bp, ridge 3e-4), seeds pooled, 2014–2021 | 6.51 | 5.34 | 1.22 | 0.71 |  |
| ours: model, hindsight (10 bp, ridge 3e-4), seeds pooled, 1998–2021 | 14.42 | 4.76 | 3.03 | 1.93 |  |

## 3. By year, K = 30

A Sharpe ratio measured over one year has a standard error of about 1 (more when it is high), and one year's σ from about 24 blocks is uncertain by about ±15%. The paper's 1998–2001 are the least reliable: the curve sits at a level of 1–5 there, where a drawing tolerance is a large share of a day's move.

| year | paper gross μ % | paper σ % | paper gross SR | paper net SR, constant cost | ours, reference model: net SR | ours, ensemble + band (hindsight): net SR | ours, model (hindsight): net SR | our reference model's cost, % a year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1998 | 40.68 | 6.03 | 6.75 | 5.56 | 6.07 | 6.68 | 5.54 | 11.24 |
| 1999 | 41.09 | 4.78 | 8.60 | 7.11 | 5.46 | 5.22 | 5.05 | 11.50 |
| 2000 | 53.16 | 6.39 | 8.32 | 7.20 | 9.32 | 8.71 | 8.74 | 13.31 |
| 2001 | 41.40 | 4.32 | 9.58 | 7.93 | 1.94 | 3.00 | 2.60 | 13.36 |
| 2002 | 29.58 | 4.27 | 6.93 | 5.26 | 3.23 | 4.50 | 4.81 | 12.75 |
| 2003 | 19.24 | 1.97 | 9.78 | 6.15 | −0.09 | 2.46 | 2.28 | 12.15 |
| 2004 | 15.80 | 2.10 | 7.54 | 4.13 | 0.33 | 2.95 | 2.66 | 11.85 |
| 2005 | 13.61 | 1.18 | 11.57 | 5.50 | 0.04 | 2.48 | 1.07 | 11.77 |
| 2006 | 14.89 | 1.60 | 9.33 | 4.86 | −0.40 | 1.79 | 1.50 | 10.72 |
| 2007 | 12.45 | 4.34 | 2.87 | 1.22 | −1.49 | 0.78 | −0.08 | 10.90 |
| 2008 | 24.55 | 8.34 | 2.94 | 2.09 | 1.17 | 0.89 | 0.36 | 11.27 |
| 2009 | 15.88 | 5.16 | 3.08 | 1.69 | 1.17 | 3.34 | 3.43 | 11.44 |
| 2010 | 8.09 | 2.36 | 3.42 | 0.40 | −0.71 | 0.95 | 0.62 | 10.51 |
| 2011 | 13.06 | 3.13 | 4.18 | 1.89 | −0.78 | 0.41 | 0.20 | 10.55 |
| 2012 | 6.13 | 2.65 | 2.32 | −0.38 | 0.02 | 1.43 | 1.51 | 8.30 |
| 2013 | 4.45 | 1.81 | 2.46 | −1.48 | 2.16 | 2.81 | 2.81 | 6.17 |
| 2014 | 3.81 | 1.89 | 2.01 | −1.76 | 0.74 | 1.60 | 0.71 | 6.21 |
| 2015 | 7.67 | 2.49 | 3.08 | 0.21 | 0.24 | 0.74 | 0.29 | 6.32 |
| 2016 | 6.76 | 3.19 | 2.12 | −0.12 | −1.25 | −0.01 | −0.20 | 8.19 |
| 2017 | 6.30 | 2.25 | 2.80 | −0.37 | 0.52 | 1.24 | 2.19 | 6.50 |
| 2018 | 2.75 | 2.79 | 0.98 | −1.57 | −0.30 | 0.34 | −0.39 | 5.74 |
| 2019 | 7.06 | 3.52 | 2.00 | −0.02 | 0.02 | 0.77 | 1.90 | 6.97 |
| 2020 | 4.15 | 7.63 | 0.54 | −0.39 | 1.55 | 2.55 | 1.25 | 4.94 |
| 2021 | 7.12 | 3.51 | 2.03 | −0.01 | 0.76 | 1.45 | 0.85 | 4.81 |

## 4. What it shows

- **The paper's 2.28 is a 1998–2006 result.** Its estimated net SR is 5.77 on 1998–2005, 1.25 on 2006–2013 and −0.38 on 2014–2021 (+0.27 if its cost fell as ours did).
- **The whole gap to the paper is in 2001–2006.** Our edge broke in spring 2001 (EXTENSION.md); the paper's held to about 2006, at 14–19% gross a year with 1–2% volatility. From 2007 on, our hindsight objects are level with the paper's estimated net or above it.
- **Our best fixed ensemble + band matches the paper's 2.28 over 1998–2021 with less gross return** (13.9% against 16.7%), because it trades less and at lower σ. It is a deployment layer chosen in hindsight, not the paper's object.
- **Caveat.** The paper's yearly cost is unknown; the two cost cases bracket it only if its turnover moved the way ours did.
