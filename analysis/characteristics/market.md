# The 13 monthly market characteristics

Written 2026-11-13. Code: `afsa\chars_market.py`; tests: `tests\test_chars_market.py`. "The panel" is the Chen-Pelger-Zhu characteristic file (`Data\WRDS\pelger\characteristics.parquet`, complete cases only, holding months to 2016-12). "Equal" is the match rate of `afsa.pelger_check.compare`: the share of stock-months with |ours − theirs| ≤ 1e-6 + 1e-4 |theirs|; "within 1%" is the same with 1e-2. "Relative 1e-4" has no absolute term (|ours − theirs| ≤ 1e-4 |theirs|, the helper's `rel1e4`), and "ours missing" counts the stock-months where the panel has a value and ours is NaN (`ours_nan`); those are in no rate. Only aggregate statistics appear here.

Corrected 2026-11-13 after four independent audits: how much French's factors have changed (conclusion 2, section 5), the effect of the factor vintage on Resid_Var's ranks in the universe (section 5), rf in Resid_Var (section 3), the 1985 dip (sections 5 and 6), the two new columns (section 4) and the share of the universe the panel can verify (section 4). Numbers marked "audit" were measured by the auditors with their own code; the others come from `check` or from the scripts of this report.

```powershell
.venv\Scripts\python.exe -m afsa.chars_market build    # Data\WRDS\characteristics\market.parquet (15 s, 4 GB of memory at the peak)
.venv\Scripts\python.exe -m afsa.chars_market check    # section 4 below (10 s)
.venv\Scripts\python.exe -m afsa.chars_market check --ff-vintage Data\WRDS\characteristics\reference\F-F_Research_Data_Factors_daily_201611.txt   # section 5
```

Sections 4 and 5 are the output of `check`. The formula variants of section 3 were measured with one-off scripts that are not in the repository; the counts are given so they can be redone.

## 1. Conclusions

1. **Eleven characteristics reproduce the panel with today's data**: equal is 98.7% (SUV) to 100.0% of about 256,000 stock-months (target 98%). Without the absolute term of the tolerance (relative 1e-4) the range is 98.0% (SUV) to 100.0% (section 4).
2. **Resid_Var and Beta reproduce the panel too, but only with the Fama-French factors of the panel's time**: equal is 99.95% and 99.77% with Kenneth French's daily file of November 2016. With today's WRDS factors (the file on disk) they are within 1% of the panel in 80.0% and 99.7% of stock-months, with rank correlations of 0.9998 and 1.0000. French's revisions since then are not small: against his 2016 file, Mkt-RF differs on 19% of the trading days of 1984–2016, SMB on 83% and HML on 89% (audit). Beta's ranks do not notice. Resid_Var's do, a little: in the broad universe of 1990–2016 its rank on [−0.5, 0.5] differs from the one the 2016 factors give by 0.0051 on average, by 0.038 at the 99th percentile, and by more than 0.01 in 15.5% of stock-months, most in 1994–95 (audit; section 5).
3. **Beta's formula was the open point and is now pinned down.** The panel takes the security's volatility from ln(1 + R) without subtracting the risk-free rate, the market's volatility over every trading day, and the correlation from 3-observation sums of ln(1 + R − rf). The first reading (excess log returns everywhere) gave 90% within a relative distance of 1e-4 on the sample of the time, and 50.8% in an auditor's recount on universe rows; this one gives 99.8%.
4. **Four definitions differ from the written ones** (section 3): r2_1 is the return of the month before last; Spread is the daily high-low range, not a quoted spread; Resid_Var and SUV divide the sum of squared residuals by n − 1; LTurnover is in the units of the legacy CRSP monthly file. ST_Rev needs legacy CRSP's monthly return, which the CIZ monthly file no longer has.
5. **The panel can verify 42.4% of the universe.** It has a row for 56.6% of the broad universe's stock-months of 1990–2016, for none after 2016, and for nearly no bank or insurer (section 4). Where it has a row, no value of ours is missing.
6. **What is left is data vintage.** LME and LTurnover differ for 29 of the panel's 76 ADRs from 2011 on (shares outstanding revised by CRSP). Four PERMNOs have a longer history in the panel than in today's CRSP. The past returns differ in 0.02% of months, which a 24-month window turns into 0.3%.
7. **No input is filled with 0 and nothing is carried forward**: a characteristic whose window lacks data is NaN. A strict switch is therefore not needed for these 13.

## 2. What is built

**Rows.** One per security (PERMNO) and holding month: 619,608 rows, 2,524 PERMNOs, holding months 1984-02 to 2025-12. A security has a row where it has at least one daily row in the formation month (the month before). All 217,000 stock-months of the broad universe and all 217,000 of the common universe (1989-11 to 2025-12) have a row, with the same formation date and with LME × 1000 equal to the cap that ranks the universe.

**Columns of `market.parquet`:** `permno` (int32), `month` (Period[M], the holding month), `formation_date` (the last trading day of the previous month) and the 13 characteristics as float64, NaN where not computable.

**No look-ahead.** Every value of holding month m uses daily rows dated up to the formation date only. Two tests check it bit for bit (section 7).

**Coverage in the broad universe** (share of the 217,000 stock-months with a value):

| ST_Rev | r2_1 | r12_2 | r12_7 | r36_13 | LME | LTurnover | Rel2High | Spread | Variance | Resid_Var | SUV | Beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 99.99% | 99.85% | 98.40% | 98.40% | 94.79% | 100% | 100% | 100% | 100% | 99.99% | 99.87% | 99.83% | 94.71% |

- 94.7% of stock-months have all 13; 99.8% have the eight that need at most three months of history.
- The missing values are recent listings: r36_13 needs 36 months of returns, Beta about three years of daily returns.
- The common universe is within 0.2 points of these numbers.

**Inputs.** `crsp_dsf_v2.parquet` (daily CIZ, 1984-01-03 to 2025-12-31) and `ff_factors_daily.parquet` (WRDS `ff.factors_daily`, every CRSP trading day covered). Delisting rows (`dlydelflg` = Y) are dropped: they are not trading days.

## 3. Definitions as implemented

f is the formation month (holding month − 1). The match rates of the variants are equal rates against the panel.

### Monthly return R(k), ST_Rev

R(k) = ∏(1 + retx_d) − 1 + Σ_d (ret_d − retx_d) ∏_{s<d}(1 + retx_s) over the days of month k with a return, rounded to 6 decimals. This is the monthly return of legacy CRSP, in which a dividend is not reinvested within the month. ST_Rev = R(f).

- The price return and the dividend yield of a day are recomputed from prices and dividend amounts where that agrees with CRSP's own 6-decimal daily fields, because the rounded fields compound into errors of the size of the tolerance.

| ST_Rev variant | Equal |
| --- | --- |
| CIZ `mthret` (daily returns compounded; from 1985-12) | 81.4% |
| Daily `dlyret` compounded | 78.6% |
| Legacy formula from the rounded daily fields | 96.9% |
| As implemented | 99.98% |

- Availability is that of CIZ `mthret`: of 599,921 security-months from 1985-12, none has a return in only one of the two.
- In its listing month a security's return runs from its first close. CIZ does the same; legacy CRSP has no return there. The panel cannot tell (its securities have three years of returns). It concerns 297 of the 217,000 broad-universe stock-months.

### r2_1, r12_2, r12_7, r36_13

| Characteristic | As implemented | Equal | Nearest alternatives |
| --- | --- | --- | --- |
| r2_1 | R(f−1) | 99.98% | R(f), the written "lagged one-month return": 0.04% |
| r12_2 | months f−11..f−1 (11 months) | 99.83% | f−12..f−1: 1.0%; f−11..f: 0.9% |
| r12_7 | months f−11..f−6 (6 months) | 99.91% | f−12..f−6: 1.0%; f−11..f−5: 1.0% |
| r36_13 | months f−35..f−12 (24 months) | 99.68% | f−36..f−12: 1.1%; f−35..f−11: 0.9% |

- In the panel r2_1 is therefore last month's ST_Rev: the two are the same series one month apart.
- Every month of a window must have a return. Counting a missing month as 0 lowers r12_2 to 99.65%.

### LME

The security's own CRSP market cap on its last day with a price in f, in USD millions. It equals CIZ `mthcap` / 1000 on all 598,874 stock-months that are in both files. Equal to the panel: 99.44%; see section 6 for the ADRs.

### LTurnover

round(shares traded in f / 100) / shares outstanding in thousands at the end of f. Legacy CRSP's monthly volume is in whole hundreds of shares and the panel divides it by `shrout` as it is, so the value is 10 times the fraction of shares traded. Ranks do not depend on the unit.

| Variant | Equal |
| --- | --- |
| Shares / (1000 × shrout): a fraction | 0.0% |
| (Shares / 100) / shrout, not rounded | 98.81% |
| As implemented | 99.43% |

### Rel2High

Split-adjusted close at the end of f over the highest split-adjusted close of months f−11..f. Equal: 100.00%.

| Variant | Equal |
| --- | --- |
| Prices not adjusted for splits | 89.7% |
| 13 months, or 11 | 87.7%, 87.1% |
| Highest intraday high instead of close | 6.9% |

- Prices are adjusted with CRSP's cumulative factor. The factor of a past day changes when the security splits later; the ratio does not (there is a test).
- A security listed inside the 12 months uses the days it has, so coverage in the universe is 100%. The panel cannot tell. If a full year were required, about 1.6% of universe stock-months would be NaN (those without r12_2).

### Spread

Mean over the days of f of (high − low) / ((high + low) / 2), with ask and bid in place of high and low on days without a trade (what legacy CRSP's ASKHI and BIDLO hold). Equal: 100.00%.

| Variant | Equal |
| --- | --- |
| Mean quoted spread (ask − bid) / midpoint, the written "bid-ask spread" | 0.2% |
| High-low range on trade days only | 98.7% |

### Variance

Sample variance (divisor n − 1) of the daily returns of months f−1 and f, at least 2 returns. Equal: 99.99%; relative 1e-4: 99.98%. One month gives 0.9%, three months 1.3%, divisor n 0.5%.

### Resid_Var

Sum of squared residuals / (n − 1) from the regression of daily returns in excess of rf on a constant, Mkt-RF, SMB and HML over the days of f−1 and f, at least 20 days. With the factors of November 2016, equal is 99.95% and relative 1e-4 is 99.68%.

| Variant (factors of November 2016) | Equal |
| --- | --- |
| Divisor n − 4, the regression's degrees of freedom | 0.1% |
| Divisor n | 2.5% |
| One month | 1.4% |
| Market factor only | 6.5% |
| Returns not in excess of rf | 99.95% (relative 1e-4: 71.73% against 99.93% in excess of rf, audit) |

- **The panel subtracts rf, as the build does.** "Equal" cannot see it (99.95% either way), because its absolute term of 1e-6 is 0.2 to 0.6% of a typical Resid_Var. Without that term it shows: on 1,500 random broad-universe stock-months that are in the panel (1990-01 to 2016-11, factors of November 2016, the auditor's own code), SSE / (n − 1) of excess returns is within a relative 1e-4 of the panel in 99.93% and SSE / (n − 1) of raw returns in 71.73%. An earlier version of this report said the panel could not tell.
- For Resid_Var and Variance, whose values are of the order of 1e-4 to 1e-3, the relative rate is the one to quote. The other rows of the table are far enough from 100% for either measure.
- The written definition says "variance of the residuals"; the divisor n − 1 is the plain sample variance of the residuals.

### SUV

Regression of daily share volume on a constant, max(R, 0) and max(−R, 0) over the days of f−2 and f−1 (at least 5); SUV is the mean of (volume − prediction) over the days of f divided by √(SSE / (n − 1)) of the estimation. Equal: 98.74%; within 1%: 99.98%.

| Variant | Equal | Within 1% |
| --- | --- | --- |
| Divisor n − 3 | 0.0% | 0.0% |
| Estimated on f−1 only | 0.0% | 1.1% |
| Estimated on f−3..f−1 | 0.0% | 1.7% |
| Estimated on f−1 and f | 0.0% | 0.4% |

- The 1.3% that are not equal are numerical, not a formula difference. The absolute difference has a median of about 1e-6 at every level of SUV and is below 1e-4 in 99.98% of stock-months. The tolerance of "equal" is close to 1e-6 when |SUV| is small: equal is 66% for |SUV| < 0.01, 83% for 0.01–0.03, 97% for 0.03–0.1 and 99.8% or more above 0.1.
- The source of the noise is the sixth decimal of the daily returns. CIZ `dlyret` differs from the return recomputed from prices and rounded to 6 decimals on 2.4% of days, by one unit of the last digit. With the recomputed returns, SUV's equal rises to 99.2% and the share of Variance values identical to the panel (relative 1e-9) from 31% to 58%. The build keeps CRSP's field.
- Monthly volume does not explain it: 99.8% of LTurnover values are identical to the panel.

### Beta

ρ · σ_i / σ_M (Frazzini and Pedersen 2014), no shrinkage.

- σ_i: standard deviation of ln(1 + R) over the security's returns of months f−11..f, at least 120. The risk-free rate is not subtracted.
- σ_M: standard deviation of ln(1 + Mkt-RF) over every trading day of months f−11..f.
- ρ: correlation, over months f−59..f, between sums over 3 consecutive observations of the security of ln(1 + R − rf) and the sums of ln(1 + Mkt-RF) over the same days. At least 750 sums. The sums are formed inside the window, so its first two observations have none.

With the factors of November 2016: equal 99.77%, within 1% 99.95%.

| Variant (factors of November 2016) | Within a relative distance of 1e-4 |
| --- | --- |
| As implemented | 99.76% |
| σ_M over the security's days only | 99.68% |
| σ_i and ρ both from ln(1 + R) − ln(1 + rf) | 90.0% |
| σ_i and ρ both from ln(1 + R − rf) | 49.5% |
| σ_i and ρ both from ln(1 + R) | 35.5% |
| Market as ln(1 + Mkt) with rf added back, security not in excess | 25.4% |
| Simple returns instead of log returns | under 1% |
| Sums that may start before the 60-month window | 14.6% |
| Sums over 3 consecutive trading days only | 99.24% |
| Market's own 3-trading-day sums | 99.22% |
| Volatility over 11 or 13 months | 0.4% |
| Correlation over 59 or 61 months | 1.5% |
| Sums over 2 or 5 observations, or daily returns | 0.2% or less |

- How the rf treatment was found. With excess log returns everywhere, the match was 99.7% in holding years 2014–2015, when rf was zero throughout the windows, and 65–80% in 1989–1992, 2002 and 2008, the years in which rf moved most inside the one-year window. Splitting the convention between the volatility and the correlation raised 1989–1992 from 70.1% to 100.0%.
- The rows of the table that change the return convention or the windows were measured with σ_M on the security's days, the version before the last refinement; it moves a rate by 0.1 point at most.
- σ_M on every trading day against the security's days: they differ in 201 stock-months (securities with days without a return). The panel agrees with every trading day in 97% of them and with the security's days in 2%.
- An auditor's recount on 1,500 universe stock-months that are in the panel (factors of November 2016, own code): excess log returns in both the volatilities and the correlation give 50.8% within a relative 1e-4 (market on the security's days) or 50.9% (market on all days), the formula as implemented 99.8%. That is the 49.5% row. The 90.0% row was measured on another sample before the last refinement and was not recounted; the conclusion does not rest on it.
- The minimum numbers (120 and 750) are CPZ's written ones. The panel cannot test them: requiring 700 or 800 sums changes the compared stock-months by 11 and 12.
- The written definition says "log excess returns" for both parts.

## 4. Match with the panel (the file on disk, today's factors)

| Characteristic | Stock-months | Equal | Relative 1e-4 | Within 1% | Rank correlation | Median relative difference | Ours missing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ST_Rev | 256,510 | 99.98% | 99.81% | 99.99% | 1.0000 | 0 | 18 |
| r2_1 | 256,095 | 99.98% | 99.81% | 99.99% | 1.0000 | 0 | 433 |
| r12_2 | 251,902 | 99.83% | 99.46% | 99.94% | 1.0000 | 2e-15 | 4,626 |
| r12_7 | 251,907 | 99.91% | 99.60% | 99.96% | 1.0000 | 2e-15 | 4,621 |
| r36_13 | 241,353 | 99.68% | 99.30% | 99.89% | 1.0000 | 3e-15 | 15,175 |
| LME | 256,510 | 99.44% | 99.44% | 99.49% | 0.9992 | 0 | 18 |
| LTurnover | 256,510 | 99.43% | 99.43% | 99.48% | 0.9987 | 7e-16 | 18 |
| Rel2High | 251,934 | 100.00% | 100.00% | 100.00% | 1.0000 | 3e-16 | 4,594 |
| Spread | 256,510 | 100.00% | 100.00% | 100.00% | 1.0000 | 9e-16 | 18 |
| Variance | 256,098 | 99.99% | 99.98% | 99.99% | 1.0000 | 2e-6 | 430 |
| SUV | 255,682 | 98.74% | 97.97% | 99.98% | 1.0000 | 2e-6 | 846 |
| Resid_Var | 256,094 | 45.27% | 1.36% | 79.96% | 0.9998 | 5e-3 | 434 |
| Beta | 230,180 | 11.11% | 11.01% | 99.70% | 1.0000 | 5e-4 | 26,348 |

- **Relative 1e-4** is below equal where values come close to zero: a monthly return that differs in its sixth decimal (past returns, 0.2 to 0.4 points), small SUV values (0.8 point), and Resid_Var, whose whole scale is small (with today's factors 1.36% against 45.27%; with the factors of 2016, 99.68% against 99.95%, section 5).
- **Ours missing** are rows of `market.parquet` where the panel has a value and ours is NaN. They are not in "stock-months" and lower no rate. All of them are outside the universe: on the 92,214 panel rows that are broad-universe stock-months the count is 0 for all 13. According to the audit, all but 18 to 136 per characteristic are holding months of 1984–89, whose windows start before the daily file does (1984-01); the 18 of LME, LTurnover, Spread and ST_Rev are the stock-months with daily rows but no price in the formation month (below).

**How much of the universe the panel covers.** The panel has a row for 91,690 of the 162,000 broad-universe stock-months of 1990–2016 (56.6%) and for none of the 54,000 of 2017–2025: 42.4% of the 216,000 stock-months of 1990–2025. With the two lookback months of 1989 it is 92,214 of 217,000. It is nearly silent on financial firms. Its rows from 1984, by the SIC code of our securities (audit): banks 175, insurers 1,931, other financial firms 4,273, non-financial firms 250,149; 539 are REITs, 2,053 MLPs and units, 10,284 ADRs. On the 92,214 universe rows the match is that of the whole panel: equal is 99.50% (LME, LTurnover) to 100.00% for ten characteristics, 98.34% for SUV, 51.0% for Resid_Var and 11.6% for Beta (today's factors). For the other 57.6% of the universe, and for banks and insurers in particular, the formulas rest on the written definitions and on the continuity of each cross-section through 2016–2017, which shows no break.

The panel has 257,355 rows from 1984 for the 2,524 PERMNOs; 256,510 have an LME here. The other 845 are holding month 1984-01 (414, formed before the daily file starts), four PERMNOs whose history is longer in the panel than in today's CRSP (413, section 6), and 18 stock-months with daily rows but no price in the formation month (not investigated). The counts of r12_2, Rel2High, r36_13 and Beta are lower because windows that reach back before 1984 are NaN.

**Equal by period** (holding years):

| Characteristic | 1984–89 | 1990–97 | 1998–2005 | 2006–10 | 2011–16 |
| --- | --- | --- | --- | --- | --- |
| ST_Rev | 99.98% | 99.97% | 99.99% | 99.99% | 99.99% |
| r2_1 | 99.98% | 99.97% | 99.99% | 99.99% | 99.99% |
| r12_2 | 99.82% | 99.79% | 99.83% | 99.88% | 99.85% |
| r12_7 | 99.90% | 99.89% | 99.90% | 99.94% | 99.91% |
| r36_13 | 99.59% | 99.72% | 99.61% | 99.71% | 99.73% |
| LME | 99.99% | 100.00% | 99.99% | 99.90% | 97.52% |
| LTurnover | 99.93% | 99.97% | 99.98% | 99.88% | 97.51% |
| Rel2High | 100.00% | 100.00% | 100.00% | 99.98% | 100.00% |
| Spread | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Variance | 100.00% | 99.99% | 100.00% | 99.99% | 99.99% |
| SUV | 99.33% | 99.37% | 98.82% | 98.38% | 97.98% |
| Resid_Var | 57.44% | 41.21% | 27.14% | 44.82% | 64.80% |
| Beta | 19.16% | 12.93% | 6.99% | 10.13% | 14.29% |

**Equal by security type** (CIZ classification on the formation date):

| Characteristic | US common | Foreign | ADR | REIT | Other |
| --- | --- | --- | --- | --- | --- |
| Stock-months | 228,395 | 15,308 | 10,424 | 539 | 1,844 |
| ST_Rev, r2_1 | 99.98% | 99.97% | 99.99% | 100% | 100% |
| r12_2 | 99.82% | 99.95% | 99.89% | 99.81% | 99.95% |
| r12_7 | 99.90% | 99.95% | 99.91% | 99.81% | 100% |
| r36_13 | 99.67% | 99.71% | 99.71% | 100% | 99.83% |
| LME | 99.97% | 99.98% | 86.98% | 100% | 100% |
| LTurnover | 99.96% | 99.90% | 86.95% | 100% | 100% |
| Rel2High | 100.00% | 99.94% | 100.00% | 100% | 100% |
| Spread | 100.00% | 99.97% | 100.00% | 100% | 100% |
| Variance | 100.00% | 99.95% | 99.98% | 100% | 100% |
| SUV | 98.75% | 98.93% | 98.41% | 98.33% | 97.72% |

No type other than the ADRs stands out, so the formulas hold for every kind of security in the universe.

## 5. Resid_Var and Beta: the vintage of the Fama-French factors

The panel was built with Kenneth French's factor file of its time. The archived public file `F-F_Research_Data_Factors_daily.txt` created "using the 201611 CRSP database" (Internet Archive snapshot of 2017-01-06, stored as `Data\WRDS\characteristics\reference\F-F_Research_Data_Factors_daily_201611.txt`) reproduces it:

| | Stock-months | Equal | Within 1% | Relative distance ≤ 1e-4 | Median relative difference |
| --- | --- | --- | --- | --- | --- |
| Resid_Var, factors of 2016-11 | 256,094 | 99.95% | 99.99% | 99.68% | 1.5e-6 |
| Beta, factors of 2016-11 | 230,180 | 99.77% | 99.95% | 99.76% | 1.3e-6 |
| Resid_Var, today's WRDS factors | 256,094 | 45.27% | 79.96% | 1.36% | 5.2e-3 |
| Beta, today's WRDS factors | 230,180 | 11.11% | 99.70% | 11.01% | 5.0e-4 |

- **How much the factors changed** (audit). Today's WRDS file against the 2016 file on their 8,300 common trading days, 1984-01-03 to 2016-11-30, counting differences above 5e-5 (both files have 4 decimals): Mkt-RF differs on 19.05% of days (by at most 0.0007), SMB on 82.84% (at most 0.0056; correlation 0.99694), HML on 89.49% (at most 0.0076; correlation 0.98874), rf on 2.0%. At least one of the three factors differs on 97.65% of days. An earlier version of this report said "15% of the daily factor returns", which is roughly Mkt-RF alone; SMB and HML are what move Resid_Var. WRDS also keeps rf to 4 decimals (French's file has 5), which moves Beta in the sixth digit.
- The file on disk uses today's factors: they are French's current estimates and run to 2025. The 2016 file ends on 2016-11-30 and serves only to verify the formulas.
- With today's factors the cross-sectional rank of a stock, on the scale of the model's input ([−0.5, 0.5]) and among the panel's securities, differs from the panel's by 0.004 on average for Resid_Var (99th percentile 0.028) and by 0.0004 for Beta (0.003).
- **In the universe the vintage moves Resid_Var's ranks more than a rank correlation of 0.9998 suggests** (audit). Broad universe, 1990-01 to 2016-11, 161,262 stock-months, within-month ranks on [−0.5, 0.5] from today's factors against the same formula with the 2016 factors:

  | | Mean absolute difference | 99th percentile | Largest | Moved by more than 0.01 | By more than 0.05 |
  | --- | --- | --- | --- | --- | --- |
  | Resid_Var | 0.0051 | 0.038 | 0.267 | 15.5% | 0.55% |
  | Beta | 0.0003 | – | 0.011 | – | none |

  The worst years for Resid_Var are 1994–95 (mean 0.0115). Its levels under the two vintages are within 1% of each other in only 64.9% of these stock-months. The missing values are the same under both. For Beta the vintage does not matter in practice.
- **The decision stands**: one vintage, today's, for 1990–2025, and no splice at 2016. The input the model sees for Resid_Var is therefore not exactly the panel's, by the amounts above; for comparison, a rank step between two neighbours of 500 stocks is 0.002.

**By holding year.** "Today" columns: the file on disk against the panel; p5 and p95 are percentiles of ours / theirs. "2016" columns: equal with the factors of November 2016.

| Year | Resid_Var within 1%, today | p5 – p95 | Rank corr. | Equal, 2016 | Beta within 1%, today | p5 – p95 | Equal, 2016 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1984 | 90.7% | 0.990 – 1.009 | 0.9999 | 100.00% | – | – | – |
| 1985 | 75.0% | 0.980 – 1.018 | 0.9998 | 98.11% | – | – | – |
| 1986 | 87.3% | 0.988 – 1.011 | 0.9999 | 100.00% | – | – | – |
| 1987 | 87.3% | 0.988 – 1.010 | 1.0000 | 100.00% | – | – | – |
| 1988 | 79.4% | 0.985 – 1.017 | 0.9999 | 100.00% | – | – | – |
| 1989 | 78.5% | 0.985 – 1.016 | 0.9999 | 100.00% | 99.9% | 0.9992 – 1.0007 | 100.00% |
| 1990 | 86.0% | 0.988 – 1.011 | 1.0000 | 100.00% | 99.8% | 0.9990 – 1.0004 | 100.00% |
| 1991 | 89.6% | 0.990 – 1.010 | 1.0000 | 100.00% | 99.8% | 0.9993 – 1.0005 | 100.00% |
| 1992 | 89.6% | 0.990 – 1.010 | 1.0000 | 100.00% | 99.6% | 0.9995 – 1.0007 | 100.00% |
| 1993 | 73.4% | 0.976 – 1.025 | 0.9997 | 100.00% | 99.7% | 0.9994 – 1.0016 | 100.00% |
| 1994 | 42.4% | 0.944 – 1.055 | 0.9990 | 99.99% | 99.9% | 0.9995 – 1.0017 | 99.86% |
| 1995 | 47.0% | 0.958 – 1.057 | 0.9992 | 100.00% | 99.6% | 0.9982 – 1.0007 | 99.47% |
| 1996 | 52.0% | 0.962 – 1.044 | 0.9995 | 99.99% | 99.4% | 0.9985 – 1.0018 | 99.63% |
| 1997 | 59.2% | 0.972 – 1.033 | 0.9997 | 99.92% | 99.4% | 0.9977 – 1.0014 | 99.71% |
| 1998 | 60.1% | 0.971 – 1.030 | 0.9997 | 100.00% | 99.8% | 0.9984 – 1.0013 | 99.69% |
| 1999 | 46.7% | 0.958 – 1.048 | 0.9993 | 100.00% | 99.9% | 0.9993 – 1.0022 | 99.70% |
| 2000 | 43.3% | 0.953 – 1.069 | 0.9986 | 100.00% | 99.6% | 0.9981 – 1.0038 | 99.74% |
| 2001 | 53.3% | 0.967 – 1.043 | 0.9997 | 99.97% | 98.7% | 0.9986 – 1.0052 | 99.58% |
| 2002 | 65.6% | 0.978 – 1.032 | 0.9998 | 100.00% | 99.3% | 0.9983 – 1.0036 | 99.78% |
| 2003 | 62.9% | 0.976 – 1.029 | 0.9998 | 100.00% | 99.7% | 0.9984 – 1.0030 | 99.83% |
| 2004 | 59.3% | 0.974 – 1.036 | 0.9998 | 100.00% | 99.1% | 0.9986 – 1.0036 | 99.64% |
| 2005 | 61.5% | 0.973 – 1.035 | 0.9996 | 100.00% | 99.3% | 0.9991 – 1.0020 | 99.58% |
| 2006 | 59.8% | 0.973 – 1.036 | 0.9996 | 99.97% | 99.9% | 0.9991 – 1.0006 | 99.91% |
| 2007 | 61.9% | 0.974 – 1.028 | 0.9998 | 100.00% | 100.0% | 0.9996 – 1.0015 | 99.90% |
| 2008 | 71.1% | 0.977 – 1.021 | 0.9999 | 99.99% | 100.0% | 0.9993 – 1.0011 | 99.97% |
| 2009 | 85.5% | 0.988 – 1.013 | 1.0000 | 100.00% | 100.0% | 0.9992 – 1.0008 | 99.87% |
| 2010 | 84.7% | 0.988 – 1.013 | 1.0000 | 100.00% | 99.9% | 1.0001 – 1.0025 | 99.72% |
| 2011 | 77.0% | 0.985 – 1.018 | 0.9999 | 99.99% | 99.9% | 0.9995 – 1.0008 | 99.54% |
| 2012 | 81.3% | 0.987 – 1.016 | 0.9999 | 100.00% | 99.9% | 0.9996 – 1.0009 | 99.74% |
| 2013 | 78.8% | 0.986 – 1.017 | 0.9999 | 99.99% | 100.0% | 0.9993 – 1.0004 | 99.77% |
| 2014 | 86.6% | 0.988 – 1.011 | 1.0000 | 100.00% | 100.0% | 0.9999 – 1.0012 | 99.87% |
| 2015 | 81.1% | 0.984 – 1.014 | 1.0000 | 99.97% | 99.9% | 0.9993 – 1.0006 | 99.61% |
| 2016 | 84.9% | 0.990 – 1.014 | 1.0000 | 100.00% | 99.8% | 0.9997 – 1.0020 | 99.58% |

- **The 98.11% of 1985 is two months** (audit). With the 2016 factors, the share within a relative 1e-4 is 1.000 in every holding month from 1984-11 to 1986-03 except 1985-08 (0.107) and 1985-09 (0.116), where the median of ours / theirs is still 0.9999 and 1.0000. The two windows share July 1985, so the panel's factor file differed from the file of November 2016 on some day of that month. Both months are four years before the universe starts.
- With today's factors the level differences of Resid_Var are centred on 1 in every year (median ratio 0.9997 to 1.0018) and widest in 1994–2007, the years French's revisions changed most. Beta's rank correlation with the panel is 0.9999 or more in every year.
- With the 2016 factors Beta is equal in 100% of stock-months until 1993. From 1994 on, about 550 stock-months of 29 securities differ (20 securities hold 97% of them): single revised daily returns that stay in a 60-month window.

## 6. What remains unexplained or outside the panel's reach

- **ADRs, LME and LTurnover.** 1,357 of the 1,442 LME mismatches are ADRs: 29 of the panel's 76 ADRs, almost all from 2011 on (ADR equal: 98% in 2010, 92% in 2011, 52% in 2013, 43% in 2016). In 98.6% of all LME mismatches the LME ratio is the inverse of the LTurnover ratio, so the shares outstanding differ; CRSP has revised them since the panel was built. Our value equals today's `mthcap` everywhere. Which vintage is right for those ADRs was not investigated.
- **Four PERMNOs** have 413 panel rows (1993–2015) in months for which today's CRSP, daily and monthly, has no row for that PERMNO. In the months after their first row here, r36_13 and Beta are NaN where the panel has values (about 110 more rows each).
- **Resid_Var in holding year 1985: closed.** The 98.1% with the 2016 factors is holding months 1985-08 and 1985-09 only, whose windows share July 1985: one month in which the panel's factor file and the file of November 2016 differ (section 5). It is outside the universe period.
- **SUV**: 1.3% not equal, numerical (section 3).
- **Past returns**: 0.02% of monthly returns differ from the panel (CRSP corrections since 2016); the windows multiply that to 0.1–0.3%.
- **The panel cannot verify**: the minimum numbers of observations (Variance 2, Resid_Var 20, SUV 5, Beta 120 and 750); the short windows of recent listings (Rel2High, the listing month's return); anything after 2016 (25% of the universe stock-months of 1990–2025); securities with less than three years of history, since the panel keeps complete cases only; banks and insurers, of which it has almost none. In all it has a row for 42.4% of the universe's stock-months of 1990–2025 (section 4). That Resid_Var subtracts rf is no longer on this list (section 3).
- **Not point-in-time.** The inputs are today's vintage of CRSP (returns, and share counts, which CRSP dates at the statement or month-end date, not on the day they became public: LME, LTurnover) and of the Fama-French factors (Resid_Var, Beta). Nothing on disk can measure what was known at the time. The list for all three modules is in `training_data.md`, section 10.
- **Not checked against Ander's X file.** The task was the panel.

## 7. Tests

`tests\test_chars_market.py`, 23 tests, all passing (10 s):

- **Formulas on synthetic data** with hand-computed or directly computed expectations: the monthly return with a dividend, its rounding, the five past-return windows and their missing-month rule, LME and LTurnover, Rel2High with a split, Spread with and without a trade, Variance, Resid_Var and SUV against `numpy.linalg.lstsq`, SUV with a regressor that is zero, Beta against a direct calculation (a security with gaps tells the two market volatilities apart), Beta = 1.5 for a security that moves 1.5 times the market, the rf treatment of Beta, the minimum numbers of observations, delisting rows, rows in any order, the output layout.
- **No look-ahead.** (a) Every numeric input dated after a formation date is rescaled at random, 10% of the later rows are removed and the later factors replaced; all rows up to that holding month must be identical bit for bit, and every characteristic of the next month must change. (b) Only the first day after a formation date is changed. Both were run against nine mutants that shift one characteristic a month ahead and one that counts the first day of a month in the month before: test (a) fails on all ten.
- **The comparison helper** (`afsa.pelger_check.compare`) on a synthetic panel: a small value off by 5e-7 counts as equal and not as within a relative 1e-4; a value that is missing on our side where the panel has one is counted in `ours_nan` and lowers no rate; a gap of the panel is not ours.
- **Data tests** (skipped without the files): layout and universe coverage of `market.parquet`; the match rates of section 4 with a small margin; Resid_Var and Beta with the factors of November 2016 (equal above 99.8% and 99.6%).
