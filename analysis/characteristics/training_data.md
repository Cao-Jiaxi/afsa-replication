# Training arrays from our own WRDS downloads

Written 2026-11-13. Code: `afsa\build_wrds.py`; tests: `tests\test_build_wrds.py`. The 35 monthly characteristics come from `afsa\chars_market.py` and `afsa\chars_accounting.py` (reports `market.md` and `accounting.md` in this folder); this report covers what is added on top: the calendar, membership, returns, the risk-free rate, the 4 daily characteristics, the ranking and the checks. Only aggregate statistics appear here.

**Changed 2026-11-13 after four independent audits** (they found no look-ahead in the assembly and reproduced the arrays bit for bit): rf is spread over the scheduled trading days (sections 4 and 5); Ret_W1 and STD_W1 are rounded to 12 decimals so that exact ties share a rank (section 4); `train_span` rejects 1989 as a training year (section 5); two tests of the arrays on disk detect stale arrays (section 9); and the report is corrected on membership (section 2), the D2A coverage of 1989–1996 (section 6), Rel2High against Ander's file (section 7) and the sources that are not point-in-time (section 10). The same day the accounting file became the production build of `accounting.md`: no zero-fill for XSGA, DP, COGS, CEQ and LCT, and a filing lag of six months for foreign filers.

**Both folders were rebuilt on 2026-11-13 at 12:12 with all of this** (19 checks passed for each, then the whole suite: 157 tests). Every number below describes the rebuilt arrays, except those marked "audit", which an auditor measured on the build of 11:06 and which were not measured again. Section 11 lists what the rebuild changed against the build of 11:06.

```powershell
.venv\Scripts\python.exe -m afsa.build_wrds                 # wrds_broad_fill-none\ and wrds_common_fill-none\ (1 minute each, 5.6 GB of memory at the peak)
.venv\Scripts\python.exe -m afsa.build_wrds --check-only    # the 19 checks of section 5, nothing written
.venv\Scripts\python.exe -m afsa.build_wrds --fill last     # the paper's section 4.1 instead -> wrds_broad\, wrds_common\ (checked, not written)
.venv\Scripts\python.exe -m afsa.build_wrds --compare       # section 7: our ranks against the arrays built from Ander's files (3 minutes)
.venv\Scripts\python.exe -m pytest tests\test_build_wrds.py -q    # 29 tests, 4 s (15 synthetic, 7 on each folder on disk)
```

## 1. Conclusions

1. **The arrays are built** for both universes under the user's missing-value policy (`fill = none`): `Data\TrainingData\wrds_broad_fill-none\` and `wrds_common_fill-none\`, 9,108 trading days (1989-11-01 to 2025-12-31) × 500 slots, in the layout of `afsa.build_data`. `Config(universe="wrds_broad", fill="none")` finds them. All 19 checks pass for both. No model was trained.
2. **Loss days now start on 1990-01-02.** The arrays begin 41 trading days earlier, and `afsa.train.train_span` needs 31. A `Panel` on the CPU confirms it. The first window has 2,024 loss days instead of 1,993. The 41 days of 1989 are lookback only: `train_span` now refuses a window whose first training year is 1989 (section 5).
3. **The stock-days of 1990–2025 are exactly Ander's**: 4,527,395 for the broad universe and 4,526,484 for the common one, with no stock-day on one side only.
4. **Nothing suggests an assembly bug.** Against the arrays built from Ander's files, 34 of the 39 characteristics have a rank correlation of 0.981 (OA) to 1.000 and at least 97.0% of ranks within 0.01, except Beta (94.9%). The other five (FC2Y, OP, OL, CF, CF2P) are now far from his by design: his file counts a missing XSGA or DP as 0 and ours leaves the characteristic empty (rank 0) on 22% and 5% of stock-days, so the others are ranked among fewer. Among the stocks that have a value in ours, his ranks re-ranked agree as well as the rest (rank correlation 0.998 to 1.000, 98.8 to 99.5% within 0.01). The gaps follow the documented differences in the raw values (section 7).
5. **The companion code defines three of the four daily characteristics differently from the paper's words** (section 3): a "week" of 7 trading days that skips the latest day, and one day's volume instead of a week's. The arrays follow the paper (5 days up to t−1, weekly volume). The two versions are far apart for Ret_W1 (mean daily rank correlation 0.62) and STD_W1 (0.77). **This is a decision for the user.**
6. **rf comes from the monthly Fama-French rate**, spread over the month's scheduled trading days: the days it has plus the closures nobody knew of when it began (four months, section 4). WRDS's daily series takes 4 distinct values in 1989–2025 and is 0 on 36% of days; ours is 0 on 16.5% (months whose rate rounds to 0.00%, mostly 2011–2015 and 2021).

## 2. What is built

| Folder | Days | Stock-days | From 1990 | PERMNOs | Members a day | Missing returns set to 0 |
| --- | --- | --- | --- | --- | --- | --- |
| `wrds_broad_fill-none` | 9,108 | 4,547,826 | 4,527,395 | 2,125 | 493–500, mean 499.3 | 87 |
| `wrds_common_fill-none` | 9,108 | 4,546,915 | 4,526,484 | 2,122 | 492–500, mean 499.2 | 56 |

Each folder holds the 9 arrays of `afsa.build_data` (`dates`, `permnos`, `permno`, `slot_of`, `ranks`, `levels`, `ret`, `rf`, `run`; 759 MB) and `meta.json`: sources with their sizes and dates, the definitions, the 19 checks and 7 notes, the share of stock-days at rank 0 per characteristic, and the list of missing returns.

**Row t pairs what was known at the close of t−1 with the return of day t.**

- **Calendar.** The days of the CRSP market index, 1989-11-01 to 2025-12-31. Every day of the daily file is in it, and the Fama-French daily file has the same days.
- **Membership.** On the list of the day's holding month (`Data\WRDS\universe\<kind>.parquet`) and with a row in `crsp_dsf_v2` that day. 6,174 of 4,554,000 list stock-days (broad) have no row. All 6,174 fall after the security's last row in the daily file (it was delisted); none falls inside a security's life and none before its first row (the build now counts the three cases in its note). So a member never drops out for a day and returns, and membership on day t needs nothing that is unknown at the close of t−1: a security leaves only after its delisting row. The path "no row for a day, back the next day with a new spell" exists in the synthetic tests only.
- **Slots.** The day's members sorted by PERMNO, from slot 0.
- **Returns.** `dlyret`. The row CRSP adds on the day after the last trade carries the delisting return and counts as a day in the universe (531 broad stock-days, one of them without a return). A missing return counts as 0 and the stock stays.
- **Inputs.** 39 ranks per stock-day, 39 medians and the previous day's rf per day (sections 4 and 6).

## 3. What the companion code shows

`companionpaper\set-sequence-model\src\dataloaders\dataloader_equities.py`, read and not run. Its feature list for the attention-factors experiment (`configs\experiment\equities\attention_factors_equities.yaml`) gives 39 ranked features, their 39 medians and the risk-free rate: the paper's 79 inputs. Line numbers refer to the data loader.

**Timing.** Row t of X holds features computed up to day t, and the loss multiplies the weights of row t by the return of row t+1 (`src\tasks\metrics.py:489`). So row t is the input for target day d = t+1, as in our arrays.

| Topic | Companion code | Our arrays |
| --- | --- | --- |
| Daily return | Return of row t (line 665): day d−1 | The same |
| Weekly return | `"return_1week": (7, 0)` (line 509): the product over rows t−7..t−1, that is days d−8..d−2. Seven trading days, and the latest day is left out | Days d−5..d−1 |
| Weekly volatility | `np.std` (divisor n) of the returns of rows t−7..t−1 (line 738), clipped at the 99.99th percentile. A second one over 50 days stands where CPZ's Variance would be | Sample standard deviation of days d−5..d−1 |
| Volume | The share volume of row t alone (line 683): one day, d−1 | Shares traded over days d−5..d−1 |
| Monthly characteristics on a day | The latest monthly date on or before row t (line 531): for a target day in month m, the values at the end of m−1 | The same (holding month) |
| Ranking | Average ranks over the window's 500 stocks, scaled to [−1, 1] (lines 117–120) | The same over the day's members with a value, scaled to [−0.5, 0.5] |
| Missing values | None reach the ranking. A security with a missing value in any of 5 daily fields over the 10 years is dropped (line 409). Accounting items are filled forward, then backward from the first observation, then with the item's global median (lines 873–894). Missing returns inside a past-return window count as 0 (line 274) | `none`: no value, rank 0. `last`: the last observed value first |
| Medians | The median of the raw values across the 500 stocks, then differenced over time: first differences for returns, differences of log(0.1 + median) or second differences of it for the others (lines 30–59, 130, 314). Not standardized (line 137 is commented out) | Raw medians; the model standardizes them on the training window |
| Risk-free rate | A daily series in the data file, missing days filled forward, standardized with the training window's mean and deviation (line 729); the rate of row t, so of day d−1 | The rate of day d−1; the model standardizes it |
| Target | Return minus the risk-free rate (line 1190) | The same |

**Other things the code shows**, outside what this task had to settle:

- **Its characteristics are not CPZ's.** Past returns are products of daily returns over fixed windows (21, 42–21, 252–42, 252–147, 756–273 rows). Accounting ratios use quarterly Compustat items with no reporting lag, simplified formulas and clipping.
- **The spread feature is constant across stocks.** Line 609 takes `np.mean` of a days × stocks block without an axis, so every stock gets the same value and the same rank.
- **Turnover looks ahead.** Line 963 adds each day's volume to the month the day belongs to, and the days of that month then use the month's total.
- **The backward fill of accounting items looks ahead** for a firm's first reports.

These are observations from reading the code; none was confirmed by running it.

**How far apart the two readings of the daily characteristics are** (mean over the 9,108 days of the daily rank correlation across the broad universe's members; lowest day in brackets):

| | Ret_W1 | STD_W1 | Vol |
| --- | --- | --- | --- |
| Companion's version against ours | 0.62 (−0.18) | 0.77 (0.33) | 0.95 (0.82) |

For Ret_W1, the skipped day alone gives 0.76 (5 days, ending at d−2) and the longer window alone 0.80 (7 days, ending at d−1).

## 4. Definitions as built

**The 4 daily characteristics** of day t, from the five trading days before it (`daily_characteristics`):

| Name | Definition | Missing when |
| --- | --- | --- |
| Ret_D1 | Return of day t−1 | No return on t−1 |
| Ret_W1 | Product of (1 + return) over t−5..t−1, minus 1 | A return is missing on any of the five days |
| STD_W1 | Sample standard deviation (divisor 4) of those five returns | The same |
| Vol | Shares traded over those five days (`dlyvol`, as reported) | A volume is missing on any of the five days |

- The five days are the five trading days of the market calendar before t, not the security's last five rows. After a day without a row, the weekly ones are missing for five days.
- The window follows the paper ("previous day and week return and volatility", "weekly trading volume"). The unit of Vol follows the companion code (shares, not dollars or turnover).
- The divisor of STD_W1 does not affect the ranks. The companion's divisor 5 would scale the median by a constant, which the model's standardization removes.
- **Ret_W1 and STD_W1 are rounded to 12 decimals** (`WEEK_DECIMALS`). Two securities with the same five returns (two share classes, say) have the same weekly return and volatility in exact arithmetic, but a product or a sum taken in another order can differ in the last bit (about 1e-16), and the two are then ranked one position apart instead of sharing an average rank. An auditor found 8 such entries for Ret_W1 and 20 for STD_W1 in the broad arrays when he multiplied the returns newest-first. Rounded, the two values are equal whatever the order. Against the build of 11:06 (measured again after the rebuild), the rounding moves 14 rank entries of Ret_W1 and 18 of STD_W1 (of 4,547,826 each, broad universe; 12 and 18 in the common one; each by 0.001, half a rank step, and none by more) and no entry of Ret_D1, Vol or the 13 market characteristics. The stored medians (float32) change on 181 days for Ret_W1 and 4 for STD_W1, by one float32 step (at most 1.9e-9). A tied pair whose two values straddle a rounding boundary would still be ranked apart: a chance of the order of 1 in 10,000 per pair. Ret_D1 is CRSP's own number and Vol is a sum of whole share counts: both are exact.

**The 35 monthly characteristics** of day t are those of t's holding month in `market.parquet` and `accounting.parquet`. The accounting file is the production build of `accounting.md`: the fiscal rule `safe` (accounts enter four months after the fiscal year-end, six for foreign filers) and no zero-fill for XSGA, DP, COGS, CEQ and LCT, so that FC2Y, OP and OL have no value where XSGA is missing, CF and CF2P where DP is missing. Before the rebuild both files on disk were compared with a fresh build from the current code: identical.

**rf.** rf_d = (1 + rf_m)^(1/n_m) − 1, with rf_m the Fama-French monthly rate (`ff_factors_monthly.rf`) and n_m the trading days the month was scheduled to have when it began (`spread_rf`, `scheduled_days`).

- The monthly rate is that of a one-month bill bought at the end of the previous month, so it is known when the month starts. So is the exchange calendar, except for a closure decided during the month. **n_m is therefore the month's trading days plus its unscheduled closures**, which are written out in `UNSCHEDULED_CLOSURES` with the reason for each:

  | Month | Closed | Why unscheduled | Trading days | Scheduled | rf_d, basis points | With the realized days |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1994-04 | 27 April | Day of mourning for President Nixon, who died on 22 April | 19 | 20 | 1.348 | 1.419 (+0.071) |
  | 2001-09 | 11–14 September | The attacks of 11 September | 15 | 19 | 1.472 | 1.864 (+0.392) |
  | 2004-06 | 11 June | Day of mourning for President Reagan, who died on 5 June | 21 | 22 | 0.363 | 0.381 (+0.017) |
  | 2012-10 | 29–30 October | Hurricane Sandy | 21 | 23 | 0.043 | 0.048 (+0.004) |

- **What this corrects.** The first build divided by the realized days, so the days before a closure carried a rate nobody could have computed then: 0.39 basis points a day too high on the 5 trading days before 11 September 2001 (the 10 days after the closure carried the same rate). The first version of this report put the effect "below 0.01 basis points a day" without measuring it; an auditor measured the figures of the last column, and the rebuild reproduces them (76 days of rf change against the build of 11:06, all in these four months; no other day changes; in September 2001 the rate of each of the 15 days goes from 1.8642 to 1.4717 basis points).
- **What it costs.** Those four months no longer compound to the monthly rate: 25.65 against 27 basis points (−1.35), 22.10 against 28 (−5.90), 7.64 against 8 (−0.36) and 0.91 against 1 (−0.09). The interest of the closed days is not booked on any trading day. Every other month compounds to the monthly rate exactly (largest difference 2.5e-15 over 430 months).
- Closures announced before the month's first trading day are in nobody's count and need nothing: 2 January 2007, 5 December 2018 and 9 January 2025 (days of mourning for Presidents Ford, G. H. W. Bush and Carter). The list also holds 27 September 1985 (hurricane Gloria), which is before the arrays start and changes nothing that is written. None of the listed days is in the CRSP calendar; the build checks it.
- The effect on the model is small either way: the rf input is the same for every stock of a day, which the softmax over stocks cancels (CLAUDE.md, section 5), and the excess return of 15 days of September 2001 rises by 0.39 basis points.
- Against the WRDS daily series: the largest daily difference is 5.3e-5, half a rounding step of that series; ours rounded to 4 decimals equals it on 99.1% of days; both average 2.70% a year. In single years the daily series is off by up to 1.02 points (2003: 0.00% against 1.02%; 2025: 5.00% against 4.16%).
- X's rf input is the rate of the previous trading day (for 1989-11-01, that of 1989-10-31).

**Ranking** (`afsa.build_window.ranks_and_medians`, reused). Each characteristic is ranked across the day's members that have a value: (rank − 1)/(n − 1) − 0.5, average ranks for ties. A member without a value gets 0. The median is over the same values. All 39 are ranked over the day's members; Ander's X ranks the monthly ones over the month's list.

**Missing values.**

- `none` (built): a value that cannot be computed stays missing. No imputation and no carry-forward.
- `last` (supported, checked, not written): the security's last observed value first, with no age limit and including values from before it entered the universe; rank 0 if it never had one.

## 5. Checks

All 19 pass for both universes and are recorded in `meta.json`. Numbers are for the broad universe; the common universe is within the same bounds.

| Check | Result |
| --- | --- |
| Source files exist | 7 files |
| Calendar runs from 1989-11-01 to 2025-12-31, sorted | 9,108 days, 9,067 from 1990 |
| Lookback days exist: loss days of the first window start on 1990-01-02 | 41 days before it, 31 needed; 96.8% of that day's members have 30 prior days in the universe |
| A list of 500 for every month, formed on the last trading day of the month before | 434 lists, 1989-11 to 2025-12 |
| At most 500 stocks a day | 493 to 500 |
| Slots sorted by PERMNO, filled from slot 0 | 2,125 PERMNOs |
| Returns are plausible | −0.954 to 1.039; 71 with a move above 50% |
| rf: each month's days compound to the monthly rate, and to less of it in the four months with an unscheduled closure; never negative; no listed closure is a day of the calendar | Largest difference 2.5e-15 over 430 months; 1994-04 −1.35 bp, 2001-09 −5.90 bp, 2004-06 −0.36 bp, 2012-10 −0.09 bp |
| rf agrees with the daily series up to its rounding | Section 4 |
| X's rf is the rate of the previous trading day | Exact |
| Array shapes and dtypes are those of `afsa.build_data` | ranks [9108, 500, 39], levels [9108, 40] |
| No missing value in any array | |
| Ranks lie in [−0.5, 0.5]; empty slots hold 0 | |
| Ranks are centred on zero each day | Largest cross-sectional mean 3.9e-9 |
| Every characteristic has a value for at least 100 members every day | Fewest: OP 345, OL 345, FC2Y 346, OA 361 (common: OP 341, OL 341, FC2Y 342, OA 365) |
| Members, returns, ranks and medians of 12 sample days equal a direct recomputation | 5,995 stock-days recomputed with pandas from the parquet files: largest rank difference 1.5e-8, medians and returns identical |
| Ret_D1 on day t ranks the return of t−1 | Mean daily rank correlation 1.0000 with t−1, −0.012 with t, −0.017 with t+1 |
| A monthly characteristic's rank is constant within the month, up to re-ranking | In the 164 months whose members never change, no rank moves at all. Elsewhere the largest move is 0.0113 (common: 0.0112), never above the bound implied by the members that changed; 0.001% of stock-month-characteristics move by more than 0.01 |
| Equal-weighted excess return 1998–2021 near the paper's market row | 9.97% mean, 20.52% volatility, Sharpe 0.49. Paper: 8.61%, 20.37%, 0.42. Common universe: 10.53%, 20.69%, 0.51 |

- The market row passes a loose bound (3 points on the mean, 2 on the volatility). The mean is 1.36 points above the paper's, as with Ander's files (9.94%): the universe differs from the paper's, which CLAUDE.md already records.
- After writing, the build reads the 9 files back (equal to memory) and loads a `Panel` on the CPU: `train_span` starts the loss days on 1990-01-02, and a batch has 484 stocks with a signal on that day.
- Stock-days with fewer than 30 prior days in the universe (no signal): 3.76% from 1990-01-02, as before.
- **1989 is not a training year.** The arrays hold 41 days of 1989, enough for a batch of the tests but not a year. `afsa.train.train_span` used to accept test year 1997 with 8 training years (or 1996 with 7) on these arrays and return loss days from row 31, a "1989–1996" window with 10 loss days of 1989; with Ander's arrays the same request raised an error. It now raises "the sample does not cover the 8 years before 1997: it starts on 1989-11-01, inside 1989" whenever the sample starts inside the first training year and later than 7 January (no year's first trading day is later than 4 January in the CRSP calendar of 1984–2025). Every window a run uses is unchanged: for both `wrds_*` folders and the three folders built from Ander's files, the 58 windows of test years 1998–2025 with 8 training years and 1996–2025 with 6 give the same loss days as before (first window of `wrds_broad_fill-none`: 1990-01-02 to 1997-12-31, 2,024 days; of `broad`: 1990-02-14, 1,993 days), and the 52 folders of the window universes start in the November before their first training year.
- **A missing folder now names its builder.** `Config(universe="wrds_broad")` has `fill = "last"` by default and points at `wrds_broad\`, which is not built. The error used to recommend `python -m afsa.build_data --universe wrds_broad`, which cannot build it; it now says `python -m afsa.build_wrds --universe wrds_broad --fill last`. A run on the arrays that exist needs `--universe wrds_broad --fill none`.

## 6. Share of stock-days at rank 0 (no value)

Per cent of the universe's stock-days. Characteristics not listed are below 0.005% (Ret_D1, LME, LTurnover, Rel2High, Spread). The first column is the build of 11:06, which had every accounting zero-fill on and a four-month filing lag for everyone; the others are the rebuilt arrays (`last` is checked in memory, not written).

| | Broad, `none`, build of 11:06 | Broad, `none` | Common, `none` | Broad, `last` |
| --- | --- | --- | --- | --- |
| FC2Y | 0.29 | 22.67 | 22.20 | 19.92 |
| OP | 0.25 | 22.64 | 22.13 | 19.88 |
| OL | 0.23 | 22.62 | 22.11 | 19.88 |
| OA | 20.54 | 20.83 | 18.55 | 18.03 |
| DPI2A | 11.03 | 11.03 | 9.16 | 8.42 |
| CF2P | 1.72 | 7.17 | 5.00 | 5.09 |
| CF | 0.23 | 5.74 | 3.59 | 3.56 |
| D2A | 5.73 | 5.73 | 3.59 | 3.56 |
| Beta | 5.28 | 5.28 | 5.14 | 5.27 |
| r36_13 | 5.20 | 5.20 | 5.05 | 5.16 |
| E2P | 1.72 | 1.72 | 1.64 | 1.66 |
| A2ME, BEME | 1.71 | 1.72 | 1.64 | 1.65 |
| r12_2, r12_7 | 1.60 | 1.60 | 1.52 | 1.59 |
| Q | 1.37 | 1.49 | 1.46 | 1.30 |
| RNA | 0.79 | 0.94 | 0.72 | 0.70 |
| NOA | 0.75 | 0.89 | 0.66 | 0.66 |
| CTO | 0.75 | 0.76 | 0.64 | 0.59 |
| Investment | 0.73 | 0.73 | 0.62 | 0.59 |
| PM | 0.32 | 0.32 | 0.32 | 0.23 |
| PCM | 0.30 | 0.30 | 0.30 | 0.23 |
| PROF | 0.25 | 0.25 | 0.22 | 0.18 |
| C | 0.23 | 0.23 | 0.19 | 0.15 |
| AT, Lev | 0.21 | 0.21 | 0.17 | 0.15 |
| SUV | 0.17 | 0.17 | 0.15 | 0.17 |
| r2_1 | 0.15 | 0.15 | 0.14 | 0.15 |
| Resid_Var | 0.13 | 0.13 | 0.12 | 0.13 |
| Variance, ST_Rev | 0.01 | 0.01 | 0.01 | 0.01 |
| Ret_W1, STD_W1, Vol | 0.01 | 0.01 | 0.01 | 0 |

- **What the accounting change did** (broad, in points): FC2Y +22.38, OP +22.39 and OL +22.39 (no value without XSGA); CF +5.51 and CF2P +5.45 (no value without DP); OA +0.28 (DP and LCT); Q +0.13, RNA +0.15 and NOA +0.14 (CEQ). The other 13 accounting characteristics move by 0.003 point or less (the six-month lag of foreign filers leaves a few stock-months without an earlier fiscal year); in the common universe (US common stocks) not one of their rank entries moves. The 4 daily and the 13 market characteristics are unchanged.
- **FC2Y is no longer one large tie.** With every fill on, FC2Y was exactly 0 wherever XSGA, XRD and XAD were all missing, and those stocks shared one rank at the bottom of the scale: the largest group of equal non-zero FC2Y ranks on a day was 121 stocks (1998-09-01; 120 on the 300 days the auditor sampled) and 18.38% of stock-days were in such a group. In the rebuilt arrays the largest group is 2 on every one of the 9,108 days and 0.89% of stock-days are in one (common universe: 119 → 2, 18.19% → 1.06%). Stock-days with an FC2Y rank below −0.3: 20.35% before, 15.51% now. OP, OL and CF never had a group above 2.
- FC2Y, OP and OL are missing where XSGA is not reported: 19% of banks' stock-months, 76% of insurers', 49% of other financials' and 17% of non-financial firms' (`accounting.md`, section 5). By year the share falls from 29–30% in 1989–1994 to 24–28% in 1995–2003, 20–24% in 2004–2009 and 16–19% from 2010 (common: 28–31%, 24–28%, 18–21% and 17–20%), with no change larger than 3.4 points from one year to the next.
- OA, DPI2A and D2A are missing mostly for financial firms (banks report no current assets or gross plant; see `accounting.md`).
- **D2A, CF and CF2P have a coverage cliff inside the first training window** (a property of the data, not of the build; first found for D2A by an auditor). D2A is missing for 12.6 to 15.1% of the broad universe's stock-months in 1989–1993, 10.7% in 1994, 4.35% in 1995 and 2.85% in 1996 (common universe: 13.7–15.4%, 10.4%, 4.2%, 2.8%). **CF now has no value in the same share of stock-months in each of those years** (it needs DP too), and CF2P in 13.5–16.2%, 11.9%, 5.6% and 5.0%; with the DP fill on they were computed with DP = 0. Compustat has no depreciation (DP) for most financial firms before 1994: among universe stock-months with total assets, DP is missing for 77.8, 77.1, 79.1, 80.2 and 80.5% of financial firms in 1989–1993, then 53.7% (1994), 19.9% (1995) and 12.8% (1996), against 0.1 to 0.8% of the other firms throughout (audit). Financial firms are 15.5 to 22.3% of universe stock-months. Under the rank-0 policy about four fifths of them sit at rank 0 for D2A, CF and CF2P until 1993 and about one tenth afterwards, so these inputs mean something different before and after 1994, within the training window 1990–1997. The arrays built from Ander's files show the same for D2A (audit, on the build of 11:06: their rank-0 share of D2A differs from ours by at most 0.08 point in any year); his CF and CF2P are computed with DP = 0. For scale: the largest year-on-year jumps of a missing share over the 35 monthly characteristics are 7.2 points for r36_13 and Beta (2000 to 2001, new listings), 6.4 points for D2A and CF and 6.3 for CF2P (1994 to 1995). This matters if one of them is studied on its own, for instance when Table 3 drops a group.
- Beta and r36_13 are missing for securities listed less than about three years.
- The four ratios to market value are missing until the July after a security's first December.
- A rank of exactly 0 also occurs for a real value at the middle of the scale: 0.06 to 0.11% of stock-days per characteristic on top of the shares above.

## 7. Comparison with the arrays built from Ander's files

A bug detector, not a target. Broad universe, the 4,527,395 stock-days of 1990–2025, slot by slot (the stock-days and their slots are identical). `last`: our `fill = last` ranks against `Data\TrainingData\broad\` (his X as is). `none`: our `fill = none` ranks against `broad_fill-none\`. "Raw equal": share of universe stock-months where our monthly value equals the one in his `characteristics.parquet` (|difference| ≤ 1e-6 + 1e-4 |his|), among those where both have a value.

| Characteristic | Rank corr., `last` | Within 0.01, `last` | Rank corr., `none` | Within 0.01, `none` | Raw equal |
| --- | --- | --- | --- | --- | --- |
| Ret_D1, STD_W1, Vol | 1.0000 | 100.00% | 1.0000 | 100.00% | – |
| Ret_W1 | 1.0000 | 99.95% | 1.0000 | 99.95% | – |
| LME, Resid_Var, Variance | 1.0000 | 100.00% | 1.0000 | 100.00% | 100.00% |
| LTurnover, SUV | 1.0000 | 100.00% | 1.0000 | 100.00% | 99.80%, 98.84% |
| r12_2, r12_7, r36_13 | 1.0000 | 100.00% | 1.0000 | 100.00% | 92.9–93.2% |
| r2_1, ST_Rev | 1.0000 | 99.97%, 99.96% | 1.0000 | 99.97%, 99.96% | 96.2% |
| Spread | 0.9994 | 99.77% | 0.9994 | 99.77% | 99.84% |
| Rel2High | 0.9998 | 98.98% | 0.9998 | 98.98% | 98.66% |
| Beta | 0.9997 | 94.87% | 0.9997 | 94.87% | 7.61% |
| AT | 0.9998 | 99.46% | 0.9998 | 99.46% | 99.18% |
| A2ME, BEME, E2P | 0.9975–0.9995 | 98.56–99.07% | 0.9975–0.9995 | 98.56–99.07% | 99.18% |
| C, Q, Lev | 0.9981–0.9994 | 99.05–99.39% | 0.9981–0.9994 | 99.05–99.39% | 99.13–99.25% |
| PROF, CTO, PM, D2A | 0.9991–0.9995 | 99.13–99.27% | 0.9991–0.9994 | 99.13–99.28% | 99.14–99.15% |
| RNA | 0.9978 | 99.07% | 0.9979 | 99.06% | 99.08% |
| Investment, NOA, DPI2A | 0.9929–0.9981 | 97.38–99.00% | 0.9934–0.9982 | 97.39–99.00% | 98.81–99.18% |
| OA | 0.9809 | 97.03% | 0.9842 | 97.50% | 98.92% |
| PCM | 0.9998 | 99.52% | 0.9998 | 99.52% | 99.15% |
| CF | 0.9783 | 73.44% | 0.9794 | 65.07% | 99.14% |
| CF2P | 0.9679 | 70.21% | 0.9656 | 52.54% | 99.13% |
| OP | 0.8833 | 37.24% | 0.8804 | 36.55% | 99.04% |
| OL | 0.8478 | 10.19% | 0.8377 | 9.48% | 99.04% |
| FC2Y | 0.7123 | 5.04% | 0.6853 | 4.51% | 99.05% |

**The last five rows are far from his by design.** His file computes them with a missing XSGA or DP counted as 0; ours leaves them without a value (section 6). On those stock-days ours is at rank 0 and his is not, and on every other stock-day our rank is taken among fewer stocks, so nearly every rank moves (FC2Y most: his has a fifth of the cross-section tied at the bottom). Where both files have a value the raw values agree as for the other accounting characteristics (last column). The test that isolates the assembly: on each day keep the stocks that have a value in ours, rank his ranks again among them, and compare (`none`, 1990–2025):

| | Stock-days with a value in his only | Rank corr., his re-ranked | Within 0.01, his re-ranked |
| --- | --- | --- | --- |
| CF | 5.48% | 0.9977 | 98.96% |
| CF2P | 5.42% | 0.9987 | 98.79% |
| OP | 22.36% | 0.9987 | 98.96% |
| OL | 22.35% | 0.9996 | 99.13% |
| FC2Y | 22.35% | 0.9998 | 99.45% |

The same re-ranking gives 97.3% (Investment) and 97.4% (OA) to 99.5% (PCM) for the other 17 accounting characteristics, so the five agree with his as well as the rest once the zero-filled stocks are set aside. A value in ours only is rare for all 22: at most 4 universe stock-months (D2A).

**Where the differences come from.** Every gap traces to a difference in the raw values that the two characteristic reports document, or to the ranking list.

- **The 22 accounting characteristics: the fiscal rule.** Our file uses `safe` (accounts enter four months after the fiscal year-end, six for foreign filers), which changes 0.82% of universe stock-months against the panel's rule (`accounting.md`); his values follow the panel's. Where both have a value, raw values are equal in 98.8% (DPI2A) to 99.3% (Lev). A changed value also moves its neighbours' ranks slightly. For the 17 characteristics outside the five above: among stock-days whose own raw value is equal, 98.2% (Investment), 98.5% (OA) and 98.8% (DPI2A) to 100.00% (Lev, PCM) of ranks are within 0.01; among those whose raw value differs, 3% (Investment, OA) to 44% (PCM). The differences sit where the rule acts: for Investment, 16.2% of July's stock-days, 7.5% of August's and 6.7% of September's are more than 0.01 from his rank, against 0.04 to 0.17% in the other nine months (OA, DPI2A, E2P and AT show the same three months).
- **The six-month lag of foreign filers lowered the agreement a little**, as it should: his file has no such lag. Against the build of 11:06, the share of ranks within 0.01 of his fell for AT from 99.52 to 99.46%, for Investment from 98.16 to 97.39% and for DPI2A from 98.43 to 97.87% (`none`), mostly in August and September, the two months a March year-end now waits longer (Investment: 4.1 → 7.5% and 2.2 → 6.7% of those months' stock-days beyond 0.01). In the common universe the rebuild moved none of these ranks.
- **OA, Investment, NOA, DPI2A, RNA** are the lowest because they are changes or use the previous year's assets: taking the accounts one year earlier moves the value a lot. OA also has a fifth of its stock-days at rank 0, and 0.23% of stock-days have a value in his file only (he counts a missing DP or LCT as 0).
- **Beta.** Our file implements the formula that reproduces the CPZ panel (`market.md`, conclusion 3) with today's factors; his differs in level on 92% of stock-months. The ranks agree: correlation 0.9997.
- **Past returns.** Our monthly return is legacy CRSP's (dividends not reinvested within the month), which the panel has. Raw values differ from his in 4–7% of stock-months, by a median of 2e-6 (ST_Rev) to 8e-6 (r36_13); 1 stock-month in 100,000 differs by more than 0.001. The ranks are the same.
- **Rel2High: a different window, and ours is the panel's** (audit). Raw values differ in 1.3% of stock-months (2,897 of 216,000), by a median of 0.008. Ander's file takes the high over the security's last 252 trading rows; ours takes it over 12 calendar months (f−11..f). On a random sample of 400 of the differing stock-months, a recomputation from prices with 12 calendar months equals ours in 400 and one with the last 252 rows equals his in 399 (253 rows: 155; a rolling calendar year: 79 to 118; 13 months: 49). Of the differing stock-months, 1,254 are in the CPZ panel: ours equals the panel on all 1,254 and his on none. The differences cluster where the 12-month high sits at the edge of the window (6.4% of stock-months in 2002, 3.6% in 2003, 4.6% in 2023, 3.1% in 2025), which is why the rank agreement within 0.01 falls to 94.6% in 2023. Stocks at their 52-week high tie at exactly 1, so a value a hair below 1 on one side breaks a large tie, which gives single rank differences of up to 0.13. The first version of this report suspected a price adjustment and left the cause open. Worth telling Ander.
- **Monthly characteristics, ranking list.** Ours are ranked over the day's members, his over the month's list. The largest effect on a rank is 0.0113 (section 5).
- **Daily characteristics.** The same recipe on returns that differ by exactly 1e-6 on 1.5% of stock-days and nowhere by more (his file has returns truncated or rounded differently in the sixth decimal). Volumes are identical.

**No suspected bug, and nothing left unexplained.** A misalignment by a day or a month would show as rank correlations far below 1 in every row. An auditor used the CPZ panel as arbiter on the universe stock-months where the two files differ; his script was run again on the rebuilt files. The 22 accounting characteristics differ on 1,073 to 1,087 panel rows (Lev 967, DPI2A 1,239) where his value equals the panel (on all of them, or all but 6 for A2ME, BEME, CF2P and E2P) and ours does not, which is the intended effect of the `safe` fiscal rule (968 rows before the six-month lag of foreign filers); the exception is DPI2A, where ours equals the panel on 152 further rows and his does not. The market rows are unchanged: for r2_1 (3,398 rows), r12_2 (6,458 of 6,491), Spread (127), LTurnover (23) and Rel2High (1,254) ours equals the panel; Beta is closer to the panel in ours on 73,588 of 86,560 rows; for SUV his is nearer in the sixth decimal (723 against 278 of 1,062).

## 8. Decisions

| Decision | Reason |
| --- | --- |
| Daily characteristics follow the paper's words: 5 trading days up to t−1, weekly volume. The companion's versions (section 3) are not built | The user's instruction: the paper first, the companion where the paper is silent. The paper says "previous day and week" and "weekly trading volume" |
| Vol is in shares, as CRSP reports them | The paper does not say; the companion code ranks share volume |
| STD_W1 uses the sample standard deviation (divisor 4) | Same ranks as the companion's divisor 5; it matches `afsa.build_window`, so the three builders agree |
| The five days are market trading days; a day without a row or without a return makes the weekly characteristics missing for five days | "All five days required"; a stale window is not a weekly figure |
| All 39 characteristics are ranked over the day's members | The task's definition. A rank then never depends on a security that is not in that day's cross-section |
| Ranks stay in [−0.5, 0.5], medians stay raw | The layout of `afsa.build_data`. The companion scales ranks to [−1, 1] and differences the medians (section 3) |
| The delisting row counts as a day in the universe | The universe rule holds a stock through its delisting return; the stock-days then equal Ander's |
| A missing return counts as 0 and the stock stays | As in `afsa.build_data` |
| rf from the monthly Fama-French rate, spread geometrically over the month's scheduled trading days: the days it has plus the unscheduled closures of 1994-04-27, 2001-09-11 to 14, 2004-06-11 and 2012-10-29 to 30 | The daily series has 4 distinct values in the sample. With the realized days, the rate of the days before a closure would use the closure (0.39 bp a day in September 2001). The four months then compound to slightly less than the monthly rate |
| Ret_W1 and STD_W1 are rounded to 12 decimals before ranking | Values that are equal in exact arithmetic must share a rank whatever the order of the floating-point operations; 12 decimals are far below the 6 decimals of CRSP's returns |
| A training window may not start in a year the arrays hold only in part (1989) | The 41 days of 1989 are the lookback of 1990-01-02, not a training year (`afsa.train.train_span`) |
| The comparison with Ander's file is recorded as a note, not a pass/fail check | His file is not a target; the stock-day count is asserted in the tests instead |
| `fill = last` is supported and passes the checks, but its folders are not written | The task asks for `none` by default |
| `afsa\config.py`: a new tuple `WRDS_UNIVERSES = ("wrds_broad", "wrds_common")`, accepted by `Config`; no default and no folder name changed | `afsa.build_data --universe` lists the keys of `UNIVERSES`, which map to Ander's files, so the new names are kept apart |

## 9. Tests

`tests\test_build_wrds.py`: 29 tests, 15 on synthetic data and 7 on each of the two folders on disk. After the rebuild of 12:12 all 29 pass on the folders in `Data\TrainingData`, and the whole suite passes: 157 tests in 31 s. Before the rebuild 8 of the 14 tests on disk failed, as they should: the folders were older than `accounting.parquet` and than the two changes of section 4.

- **Daily characteristics** on small matrices: values by hand for each of the four; a missing return or volume blanks exactly the five following days; replacing day t and everything after it leaves rows up to t bit-identical and changes row t+1.
- **Daily characteristics, exact ties**: two securities with the same five returns in another order differ in the last bit as computed and are ranked apart; rounded they hold one value and one average rank; five equal returns have a volatility of exactly 0; Ret_D1 and Vol are not rounded.
- **rf spreading**: the daily rate by hand for months of 23, 20 and 22 days; months compound to the monthly rate; a zero month stays zero; a month without a rate raises.
- **rf and unscheduled closures**: September 2001 on a business-day calendar without 11–14 September has 15 days and a rate of (1 + rf_m)^(1/19) − 1 on each; the rates equal those of a calendar on which the exchanges stayed open, so no day depends on the closure; the month compounds to (1 + rf_m)^(15/19) − 1 and the neighbours to their monthly rate; every listed closure adds one day to its month and the three closures announced in advance add none; the build's check passes on these rates and fails on rates spread over the realized days, on a calendar that contains a closure day, and on a negative rate.
- **Ranking with missing values**: three of five ranked among themselves, a tie, a row with no value.
- **A synthetic market** (8 securities, 6 months, lists of 6, 15% of characteristic values missing, a day without a row, a row without a return, a late listing, a delisting row). The assembled arrays equal a reference written with loops and dictionaries, for `none` and for `last`. Membership, the return of 0, the delisting row, the month a characteristic belongs to and the carried value are asserted one by one.
- **List days without a row**: counted after the last row, before the first and inside a life, against a count with loops (1 inside: the planted gap).
- **The build's own recomputation** (`direct_inputs`, pandas from the parquet files) equals the assembled arrays on every fifth day of the synthetic market, for `none` and `last`. Before, only the build exercised it.
- **A missing folder names its builder**: `wrds_broad` asks for `afsa.build_wrds --universe wrds_broad --fill last`, `wrds_common_fill-none` for `afsa.build_wrds --universe wrds_common`, `broad` for `afsa.build_data` as before.
- **No look-ahead in the assembled arrays**: everything dated on or after day D is replaced (returns, volumes, the characteristics and rates of later months); X up to and including D and the returns before D are bit-identical, and X of D+1 changes.
- **Mutants.** Nine deliberately wrong versions of the module were run against the synthetic tests; all nine fail. The four that leak information (same-day Ret_D1, a weekly window that includes day t, a volume window one day late, next month's characteristics) fail the no-look-ahead tests themselves. Seven more for the changes of 2026-11-13 (rf over the realized days, no rounding, a partial first year accepted, the old folder message, gaps counted as delistings, and the two new columns of `afsa.pelger_check.compare` disabled) each fail the test written for them.
- **The arrays on disk** (skipped without them), for both universes: no check failed and the checks that matter are among those passed (their number is not asserted); 9,108 days; 4,527,395 broad stock-days from 1990; shapes, dtypes, sorted slots, finite values, centred ranks; Ret_D1 against the stored returns of t−1; a `Panel` on the CPU whose first window of the rolling experiment (test year 1998) has 2,024 loss days from 1990-01-02 to 1997-12-31. The rank-0 shares are bounded loosely (below 50%, OA above 15%), because they depend on the accounting definitions.
- **Stale arrays** (new, on disk). `meta.json` records the size and the modification time of the 7 source files at the build; the test fails when one of them differs now or is newer than the build, and names the file. Before, arrays left over from an older `market.parquet` or `accounting.parquet` passed the whole suite.
- **Ranks and medians recomputed from the source files** (new, on disk). For three days (1990-01-02, the first loss day and the first day of a month; 2001-09-17, the first day after the closure, whose five-day window spans it; 2025-12-31) the test rebuilds, with code of its own, the members from the month's list and the daily file, the returns, the 4 daily characteristics, the 35 monthly ones from `market.parquet` and `accounting.parquet`, the 39 ranks with the missing-value rule and the 39 medians, and compares them with the arrays (ranks to 2e-6, medians to a relative 1e-6). On the folders of 11:06 it reported exactly the accounting characteristics (the 17 others agreed); on the rebuilt folders all 39 agree.
- **rf on disk** (new): every month recomputed from the monthly file with the closures written out again in the test.

## 10. Open issues and what was not verified

**For the user or the orchestrator.**

- **The daily characteristics: paper or companion?** If the paper's results were produced by the companion pipeline, its Ret_W1 and STD_W1 are different inputs from ours (section 3). `daily_characteristics` takes the window as an argument; the companion's one-day skip and one-day volume would need a few lines more and a folder name of their own.
- **The medians.** The companion code feeds changes in the medians, not levels. Under the paper's equation (1) the medians are inert either way (CLAUDE.md, section 5), so this matters only if the model is changed to use them.
- **Rank scale.** The companion's ranks span [−1, 1], ours [−0.5, 0.5]. That is a factor of 2 on the input of W^K.
- **Zero-filled accounting inputs.** The arrays now hold the production build: five fills off (XSGA, DP, COGS, CEQ, LCT), fifteen on. Three points of `accounting.md` remain for the user: FC2Y is empty in 6.0% of universe stock-months where XSGA is missing but XRD or XAD exists; four kept fills rest on few panel rows (DLTT, DLC, PSTK, PS); and for CEQ and LCT the panel does show the fill (on 5 and 3 securities), so leaving Q, NOA, RNA and OA empty there follows the missing-value policy and not the panel.
- **Three inputs with a fifth of the cross-section at rank 0.** FC2Y, OP and OL now have no value on 22.6% of stock-days (30% in 1990–1993, 16–19% from 2010), OA on 20.8%. Whether the model does better with them empty or zero-filled was not tested: nothing was trained.
- **Resid_Var and Beta** use today's Fama-French factors (the market agent's open issue).
- **CLAUDE.md and NOTES.md need updating by their owner**: a row and commands for `afsa/build_wrds.py`, the two new folders in section 2, the new universes in section 4, the first loss day (1990-01-02, section 4 and the decision "loss days start on the 32nd trading day of 1990"), the test count (157), the two accounting decisions (production fills, six-month lag of foreign filers), and NOTES.md section 2, which is now done.
- **`tests\test_data.py` and `afsa.build_data` still describe Ander's arrays** (9,067 days from 1990-01-02). They are untouched and still pass.

**Not verified.**

- Nothing was trained: whether results change with these arrays is unknown.
- `fill = last` passes the 19 checks in memory for both universes (run again after the rebuild, with the production accounting file) and matches the loop reference on synthetic data, but its folders were not written and no test reads them.
- The direct recomputation covers 12 days per universe. Under `fill = last` it leaves out a day's daily characteristics when a member has a carried value.
- The companion findings come from reading the data loader and the loss; the model code between them was not traced, and nothing was run.
- The rank correlations of section 3 between the two readings of the daily characteristics come from a one-off script, run once.
- The source of the companion's daily risk-free series is not in the code.
- The comparison with Ander's arrays covers the broad universe only. The 1.3% of Rel2High values that differ from his are explained (section 7: his window is the last 252 trading rows, ours 12 calendar months).
- The numbers marked "audit" were measured on the build of 11:06 and not again: the DP coverage of financial firms and the comparison of D2A with Ander's arrays by year (section 6), the Rel2High samples (section 7), and the share counts, delisting rows and factor revisions below. The auditors' mutants and their bit-for-bit reproduction of the arrays were not repeated on the rebuilt folders; the build's own recomputation of 12 days and the tests' recomputation of 3 days were.

**Sources that are not point-in-time.** Everything is built from one download, made in November 2026. Where a source rewrites its past, an input of day t can hold a number that was not known at t−1. None of this can be measured or repaired with the files on disk; the list is for the report.

- **Compustat restatements.** The annual file holds the values as restated and backfilled, not as first filed. Also, 3,475 of the 216,000 broad-universe stock-months (1.61%) use a fiscal year that ended before the security's first CRSP daily row: pre-listing accounts, public through the prospectus at best. Reporting lags, and the foreign filers who now wait six months (1,111 stock-months still enter on the old 20-F deadline day), are in `accounting.md`.
- **CCM links.** The link dates between PERMNO and GVKEY are assigned retroactively.
- **CRSP share counts.** A change in shares outstanding is dated on the as-of date of its source (a month-end or the balance-sheet date of a quarterly report), not on the day it became public. Of 236,271 changes of `shrout` on days without a split, 70% / 94% / 83% / 80% / 77% fall on a month's last trading day in 1984–89 / 1990–99 / 2000–09 / 2010–19 / 2020–25, and 64% / 57% / 29% / 27% / 27% on a quarter's last trading day; about 4.8% and 1.6% would be expected if they were dated when known. So the cap that ranks the top 500 at the end of month m−1, LME, LTurnover and the December caps of A2ME, BEME, CF2P and E2P can contain a share count published weeks later. It is the convention of every paper that uses CRSP, CPZ and AFSA included.
- **CRSP revisions.** Returns and share counts are today's vintage (the revised ADR share counts and the handful of revised returns in `market.md`).
- **Fama-French revisions.** The daily factors behind Resid_Var and Beta are today's WRDS file; French has changed Mkt-RF on 19% of the days of 1984–2016, SMB on 83% and HML on 89% since November 2016 (`market.md`, section 5). The monthly rf is a rate set before the month and is not affected.
- **Delisting amounts.** The delisting return is booked on the delisting row, always the trading day after the last trade here (531 rows in the broad universe), although the amount can be settled later. 20 of the 1,384 delisting rows of the file have no return; 1 is in the broad universe and counts as 0.
- **What is point-in-time by construction:** the daily characteristics (days t−5..t−1), the monthly ones (rows up to the formation date; truncation tests), membership (a row on day t is known at t−1 because no security has a gap inside its life), the fiscal rule `safe`, and now rf in the months with an unscheduled closure.

## 11. What the rebuild of 12:12 changed

Measured by comparing the rebuilt folders with copies of the folders of 11:06, array by array (the copies were deleted afterwards). Before the rebuild `accounting.parquet`, `link.parquet` and `market.parquet` on disk were compared with a fresh build from the current code: identical, so none was rebuilt.

| | Broad | Common |
| --- | --- | --- |
| `dates`, `permnos`, `permno`, `slot_of`, `ret`, `run` | identical | identical |
| `rf`: days that change | 76, all in 1994-04, 2001-09, 2004-06 and 2012-10 | the same |
| rf of a day of September 2001, basis points | 1.8642 → 1.4717 (the month compounds to 22.10 instead of 28) | the same |
| Rank entries of Ret_W1 that change (each by 0.001) | 14 of 4,547,826 | 12 of 4,546,915 |
| Rank entries of STD_W1 that change (each by 0.001) | 18 | 18 |
| Rank entries of Ret_D1, Vol and the 13 market characteristics that change | 0 | 0 |
| Accounting characteristics with a changed rank entry | all 22 | 9: CF, CF2P, Q, FC2Y, OP, RNA, NOA, OA, OL |
| Stock-days without a value: FC2Y / OP / OL | 0.29 / 0.25 / 0.23% → 22.67 / 22.64 / 22.62% | 0.29 / 0.21 / 0.20% → 22.20 / 22.13 / 22.11% |
| Stock-days without a value: CF / CF2P | 0.23 / 1.72% → 5.74 / 7.17% | 0.19 / 1.64% → 3.59 / 5.00% |
| Stock-days without a value: OA / Q / RNA / NOA | 20.54 / 1.37 / 0.79 / 0.75% → 20.83 / 1.49 / 0.94 / 0.89% | 18.41 / 1.45 / 0.69 / 0.64% → 18.55 / 1.46 / 0.72 / 0.66% |
| Largest group of equal non-zero FC2Y ranks on a day | 121 → 2 | 119 → 2 |
| FC2Y stock-days in such a group | 18.38% → 0.89% | 18.19% → 1.06% |
| Rank correlation of the new FC2Y / OL / OP / CF2P / CF ranks with the old | 0.70 / 0.86 / 0.89 / 0.97 / 0.98 | 0.70 / 0.85 / 0.90 / 0.99 / 0.99 |
| Fewest members with a value on a day | OA 363 → OP 345 | OP 341 |
| Equal-weighted excess return 1998–2021 (the rf change) | 9.964% → 9.966% | 10.528% → 10.530% |

- In the broad universe the other 13 accounting characteristics change on 110,000 (PCM) to 515,000 (Investment) rank entries, of which 2,900 to 10,700 by more than 0.01, with a rank correlation of 0.999 or more between old and new: the six-month lag of foreign filers changes the value of 257 stock-months (`accounting.md`), and each changed value shifts the ranks of the stocks between its old and its new position by one step.
- FC2Y, OP and OL are different inputs from before: the fifth of the cross-section that sat in a tie at the bottom of FC2Y, or carried an OP and an OL computed without XSGA, is now at rank 0 and the rest are spread over the whole scale.
- The medians in `levels` change with the values: on every day for FC2Y and OL, on 8,861 to 8,964 days for CF, CF2P and OP (broad), on fewer for the others.
