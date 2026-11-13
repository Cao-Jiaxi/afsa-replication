# Project notes

Details moved out of `CLAUDE.md` (2026-11-13) because sessions rarely need them. `CLAUDE.md` points here; keep both in step.

## 1. Missing-characteristic policy: effect (`broad_fill-none`, built 2026-11-13)

- **Check.** With `--fill last`, the same recipe reproduces Ander's X: 0.0001% of rank entries are off by more than 1e-4.
- **Share of stock-days at rank 0, under the policy:**

  | | OA | DPI2A | D2A | Q | Elsewhere |
  | --- | --- | --- | --- | --- | --- |
  | `--fill last` → `--fill none` | 18.0 → 20.6% | 8.4 → 11.0% | 3.5 → 5.7% | 1.3 → 1.5% | ≤ 0.2 points |

- **Re-ranking.** 22% of rank entries move slightly, because the other stocks are ranked among fewer.
- **Data read.** `--fill none` (and every `afsa.build_window` run) reads `characteristics.parquet` and `crsp_daily` `ret`, `vol` and `cap`, beyond Ander's X. The 4 daily characteristics are recomputed from `crsp_daily`.

## 2. Lookback before 1990 (done in the WRDS-built arrays, 2026-11-13; §6)

**Why.** A loss day needs X, returns and universe membership on its 31 previous trading days: 30 lags, plus the day before for its cost. So starting loss days on 1990-01-02 takes a panel from 1989-11-15.

**What that needs:**
- The Nov and Dec 1989 lists: the 500 largest at 1989-10-31 and 1989-11-30, same rule.
- Their X rows. These need raw CRSP from about 1986 (r36_13, Beta's 750 days, Rel2High) and Compustat from about 1984. Under `--fill last`, the carry-forward also needs history before 1989.
- Under the alternative lag rule (today's projection on past returns), only returns back to 1989-11-15 would be needed.

**Who can build it.** Ander's `crsp_daily` holds only securities that are in a universe from 1990 on, so a rebuild from his files would miss about 5–11 names in each 1989 list. Our own WRDS download covers every security from 1985-12, so `afsa.wrds_universe` builds the 1989-11 and 1989-12 lists exactly (2026-11-13; §5). Five of their securities are in no list from 1990 on. Their X rows are in `wrds_broad_fill-none` and `wrds_common_fill-none`, which start on 1989-11-01: `train_span` puts the first loss day of the 1998 model on 1990-01-02, with 2,024 loss days. The Ander-based folders still start on 1990-01-02.

**Effect:**
- The first window gets about 2,024 loss days instead of 1,993, so its 16 × 126-day batches stop overlapping.
- Only the 1998 model and the 1996–1997 validation fit change.
- The lookback days feed LongConv, and through their residuals the gradients of W^K and Q, but never the loss.
- The scaling statistics stay on the training years.

## 3. Results by period (seed averages, 1998–2021 baseline)

SR / SR_net, and turnover per day. Gross SR and turnover by period are recorded only here; SR_net by period is also in `analysis\gap\RESULTS.md`.

| | 1998–2005 | 2006–2013 | 2014–2021 |
| --- | --- | --- | --- |
| K = 8 | 6.07 / 3.29, 0.83 | 2.96 / 0.44, 0.72 | 1.81 / 0.45, 0.43 |
| K = 30 | 6.68 / 3.45, 0.81 | 3.06 / −0.06, 0.69 | 1.89 / −0.30, 0.51 |

After 2005, the K = 30 strategy does not earn its costs. Each model is fitted on 8 years whose net Sharpe is far above what the next year delivers (in-sample 5.6 and 2.4 for the last two blocks), so it trades as if the older, richer signal were still there.

## 4. Companion code (`companionpaper\set-sequence-model\`)

From OpenReview forum `c1A3DCni3K`, downloaded 2026-11-12. It is git-ignored and untrusted: read it, never run it. It includes an attention-factors model; most of it is unreviewed.

**Universe rule** (`src\dataloaders\dataloader_equities.py:387-502`):
- *Span:* from train start to test end, i.e. 8 training years + a validation year + a test year.
- *Data filter:* keep securities with no missing value in any of 5 daily CRSP fields (price, two high/low fields, return, volume).
- *Size:* the stored price on the last training day × Compustat quarterly company shares `cshoq` at the last quarter end. `np.abs` is applied only after selection (line 505). Every share class gets the whole company's cap, and a security with no `cshoq` is excluded.
- *Selection:* the 500 largest, fixed for the 10 years.
- *Candidates:* `daily_price_data_n_equities_4000.npz`, the first 4,000 of 36,669 PERMNOs, in an order the code does not show (`configs\experiment\equities\attention_factors_equities.yaml:35`, `scripts\notebooks\equities\process_raw_equities_data.py:57-67`).

**Evaluation** (`scripts\notebooks\equities\eval_equities.py`):
- It reports the validation year (1998–2021) from `last.ckpt`, cut to its first 246 days: 245 returns a year. That gives the 5,880 days of the paper's Figure 2.
- So the list needs complete data through the year after the reported year.
- A like-for-like comparison with Table 2 uses the first 245 returns of each year; `afsa.metrics` scores all 6,040 days.

**Differences from our `window` universe:** `window` requires returns only (through the test year), sizes by CRSP security cap, and draws on Ander's 2,521 securities. Its build docstring still calls it "the companion paper's rule".

**Config:** `attention_factors_equities.yaml` has `n_attn_summary_statistics: True` and `num_states: 77`. These may bear on whether the medians are really inert in the authors' model.

## 5. Universe from WRDS (`afsa.wrds_universe`, 2026-11-13)

**Downloads** (`Data\WRDS\raw\`, each with a `.json` manifest):
- `crsp.msf_v2`, 1985-12 to 2025-12: every security, type and exchange (3,708,556 rows, 34,678 PERMNOs), so filters can vary locally;
- `crsp.stkdelists` in full: one row per PERMNO, with `delistingdt`.

**Rule.** Holding month m takes the 500 largest `mthcap` (thousands of USD) at the month-end row of m−1. Eligible securities:
- are equities, `securitytype` EQTY (this drops FUND, i.e. CEFs and ETFs, and DERV);
- have `primaryexch` N, A or Q;
- trade regular-way, `conditionaltype` RW (this drops 2 when-issued stock-months);
- have a last trading day (`delistingdt`) no earlier than the formation day.

**The delisting test, three variants tried against Ander's file:**
- No delisting test: 428 stock-months differ. Securities that delisted earlier in m−1 still have a month-end row with the cap of their last trade.
- Requiring a price on the formation day: 1 differs. YPF in 2000-09 had no trade on Aug 31 but was still listed.
- Excluding `delistingdt` equal to the formation day: 98 differ. CRSP books the delisting return on the next trading day, the first day of m, so those stocks are held at the formation close and earn it.
- The chosen rule (strictly earlier) gives 0 differences.

**Types,** from the CIZ fields as of each month:
- `sharetype` AD = ADR;
- `issuertype` REIT = REIT;
- `sharetype` UG = MLP (units);
- `usincflg` N = foreign;
- otherwise common.

**Common universe:** EQTY/COM, `sharetype` NS, `issuertype` CORP or ACOR, `usincflg` Y, the counterpart of share codes 10/11.

**Result.** The lists run from holding month 1989-11 (the lookback, §2) to 2025-12: 217,000 stock-months; 2,125 PERMNOs in `broad`, 2,122 in `common`. On 1990–2025 both equal Ander's `universe.parquet` and `universe_comun.parquet` on all 216,000 stock-months, with the same ranks, `mthcap` and types. `me_company` sums the eligible classes of a PERMCO. Not yet built: `gvkey`, which needs the CCM link.

**Sensitivity.** Adding NYSE Arca (R) and Cboe (B) changes 39 stock-months, all Cboe Global Markets (2018-11 to 2025-12, on and off).

## 6. Characteristics and training arrays from WRDS (2026-11-13)

Built by `afsa.chars_market`, `afsa.chars_accounting` and `afsa.build_wrds`, then audited twice by independent agents. The full record is in `analysis\characteristics\` (`market.md`, `accounting.md`, `training_data.md`); this section is the short version.

**Raw files** (`Data\WRDS\raw\`, listed in the docstring of `afsa.wrds_data`):
- daily CRSP (`crsp.dsf_v2`) from 1984 for the 2,524 securities ever in a list, 13.0 M rows;
- Compustat annual from 1978 (INDL/STD, USD and CAD reporters, 140 columns);
- the CCM links, the CRSP market indices, and the Fama-French daily and monthly factors.

**The arbiter.** The CPZ panel (`Data\PelgerOldData\characteristics-replication.csv`, as parquet in `Data\WRDS\pelger\`) holds raw values of 46 characteristics per stock-month to 2016-12, complete cases only. It has 35 of AFSA's 39. It covers 91,690 of the 162,000 universe stock-months of 1990–2016 (56.6%): 69% of non-financial firms' and under 1% of banks'.

**Match** (share of common stock-months equal within 1e-6 + 1e-4 relative; formulas run under the panel's own rules):
- 22 accounting characteristics: 99.30% to 99.95%.
- Past returns 99.68% to 99.98%; Spread, Rel2High and Variance 99.99% or more; LME and LTurnover 99.4% (87% for ADRs from 2011, where CRSP has revised share counts); SUV 98.74% (99.98% within 1%).
- Resid_Var 99.95% and Beta 99.77% with French's daily factor file of 2016-11 (`Data\WRDS\characteristics\reference\`, from the Internet Archive, used only by `check --ff-vintage` and one test). With today's WRDS factors: 45% and 11% equal, rank correlations 0.9998 and 1.0000. French has since changed Mkt-RF on 19% of the days of 1984–2016, SMB on 83% and HML on 89%.

**Where the panel overrides the written definition:**
- r2_1 is the return of month t−2; ST_Rev of t−1.
- Monthly returns are legacy CRSP's (dividends not reinvested within the month), rebuilt from daily data; CIZ `mthret` equals the panel in only 81%.
- Spread is the month's mean daily high-low range over its midpoint, not a quoted spread.
- LTurnover is in the legacy file's units (hundreds of shares over thousands of shares outstanding).
- Resid_Var and SUV divide the squared residuals by n − 1; SUV is estimated on months t−3 and t−2 and evaluated on t−1.
- Beta: log returns; the security's one-year volatility without subtracting `rf`; `rf` only in the five-year correlation of three-day sums.
- Book equity adds TXDITC only up to Compustat fiscal year 1992.
- Q uses Compustat's CSHO × PRCC_F at the fiscal year-end, not CRSP's cap.
- OA is the change of the whole working-capital bracket, depreciation included, over lagged assets.
- Ratios to market equity use the security's own December cap, not the firm's (2.9% of universe stock-months have another share class; 3.5% are ADRs).
- Accounts of CAD reporters are not converted (3.5% of universe stock-months).

**Departures from the panel in the production files:**
- Fiscal rule `safe`: the panel's `fyear` rule with an availability gate of 4 months (6 for ADRs and non-US issuers, classified at the formation month). It differs from the panel's rule on 1,762 universe stock-months (0.8%); the production file equals the panel in 98.1% to 98.8%.
- Today's Fama-French factors.

**Missing values.**
- Market characteristics need: every month of a past-return window; 2 daily returns for Variance, 20 for Resid_Var, 5 estimation days for SUV; 120 daily returns and 750 three-day sums for Beta. The daily four need all five previous trading days. The panel cannot test these minimums: its securities all have three years of history.
- Accounting, counted as 0 when missing (the panel shows each): WCAPCH, CAPX, XRD, XAD, TIE, MIB, IVAO, TXDB, TXP, INVT, TXDITC, DLTT, DLC, PSTK, PS.
- Accounting, left empty when missing: every main item, plus XSGA, DP, COGS (no panel evidence), CEQ and LCT (in the panel on 4 and 3 securities, but a missing equity or liability is not 0).
- Share of universe stock-days at rank 0 (broad): FC2Y 22.7%, OP 22.6%, OL 22.6%, OA 20.8%, DPI2A 11.0%, CF2P 7.2%, CF and D2A 5.7%, Beta 5.3%, r36_13 5.2%; the four market-value ratios 1.7%; r12_2 and r12_7 1.6%; Q 1.5%; the rest under 1%. Most of the accounting gaps are financial firms.
- 151 universe stock-months have a Compustat row with no data for the fiscal year: for 32 of the 46 firm-years Compustat has figures only in the SUMM_STD format (checked on WRDS), which we do not use.

**Arrays** (`afsa.build_wrds`): members of day t are the month's list with a daily row that day; all 39 characteristics are ranked over them, (rank − 1)/(n − 1) − 0.5 with average ranks for ties, 0 without a value; the medians are over the same values; `rf` is the monthly rate over the month's scheduled trading days, and the input is the previous day's.

**Audits.** Round 1 (four agents): no look-ahead under truncation at 10–12 dates; every match rate reproduced with independent code; arrays rebuilt independently. It found the fabricated FC2Y zeros, the 20-F lag, the `rf` of September 2001 and smaller points, all fixed. Round 2 (two agents): both folders rebuilt without `afsa` code on all 9,108 days, bitwise equal; no critical or major finding.

**Not point-in-time, and not fixable from disk:** Compustat restatements and backfill; CCM link dates; CRSP share counts (dated at the statement date); French's factor revisions; delisting amounts.

**Other findings:**
- Ander's Rel2High uses the last 252 trading rows; ours uses 12 calendar months, and equals the panel on all 1,254 differing rows that are in it. His Beta departs from the panel in 1989–1990.
- D2A is missing for about four fifths of financial firms until 1993 (Compustat has no DP for them), so its coverage changes inside the first training window.
- The companion code's inputs are not CPZ's: quarterly accounts with no reporting lag, ranks in [−1, 1], differenced medians, a spread feature that is the same for every stock, and a turnover that looks ahead within the month (read only; `training_data.md`).
