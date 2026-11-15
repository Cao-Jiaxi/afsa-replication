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

## 7. Month blocks for the rebuilt-history entry rule (`afsa.build_history`, 2026-11-14)

**Why.** Under the current rule (`entry_rule = run`), a member gets a LongConv signal only after 30 consecutive days in the universe. Each lag is its residual from that day's own projection. A stock that entered fewer than 30 days ago has no residuals for its days before entry, because the day × slot arrays hold nothing for it then, so it gets no signal (3.76% of broad member stock-days from 1990-01-02). The decided rule (CLAUDE.md §7) recomputes the 30 lags over day t's universe, the month's list, including the days before a stock entered. On a March day every lag day, January and February ones included, is computed over March's list.

**Layout** (`Data\TrainingData\history\wrds_<universe>_fill-none\`, 14 arrays + `meta.json`; the docstring of `afsa.build_history` defines each):
- one block per holding month, 1989-11 to 2025-12 (434); its slots are the month's list of 500, sorted;
- rows: the 30 trading days before the month's first day, then its 15–23 days (53 with padding); the calendar starts on 1989-09-20;
- `member`, `ranks` [434, 53, 500, 39] float32, `levels`, `ret`, `listed`, `run`, `accounts`, `accounts_post`.

**Example: March 2000.** The lookback rows run from 2000-01-18 to 2000-02-29 (10 January days, 20 February days). 46 of March's 500 securities were not on February's list, 60 not on January's. On 2000-01-18, 494 of March's list traded, 54 of them outside that day's own universe. On March 1, 435 members get a signal under the current rule and 494 under the decided one.

**Checks** (13, all passing for both universes, plus 5 notes, in `meta.json`):
- the daily file has no gap inside a security's life, so `listed` counts the days since listing;
- the months' own rows equal the `build_wrds` arrays on every stock-day (broad: 4,547,826), bit for bit: members, ranks, medians, rf input, returns and run;
- the rows of 12 blocks (611 rows, 360 of them lookback) equal a pandas recomputation from the whole table and the raw daily file (ranks within 1.5e-8);
- a member with `listed` > 30 has a CRSP row on each of its 30 lag rows;
- the accounts flags never turn false;
- no missing value, and ranks are centred;
- Ret_D1 on lookback rows ranks the previous day's return (rank correlation 1.0000 with s−1, −0.018 with s).

**Coverage** (member stock-days from 1990-01-02):

| | Broad | Common |
| --- | --- | --- |
| Current rule (31 days in the universe) | 96.243% | 96.390% |
| Decided rule (listed > 30 and accounts) | 99.787% | 99.843% |
| Decided, counting only fiscal years that end after listing | 98.289% | 98.362% |
| No signal: listed ≤ 30 days / no accounts / both | 4,949 / 4,186 / 513 | 4,575 / 2,227 / 325 |
| Gain a signal / lose one, against the current rule | 164,117 / 3,680 | 158,247 / 1,988 |
| Securities with no accounts on any member day | 6: PERMNOs 16267, 24164, 75142, 75936, 77735, 89457 (1,389 stock-days) | 3: 16267, 69729, 93194 (585) |

- **Lags of the stock-days that gain a signal:** 4,923,510 lag rows (broad), 502 of them on a day without a CRSP return (counted as 0).
- **How much the cross-section moves a rank.** On lookback rows, for securities on both lists, the rank over the block's list differs from the rank in that day's own universe by 0.0045 on average (common: 0.0043).
- **The largest differences**, up to 0.11 (LME), come in months with many entrants. Lists change by 14.4 securities a month on average, and by 30–46 in 2000 and late 2008. A March 2000 day in May 2000's block has 56 securities that were not on March's list, mostly smaller then, so a stock near the bottom of March's list moves up 54 places.

**What the model needs beyond the day × slot arrays:**
1. Returns and characteristics of the days before a member entered: in the blocks.
2. Ranks over day t's list on lag days in earlier months: in the blocks. On the month's own days they equal today's ranks.
3. For the exception: the trading days since listing (`listed`) and the accounts flag: in the blocks.
4. Each day's weights come from the block of its own month, so the weights of the day before a month's first day (for that day's cost) come from the previous month's block.
5. Unchanged: rf; the medians (inert) and their training-window statistics (own rows equal the arrays); explained variance and the loss days (own rows); the first test day's trade from the previous model.
6. Compute: residuals over every block a batch touches (7 or 8 × 53 rows for a 126-day batch, against 157 rows today); LongConv is unchanged. Measured on the RTX 5080 (2026-11-14, 16 models per pass): a step takes 1.0 / 1.3 / 1.7 times as long for K = 8 / 30 / 100, and the peak memory per model is 127 / 168 / 379 MB against 127 / 148 / 271. All 434 blocks take 1.77 GB on the GPU (the day arrays 0.73 GB), which still leaves room for 36 models at K = 30, so `BlockPanel` loads every month by default (`first_year` / `last_year` limit it).

**Model** (2026-11-14): `afsa.panel.BlockPanel` serves the blocks, and `AFSAModel._forward_blocks` computes factors and residuals on every row of the blocks a batch touches, each over its own month's list. Run it with `--universe wrds_broad --fill none --entry_rule history` (`Config.entry_rule` stays `run`). Checks (`tests/test_blocks.py`): on a fixed list it trains exactly as the day × slot path, dropout included; on a changing list (entrants with a history, a new listing, a delisting, missing accounts) it equals a recomputation that builds each lag row by row; no look-ahead; models in one pass are independent; on the blocks on disk its own-row residuals equal the day arrays' (2008, float64, to 5e-15).

**First run** (2026-11-14): K = 30, ridge 1e-4, training at 5 bp, seeds 0–2. Validation 1996–1997 (`attn_K030_universe-wrds_broad_train_years-6_entry_rule-history_ridge-0.0001_fill-none`) and 1998–2025 (`attn_K030_universe-wrds_broad_entry_rule-history_ridge-0.0001_fill-none`): 90 models in 2.6 minutes. Against the 30-day wait with the same setting and arrays (seed averages; pooled = the paired block bootstrap of `analysis\gap\paired_test.py` on the seed-mean net series):

| Period | SR, history / wait | SR_net, history / wait | Pooled SR_net difference [95%] | Turnover | Beta |
| --- | --- | --- | --- | --- | --- |
| 1996–1997 (validation) | 12.69 / 11.73 | 8.00 / 7.52 | +0.53 [−0.20, +1.18] | 0.83 / 0.75 | 0.03 / 0.04 |
| 1998–2021 | 3.68 / 3.72 | 1.23 / 1.18 | +0.05 [−0.10, +0.22] | 0.68 / 0.67 | 0.07 / 0.05 |
| 1998–2005 | 6.94 / 6.55 | 3.64 / 3.42 | +0.25 [−0.00, +0.51] | 0.87 / 0.81 | 0.01 / 0.01 |
| 2006–2013 | 3.05 / 3.15 | 0.02 / 0.06 | −0.04 [−0.32, +0.27] | 0.74 / 0.74 | 0.04 / 0.03 |
| 2014–2021 | 1.73 / 1.82 | 0.27 / 0.18 | +0.10 [−0.14, +0.34] | 0.45 / 0.46 | 0.18 / 0.15 |
| 2022–2025 | 0.89 / 0.88 | −0.33 / −0.40 | +0.06 [−0.46, +0.59] | 0.35 / 0.30 | 0.13 / 0.10 |

- The rule moves net return by year in both directions (from −2.8 to +4.7 points, 2008 and 2000) and adds 0.4 points a year net over 1998–2021, with 0.2 points more volatility. The two rules' daily gross returns correlate at 0.92.
- Not separated yet: what comes from the rebuilt lags (over the month's list) and what from the signals the rule adds (member stock-days with a signal: 96.2% → 99.8%). A block run gated by the old rule (`run` > 30) would split them; it needs a small option that is not built.

## 8. Buffered list (`wrds_broad_buffer600`, 2026-11-14)

**Rule** (the user's proposal; `afsa.wrds_universe.buffered`). Holding month m takes the top 500 by market cap at the end of m−1, plus every security of m−1's buffered list that ranks within 600. Eligibility and ranking are those of the plain lists (`ranked`; the refactored `build` reproduces `broad.parquet` and `common.parquet` exactly). The rule starts from the plain list of 1989-11, so every buffered security entered through a plain list, and the daily download (PERMNOs ever on a plain list) holds its data.

**Why the list grows.** With 500 fixed, every top-500 stock cannot enter while incumbents ranked 501–600 stay. The model takes any number of stocks a day (the softmax runs over the day's stocks, ranks scale to [−0.5, 0.5]), so the list holds 500 to 600 and the arrays get 600 slots (`config.n_slots`), the empty ones last (PERMNO −1).

**Numbers** (broad, holding months 1989-12 to 2025-12):
- List size 500 in 1989-11, 512 in 1990-01, 536 by 1990-06, then 524–564 (mean 545 over 1992–2025). 19,144 stock-months are held through the band.
- Securities entering a month: 6.6 instead of 14.4; leaving: 6.4 instead of 14.4.
- Under the plain rule, securities leave a list 5,608 times in 1990–2021, and 3,343 entries are securities coming back within 12 months of leaving.
- Other bands (1990–2021 simulation): 550 → mean size 523, 8.6 entries a month; 700 → 581, 5.6.

**Turnover attributable to list changes** (measured 2026-11-14 on the saved daily weights of the K = 30 baseline, `broad`, seeds 0–2, 1998–2021):
- Daily turnover 0.669: 0.892 on a month's first day, 0.658 on other days.
- On a month's first day, closing the 13.7 leaving names costs 0.044, opening the 14.9 entering names 0.064, and the stocks in both trade 0.784 (0.126 above an ordinary day, mostly the 35 monthly characteristics updating).
- Spread over all days, exits and entries are 0.71% of turnover; everything extra on month starts is 1.53%. Halving list churn saves about 0.35% of turnover, about 0.03% a year of return.

**Built** (all checks pass):
- `Data\WRDS\universe\broad_buffer600.parquet`: 236,144 stock-months, the columns of the plain lists plus `buffered` (on the list only through the band).
- `Data\TrainingData\wrds_broad_buffer600_fill-none\` (`afsa.build_wrds`, 904 MB): 4,949,552 stock-days. Under the current 30-day rule, 1.70% of member stock-days from 1990-01-02 get no signal (plain list: 3.76%).
- `history\wrds_broad_buffer600_fill-none\` (`afsa.build_history`, 2.1 GB): 13 checks. The own rows equal the day arrays on every stock-day. Lookback rows of 12 blocks equal a pandas recomputation from the whole table. Under the decided rule, 99.80% of member stock-days from 1990-01-02 get a signal (current rule on the same list: 98.30%). Lookback ranks move by 0.0029 on average against each day's own universe (plain list: 0.0045).
- Smoke run through `afsa.run.run_config`: K = 30, 2 epochs, test year 1998, 0.8 GB of GPU memory at the peak, scored with the buffered day arrays' market row.

**Caveats.** It departs from the paper's "500 largest": a robustness variant, judged on 1996–1997 (CLAUDE.md §8), never a Table 2 row. The flat 5 bp cost does not charge more for the smaller stocks ranked 501–600. Its runs' market row and beta are against the buffered list's equal-weighted return. `tests\test_buffered.py` covers the rule, the 600-slot builders against loop references, the configuration and the files on disk.

**First run** (2026-11-14; `analysis\gap\buffer.py` → `analysis\gap\BUFFER.md`). K = 30, ridge 1e-4, training at 5 bp, 30 epochs, rebuilt-history entry, seeds 0–2, 1996–2025: 90 models in 3 minutes. It is compared with the plain list's run at the same setting (SCAN_HISTORY.md). Figures are seed averages; *pooled* is the paired block bootstrap of the pooled net series, buffered minus plain:

| Period | SR, buffered / plain | SR_net, buffered / plain | Pooled SR_net difference [95%] | Turnover |
| --- | --- | --- | --- | --- |
| 1996–1997 (validation) | 13.21 / 12.69 | 8.42 / 8.00 | +0.45 [−0.11, +1.02] | 0.84 / 0.83 |
| 1998–2021 | 3.70 / 3.68 | 1.31 / 1.23 | +0.08 [−0.03, +0.21] | 0.66 / 0.68 |
| 1998–2005 | | 3.56 / 3.64 | −0.09 [−0.33, +0.14] | |
| 2006–2013 | | 0.26 / 0.02 | +0.25 [+0.03, +0.47] | |
| 2014–2021 | | 0.36 / 0.27 | +0.09 [−0.06, +0.25] | |
| 2022–2025 | 0.76 / 0.89 | −0.34 / −0.33 | −0.02 [−0.25, +0.23] | 0.30 / 0.35 |

- **Return and risk do not change:** μ 14.65% against 14.68%, σ 3.96% against 3.99%. Beta is 0.07 against either list's equal-weighted market (0.18 for both in 2014–2021). The pooled gross SR difference is +0.02 [−0.11, +0.15]. The paper has SR 3.97, SR_net 2.28 and an implied turnover of 0.47.
- **Turnover falls by 0.024 a day** (0.685 → 0.661), for every seed (0.656–0.665 against 0.678–0.691). Cost falls by 0.13 bp a day, which is 0.28 points a year more net return (μ_net 5.17% against 4.89%).
    - List changes explain 0.002 of the drop. Entries fall from 14.9 to 7.0 a month, and closing and opening positions from 0.55% to 0.28% of turnover. On a month's first day only these list trades fall (0.076 → 0.035); stocks on both lists trade as much as before (0.81).
    - The rest is less trading on other days (0.675 → 0.651), in stocks that stay members.
- **The securities held through the band** (44 a day) carry 8.8% of the gross exposure and 9.2% of the turnover, and net to about zero (+0.02). They earn 1.08 of the 14.65 points of gross return: 12.3% per unit of exposure, against 14.9% for the top 500.
- **The gain rests on the flat cost.** At 10 bp per unit of turnover on those securities' trades, the net SR is 1.12, below the plain list's 1.23. The two are equal at 7.1 bp.
- **Layers:** both lists' validation years choose a band of 4e-4. Ensemble 1.70 against 1.62; ensemble + band 2.08 against 2.01 (pooled +0.08 [−0.04, +0.20]). In 2022–2025, ensemble + band 0.08 against 0.09.
- **Reading:**
    - The buffer costs nothing in return or risk and trades a little less. Every gain is inside the noise: of the six net differences above, only 2006–2013 excludes zero.
    - It does not narrow the gap to the paper, and the gain disappears if the smaller securities cost 2 bp more to trade.
    - Keep it as a robustness row, not a change of universe.

## 9. The four `broad` datasets from WRDS (2026-11-15)

**Design** (the user's request; CLAUDE.md §7 row of 2026-11-15; `analysis\gap\datasets.py` → `analysis\gap\DATASETS.md`). A 2×2: the plain list (`wrds_broad`) or the buffered one (`wrds_broad_buffer600`), crossed with the 30-day wait or rebuilt history. One setting on all four: K = 30, ridge 1e-4, training at 5 bp, 30 epochs, seeds 0–2, validation 1996–1997, test 1998–2021, extension 2022–2025. Only C (buffered list, 30-day wait) was new: 90 models in about 2 minutes, the first run on the buffered day arrays. Its weights replay the stored results exactly. A comparison, not a selection: Table 2 stays on B.

**Net SR** (the model is the seed average; the layer is the ensemble through the band its own validation years choose, 4e-4 for all four):

| Dataset | Validation | 1998–2021 | 2022–2025 | 1998–2025 | Layer, 1998–2021 | Layer, 2022–2025 |
| --- | --- | --- | --- | --- | --- | --- |
| A: plain list, 30-day wait | 7.52 | 1.18 | −0.39 | 0.96 | 1.93 | −0.02 |
| B: plain list, rebuilt history | 8.00 | 1.23 | −0.33 | 0.99 | 2.01 | +0.09 |
| C: buffered list, 30-day wait | 8.01 | 1.22 | −0.42 | 0.97 | 1.99 | −0.02 |
| D: buffered list, rebuilt history | 8.42 | 1.31 | −0.34 | 1.06 | 2.08 | +0.08 |

The ensemble alone: 1.56 / 1.62 / 1.61 / 1.70 on 1998–2021. Gross SR 3.68–3.72 for all four; turnover 0.66–0.68.

**Contrasts** (model, net, pooled paired bootstrap; 1998–2021 unless stated):
- Rebuilt history: +0.05 [−0.10, +0.22] on the plain list, +0.09 [−0.03, +0.21] on the buffered one.
- Buffered list: +0.05 [−0.10, +0.19] under the 30-day wait, +0.08 [−0.03, +0.21] under rebuilt history.
- Interaction (D − C) − (B − A): +0.04 [−0.08, +0.16]. Both changes (D − A): +0.13 [−0.05, +0.34]; 1998–2025 +0.10 [−0.07, +0.28]; validation +0.99 [+0.14, +1.75].
- Gross, the four are equal on 1998–2021 (every contrast within ±0.04). In 1998–2005 rebuilt history gains +0.43 [+0.15, +0.73] gross on the plain list, and the two changes partly substitute (gross interaction −0.25 [−0.48, −0.03]).
- The layers show the same pattern: on 1998–2021 the ensemble + band contrasts are within 0.03 of the model's.

**Where it comes from** (the model, 1998–2021, from the saved weights). *Waiting* stock-days are members that rebuilt history gives a signal and the 30-day wait does not:
- They are 3.78% of member stock-days on the plain list and 1.70% on the buffered one. The buffer halves them because 7.0 securities enter a month instead of 14.9.
- Under the 30-day wait the model still holds them through the hedge: 7.45% of gross exposure on the plain list (3.98% buffered), losing 0.77 (0.64) points of gross return a year. Under rebuilt history they carry 4.57% (2.12%) and earn +1.00 (+0.65).
- The other stock-days earn less under rebuilt history: 14.85 → 13.68 points (plain), 14.99 → 14.00 (buffered). Their lags change too, since every lag day of the previous month is recomputed over the current list. So gross return rises only 0.6 (0.3) points, with σ up 0.2 (0.07).
- List changes are 0.74% of turnover under the wait on the plain list, 0.33% on the buffered one.

**Cost of the buffered list's extra securities** (44 a day, ranks 501–600, 8.8% of gross exposure, 9.2% of turnover). At 10 bp on their trades the net SR is 1.02 for C (A: 1.18) and 1.12 for D (B: 1.23). The break-even is 6.0 bp for C and 7.1 bp for D.

**Reading:**
- On the test years the four are within 0.13 net SR of each other, and no single contrast excludes zero; the same holds in 2022–2025. Validation and test rank them in the same order: D, then B and C about equal, then A.
- The two changes add up rather than overlap: the buffer halves the waiting stock-days, yet rebuilt history gains at least as much on the buffered list. They substitute only in 1998–2005, when entrants were most frequent.
- Rebuilt history's gain is real on the entrants (+1.8 points of gross return from the waiting stock-days on the plain list), but it is mostly offset elsewhere.
- None of them narrows the gap to the paper (2.28 net). The buffer's gain rests on the flat 5 bp, so the decision stands: Table 2 on B, the buffered list a robustness row.
