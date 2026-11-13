# The 22 accounting characteristics

Written 2026-11-13; revised the same day after four independent audits (the zero-fills of section 5 and the filing lag of foreign issuers in section 2). Code: `afsa\chars_accounting.py`; tests: `tests\test_chars_accounting.py`. "The panel" is the Chen-Pelger-Zhu characteristic file (`Data\WRDS\pelger\characteristics.parquet`, complete cases only, to 2016-12). "Equal" is the match rate of `afsa.pelger_check`: the share of common stock-months with |ours − theirs| ≤ 1e-6 + 1e-4 |theirs|. "The universe" is the broad universe, 216,000 stock-months from 1990 to 2025.

```powershell
.venv\Scripts\python.exe -m afsa.chars_accounting build              # Data\WRDS\characteristics\{link,accounting}.parquet (3 s)
.venv\Scripts\python.exe -m afsa.chars_accounting check --detail     # sections 4, 5 and 6 below (25 s)
.venv\Scripts\python.exe -m afsa.chars_accounting build --all-fills  # every zero-fill on, the panel's convention -> accounting_allfills.parquet
.venv\Scripts\python.exe -m afsa.chars_accounting build --strict     # no zero-fill at all -> accounting_strict.parquet
```

Sections 4 to 6 are the output of `check`. The formula variants of section 3, the "panel rows" counts of section 5 and the counts of the revision (stock-months moved by the foreign lag, the known limit, the empty fiscal-year rows) were measured with one-off scripts that are not in the repository; the counts are given so they can be redone. Not recounted after the revision: the sizes in the special-cases table of section 2 other than the rows marked "recounted". The foreign lag moved the fiscal year of 257 universe stock-months, so those sizes can be off by a few.

## 1. Conclusions

1. **All 22 formulas reproduce the panel**: equal is 99.3% to 99.95% of about 246,700 common stock-months under the panel's fiscal-year rule and with every zero-fill on, as the panel was built (target 98%). The lowest are the four ratios to market cap (99.3–99.5%). The revision did not change these rates.
2. **What is left is data vintage, not formulas.** Up to 2012 every characteristic is equal in at least 99.86% of rows; in 2013–2016, the last years before the panel was built, 95.9–99.8%. The ratios to market cap differ almost only for ADRs from 2007 on (88.9% equal against 99.94% for US common stocks), where CRSP has since revised the share counts. No older vintage is on disk, so this is inferred from the pattern, not proven.
3. **The file on disk departs from the panel in two deliberate ways**, and matches it in 98.1–98.8% of rows.
    - *Fiscal rule `safe`:* no accounts younger than 4 months, 6 months for ADRs and non-US issuers. It changes 0.82% of universe stock-months relative to the panel's rule.
    - *Production zero-fills:* a missing input counts as 0 only where the panel confirms it on many rows. XSGA, DP, COGS, CEQ and LCT are not filled: the characteristic is NaN, and gets rank 0 under the missing-value policy.
4. **What the production fills change in the universe.**
    - FC2Y, OP and OL are NaN in 22.6% of stock-months (0.2–0.3% with every fill on), CF in 5.7% and CF2P in 7.1% (0.2% and 1.7%), Q, NOA, RNA and OA in 0.1–0.3 points more. The other 13 characteristics do not change.
    - FC2Y is exactly 0 in 13 stock-months (0.006%) instead of 37,709 (17.46%); the largest group of stocks with the same FC2Y in one month is 2 instead of 121.
    - No value changes: a cell is either the same or NaN. No panel match rate falls.
5. **The panel cannot verify** banks (0.4% of their universe stock-months are in it), insurers (9%), REITs (9%), anything after 2016 (25% of the universe stock-months of 1990–2025), and what a characteristic is when XSGA, DP or COGS is missing (section 5).

## 2. What is built

**Rows.** One per security (PERMNO) and holding month: 600,361 rows, 2,524 PERMNOs, 1986-01 to 2025-12. A security has a row for every month from the month after its first daily CRSP row to the month after its last. All 217,000 stock-months of the broad universe (1989-11 on) are there.

**Columns of `accounting.parquet`:** `permno` (int32), `month` (Period[M], the holding month), `gvkey`, `fyear` and `datadate` of the fiscal year used, `curcd` (its currency), `cap_year` (the December whose market cap the ratios use), and the 22 characteristics (float64, NaN where not computable). `link.parquet` has `permno`, `month`, `formation_date`, `gvkey`, `linkprim`, `linktype` and `foreign_filer` (new) for the same rows.

**Link.** The CCM link valid on the formation date (the last trading day of the previous month): link types LC, LU, LS; any `linkprim`, in the order P > C > J > N, then LC > LU > LS, then the latest link date.
- 114 of the 216,000 universe stock-months have no GVKEY (0.053%). The GVKEY and `linkprim` equal Ander's on all 216,000.
- No security-month has valid links to two GVKEYs.
- Secondary classes (J and N) match the panel as well as primary ones: AT is equal in 100% of their 1,995 panel rows. Keeping only P and C would leave 0.69% of universe stock-months without accounts instead of 0.05%.

**Fiscal year** (`fiscal_rule`). Y is the year such that the holding month lies in July of Y to June of Y+1.

| Rule | Fiscal year used | Market cap in the ratios | AT equal to the panel |
| --- | --- | --- | --- |
| `pelger` | Compustat `fyear` = Y−1 (year-ends June of Y−1 to May of Y) | December of Y−1 | 99.93% |
| `ff` | The fiscal year ending in calendar year Y−1 (the written rule) | December of Y−1 | 85.61% |
| `safe` (the file) | `pelger`; while year-end + lag is after the first day of the month, the previous fiscal year. The lag is 4 months, 6 for a foreign filer | December of Y−1; December of Y−2 in the gated months | 98.70% |

- Under `pelger`, 1,505 universe stock-months (0.70%) use accounts less than 4 months old at the formation date: year-ends in March (366, used from July, 3 months old), April (260, 2 months) and May (879, 1 month). This is the panel's look-ahead.
- **US filers** (185,303 universe stock-months) wait 4 months: the 10-K was due in 90 days, later 60 to 75. A March year-end enters in August, April in September, May in October. 1,375 stock-months use the previous fiscal year for this reason. Their youngest accounts are 120 days old at the formation date.
- **Foreign filers** (30,697 universe stock-months, 14.2%) wait 6 months, in every year: a 20-F was due 6 months after the year-end for fiscal years ending before 2011-12-15, and 4 months since. A January year-end enters in August, February in September, …, May in December. 383 stock-months use the previous fiscal year. Their youngest accounts are 179 days old at the formation date (120 before the revision).
    - A foreign filer is an ADR (`sharetype` = AD) or a non-US issuer (`usincflg` = N) in the CRSP monthly file at the formation month. The flag is dated: 51 securities change it. It equals the universe file's own classification on every universe stock-month.
    - 212 of the 600,361 rows (none in the universe) have no monthly row for the formation month and take the security's latest earlier row. The fallback to a security's earliest row, for a grid that starts before the monthly file, is not needed here.
- **What the 6 months changed.** The fiscal year of 257 universe stock-months (0.12%; 29 securities; 134 ADR, 123 other non-US issuers; year-ends in March 176, April 37, January 22, February 16, May 6). Before, all 257 used accounts younger than 6 months, 134 of them for fiscal years that ended before 2011-12-15, when the deadline was 6 months. 3 of the 257 have no earlier fiscal year and are now NaN. Among the panel's rows 412 moved, and AT's match under `safe` fell by exactly those 412 rows (98.87% to 98.70%).
- `safe` now differs from `pelger` in 1,762 universe stock-months (0.82%): the 1,505 above and the 257.
- Accounts are 120 to 701 days old at the formation date (median 365); the holding month is 5 to 24 months after the year-end month (median 13).
- `ff` picks another fiscal year than `pelger` in 9.4% of universe stock-months (every year-end from January to May).

**Limits of `safe`.**
- *The lag runs to the first day of the holding month*, which is 1 to 4 calendar days after the formation date. For a US filer the 4 months leave a month of margin over the 90 days. For a foreign filer 6 months is the old deadline itself. The 20-F of a December year-end was due on the last day of June, the formation date of July: 1,037 universe stock-months of 166 securities, for fiscal years ending before 2011-12-15 (313 ADR, 724 other non-US issuers). In 246 of them June does not end on a trading day, so the formation date is 1 to 3 days before the deadline. January to May year-ends are in the same position in the month they enter: 74 stock-months of 16 securities, 24 with the formation date before the deadline. Together 1,111 stock-months, 0.51% of the universe. They are not gated by a seventh month: this is a known limit. From fiscal years ending 2011-12-15 on, the deadline is 4 months and the 1,034 stock-months that enter after 6 have two months of margin.
    - An upper bound from the look-ahead audit, not recounted here: Compustat's own final date (`fdate`, filled from about 2005) is after the formation date for 39 of 1,361 ADR and foreign December year-ends first used in July.
- *The fallback needs the newer row.* A month falls back to the previous fiscal year only when the row of the newer fiscal year exists in the Compustat file and is too young. A firm with no row for the fiscal year asked for gets NaN, never an older year: nothing is carried forward. No value of a later row is used (the audit scrambled them: 0 cells changed), but whether the newer row is in the download does matter. So a truncation test that cuts Compustat at year-end + lag turns the gated months into NaN and is not bit-identical; that is this rule, not a leak. Cut at the formation date (Compustat by year-end, CRSP daily and monthly, CCM links by start date), the file is bit-identical: every row up to the cut, at 12 dates from 1990-06 to 2025-11, after the revision.
- *Deadlines, not filing dates.* A late filer is not detected. CRSP's two flags stand in for the form a company files: a non-US issuer that files a 10-K waits 6 months for nothing, and Canadian issuers (40-F) are treated like the rest.
- *Restated accounts.* Compustat's values are today's, not those public at the time.

**Market cap.** A2ME, BEME, CF2P and E2P divide by the security's own CRSP cap (`dlycap`, in USD millions) on its last trading day of December. It equals the monthly file's December `mthcap` on all 50,954 security-years that are in both files. Q uses Compustat's CSHO × PRCC_F at the fiscal year-end. Using the security's cap is a deviation from the written definition (section 3).

**Special cases.**

| Case | How it is handled | Size |
| --- | --- | --- |
| Two rows for one (gvkey, fyear), after a change of year-end | The row with total assets is kept, then the later year-end | 2 pairs among the linked firms; in both, one row is empty |
| Rows without `fyear` | Dropped | 29 rows, none with total assets |
| The fiscal-year row exists but is empty | All 22 characteristics are NaN: the row is the fiscal year the rule asks for, and an older year is not used in its place | 151 universe stock-months (0.07%), 22 securities, 26 firm-years (recounted; below) |
| Restated accounts | The download holds one row per firm and year-end (`datafmt` = STD), so there is no restated row to choose. Values are today's, not those public at the time | Not measurable here |
| Change of year-end month between two fiscal years | The lag is still `fyear` − 1, as in the panel (equal in 100% of its 1,135 such rows, where the two year-ends are 6 to 21 months apart) | 689 universe stock-months (0.32%) |
| Previous row is not `fyear` − 1 | Lagged characteristics are NaN | 5 of 62,721 firm-years |
| Firm reports in CAD | Not converted, as in the panel (AT equal in 100% of 2,980 rows). AT is in CAD; A2ME, BEME, CF2P and E2P divide CAD by USD | 7,649 universe stock-months (3.5%), 65 securities (recounted) |
| Firm switches between CAD and USD | The six characteristics that use the previous year are NaN that year. **The panel instead mixes the two currencies** (equal in 100% of 134 rows) | 101 universe stock-months (0.05%) |
| Several share classes | Same accounts, each class its own December cap. The panel does the same: 99.3% equal on 7,923 rows, 0% with the company's total cap | 6,229 universe stock-months (2.9%) belong to a company with another share class; in 2,444 (1.1%) both classes are in the universe |
| ADRs | The whole company's accounts over the cap of the US receipts only, as in the panel | 7,600 universe stock-months (3.5%) (recounted) |
| Negative book equity | Kept, as in the panel (6,110 rows, 99.9% equal) | 2.3% of universe stock-months (recounted) |
| Zero denominator | NaN | No panel row has SALE or AT equal to 0 |

**The empty fiscal-year rows.** In 151 universe stock-months the security has a GVKEY and Compustat has a row for the fiscal year, but the row holds no accounts: no AT, SEQ, SALE, IB or CSHO (5 of the 26 firm-years have a year-end price, PRCC_F, and nothing else).
- All are rows in USD; 138 stock-months are US common stocks, 13 are ADRs or other foreign shares.
- 65 stock-months fall in 1990–1999 (14 security-years), 18 in 2000–2009 (3), none in 2010–2016 and 68 in 2017–2025 (9).
- A security-year lasts 1 to 12 months (median 4; 4 last the full 12).
- These are a third of the 451 universe stock-months (0.21%) without total assets; the others have no GVKEY (114) or no row for the fiscal year (186).
- Not checked: whether the accounts exist in WRDS under another format (for example `indfmt` = FS) or arrived after the download. That needs a WRDS query.

## 3. Formulas as implemented, and the evidence

τ is the fiscal year used, τ−1 the previous one. ME is the December cap of the security.
- A star (\*) marks an input counted as 0 when missing, in the panel and in the production file.
- A dagger (†) marks an input that the panel's convention counts as 0 but the production file does not: when it is missing the characteristic is NaN (section 5). `check` verifies the formulas with the daggers filled.

- BE = SH − PS\* + TXDITC\* (TXDITC only if `fyear` ≤ 1992). SH = SEQ, else CEQ + PSTK, else AT − LT. PS = PSTKRV, else PSTKL, else PSTK.
- NOAraw = (AT − CHE − IVAO\*) − (AT − DLC\* − DLTT\* − MIB\* − PSTK\* − CEQ†).
- WC = ACT − CHE − LCT† − DLC\* − TXP\* − DP†.

| | Formula | Differs from the written definition |
| --- | --- | --- |
| AT | AT | |
| A2ME | AT / ME | ME of the security, not of the firm |
| BEME | BE / ME | ME of the security, not of the firm. TXDITC only up to fiscal 1992 |
| C | CHE / AT | |
| CF | (NI + DP† − WCAPCH\* − CAPX\*) / BE | |
| CF2P | (IB + DP† + TXDB\*) / ME | ME of the security, not of the firm |
| Q | (AT + CSHO × PRCC_F − CEQ† − TXDB\*) / AT | Compustat's market value at the fiscal year-end, not CRSP's |
| Lev | (DLTT\* + DLC\*) / (DLTT\* + DLC\* + SEQ) | |
| E2P | IB / ME | ME of the security, not of the firm |
| PROF | (REVT − COGS) / BE | |
| CTO | SALE / AT(τ−1) | |
| FC2Y | (XSGA† + XRD\* + XAD\*) / SALE | |
| OP | (REVT − COGS† − TIE\* − XSGA†) / BE | TIE exists only for financial firms, so no interest is subtracted elsewhere |
| PM | OIADP / SALE | |
| RNA | OIADP / NOAraw(τ−1) | |
| D2A | DP / AT | |
| Investment | AT / AT(τ−1) − 1 | |
| NOA | NOAraw / AT(τ−1) | |
| DPI2A | (ΔPPEGT + ΔINVT\*) / AT(τ−1), each INVT level zero-filled | |
| OA | (WC − WC(τ−1)) / AT(τ−1) | The change is taken of the whole bracket, DP included; DLC and TXP are subtracted |
| OL | (COGS + XSGA†) / AT | |
| PCM | (SALE − COGS) / SALE | |

**Deviation: the market equity of the security.** Freyberger, Neuhierl and Weber write that market equity is "the total market capitalization at the firm level". The panel divides by the cap of the security: on its 7,923 rows of companies with several share classes, 99.3% are equal with the security's cap and 0% with the company's. We follow the panel, so this is a panel-driven deviation from the written definition, and it has an economic effect on three groups of universe stock-months:
- 2.9% (6,229) belong to a company with another share class: the ratio is too high by company cap / class cap.
- 3.5% (7,600) are ADRs: the whole company's accounts over the cap of the US receipts.
- 3.5% (7,649) report in CAD: accounts in CAD over a cap in USD.

Together 9.6% of universe stock-months. A reader of Table 3 (the value group) should know it.

**Where the panel decides** (equal on all common rows unless a subset is named):

| Choice | Implemented | Alternatives |
| --- | --- | --- |
| Fiscal year | `fyear` = Y−1: 99.93% (AT) | Year-end in calendar Y−1: 85.6% |
| Market cap | The security's own December cap: 99.46% (A2ME) | Company total, on 7,923 multi-class rows: 0% against 99.3% |
| TXDITC in BE | `fyear` ≤ 1992: 99.46%; 100% on the 1,089 rows of fiscal 1992 ending in 1993, of which 777 have a TXDITC other than 0 | Year-end ≤ 1992: 0% on those 777 rows. Always: 46.1%. Never: 82.1% |
| Preferred stock | PSTKRV, PSTKL, PSTK: 100% on the 9,906 rows where the order matters | PSTKL first: 15.6% on them. PSTK only: 6.1% on the 23,812 rows where it differs |
| Q, market value | CSHO × PRCC_F: 99.92% | CRSP December cap: 18.0%. CSHO × PRCC_C: 67.8% |
| Q, deferred taxes | TXDB: 99.92% | TXDITC: 98.9%. None: 29.1% |
| CF | NI + DP − WCAPCH − CAPX: 99.81%; 100% on the 6,035 rows with a WCAPCH | Without WCAPCH: 0% on those rows. IB for NI: 74.3%. OANCF − CAPX: 0.03% |
| CF2P | TXDB: 99.30% | TXDI: 4.6%. TXDITC: 98.3% |
| OP | No XINT: 99.63% | Subtracting XINT: 7.3% |
| NOA, RNA | With IVAO and MIB: 99.75%, 99.60% | Without IVAO: 52.9%, 53.4%. MIBT for MIB: 88.9%. Over current AT or NOA: under 0.1% |
| DPI2A | PPEGT, INVT levels zero-filled: 99.82%; 100% on the 615 rows with one INVT missing | Zero-filling the change: 29.6% on those rows. PPENT: 0.04% |
| OA | Change of the bracket with DP inside: 99.73% | Level of DP: 0.05%. Sloan's accrual (DLC and TXP added back): 0.03% |
| CTO | Lagged AT: 99.89% | Current AT: 0.06% |

**Where the panel cannot decide.**
- REVT against SALE (PROF, OP, PCM, PM, CTO, FC2Y): the two give the same match rate on the panel's rows. We follow the written definitions.
- PROF's numerator: the written definition says gross profit (GP). Compustat's GP and REVT − COGS give the same match rate (99.64%); the code uses REVT − COGS.
- The SH fallbacks: SEQ is never missing when AT exists, in the panel or in the universe, so they are never used.
- OP's interest term: TIE is missing in every panel row, and OP equals the panel in 99.63% of them with no interest subtracted. So the panel confirms the fill and shows that XINT is not subtracted; it cannot show that TIE is subtracted where it exists. TIE exists in 5.8% of universe stock-months: 28.7% of the financial firms' and 0.04% of the others'.
- Q when CSHO or PRCC_F is missing: the panel has a value in 92 such rows (9 securities; PRCC_F is missing in all 92) that no alternative tried reproduces (CRSP's cap in December or at the year-end: 0 rows; CSHO × CRSP's year-end price: 10 of 55). We leave Q missing: 2,508 universe stock-months (1.16%), 2,365 of them fiscal years that ended before the security's first day in CRSP.

## 4. Match with the panel

About 246,700 common stock-months, 1986-01 to 2016-12. The first six columns verify the formulas: every zero-fill on, as in the panel; `within 1%`, `Spearman` and n are under `pelger`. The last two are the file on disk: `safe` and the production fills.

| | pelger | ff | safe | n | within 1% | Spearman | The file | n (file) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT | 0.9993 | 0.8561 | 0.9870 | 246,767 | 0.9993 | 0.9999 | 0.9870 | 246,767 |
| A2ME | 0.9946 | 0.8520 | 0.9825 | 246,734 | 0.9948 | 0.9994 | 0.9825 | 246,730 |
| BEME | 0.9946 | 0.8521 | 0.9825 | 246,734 | 0.9948 | 0.9993 | 0.9825 | 246,730 |
| C | 0.9995 | 0.8570 | 0.9872 | 246,756 | 0.9995 | 0.9998 | 0.9872 | 246,755 |
| CF | 0.9981 | 0.8551 | 0.9858 | 246,767 | 0.9984 | 0.9994 | 0.9859 | 246,745 |
| CF2P | 0.9930 | 0.8507 | 0.9810 | 246,745 | 0.9936 | 0.9988 | 0.9811 | 246,708 |
| Q | 0.9992 | 0.8560 | 0.9869 | 246,675 | 0.9996 | 0.9999 | 0.9870 | 246,459 |
| Lev | 0.9988 | 0.8722 | 0.9880 | 246,767 | 0.9989 | 0.9999 | 0.9880 | 246,767 |
| E2P | 0.9944 | 0.8520 | 0.9824 | 246,745 | 0.9948 | 0.9992 | 0.9824 | 246,741 |
| PROF | 0.9964 | 0.8534 | 0.9841 | 246,767 | 0.9972 | 0.9995 | 0.9841 | 246,767 |
| CTO | 0.9989 | 0.8556 | 0.9866 | 246,589 | 0.9991 | 0.9993 | 0.9866 | 246,586 |
| FC2Y | 0.9973 | 0.8544 | 0.9850 | 246,778 | 0.9978 | 0.9999 | 0.9851 | 246,759 |
| OP | 0.9963 | 0.8533 | 0.9840 | 246,767 | 0.9970 | 0.9984 | 0.9841 | 246,748 |
| PM | 0.9973 | 0.8542 | 0.9851 | 246,778 | 0.9978 | 0.9995 | 0.9851 | 246,778 |
| RNA | 0.9960 | 0.8529 | 0.9837 | 246,549 | 0.9967 | 0.9992 | 0.9838 | 246,364 |
| D2A | 0.9983 | 0.8553 | 0.9860 | 246,748 | 0.9984 | 0.9965 | 0.9860 | 246,745 |
| Investment | 0.9991 | 0.8558 | 0.9868 | 246,589 | 0.9991 | 0.9991 | 0.9868 | 246,586 |
| NOA | 0.9975 | 0.8543 | 0.9852 | 246,589 | 0.9978 | 0.9996 | 0.9853 | 246,369 |
| DPI2A | 0.9982 | 0.8549 | 0.9859 | 246,575 | 0.9982 | 0.9990 | 0.9859 | 246,572 |
| OA | 0.9973 | 0.8545 | 0.9850 | 246,529 | 0.9979 | 0.9990 | 0.9851 | 246,217 |
| OL | 0.9965 | 0.8536 | 0.9842 | 246,767 | 0.9977 | 0.9995 | 0.9843 | 246,748 |
| PCM | 0.9963 | 0.8536 | 0.9840 | 246,778 | 0.9971 | 0.9992 | 0.9840 | 246,778 |

- The `pelger` and `ff` columns, n, within 1% and Spearman are the same as before the revision for all 22. The `safe` column is 0.14 to 0.17 points lower, because 412 panel rows of foreign filers now wait two more months (section 2).
- The production fills cost no match (section 5): under `pelger` they remove 19 to 301 panel rows of six characteristics and change no value.

Of the 246,819 panel rows on our grid, 52 have no total assets on our side: 28 without GVKEY, 13 without a row for the fiscal year, 11 with a row but no AT.

**Where the remaining differences are** (`pelger`, every zero-fill on, equal):

| | AT | A2ME | Q | CF | OP | RNA | OA | PCM | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1986–1994 | 0.9995 | 0.9993 | 1.0000 | 0.9995 | 0.9995 | 0.9993 | 0.9993 | 0.9995 | 55,191 |
| 1995–2004 | 0.9995 | 0.9995 | 0.9999 | 0.9995 | 0.9995 | 0.9991 | 0.9990 | 0.9994 | 79,366 |
| 2005–2012 | 0.9997 | 0.9987 | 0.9998 | 0.9988 | 0.9977 | 0.9975 | 0.9982 | 0.9973 | 76,147 |
| 2013–2016 | 0.9979 | 0.9679 | 0.9955 | 0.9912 | 0.9817 | 0.9812 | 0.9887 | 0.9826 | 36,063 |
| US common | 0.9995 | 0.9994 | 0.9994 | 0.9980 | 0.9967 | 0.9966 | 0.9973 | 0.9964 | 219,119 |
| ADR | 1.0000 | 0.8886 | 1.0000 | 0.9994 | 0.9895 | 0.9866 | 0.9982 | 0.9901 | 10,284 |
| Foreign | 0.9965 | 0.9965 | 0.9965 | 0.9990 | 0.9965 | 0.9945 | 0.9973 | 0.9990 | 14,772 |
| MLP | 0.9946 | 0.9946 | 0.9946 | 0.9946 | 0.9946 | 0.9917 | 0.9917 | 0.9946 | 2,053 |
| REIT | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 539 |
| Reports in CAD | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2,980 |
| Bank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 175 |
| Insurer | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1,931 |
| Other financial | 1.0000 | 0.9802 | 1.0000 | 0.9986 | 0.9958 | 0.9887 | 0.9972 | 0.9972 | 4,254 |
| Non-financial | 0.9993 | 0.9948 | 0.9992 | 0.9981 | 0.9963 | 0.9961 | 0.9973 | 0.9963 | 240,407 |

- **Total assets:** the 163 rows that differ are 19 firm-years of 14 securities. In none of them does the panel's value equal our AT of fiscal year τ±1 or τ±2, so they are changed values, not another year.
- **Market cap:** 1,173 rows have the panel's AT and another A2ME. 1,146 of them are ADRs of 2007–2016, constant within each security-year: another share count.
- **Cross-check after 2016.** Ander's independent `characteristics.parquet` equals ours (`pelger`, every zero-fill on) in 99.6% to 100% of stock-months for all 22, both in 1990–2016 and in 2017–2025. This checks the code, not the definitions. It was measured before the revision and not redone.

## 5. Zero-filled inputs

**The rule.** The main items of a characteristic must exist. A secondary item can count as 0 when it is missing, which is what the panel does wherever it can be seen. The production file keeps such a fill only where the panel confirms it on many rows, and turns it off for five inputs (`PRODUCTION_STRICT` in the code):

| Input off | Used in | Why |
| --- | --- | --- |
| XSGA | FC2Y, OP, OL | The panel has no row with XSGA missing, so it cannot show a fill. And the fill has a consequence the panel does not have: FC2Y became exactly 0 where XSGA, XRD and XAD are all missing, while the panel's own SGA2S is 0 in 9 of its 908,040 rows from 1986 |
| DP | CF, CF2P, OA | For CF and CF2P the panel cannot tell: it has 19 and 30 rows without DP, and the fill reproduces none (the panel had a DP then). D2A is NaN for the same firm. For OA the fill matches 90.6% of 127 rows |
| COGS | OP | The panel has no row with COGS missing |
| CEQ | Q, NOA, RNA | The panel does fill it (92–97% of 181–216 rows), but these are 4 securities, and a missing common equity is not 0: the fill raises Q |
| LCT | OA | The panel does fill it (100% of 174 rows), but these are 3 securities, and missing current liabilities are not 0 |

Still on: WCAPCH, CAPX, XRD, XAD, TIE, MIB, IVAO, TXDB, TXP, INVT, TXDITC (to fiscal 1992), DLTT, DLC, PSTK and PS. The table at the end of this section gives the evidence for each. The first eleven rest on 1,754 to 246,767 panel rows. DLTT, DLC, PSTK and PS rest on far fewer (12 to 382 rows; 100%, 100%, 92% and 95% equal): they are kept because nothing contradicts them and a missing debt or preferred-stock item is read as "none", but the panel's support for them is thin.

One helper, `_zero`, does every fill. `ZERO_FILLED` lists every fill the code can make and a test checks the list against the code, item by item, for all three conventions. `fiscal_years(funda)` uses the production list; `strict=False` turns every fill on (the panel's convention); `strict=True` turns every fill off; `strict=("xsga", "ceq")` only those. On the command line `build` writes `accounting.parquet`, `build --all-fills` writes `accounting_allfills.parquet`, `build --strict` writes `accounting_strict.parquet` and `build --strict ceq lct` writes `accounting_strict-ceq-lct.parquet`.

**Share of universe stock-months that are NaN, in %** (`safe`):

| | Every fill on | Production (the file) | Every fill off | Inputs filled in production | Inputs not filled |
| --- | --- | --- | --- | --- | --- |
| AT | 0.21 | 0.21 | 0.21 | | |
| A2ME | 1.72 | 1.72 | 1.72 | | |
| BEME | 1.72 | 1.72 | 3.17 | PS, TXDITC | |
| C | 0.23 | 0.23 | 0.23 | | |
| CF | 0.23 | **5.70** | 99.96 | WCAPCH, CAPX, PS, TXDITC | DP |
| CF2P | 1.73 | **7.14** | 16.88 | TXDB | DP |
| Q | 1.37 | **1.50** | 13.10 | TXDB | CEQ |
| Lev | 0.21 | 0.21 | 0.58 | DLTT, DLC | |
| E2P | 1.73 | 1.73 | 1.73 | | |
| PROF | 0.25 | 0.25 | 1.75 | PS, TXDITC | |
| CTO | 0.75 | 0.75 | 0.75 | | |
| FC2Y | 0.30 | **22.63** | 77.10 | XRD, XAD | XSGA |
| OP | 0.25 | **22.60** | 95.24 | TIE, PS, TXDITC | XSGA, COGS |
| PM | 0.32 | 0.32 | 0.32 | | |
| RNA | 0.79 | **0.94** | 23.12 | IVAO, DLC, DLTT, MIB, PSTK (of τ−1) | CEQ (of τ−1) |
| D2A | 5.70 | 5.70 | 5.70 | | |
| Investment | 0.73 | 0.73 | 0.73 | | |
| NOA | 0.75 | **0.88** | 22.85 | IVAO, DLC, DLTT, MIB, PSTK | CEQ |
| DPI2A | 11.04 | 11.04 | 12.86 | INVT | |
| OA | 20.55 | **20.83** | 24.41 | DLC, TXP | LCT, DP |
| OL | 0.23 | **22.58** | 22.58 | | XSGA |
| PCM | 0.30 | 0.30 | 0.30 | | |

- In the production file 5.9% of the universe's 4.75 million cells are NaN (2.3% with every fill on); 35.2% of stock-months have at least one NaN among the 22 (21.9%); a stock-month has 1.30 NaN on average (0.51).
- Where the production NaN are, in % of universe stock-months by sector:

  | | FC2Y, OP, OL | CF | CF2P | OA | DPI2A | D2A | Stock-months |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Banks | 19.3 | 24.8 | 26.0 | 99.4 | 76.4 | 24.8 | 15,319 |
  | Insurers | 75.8 | 22.4 | 23.2 | 81.5 | 30.4 | 22.4 | 12,405 |
  | Other financials | 48.5 (OL 48.3) | 31.0 | 32.6 | 73.6 | 40.6 | 31.0 | 15,858 |
  | Non-financial | 16.7 | 0.5 | 2.0 | 4.6 | 1.1 | 0.5 | 172,418 |

- By period: FC2Y, OP and OL are NaN in 28.5% of stock-months in 1990–1997, 24.7% in 1998–2009, 18.2% in 2010–2021 and 18.0% in 2022–2025; CF in 9.6%, 4.2%, 5.0% and 4.4%.
- With every fill on, the NaN come from: no GVKEY (114 stock-months), no row for the fiscal year (186), a row without AT (151), no December cap (3,265, of which 3,246 are securities first listed after that December), and main items that financial firms do not report.

**What the revision verified** (the file before it had every fill on and a 4-month lag for everyone):

| | Before | After |
| --- | --- | --- |
| FC2Y exactly 0, universe stock-months | 37,709 (17.46%) | 13 (0.006%): 2 securities whose XSGA is reported as 0 |
| XSGA, XRD and XAD all missing: 35,440 stock-months (16.4%) | FC2Y = 0 in 35,391 | FC2Y is NaN in all |
| Share of a month's stocks at FC2Y = 0 | mean 17.5%, at most 24.2% | mean 0.01%, at most 0.2% |
| Largest group of stocks with one FC2Y value in a month | 121 | 2 |
| Stock-months that share their FC2Y with another stock of the month | 18.36% | 0.90% |
| XSGA missing: 48,303 stock-months (22.4%) | OP and OL computed without it in 48,264 | NaN in all |
| DP missing: 11,860 stock-months (5.5%) | CF computed in 11,821, CF2P in 11,695, with D2A NaN | NaN in all |

- The remaining ties are of size 2 or 3 (two share classes with the same accounts), except true values: Lev is exactly 0 for 3.7% of stock-months (firms without debt; at most 52 in a month), and C, D2A and DPI2A have groups of 5 to 7. DLTT and DLC are never both missing in the universe, so no Lev of 0 is made of fills.
- Against every fill on, under the same fiscal rule, over all 600,361 rows: no value differs and no value appears. The cells that became NaN are in CF, CF2P, Q, FC2Y, OP, RNA, NOA, OA and OL only.
- **No panel match rate falls.** Under `pelger` the production fills remove these panel rows, and every other row is unchanged:

  | | Panel rows that became NaN | Of them equal before | Equal, every fill on | Equal, production fills |
  | --- | --- | --- | --- | --- |
  | FC2Y, OP, OL | 0 | – | 0.9973, 0.9963, 0.9965 | the same |
  | CF | 19 | 0 | 0.9981 | 0.9982 |
  | CF2P | 30 | 0 | 0.9930 | 0.9932 |
  | Q | 216 | 205 | 0.9992 | 0.9993 |
  | RNA | 181 | 175 | 0.9960 | 0.9961 |
  | NOA | 216 | 199 | 0.9975 | 0.9976 |
  | OA | 301 | 289 | 0.9973 | 0.9974 |

**Each input on its own.** The middle columns say what the panel shows: the rows where our input is missing and the panel has a value, and how many of them the zero-fill reproduces. Then the extra NaN, in % of universe stock-months, if that one fill alone is turned off, and whether the production file fills it.

| Input | Used in | Panel rows with the input missing | Equal with 0 | Extra NaN without the fill | Production |
| --- | --- | --- | --- | --- | --- |
| WCAPCH | CF | 240,720 | 99.8% | 99.73 | filled |
| TIE | OP | all 246,767 | 99.6%: OP equals the panel with no interest subtracted. No panel row has a TIE, so the subtraction itself is not verified | 93.94 | filled |
| XAD | FC2Y | 147,289 | 99.8% | 62.30 | filled |
| XRD | FC2Y | 86,390 | 99.7% | 48.63 | filled |
| XSGA | FC2Y, OP, OL | 0 | **cannot tell** | 22.34 | **not filled** |
| MIB | NOA, RNA | 41,632; 42,735 | 99.95% | 13.75; 14.06 | filled |
| TXDB | Q, CF2P | 20,498; 20,580 | 99.7%; 99.8% | 11.62; 11.60 | filled |
| IVAO | NOA, RNA | 30,665; 30,883 | 99.9%; 99.7% | 10.02; 9.97 | filled |
| DP | CF, CF2P | 19; 30 | **cannot tell** (0%: the panel had a DP then) | 5.47; 5.41 | **not filled** |
| DP | OA | 127 (lagged) | 90.6% | 0.18 | **not filled** |
| CAPX | CF | 1,754 | 98.9% | 4.39 | filled |
| TXP | OA | 3,649 | 100% | 3.61 | filled |
| INVT | DPI2A | 1,779 | 100% | 1.81 | filled |
| TXDITC (to 1992) | BEME, CF, PROF, OP | 1,930 | 100% | 1.38 | filled |
| DLTT | Lev, NOA, RNA | 12; 12; 382 | 100% | 0.25; 0.24; 0.28 | filled |
| DLC | Lev, NOA, RNA | 12 | 100% | 0.12; 0.12; 0.14 | filled |
| DLC | OA | 24 (missing in one of the two years; 1 security) | 100% | 0.00 | filled |
| PSTK | NOA, RNA | 216 (with CEQ) | 92.1% | 0.21; 0.22 | filled |
| CEQ | Q, NOA, RNA | 216; 216; 181 (4 securities) | 94.9%; 92.1%; 96.7% | 0.13; 0.14; 0.15 | **not filled** |
| PS (all three) | BE | 210 | 94.8% | 0.07 to 0.11 | filled |
| LCT | OA | 174 (3 securities, ACT present) | 100% | 0.10 | **not filled** |
| COGS | OP | 0 | **cannot tell** | 0.01 | **not filled** |

- "Cannot tell" has one cause: the panel keeps a stock-month only if all 46 of its characteristics exist. SGA2S needs XSGA, D2A needs DP, PCM needs COGS, so a row with one of these missing is never in the panel, whatever CF, OP or OL did with it.
- The "panel rows" counts depend on whether the input is missing in the year used, the year before or either, so a recount can differ by a few rows in ten thousand; the shares do not move. They were measured before the revision and are under `pelger`, which the revision did not touch. The XSGA, DP and COGS rows and all "extra NaN" columns were recounted.
- In the panel, PSTK is missing in NOA and RNA only on the 216 rows where CEQ is missing too (the same 4 securities). PSTK stays filled; those rows are NaN in production through CEQ.
- **Main items are never filled:** AT, SEQ (Lev), CHE, ACT, PPEGT, SALE, REVT, COGS (PROF, OL, PCM), OIADP, NI, IB, DP in D2A, CSHO, PRCC_F. The panel has no row where filling one of them with 0 reproduces its value (ACT: 20 rows, 0%; DP in D2A: 19, 0%; CHE: 11, 0%; PPEGT: 14, 0%).

## 6. What is not verified

- **After 2016.** The panel ends in 2016-12; 25% of the universe stock-months of 1990–2025 are later. Only the agreement with Ander's file covers them.
- **Financial firms.** The panel holds 56.6% of the universe stock-months of 1990–2016: 69% of the non-financial ones, 0.4% of the banks', 9% of the insurers', 13% of the other financials', 9% of the REITs'. Banks and insurers are 12.8% of the universe. Their OP (the TIE term) and their COGS-based ratios (Compustat puts interest expense in a bank's COGS) are unchecked.
- **The complete-case filter.** A stock-month with any of the 46 characteristics missing is not in the panel. So the panel never shows what a characteristic is when a sibling is missing (section 5), and it says nothing on young listings.
- **Which fills to keep is a decision, not a measurement.** The panel is silent on XSGA, DP (in CF and CF2P) and COGS; leaving them empty follows the missing-value policy, not the panel. Two things the decision costs:
    - In 12,862 universe stock-months (6.0%) XSGA is missing but XRD or XAD exists. FC2Y was (XRD + XAD) / SALE there and is now NaN. A narrower rule (NaN only when all three are missing, 16.4%) would keep them; it was not chosen.
    - Among non-financial firms 16.7% of stock-months have no XSGA. Compustat leaves XSGA empty when a firm does not report it separately from cost of goods sold, so "missing" there does not mean "zero", but it also is not a firm without overhead.
- **The remaining fills are the panel's, not economics.** WCAPCH is missing in 99.8% of universe stock-months, so CF is in effect (NI + DP − CAPX) / BE. A missing XRD or XAD counts as no spending.
- **Point-in-time accounts.** Compustat's values are today's. `safe` controls when a fiscal year enters, not later corrections to it.
- **Filing dates.** `safe` uses deadlines: 4 months for US filers, 6 for foreign filers. No filing date was consulted (the annual file has none before about 2005). The 1,111 stock-months of foreign filers that enter on the old deadline day are a known limit (section 2), and who is a foreign filer rests on two CRSP flags, not on the form filed.
- **The empty fiscal-year rows** (151 stock-months, section 2): whether WRDS has those accounts elsewhere was not checked.
- **First December.** A security listed after December of Y−1 has no cap for the four ratios until the July after its first December: 3,265 universe stock-months (1.5%). The panel cannot show what it did there.
- **The FAS 109 reading.** Dropping TXDITC after fiscal 1992 looks like Fama and French's later treatment of deferred taxes, but the panel cuts by Compustat `fyear`, not by the year-end's calendar year. I did not re-read French's note.
- **Downstream.** `Data\TrainingData\wrds_*_fill-none\` were rebuilt from this file on 2026-11-13 at 12:12 (`afsa.build_wrds`; `training_data.md`). Against the file before the revision, 26.2% of universe stock-months have at least one changed cell.
