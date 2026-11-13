# CLAUDE.md

Guide for Claude Code in `c:\Claude\AFSA`, loaded every session. Keep it short and true: update §7, §9 and §10 as work moves, replace stale text, and put long detail in `NOTES.md`. Keep the section numbers, because repo files cite §7 (the baseline) and §8 (no look-ahead).

## 1. Project

- **Course:** AMS 520 *Machine Learning in Quantitative Finance*, Stony Brook, Fall 2026 (Prof. Paweł Polak). Group project with Ander Ibarrondo, who built the data.
- **Paper:** Epstein, Wang, Choi and Pelger, *Attention Factors for Statistical Arbitrage* (AFSA), ICAIF '25.
- **Goal now:** settle the open specification (§10), then rerun the Attention Factors + LongConv rows of Table 2 (1998–2021) under it and under `--fill none`.
- **Later:** Table 3, Figures 2–4, PCA + LongConv, and the 2022–2025 extension. The extension is the professor's main ask; he has already replicated AFSA elsewhere.
- **Deliverables:** a Jupyter/Colab report; a GitHub repo with code and docs (no credentials); presentations at Phase II (Lecture 22, "first empirical results"; date to confirm) and Phase III (final-exam day).

## 2. Workspace

```
afsa\                the code (§3)
tests\               unit tests; tests\hardware\ = resource tests (REPORT.md)
analysis\gap\        why our results differ from the paper's: REPORT.md, RESULTS.md, scripts
analysis\characteristics\  the WRDS-built characteristics: market.md, accounting.md, training_data.md (definitions, match with the CPZ panel, limits)
results\runs\        one folder per configuration (git-ignored)
Data\TrainingData\   generated arrays (git-ignored): broad, common, broad_fill-none (build_data); window, window_train (build_window);
                     wrds_broad_fill-none, wrds_common_fill-none (build_wrds: built from WRDS only, §4)
Data\WRDS\           our WRDS data (git-ignored): raw\ as downloaded, never edited; universe\ (top-500 lists); characteristics\ (raw
                     monthly values); pelger\ (the CPZ panel as parquet, for checks)
Data\PelgerOldData\  Chen-Pelger-Zhu panel (3.2 GB), only for checking characteristics: never read its CSVs whole
Ander\               copy of Ander's Drive folder (2026-11-12): data\*.parquet, README.md, two HTML references, his CLAUDE.md
companionpaper\      companion paper's code, in set-sequence-model\ (git-ignored; untrusted: read, never run; parts derive from CRSP)
NOTES.md             details moved out of this file
JiaxiCaoShare\       copy for the instructor's review (2026-11-13; git-ignored): code, tests, analysis, docs, Data\WRDS, Data\TrainingData
                     without window*, results, README.md (4.9 GB). A snapshot: it does not follow later changes
*.pdf                3768292.3770398 = AFSA; 1904.00745v6 = the 39 characteristics; 2106.04028v2 = predecessor (costs, OU); syllabus
```

- **Read-only inputs:** `Ander\` and `Data\PelgerOldData\`. `Data\TrainingData\` is rebuilt, never edited.
- **`Ander\CLAUDE.md`** documents Ander's pipeline and decisions. Its paths, hooks and code are on his machine, and it may load into context when files under `Ander\` are read.
- **Licence.** Everything in `Ander\data\` and `Data\` derives from CRSP and Compustat (Stony Brook's WRDS subscription).
    - **GitHub:** WRDS-processed data may be committed to the private repo `Cao-Jiaxi/afsa-replication` (user's decision, 2026-11-13). Raw downloads (`Data\WRDS\raw\`) stay off it, and no WRDS data goes to any other external service.
    - `.gitignore` and the pre-commit hook still block all data, so processed data must be un-ignored deliberately, file by file. The hook also blocks `*.pdf` and files over 5 MB; GitHub itself refuses files over 100 MB.
    - The hook lives in `.git\hooks\`, so a clone has only `.gitignore`.

## 3. Code and commands

Each `afsa\` module opens with a docstring saying what it holds:
- `config.py`: `Config` holds every parameter; its defaults are Table 4 plus §7.
- `model.py`: §5 for M independent models at once.
- `metrics.py`: the Table 2 metrics and the paper's rows (`PAPER_TABLE2`).
- Also `panel.py` (GPU arrays, day × stock batches), `train.py` (training loop, out-of-sample evaluation), `run.py` (the rolling experiment), `build_data.py` and `build_window.py` (the arrays of §4).
- `wrds_db.py`: `query(sql, params)` → DataFrame, not via the `wrds` package (it pins pandas < 2.3). The user runs its setup (`-m afsa.wrds_db setup`; login in their `pgpass.conf`) and approves Duo pushes. Legacy `crsp.dsf` ends 2024-12-31, so use `crsp.dsf_v2` for 2025. Downloads go under `Data\WRDS\raw\`. The account has no international data (Compustat Global, Datastream, Worldscope: permission denied).
- `wrds_universe.py`: the monthly top 500 from our own downloads (§4); `download`, `build`, `compare` (against Ander's).
- `wrds_data.py`: the other raw downloads (its docstring lists every file and its units) and the CPZ panel as parquet. `wrds_db.copy_query` is the fast path for large tables.
- `chars_market.py` (13 monthly market characteristics) and `chars_accounting.py` (22 accounting ones): `build` and `check`; `check` compares with the CPZ panel through `pelger_check.compare`.
- `build_wrds.py`: the 4 daily characteristics, `rf`, the ranking and the training arrays `wrds_<universe>_fill-<rule>` (§4).

```powershell
.venv\Scripts\python.exe -m afsa.wrds_data download           # daily CRSP, Compustat annual, CCM links, indices, Fama-French (~2 min; skips files on disk)
.venv\Scripts\python.exe -m afsa.chars_market build           # -> Data\WRDS\characteristics\market.parquet (~15 s); `check` = match with the CPZ panel
.venv\Scripts\python.exe -m afsa.chars_accounting build       # -> accounting.parquet, link.parquet (~5 s); `check --detail`; `build --all-fills` / `--strict`
.venv\Scripts\python.exe -m afsa.build_wrds                   # -> TrainingData\wrds_broad_fill-none, wrds_common_fill-none (~1 min each, 5.6 GB RAM)
.venv\Scripts\python.exe -m afsa.run --universe wrds_broad --fill none --K 8 --seeds 0 1 2   # a run on the WRDS-built arrays (only a K = 30 validation run so far)
.venv\Scripts\python.exe -m afsa.build_data                   # check sources, rebuild TrainingData\broad (~10 s; --universe common)
.venv\Scripts\python.exe -m afsa.build_data --fill none       # the §4 policy -> broad_fill-none (~100 s)
.venv\Scripts\python.exe -m afsa.build_window                 # fixed-per-window universes (--rule window_train, --fill none, --check)
.venv\Scripts\python.exe -m pytest tests -q                   # 159 tests (~30 s)
.venv\Scripts\python.exe -m afsa.wrds_universe download       # crsp.msf_v2 + crsp.stkdelists -> Data\WRDS\raw\ (~1 min; may need a Duo approval)
.venv\Scripts\python.exe -m afsa.wrds_universe build          # top 500, broad and common -> Data\WRDS\universe\ (~10 s); then `compare`
.venv\Scripts\python.exe -m afsa.run --K 8 30 --seeds 0 1 2   # rolling 1998–2021 -> results\runs\<configuration>\ (--help lists all)
.venv\Scripts\python.exe tests\hardware\run_tests.py          # ~20 min of GPU training -> tests\hardware\RESULTS.md
.venv\Scripts\python.exe analysis\gap\variants.py --list      # gap variants; other scripts in analysis\gap\REPORT.md §8
```

- **Flag spelling.** `Config` flags take underscores (`--train_years`, `--drop_groups`, `--cumulative`, `--universe`, `--fill`). Run flags take hyphens (`--first-year`, `--last-year`, `--max-models`, `--tag`).
- **Tests** run on synthetic data, except those that read files under `Data\` (each is skipped without its files) and one that needs CUDA. The data tests of `test_build_wrds.py` fail on arrays older than their sources: rebuild with `afsa.build_wrds`. No test trains a model on a `fill none` folder.
- **Run folders** are named after their non-default parameters, e.g. `attn_K030_drop-past_returns`, plus `_<tag>` if given.
    - Each holds `config.json` and `seed<s>\<year>.npz` + `.pt`.
    - Finished (year, seed) pairs are skipped, so runs resume. If the folder already holds a different `config.json`, the run stops with an error: use `--tag`.
    - Only `afsa.run` folders have `summary.json`. Folders made by `variants.py` are scored in `analysis\gap\RESULTS.md`, without beta.
    - `train_years-6` folders are 1996–1997 validation runs.
- **Before changing a `Config` default,** write the old value into the existing `config.json` files, or move those folders. Missing keys load as the current default, and none of the 69 files has `fill`, so `afsa.run` would relabel the old runs and skip them as done.

## 4. Data

- **Sources:** Ander's `X_amplio.parquet` (79 inputs per universe stock-day), `crsp_daily.parquet` (`ret`) and `factors_daily.parquet` (`rf` and the calendar). His `README.md` and HTML references explain how they were built.
- **Arrays.** Each build runs 16 checks (18 with `--fill none`), all passing; results are in each folder's `meta.json`. `broad` is 9,067 days (1990-01-02 to 2025-12-31) × 500 slots: 4,527,395 stock-days and 2,120 PERMNOs.
- **Timing.** Row t pairs what was known at the close of t−1 with the return of day t, i.e. the paper's X_{t−1} and R_t (checked on `Ret_D1`).
- **`broad` (main universe):** each month, the 500 largest NYSE/AMEX/NASDAQ equities by previous month-end market cap, ADRs, foreign shares, REITs and MLPs included. It holds only 70 of the 109 firms in the paper's Figures 3–4, so results are comparable, not identical.
- **`common`** (US common stocks only) is for robustness.
- **Rebuilt from WRDS** (2026-11-13, `afsa.wrds_universe`): both universes equal Ander's on all 216,000 stock-months of 1990–2025, with the same ranks, caps and types (`tests\test_wrds_universe.py`). They also hold the lookback lists of 1989-11 and 1989-12. Rule and CIZ fields in `NOTES.md` §5.
- **Fixed per window** (robustness): the 500 largest at the end of the training years, among the 2,521 securities in Ander's files.
    - `window` keeps securities with a return on every day of the training and test years, so it looks ahead. It approximates the companion paper's rule (arXiv 2505.11243; `NOTES.md` §4).
    - `window_train` requires the training years only.
    - Stored as `<rule>\<n>y\<test year>\`: 8y for 1998–2021, 6y for 1996–1997.
- **Missing characteristics (policy, Prof. Polak, 2026-11-13):** compute each characteristic, and leave it empty if its raw inputs are not available. When ranking, an empty characteristic gets rank 0 (the median, harmless to the network), and the stocks with a value are ranked among themselves into [−0.5, 0.5]. No carry-forward.
    - This is `--fill none`, built for `broad` only (effects in `NOTES.md` §1).
    - Ander's X (`--fill last`, still the default) carries the last value forward with no age limit, as the paper's §4.1 says.
- **Returns** are in excess of `rf`, stock by stock. The 87 missing returns (suspensions) count as total returns of 0. `rf` has 4 decimals, so it is 0 throughout 2003, 2009–2017 and 2021.
    - On WRDS, `ff.factors_daily.rf` takes only 7 distinct values ever.
    - `ff.factors_monthly.rf` has 4 decimals per month: 0.21% in 2016 and 0.79% in 2017, where the daily series is 0. It runs to 2026-09.
    - `frb.rates_daily.dtb3` (3-month bill) ends 2025-02-11. `crsp.tfz_*` is not readable.
- **Loss days** start on 1990-02-14 in the Ander-based folders (§7). After that, 3.8% of stock-days get no signal, because they have fewer than 30 prior days in the universe.
- **WRDS-built arrays** (2026-11-13; `wrds_broad_fill-none`, `wrds_common_fill-none`). Nothing of Ander's is used. Rules in `NOTES.md` §6, detail in `analysis\characteristics\`.
    - **Span:** 9,108 days from 1989-11-01. The 41 days of 1989 are lookback only, so loss days start on 1990-01-02. The 4,527,395 broad stock-days of 1990–2025 equal Ander's.
    - **Monthly formulas** reproduce the CPZ panel: exactly equal in 98.7–100% of stock-months for 33 of 35. Resid_Var and Beta reach 99.95% and 99.77% only with French's factor file of 2016-11; with today's WRDS factors, which the arrays use, they agree in rank (0.9998, 1.0000).
    - **The panel covers** 56.6% of the universe's stock-months of 1990–2016, almost no bank or insurer, and nothing after 2016.
    - **Departures from the panel, to avoid look-ahead:** accounts enter 4 months after the fiscal year-end (6 for ADRs and non-US issuers); today's factors.
    - **Missing inputs:** 15 secondary accounting items count as 0 where the panel shows it; a missing XSGA, DP, COGS, CEQ or LCT leaves the characteristic empty. FC2Y, OP and OL are at rank 0 on 22.6% of stock-days, OA on 20.8%, DPI2A on 11.0%.
    - **Daily characteristics** follow the paper's words: Ret_D1 = return of t−1; Ret_W1, STD_W1 and Vol (shares) over days t−5..t−1.
    - **`rf`** is the monthly Fama-French rate spread over the month's scheduled trading days.
    - **Audited** twice by independent agents (look-ahead by truncation, panel match recomputed, arrays rebuilt on every day): no critical or major finding left.

## 5. Model and protocol (AFSA §3–4, Appendices A–B)

On day t, over the N ≤ 500 stocks of the universe, with X_{t−1} ∈ R^{N×79}:

1. ω^F = Softmax(Q (X W^K)ᵀ / √d_x), with the softmax over stocks (K × N). Factors F_t = ω^F R_t.
2. βᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I)⁻¹. Residuals ε_t = R_t − βᵀ F_t = ω^ε R_t.
3. ω^port_i = LongConv(ε_{i, t−30..t−1}): one network for all stocks.
4. ω = (ω^ε)ᵀ ω^port, scaled to ‖ω‖₁ = 1. Cost_t = 0.0005 ‖ω_t − ω_{t−1}‖₁ + 0.0001 ‖max(−ω_t, 0)‖₁.
5. Maximize net Sharpe + λ_Var · (1/N) Σ_i (1 − Var(ε_i)/Var(R_i)). W^K, Q and LongConv are learned jointly.

- **Protocol:** a rolling 8-year window, retrained yearly. 24 test years (1998–2021); the first window is 1990–1997. K ∈ {1, 3, 5, 8, 15, 30, 100}, 3 seeds.
- **Medians and `rf`** are equal across a day's stocks, so the softmax cancels them: the model as written ignores them (there is a test).

## 6. Targets

- **Table 2** (1998–2021, annualized; SR = μ/σ; beta against the equal-weighted market): the attention rows are `PAPER_TABLE2`, also shown in §9. Their costs imply 2.6–2.8 bp a day for K ≤ 30 and 3.4 for K = 100. That is a daily turnover of 0.42–0.47 (0.58 for K = 100), if short exposure is 0.5.
- **Benchmarks** (SR / SR_net): PCA + LongConv 2.64 / 1.50 (K = 8) and 2.79 / 1.57 (K = 30); PCA + OU 0.78 / −4.15 (K = 8); equal-weighted market 0.42.
- **Table 3** (K = 30, one characteristic group dropped; SR / SR_net):

  | Group dropped | SR / SR_net |
  | --- | --- |
  | none | 3.97 / 2.28 |
  | past returns | 1.50 / 0.59 |
  | trading frictions | 2.90 / 1.34 |
  | investment | 3.88 / 2.19 |
  | profitability | 3.94 / 2.26 |
  | intangibles | 3.91 / 2.24 |
  | value | 4.08 / 2.32 |

- **Figure 2** (cumulative gross returns) is digitized in `analysis\gap\paper_fig2.csv`. It has 5,880 days, the first 245 returns of each year, while `afsa.metrics` scores all 6,040 days. Compare like with like.
- **Figures 3–4:** the K = 8 model trained on 2013–2020, evaluated on the first trading day of 2021. They show the t-SNE of the betas and the top 10 weights of factors 1–6.

## 7. Decisions (what the paper does not specify; 2026-11-12 unless dated)

| Decision | Reason |
| --- | --- |
| No tuning and no early stopping: each window trains 30 epochs (AdamW, constant learning rate) and keeps the last model | Table 4 gives the hyperparameters; Ander settled the open choices on 1990–95 / 1996–97 |
| Ander's choices: annualized Sharpe in the objective; batches of 126 consecutive days; λ_ridge = 1e-5; block = Linear 1→32, LongConv, GELU, dropout, Linear, residual + LayerNorm, read-out of the newest day, Linear → scalar; residuals in %, not cumulative; each lag from its own day's projection; no signal under 30 prior days in the universe; per-stock excess returns (the paper's printed formula is `--rf_hurdle true`); medians and `rf` standardized over all days of the training years; a test year's first day pays the trade from the previous model; 3 seeds | His log (`Ander\CLAUDE.md` §7) records validation for the Sharpe scale, batching, λ_ridge and the residual input, and arguments for the rest. The lag projection is inferred from his docs |
| Models train together in one GPU pass, with a leading model dimension; tests show this equals training them one by one. `window*` universes put one test year's seeds in each pass | 3.5–4 times faster (`tests\hardware\REPORT.md`) |
| An epoch is 16 batches of 126 days tiling the window from a random offset, in random order. Windows short of 2,016 days (1998 by 23 days; 2002–2025 by 1–6) get evenly spread, slightly overlapping batches | Every model takes the same number of steps, which batching needs |
| Explained variance: per-stock variances over the batch's loss days, for stocks with ≥ 21 days; plain average | The paper gives only the formula; a few days give a noisy ratio |
| Initialization: W^K uniform ±1/√79; Q and D ~ N(0, 1); kernel as in Appendix A with x ~ N(0, 1); other layers as `nn.Linear`. Weight decay on the sequence model, none on W^K and Q | The paper gives only the kernel and "weight decay in LongConv model" |
| The K × K ridge solve runs in float64, the rest in float32 | λ_ridge is 1e-5 and the solve is tiny |
| Loss days start on the 32nd trading day of 1990 until the pre-1990 lookback exists; the first test day of 1998 is bought from cash | The day before a loss day needs its own 30-day history |
| Each (seed, test year) has its own random streams for initialization, batch order and dropout | A model does not depend on which others share its pass |
| 2026-11-13: data are rebuilt from our own WRDS downloads, starting with the universe. A security leaves only if its last trading day is before the formation day (`NOTES.md` §5) | The user's decision: WRDS is the only source. This rule reproduces Ander's universe exactly; a stale-price rule or excluding formation-day delistings did not |
| 2026-11-13, WRDS build: each monthly formula is the one that reproduces the CPZ panel, also where it contradicts the written definition (list in `NOTES.md` §6) | The user's instruction: the panel is the arbiter |
| 2026-11-13, WRDS build: fiscal rule `safe`. The panel's rule (July of Y to June of Y+1 uses Compustat `fyear` Y−1), but a fiscal year enters only once its year-end + 4 months (6 for ADRs and non-US issuers) is on or before the first day of the holding month | The panel's rule uses March–May year-ends 1–3 months old. 10-K: 90 days; 20-F: 6 months until fiscal 2011 |
| 2026-11-13, WRDS build: a missing accounting input counts as 0 only where the panel shows the fill and a missing item plausibly means none (15 items). XSGA, DP, COGS, CEQ and LCT leave the characteristic empty | The user's missing-value rule. With every fill on, FC2Y was exactly 0 on 17.5% of universe stock-months, tied near the bottom instead of at rank 0. `build --all-fills` is the pure panel convention |
| 2026-11-13, WRDS build: Resid_Var and Beta use today's WRDS Fama-French factors | WRDS is the only source, and one vintage covers 1990–2025. The 2016 file only verifies the formulas |
| 2026-11-13, WRDS build: `rf` = monthly Fama-French rate over the month's scheduled trading days (realized days plus unscheduled closures) | The daily WRDS series has 4 decimals and is 0 in whole years |
| 2026-11-13, WRDS build: daily characteristics by the paper's words (5 trading days to t−1, weekly share volume; all five days required). Ret_W1 and STD_W1 are rounded to 12 decimals before ranking | The paper is explicit. The companion code uses 7 days skipping the latest and one-day volume (§10). Rounding makes exact ties share a rank |
| 2026-11-13, WRDS build: all 39 characteristics are ranked over the day's members (Ander's X ranks the monthly ones over the month's list) | One cross-section per day; moves a rank by at most 0.009 |
| 2026-11-13: `train_span` refuses a training window whose first year the arrays hold only in part | The wrds folders hold 41 days of 1989 as lookback |
| 2026-11-13: the missing-characteristic policy of §4 (`--fill none`), departing from the paper's §4.1. `Config.fill` stays `last` until Table 2 is rerun under it (see §3 before changing it) | Prof. Polak's instruction: rank 0 is the median, which does not distort the network's input |

## 8. Environment and rules

- **Machine:** Windows 11, PowerShell 5.1 (no `&&`). Python 3.13 in `.venv\`, pinned in `requirements.txt` (torch 2.11.0+cu128). RTX 5080 16 GB (needs a CUDA 12.8+ build; the desktop holds 2–3 GB), i7-12700KF, 32 GB RAM.
- **git:** git and `gh` are on the PATH; a shell opened before git was installed (2026-11-12) needs `C:\Program Files\Git\cmd\git.exe`. `origin` is `github.com/Cao-Jiaxi/afsa-replication`, branch `main`. **Commit or push only when asked.**
- **No look-ahead.** Weights for day t use information up to t−1 only, scaling statistics included (training window only). Never choose anything by looking at test-year results (1998 onward). If a choice must be revisited, compare on 1996–1997 only.
- **One GPU job at a time, with memory headroom** (`run_tests.py` counts as a job).
    - `afsa.run` sizes its passes from free GPU memory, up to 36 models; `--max-models` overrides.
    - When memory runs out, Windows slows down instead of failing: two 36-model passes together filled the 16 GB and ran about 15 times slower.
    - Timings are in `tests\hardware\REPORT.md`.
- **Reproducibility.** Results, tables and figures come from code, and each run saves its configuration with its results. Every choice the paper does not specify gets a row in §7. Code, docs and conversation are in English.

## 9. Status and results

- [x] Data (Ander, 2026-10-13). 2026-11-12: training arrays and checks; code and 35 unit tests; hardware tests; all seven K × 3 seeds, 1998–2021; gap analysis (`analysis\gap\REPORT.md`, 32 variants).
- [ ] Decide with Ander (§10): cumulative residuals as the LongConv input, and the main universe. The `Config` defaults are still the §7 baseline.
- [ ] `--fill none`: arrays built and checked; code paths in `config`, `build_data`, `build_window`, `run` and `analysis\gap`. **Paused at the user's request** before running the unit tests or any model with it, and before making it the default.
- [x] Universe rebuilt from WRDS (2026-11-13): identical to Ander's, both universes (§4).
- [x] The rest of the data from WRDS (2026-11-13): downloads, the 39 characteristics checked against the CPZ panel, and the training arrays `wrds_broad_fill-none` and `wrds_common_fill-none` (§4), audited twice.
- [ ] Runs on the WRDS-built arrays (2026-11-13), rows below. K = 30: validation 1996–1997, then 1998–2025 in one folder (`attn_K030_universe-wrds_broad_fill-none`; its `summary.json` covers 1998–2025, so score sub-periods with `metrics.summarize(..., save=False)`). The first scoring failed: `metrics.market_returns` ignored the fill suffix of the folder (fixed, with a test). Addentax's −95% day (2022-09-01) is not among any seed's three worst days. Still to do: the other K; a `--universe broad --fill none` run to split the data source from the fill rule.
- [ ] Commit and push, when asked. `origin/main` holds only `7875556` (2026-11-12, early code), and everything since is uncommitted or untracked. `.gitignore` also drops `tests\hardware\results\` and `analysis\gap\paper_fig2.csv` (rebuilt from the git-ignored PDF): decide on those first.
- [ ] Stale docs, written before the companion code arrived:
    - `build_window.py`, which calls `window` the companion's rule;
    - `analysis\gap\REPORT.md` §2 and §5, `fig2_digitize.py` and `yearly.py`.
- [ ] Ander's code and results (§10); the lookback before 1990 (§4).
- [ ] Later:
    - Table 3's other four rows. Past returns and trading frictions are done at K = 30 with raw input: SR / SR_net 3.14 / 1.14 and 3.54 / 0.85.
    - Figures 2–4, PCA + LongConv, the extension (`--first-year 2022 --last-year 2025`), the notebook and the presentations.

**Next step:** agree with Ander on the two decisions; then a validation run on `wrds_broad` (1996–1997), and Table 2 with the agreed specification on the WRDS-built arrays.

**Results** (seed averages; μ and σ in % a year; turnover per day; every variant in `analysis\gap\RESULTS.md`, by period in `NOTES.md` §3):

| Run, 1998–2021 | SR | SR_net | μ / μ_net | σ | Beta | Turnover | Paper SR / SR_net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K = 1 | 2.24 | 1.05 | 16.50 / 7.75 | 7.35 | 0.14 | 0.62 | 3.05 / 1.68 |
| K = 3 | 2.88 | 1.44 | 15.76 / 7.83 | 5.47 | 0.10 | 0.55 | 3.05 / 1.69 |
| K = 5 | 3.08 | 1.35 | 14.92 / 6.52 | 4.84 | 0.08 | 0.58 | 2.92 / 1.58 |
| K = 8 | 3.53 | 1.36 | 15.38 / 5.93 | 4.36 | 0.07 | 0.66 | 3.35 / 1.94 |
| K = 15 | 3.65 | 1.20 | 14.39 / 4.70 | 3.95 | 0.05 | 0.67 | 3.81 / 2.25 |
| K = 30 | 3.86 | 1.06 | 13.32 / 3.64 | 3.45 | 0.03 | 0.67 | 3.97 / 2.28 |
| K = 100 | 3.73 | 0.39 | 10.45 / 1.10 | 2.80 | 0.02 | 0.64 | 4.52 / 2.19 |
| K = 8, cumulative input | 2.94 | 1.51 | 13.09 / 6.71 | 4.45 | 0.08 | 0.41 | 3.35 / 1.94 |
| K = 30, cumulative input | 3.37 | 1.26 | 11.21 / 4.18 | 3.33 | – | 0.46 | 3.97 / 2.28 |
| K = 30, `wrds_broad --fill none` | 3.81 | 0.98 | 13.04 / 3.34 | 3.42 | 0.03 | 0.67 | 3.97 / 2.28 |
| K = 30, `wrds_broad --fill none`, 2022–2025 | 1.08 | −0.43 | 3.85 / −1.55 | 3.58 | 0.06 | 0.34 | – |
| K = 8, validation 1996–1997 | 10.91 | 7.19 | 32.29 / 21.26 | 2.96 | 0.06 | 0.78 | – |
| K = 30, validation 1996–1997 | 11.67 | 7.31 | 29.03 / 18.16 | 2.49 | 0.04 | 0.76 | – |
| K = 30, validation, `wrds_broad --fill none` | 11.37 | 7.00 | 27.82 / 17.11 | 2.45 | 0.04 | 0.75 | – |

- **Validation** is `--train_years 6 --first-year 1996 --last-year 1997`, which uses nothing after 1997. Judge any change of §7 against it, never against test years.
- **WRDS arrays, K = 30:** validation SR_net 0.31 below the baseline (seed sd 0.08–0.09), from gross return at the same turnover. On 1998–2021 they track Ander's arrays year by year: SR_net 0.98 against 1.06 (3.28 / −0.08 / −0.35 in 1998–2005 / 2006–2013 / 2014–2021, against 3.45 / −0.06 / −0.30). Three things differ at once: data source, fill rule, and loss days from 1990-01-02 instead of 1990-02-14.
- **Extension 2022–2025 (K = 30, WRDS):** SR 1.08, SR_net −0.43, turnover 0.34; net return by year 0.7%, −0.3%, −2.3%, −4.5%. Equal-weighted market: SR 0.27.
- **The gap** (`analysis\gap\REPORT.md`, Conclusions):
    - Before costs K = 8 matches the paper, but we trade 0.55–0.67 a day against the paper's implied 0.42–0.47, so every net SR falls short. The paper's net SR rises with K up to 30; ours peaks at K = 3.
    - No code error: an independent audit replayed saved models and matched the stored results.
    - Cumulative input brings turnover to the paper's level, but gross return still falls short (§10).
    - Models fit their window far better than they trade the next year (net SR 5–7 against 1.1–1.4), and more fitting means more trading. After 2005, K = 30 does not earn its costs.

## 10. Open questions

- **Ander's code and results.** Ask for `src/`, `tests/`, `docs/implementation_decisions.md`, `notebooks/attention_factors_results.ipynb` and `results/runs/`. His `CLAUDE.md` §6 marks Tables 2–3, PCA + OU, PCA + LongConv and the extension as done, but records no Sharpe ratios. Compare his choices with §7, above all batching, initialization and explained variance.
- **Sequence-model input (pending).**
    - Cumulative residuals reproduce the paper's turnover and are what the predecessor paper states.
    - The validation years prefer raw residuals (SR_net 7.19 against 6.32 for K = 8; 7.31 against 6.68 for K = 30), because 1996–1997 rewards speed.
    - `analysis\gap\REPORT.md` §6 recommends cumulative as the main specification, raw beside it.
    - Cumulative has not been run for K = 1, 3, 5, 15, 100 or for Table 3.
- **Gross return at the paper's turnover:** 13.1% against 15.7% (K = 8), and 11.2% against 16.7% (K = 30). The paper's edge comes in 1998–2007, and it gains from more factors where we do not. Unexplained; the universe is the main suspect.
- **Universe.** About a third of the paper's named stocks are not in ours, so expect gaps that no code fix closes.
    - `REPORT.md` §6 keeps the monthly top 500 as the main universe, with fixed-per-window as a robustness check.
    - `window` reproduces the paper's weak returns after 2013, but not its strong early years.
    - The companion code fixes 500 stocks for 10 years by company market cap, and needs complete data through the year after the reported one (`NOTES.md` §4).
- **For the professor:** the code or results of the earlier AFSA replication, and which universe it used.
- **Missing raw inputs inside a characteristic.** Decided for the WRDS build (§7): 15 items the panel confirms count as 0; XSGA, DP, COGS, CEQ and LCT do not. Still for Prof. Polak and the user: read literally, the §4 policy would also empty the 15 (`build --strict`: CF 99.96% empty, OP 95%, FC2Y 77%). CEQ and LCT are a judgement: the panel does fill them, on 4 and 3 securities.
- **Weekly characteristics (user's call).** Built by the paper's words. The companion code uses days t−8..t−2 for the weekly return and volatility and one day's volume; the daily rank correlation between the two versions is 0.62 (Ret_W1), 0.77 (STD_W1) and 0.95 (Vol). If the paper's numbers came from that code, these inputs differ materially.
- **20-F on the deadline day.** A foreign filer's December year-end enters in July, formed on the last trading day of June, the old 20-F deadline itself: 1,111 universe stock-months (0.51%) rely on the issuer having filed by then. A 7-month lag would close it (`min_lag_months_foreign`).
- **`Config.fill` defaults to `last`,** so runs on the wrds folders need `--fill none` (see §3 before changing a default).
- **Medians.** They are inert under eq. (1). Did the authors' code differ, say with a nonlinear embedding? The companion config gives a hint (`NOTES.md` §4).
- **Exchanges in the universe.** The paper says only "securities on CRSP". The NYSE/AMEX/NASDAQ filter (Ander's) excludes only Cboe Global Markets, 39 months in 2018–2025. Keep the filter unless the professor says otherwise.
- **Outlier filter for the extension.** Addentax (ATXG) entered the top 500 in September 2022 after a post-IPO surge, and lost 95% on its first day there. The paper has no price or liquidity filter.
