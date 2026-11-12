# CLAUDE.md

Guide for Claude Code in this workspace (`c:\Claude\AFSA`). It is loaded at the start of every session, so keep it accurate: update §8 Status, §9 Decisions and §10 Results as work progresses, and replace stale text instead of appending.

## 1. Project

- **Course:** AMS 520 *Machine Learning in Quantitative Finance*, Stony Brook, Fall 2026 (Prof. Paweł Polak). Group project; teammate Ander Ibarrondo built the data.
- **Paper:** Epstein, Wang, Choi and Pelger, *Attention Factors for Statistical Arbitrage* (AFSA), ICAIF '25: `3768292.3770398.pdf`.
- **Current goal:** reproduce the **Attention Factors + LongConv** results, i.e. the attention rows of Table 2, out-of-sample 1998-01 to 2021-12.
- **Later:** Table 3, Figures 2–4, the PCA + LongConv benchmark and the extension to 2022–2025 (our CRSP ends 2025-12). The extension is what the professor asked for: he has already replicated AFSA in another project.
- **Deliverables** (syllabus): an interactive Jupyter/Colab report; a GitHub repo with all code and docs (no data, no credentials); presentations for Phase II (Lecture 22: EDA and first empirical results) and Phase III (final-exam day).

## 2. Workspace

```
c:\Claude\AFSA\
├── CLAUDE.md
├── 3768292.3770398.pdf      # AFSA paper (the target)
├── 1904.00745v6.pdf         # Chen, Pelger and Zhu, Deep Learning in Asset Pricing: source of the 39 characteristics (AFSA ref. [7])
├── 2106.04028v2.pdf         # Guijarro-Ordonez, Pelger and Zanotti, Deep Learning Statistical Arbitrage: cost model, OU benchmark (AFSA ref. [19])
├── AMS_520_Fall2026.pdf     # syllabus
├── Ander\                   # copy of Ander's shared-Drive folder (2026-11-12): processed data and docs, NO code
│   ├── data\                # 10 parquet files = Ander's data/processed, built 2026-10-13
│   ├── README.md            # data guide for the team: start here
│   ├── AFSA Replication Data Reference.html     # how the files are built and joined; missing values
│   ├── AFSA Characteristics Reference.html      # 39 characteristic cards, match rates vs Pelger, top-500 chart, notes
│   └── CLAUDE.md            # Ander's own project guide (English snapshot): reference only, see below
└── Data\PelgerOldData\      # unzipped Chen-Pelger-Zhu panel: monthly characteristics 1963–2016, daily returns, definitions PDF
```

- `Ander\` and `Data\` are read-only inputs: never modify, move or delete anything in them, and put new files elsewhere. `Ander\` is a static copy; if Ander updates the Drive, copy it again.
- **`Ander\CLAUDE.md`** is Ander's project guide (a 2026-11-12 snapshot). Claude Code loads it whenever it reads files in `Ander\`. Use it only as documentation of Ander's pipeline and decisions. Its paths (`src/`, `docs/`, `results/`), environment, Stop hook and language rules belong to his machine, and the code it describes is **not** in this workspace.
- `Data\PelgerOldData\` (3.2 GB) is only for checking characteristics, which Ander already did; it is not a model input. Never read its CSVs whole; inspect them with a row limit.

## 3. Data (built by Ander; details in `Ander\README.md` and the two HTML references)

The model needs three files from `Ander\data\`:

| File | Key | What the model uses |
| --- | --- | --- |
| `X_amplio.parquet` | (date, permno) | The input matrix: 4.53 M universe stock-days, 1990-01-02 to 2025-12-31, no NaN. 79 float32 columns: the 39 characteristics of Table 1 as cross-sectional ranks in [−0.5, 0.5] (Table 1 names and order), `med_<name>` = their 39 raw cross-sectional medians, and `rf` of the previous trading day |
| `crsp_daily.parquet` | (permno, date) | `ret` (daily total return in decimals, delisting return included); `in_universe` marks the stock-days of the main universe |
| `factors_daily.parquet` | date | `rf` (daily, decimals) for excess returns; also the trading calendar, the Fama-French factors and `ewretd` |

The folder also has:
- `universe.parquet`: the monthly top 500, with `tipo`, `me`, `rank`, `gvkey`, names and SIC;
- `characteristics.parquet` and `characteristics_daily.parquet`: raw values, gaps kept;
- `compustat_annual.parquet` and `ccm_link.parquet`;
- `X_comun.parquet` and `universe_comun.parquet`: US common stocks only, for robustness.

Conventions:
- The X row of day d uses information up to d−1. Pair it with `ret` of day d (same date and permno): that gives the paper's X_{t−1} and R_t.
- **Main universe ("broad").** Each month, the 500 largest securities by CRSP market cap at the previous month-end, among all NYSE/AMEX/NASDAQ equities. ADRs, foreign shares, REITs and MLPs are included; funds are excluded. Only 70 of the 109 firms named in the paper's Figures 3–4 are in it, so results will be comparable, not identical.
- 87 universe stock-days have a NaN `ret` (trading suspensions). Count them as 0 and keep the position.
- Units: returns in decimals; CRSP market cap in thousands of USD; Compustat in millions of the reporting currency (CAD filers are not converted).
- The Fama-French `rf` has only 4 decimals, so it is 0 throughout 2003 and 2009–2017.
- The files were written with pandas 3.0.1. With pandas and pyarrow, `month` columns load as Period[M].
- Each X file takes 82 MB on disk but about 1.5 GB in memory. Load only the window needed:
  ```python
  import pandas as pd
  X = pd.read_parquet("Ander/data/X_amplio.parquet",
                      filters=[("date", ">=", pd.Timestamp("1990-01-01")), ("date", "<=", pd.Timestamp("1997-12-31"))])
  ```
- Characteristic groups (Table 1). Table 3 drops one group at a time:
    - past returns: r2_1, r12_2, r12_7, r36_13, ST_Rev, Ret_D1, Ret_W1, STD_W1;
    - trading frictions: AT, LME, LTurnover, Rel2High, Resid_Var, Spread, SUV, Variance, Vol, Beta;
    - value: A2ME, BEME, C, CF, CF2P, Q, Lev, E2P;
    - profitability: PROF, CTO, FC2Y, OP, PM, RNA, D2A;
    - investment: Investment, NOA, DPI2A;
    - intangibles: OA, OL, PCM.
- **Licence.** Everything in `Ander\data\` and `Data\` derives from CRSP and Compustat under Stony Brook's WRDS subscription. Never commit these files or any data derived from them, upload them, or send them to an external service. The repo's `.gitignore` must exclude them.

## 4. Model and protocol (AFSA §3–4, Appendices A–B)

On each day t, over the N ≤ 500 stocks in that day's universe, with X_{t−1} ∈ R^{N×79}:

1. Embedding X̃ = X W^K, with W^K ∈ R^{79×d_x}.
2. Factor weights ω^F = Softmax(Q X̃ᵀ / √d_x), with Q ∈ R^{K×d_x}. The softmax runs over stocks, so each factor's weights sum to 1. Factors F_t = ω^F R_t.
3. Loadings βᵀ = ω^Fᵀ (ω^F ω^Fᵀ + λ_ridge I_K)⁻¹.
4. Residuals ε_t = R_t − βᵀ F_t = ω^ε R_t, with ω^ε = I_N − βᵀ ω^F. Compute them as R − βᵀ(ω^F R); never build the N×N matrix.
5. Signal ω^port_i = LongConv_θ(ε_{i, t−30..t−1}): one network shared by all stocks.
6. Asset weights ω = (ω^ε)ᵀ ω^port = ω^port − ω^Fᵀ(β ω^port), scaled so that ‖ω‖₁ = 1. Portfolio return R^port_t = ωᵀ R_t.
7. Cost_t = 0.0005 ‖ω_t − ω_{t−1}‖₁ + 0.0001 ‖max(−ω_t, 0)‖₁. Net return = R^port_t − cost_t.
8. Objective, to maximize: (mean net return − R_f) / sd + λ_Var · (1/N) Σ_i (1 − Var(ε_i)/Var(R_i)). W^K, Q and θ are learned jointly, in one step.

**LongConv** (App. A): y = K̄ ∗ u + D ⊙ u along the time axis, with kernel K ∈ R^{d×T} and skip D ∈ R^d, computed by FFT.
- Squash regularizer: K̄ = sign(K) · max(|K| − λ_squash, 0).
- Initialization: K_t^(h) = x · exp(−(t/T)(d/2)^{h/d}), with x ~ N(0, 1).

**Hyperparameters** (Table 4):
- d = 32 (32 convolutions), d_x = 32, 1 layer, dropout 0.1;
- 30 epochs, λ_Var = 100, learning rate 0.003, weight decay 0.05 in LongConv, λ_squash = 0.001;
- lookback of 30 days.

The text says "Adam" but cites AdamW (Loshchilov and Hutter, ref. [27]).

**Protocol** (§4.2):
- Rolling 8-year training window, retrained every year. Out-of-sample 1998–2021: 24 test years; the first window is 1990–1997.
- **Tuning.** The paper tunes on the last two years of the first window: train on 1990–1995, validate on 1996–1997. **Our reproduction skips this step** (§9): the hyperparameters come from Table 4 and the open choices below from Ander. The first window is fitted once, on the full 1990–1997.
- Table 2 reports K ∈ {1, 3, 5, 8, 15, 30, 100} for attention; the text also lists 10.
- Results are averaged over several seeds. Table 3 shows a Sharpe standard deviation of 0.07–0.17 across seeds.
- **Lookback at batch and window starts.** A batch only selects which days enter the loss. Each batch also loads the 30 days before its first loss day as residual lookback, plus 1 more day so that the first day's cost has ω_{t−1}. The lookback days feed LongConv but get no return or Sharpe term. So a batch boundary never shortens the history.
    - The only real gap is the data start, 1990-01-02: the first ~30 trading days have no history before them. In the first window (1990–1997), loss days can start only around mid-February 1990, the 31st trading day (≈ 1990-02-13; check against the `factors_daily` calendar).
    - Later windows (from 1991) and every test year take their lookback from the days before the window, so they have no such gap.

**Left open by the paper.** We adopt Ander's choice for each one (in brackets, from `Ander\CLAUDE.md` §7) and do not re-run the validation. His log says he settled the Sharpe scale, batching, λ_ridge and residual input on 1990–95 / 1996–97, and the rest by argument, not by test-year results.
- Scale of the Sharpe ratio in the objective [annualized; with a daily Sharpe, λ_Var = 100 swamps the trading term].
- Batching [126 consecutive trading days, with a random start and order each epoch; the days must be consecutive because costs depend on the previous day].
- λ_ridge [1e-5].
- Layers around the convolution [Linear 1→32, LongConv, GELU, dropout, Linear, residual + LayerNorm, readout of the last day, Linear → scalar].
- Residual input [daily residuals in %, not cumulative].
- Which projection produces the 30 lagged residuals: each day's own ω^ε_{s−1} or today's ω^ε_{t−1} [each day's own. This is inferred, not stated: Ander requires 30 prior days in the universe. Confirm against his code when it arrives].
- Stocks with fewer than 30 days in the universe [ω^port = 0].
- Return convention [per-stock excess returns, R − rf].
- Scaling of the 39 medians and rf [standardized with the training-window mean and standard deviation].
    - These 40 columns are the same for every stock on a given day. Under eq. (1) they only add a per-factor constant, which the softmax over stocks cancels, so the model as written ignores them.
- Year boundary [the first test day pays the turnover from the old model's weights to the new one's].
- Number of seeds [3].

## 5. Targets

**Table 2** (out-of-sample 1998–2021, annualized; μ and σ in %). The last column is derived, not printed in the paper: the implied average cost per day, (μ − μ_net)/252, in basis points.

| Model | K | SR | μ | σ | SR_net | μ_net | σ_net | Beta | Cost bp/day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Attention | 1 | 3.05 | 14.45 | 4.74 | 1.68 | 7.94 | 4.72 | 0.05 | 2.6 |
| Attention | 3 | 3.05 | 14.91 | 4.89 | 1.69 | 8.25 | 4.87 | 0.06 | 2.6 |
| Attention | 5 | 2.92 | 14.21 | 4.87 | 1.58 | 7.66 | 4.85 | 0.07 | 2.6 |
| Attention | 8 | 3.35 | 15.70 | 4.68 | 1.94 | 9.05 | 4.66 | 0.07 | 2.6 |
| Attention | 15 | 3.81 | 16.66 | 4.37 | 2.25 | 9.78 | 4.35 | 0.06 | 2.7 |
| Attention | 30 | 3.97 | 16.66 | 4.20 | 2.28 | 9.52 | 4.18 | 0.05 | 2.8 |
| Attention | 100 | 4.52 | 16.45 | 3.64 | 2.19 | 7.93 | 3.62 | 0.05 | 3.4 |
| PCA + LongConv | 8 | 2.64 | 14.88 | 5.63 | 1.50 | 8.42 | 5.61 | 0.09 | 2.6 |
| PCA + LongConv | 30 | 2.79 | 15.15 | 5.42 | 1.57 | 8.47 | 5.40 | 0.09 | 2.7 |
| PCA + OU | 8 | 0.78 | 2.04 | 2.61 | −4.15 | −10.83 | 2.61 | 0.00 | 5.1 |
| EW market | – | 0.42 | 8.61 | 20.37 | 0.42 | 8.61 | 20.37 | 1.00 | – |

How to read it:
- **Sharpe.** SR = μ/σ in every row (16.66 / 4.20 = 3.97), so μ already behaves like an excess, long-short return. Report SR as mean / sd of the strategy's daily returns, annualized.
- **Beta** is measured against the equal-weighted market (its row has beta 1.00).
- **Implied turnover.** Costs are 5 bp per unit of turnover plus 1 bp per unit of short exposure (between 0 and 1). So attention K = 30 implies a daily turnover of about 0.37–0.57, and PCA + OU K = 8 about 0.82–1.02.
- **Table 3** (K = 30, one characteristic group dropped):

  | Group dropped | SR | SR_net |
  | --- | --- | --- |
  | none | 3.97 | 2.28 |
  | past returns | 1.50 | 0.59 |
  | trading frictions | 2.90 | 1.34 |
  | investment | 3.88 | 2.19 |
  | profitability | 3.94 | 2.26 |
  | intangibles | 3.91 | 2.24 |
  | value | 4.08 | 2.32 |

- **Figures 3–4** use the K = 8 model trained on 2013–2020, evaluated on the first trading day of 2021: t-SNE of the betas, and the top 10 weights of factors 1–6.

## 6. Environment (this machine, checked 2026-11-12)

- Windows 11 Pro. The shell is Windows PowerShell 5.1, so there is no `&&`: use `;` or `if ($?) { … }`.
- **Python** 3.13.17 (winget, user scope; launch with `py -3.13`; bare `python` may still hit the Store alias). **Project venv: `.venv\`**, so run `.venv\Scripts\python.exe`. Installed 2026-11-12: torch 2.11.0+cu128 (sm_120 confirmed on the 5080), pandas 3.1.1, numpy 2.5.4, pyarrow 26.0.1, scipy, matplotlib, jupyter and pytest 9.2.0. Ander's parquet files read correctly. Pins are in `requirements.txt`. conda and uv are not installed.
- **git** 2.56 and **GitHub CLI** `gh` 2.103 (winget). A new shell is needed for them to be on PATH. `.gitignore` excludes `Ander/`, `Data/`, all data, model and result file types, and the PDFs. The local `.git/hooks/pre-commit` refuses data paths and files over 5 MB, even with `git add -f`.
- GPU: NVIDIA RTX 5080 with 16 GB, driver 610.88. It is a Blackwell card (sm_120), so PyTorch must be a CUDA 12.8+ build, such as the cu128 wheels.
- CPU: i7-12700KF (12 cores, 20 threads). RAM: 32 GB.
- Ander's environment, for compatibility: Python 3.13.5, pandas 3.0.1, numpy 2.4.2, pyarrow 23.0.1 and PyTorch 2.11.0+cu128. He ran on an RTX 4060 laptop (8 GB), where a full rolling run took about 6 minutes per model and seed.

## 7. Working rules

- **No look-ahead:** weights for day t use information up to t−1 only. This includes the scaling statistics (training window only) and every model choice: never pick one by looking at test-year results (1998 onward). We run no validation (§9); if a choice ever has to be revisited, compare on 1996–1997 only.
- Results, tables and figures are produced by code. Each run fixes its seeds and saves its config next to its results.
- Every choice the paper does not specify gets a row in §9 with its reason.
- Code, docs and conversation are in English.

## 8. Status

- [x] Data (Ander, 2026-10-13): universe 1990–2025 and the 39 characteristics, checked against Pelger's panel. 33 of the 35 monthly characteristics match his values in 98–100% of stock-months, and Resid_Var and Beta match in rank. X matrices have no look-ahead. The schemas in `Ander\data` were checked against his docs on 2026-11-12.
- [ ] Get Ander's code and results. Ask for `src/`, `tests/`, `notebooks/`, `docs/implementation_decisions.md`, `docs/paper_targets.md`, `requirements.txt` and `results/runs/`.
    - His `CLAUDE.md` says the models, the training, the Table 2 and 3 runs and the 2022–2025 extension exist on his machine (2026-11-12).
    - But it records no numbers, its "differences with the paper" section is an unfilled placeholder, and its run listing shows some runs still in progress.
- [x] Environment (2026-11-12): Python 3.13 venv with PyTorch cu128 and the data/test stack (§6). Local git repo (`main`) with a data-excluding `.gitignore` and a pre-commit guard.
- [ ] GitHub remote: waiting for `gh auth login`, the repo name and its visibility (private recommended); then the first commit and push.
- [ ] Panel loader: X window + excess returns + membership mask → day × stock tensors (training window, test year, 30-day lookback).
- [ ] Model, with unit tests: attention factors (softmax masked to the day's universe), ridge loadings, residuals, LongConv, weights, costs and objective.
- [ ] Rolling out-of-sample 1998–2021: K = 8 and 30 first (3 seeds), then 1, 3, 5, 15 and 100.
- [ ] Metrics and comparison with Table 2, including turnover and short exposure.
- [ ] Afterwards: Table 3, Figures 2–4, PCA + LongConv, the 2022–2025 extension, the report notebook and the Phase II and III presentations.

**Next step:** authenticate `gh` and push the repo; ask Ander for his code and results.

## 9. Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-12 | No validation step. The paper tunes on 1990–95 / 1996–97; we do not. The first window is fitted once on 1990–1997 | Table 4 gives every hyperparameter, and Ander already settled the choices the paper leaves open. Re-running the search would add nothing for a reproduction |
| 2026-11-12 | Adopt Ander's open choices as listed in §4: annualized Sharpe in the objective, batches of 126 consecutive days, λ_ridge = 1e-5, his LongConv block, daily residuals in %, each day's own projection for lagged residuals, ω^port = 0 under 30 days in the universe, per-stock excess returns, medians and rf standardized on the training window, turnover paid at the year boundary, 3 seeds | His log records validation on 1990–95 / 1996–97 for the Sharpe scale, batching, λ_ridge and residual input, and arguments for the rest. Caveats: the lagged-residual projection is inferred from his docs, and his case for excess returns cites the paper's EW-market row (a 1998–2021 number, though not a strategy result). Check both against his code |

## 10. Results

| Run | Period | SR | SR_net | Paper (SR / net) | File |
| --- | --- | --- | --- | --- | --- |

_None yet._

## 11. Open questions

- **For the professor:** the code or results of the earlier AFSA replication, and which universe it used.
- **Universe.** The paper's exact universe cannot be rebuilt (§3), so expect gaps against Table 2 that no code fix will close.
- **Outlier filter for the extension.** Addentax (ATXG) entered the top 500 in September 2022 after a post-IPO surge and lost 95% on its first day in the universe. Decide whether to add a price or liquidity filter; the paper has none.
- **Medians.** If the medians are inert under eq. (1), did the authors' code differ from the paper (e.g. a nonlinear embedding)? Ander's results may help tell.
- **Phase II date** (Lecture 22): confirm it.
