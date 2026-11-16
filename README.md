# Attention Factors for Statistical Arbitrage: a replication

Jiaxi Cao and Ander Ibarrondo. AMS 520 *Machine Learning in Quantitative Finance*, Stony Brook University, Fall 2026 (Prof. Paweł Polak).

We replicate Epstein, Wang, Choi and Pelger, *Attention Factors for Statistical Arbitrage* (ICAIF '25): attention-based factors learned from 39 firm characteristics, a LongConv network trading the residuals, and all of it trained jointly on the net Sharpe ratio. The target is the Attention + LongConv rows of the paper's Table 2 (out of sample 1998–2021), and the extension is 2022–2025.

## Slides

- [Phase I presentation](slides/Phase1/main.pdf), with its [source](slides/Phase1/) (report, speech text and speaker notes as LaTeX).
- [Final report presentation](slides/FinalReport/main.pdf), with its [source](slides/FinalReport/). It is a work in progress.

## Where we stand

- **Data.** Everything is rebuilt from our own WRDS downloads (CRSP and Compustat).
    - The universe is the monthly 500 largest NYSE/AMEX/NASDAQ securities. It equals Ander's original build on every stock-month of 1990–2025.
    - 33 of the 35 monthly characteristics equal the Chen-Pelger-Zhu panel in 98.7–100% of stock-months. The other two (Resid_Var, Beta) agree in rank.
    - Missing characteristics are left at rank 0, not carried forward.
- **Model.** The paper's model as specified, with each choice the paper leaves open documented in [CLAUDE.md §7](CLAUDE.md).
- **Results** at K = 30, as net Sharpe ratios on 1998–2021 after the paper's costs (the paper: 2.28).
    - Before costs we are close to the paper (gross SR 3.7 against 3.97). After costs our model is well below it (1.31 on dataset D, defined in the table below), because it trades more.
    - Averaging the seeds' portfolio weights and adding a no-trade band lifts this to 2.09 when the training setting and the band width are chosen on 1996–1997, as the paper chooses its hyperparameters, and to 1.93 when they are re-chosen each January from the years before. It reaches the paper's 2.28 only with the setting and band that turn out best on 1998–2021 itself, which no one could have picked in advance ([SCAN_BUFFER_GRID.md](analysis/gap/SCAN_BUFFER_GRID.md)). These are deployment layers, not Table 2 rows.
    - The gap to the paper is in 2001–2006. Rebuilt year by year from its Figure 2 (net of Table 2's average cost), the paper earned 14–19% a year before costs at 1–2% volatility until about 2006, while our edge broke in spring 2001. From 2007 on, our best result in hindsight is level with the paper's or above it ([PAPER_BY_YEAR.md](analysis/gap/PAPER_BY_YEAR.md)).
    - In 2022–2025 the model loses money after costs (−0.34). The ensemble with the band nets +0.13, but only thanks to the hedge's short positions in stocks too new to have a signal, three of which collapsed. With those positions set aside it nets −0.37, and none of the fixes we tried lifts it above zero ([EXTENSION.md](analysis/gap/EXTENSION.md)).
    - We tested designs from the course's reinforcement-learning lecture: a learned trading rule and two richer versions of it, attention across stocks, and pessimism where the seeds disagree. Each decision rule was written down before its test ran. All came out inconclusive or not kept except one: on fresh runs, the richest trading rule (SM) beat D's model with a band added after training on 1996–1997. SM is reported below as a deployment variant ([analysis/rl/](analysis/rl/)).
    - The main specification (training cost, ridge, universe, λ_Var) is still being settled.

K = 30, net Sharpe ratio, annualized, at the paper's costs (5 bp per unit of turnover, 1 bp per unit short). "Model" is the average over 3 seeds, which is what the paper's tables report. "Ensemble" averages the 3 seeds' portfolio weights. All runs use one setting: ridge 1e-4, trained at 5 bp, 30 epochs.

| Dataset | Model, 1998–2021 | Ensemble, 1998–2021 | Model, 2022–2025 | Ensemble, 2022–2025 |
| --- | --- | --- | --- | --- |
| A: top 500, 30-day wait for new members | 1.18 | 1.56 | −0.39 | −0.23 |
| B: top 500, rebuilt history for new members | 1.23 | 1.62 | −0.33 | −0.15 |
| C: buffered list (top 500 + incumbents to rank 600), 30-day wait | 1.22 | 1.61 | −0.42 | −0.28 |
| D: buffered list, rebuilt history | 1.31 | 1.70 | −0.34 | −0.18 |
| Paper, Table 2 | 2.28 | – | – | – |

Sources: [analysis/gap/DATASETS.md](analysis/gap/DATASETS.md) for this table; [analysis/gap/RESULTS.md](analysis/gap/RESULTS.md) for every run; [analysis/gap/REPORT.md](analysis/gap/REPORT.md) for why we differ from the paper; the scans in `analysis/gap/SCAN*.md`.

**A deployment variant, not a Table 2 row.** It is dataset D's model trained with a learned trading rule, from the course's reinforcement-learning material, traded as a seed ensemble. The rule sets each stock's no-trade band and trading speed every day from the stock's history, its position and the day's market. It trades 35–45% less than the plain ensemble with a fixed band added after training. Net Sharpe ratio, same costs, 1998–2021 unless the column says otherwise:

| K | Paper, Table 2 (model) | D model | D ensemble + band after | Learned rule (SM), ensemble, 1998–2021 | Learned rule (SM), ensemble, 2022–2025 |
| --- | --- | --- | --- | --- | --- |
| 8 | 1.94 | 1.52 | 2.13 | 1.95 | −0.67 |
| 30 (6 seeds) | 2.28 | 1.32 | 2.22 | 2.61 | −0.50 |
| 100 | 2.19 | 0.77 | 1.83 | 2.09 | −0.73 |

It beats the fixed band on 1998–2021 at K = 30 and 100, mostly in 1998–2005, and trails it in 2022–2025 at every K (D ensemble + band after: −0.50, +0.08, +0.36). An ensemble with a trading rule is not the paper's object, whose rows are seed-averaged models. See [analysis/rl/DEPLOYMENT.md](analysis/rl/DEPLOYMENT.md).

## Repository

| Path | What it holds |
| --- | --- |
| `afsa/` | The code. Each module opens with a docstring saying what it does |
| `tests/` | Unit tests (synthetic data; tests that need the data files are skipped without them). `tests/hardware/` has GPU timing tests |
| `analysis/gap/` | Scripts and reports on the gap to the paper, the hyperparameter scans and the dataset comparison |
| `analysis/characteristics/` | How each of the 39 characteristics is built from WRDS, and how it matches the Chen-Pelger-Zhu panel |
| `analysis/rl/` | The course's reinforcement-learning lecture condensed, how it applies to AFSA, and pre-registered tests of the designs built from it: a learned trading rule and its views of the book and the market, cross-stock attention, pessimism; the deployment variant |
| `slides/` | The Phase I and final presentations (Beamer, pdfLaTeX) |
| `CLAUDE.md` | The project guide: data, model, every open decision with its reason, status, all results and open questions. It also serves as instructions for the coding assistant |
| `NOTES.md` | Details behind the guide |

**No data is included.** The data come from CRSP and Compustat under Stony Brook's WRDS licence. To reproduce the results you need your own WRDS access; the commands below rebuild everything from it.

## Setup

Python 3.13 and an NVIDIA GPU. An RTX 50-series card needs a CUDA 12.8+ build of PyTorch.

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest tests -q
```

## Rebuilding the data and running

Run from the repository root (`python` = `.venv\Scripts\python.exe`). Each step's time is on an RTX 5080.

```
python -m afsa.wrds_db setup              # once: stores your WRDS login in your Postgres password file (never in the repo)
python -m afsa.wrds_universe download     # CRSP monthly file and delistings
python -m afsa.wrds_universe build        # the monthly top-500 lists
python -m afsa.wrds_data download         # daily CRSP, Compustat, links, Fama-French (~2 min)
python -m afsa.chars_market build         # 13 monthly market characteristics
python -m afsa.chars_accounting build     # 22 accounting characteristics
python -m afsa.build_wrds                 # model-ready arrays (dataset A)
python -m afsa.build_history              # month blocks for rebuilt history (dataset B)
python -m afsa.wrds_universe buffer       # the buffered list; then build_wrds and build_history with --universe wrds_broad_buffer600 (C, D)

# A rolling run: 8-year windows retrained every year, test years 1998-2025, 3 seeds (a few minutes)
python -m afsa.run --universe wrds_broad --fill none --entry_rule history --ridge 1e-4 --K 30 --first-year 1998 --last-year 2025
```

`python -m afsa.run --help` lists every option. Results go to `results/runs/<configuration>/`, with the configuration saved next to them.
