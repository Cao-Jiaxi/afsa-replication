"""The no-trade band trained inside the model, against the same band added after training (CLAUDE.md section 7).

    python analysis/gap/band_trained.py                  -> analysis/gap/BAND_TRAINED.md (CPU, about a minute)
    python analysis/gap/band_trained.py --delta 1.2e-3   -> analysis/gap/BAND_TRAINED_0.0012.md

Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0-2. The band is
delta = 4e-4 (the width D's validation years chose for the ensemble, DATASETS.md) or 1.2e-3 (the width hindsight chose
for D's ensemble + band at K = 30, SCAN_BUFFER_GRID.md: a diagnostic), eta = 1, with the movers-only rescale (only the
stocks that trade or are closed are resized; the others keep their position exactly).

Objects, each on one path per period from cash on its first day (1998-2025, scored by sub-period; validation 1996-1997):
  D model                  the stored model of datasets.py (its target is its portfolio)
  D model + band after     the same targets through the band (movers-only)
  trained model            the model trained with the band inside (`afsa.run --band 4e-4`): its book
  trained model's target   its target without the band (a diagnostic: what the network learned to aim at)
  ensembles                seed-averaged targets, rescaled, through the band; D's also with the stored results' rescale
Contrasts are paired circular block bootstraps (paired_test.py) of pooled seed-mean series on the same days.

Before running, train (GPU, about 7 minutes) with --save-weights, so the targets and books are in each <year>.npz:
    python -m afsa.run --universe wrds_broad_buffer600 --fill none --entry_rule history --ridge 1e-4 --band 4e-4 --K 30 --seeds 0 1 2 --first-year 1998 --last-year 2025 --save-weights
    python -m afsa.run ... the same ... --train_years 6 --first-year 1996 --last-year 1997 --save-weights
D's weights (<year>_w.npz) come from datasets.py's runs (REPORT.md section 8).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy import band_path, ensemble, market, seed_weights, trade  # noqa: E402
from paired_test import block_draws, sharpe  # noqa: E402
from variants import stats  # noqa: E402

from afsa.config import RESULTS_DIR, Config  # noqa: E402
from afsa.metrics import stitch  # noqa: E402

ORIGIN = {4e-4: "D's validation choice for the ensemble band",
          1.2e-3: "the width hindsight chose for D's ensemble + band at K = 30 in the scan's stage 2 (the best fixed band over "
                  "1998–2021, `SCAN_BUFFER_GRID.md`). It was chosen on the test years, so this is a diagnostic against "
                  "hindsight, not a selection (CLAUDE.md §8)"}
BASE = Config(universe="wrds_broad_buffer600", fill="none", entry_rule="history", ridge=1e-4, K=30)
TEST, HOLDOUT, VALIDATION = (1998, 2021), (2022, 2025), (1996, 1997)
SPANS = (("1998–2021", TEST), ("1998–2005", (1998, 2005)), ("2006–13", (2006, 2013)), ("2014–21", (2014, 2021)),
         ("2022–25", HOLDOUT), ("1998–2025", (1998, 2025)))
HERE = Path(__file__).resolve().parent


def run_dir(cfg: Config) -> Path:
    return RESULTS_DIR / "runs" / cfg.name()


def saved(seed_dir: Path, dates: np.ndarray, permnos: np.ndarray, key: str) -> np.ndarray:
    """Daily `weights` (the target) or `book` [T, S] of one seed of a run saved with --save-weights, from <year>.npz."""
    W = np.zeros((len(dates), len(permnos)))
    for year in np.unique(dates.astype("datetime64[Y]").astype(int) + 1970):
        z = np.load(seed_dir / f"{year}.npz")
        rows = np.searchsorted(dates, z["dates"].astype("datetime64[D]"))
        col = np.minimum(np.searchsorted(permnos, z["permnos"]), len(permnos) - 1)
        known = permnos[col] == z["permnos"]
        if np.abs(z[key][:, ~known]).max(initial=0.0) > 0:
            raise ValueError(f"{seed_dir / str(year)}: weight on a stock outside the universe")
        W[np.ix_(rows, col[known])] = z[key][:, known]
    return W


def period(first: int, last: int, train_years: int, delta: float) -> dict[str, list[pd.DataFrame]]:
    """object -> daily frames (one per seed, or one for an ensemble) on first..last, each path from cash on day one."""
    base = dataclasses.replace(BASE, train_years=train_years)
    new = dataclasses.replace(base, band=delta)
    dates, permnos, R, member = market(base, first, last)
    seeds = sorted(run_dir(base).glob("seed*"), key=lambda p: int(p.name[4:]))
    W_d = [seed_weights(d, dates, permnos) for d in seeds]
    new_seeds = [run_dir(new) / d.name for d in seeds]
    W_n = [saved(d, dates, permnos, "weights") for d in new_seeds]
    for d, W in zip(seeds, W_d):                                             # the inputs reproduce the stored results
        check(stitch(d), trade(W, R, dates), d)
    for d in new_seeds:
        check(stitch(d), trade(saved(d, dates, permnos, "book"), R, dates), d)
    band = lambda W, rescale="moved": band_path(W, member, delta, rescale=rescale)   # noqa: E731
    return {
        "D model": [trade(W, R, dates) for W in W_d],
        "D model + band after": [trade(band(W), R, dates) for W in W_d],
        "trained model's target": [trade(W, R, dates) for W in W_n],
        "trained model": [trade(band(W), R, dates) for W in W_n],
        "D ensemble": [trade(ensemble(W_d), R, dates)],
        "D ensemble + band after (stored rescale)": [trade(band(ensemble(W_d), "all"), R, dates)],
        "D ensemble + band after": [trade(band(ensemble(W_d)), R, dates)],
        "trained ensemble + band": [trade(band(ensemble(W_n)), R, dates)],
    }


def check(stored: pd.DataFrame, mine: pd.DataFrame, where: Path) -> None:
    stored = stored.loc[mine.index]
    gap = max(float(np.abs(mine["gross"] - stored["gross"]).max()), float(np.abs(mine["turnover"] - stored["turnover"])[1:].max()))
    if gap > 2e-5:
        raise ValueError(f"{where}: the saved weights do not reproduce the stored results (largest difference {gap:.1e})")


def span(f: pd.DataFrame | pd.Series, s: tuple[int, int]) -> pd.DataFrame | pd.Series:
    return f[(f.index.year >= s[0]) & (f.index.year <= s[1])]


def pooled(frames: list[pd.DataFrame]) -> pd.Series:
    return sum(f["net"] for f in frames) / len(frames)


def table(test: dict, val: dict) -> pd.DataFrame:
    rows = {}
    for name, frames in test.items():
        m = lambda s, k: float(np.mean([stats(span(f, s))[k] for f in frames]))   # noqa: E731
        rows[name] = {"Validation SR_net": float(np.mean([stats(f)["SR_net"] for f in val[name]])),
                      "SR": m(TEST, "SR"), "SR_net": m(TEST, "SR_net"), "μ_net": m(TEST, "mu_net"), "σ": m(TEST, "sigma"),
                      "Turnover": m(TEST, "turnover"), "Short": m(TEST, "short"),
                      **{f"SR_net {n}": m(s, "SR_net") for n, s in SPANS[1:]}}
    return pd.DataFrame(rows).T


def contrasts(test: dict, val: dict, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for a, b in pairs:
        cells = {}
        for name, s in (("Validation 1996–97", None),) + SPANS:
            x, y = ((pooled(val[k]) if s is None else span(pooled(test[k]), s)).to_numpy() for k in (a, b))
            idx = block_draws(len(x))
            d = sharpe(x[idx]) - sharpe(y[idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            cells[name] = f"{float(sharpe(x) - sharpe(y)):+.2f} [{lo:+.2f}, {hi:+.2f}]"
        rows[f"{a} − {b}"] = cells
    return pd.DataFrame(rows).T


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    head = "| | " + " | ".join(df.columns) + " |\n|" + " --- |" * (len(df.columns) + 1) + "\n"
    body = "".join("| " + str(i) + " | " + " | ".join(v if isinstance(v, str) else fmt.format(v) for v in r) + " |\n"
                   for i, r in zip(df.index, df.to_numpy(dtype=object)))
    return head + body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delta", type=float, default=4e-4, choices=sorted(ORIGIN), help="band width [default: 4e-4]")
    delta = p.parse_args().delta
    out = HERE / ("BAND_TRAINED.md" if delta == 4e-4 else f"BAND_TRAINED_{delta:g}.md")
    test = period(TEST[0], HOLDOUT[1], 8, delta)
    val = period(*VALIDATION, 6, delta)
    t = table(test, val)
    stored = [stats(span(stitch(run_dir(dataclasses.replace(BASE, band=delta)) / f"seed{s}"), TEST))["SR_net"] for s in range(3)]
    pairs = [("trained model", "D model + band after"), ("trained model", "D model"), ("D model + band after", "D model"),
             ("trained ensemble + band", "D ensemble + band after")]
    c = contrasts(test, val, pairs)
    lines = [
        "# The band trained inside the model",
        "",
        "Written by `analysis/gap/band_trained.py` (CLAUDE.md §7). Dataset D (buffered list, rebuilt history), K = 30, "
        "training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0–2; test years 1998–2021 and 2022–2025 with 8-year windows, "
        f"validation 1996–1997 with 6-year windows. Band δ = {delta:g} ({ORIGIN[delta]}), η = 1, "
        "movers-only rescale: only stocks that trade or are closed are resized.",
        "",
        "*Trained model*: the network is scored during training on the book that the band holds, so it is charged only for "
        "trades that happen. *D model + band after*: D's stored model, which never saw the band, with the same band applied "
        "to its targets. Every path starts from cash on the first day of its period (1998 and 1996) and carries the book "
        "across years. Scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).",
        "",
        "## Levels",
        "",
        "Model rows are seed averages; ensemble rows are one portfolio (the seeds' targets averaged, rescaled, then the "
        "band). Columns are 1998–2021 unless named; μ_net and σ in % a year.",
        "",
        md(t),
        f"The run's own stored results restart the book from its target at each year's start; their 1998–2021 SR_net, "
        f"{np.mean(stored):.2f}, differs from the continuous book's by {np.mean(stored) - t.loc['trained model', 'SR_net']:+.2f}.",
        "",
        "## Contrasts",
        "",
        "Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws, the same days for both), "
        "95% interval.",
        "",
        md(c),
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")                                 # the Windows console's code page lacks μ and σ
    pd.set_option("display.width", 250)
    print(t.round(2).to_string())
    print(c.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
