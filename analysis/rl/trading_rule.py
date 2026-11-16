"""The learned trading rule against the fixed band (CLAUDE.md section 7, rows of 2026-11-16; TRADING_RULE.md).

    python analysis/rl/trading_rule.py      -> analysis/rl/TRADING_RULE_RESULTS.md (CPU, a few minutes)

Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0-2. Every
object is one path per period: from cash on the period's first day (1998 for the test years, 1996 for validation),
with the book carried across years, scored at the paper's costs.

Objects:
  rule                               the model trained with the learned rule: its target through its own dials
  rule's target                      that target traded as it is (a diagnostic)
  rule's targets + fixed band        that target through the fixed band (δ = 4e-4, η = 1): what the dials add
  fixed band trained                 the model trained through the fixed band (BAND_TRAINED.md), through that band
  D model + band after               D's stored model, with the fixed band added after training
  rule ensemble                      the seeds' targets averaged and rescaled, their dials averaged, then the rule
  fixed band trained, ensemble + band, and D ensemble + band after: the same for the comparators

The pre-registered decision compares the rule with the fixed band trained inside the model, on 1996-1997 only, for
the model and for the ensemble. The test years are reported and never decide.

Train first (GPU, about 10 minutes); the commands are in TRADING_RULE.md. The fixed band's runs and D's weights come
from analysis/gap/band_trained.py's runs and datasets.py's runs.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "gap"))
import band_trained as bt  # noqa: E402
from deploy import band_path, ensemble, market, seed_weights, trade  # noqa: E402
from paired_test import block_draws, sharpe  # noqa: E402

from afsa.metrics import stitch  # noqa: E402

DELTA = 4e-4                                         # the fixed band: D's validation choice, and the rule's starting width
HERE = Path(__file__).resolve().parent
PERIODS = (("1996–97", None), ("1998–2005", (1998, 2005)), ("2006–13", (2006, 2013)), ("2014–21", (2014, 2021)),
           ("2022–25", (2022, 2025)))


def period(first: int, last: int, train_years: int) -> tuple[dict[str, list[pd.DataFrame]], dict]:
    """object -> daily frames (one per seed, or one for an ensemble) on first..last, and the rule's dials and books."""
    base = dataclasses.replace(bt.BASE, train_years=train_years)
    rule, fixed = dataclasses.replace(base, trade_head=True), dataclasses.replace(base, band=DELTA)
    dates, permnos, R, member = market(base, first, last)
    seeds = sorted(bt.run_dir(base).glob("seed*"), key=lambda p: int(p.name[4:]))
    W_d = [seed_weights(d, dates, permnos) for d in seeds]
    W_f = [bt.saved(bt.run_dir(fixed) / d.name, dates, permnos, "weights") for d in seeds]
    rule_seeds = [bt.run_dir(rule) / d.name for d in seeds]
    W_r, D_r, E_r = ([bt.saved(d, dates, permnos, key) for d in rule_seeds] for key in ("weights", "delta", "eta"))
    for d, W in zip(seeds, W_d):                                             # the inputs reproduce the stored results
        bt.check(stitch(d), trade(W, R, dates), d)
    for d in rule_seeds:
        bt.check(stitch(d), trade(bt.saved(d, dates, permnos, "book"), R, dates), d)

    def band(W):
        return band_path(W, member, DELTA, rescale="moved")

    def rule_books(W, dl, et):
        return band_path(W, member, dl, rescale="moved", eta=et)

    H_r = [rule_books(W, dl, et) for W, dl, et in zip(W_r, D_r, E_r)]
    H_f = [band(W) for W in W_f]
    objects = {
        "rule": [trade(H, R, dates) for H in H_r],
        "rule's target": [trade(W, R, dates) for W in W_r],
        "rule's targets + fixed band": [trade(band(W), R, dates) for W in W_r],
        "fixed band trained": [trade(H, R, dates) for H in H_f],
        "D model + band after": [trade(band(W), R, dates) for W in W_d],
        "rule ensemble": [trade(rule_books(ensemble(W_r), sum(D_r) / len(D_r), sum(E_r) / len(E_r)), R, dates)],
        "fixed band trained, ensemble + band": [trade(band(ensemble(W_f)), R, dates)],
        "D ensemble + band after": [trade(band(ensemble(W_d)), R, dates)],
    }
    return objects, {"dates": dates, "member": member, "W": W_r, "delta": D_r, "eta": E_r, "H": H_r, "H_fixed": H_f}


def interval(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Sharpe difference of two daily net series and its 95% paired circular block bootstrap interval."""
    a, b = x.to_numpy(), y.to_numpy()
    idx = block_draws(len(a))
    lo, hi = np.percentile(sharpe(a[idx]) - sharpe(b[idx]), [2.5, 97.5])
    return float(sharpe(a) - sharpe(b)), float(lo), float(hi)


def decision(val: dict) -> tuple[str, list[str]]:
    """The pre-registered verdict (CLAUDE.md section 7): the rule against the fixed band trained inside the model."""
    lines, signs = [], []
    for a, b in (("rule", "fixed band trained"), ("rule ensemble", "fixed band trained, ensemble + band")):
        d, lo, hi = interval(bt.pooled(val[a]), bt.pooled(val[b]))
        signs.append(1 if lo > 0 else -1 if hi < 0 else 0)
        lines.append(f"{a} − {b}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    verdict = ("better" if max(signs) == 1 and min(signs) >= 0 else
               "worse" if min(signs) == -1 and max(signs) <= 0 else "inconclusive")
    return verdict, lines


def spearman_by_day(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean over days of the rank correlation of a and b across the stocks in mask."""
    A = pd.DataFrame(np.where(mask, a, np.nan)).rank(axis=1)
    B = pd.DataFrame(np.where(mask, b, np.nan)).rank(axis=1)
    A, B = A.sub(A.mean(axis=1), axis=0), B.sub(B.mean(axis=1), axis=0)
    r = (A * B).sum(axis=1) / np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    return float(r[mask.sum(1) >= 10].mean())


def dials(info: dict, years: tuple[int, int] | None) -> dict[str, float]:
    """The rule's band widths and speeds on member stock-days with a learned dial (members without a signal keep the
    fixed band, stored as exactly 4e-4), how often it leaves a held position untouched, and what the width follows."""
    y = pd.DatetimeIndex(info["dates"]).year.to_numpy()
    keep = np.ones(len(y), dtype=bool) if years is None else (y >= years[0]) & (y <= years[1])
    member = info["member"][keep]
    d_all, e_all, rows = [], [], {"δ ρ target size": [], "δ ρ target change": [], "η ρ target change": [],
                                  "untouched": [], "untouched, fixed band": []}
    for W, dl, et, H, Hf in zip(info["W"], info["delta"], info["eta"], info["H"], info["H_fixed"]):
        W, dl, et, H, Hf = W[keep], dl[keep], et[keep], H[keep], Hf[keep]
        learned = member & (dl != float(np.float32(DELTA)))
        d_all.append(dl[learned]), e_all.append(et[learned])
        change = pd.DataFrame(np.abs(np.diff(W, axis=0, prepend=W[:1]))).rolling(20, min_periods=10).mean().to_numpy()
        rows["δ ρ target size"].append(spearman_by_day(dl, np.abs(W), learned))
        rows["δ ρ target change"].append(spearman_by_day(dl, change, learned & np.isfinite(change)))
        rows["η ρ target change"].append(spearman_by_day(et, change, learned & np.isfinite(change)))
        held = member[1:] & member[:-1] & (H[:-1] != 0)
        rows["untouched"].append(float((H[1:] == H[:-1])[held].mean()))
        held_f = member[1:] & member[:-1] & (Hf[:-1] != 0)
        rows["untouched, fixed band"].append(float((Hf[1:] == Hf[:-1])[held_f].mean()))
    d, e = np.concatenate(d_all) * 1e4, np.concatenate(e_all)                # δ in bp of the book
    return {"δ p10": np.percentile(d, 10), "δ median": np.median(d), "δ p90": np.percentile(d, 90),
            "η p10": np.percentile(e, 10), "η median": np.median(e), "η p90": np.percentile(e, 90),
            **{k: float(np.mean(v)) for k, v in rows.items()}}


def main() -> None:
    test, info_t = period(bt.TEST[0], bt.HOLDOUT[1], 8)
    val, info_v = period(*bt.VALIDATION, 6)
    verdict, decided = decision(val)
    levels = bt.table(test, val)
    pairs = [("rule", "fixed band trained"), ("rule ensemble", "fixed band trained, ensemble + band"),
             ("rule", "rule's targets + fixed band"), ("rule", "D model + band after"),
             ("rule ensemble", "D ensemble + band after")]
    c = bt.contrasts(test, val, pairs)
    dl = pd.DataFrame({name: dials(info_v if s is None else info_t, s) for name, s in PERIODS}).T
    lines = [
        "# The learned trading rule: results",
        "",
        "Written by `analysis/rl/trading_rule.py`, as pre-registered in CLAUDE.md §7 (row of 2026-11-16; the rule itself "
        "is in [TRADING_RULE.md](TRADING_RULE.md)). Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, "
        "ridge 1e-4, 30 epochs, seeds 0–2. Test years 1998–2021 and 2022–2025 use 8-year windows; the validation years "
        "1996–1997 use 6-year windows. Every path starts from cash on its period's first day and carries its book across "
        "years, scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).",
        "",
        "## Decision (pre-registered: 1996–1997 only, against the fixed band trained inside the model)",
        "",
        f"**{verdict.capitalize()}.**",
        "",
        *[f"- {x}" for x in decided],
        "",
        "Pooled seed-mean net series for the models; one series for each ensemble. Paired circular block bootstrap "
        "(21-day blocks, 2,000 draws, the same days for both), 95% interval.",
        "",
        "## Levels",
        "",
        "Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; μ_net and σ in "
        "% a year.",
        "",
        bt.md(levels),
        "## Contrasts",
        "",
        "Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws), 95% interval. Only the "
        "validation column decides; the others are reported.",
        "",
        bt.md(c),
        "## The dials",
        "",
        "Member stock-days with a signal, pooled over seeds; δ in basis points of the book (4 = the fixed band, 0.04%). "
        "ρ: the mean over days of the rank correlation across stocks, with the size of the stock's target (its absolute "
        "value), or with its mean absolute daily target change over the last 20 days. *Untouched*: the share of held positions (member on both "
        "days) that do not trade on a day, for the rule and for the fixed band trained inside the model.",
        "",
        bt.md(dl),
    ]
    out = HERE / "TRADING_RULE_RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")                                 # the Windows console's code page lacks μ and σ
    pd.set_option("display.width", 250)
    print(f"Decision: {verdict}", *decided, sep="\n")
    print(levels.round(2).to_string())
    print(c.to_string())
    print(dl.round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
