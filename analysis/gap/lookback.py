"""Look-back rules for walk-forward (CLAUDE.md section 7, the row of 2026-11-15 written before it ran).

Walk-forward re-chooses a candidate every January from the candidates' records before that year. The current rule
scores a candidate on every test year so far; the user's proposal scores it on a fixed look-back, so that the choice
can follow a market that changes. This script compares the rules on stage 2 of the scan on D (the 28 settings x 9
band widths at K = 8 and 30, and K chosen too) and, as a second check outside the decision, on the first scan
(`broad`, 24 settings x 9 band widths at each of the 7 K, and K chosen among all 7).

The object is the seed ensemble through a no-trade band: a deployment layer, never a Table 2 row. A candidate is a
K, a training setting and a band width; its record is its daily net return traded alone from cash on the first test
day, at the paper's costs (selection.paths). Each rule takes the validation choice in 1998, and the earlier candidate
on a tie (smaller K, earlier grid setting, narrower band):
- validation:               the best net Sharpe ratio on 1996-1997, kept for every year (the paper's rule)
- all years:                the best net Sharpe ratio over every test year before (the current walk-forward)
- last 8 years:             the best net Sharpe ratio over the 8 most recent test years
- fading memory:            the best weighted net Sharpe ratio over every test year before, each year weighted
                            0.5 ** (age / 8) with last year's age 0 (weighted mean over weighted standard deviation)
- recent return, full risk: the best mean net return over the 8 most recent test years, divided by the standard
                            deviation over every test year before
- hindsight:                the best net Sharpe ratio over 1998-2021, kept for every year (a diagnostic)
- last 4 years:             the best net Sharpe ratio over the 4 most recent test years. Added on 2026-11-16 at the
                            user's request, after the run: reported with the others, outside the decision below
The book is carried over from one year to the next, so a switch is traded and paid for (selection.realize).

Decision, fixed before the run: a rule beats *all years* only if its 1998-2021 net Sharpe ratio is higher at K = 8,
at K = 30 and with K chosen; no single year of 1998-2021, traded as *all years* instead, reverses any of the three;
and its 2022-2025 net Sharpe ratio is not lower in any of the three. A difference is real only if its paired bootstrap
interval (21-day blocks, 2,000 draws) excludes 0.

    python analysis/gap/lookback.py              # -> analysis/gap/LOOKBACK.md (CPU, a few minutes)
    python analysis/gap/lookback.py --no-broad   # dataset D only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection as S  # noqa: E402
from selection import HOLDOUT, SHOW, TEST, Candidate, Scan, cell, compare, describe, md, parse_set, parse_setting, sharpe, stats  # noqa: E402

OUT = Path(__file__).resolve().parent / "LOOKBACK.md"
OBJ = "ensemble + band"
WINDOW, HALF_LIFE, SHORT = 8, 8.0, 4                                         # years
WALK = ("all years", "last 8 years", "fading memory", "recent return, full risk")
GUARDS = WALK[1:]                                                            # the rules judged against *all years*
ADDED = ("last 4 years",)                                                    # added after the run (2026-11-16): outside the decision
RULES = ("validation",) + WALK + ADDED + ("hindsight",)
D_SET = ["universe=wrds_broad_buffer600", "fill=none", "entry_rule=history"]
D_GRID = tuple([f"{c}/{r}/30" for c in ("2.5", "5", "7.5", "10") for r in ("1e-5", "3e-5", "1e-4", "3e-4", "1e-3")]
               + [f"{c}/1e-4/{e}" for e in (20, 50) for c in ("2.5", "5", "7.5", "10")])   # stage 2, as in REPORT.md section 8
KEYS = ("SR", "mu", "sigma", "turnover", "SR_net", "SR_net 1998-05", "SR_net 2006-13", "SR_net 2014-21", "beta 2014-21", "SR_net 2022-25")


# ---------------------------------------------------------------------------------------------- the rules
def moments(nets: dict[Candidate, np.ndarray], years: np.ndarray) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    """Candidates, test years, and per year and candidate the number of days, the sum and the sum of squares of the
    daily net return [3, years, candidates]. `nets` maps each candidate to its daily net returns [T, 1] (a layer)."""
    cands = list(nets)
    if any(nets[c].shape[1] != 1 for c in cands):
        raise ValueError("the look-back rules score one series per candidate (a layer), not seeds")
    X = np.stack([nets[c][:, 0] for c in cands], axis=1)                     # [T, candidates]
    uniq = np.unique(years)
    mom = np.zeros((3, len(uniq), len(cands)))
    for j, y in enumerate(uniq):
        x = X[years == y]
        mom[0, j], mom[1, j], mom[2, j] = len(x), x.sum(0), (x * x).sum(0)
    return cands, uniq, mom


def _sharpe(n: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    var = (s2 - s1 * s1 / n) / (n - 1)
    return np.where(var > 0, (s1 / n) / np.sqrt(np.where(var > 0, var, 1.0)), -np.inf)


def score(mom: np.ndarray, j: int, rule: str) -> np.ndarray:
    """Each candidate's criterion at the start of the j-th test year, from the years before it (j >= 1)."""
    lo = max(0, j - WINDOW)
    if rule == "all years":
        return _sharpe(*mom[:, :j].sum(1))
    if rule == "last 8 years":
        return _sharpe(*mom[:, lo:j].sum(1))
    if rule == "last 4 years":
        return _sharpe(*mom[:, max(0, j - SHORT):j].sum(1))
    if rule == "fading memory":
        w = 0.5 ** ((j - 1 - np.arange(j)) / HALF_LIFE)                      # last year's weight 1
        n, s1, s2 = (mom[:, :j] * w[None, :, None]).sum(1)
        mean = s1 / n
        var = s2 / n - mean * mean
        return np.where(var > 0, mean / np.sqrt(np.where(var > 0, var, 1.0)), -np.inf)
    if rule == "recent return, full risk":
        n, s1, _ = mom[:, lo:j].sum(1)
        N, S1, S2 = mom[:, :j].sum(1)
        var = (S2 - S1 * S1 / N) / (N - 1)
        return np.where(var > 0, (s1 / n) / np.sqrt(np.where(var > 0, var, 1.0)), -np.inf)
    raise ValueError(f"unknown rule {rule!r}")


def make_plans(nets: dict[Candidate, np.ndarray], years: np.ndarray, start: Candidate, rules=WALK) -> dict[str, dict[int, Candidate]]:
    """For each walk-forward rule, the candidate of each test year; the first year takes `start`."""
    cands, uniq, mom = moments(nets, years)
    plans = {}
    for rule in rules:
        plan = {int(uniq[0]): start}
        for j in range(1, len(uniq)):
            plan[int(uniq[j])] = cands[int(np.argmax(score(mom, j, rule)))]
        plans[rule] = plan
    return plans


def make_plan(nets: dict[Candidate, np.ndarray], years: np.ndarray, start: Candidate, rule: str) -> dict[int, Candidate]:
    return make_plans(nets, years, start, (rule,))[rule]


# ---------------------------------------------------------------------------------------------- one set of candidates
def realized(scan: Scan, plan: dict[int, Candidate]) -> pd.DataFrame:
    f = scan.realize(plan, OBJ)[0]
    f["net"] = f["gross"] - S.COST_TRADE * f["turnover"] - S.COST_SHORT * f["short"]
    return f


def case(scan: Scan, Ks: list[int] | None) -> dict:
    """Plans and realized daily results of every rule for the candidates of one K (or of every K: Ks None)."""
    cands = scan.candidates(OBJ, Ks)
    test, val = scan.nets(OBJ, cands), scan.nets(OBJ, cands, validation=True)
    in_test = (scan.years >= TEST[0]) & (scan.years <= TEST[1])
    v, h = S.best(val), S.best({c: x[in_test] for c, x in test.items()})
    years = range(TEST[0], scan.last + 1)
    plans = {"validation": {y: v for y in years}, **make_plans(test, scan.years, v, WALK + ADDED), "hindsight": {y: h for y in years}}
    ref = scan.plans(OBJ, Ks)                                                # the plans behind SCAN_BUFFER_GRID.md and SCAN.md
    if (ref["validation"], ref["walk-forward"], ref["hindsight"]) != (plans["validation"], plans["all years"], plans["hindsight"]):
        raise AssertionError("the validation, all-years and hindsight plans must equal selection.py's")
    return {"plans": plans, "frames": {r: realized(scan, plans[r]) for r in RULES}, "with_k": Ks is None,
            "candidates": len(cands), "market": scan.market}


def period(f: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return f[(f.index.year >= lo) & (f.index.year <= hi)]


def switches(plan: dict[int, Candidate], last: int) -> int:
    years = sorted(plan)
    return sum(plan[y] != plan[y - 1] for y in years[1:] if y <= last)


def swaps(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, pd.Series]:
    """Net Sharpe ratio of `a` minus `b` on 1998-2021, and the same with each single year of `a` traded as `b`."""
    a, b = period(a, *TEST), period(b, *TEST)
    if not a.index.equals(b.index):
        raise ValueError("the two paths must cover the same days")
    x, z, yrs = a["net"].to_numpy(), b["net"].to_numpy(), a.index.year
    base = float(sharpe(z))
    out = {}
    for y in range(TEST[0], TEST[1] + 1):
        s = x.copy()
        s[yrs == y] = z[yrs == y]
        out[y] = float(sharpe(s)) - base
    return float(sharpe(x)) - base, pd.Series(out)


def detail(c: dict) -> pd.DataFrame:
    rows = {}
    for rule in RULES:
        f, plan = c["frames"][rule], c["plans"][rule]
        d = describe([f], c["market"])
        r = {"Chosen": S.summary_of(plan, c["with_k"]), "Switches to 2021": switches(plan, TEST[1]),
             "Switches to 2025": switches(plan, HOLDOUT[1]), **{SHOW[k]: d[k] for k in KEYS}, "SR net 1998–2025": stats(f)["SR_net"]}
        for other in ("all years", "validation"):
            if rule != other:
                g = c["frames"][other]
                r[f"Δ vs {other}, 1998–2021"] = cell(compare(period(f, *TEST), period(g, *TEST))["all"])
                r[f"Δ vs {other}, 2022–25"] = cell(compare(period(f, *HOLDOUT), period(g, *HOLDOUT))["all"])
        rows[rule] = r
    return pd.DataFrame(rows).T


def net_sr(c: dict, rule: str, lo: int, hi: int) -> float:
    return stats(period(c["frames"][rule], lo, hi))["SR_net"]


# ---------------------------------------------------------------------------------------------- tables
def decision(cases: dict[str, dict], rules=GUARDS) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows, verdict = {}, {}
    for rule in rules:
        r, ok = {}, True
        for name, c in cases.items():
            d, sw = swaps(c["frames"][rule], c["frames"]["all years"])
            late = net_sr(c, rule, *HOLDOUT) - net_sr(c, "all years", *HOLDOUT)
            reversing = sw[sw <= 0] if d > 0 else sw[sw >= 0]
            r[f"{name}: Δ 1998–2021"] = f"{d:+.2f}"
            r[f"{name}: one year reverses it"] = (f"no ({sw.min():+.2f} to {sw.max():+.2f})" if reversing.empty
                                                  else "yes: " + ", ".join(f"{y} ({v:+.2f})" for y, v in reversing.items()))
            r[f"{name}: Δ 2022–25"] = f"{late:+.2f}"
            ok &= d > 0 and reversing.empty and late >= 0
        r["Beats all years"] = "yes" if ok else "no"
        rows[rule], verdict[rule] = r, ok
    return pd.DataFrame(rows).T, verdict


def summary(cases: dict[str, dict], periods=((TEST, "1998–2021"), (HOLDOUT, "2022–25"), ((TEST[0], HOLDOUT[1]), "1998–2025"))) -> pd.DataFrame:
    return pd.DataFrame({f"{name}, {label}": {rule: net_sr(c, rule, *span) for rule in RULES}
                         for span, label in periods for name, c in cases.items()})


def differences(cases: dict[str, dict], periods=((TEST, "1998–2021"), (HOLDOUT, "2022–25"))) -> pd.DataFrame:
    rows = {}
    for rule in RULES:
        if rule == "all years":
            continue
        rows[rule] = {f"{name}, {label}": cell(compare(period(c["frames"][rule], *span), period(c["frames"]["all years"], *span))["all"])
                      for span, label in periods for name, c in cases.items()}
    return pd.DataFrame(rows).T


def picks(c: dict) -> pd.DataFrame:
    return pd.DataFrame({rule: {y: p.code(c["with_k"]) for y, p in c["plans"][rule].items()} for rule in WALK + ADDED})


def write(d_cases: dict[str, dict], b_cases: dict[str, dict] | None, out: Path) -> dict[str, bool]:
    table, verdict = decision(d_cases, GUARDS + ADDED)
    passed = [r for r in GUARDS if verdict[r]]
    parts = [
        "# Look-back rules for walk-forward",
        "Generated by `analysis/gap/lookback.py`. The design is CLAUDE.md §7 (the row of 2026-11-15, written before it ran); "
        "the reading is REPORT.md §10.8. Object: the seed ensemble (seeds 0–2) through a no-trade band, a deployment layer, never a "
        "Table 2 row. A candidate is a K, a training setting and a band width; its record is its daily net return traded alone from "
        "cash on 1998-01-02. Every result is scored at the paper's costs (5 bp per unit of turnover, 1 bp per unit short); Sharpe "
        "ratios are annualized, on stock-by-stock excess returns.",
        "- **Rules.** Each January a rule scores every candidate on its record before that year and trades the best. 1998 takes the "
        "validation choice, ties go to the earlier candidate, and the book is carried over, so a switch is traded and paid for. "
        "*validation*: the best on 1996–1997, kept (the paper's rule). *all years*: the best net Sharpe ratio since 1998 (the current "
        "walk-forward). *last 8 years*: the best over the 8 most recent test years. *fading memory*: the best weighted net Sharpe ratio "
        "since 1998, each year weighted 0.5^(age/8), last year's age 0. *recent return, full risk*: the best mean net return over the "
        "last 8 years divided by the standard deviation since 1998. *hindsight*: the best over 1998–2021, kept (a diagnostic). The two "
        "rules with an 8-year window equal *all years* until 2006. *last 4 years*: the best over the 4 most recent test years; it "
        "equals *all years* until 2002. It was added on 2026-11-16 at the user's request, after the run, so it is reported with the "
        "others but is outside the decision.",
        "- **Decision, fixed before the run.** A rule beats *all years* only if its 1998–2021 net Sharpe ratio is higher at K = 8, at "
        "K = 30 and with K chosen; no single year of 1998–2021, traded as *all years* instead, reverses any of the three; and its "
        "2022–2025 net Sharpe ratio is not lower in any of the three. A difference is real only if its paired bootstrap interval "
        "(21-day blocks, 2,000 draws) excludes 0.",
        "- **Check.** The *validation*, *all years* and *hindsight* plans equal `selection.py`'s (asserted), so those rows reproduce "
        "`SCAN_BUFFER_GRID.md`" + (" and, in the second check, `SCAN.md`." if b_cases else "."),
        "## Decision: dataset D, stage 2 of the scan (K = 8 and 30: 28 settings × 9 band widths)",
        "**" + ("No rule beats *all years*." if not passed else f"Beats *all years*: {', '.join(passed)}.") + "** "
        "*Δ*: the rule's net Sharpe ratio minus *all years'*. *One year reverses it*: the 1998–2021 Δ with each single year traded "
        "as *all years* instead (its range when no year reverses the sign). The row of *last 4 years*, added after the run, applies "
        "the same test outside the decision.",
        md(table),
        "## Net Sharpe ratio of each rule, dataset D", md(summary(d_cases)),
        "## Paired differences from *all years*, dataset D", md(differences(d_cases)),
    ]
    for name, c in d_cases.items():
        parts += [f"### {name} ({c['candidates']} candidates)", md(detail(c)), "**Each year's choice**", md(picks(c), "Year")]
    if b_cases:
        t1 = summary(b_cases, ((TEST, "1998–2021"),))
        t2 = summary(b_cases, ((HOLDOUT, "2022–25"),))
        counts = []
        for rule in GUARDS + ADDED:
            n1 = sum(t1.loc[rule].iloc[i] > t1.loc["all years"].iloc[i] for i in range(t1.shape[1]))
            n2 = sum(t2.loc[rule].iloc[i] >= t2.loc["all years"].iloc[i] for i in range(t2.shape[1]))
            counts.append(f"*{rule}*: higher on 1998–2021 in {n1} of {t1.shape[1]}, not lower on 2022–2025 in {n2} of {t2.shape[1]}")
        parts += ["## Second check: the first scan (`broad`, Ander's arrays, 30-day wait; 24 settings × 9 band widths at each K)",
                  "Outside the decision: the same rules, unchanged. Against *all years*: " + "; ".join(counts) + ".",
                  "**Net Sharpe ratio, 1998–2021**", md(t1.set_axis([c.split(",")[0] for c in t1.columns], axis=1)),
                  "**Net Sharpe ratio, 2022–2025**", md(t2.set_axis([c.split(",")[0] for c in t2.columns], axis=1)),
                  "**Paired differences from *all years*, 1998–2021**",
                  md(differences(b_cases, ((TEST, "1998–2021"),)).rename(columns=lambda c: c.split(",")[0]))]
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-broad", action="store_true", help="skip the second check on the first scan")
    p.add_argument("--out", type=Path, default=OUT, help="markdown file [default: analysis/gap/LOOKBACK.md]")
    args = p.parse_args()
    t0 = time.time()
    d_scan = Scan([8, 30], tuple(parse_setting(g) for g in D_GRID), parse_set(D_SET))
    d_cases = {}
    for name, Ks in (("K = 8", [8]), ("K = 30", [30]), ("K chosen", None)):
        d_cases[name] = case(d_scan, Ks)
        print(f"D, {name}: done ({time.time() - t0:.0f} s)", flush=True)
    b_cases = None
    if not args.no_broad:
        b_scan = Scan([1, 3, 5, 8, 15, 30, 100])
        b_cases = {}
        for name, Ks in [(f"K = {K}", [K]) for K in b_scan.Ks] + [("K chosen", None)]:
            b_cases[name] = case(b_scan, Ks)
            print(f"broad, {name}: done ({time.time() - t0:.0f} s)", flush=True)
    verdict = write(d_cases, b_cases, args.out)
    print(f"written: {args.out} ({time.time() - t0:.0f} s); beats all years: {[r for r, ok in verdict.items() if ok] or 'none'}")


if __name__ == "__main__":
    main()
