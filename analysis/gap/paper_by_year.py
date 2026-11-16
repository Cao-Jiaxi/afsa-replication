"""The paper's results by year and by era, read back from its Figure 2, next to ours on dataset D (K = 30).

    python analysis/gap/paper_by_year.py            # -> analysis/gap/PAPER_BY_YEAR.md (CPU, about 2 minutes)

The paper reports net Sharpe ratios only for 1998-2021 as a whole (Table 2). Figure 2 holds its gross growth day by
day (fig2_digitize.py reads the vector paths: 5,880 days, 245 to a year), so by year:
- gross return is exact: log growth over the year's 245 days, plus sigma^2 / 2 to make it an arithmetic mean like
  Table 2's;
- volatility comes from 10-day blocks within each year. Matplotlib dropped about half the points as invisible and the
  digitizer filled them by interpolation, which smooths single days (1-day returns understate sigma); blocks whose
  ends are within a drawing tolerance do not. The check: the years pooled give Table 2's sigma;
- net return needs the year's cost, which the paper does not give. Only the average is known: mu - mu_net in Table 2.
  Three cases: the cost constant; the cost moving by era as our D model's does (it falls as turnover falls); or the
  cost proportional to each year's gross return, every year paying the same share of it (42.9% at K = 30). The final
  deck uses the proportional case since 2026-11-16 (the user's decision); section 5 says what it requires of turnover.
Ours: the D runs at K = 30 (buffered list, rebuilt history), each traded from cash at the paper's costs over all of
each year's days: the reference model (seeds pooled), the best fixed model in hindsight (10 bp, ridge 3e-4) and the
best fixed ensemble + band in hindsight (5 bp, ridge 1e-4, band 1.2e-3; SCAN_BUFFER_GRID.md). Hindsight settings
are chosen on 1998-2021 itself: upper bounds, not out-of-sample results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import deploy  # noqa: E402
from fig2_digitize import OUT as FIG2  # noqa: E402

from afsa.config import RESULTS_DIR, Config  # noqa: E402
from afsa.metrics import PAPER_TABLE2  # noqa: E402

DAYS, BLOCK = 245, 10                                         # the figure's days a year; block length for sigma
ERAS = {"1998–2005": (1998, 2005), "2006–2013": (2006, 2013), "2014–2021": (2014, 2021), "1998–2021": (1998, 2021)}
D = {"universe": "wrds_broad_buffer600", "fill": "none", "entry_rule": "history"}
OUT = Path(__file__).with_name("PAPER_BY_YEAR.md")


def sr(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252))


def paper_after_costs(K: int = 30, how: str = "proportional", era_cost: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Figure 2's growth of one unit before costs (index 0: the start of 1998; 245 days a year) and the estimated path
    after costs. Table 2 gives only the average cost (mu - mu_net), taken off as a cost per figure day (a year's % / 252,
    as Table 2's annualization counts days), spread over the years in one of three ways:
      constant      the same every day;
      proportional  every year pays the same share of its own gross return: each day of year y pays share x the year's
                    mean daily gross return, share = the average cost / the mean daily gross return (42.9% at K = 30);
      ours          each era pays the average scaled as our reference model's cost moves (era_cost: era -> % a year).
    All three take the same total cost; they differ in when it is paid."""
    paper = np.r_[1.0, pd.read_csv(FIG2)[f"K={K}"].to_numpy()]
    r = paper[1:] / paper[:-1] - 1
    year = 1998 + np.arange(len(r)) // DAYS
    avg = (PAPER_TABLE2[K][1] - PAPER_TABLE2[K][4]) / 100 / 252
    if how == "constant":
        cost = np.full(len(r), avg)
    elif how == "proportional":
        cost = avg / r.mean() * pd.Series(r).groupby(year).transform("mean").to_numpy()
    elif how == "ours":
        era = {y: e for e, (a, b) in ERAS.items() if e != "1998–2021" for y in range(a, b + 1)}
        cost = avg * np.array([era_cost[era[y]] for y in year]) / era_cost["1998–2021"]
    else:
        raise ValueError(f"unknown cost path {how!r}")
    return paper, np.r_[1.0, np.cumprod(1 + r - cost)]


def paper_years(K: int) -> pd.DataFrame:
    """Gross mu (arithmetic, % a year), sigma (% a year) and gross SR of each of the 24 figure years."""
    lc = np.log(np.r_[1.0, pd.read_csv(FIG2)[f"K={K}"].to_numpy()])
    rows = {}
    for y in range(24):
        seg = lc[y * DAYS:(y + 1) * DAYS + 1]
        var = np.diff(seg[::BLOCK]).var(ddof=1) * 252 / BLOCK
        mu = (seg[-1] - seg[0]) / DAYS * 252 + var / 2
        rows[1998 + y] = {"mu": mu * 100, "var": var * 1e4}
    t = pd.DataFrame(rows).T
    t["sigma"] = np.sqrt(t["var"])
    return t


def sigma_by_block(K: int) -> dict[int, float]:
    lc = np.log(np.r_[1.0, pd.read_csv(FIG2)[f"K={K}"].to_numpy()])
    return {h: float(np.sqrt(np.mean([np.diff(lc[y * DAYS:(y + 1) * DAYS + 1][::h]).var(ddof=1) for y in range(24)]) * 252 / h) * 100)
            for h in (1, 2, 5, 10, 21)}


def ours() -> dict[str, pd.DataFrame]:
    run = RESULTS_DIR / "runs" / Config(K=30, ridge=1e-4, **D).name()
    _, dates, R, member, Ws = deploy.load(run, 1998, 2021)
    out = {"reference model (5 bp, ridge 1e-4), seeds pooled": sum(deploy.trade(W, R, dates) for W in Ws) / len(Ws),
           "ensemble + band, hindsight (5 bp, ridge 1e-4, band 1.2e-3)": deploy.trade(deploy.band_path(deploy.ensemble(Ws), member, 1.2e-3), R, dates)}
    run = RESULTS_DIR / "runs" / Config(K=30, ridge=3e-4, cost_trade=1e-3, **D).name()
    _, dates, R, _, Ws = deploy.load(run, 1998, 2021)
    out["model, hindsight (10 bp, ridge 3e-4), seeds pooled"] = sum(deploy.trade(W, R, dates) for W in Ws) / len(Ws)
    return out


def num(v: float) -> str:
    return f"{v:+.2f}".replace("-", "−")


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else fmt.format(v).replace("-", "−")          # a typographic minus on numbers only
        return str(v)
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |")
    return "\n".join(lines)


def main() -> None:
    paper = {K: paper_years(K) for K in (8, 30, 100)}
    mine = ours()
    ref = mine["reference model (5 bp, ridge 1e-4), seeds pooled"]
    cost_year = (ref.gross - ref.net).groupby(ref.index.year).mean() * 25200            # % a year
    cost_era = {e: float(cost_year.loc[a:b].mean()) for e, (a, b) in ERAS.items()}
    text = [f"# The paper by year, from its Figure 2, against ours (K = 30)\n\nWritten by `analysis/gap/paper_by_year.py`; how each number is made is in its docstring. "
            "Table 2 gives net Sharpe ratios only for 1998–2021 as a whole; everything by year or era below is rebuilt from Figure 2 (gross) "
            "and, for the net figures, an assumption about how the paper's cost moved over time.\n"]

    # 1. the check against Table 2
    rows = {}
    for K, t in paper.items():
        mu, sig = t["mu"].mean(), np.sqrt(t["var"].mean())
        p = dict(zip(("SR", "mu", "sigma", "SR_net", "mu_net", "sigma_net", "beta"), PAPER_TABLE2[K]))
        rows[f"K = {K}"] = {"μ rebuilt": mu, "μ Table 2": p["mu"], "σ rebuilt": sig, "σ Table 2": p["sigma"], "SR rebuilt": mu / sig, "SR Table 2": p["SR"],
                            "net SR rebuilt (constant cost)": (mu - (p["mu"] - p["mu_net"])) / sig, "net SR Table 2": p["SR_net"]}
    chk = pd.DataFrame(rows).T
    chk.index.name = "1998–2021"
    blocks = sigma_by_block(30)
    dev = (chk["σ rebuilt"] / chk["σ Table 2"] - 1).abs() * 100
    text += ["## 1. How well the rebuilt figures match Table 2\n",
             "Gross μ and σ in % a year. σ is from 10-day blocks within each figure year, pooled over the 24 years. μ is exact. "
             f"σ is within {dev['K = 30']:.0f}% of Table 2's at K = 30 and within {dev.max():.0f}% at K = 8 and 100, which moves those Sharpe ratios by up to "
             f"{(chk['SR rebuilt'] - chk['SR Table 2']).abs().max():.1f}: read the era and year figures below to about that precision.\n", md(chk), "",
             "σ of K = 30 by block length (Table 2: 4.20%): " + ", ".join(f"{h} days {v:.2f}%" for h, v in blocks.items())
             + ". Single days understate it because the digitizer interpolated the points matplotlib dropped; 5 and 10 days agree.\n"]

    # 2. by era
    era_rows = {}
    for K in (8, 30, 100):
        t, p = paper[K], dict(zip(("SR", "mu", "sigma", "SR_net", "mu_net"), PAPER_TABLE2[K][:5]))
        avg = p["mu"] - p["mu_net"]
        share = avg / t["mu"].mean()                                                     # the proportional path's share
        for e, (a, b) in ERAS.items():
            s = t.loc[a:b]
            mu, sig = s["mu"].mean(), np.sqrt(s["var"].mean())
            prof = avg * cost_era[e] / cost_era["1998–2021"]
            era_rows[(f"paper K = {K}", e)] = {"gross μ %": mu, "σ %": sig, "gross SR": mu / sig, "net SR, constant cost": (mu - avg) / sig,
                                               "net SR, cost moving like ours": (mu - prof) / sig,
                                               "net SR, cost proportional to return": (1 - share) * mu / sig}
    for name, d in mine.items():
        for e, (a, b) in ERAS.items():
            x = d[(d.index.year >= a) & (d.index.year <= b)]
            era_rows[(f"ours: {name}", e)] = {"gross μ %": x.gross.mean() * 25200, "σ %": x.gross.std() * np.sqrt(252) * 100, "gross SR": sr(x.gross),
                                              "net SR, constant cost": sr(x.net), "net SR, cost moving like ours": np.nan,
                                              "net SR, cost proportional to return": np.nan}
    et = pd.DataFrame(era_rows).T
    et.index = [f"{a}, {b}" for a, b in et.index]
    et.index.name = "run, period"
    text += ["## 2. By era\n",
             "For the paper, the net columns subtract its average cost (Table 2's μ − μ_net, 7.14% a year at K = 30): constant, scaled by era "
             f"as our reference model's cost moves ({', '.join(f'{e}: {v:.1f}%' for e, v in cost_era.items())} a year), or in proportion "
             "to each year's gross return (every year pays the same share of it; the final deck's case). "
             "Ours are measured; for them the three net columns would be the same, so only the first is filled.\n", md(et), ""]

    # 3. by year, K = 30
    t = paper[30]
    avg = PAPER_TABLE2[30][1] - PAPER_TABLE2[30][4]
    share30 = avg / t["mu"].mean()
    yr = pd.DataFrame({"paper gross μ %": t["mu"], "paper σ %": t["sigma"], "paper gross SR": t["mu"] / t["sigma"],
                       "paper net SR, constant cost": (t["mu"] - avg) / t["sigma"],
                       "paper net SR, cost proportional to return": (1 - share30) * t["mu"] / t["sigma"],
                       "paper cost proportional to return, % a year": share30 * t["mu"]})
    for name, short in zip(mine, ("ours, reference model", "ours, ensemble + band (hindsight)", "ours, model (hindsight)")):
        d = mine[name]
        yr[f"{short}: net SR"] = d["net"].groupby(d.index.year).apply(sr)
    yr["our reference model's cost, % a year"] = cost_year
    yr.index.name = "year"
    text += ["## 3. By year, K = 30\n",
             "A Sharpe ratio measured over one year has a standard error of about 1 (more when it is high), and one year's σ from about 24 blocks is "
             "uncertain by about ±15%. The paper's 1998–2001 are the least reliable: the curve sits at a level of 1–5 there, where a drawing tolerance "
             "is a large share of a day's move.\n", md(yr), ""]

    # 4. reading
    p30 = {e: et.loc[f"paper K = 30, {e}"] for e in ERAS}
    o30 = {e: et.loc[f"ours: ensemble + band, hindsight (5 bp, ridge 1e-4, band 1.2e-3), {e}"] for e in ERAS}
    era3 = ("1998–2005", "2006–2013", "2014–2021")
    by = lambda d, col: " / ".join(num(d[e][col]) for e in era3)                         # noqa: E731
    text += ["## 4. What it shows\n",
             "- **The paper's 2.28 leans on its early years.** Its estimated net SR on 1998–2005 / 2006–2013 / 2014–2021 is "
             f"{by(p30, 'net SR, constant cost')} with a constant cost, {by(p30, 'net SR, cost moving like ours')} if its cost fell as ours did, "
             f"and {by(p30, 'net SR, cost proportional to return')} with a cost proportional to each year's return (the final deck's case). "
             f"Before costs, which needs no assumption: {by(p30, 'gross SR')}.",
             "- **Where the gap to the paper sits depends on the cost path.** Before costs, our edge broke in spring 2001 (EXTENSION.md) while the "
             "paper's held to about 2006, at 14–19% gross a year with 1–2% volatility. Against our best fixed ensemble + band in hindsight "
             f"({by(o30, 'net SR, constant cost')}): with a constant cost the paper leads only in 1998–2005, and from 2007 on our hindsight objects "
             "are level with it or above; with a cost proportional to return it leads only in 2006–2013.",
             f"- **Our best fixed ensemble + band matches the paper's 2.28 over 1998–2021 with less gross return** ({o30['1998–2021']['gross μ %']:.1f}% against "
             f"{p30['1998–2021']['gross μ %']:.1f}%), because it trades less and at lower σ. It is a deployment layer chosen in hindsight, not the paper's object.",
             "- **Caveat.** The paper's yearly cost is unknown. The constant and proportional paths are the two ends: costs come from trading, "
             "not from returns, so a proportional cost needs the paper's turnover to fall with its return (section 5).\n"]

    # 5. the after-cost curve under each cost path (K = 30)
    paths = {"constant": "the same cost every day", "ours": "cost moving like ours", "proportional": "cost proportional to each year's return"}
    rows = {}
    for how, label in paths.items():
        v, net = paper_after_costs(30, how, cost_era)
        r = v[1:] / v[:-1] - 1
        cost = (1 + r) - net[1:] / net[:-1]                                                 # each figure day's cost
        cyear = pd.Series(cost).groupby(1998 + np.arange(len(r)) // DAYS).mean() * 25200     # % a year, as Table 2 counts
        k10, k14, peak = 12 * DAYS, 16 * DAYS, int(np.argmax(net))
        rows[label] = {"start of 2010": net[k10], "peak": net[peak], "peak year": str(1998 + peak // DAYS), "end of 2021": net[-1],
                       "change 2010–2021, %": 100 * (net[-1] / net[k10] - 1), "change 2014–2021, %": 100 * (net[-1] / net[k14] - 1),
                       **{f"cost {e}, % a year": float(cyear.loc[a:b].mean()) for e, (a, b) in ERAS.items() if e != "1998–2021"},
                       "lowest year's cost, % a year": float(cyear.min())}
    pt = pd.DataFrame(rows).T
    pt.index.name = "cost path"
    short_fee = 0.0001 * 0.5 * 252 * 100                                                    # 1 bp a day on half the book, % a year
    per_turnover = 0.0005 * 252 * 100                                                       # % a year per unit of daily turnover
    prop = pt.loc[paths["proportional"]]
    late, low = float(prop["cost 2014–2021, % a year"]), float(prop["lowest year's cost, % a year"])
    implied = (avg - short_fee) / per_turnover
    text += ["## 5. The paper's curve after costs, K = 30, under each cost path\n",
             "Growth of one unit from the start of 1998, on Figure 2's 245 days a year (the curve `fig2_ours.py` draws). The three paths take the same "
             "total cost, so they end at the same value; they differ in when it is paid. The final deck draws the proportional path (2026-11-16, the "
             "user's decision).\n", md(pt), "",
             f"What the proportional path requires: at 5 bp per unit traded and 1 bp per unit short, with half the book short (a short fee of "
             f"{short_fee:.2f}% a year), its 2014–2021 cost of {late:.2f}% a year leaves a turnover of about {(late - short_fee) / per_turnover:.2f} "
             f"a day, against {implied:.2f} implied by Table 2's average cost; its lowest year's cost, {low:.2f}% a year, is "
             f"{'below' if low < short_fee else 'above'} the short fee alone.\n"]
    OUT.write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
