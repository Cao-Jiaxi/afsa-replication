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
  Two cases: the cost constant, or the cost moving by era as our D model's does (it falls as turnover falls).
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
        for e, (a, b) in ERAS.items():
            s = t.loc[a:b]
            mu, sig = s["mu"].mean(), np.sqrt(s["var"].mean())
            prof = avg * cost_era[e] / cost_era["1998–2021"]
            era_rows[(f"paper K = {K}", e)] = {"gross μ %": mu, "σ %": sig, "gross SR": mu / sig, "net SR, constant cost": (mu - avg) / sig,
                                               "net SR, cost moving like ours": (mu - prof) / sig}
    for name, d in mine.items():
        for e, (a, b) in ERAS.items():
            x = d[(d.index.year >= a) & (d.index.year <= b)]
            era_rows[(f"ours: {name}", e)] = {"gross μ %": x.gross.mean() * 25200, "σ %": x.gross.std() * np.sqrt(252) * 100, "gross SR": sr(x.gross),
                                              "net SR, constant cost": sr(x.net), "net SR, cost moving like ours": np.nan}
    et = pd.DataFrame(era_rows).T
    et.index = [f"{a}, {b}" for a, b in et.index]
    et.index.name = "run, period"
    text += ["## 2. By era\n",
             "For the paper, the net columns subtract its average cost (Table 2's μ − μ_net, 7.14% a year at K = 30), either constant or scaled by era "
             f"as our reference model's cost moves ({', '.join(f'{e}: {v:.1f}%' for e, v in cost_era.items())} a year). "
             "Ours are measured; for them both net columns would be the same, so only the first is filled.\n", md(et), ""]

    # 3. by year, K = 30
    t = paper[30]
    avg = PAPER_TABLE2[30][1] - PAPER_TABLE2[30][4]
    yr = pd.DataFrame({"paper gross μ %": t["mu"], "paper σ %": t["sigma"], "paper gross SR": t["mu"] / t["sigma"],
                       "paper net SR, constant cost": (t["mu"] - avg) / t["sigma"]})
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
    text += ["## 4. What it shows\n",
             f"- **The paper's 2.28 is a 1998–2006 result.** Its estimated net SR is {p30['1998–2005']['net SR, constant cost']:.2f} on 1998–2005, "
             f"{p30['2006–2013']['net SR, constant cost']:.2f} on 2006–2013 and {num(p30['2014–2021']['net SR, constant cost'])} on 2014–2021 "
             f"({num(p30['2014–2021']['net SR, cost moving like ours'])} if its cost fell as ours did).",
             f"- **The whole gap to the paper is in 2001–2006.** Our edge broke in spring 2001 (EXTENSION.md); the paper's held to about 2006, at 14–19% gross a year "
             "with 1–2% volatility. From 2007 on, our hindsight objects are level with the paper's estimated net or above it.",
             f"- **Our best fixed ensemble + band matches the paper's 2.28 over 1998–2021 with less gross return** ({o30['1998–2021']['gross μ %']:.1f}% against "
             f"{p30['1998–2021']['gross μ %']:.1f}%), because it trades less and at lower σ. It is a deployment layer chosen in hindsight, not the paper's object.",
             "- **Caveat.** The paper's yearly cost is unknown; the two cost cases bracket it only if its turnover moved the way ours did.\n"]
    OUT.write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
