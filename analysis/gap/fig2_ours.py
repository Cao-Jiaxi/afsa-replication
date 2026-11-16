"""Figure 2 of the paper, redrawn for our K = 30 ensemble + band on dataset D, beside the paper's K = 30 curves.

    python analysis/gap/fig2_ours.py                    # -> analysis/gap/fig2_ours_K030.png, 1998-2025 (CPU, about a minute)
    python analysis/gap/fig2_ours.py --last-year 2021   # the paper's years only -> fig2_ours_K030_2021.png
    python analysis/gap/fig2_ours.py --deck             # the final report's two figures (CPU, about two minutes):
        slides/FinalReport/assets/fig2_K030.png   the paper before and after costs, beside ours by hindsight
        slides/FinalReport/assets/net_K030.png    after costs only: the paper, ours by walk-forward and by hindsight, and
                                                  the band of 4e-4 on D's reference setting, added after or in the loss

Ours: dataset D (buffered list, rebuilt history), K = 30, the seed ensemble (seeds 0-2) through a no-trade band: a
deployment layer, never a Table 2 row. The candidates are stage 2 of the scan on D, 28 training settings x 9 band
widths (SCAN_BUFFER_GRID.md), and two rules pick among them, exactly as selection.py and lookback.py do:
- hindsight: the best fixed candidate over 1998-2021 (5 bp, ridge 1e-4, 30 epochs, band 1.2e-3), kept through 2025.
  It is chosen on 1998-2021 itself, so it is an upper bound there; 2022-2025 is out of sample for it.
- walk-forward: each January, the candidate with the best net Sharpe ratio over every test year before (1998 takes
  the validation choice). It uses no future data but chooses from test years (CLAUDE.md section 8).
Both are traded from cash on 1998-01-02 at the paper's costs over every trading day; the book is carried from one
year to the next, so a switch of candidate is traded and paid for. The script checks both against the net Sharpe
ratios stored in SCAN_BUFFER_GRID.md before it draws.

The paper: its Figure 2 curve for K = 30 (fig2_digitize.py), growth of one unit before costs, 245 days a year. Its
after-cost line is an estimate: the paper gives no daily cost, only the average in Table 2 (mu - mu_net, 7.14% a
year at K = 30). Since 2026-11-16 (the user's decision) it is spread over the years in proportion to each year's gross
return, every year paying the same share of it, 42.9% (paper_by_year.paper_after_costs, "proportional"); until then it
was taken off every day evenly. PAPER_BY_YEAR.md section 5 compares the cost paths: they end at the same value.

The deck's figures (2026-11-16, the user's requests): fig2_K030.png shows the paper and ours by hindsight only;
net_K030.png shows every curve after costs, with two more of ours from band_trained.py: the seed ensemble of D's
reference setting (5 bp, ridge 1e-4, 30 epochs) through the band of 4e-4 that D's validation years chose, added after
training or trained inside the model (movers-only rescale, as in BAND_TRAINED.md), from cash on 1998-01-02. The script
checks those two against the net Sharpe ratios stored in BAND_TRAINED.md as well.
Every curve is drawn against the year: day k of a year with n days sits at year + k / n, so a year spans one tick.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fig2_digitize import OUT as FIG2  # noqa: E402
from lookback import D_GRID, D_SET, OBJ, realized  # noqa: E402
from paper_by_year import paper_after_costs  # noqa: E402
from selection import HOLDOUT, TEST, Scan, parse_set, parse_setting  # noqa: E402

from afsa.metrics import PAPER_TABLE2  # noqa: E402

PAPER_DAYS = 245                                              # the figure's days a year (fig2_digitize.py)
PAPER_COST = "proportional"                                   # how the paper's average cost is spread (paper_by_year.py)
STORED = {"hindsight": (2.28, 0.13), "walk-forward": (1.93, 0.13)}   # SCAN_BUFFER_GRID.md, K = 30: 1998-2021, 2022-2025
COLORS = {"hindsight": "tab:blue", "walk-forward": "tab:orange"}
HERE = Path(__file__).parent
# The deck's version: its text width (114 mm, a 4:3 Beamer frame less 7 mm margins), its text colour (Ink) for every
# label, and line colours that pass the dataviz palette checks on white (CVD, normal vision, contrast): a lighter SBU
# red (the deck's 990000 sits just below the lightness band), the deck's AttentionBlue and a dark orange.
DECK_OUT = HERE.parents[1] / "slides" / "FinalReport" / "assets" / "fig2_K030.png"
DECK_NET_OUT = DECK_OUT.with_name("net_K030.png")
DECK_SIZE_MM = (114, 52)
INK = "#262626"
# The after-cost figure adds the band of 4e-4 on D's reference setting, in one more colour (teal), solid when the band
# is added after training and dashed when it is in the loss: the four colours pass the palette checks on all pairs.
DECK_COLORS = {"paper": "#A61C1C", "hindsight": "#235FAA", "walk-forward": "#C77700", "band": "#0E8C6A"}
BAND_DELTA = 4e-4
BAND_STORED = {"band added after": (2.10, 0.10), "band in the loss": (2.17, -0.13)}   # BAND_TRAINED.md, ensembles + band


def sr(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252))


def num(v: float, sign: bool = False) -> str:
    return (f"{v:+.2f}" if sign else f"{v:.2f}").replace("-", "−")


def span(f: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return f[(f.index.year >= lo) & (f.index.year <= hi)]


def year_axis(index: pd.DatetimeIndex) -> np.ndarray:
    """Day k (1-based) of a year with n trading days -> year + k / n, so a year's last day lands on the next tick."""
    s = pd.Series(1, index=index)
    return index.year.to_numpy() + s.groupby(index.year).cumsum().to_numpy() / s.groupby(index.year).transform("size").to_numpy()


def ours() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    scan = Scan([30], tuple(parse_setting(g) for g in D_GRID), parse_set(D_SET))
    plans = scan.plans(OBJ, [30])
    frames = {rule: realized(scan, plans[rule]) for rule in STORED}
    for rule, (test, holdout) in STORED.items():
        got = (sr(span(frames[rule].net, *TEST)), sr(span(frames[rule].net, *HOLDOUT)))
        if abs(got[0] - test) > 0.006 or abs(got[1] - holdout) > 0.006:
            raise ValueError(f"{rule}: net SR {got[0]:.3f} / {got[1]:.3f}, not the stored {test} / {holdout}")
    return frames, plans


def bands() -> dict[str, pd.DataFrame]:
    """D's reference setting, seed ensemble, through the band of 4e-4: added after training, or trained in the loss
    (band_trained.py's objects, one path from cash on 1998-01-02), checked against BAND_TRAINED.md."""
    from band_trained import period
    objects = period(TEST[0], HOLDOUT[1], 8, BAND_DELTA)
    frames = {"band added after": objects["D ensemble + band after"][0], "band in the loss": objects["trained ensemble + band"][0]}
    for name, (test, holdout) in BAND_STORED.items():
        got = (sr(span(frames[name].net, *TEST)), sr(span(frames[name].net, *HOLDOUT)))
        if abs(got[0] - test) > 0.006 or abs(got[1] - holdout) > 0.006:
            raise ValueError(f"{name}: net SR {got[0]:.3f} / {got[1]:.3f}, not the stored {test} / {holdout}")
    return frames


def label_ends(ax, curves: list[tuple[np.ndarray, np.ndarray]], fontsize: float, color: str, dx: float = 0.45,
               gap_share: float = 0.05) -> None:
    """The end value of each curve, in text ink: right of the curves that reach the last year, spread apart; above-left
    otherwise."""
    x_end = max(xs[-1] for xs, _ in curves)
    gap = gap_share * max(ys.max() for _, ys in curves)
    right = sorted(ys[-1] for xs, ys in curves if xs[-1] >= x_end - 1e-9)
    pos = list(right)
    for _ in range(20):
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
    for y, py in zip(right, pos):
        ax.text(x_end + dx, py, f"{y:.1f}", va="center", fontsize=fontsize, color=color)
    for xs, ys in curves:
        if xs[-1] < x_end - 1e-9:
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(-3, 6), textcoords="offset points", ha="right", va="bottom",
                        fontsize=fontsize, color=color)


def deck_style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 6.5,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": INK, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.0, "ytick.major.size": 2.0, "legend.fontsize": 6.2})


def deck_axes(ax, ymax: float, ystep: float, ylabel: str) -> None:
    """What the deck's two figures share: the extension shaded and named, the year ticks, the grid."""
    ax.axvspan(HOLDOUT[0], HOLDOUT[1] + 1, color="#EDEDED", lw=0, zorder=0)
    ax.text((HOLDOUT[0] + HOLDOUT[1] + 1) / 2, 0.97, "Extension\n2022–2025", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.2, color=INK, linespacing=1.15)
    ticks = np.arange(TEST[0], HOLDOUT[1] + 2, 2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(TEST[0] - 0.4, HOLDOUT[1] + 2.6)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax - 1e-9, ystep))                    # every step below the top edge
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#E2E2E2", lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def deck(frames: dict[str, pd.DataFrame], xp: np.ndarray, paper: np.ndarray, paper_net: np.ndarray) -> None:
    """The final report's version (slides/FinalReport, the part on the paper's Figure 2), drawn at the size it is shown:
    no title or notes, since the frame carries them; colour for whose curve, solid before costs and dashed after; the
    Sharpe ratios of 1998-2021 in the legend (the paper's from Table 2), the end values at the ends. Ours by hindsight
    only (the user's request, 2026-11-16: walk-forward is in the after-cost figure)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    frames = {"hindsight": frames["hindsight"]}
    deck_style()
    fig, ax = plt.subplots(figsize=tuple(v / 25.4 for v in DECK_SIZE_MM), dpi=300)
    lw, dash = 1.1, (0, (3.6, 1.6))
    ax.plot(xp, paper, color=DECK_COLORS["paper"], lw=lw)
    ax.plot(xp, paper_net, color=DECK_COLORS["paper"], lw=lw, ls=dash)
    curves = [(xp, paper), (xp, paper_net)]
    srs = {"paper": (PAPER_TABLE2[30][0], PAPER_TABLE2[30][3])}
    ends_2021 = {}
    for rule, f in frames.items():
        f = span(f, TEST[0], HOLDOUT[1])
        x = np.r_[1998.0, year_axis(f.index)]
        for col, ls in (("gross", "-"), ("net", dash)):
            y = np.r_[1.0, (1 + f[col]).cumprod().to_numpy()]
            ax.plot(x, y, color=DECK_COLORS[rule], lw=lw, ls=ls)
            curves.append((x, y))
            ends_2021[rule, col] = float((1 + span(f, *TEST)[col]).prod())
        srs[rule] = (sr(span(f, *TEST).gross), sr(span(f, *TEST).net))
    label_ends(ax, curves, 6.2, INK, dx=0.35, gap_share=0.045)

    names = {"paper": "Paper", "hindsight": "Ours, hindsight", "walk-forward": "Ours, walk-forward"}
    who = ax.legend(handles=[Patch(color=DECK_COLORS[k], label=f"{names[k]}: {num(a)} / {num(b)}") for k, (a, b) in srs.items()],
                    title="Sharpe ratio 1998–2021,\nbefore / after costs", loc="upper left", bbox_to_anchor=(0.005, 0.995),
                    frameon=False, handlelength=1.2, handleheight=0.7, title_fontsize=6.2, alignment="left",
                    borderaxespad=0.2)
    ax.add_artist(who)
    ax.legend(handles=[Line2D([], [], color=INK, lw=lw, label="Before costs"),
                       Line2D([], [], color=INK, lw=lw, ls=dash, label="After costs")],
              loc="upper left", bbox_to_anchor=(0.43, 0.995), frameon=False, handlelength=2.6, borderaxespad=0.2)

    deck_axes(ax, 54, 10, "Cumulative return (growth of 1)")
    fig.tight_layout(pad=0.25)
    fig.savefig(DECK_OUT)

    k10 = 12 * PAPER_DAYS                                     # the figure's first day of 2010
    peak = int(np.argmax(paper_net))
    print(f"paper before costs: {paper[k10]:.1f} at the start of 2010, {paper[-1]:.1f} at the end of 2021")
    print(f"paper after costs: {paper_net[k10]:.1f} at the start of 2010, peak {paper_net[peak]:.1f} in "
          f"{1998 + peak // PAPER_DAYS}, {paper_net[-1]:.1f} at the end of 2021")
    for (rule, col), v in ends_2021.items():
        print(f"ours, {rule}, {col}: {v:.2f} at the end of 2021")
    print(f"wrote {DECK_OUT}")


def deck_net(frames: dict[str, pd.DataFrame], banded: dict[str, pd.DataFrame], xp: np.ndarray, paper_net: np.ndarray) -> None:
    """The deck's after-cost figure (slides/FinalReport, the conclusions): every curve after costs, at the size it is
    shown. Colour for whose curve; the band of 4e-4 on D's reference setting is one colour, solid when added after
    training and dashed when in the loss (the frame's source line gives the width). Net Sharpe ratios of 1998-2021 and 2022-2025 in the legend (the paper's from
    Table 2), the end values at the ends."""
    from matplotlib.lines import Line2D

    deck_style()
    fig, ax = plt.subplots(figsize=tuple(v / 25.4 for v in DECK_SIZE_MM), dpi=300)
    lw, dash = 1.1, (0, (3.6, 1.6))
    ax.plot(xp, paper_net, color=DECK_COLORS["paper"], lw=lw)
    curves = [(xp, paper_net)]
    handles = [Line2D([], [], color=DECK_COLORS["paper"], lw=lw, label=f"Paper, estimated: {num(PAPER_TABLE2[30][3])}")]
    series = (("walk-forward", "Walk-forward", frames["walk-forward"], "walk-forward", "-"),
              ("hindsight", "Hindsight", frames["hindsight"], "hindsight", "-"),
              ("band added after", "Band added after", banded["band added after"], "band", "-"),
              ("band in the loss", "Band in the loss", banded["band in the loss"], "band", dash))
    ends = {}
    for key, label, f, colour, ls in series:
        f = span(f, TEST[0], HOLDOUT[1])
        x = np.r_[1998.0, year_axis(f.index)]
        y = np.r_[1.0, (1 + f.net).cumprod().to_numpy()]
        ax.plot(x, y, color=DECK_COLORS[colour], lw=lw, ls=ls)
        curves.append((x, y))
        a, b = sr(span(f, *TEST).net), sr(span(f, *HOLDOUT).net)
        handles.append(Line2D([], [], color=DECK_COLORS[colour], lw=lw, ls=ls, label=f"{label}: {num(a)} / {num(b, sign=True)}"))
        ends[key] = (float((1 + span(f, *TEST).net).prod()), float(y[-1]), float(y.max()), float(x[int(np.argmax(y))]))
    label_ends(ax, curves, 6.2, INK, dx=0.35, gap_share=0.045)
    ax.legend(handles=handles, title="Net Sharpe ratio, 1998–2021 / 2022–2025", loc="upper left",
              bbox_to_anchor=(0.005, 0.995), frameon=False, handlelength=2.6, title_fontsize=6.2, alignment="left",
              borderaxespad=0.2)
    top = max(c[1].max() for c in curves)                                # room above the curves for the legend
    deck_axes(ax, float(np.ceil(top * 1.45 / 2) * 2), 2, "Cumulative return after costs (growth of 1)")
    fig.tight_layout(pad=0.25)
    fig.savefig(DECK_NET_OUT)

    k10 = 12 * PAPER_DAYS
    peak = int(np.argmax(paper_net))
    print(f"paper after costs: {paper_net[k10]:.2f} at the start of 2010, peak {paper_net[peak]:.2f} at "
          f"{xp[peak]:.2f}, {paper_net[-1]:.2f} at the end of 2021")
    for key, (e21, e25, top_y, when) in ends.items():
        print(f"{key}: {e21:.2f} at the end of 2021, {e25:.2f} at the end of 2025; peak {top_y:.2f} at {when:.2f}")
    print(f"wrote {DECK_NET_OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-year", type=int, default=HOLDOUT[1], choices=(TEST[1], HOLDOUT[1]))
    ap.add_argument("--deck", action="store_true", help="draw the final report's version instead (1998-2025)")
    args = ap.parse_args()
    last = args.last_year

    frames, plans = ours()
    paper, paper_net = paper_after_costs(30, PAPER_COST)
    if not np.allclose(paper[1:], pd.read_csv(FIG2)["K=30"].to_numpy()):
        raise ValueError("paper_after_costs must start from the digitized Figure 2 curve")
    xp = 1998 + np.arange(len(paper)) / PAPER_DAYS
    p_sr, p_mu, p_sr_net, p_mu_net = (PAPER_TABLE2[30][i] for i in (0, 1, 3, 4))
    if args.deck:
        deck(frames, xp, paper, paper_net)
        deck_net(frames, bands(), xp, paper_net)
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=200)
    curves = [(xp, paper), (xp, paper_net)]
    ax.plot(xp, paper, color="tab:brown", lw=1.5, label=f"Paper, before costs (Figure 2; SR {num(p_sr)})")
    ax.plot(xp, paper_net, color="tab:brown", lw=1.5, ls="--", label=f"Paper, after costs (estimated; net SR {num(p_sr_net)})")
    for rule, f in frames.items():
        f = span(f, TEST[0], last)
        x = np.r_[1998.0, year_axis(f.index)]
        for col, ls, what in (("gross", "-", "before costs (SR"), ("net", "--", "after costs (net SR")):
            y = np.r_[1.0, (1 + f[col]).cumprod().to_numpy()]
            text = f"Ours, {rule}, {what} {num(sr(span(f, *TEST)[col]))}"
            if last > TEST[1]:
                text += f"; 2022–25: {num(sr(span(f, *HOLDOUT)[col]), sign=True)}"
            ax.plot(x, y, color=COLORS[rule], lw=1.5, ls=ls, label=text + ")")
            curves.append((x, y))

    label_ends(ax, curves, 8, "0.15")
    if last > TEST[1]:
        ax.axvspan(HOLDOUT[0], last + 1, color="0.92", zorder=0)
        ax.text((HOLDOUT[0] + last + 1) / 2, 0.985, "Extension\n2022–2025", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="0.25")

    ticks = np.arange(TEST[0], last + 2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_xlim(TEST[0] - 0.5, last + 1.9)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, ls=":", color="0.6", lw=0.6)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax.set_title("Cumulative Returns of Arbitrage Portfolios, K = 30", fontsize=11, fontweight="bold")

    wf = {y: c for y, c in plans["walk-forward"].items() if y <= last}
    years = sorted(wf)
    switches = sum(wf[y] != wf[y - 1] for y in years[1:])
    fig.text(0.01, 0.005,
             "Ours: buffered list, rebuilt history; seed ensemble (0–2) through a no-trade band, at the paper's costs. Candidates: 28 training settings "
             "× 9 band widths.\nHindsight: the best fixed candidate over 1998–2021 (5 bp, ridge 1e-4, 30 epochs, band 1.2e-3), kept through "
             f"{last}.\nWalk-forward: each January the best net SR over all earlier years (1998: the validation choice); "
             f"{len(set(wf.values()))} candidates, {switches} switches to {last}, each paid for.\nThe paper's after-cost line takes Table 2's "
             f"average cost ({p_mu - p_mu_net:.2f}% a year) in proportion to each year's gross return, since the paper gives no daily cost.\nThe paper's curves "
             f"have {PAPER_DAYS} days a year; ours have every trading day.",
             fontsize=6.5, color="0.25", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = HERE / ("fig2_ours_K030.png" if last == HOLDOUT[1] else f"fig2_ours_K030_{last}.png")
    fig.savefig(out)
    for rule, f in frames.items():
        f = span(f, TEST[0], last)
        print(f"{rule}: net SR 1998-2021 {sr(span(f, *TEST).net):.3f}, 2022-2025 {sr(span(f, *HOLDOUT).net):.3f}; "
              f"ends gross {(1 + f.gross).prod():.2f}, net {(1 + f.net).prod():.2f}")
    print(f"walk-forward picks: " + ", ".join(f"{y}: {wf[y].code()}" for y in years if y == years[0] or wf[y] != wf[y - 1]))
    print(f"paper ends {paper[-1]:.2f} / net {paper_net[-1]:.2f}\nwrote {out}")


if __name__ == "__main__":
    main()
