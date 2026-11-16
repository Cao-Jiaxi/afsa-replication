"""The learned trading rule with views of the book and of the market, against the rule without views (CLAUDE.md
section 7, rows of 2026-11-16; TRADING_STATE.md).

    python analysis/rl/trading_state.py                     -> analysis/rl/TRADING_STATE_RESULTS.md (CPU, a few minutes)
    python analysis/rl/trading_state.py --validation-only   -> the pre-registered decision only (1996-1997)

Dataset D (buffered list, rebuilt history), K = 30, training cost 5 bp, ridge 1e-4, 30 epochs, seeds 0-2. Arms:
  S    --trade_head true --trade_state true                        the dials also see the book
  SM   --trade_head true --trade_state true --trade_market true    and the day's market
Every object is one path per period from cash on its first day (1998 for the test years, 1996 for validation), with
the book carried across years, at the paper's costs. For S and SM the dials depend on the book, so they are recomputed
along that path from the saved logits and view weights (deploy.state_path). Checked one day at a time: applied to the
stored book of the day before, the numpy rule must give the stored book of the day, except through a float32 tie at the
band's edge (`check_state`).

Objects: each arm's model (one path per seed, statistics averaged) and its ensemble (the seeds' targets averaged and
rescaled; each day each seed's dials set on the ensemble's own book and averaged); the rule without views
(TRADING_RULE.md) as model and ensemble; the fixed band trained inside the model and D + band after (trading_rule.py).

The pre-registered decision compares each arm with the rule without views on 1996-1997 only, for the model and the
ensemble. The test years are reported and never decide.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "analysis" / "gap", HERE):
    sys.path.insert(0, str(p))
import band_trained as bt  # noqa: E402
from deploy import band_path, ensemble, market, seed_weights, state_path, trade  # noqa: E402
from trading_rule import DELTA, interval  # noqa: E402

from afsa.metrics import stitch  # noqa: E402

ARMS = {"S": {"trade_state": True}, "SM": {"trade_state": True, "trade_market": True}}
PERIODS = (("1996–97", None), ("1998–2005", (1998, 2005)), ("2006–13", (2006, 2013)), ("2014–21", (2014, 2021)),
           ("2022–25", (2022, 2025)))
KINDS = (("opens", 0), ("adds", 1), ("reduces", -1))       # sign(h) sign(w - h): no position, adds to it, reduces it
VIEW = ("position", "target", "adds")


def view_weights(seed_dir: Path, dates: np.ndarray) -> np.ndarray:
    """Each day's view weights [T, 2, 3] (rows: band width, speed), from the `dial_v` of that day's model."""
    V = np.full((len(dates), 2, 3), np.nan)
    for year in np.unique(dates.astype("datetime64[Y]").astype(int) + 1970):
        z = np.load(seed_dir / f"{year}.npz")
        V[np.searchsorted(dates, z["dates"].astype("datetime64[D]"))] = z["dial_v"]
    if not np.isfinite(V).all():
        raise ValueError(f"{seed_dir}: days without view weights")
    return V


def check_state(x: dict, dates: np.ndarray, member: np.ndarray, const: dict) -> tuple[float, int]:
    """One day at a time, the numpy rule applied to the stored book of the day before must give the stored book of the
    day. Each year's first day is skipped: the model holds its target the day before. A day may differ only through a
    tie, a stock whose gap to its target equals its band width to within 1e-6: float32 (the model) and float64 (numpy)
    can then decide differently whether it trades, and a stock that trades is resized with the other movers. Returns
    the largest daily L1 gap on the other days and the number of tie days."""
    years, worst, ties = pd.DatetimeIndex(dates).year.to_numpy(), 0.0, 0
    for t in np.flatnonzero(years[1:] == years[:-1]) + 1:
        h, D, _ = state_path(x["W"][t:t + 1], member[t:t + 1], [(x["A"][t:t + 1], x["K"][t:t + 1], x["V"][t:t + 1])],
                             x["has"][t:t + 1], h0=x["book"][t - 1], **const)
        gap = np.abs(h[0] - x["book"][t])
        if gap.sum() <= 1e-5:
            worst = max(worst, float(gap.sum()))
            continue
        edge = np.abs(np.abs(x["W"][t] - x["book"][t - 1]) - D[0]) < 1e-6
        if not (edge & member[t] & x["has"][t] & (gap > 1e-6)).any():
            raise ValueError(f"{x['dir']}, {dates[t]}: the numpy rule does not reproduce the stored book (L1 gap {gap.sum():.1e})")
        ties += 1
    return worst, ties


def period(first: int, last: int, train_years: int, arms: tuple[str, ...] = tuple(ARMS)) -> tuple[dict, dict]:
    """object -> daily frames (one per seed, or one for an ensemble) on first..last, and what the dial tables need."""
    base = dataclasses.replace(bt.BASE, train_years=train_years)
    rule, fixed = dataclasses.replace(base, trade_head=True), dataclasses.replace(base, band=DELTA)
    const = {"band0": rule.head_band0, "band_max": rule.head_band_max, "eta0": rule.head_eta0}
    dates, permnos, R, member = market(base, first, last)
    seeds = sorted(bt.run_dir(base).glob("seed*"), key=lambda p: int(p.name[4:]))
    W_d = [seed_weights(d, dates, permnos) for d in seeds]
    W_f = [bt.saved(bt.run_dir(fixed) / d.name, dates, permnos, "weights") for d in seeds]
    rule_dirs = [bt.run_dir(rule) / d.name for d in seeds]
    W_r, D_r, E_r = ([bt.saved(d, dates, permnos, key) for d in rule_dirs] for key in ("weights", "delta", "eta"))

    def band(W):
        return band_path(W, member, DELTA, rescale="moved")

    H_r = [band_path(W, member, dl, rescale="moved", eta=et) for W, dl, et in zip(W_r, D_r, E_r)]
    E_rule = ensemble(W_r)
    H_re = band_path(E_rule, member, sum(D_r) / len(D_r), rescale="moved", eta=sum(E_r) / len(E_r))
    objects = {"rule": [trade(H, R, dates) for H in H_r], "rule ensemble": [trade(H_re, R, dates)]}
    info = {"dates": dates, "member": member, "checks": {},
            "rule": {"W": W_r, "H": H_r, "D": D_r, "E": E_r, "ens": (E_rule, H_re, sum(D_r) / len(D_r), sum(E_r) / len(E_r))}}
    for name in arms:
        cfg = dataclasses.replace(rule, **ARMS[name])
        xs = []
        for s in seeds:
            d = bt.run_dir(cfg) / s.name
            W, A, Kl, G, B = (bt.saved(d, dates, permnos, key) for key in ("weights", "dial_a", "dial_k", "signal", "book"))
            bt.check(stitch(d), trade(B, R, dates), d)                       # the stored book reproduces the stored results
            xs.append({"dir": d, "W": W, "A": A, "K": Kl, "has": G != 0, "V": view_weights(d, dates), "book": B})
        if any((x["has"] != xs[0]["has"]).any() for x in xs):
            raise ValueError(f"{name}: the seeds disagree on who has a signal")
        found = [check_state(x, dates, member, const) for x in xs]
        info["checks"][name] = (max(f[0] for f in found), sum(f[1] for f in found), len(xs) * len(dates))
        paths = [state_path(x["W"], member, [(x["A"], x["K"], x["V"])], x["has"], **const) for x in xs]
        E_arm = ensemble([x["W"] for x in xs])
        ens = state_path(E_arm, member, [(x["A"], x["K"], x["V"]) for x in xs], xs[0]["has"], **const)
        objects[name] = [trade(H, R, dates) for H, _, _ in paths]
        objects[f"{name} ensemble"] = [trade(ens[0], R, dates)]
        info[name] = {"W": [x["W"] for x in xs], "H": [p[0] for p in paths], "D": [p[1] for p in paths],
                      "E": [p[2] for p in paths], "V": [x["V"] for x in xs], "has": xs[0]["has"], "ens": (E_arm, *ens)}
        info["has"] = xs[0]["has"]
        for x in xs:
            x.clear()
    objects["fixed band trained"] = [trade(band(W), R, dates) for W in W_f]
    objects["fixed band trained, ensemble + band"] = [trade(band(ensemble(W_f)), R, dates)]
    objects["D model + band after"] = [trade(band(W), R, dates) for W in W_d]
    objects["D ensemble + band after"] = [trade(band(ensemble(W_d)), R, dates)]
    return objects, info


def decision(val: dict, arm: str) -> tuple[str, list[str]]:
    """The pre-registered verdict for one arm (CLAUDE.md section 7): against the rule without views, model and ensemble."""
    lines, signs = [], []
    for a, b in ((arm, "rule"), (f"{arm} ensemble", "rule ensemble")):
        d, lo, hi = interval(bt.pooled(val[a]), bt.pooled(val[b]))
        signs.append(1 if lo > 0 else -1 if hi < 0 else 0)
        lines.append(f"{a} − {b}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    verdict = ("better" if max(signs) == 1 and min(signs) >= 0 else
               "worse" if min(signs) == -1 and max(signs) <= 0 else "inconclusive")
    return verdict, lines


def kinds(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """sign(h_{t-1}) sign(w_t - h_{t-1}) on each day after the first: +1 adds to the position, -1 reduces it, 0 opens."""
    prev = H[:-1]
    return np.sign(prev) * np.sign(W[1:] - prev)


def dial_table(info: dict, years: tuple[int, int] | None, names: tuple[str, ...]) -> pd.DataFrame:
    """Median band width (bp of the book) and speed by kind of trade, on member stock-days with a signal along each
    seed's own path (pooled over seeds), and the share of held positions left untouched."""
    y = pd.DatetimeIndex(info["dates"]).year.to_numpy()
    keep = np.ones(len(y), dtype=bool) if years is None else (y >= years[0]) & (y <= years[1])
    rows = {}
    for name in names:
        x = info[name]
        cells, by = {}, {k: ([], []) for k, _ in KINDS}
        untouched = []
        for W, H, D, E in zip(x["W"], x["H"], x["D"], x["E"]):
            m = info["member"][1:] & info["has"][1:] & keep[1:, None]
            kd = kinds(H, W)
            for k, v in KINDS:
                sel = m & (kd == v) & ((H[:-1] != 0) if v else (H[:-1] == 0))
                by[k][0].append(D[1:][sel]), by[k][1].append(E[1:][sel])
            held = info["member"][1:] & info["member"][:-1] & (H[:-1] != 0) & keep[1:, None]
            untouched.append(float((H[1:] == H[:-1])[held].mean()))
        for k, _ in KINDS:
            d, e = np.concatenate(by[k][0]), np.concatenate(by[k][1])
            cells[f"δ {k}"], cells[f"η {k}"], cells[f"share {k}"] = np.median(d) * 1e4, np.median(e), float(len(d))
        total = sum(cells[f"share {k}"] for k, _ in KINDS)
        for k, _ in KINDS:
            cells[f"share {k}"] /= total
        cells["untouched"] = float(np.mean(untouched))
        rows[name] = cells
    order = [f"{q} {k}" for q in ("δ", "η") for k, _ in KINDS] + [f"share {k}" for k, _ in KINDS] + ["untouched"]
    return pd.DataFrame(rows).T[order]


def view_table(info_v: dict, info_t: dict) -> pd.DataFrame:
    """The view weights (v_delta, v_eta on position, target, adds), averaged over seeds and the period's windows."""
    rows = {}
    for label, s in PERIODS:
        info = info_v if s is None else info_t
        y = pd.DatetimeIndex(info["dates"]).year.to_numpy()
        keep = np.ones(len(y), dtype=bool) if s is None else (y >= s[0]) & (y <= s[1])
        for name in ARMS:
            if name not in info:
                continue
            years = np.unique(y[keep])
            V = np.mean([np.mean([Vs[np.flatnonzero(y == yr)[0]] for yr in years], 0) for Vs in info[name]["V"]], 0)
            rows[(name, label)] = {f"{dial} {v}": V[i, j] for i, dial in enumerate(("v_δ", "v_η")) for j, v in enumerate(VIEW)}
    t = pd.DataFrame(rows).T
    t.index = [f"{a}, {b}" for a, b in t.index]
    return t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation-only", action="store_true", help="only the pre-registered decision, on 1996-1997")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")                                 # the Windows console's code page lacks δ and η
    val, info_v = period(*bt.VALIDATION, 6)
    verdicts = {arm: decision(val, arm) for arm in ARMS}
    for arm, (verdict, lines) in verdicts.items():
        print(f"{arm}: {verdict}", *lines, sep="\n  ")
    print("numpy rule against the stored book (largest daily L1 gap, tie days, seed-days):", info_v["checks"])
    if args.validation_only:
        return
    test, info_t = period(bt.TEST[0], bt.HOLDOUT[1], 8)
    order = ["rule", "S", "SM", "fixed band trained", "D model + band after", "rule ensemble", "S ensemble", "SM ensemble",
             "fixed band trained, ensemble + band", "D ensemble + band after"]
    levels = bt.table({k: test[k] for k in order}, val)
    pairs = [("S", "rule"), ("S ensemble", "rule ensemble"), ("SM", "rule"), ("SM ensemble", "rule ensemble"),
             ("SM", "S"), ("SM ensemble", "S ensemble"), ("S", "fixed band trained"),
             ("S ensemble", "fixed band trained, ensemble + band"), ("SM", "fixed band trained"),
             ("SM ensemble", "fixed band trained, ensemble + band")]
    c = bt.contrasts(test, val, pairs)
    dials = pd.concat({label: dial_table(info_v if s is None else info_t, s, ("rule", "S", "SM")) for label, s in PERIODS})
    dials.index = [f"{name}, {label}" for label, name in dials.index]
    views = view_table(info_v, info_t)
    checks = {k: (max(info_v["checks"][k][0], info_t["checks"][k][0]), info_v["checks"][k][1] + info_t["checks"][k][1],
                  info_v["checks"][k][2] + info_t["checks"][k][2]) for k in ARMS}
    lines = [
        "# The learned trading rule with views of the book and of the market: results",
        "",
        "Written by `analysis/rl/trading_state.py`, as pre-registered in CLAUDE.md §7 (rows of 2026-11-16; the views are "
        "specified in [TRADING_STATE.md](TRADING_STATE.md)). Dataset D (buffered list, rebuilt history), K = 30, training "
        "cost 5 bp, ridge 1e-4, 30 epochs, seeds 0–2. **S**: the dials also see the book (`--trade_state true`). **SM**: "
        "they also see the day's market (`--trade_market true` as well). *Rule*: the learned rule without views "
        "([TRADING_RULE_RESULTS.md](TRADING_RULE_RESULTS.md)). Test years 1998–2021 and 2022–2025 use 8-year windows; the "
        "validation years 1996–1997 use 6-year windows. Every path starts from cash on its period's first day and carries "
        "its book across years, scored at the paper's costs (5 bp per unit traded, 1 bp per unit short a day).",
        "",
        "For S and SM the dials depend on the book, so they are recomputed along each carried path (in float64, by "
        "`deploy.state_path`). Checked one day at a time against the model's own float32 books: applied to the stored "
        "book of the day before, the numpy rule gives the stored book of the day, with a largest daily L1 gap of "
        + ", ".join(f"{v[0]:.1e} ({k})" for k, v in checks.items()) + ". The exceptions are "
        + " and ".join(f"{v[1]} of {v[2]:,} seed-days ({k})" for k, v in checks.items())
        + " with a tie: a stock whose gap to its target equals its band width to within 1e-6, so the two precisions "
        "decide differently whether it trades. A stock that trades is resized with the other movers, so a tie can move "
        "the day's book by up to a few tenths of a per cent.",
        "",
        "## Decision (pre-registered: 1996–1997 only, each arm against the rule without views)",
        "",
        *[f"- **{arm}: {v.capitalize()}.** " + "; ".join(ls) + "." for arm, (v, ls) in verdicts.items()],
        "",
        "Pooled seed-mean net series for the models; one series for each ensemble. Paired circular block bootstrap "
        "(21-day blocks, 2,000 draws, the same days for both), 95% interval. The decision was computed from the "
        "validation runs before any test-year model of S or SM had trained.",
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
        "validation column of the first four rows decides; everything else is reported.",
        "",
        bt.md(c),
        "## The dials by kind of trade",
        "",
        "Member stock-days with a signal along each seed's own path, pooled over seeds. *Opens*: no position held; "
        "*adds*: trading towards the target would add to the position; *reduces*: it would reduce or reverse it. δ is the "
        "median band width in basis points of the book (4 = the fixed band); η the median speed; *share* the share of "
        "stock-days of each kind; *untouched* the share of held positions (member on both days) that do not trade on a day.",
        "",
        bt.md(dials),
        "## The view weights",
        "",
        "v_δ and v_η: each dial's weights on the view of the book (position size, target size, adds), averaged over seeds "
        "and the period's windows. A positive v_δ widens the band, a positive v_η speeds up trading. Log sizes are in units "
        "of an average position.",
        "",
        bt.md(views),
    ]
    out = HERE / "TRADING_STATE_RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(levels.round(2).to_string())
    print(c.to_string())
    print(dials.round(2).to_string())
    print(views.round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
