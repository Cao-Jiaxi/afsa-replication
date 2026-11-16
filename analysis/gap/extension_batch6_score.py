"""Score batch 6 of analysis/gap/EXTENSION.md (section 16) with the harness's functions, extension.py unedited.

A. lambda_var 10 and 1000 against nosig0 (extension.rows), plus the in-sample statistics of each run's last epoch.
B. nosig0 with 10 seeds (0-2 from its folder, 3-9 from the _s3-9 folder) against its 3 seeds; and 7 against 3.
C. nosig0 against the reference on seeds 3-9 only: section 2's promised retraining of a small-margin pass."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import deploy  # noqa: E402
import extension as ex  # noqa: E402

pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 500)
ex.TRIALS["lv10"] = {"nosignal_hedge": False, "lambda_var": 10.0, "ref": "nosig0"}
ex.TRIALS["lv1000"] = {"nosignal_hedge": False, "lambda_var": 1000.0, "ref": "nosig0"}
part = sys.argv[1] if len(sys.argv) > 1 else "ABC"


def show(table: list[dict], title: str) -> None:
    print(f"\n{title}\n" + pd.DataFrame(table).set_index(["trial", "object"]).to_string(float_format=lambda x: f"{x:.2f}", na_rep=""), flush=True)


def objects(Ws, R, dates, member, aside=None) -> dict[str, pd.DataFrame]:
    """extension.paths for any list of seed weights: seeds' pooled series, ensemble, ensemble + band (dev width)."""
    def frame(H):
        daily = deploy.trade(H, R, dates)
        if aside is not None:
            luck = (H * aside).sum(1)
            daily["gross"], daily["net"] = daily["gross"] - luck, daily["net"] - luck
        return daily
    seeds = [frame(W) for W in Ws]
    out = {"model": sum(seeds) / len(seeds)}
    out["model"].attrs["seed average"] = {p: float(np.mean([ex.sr(s.loc[ex._mask(s, a, b), "net"]) for s in seeds])) for p, (a, b) in ex.PERIODS.items()}
    E = deploy.ensemble(Ws)
    out["ensemble"] = frame(E)
    grid = {d: frame(deploy.band_path(E, member, d)) for d in deploy.DELTAS}
    dev = {d: ex.sr(g.loc[ex._mask(g, *ex.DEV), "net"]) for d, g in grid.items()}
    best = max(dev, key=lambda d: (dev[d], -d))
    out["ensemble + band"] = grid[best]
    out["ensemble + band"].attrs["delta"] = best
    return out


def in_sample(run: Path) -> pd.DataFrame:
    """Last-epoch training statistics of every (seed, test year) model, averaged by era."""
    recs = []
    for d in sorted(run.glob("seed*")):
        for f in sorted(d.glob("[0-9][0-9][0-9][0-9].npz")):
            z = np.load(f, allow_pickle=True)
            recs.append({"year": int(f.stem), **dict(zip([str(c) for c in z["history_columns"]], z["history"][-1]))})
    df = pd.DataFrame(recs)
    eras = {"2003-21": (2003, 2021), "2022-25": (2022, 2025)}
    return pd.DataFrame({e: df[(df.year >= a) & (df.year <= b)].drop(columns="year").mean() for e, (a, b) in eras.items()}).T


if "A" in part:
    get = {n: ex.paths(n) for n in ("nosig0", "lv10", "lv1000")}
    table = ex.rows("nosig0", get["nosig0"], None)
    for n in ("lv10", "lv1000"):
        table += ex.rows(n, get[n], get["nosig0"])
    show(table, "A. lambda_var against nosig0 (lambda_var 100); every run without no-signal positions")
    print("\nnet SR by year\n" + pd.concat({n: ex.by_year(get[n]).loc[2014:].T for n in get}).round(2).to_string())
    for n in get:
        print(f"\n{n}: in-sample, last epoch (annualized SR on the training batches)\n{in_sample(ex.run_dir(n)).round(3).to_string()}")
    del get

if "B" in part or "C" in part:
    base = ex.run_dir("nosig0")
    cfg, dates, R, member, W012 = deploy.load(base, 1998, 2025)
    _, dates7, _, _, W39 = deploy.load(Path(str(base) + "_s3-9"), 1998, 2025)
    assert np.array_equal(dates, dates7) and len(W012) == 3 and len(W39) == 7

if "B" in part:
    three, seven, ten = objects(W012, R, dates, member), objects(W39, R, dates, member), objects(W012 + W39, R, dates, member)
    table = ex.rows("nosig0, seeds 0-2", three, None) + ex.rows("nosig0, seeds 3-9", seven, three) + ex.rows("nosig0, seeds 0-9", ten, three)
    show(table, "B. more seeds (d: against seeds 0-2)")
    print("\nnet SR by year\n" + pd.concat({"0-2": ex.by_year(three).loc[2014:].T, "0-9": ex.by_year(ten).loc[2014:].T}).round(2).to_string())
    print("\nagreement of seeds 3-9 (as deploy.seed_agreement):", {k: round(v, 3) for k, v in deploy.seed_agreement(Path(str(base) + "_s3-9")).items()})
    del three, seven, ten

if "C" in part:
    ref = ex.run_dir("reference")
    rcfg, rdates, rR, rmember, R39 = deploy.load(Path(str(ref) + "_s3-9"), 1998, 2025)
    assert np.array_equal(rdates, dates) and len(R39) == 7
    n39 = objects(W39, R, dates, member)
    for label, aside in (("as measured", None), ("no-signal P&L set aside", rR * ex.no_signal(rcfg, rdates, deploy.market(rcfg, 1998, 2025)[1]))):
        r39 = objects(R39, rR, rdates, rmember, aside)
        show(ex.rows("reference, seeds 3-9", r39, None) + ex.rows("nosig0, seeds 3-9", n39, r39),
             f"C. nosig0 against the reference on new seeds 3-9 (reference {label})")
        del r39
