"""How our universe differs from the paper's, as far as the paper's own figures and tables show it.

    python analysis/gap/universe.py

1. Figure 3 labels about 53 of the "500 equities with the largest market cap on the first trading day of 2021".
   The first 45 are in descending order of CRSP PERMNO (Tesla 93436 down to Alphabet 90319), so they are the
   newest listings of the paper's universe at that date. Which of them are in ours?
2. The companion paper's rule (afsa.build_window): how much of our monthly top 500 survives it, year by year.
3. The paper's market row (equal-weighted, 1998-2021: mean 8.61%, volatility 20.37%) against each universe.

Prints names and market values derived from CRSP: for the console only, not for a committed file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsa.build_window import load_sources, select  # noqa: E402
from afsa.config import SOURCE_DIR, TRAIN_DATA_DIR  # noqa: E402
from afsa.metrics import market_returns  # noqa: E402

# Tickers at the end of 2020 of the first 45 labels of Figure 3, in the order they are drawn, and of the 8 others.
FIG3 = ["TSLA", "HTHT", "GNRC", "FTNT", "BSBR", "BUD", "EC", "UNH", "AWK", "V", "PM", "MSCI", "NWG", "MELI", "MASI", "BX", "DAL",
        "TMUS", "CQP", "BR", "IPGP", "FSLR", "MFG", "LDOS", "MA", "ET", "UAL", "CMG", "LYV", "ICE", "AMP", "LBTYK", "BIDU", "WPM",
        "EXPE", "RDS", "MPW", "NDAQ", "TAP", "CE", "LVS", "MPWR", "MKTX", "DLR", "GOOGL"]
FIG3_OTHERS = ["MSFT", "CME", "AAPL", "NFLX", "PFE", "IMO", "GOOG", "JNJ"]


def base(ticker) -> str:
    return ticker.split(".")[0] if isinstance(ticker, str) else ""


def main() -> None:
    pd.set_option("display.width", 220)
    src = load_sources()
    permnos, dates, years = src["permnos"], src["dates"], src["years"]
    broad_dates = np.load(TRAIN_DATA_DIR / "broad" / "dates.npy")
    broad_permno = np.load(TRAIN_DATA_DIR / "broad" / "permno.npy")

    # ---- 1. Figure 3
    end = pd.read_parquet(SOURCE_DIR / "crsp_daily.parquet", columns=["permno", "ticker", "cap", "in_universe"],
                          filters=[("date", "==", pd.Timestamp("2020-12-31"))])
    end["tk"] = end["ticker"].map(base)
    first_2021 = int(np.searchsorted(broad_dates, np.datetime64("2021-01-04")))
    ours = set(broad_permno[first_2021][broad_permno[first_2021] >= 0].tolist())
    _, cols, info = select(src, 2021, 8, "window")
    rule = set(permnos[cols].tolist())
    complete_cap = pd.Series(src["cap"][np.searchsorted(dates, np.datetime64("2020-12-31"))], index=permnos)
    status = {"in ours": [], "not in ours, picked by the window rule": [], "not in ours, not picked": [], "not in Ander's files": []}
    for t in FIG3:
        rows = end[end.tk == t] if t != "LBTYK" else end[end.ticker == "LBTYK"]
        if len(rows) == 0:
            status["not in Ander's files"].append(t)
        elif rows.permno.isin(ours).any():
            status["in ours"].append(t)
        elif rows.permno.isin(rule).any():
            status["not in ours, picked by the window rule"].append(t)
        else:
            status["not in ours, not picked"].append(f"{t} ({rows.cap.max() / 1e6:.1f} bn)")
    print("1. The 45 newest listings labelled in Figure 3 (first trading day of 2021)")
    for k, v in status.items():
        print(f"   {len(v):2d} {k}: {', '.join(v)}")
    newest = end[end.permno.isin(ours) & (end.permno >= 90319)]
    labelled = newest.tk.isin(FIG3) | newest.ticker.isin(["LBTYK", "RDS.A", "RDS.B"])
    print(f"   Our universe has {len(newest)} securities with PERMNO >= 90319; {int(labelled.sum())} are labelled in the figure; "
          f"the largest that is not: {newest[~labelled].sort_values('cap').tk.iloc[-1]} ({newest[~labelled].cap.max() / 1e6:.0f} bn)")
    print(f"   Our cutoff on 2020-12-31: {complete_cap[list(ours)].min() / 1e6:.1f} bn; the rule's 500th: {info['cap_smallest_bn']:.1f} bn")
    print(f"   The 8 other labels in ours: {sum(end[end.tk == t].permno.isin(ours).any() for t in FIG3_OTHERS)} of 8")

    # ---- 2. the window rule against our monthly top 500
    rows = {}
    for year in range(1998, 2022):
        last_train = int(np.flatnonzero(years == year - 1)[-1])
        t = int(np.searchsorted(broad_dates, dates[last_train]))
        mine = set(broad_permno[t][broad_permno[t] >= 0].tolist())
        out = {"ours": len(mine)}
        for r in ("window", "window_train"):
            _, cols, _ = select(src, year, 8, r)
            out[f"{r}: shared with ours"] = len(mine & set(permnos[cols].tolist()))
        rows[year] = out
    t = pd.DataFrame(rows).T
    print("\n2. Securities shared by the rule's 500 and our monthly top 500 on the last training day")
    print(t.T.to_string())
    print("   mean share of our list that the rule keeps:", (t.iloc[:, 1:].div(t["ours"], axis=0)).mean().round(3).to_dict())

    # ---- 3. the market row
    print("\n3. Equal-weighted universe, excess returns, 1998-2021 (paper: mean 8.61, volatility 20.37, Sharpe 0.42)")
    for u in ("common", "broad", "window_train", "window"):
        try:
            m = market_returns(u, 8)
        except (FileNotFoundError, ValueError):
            continue
        m = m[(m.index.year >= 1998) & (m.index.year <= 2021)]
        if len(m):
            print(f"   {u:13s} mean {m.mean() * 252 * 100:5.2f}  volatility {m.std() * np.sqrt(252) * 100:5.2f}  Sharpe {m.mean() / m.std() * np.sqrt(252):.2f}  ({len(m)} days)")


if __name__ == "__main__":
    main()
