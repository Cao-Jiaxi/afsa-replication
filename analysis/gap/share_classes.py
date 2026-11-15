"""Share classes on the top-500 lists: how often a company is on a list twice, how alike the two names' returns
are, and what the model's book holds in them (the final report's slides on stocks against companies).

    python analysis/gap/share_classes.py            # -> analysis/gap/SHARE_CLASSES.md (CPU, about half a minute)

The paper takes the 500 largest stocks, so two securities of one company (PERMCO) can both be on a list: two
classes of common stock (Alphabet, Berkshire Hathaway), or a tracking stock beside its parent (General Motors and
its Hughes class). Two classes of one stock have nearly collinear returns. This script
- counts such names on the plain and the buffered list (Data/WRDS/universe);
- measures the correlation of the two names' daily returns on the days both are on the plain list
  (Data/WRDS/raw/crsp_dsf_v2.parquet);
- reads the seed ensemble's book of one run (default: K = 30 on the buffered list with rebuilt history, the
  setting of DATASETS.md, test years 1998-2021) for the share of exposure and turnover in those names, and for how
  the two names of a pair are held.
A pair is called *twins* when its returns correlate above 0.9. Nothing is trained and nothing is chosen here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from deploy import ensemble, market, seed_weights  # noqa: E402

from afsa.config import RESULTS_DIR, WRDS_DIR, Config  # noqa: E402

OUT = HERE / "SHARE_CLASSES.md"
RUN = RESULTS_DIR / "runs" / "attn_K030_universe-wrds_broad_buffer600_entry_rule-history_ridge-0.0001_fill-none"
LISTS = {"wrds_broad": "broad", "wrds_broad_buffer600": "broad_buffer600"}      # universe of a run -> its list file
TEST = (1998, 2021)
TWIN = 0.9                                           # return correlation above which a pair counts as twins
MIN_DAYS = 250                                       # a pair needs this many common days on the list


def table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    return "\n".join(rows + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def load_list(name: str) -> pd.DataFrame:
    """A list with `multi`: the name's company has another name on the list that month."""
    u = pd.read_parquet(WRDS_DIR / "universe" / f"{name}.parquet", columns=["month", "permno", "permco", "ticker", "issuernm"])
    u["multi"] = u.groupby(["month", "permco"])["permno"].transform("size") > 1
    return u


def counts(u: pd.DataFrame) -> pd.DataFrame:
    """Names, companies and second names per month, averaged over three spans."""
    by = u.groupby("month").agg(names=("permno", "size"), companies=("permco", "nunique"), in_multi=("multi", "sum"))
    by["second"] = by["names"] - by["companies"]
    year = np.array([m.year for m in by.index])
    rows = []
    for label, keep in (("All months", year > 0), (f"{TEST[0]}–{TEST[1]}", (year >= TEST[0]) & (year <= TEST[1])),
                        (f"{TEST[1] + 1}–{year.max()}", year > TEST[1])):
        s = by[keep]
        rows.append({"Months": label, "Names a month": f"{s['names'].mean():.1f}", "Companies": f"{s['companies'].mean():.1f}",
                     "Second names": f"{s['second'].mean():.2f}", "Fewest / most": f"{s['second'].min()} / {s['second'].max()}",
                     "Names of a company with two or more": f"{s['in_multi'].mean():.2f}",
                     "Share of stock-months": f"{s['in_multi'].sum() / s['names'].sum():.2%}"})
    return pd.DataFrame(rows)


def pairs_of(u: pd.DataFrame) -> dict[tuple[int, int], list]:
    """(permno, permno) of one company, both on the list -> the months they share."""
    out: dict[tuple[int, int], list] = {}
    for (_, month), g in u[u["multi"]].groupby(["permco", "month"]):
        p = sorted(g["permno"])
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                out.setdefault((p[i], p[j]), []).append(month)
    return out


def return_pairs(u: pd.DataFrame) -> pd.DataFrame:
    """Correlation of the two names' daily returns on the days of the months both are on the list."""
    both = pairs_of(u)
    d = pq.read_table(WRDS_DIR / "raw" / "crsp_dsf_v2.parquet", columns=["permno", "dlycaldt", "dlyret"],
                      filters=[("permno", "in", sorted({p for k in both for p in k}))]).to_pandas()
    d["month"] = pd.to_datetime(d["dlycaldt"]).dt.to_period("M")
    ret = {p: g.set_index("dlycaldt") for p, g in d.groupby("permno")}
    last = u.drop_duplicates("permno", keep="last").set_index("permno")
    rows = []
    for (a, b), months in both.items():
        ms = pd.PeriodIndex(months, freq="M")
        x = (ret[a][ret[a]["month"].isin(ms)]["dlyret"].rename("a").to_frame()
             .join(ret[b][ret[b]["month"].isin(ms)]["dlyret"].rename("b"), how="inner").dropna())
        if len(x) >= MIN_DAYS:
            rows.append({"Company": last.at[a, "issuernm"].title(), "Names": f"{last.at[a, 'ticker']}, {last.at[b, 'ticker']}",
                         "PERMNOs": f"{a}, {b}", "From": str(ms.min()), "To": str(ms.max()), "Days": len(x),
                         "corr": float(x["a"].corr(x["b"]))})
    return pd.DataFrame(rows).sort_values("corr", ascending=False)


def book(run: Path) -> tuple[list[str], pd.DataFrame]:
    """The ensemble book of a run in the names of companies with two or more names on its list."""
    cfg = Config.load(run / "config.json")
    dates, permnos, R, member = market(cfg, *TEST)
    E = ensemble([seed_weights(d, dates, permnos) for d in sorted(run.glob("seed*"), key=lambda p: int(p.name[4:]))])
    u = load_list(LISTS[cfg.universe])
    month = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="M")
    col = {int(p): i for i, p in enumerate(permnos)}
    multi = np.zeros(E.shape, dtype=bool)
    for m, g in u[u["multi"]].groupby("month"):
        rows = np.flatnonzero(month == m)
        multi[np.ix_(rows, [col[p] for p in g["permno"] if p in col])] = True
    multi &= member
    other = member & ~multi
    gross, trade = np.abs(E), np.abs(np.diff(E, axis=0))
    per_name, per_other = (gross * multi).sum() / multi.sum(), (gross * other).sum() / other.sum()
    lines = [f"- Names a day: {member.sum(1).mean():.1f}; of them {multi.sum(1).mean():.2f} belong to a company with another name on the "
             f"list ({multi.sum() / member.sum():.2%} of member stock-days).",
             f"- They hold {(gross * multi).sum(1).mean():.2%} of the book's gross exposure and {(trade * multi[1:]).sum() / trade.sum():.2%} "
             f"of its turnover. Per name: {per_name:.3%} of the book, against {per_other:.3%} for the other names."]
    last = u.drop_duplicates("permno", keep="last").set_index("permno")
    rows = []
    for (a, b), months in pairs_of(u).items():
        if a not in col or b not in col:
            continue
        r = np.flatnonzero(month.isin(pd.PeriodIndex(months, freq="M")) & member[:, col[a]] & member[:, col[b]])
        if len(r) < MIN_DAYS:
            continue
        wa, wb = E[r, col[a]], E[r, col[b]]
        rows.append({"Names": f"{last.at[a, 'ticker']}, {last.at[b, 'ticker']}", "PERMNOs": f"{a}, {b}", "Days": len(r),
                     "corr": float(np.corrcoef(R[r, col[a]], R[r, col[b]])[0, 1]), "wcorr": float(np.corrcoef(wa, wb)[0, 1]),
                     "same": float(np.mean(np.sign(wa) == np.sign(wb))),
                     "net": float(np.abs(wa + wb).sum() / (np.abs(wa) + np.abs(wb)).sum()),
                     "gross": float((np.abs(wa) + np.abs(wb)).mean())})
    pairs = pd.DataFrame(rows).sort_values("corr", ascending=False)
    lines.append(f"- Two ordinary names hold {2 * per_other:.2%} of the book together.")
    for label, s in ((f"Twins (return correlation above {TWIN})", pairs[pairs["corr"] > TWIN]),
                     (f"Other pairs ({TWIN} or below)", pairs[pairs["corr"] <= TWIN])):
        if len(s):
            w = s["Days"]
            lines.append(f"- **{label}:** {len(s)} pairs, {int(w.sum()):,} pair-days. Their two weights correlate at "
                         f"{np.average(s['wcorr'], weights=w):.2f} and have the same sign on {np.average(s['same'], weights=w):.0%} of days; "
                         f"|net| / gross of the pair is {np.average(s['net'], weights=w):.2f}; a pair holds "
                         f"{np.average(s['gross'], weights=w):.2%} of the book.")
    return lines, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=RUN, help="run folder whose ensemble book is read (needs its <year>_w.npz files)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    plain, buffered = load_list("broad"), load_list("broad_buffer600")
    rp = return_pairs(plain)
    ever = plain[plain["multi"]].groupby("permco")["month"].nunique()
    now = plain[(plain["month"] == plain["month"].max()) & plain["multi"]].sort_values("permco")
    parts = ["# Share classes on the top-500 lists\n\nGenerated by `analysis/gap/share_classes.py`. The lists are the paper's: the 500 largest "
             "*stocks*, so two securities of one company (PERMCO) can both be on a list. A *second name* is a name beyond the first of its "
             "company; *twins* are pairs whose daily returns correlate above " + f"{TWIN}.",
             "## How often\n\n**Plain list** (`broad`)\n\n" + table(counts(plain)) + "\n\n**Buffered list** (`broad_buffer600`)\n\n"
             + table(counts(buffered)),
             f"- Companies ever with two or more names on the plain list: {len(ever)}.\n"
             f"- In {plain['month'].max()}: " + "; ".join(f"{g['issuernm'].iloc[0].title()} ({', '.join(map(str, g['permno']))})"
                                                        for _, g in now.groupby("permco")) + "."]
    shown = rp.assign(**{"Return correlation": rp["corr"].map("{:.3f}".format)}).drop(columns="corr")
    parts.append(f"## How alike the two names are\n\nPairs of one company with at least {MIN_DAYS} common days on the plain list; daily returns "
                 f"on those days. {int((rp['corr'] > TWIN).sum())} of {len(rp)} pairs are twins; the others are tracking stocks or spun-off "
                 "classes, which are different businesses.\n\n" + table(shown))
    lines, pairs = book(args.run)
    shown = pd.DataFrame({"Names": pairs["Names"], "PERMNOs": pairs["PERMNOs"], "Days": pairs["Days"],
                          "Return correlation": pairs["corr"].map("{:.3f}".format), "Weight correlation": pairs["wcorr"].map("{:.2f}".format),
                          "Same sign": pairs["same"].map("{:.0%}".format), "|net| / gross": pairs["net"].map("{:.2f}".format),
                          "Gross, % of book": (100 * pairs["gross"]).map("{:.2f}".format)})
    parts.append(f"## What the book holds\n\nThe seed ensemble of `{args.run.name}`, {TEST[0]}–{TEST[1]}, target weights.\n\n"
                 + "\n".join(lines) + "\n\n" + table(shown))
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
