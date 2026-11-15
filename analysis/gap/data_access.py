"""Data access: does our WRDS account miss any firm the paper uses? (the final report's slide on data access)

    python analysis/gap/data_access.py            # -> analysis/gap/DATA_ACCESS.md (CPU, about half a minute)

The paper takes "the securities on CRSP" and characteristics "from the CRSP/Compustat database" (its §4.1). Our account
has CRSP, Compustat North America, the CRSP-Compustat links and Fama-French, but no non-US database (Compustat Global,
Datastream, Worldscope: permission denied). This script
- takes the 109 securities named in the paper's Figures 3 and 4 (first trading day of 2021) and asks, for each, whether
  CRSP has it in December 2020 and whether our broad top 500 of January 2021 holds it; for those it does not hold, the
  CRSP market cap against our cut, and for ADRs the whole company's value in our Compustat (shares x price at the last
  fiscal year-end on or before 2020-12-31);
- searches CRSP's whole history for a few large non-US firms without a US exchange listing (Kia, Hyundai, Samsung);
- measures, by security type, the share of broad member stock-days with Compustat accounts (Data/WRDS/table).
Nothing is trained and nothing is chosen here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from afsa.config import WRDS_DIR  # noqa: E402

OUT = HERE / "DATA_ACCESS.md"
MONTH = pd.Period("2021-01", "M")                    # the holding month of the paper's figures (formed on 2020-12-31)
# Tickers at the end of 2020: Figure 3's labels (as in universe.py), then Figure 4's top 10 of factors 1-6, in the order drawn.
FIG3 = ["TSLA", "HTHT", "GNRC", "FTNT", "BSBR", "BUD", "EC", "UNH", "AWK", "V", "PM", "MSCI", "NWG", "MELI", "MASI", "BX", "DAL",
        "TMUS", "CQP", "BR", "IPGP", "FSLR", "MFG", "LDOS", "MA", "ET", "UAL", "CMG", "LYV", "ICE", "AMP", "LBTYK", "BIDU", "WPM",
        "EXPE", "RDS", "MPW", "NDAQ", "TAP", "CE", "LVS", "MPWR", "MKTX", "DLR", "GOOGL",
        "MSFT", "CME", "AAPL", "NFLX", "PFE", "IMO", "GOOG", "JNJ"]
FIG4 = ["TSLA", "KLAC", "ON", "NXPI", "ALGN", "MELI", "MPWR", "QRVO", "LRCX", "ENTG",      # "Kla" in the figure is KLA Corp
        "RDS", "COP", "TOT", "SU", "XOM", "CNQ", "PXD", "EQNR", "SLB", "BHP",
        "KEY", "HBAN", "FITB", "BAC", "MTB", "RF", "C", "WFC", "TFC", "USB",
        "EXR", "ESS", "UDR", "AVB", "EQR", "PSA", "CPT", "PEAK", "MAA", "ELS",
        "UL", "NVS", "SNY", "TM", "BTI", "GSK", "MRK", "PG", "ORAN", "VZ",
        "ED", "AEP", "AEE", "CMS", "D", "DUK", "XEL", "EVRG", "WEC", "LNT"]
NO_US_LISTING = ["KIA", "HYUNDAI", "SAMSUNG"]
ACCOUNTING = ["AT", "A2ME", "BEME", "C", "CF", "CF2P", "Q", "Lev", "E2P", "PROF", "CTO", "FC2Y", "OP", "PM", "RNA", "D2A",
              "Investment", "NOA", "DPI2A", "OA", "OL", "PCM"]


def table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    return "\n".join(rows + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def base(ticker) -> str:
    return ticker.split(".")[0] if isinstance(ticker, str) else ""


def named(u: pd.DataFrame, crsp: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """One row per CRSP security behind the paper's 109 names (Shell has two ADR lines)."""
    jan = u[u["month"] == MONTH].assign(tk=lambda d: d["ticker"].map(base))
    cut = float(jan["me"].min())
    dec = crsp[crsp["yyyymm"] == 202012].assign(tk=lambda d: d["ticker"].map(base))
    day = pd.Timestamp("2020-12-31")
    link = pd.read_parquet(WRDS_DIR / "raw" / "ccm_lnkhist.parquet", columns=["gvkey", "lpermno", "linkdt", "linkenddt"])
    link = link[(pd.to_datetime(link["linkdt"]) <= day) & (pd.to_datetime(link["linkenddt"]).fillna(day) >= day)]   # links valid then
    funda = pd.read_parquet(WRDS_DIR / "raw" / "comp_funda.parquet", columns=["gvkey", "datadate", "csho", "prcc_f"])
    funda = (funda[(pd.to_datetime(funda["datadate"]) <= day) & funda["csho"].notna() & funda["prcc_f"].notna()]
             .sort_values("datadate"))                                 # latest fiscal year with a value, over the linked gvkeys
    rows = []
    for t in dict.fromkeys(FIG3 + FIG4):
        d = dec[dec["ticker"] == t] if t == "LBTYK" else dec[dec["tk"] == t]
        if d.empty:
            rows.append({"Ticker": t, "Figure": ("3" if t in FIG3 else "") + ("4" if t in FIG4 else ""), "In CRSP": "no"})
            continue
        for _, s in d.iterrows():
            member = jan[jan["permno"] == s["permno"]]
            kind = ("ADR" if s["sharetype"] == "AD" else "foreign" if s["usincflg"] == "N" else "US")
            f = funda[funda["gvkey"].isin(link.loc[link["lpermno"] == s["permno"], "gvkey"])]
            whole = float((f["csho"] * f["prcc_f"]).iloc[-1]) / 1e3 if len(f) else float("nan")  # millions -> bn
            rows.append({"Ticker": t, "Figure": ("3" if t in FIG3 else "") + ("4" if t in FIG4 else ""), "In CRSP": "yes",
                         "Name": s["issuernm"].title(), "Kind": kind, "In our list": "yes" if len(member) else "no",
                         "CRSP cap, bn": round(s["mthcap"] / 1e6, 2), "Compustat whole company, bn": round(whole, 1)})
    return pd.DataFrame(rows), cut / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    u = pd.read_parquet(WRDS_DIR / "universe" / "broad.parquet")
    crsp = pd.read_parquet(WRDS_DIR / "raw" / "crsp_msf_v2_198512_202512.parquet",
                           columns=["permno", "ticker", "issuernm", "yyyymm", "mthcap", "sharetype", "usincflg"])
    n, cut = named(u, crsp)
    by = n.groupby("Ticker").agg(crsp=("In CRSP", "first"), member=("In our list", lambda s: (s == "yes").any()),
                                 kind=("Kind", "first"), cap=("CRSP cap, bn", "max"))
    out = by[~by["member"]]
    adr, other = out[out["kind"] == "ADR"], out[out["kind"] != "ADR"]
    adr_rows = n[n["Ticker"].isin(adr.index)]
    passes = adr_rows.groupby("Ticker")["Compustat whole company, bn"].max() > cut
    summary = [f"- Names: {len(by)} (Figure 3: {len(set(FIG3))}, Figure 4: {len(set(FIG4))}, both: {len(set(FIG3) & set(FIG4))}).",
               f"- In CRSP in December 2020: {int((by['crsp'] == 'yes').sum())}. In our broad top 500 of {MONTH}: {int(by['member'].sum())}; "
               f"our cut (the 500th security's CRSP cap at 2020-12-31) is {cut:.2f} bn.",
               f"- Not in our list: {len(out)}. ADRs: {len(adr)}, CRSP caps {adr['cap'].min():.2f} to {adr['cap'].max():.2f} bn (CRSP counts "
               f"only the receipts traded in the US); sized as the whole company in our Compustat, {int(passes.sum())} of {len(adr)} pass the cut. "
               f"Others: {len(other)} ({int((other['kind'] == 'US').sum())} US, {int((other['kind'] == 'foreign').sum())} non-US ordinary shares), "
               f"CRSP caps {other['cap'].min():.2f} to {other['cap'].max():.2f} bn."]
    hits = []
    for pat in NO_US_LISTING:
        h = crsp[crsp["issuernm"].fillna("").str.upper().str.contains(rf"\b{pat}\b", regex=True)]
        hits.append(f"{pat.title()}: {h['permno'].nunique()} securities")
    summary.append("- Large non-US firms without a US exchange listing, searched in CRSP's monthly file 1985-12 to 2025-12 (issuer name): "
                   + "; ".join(hits) + ".")

    t = pq.read_table(WRDS_DIR / "table" / "chars_daily_fill-none.parquet", columns=["date", "permno", "in_broad"] + ACCOUNTING,
                      filters=[("in_broad", "==", True)]).to_pandas()
    t = t[t["date"] >= pd.Timestamp("1990-01-01")]
    t["month"] = pd.to_datetime(t["date"]).dt.to_period("M")
    t = t.merge(u[["month", "permno", "type"]], on=["month", "permno"], how="left")
    t["n"] = t[ACCOUNTING].notna().sum(axis=1)
    t["accounts"] = t["n"] > 0
    cov = (t.groupby("type").agg(**{"Stock-days": ("permno", "size"), "Securities": ("permno", "nunique"),
                                    "With accounts": ("accounts", "mean"), "Accounting characteristics present (of 22)": ("n", "mean")})
           .reindex(["common", "ADR", "foreign", "REIT", "MLP"]).reset_index().rename(columns={"type": "Type"}))
    cov["With accounts"] = cov["With accounts"].map("{:.2%}".format)
    cov["Accounting characteristics present (of 22)"] = cov["Accounting characteristics present (of 22)"].map("{:.1f}".format)
    never = t.groupby("permno").agg(accounts=("accounts", "max"), type=("type", "last"), days=("permno", "size"))
    never = never[~never["accounts"]]
    names = u.drop_duplicates("permno", keep="last").set_index("permno")["issuernm"].str.title()
    lines = [f"- All broad member stock-days 1990-2025: {t['accounts'].mean():.2%} with accounts ({len(t):,} stock-days).",
             f"- Securities with no accounts on any member day: {len(never)} ({int(never['days'].sum()):,} stock-days): "
             + "; ".join(f"{names.get(p, p)} ({r.type}, {r.days})" for p, r in never.sort_values("days", ascending=False).iterrows()) + "."]

    shown = n.drop(columns="In CRSP").astype(object).where(n.drop(columns="In CRSP").notna(), "")
    parts = ["# Data access: our WRDS account against the paper's data\n\nGenerated by `analysis/gap/data_access.py`. The paper uses "
             "\"the securities on CRSP\" and the CRSP/Compustat database. Our account has CRSP, Compustat North America (which also holds "
             "foreign firms listed in the US), the CRSP-Compustat links and Fama-French; it has no Compustat Global, Datastream or "
             "Worldscope. *Kind*: ADR (CRSP share type AD), foreign (other security of a non-US issuer) or US.",
             "## The paper's named firms\n\n" + "\n".join(summary) + "\n\n" + table(shown),
             "## Accounts of foreign firms\n\nShare of broad member stock-days, 1990-2025, with at least one of the 22 accounting "
             "characteristics (that is, with an entered Compustat annual report), by security type.\n\n" + table(cov) + "\n\n" + "\n".join(lines)]
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
