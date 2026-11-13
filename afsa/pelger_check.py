"""Compares monthly characteristics with the Chen-Pelger-Zhu panel (Data\\WRDS\\pelger\\characteristics.parquet).

The panel holds raw values of 46 characteristics per (PERMNO, month), 1963-2016, complete cases only (a stock-month
is there only if all 46 exist). 35 of AFSA's 39 characteristics are in it; the 4 daily ones are not. A panel row of
month m carries the return of month m and characteristics known at the end of m-1, so its month is our holding month.

    from afsa.pelger_check import compare
    print(compare(df))        # df: permno, month (Period[M], holding month) and any of the 35 characteristic columns

Every agent and test reports the same numbers:
    n         stock-months where both have a value
    equal     share with |ours - theirs| <= 1e-6 + 1e-4 |theirs|   (the "match rate")
    within1   share within 1%: |ours - theirs| <= 1e-6 + 1e-2 |theirs|
    spearman  rank correlation over all common stock-months
    med_rel   median of |ours - theirs| / |theirs|
    rel1e4    share with |ours - theirs| <= 1e-4 |theirs|: purely relative. The 1e-6 of `equal` is lenient for a
              characteristic whose values are small (it is 0.2 to 0.6% of a typical Resid_Var): for Resid_Var and
              Variance quote rel1e4
    ours_nan  number of stock-months of `df` where the panel has a value and ours is missing. They are not in n
              and lower no rate: a characteristic that is missing where the panel has it shows only here
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afsa.wrds_data import PELGER_DIR

# our name (AFSA Table 1) -> the panel's column
PELGER_NAME = {c: c for c in (
    "A2ME", "AT", "BEME", "Beta", "C", "CF", "CF2P", "CTO", "D2A", "DPI2A", "E2P", "FC2Y", "Investment", "Lev", "LME",
    "LTurnover", "NOA", "OA", "OL", "OP", "PCM", "PM", "PROF", "Q", "r2_1", "r12_2", "r12_7", "r36_13", "Rel2High",
    "Resid_Var", "RNA", "Spread", "SUV", "Variance")}
PELGER_NAME["ST_Rev"] = "ST_REV"


def load_pelger(columns: list[str] | None = None) -> pd.DataFrame:
    """The panel from 1984 on, with `month` (Period[M]) = the month of its date; `ret` is that month's return."""
    cols = None if columns is None else ["permno", "date", *columns]
    p = pd.read_parquet(PELGER_DIR / "characteristics.parquet", columns=cols)
    p["month"] = p["date"].dt.to_period("M")
    return p


def compare(df: pd.DataFrame, chars: list[str] | None = None, by: str | None = None) -> pd.DataFrame:
    """Match statistics per characteristic (and per value of `by`, a column of `df` such as a year or a type)."""
    chars = [c for c in (chars or list(PELGER_NAME)) if c in df.columns]
    p = load_pelger([PELGER_NAME[c] for c in chars]).rename(columns={v: f"{k}__p" for k, v in PELGER_NAME.items()})
    keep = ["permno", "month", *chars] + ([by] if by else [])
    m = df[keep].merge(p.drop(columns="date"), on=["permno", "month"], how="inner")
    rows = []
    for key, g in (m.groupby(by) if by else [(None, m)]):
        for c in chars:
            a, b = g[c].to_numpy(dtype=float), g[f"{c}__p"].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            ours_nan = int((np.isfinite(b) & ~np.isfinite(a)).sum())
            a, b = a[ok], b[ok]
            row = {"characteristic": c, "n": int(ok.sum())}
            if by:
                row = {by: key, **row}
            if len(a):
                d = np.abs(a - b)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = d / np.abs(b)
                row.update(equal=float((d <= 1e-6 + 1e-4 * np.abs(b)).mean()), within1=float((d <= 1e-6 + 1e-2 * np.abs(b)).mean()),
                           spearman=float(pd.Series(a).corr(pd.Series(b), method="spearman")) if len(a) > 2 else np.nan,
                           med_rel=float(np.nanmedian(rel)), rel1e4=float((d <= 1e-4 * np.abs(b)).mean()))
            rows.append({**row, "ours_nan": ours_nan})
    columns = ([by] if by else []) + ["characteristic", "n", "equal", "within1", "spearman", "med_rel", "rel1e4", "ours_nan"]
    return pd.DataFrame(rows, columns=columns)
