"""The 22 accounting characteristics of AFSA's Table 1, from Compustat annual and CRSP, checked against the CPZ panel.

    .venv\\Scripts\\python.exe -m afsa.chars_accounting build              # -> Data\\WRDS\\characteristics\\{link,accounting}.parquet (3 s)
    .venv\\Scripts\\python.exe -m afsa.chars_accounting check              # panel match, missing shares, ties, filing lags (25 s)
    .venv\\Scripts\\python.exe -m afsa.chars_accounting check --detail     # also the match by year, security type, currency, sector
    .venv\\Scripts\\python.exe -m afsa.chars_accounting build --all-fills  # every zero-fill on (the CPZ convention) -> accounting_allfills.parquet
    .venv\\Scripts\\python.exe -m afsa.chars_accounting build --strict     # no zero-fill at all -> accounting_strict.parquet
    (also: build --strict ceq lct; build --fiscal-rule pelger. Only the default build writes accounting.parquet.)

    AT, A2ME, BEME, C, CF, CF2P, Q, Lev, E2P, PROF, CTO, FC2Y, OP, PM, RNA, D2A, Investment, NOA, DPI2A, OA, OL, PCM

One row per security (PERMNO) and holding month m, for every security of crsp_dsf_v2 and every month from 1986-01 to
2025-12 that follows a month of its daily history. A row uses only accounts whose filing deadline has passed by the
first day of m (rule "safe" below, with its limits) and a market cap of an earlier December. A characteristic that
cannot be computed is NaN: nothing is carried forward or imputed here.
The formulas, the evidence for each and what the panel cannot verify are in analysis\\characteristics\\accounting.md.

The steps (each a pure function of in-memory frames):

1. `link`: the GVKEY of the security at the formation date (the last trading day of m-1), from the CCM link history:
   link types LC, LU, LS, any `linkprim` with priority P > C > J > N. Secondary share classes get their firm's
   accounts. `foreign_filers` adds `foreign_filer`: an ADR or a non-US issuer at the formation date (CRSP monthly).
2. `fiscal_years`: one row per firm and Compustat fiscal year (`fyear`) with the 18 characteristics that need no
   CRSP price and the numerators of the four that do. "Lagged" is the firm's previous fiscal year. Values are in the
   firm's reporting currency (CAD for some Canadian firms): the panel does not convert them, and neither do we.
3. `assign_fiscal_year`: which fiscal year a holding month uses (`fiscal_rule`). With Y the June-convention year
   (the month lies in July of Y .. June of Y+1):
       "pelger"  Compustat fyear = Y-1, i.e. year-ends from June of Y-1 to May of Y. This is what the panel does. A
                 year ending in March, April or May of Y is used from July of Y, when it is 3, 2 or 1 months old and
                 the annual report may not be public: a look-ahead inherited from the panel.
       "ff"      the written rule (Fama and French): the fiscal year ending in calendar year Y-1.
       "safe"    "pelger", but a fiscal year is used only once its year-end plus a filing lag is on or before the
                 FIRST DAY OF THE HOLDING MONTH. The lag is `min_lag_months` (4) for US filers (10-K: due in 90
                 days, later 60 to 75) and `min_lag_months_foreign` (6) for ADRs and non-US issuers (20-F: due in 6
                 months for fiscal years ending before 2011-12-15, 4 months since; 6 is applied in every year).
                 Until then the month keeps what June used: the previous fiscal year and the previous December's
                 market cap. This is what `build` writes.
   Three limits of "safe":
   - The lag runs to the first day of the holding month, 1 to 4 calendar days after the formation date. For a
     foreign filer 6 months is the old deadline itself: the 20-F of a December year-end was due on the last day of
     June, which is the formation date of July or, when June ends on a weekend, later than it. Those accounts
     enter in July, on the deadline (and a January-May year-end likewise in August-December): a known limit, not
     gated by a 7th month. `check` counts the stock-months.
   - The fallback to the previous fiscal year happens only when the row of the newer fiscal year EXISTS in the
     Compustat file. A firm with no row for fyear Y-1 gets NaN, never the older year (no carry-forward). No VALUE
     of a later row is ever used, but the output of a month depends on whether the newer row is in the download:
     cutting Compustat at datadate + lag turns the gated months into NaN. That is this rule, not a leak; cutting at
     datadate <= formation date changes nothing.
   - These are deadlines, not filing dates: a late filer is not detected. And Compustat holds restated accounts,
     not what was public at the time (no point-in-time data on disk).
4. `december_caps`: A2ME, BEME, CF2P and E2P divide by the SECURITY's own CRSP market cap on its last trading day of
   December of Y-1 (fiscal year Y-1 over December Y-1), whatever month the fiscal year ends in. The written
   definition (Freyberger, Neuhierl and Weber) says firm-level market equity; the panel uses the security's, and so
   do we. Q instead uses Compustat's own market value at the fiscal year-end, CSHO x PRCC_F (the whole company).

Missing inputs (`strict`). The main items of a characteristic must exist. The secondary items listed in ZERO_FILLED
can count as 0 when missing, which is what the panel does wherever it can be seen. `strict` says for which items
that fill is OFF, so that the characteristic is NaN when the item is missing:
    strict=PRODUCTION_STRICT   the default, and what `build` writes to accounting.parquet: no fill for xsga, dp, cogs,
                               ceq and lct, which the panel does not support (a missing XSGA made FC2Y exactly 0 in
                               a sixth of the universe's stock-months); every fill the panel confirms stays on
    strict=False               every fill on: the CPZ convention, which `check` uses to verify the formulas
    strict=True                every fill off
    strict=("xsga", "ceq")     only those off
Everything written here derives from CRSP and Compustat: it stays under Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Collection

import numpy as np
import pandas as pd

from afsa.config import WRDS_DIR

OUT_DIR = WRDS_DIR / "characteristics"
LINK_FILE = OUT_DIR / "link.parquet"
OUT_FILE = OUT_DIR / "accounting.parquet"
UNIVERSE_FILE = WRDS_DIR / "universe" / "broad.parquet"

CHARS = ("AT", "A2ME", "BEME", "C", "CF", "CF2P", "Q", "Lev", "E2P", "PROF", "CTO", "FC2Y", "OP", "PM", "RNA", "D2A",
         "Investment", "NOA", "DPI2A", "OA", "OL", "PCM")
FIRST_MONTH, LAST_MONTH = pd.Period("1986-01", "M"), pd.Period("2025-12", "M")   # holding months
LINK_TYPES = ("LC", "LU", "LS")
LINKPRIM_ORDER = ("P", "C", "J", "N")       # primary, primary assigned by CRSP, joiner secondary, secondary
FISCAL_RULES = ("pelger", "ff", "safe")
MIN_LAG_MONTHS = 4                          # "safe": months from the fiscal year-end to the first day of the holding month
MIN_LAG_MONTHS_FOREIGN = 6                  # the same for ADRs and non-US issuers (Form 20-F: 6 months until fiscal 2011)
MSF_FILE = "crsp_msf_v2_198512_202512"      # CRSP monthly (CIZ): sharetype and usincflg by month
TXDITC_LAST_FYEAR = 1992                    # book equity includes deferred taxes only up to this fiscal year

# characteristic -> the items that CAN count as 0 when missing: the whole zero-fill register (`strict` turns items
# off). Inside book equity: "ps" is the preferred stock PSTKRV, else PSTKL, else PSTK (0 if all three are missing),
# and "txditc" matters up to fiscal 1992.
BE_ZERO_FILLED = ("ps", "txditc")
NOA_ZERO_FILLED = ("ivao", "dlc", "dltt", "mib", "pstk", "ceq")
ZERO_FILLED: dict[str, tuple[str, ...]] = {
    "BEME": BE_ZERO_FILLED,
    "CF": ("dp", "wcapch", "capx", *BE_ZERO_FILLED),
    "CF2P": ("dp", "txdb"),
    "Q": ("txdb", "ceq"),
    "Lev": ("dltt", "dlc"),
    "PROF": BE_ZERO_FILLED,
    "FC2Y": ("xsga", "xrd", "xad"),
    "OP": ("cogs", "tie", "xsga", *BE_ZERO_FILLED),
    "RNA": NOA_ZERO_FILLED,
    "NOA": NOA_ZERO_FILLED,
    "DPI2A": ("invt",),
    "OA": ("lct", "dlc", "txp", "dp"),
    "OL": ("xsga",),
}
ZERO_FILL_ITEMS = tuple(dict.fromkeys(i for items in ZERO_FILLED.values() for i in items))
# The fills that are OFF in the production file (report, section 5). The rule: an input counts as 0 only where the
# panel shows the fill AND a missing item plausibly means "none". xsga, dp and cogs fail the first test; ceq and lct
# fail the second (a judgement, on 0.1 to 0.15% of universe stock-months):
#   xsga  FC2Y, OP, OL   no panel row has XSGA missing; the 0 that the fill implies is in 9 of 908,040 rows of its SGA2S
#   dp    CF, CF2P, OA   the panel cannot tell for CF and CF2P (D2A is NaN there); 90.6% on 127 rows for OA
#   cogs  OP             the panel cannot tell
#   ceq   Q, NOA, RNA    in the panel, but on 4 securities; a missing common equity is not 0
#   lct   OA             in the panel, but on 3 securities; missing current liabilities are not 0
# Still ON: wcapch, capx, xrd, xad, mib, ivao, txdb, txp, invt, txditc (the panel confirms each on 1,754 to 240,720 rows);
# tie, which is missing in every panel row (OP equals the panel with no interest subtracted); and dltt, dlc, pstk, ps
# (12 to 382 panel rows, none against: thin support, kept because a missing debt or preferred item is read as none).
PRODUCTION_STRICT = ("xsga", "dp", "cogs", "ceq", "lct")
PRODUCTION_FILLED = {c: tuple(i for i in items if i not in PRODUCTION_STRICT) for c, items in ZERO_FILLED.items()}
# characteristic -> its numerator in the fiscal-year table; the denominator is the security's December market cap
ME_RATIOS = {"A2ME": "AT", "BEME": "be", "CF2P": "cfp", "E2P": "ib"}
ANNUAL_CHARS = tuple(c for c in CHARS if c not in ME_RATIOS)            # complete within a fiscal year
FUNDA_COLUMNS = [
    "gvkey", "datadate", "fyear", "curcd", "at", "lt", "seq", "ceq", "pstk", "pstkrv", "pstkl", "txditc", "txdb", "che",
    "ivao", "dlc", "dltt", "mib", "act", "lct", "txp", "dp", "ppegt", "invt", "sale", "revt", "cogs", "xsga", "xrd", "xad",
    "oiadp", "tie", "ni", "ib", "capx", "wcapch", "csho", "prcc_f",
]
Strict = bool | Collection[str]


# ---------------------------------------------------------------------------------------------------- grid and link
def _ordinal(dates: pd.Series) -> np.ndarray:
    """Month number of a date (the ordinal of its Period[M])."""
    return ((dates.dt.year - 1970) * 12 + dates.dt.month - 1).to_numpy()


def month_grid(daily: pd.DataFrame, first: pd.Period = FIRST_MONTH, last: pd.Period = LAST_MONTH) -> pd.DataFrame:
    """(permno, month, formation_date): every holding month from the month after the security's first daily row to
    the month after its last. `daily` has permno and dlycaldt. The formation date is the market's last trading day
    of month-1 (the last date of that month in `daily`)."""
    days = pd.DatetimeIndex(np.sort(daily["dlycaldt"].unique()))
    form = pd.Series(days, index=days.to_period("M")).groupby(level=0).max()
    span = daily.groupby("permno")["dlycaldt"].agg(["min", "max"])
    lo = np.maximum(_ordinal(span["min"]) + 1, first.ordinal)
    hi = np.minimum(_ordinal(span["max"]) + 1, last.ordinal)
    n = np.maximum(hi - lo + 1, 0)
    ordinal = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    g = pd.DataFrame({"permno": np.repeat(span.index.to_numpy(), n).astype("int32"),
                      "month": pd.PeriodIndex.from_ordinals(ordinal, freq="M")})
    g["formation_date"] = form.reindex(g["month"] - 1).to_numpy()
    return g


def link(grid: pd.DataFrame, lnk: pd.DataFrame) -> pd.DataFrame:
    """Adds gvkey, linkprim and linktype: the CCM link valid at the formation date (missing if there is none).

    Among several valid links the order is linkprim P > C > J > N, then link type LC > LU > LS, then the most
    recent link date."""
    k = lnk[lnk["linktype"].isin(LINK_TYPES) & lnk["linkprim"].isin(LINKPRIM_ORDER) & lnk["lpermno"].notna()].copy()
    k["permno"] = k["lpermno"].astype("int32")
    k["prim"] = k["linkprim"].map({p: i for i, p in enumerate(LINKPRIM_ORDER)})
    k["kind"] = k["linktype"].map({t: i for i, t in enumerate(LINK_TYPES)})
    m = grid[["permno", "month", "formation_date"]].merge(
        k[["permno", "gvkey", "linkprim", "linktype", "linkdt", "linkenddt", "prim", "kind"]], on="permno")
    m = m[(m["linkdt"] <= m["formation_date"]) & (m["linkenddt"].isna() | (m["formation_date"] <= m["linkenddt"]))]
    m = m.sort_values(["permno", "month", "prim", "kind", "linkdt"], ascending=[True, True, True, True, False])
    m = m.drop_duplicates(["permno", "month"])[["permno", "month", "gvkey", "linkprim", "linktype"]]
    return grid.merge(m, on=["permno", "month"], how="left")


def foreign_filers(grid: pd.DataFrame, msf: pd.DataFrame) -> np.ndarray:
    """One boolean per row of `grid` (permno, month): the security is an ADR (`sharetype` AD) or a non-US issuer
    (`usincflg` N) at the formation date, so its annual report is a 20-F with a later deadline than a 10-K.

    `msf` is the CRSP monthly file (permno, yyyymm, sharetype, usincflg); its row of month-1 describes the security
    at the formation date of holding month `month`. Without that row the security's latest earlier row decides; with
    no earlier row (the monthly file starts after the grid) its earliest row; a security not in `msf` is a US filer."""
    n = len(grid)
    m = msf[msf["permno"].isin(grid["permno"].unique())]
    m = pd.DataFrame({"permno": m["permno"].to_numpy().astype("int64"),
                      "at": ((m["yyyymm"] // 100 - 1970) * 12 + m["yyyymm"] % 100 - 1).to_numpy().astype("int64"),
                      "foreign": (m["sharetype"].eq("AD") | m["usincflg"].eq("N")).fillna(False).to_numpy(dtype=bool)})
    m = m.sort_values(["at", "permno"]).drop_duplicates(["permno", "at"], keep="last")
    g = pd.DataFrame({"permno": grid["permno"].to_numpy().astype("int64"),
                      "at": pd.PeriodIndex(grid["month"], freq="M").asi8 - 1, "row": np.arange(n)})
    hit = pd.merge_asof(g.sort_values(["at", "permno"]), m, on="at", by="permno", direction="backward")
    at_formation = hit.set_index("row")["foreign"].reindex(np.arange(n)).to_numpy(dtype=float, na_value=np.nan)
    earliest = g["permno"].map(m.drop_duplicates("permno").set_index("permno")["foreign"]).to_numpy(dtype=float, na_value=np.nan)
    return np.nan_to_num(np.where(np.isnan(at_formation), earliest, at_formation), nan=0.0).astype(bool)


# ---------------------------------------------------------------------------------------------------- fiscal years
def clean_funda(funda: pd.DataFrame) -> pd.DataFrame:
    """One row per (gvkey, fyear), sorted. Of two rows with the same fiscal year (a change of fiscal year-end, where
    one of the two is empty in every case on disk) the one with total assets is kept, then the later year-end."""
    f = funda[funda["fyear"].notna()].copy()
    f["fyear"] = f["fyear"].astype("int64")
    f["_has_at"] = f["at"].notna()
    f = f.sort_values(["gvkey", "fyear", "_has_at", "datadate"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    return f.drop(columns="_has_at").reset_index(drop=True)


def strict_items(strict: Strict) -> tuple[str, ...]:
    """The zero-fill items that `strict` turns off, in register order: all of them for True, none for False."""
    if isinstance(strict, bool):
        return ZERO_FILL_ITEMS if strict else ()
    unknown = set(strict) - set(ZERO_FILL_ITEMS)
    if unknown:
        raise ValueError(f"not zero-filled inputs: {sorted(unknown)}; choose from {ZERO_FILL_ITEMS}")
    return tuple(i for i in ZERO_FILL_ITEMS if i in strict)


def _zero(f: pd.DataFrame, item: str, strict: Strict) -> pd.Series:
    """THE zero-fill, used for every secondary item: a missing value counts as 0, unless `strict` is True or names
    the item (then it stays missing and so does the characteristic)."""
    off = strict if isinstance(strict, bool) else item in strict
    return f[item] if off else f[item].fillna(0.0)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, missing where the denominator is 0 or the result is not finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den.where(den != 0)
    return r.where(np.isfinite(r))


def book_equity(f: pd.DataFrame, strict: Strict = PRODUCTION_STRICT) -> pd.Series:
    """BE = SH - PS + TXDITC (the last only up to fiscal 1992). SH = SEQ, else CEQ + PSTK, else AT - LT.
    PS = PSTKRV, else PSTKL, else PSTK; 0 if none. A missing TXDITC counts as 0. Negative book equity is kept."""
    f = f.assign(ps=f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]))
    sh = f["seq"].fillna(f["ceq"] + f["pstk"]).fillna(f["at"] - f["lt"])
    tax = _zero(f, "txditc", strict).where(f["fyear"] <= TXDITC_LAST_FYEAR, 0.0)
    return sh - _zero(f, "ps", strict) + tax


def fiscal_years(funda: pd.DataFrame, strict: Strict = PRODUCTION_STRICT) -> pd.DataFrame:
    """Per firm and fiscal year: gvkey, fyear, datadate, curcd, the 18 characteristics that need no CRSP price
    (ANNUAL_CHARS) and the numerators be, cfp and ib of BEME, CF2P and E2P (A2ME's numerator is AT).

    `strict` says which zero-fills are off (module docstring): the production list by default, False for the CPZ
    convention (every fill on), True for none.
    "Lagged" is the firm's previous fiscal year (fyear - 1). If it has no row, or reports in another currency (the
    year a firm switches between CAD and USD), the characteristics that use it are missing."""
    f = clean_funda(funda)
    strict = strict_items(strict)
    z = lambda item: _zero(f, item, strict)                                    # noqa: E731
    by_firm = f.groupby("gvkey")
    comparable = (by_firm["fyear"].diff() == 1) & (f["curcd"] == by_firm["curcd"].shift())
    lag = lambda s: s.groupby(f["gvkey"]).shift().where(comparable)            # noqa: E731

    be = book_equity(f, strict)
    at_lag = lag(f["at"])
    debt = z("dltt") + z("dlc")
    # net operating assets = operating assets - operating liabilities
    noa = (f["at"] - f["che"] - z("ivao")) - (f["at"] - z("dlc") - z("dltt") - z("mib") - z("pstk") - z("ceq"))
    wc = f["act"] - f["che"] - z("lct") - z("dlc") - z("txp") - z("dp")        # non-cash working capital less depreciation
    invt = z("invt")

    out = f[["gvkey", "fyear", "datadate", "curcd"]].copy()
    out["AT"] = f["at"]
    out["C"] = _ratio(f["che"], f["at"])
    out["CF"] = _ratio(f["ni"] + z("dp") - z("wcapch") - z("capx"), be)
    out["Q"] = _ratio(f["at"] + f["csho"] * f["prcc_f"] - z("ceq") - z("txdb"), f["at"])
    out["Lev"] = _ratio(debt, debt + f["seq"])
    out["PROF"] = _ratio(f["revt"] - f["cogs"], be)
    out["CTO"] = _ratio(f["sale"], at_lag)
    out["FC2Y"] = _ratio(z("xsga") + z("xrd") + z("xad"), f["sale"])
    out["OP"] = _ratio(f["revt"] - z("cogs") - z("tie") - z("xsga"), be)
    out["PM"] = _ratio(f["oiadp"], f["sale"])
    out["RNA"] = _ratio(f["oiadp"], lag(noa))
    out["D2A"] = _ratio(f["dp"], f["at"])
    out["Investment"] = _ratio(f["at"] - at_lag, at_lag)
    out["NOA"] = _ratio(noa, at_lag)
    out["DPI2A"] = _ratio((f["ppegt"] - lag(f["ppegt"])) + (invt - lag(invt)), at_lag)
    out["OA"] = _ratio(wc - lag(wc), at_lag)
    out["OL"] = _ratio(f["cogs"] + z("xsga"), f["at"])
    out["PCM"] = _ratio(f["sale"] - f["cogs"], f["sale"])
    out["be"] = be
    out["cfp"] = f["ib"] + z("dp") + z("txdb")
    out["ib"] = f["ib"]
    return out


# ---------------------------------------------------------------------------------------------------- timing
def june_year(month: pd.Series) -> pd.Series:
    """Y such that the holding month lies in July of Y .. June of Y+1."""
    return (month.dt.year - (month.dt.month <= 6)).astype("int64")


def assign_fiscal_year(links: pd.DataFrame, fy: pd.DataFrame, fiscal_rule: str = "safe",
                       min_lag_months: int = MIN_LAG_MONTHS,
                       min_lag_months_foreign: int = MIN_LAG_MONTHS_FOREIGN) -> pd.DataFrame:
    """(permno, month, gvkey) plus fyear, datadate and cap_year (the December whose market cap the ratios use).

    fyear is missing when the firm has no row for the fiscal year the rule asks for and, under "safe", when that
    year is not yet available and the year before it is not there either. An older fiscal year is never used in
    place of a row that does not exist (module docstring, limits of "safe").
    "safe" needs the boolean column `foreign_filer` in `links` (`foreign_filers`): those rows wait
    `min_lag_months_foreign` months after the fiscal year-end, the others `min_lag_months`."""
    if fiscal_rule not in FISCAL_RULES:
        raise ValueError(f"fiscal_rule must be one of {FISCAL_RULES}")
    if fiscal_rule == "safe" and "foreign_filer" not in links.columns:
        raise ValueError("fiscal rule 'safe' needs links['foreign_filer'] (see foreign_filers): the filing lag depends on it")
    a = links[["permno", "month", "gvkey"]].reset_index(drop=True)
    y = june_year(a["month"])
    ends = fy[["gvkey", "fyear", "datadate"]]

    def lookup(want: pd.Series, on: str = "fyear", table: pd.DataFrame = ends) -> pd.DataFrame:    # rows aligned with a
        key = pd.DataFrame({"gvkey": a["gvkey"].to_numpy(), on: want.to_numpy()})
        hit = key.merge(table, on=["gvkey", on], how="left")
        return hit.assign(fyear=hit["fyear"].where(hit["datadate"].notna()))      # no row: no fiscal year

    if fiscal_rule == "ff":                                # the fiscal year ending in calendar year Y-1 (the latest one)
        e = ends.assign(end_year=ends["datadate"].dt.year).sort_values(["gvkey", "datadate"])
        hit = lookup(y - 1, "end_year", e.drop_duplicates(["gvkey", "end_year"], keep="last"))
    else:
        hit = lookup(y - 1)
    a["fyear"], a["datadate"], a["cap_year"] = hit["fyear"].astype("Int64"), hit["datadate"], y - 1
    if fiscal_rule == "safe":
        start, foreign = a["month"].dt.start_time, links["foreign_filer"].to_numpy(dtype=bool)

        def due(datadate: pd.Series) -> pd.Series:             # the fiscal year-end plus the filing lag of the security
            return (datadate + pd.DateOffset(months=min_lag_months)).mask(
                foreign, datadate + pd.DateOffset(months=min_lag_months_foreign))

        late = (due(a["datadate"]) > start).to_numpy()                        # the deadline has not passed: keep what June used
        prev = lookup(y - 2)
        prev = prev.where(due(prev["datadate"]) <= start)
        a.loc[late, "fyear"] = prev.loc[late, "fyear"].astype("Int64")
        a.loc[late, "datadate"] = prev.loc[late, "datadate"]
        a.loc[late, "cap_year"] = y[late] - 2
    return a


def december_caps(daily: pd.DataFrame) -> pd.DataFrame:
    """(permno, cap_year, me_dec): the security's market cap in USD millions on its last trading day with a cap in
    December of cap_year (equal to the monthly file's `mthcap` of December). `daily` has permno, dlycaldt, dlycap."""
    d = daily.loc[(daily["dlycaldt"].dt.month == 12) & (daily["dlycap"] > 0), ["permno", "dlycaldt", "dlycap"]]
    d = d.sort_values(["permno", "dlycaldt"])
    d = d.assign(cap_year=d["dlycaldt"].dt.year).drop_duplicates(["permno", "cap_year"], keep="last")
    return pd.DataFrame({"permno": d["permno"].to_numpy().astype("int32"), "cap_year": d["cap_year"].to_numpy().astype("int64"),
                         "me_dec": d["dlycap"].to_numpy() / 1000.0})


# ---------------------------------------------------------------------------------------------------- the panel
def characteristics(links: pd.DataFrame, fy: pd.DataFrame, caps: pd.DataFrame, fiscal_rule: str = "safe",
                    min_lag_months: int = MIN_LAG_MONTHS,
                    min_lag_months_foreign: int = MIN_LAG_MONTHS_FOREIGN) -> pd.DataFrame:
    """permno, month, gvkey, fyear, datadate, curcd, cap_year and the 22 characteristics, one row per row of `links`.

    `links` from `link` (with `foreign_filer` for the rule "safe"), `fy` from `fiscal_years`, `caps` from
    `december_caps`."""
    a = assign_fiscal_year(links, fy, fiscal_rule, min_lag_months, min_lag_months_foreign)
    cols = ["gvkey", "fyear", "curcd", *ANNUAL_CHARS, "be", "cfp", "ib"]
    a = a.merge(fy[cols].astype({"fyear": "Int64"}), on=["gvkey", "fyear"], how="left")
    a = a.merge(caps, on=["permno", "cap_year"], how="left")
    for char, num in ME_RATIOS.items():
        a[char] = _ratio(a[num], a["me_dec"])
    a["permno"] = a["permno"].astype("int32")
    return a[["permno", "month", "gvkey", "fyear", "datadate", "curcd", "cap_year", *CHARS]]


# ---------------------------------------------------------------------------------------------------- build and check
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(links, funda, caps) from Data\\WRDS\\raw: the linked month grid with `foreign_filer`, Compustat for the linked
    firms, December caps."""
    from afsa.wrds_data import load

    daily = load("crsp_dsf_v2", columns=["permno", "dlycaldt", "dlycap"])
    links = link(month_grid(daily), load("ccm_lnkhist"))
    links["foreign_filer"] = foreign_filers(links, load(MSF_FILE, columns=["permno", "yyyymm", "sharetype", "usincflg"]))
    caps = december_caps(daily)
    funda = load("comp_funda", columns=FUNDA_COLUMNS)
    funda = funda[funda["gvkey"].isin(set(links["gvkey"].dropna()))]
    return links, funda, caps


def out_file(fiscal_rule: str = "safe", strict: Strict = PRODUCTION_STRICT):
    """accounting.parquet is the production build ("safe", the production fills); any other build gets its own file:
    accounting_allfills.parquet (every fill on, the CPZ convention), accounting_strict.parquet (none),
    accounting_strict-ceq-lct.parquet, accounting_pelger.parquet, ..."""
    off = strict_items(strict)
    rule = "" if fiscal_rule == "safe" else f"_{fiscal_rule}"
    fills = ("" if set(off) == set(PRODUCTION_STRICT) else "_allfills" if not off else
             "_strict" if len(off) == len(ZERO_FILL_ITEMS) else "_strict-" + "-".join(sorted(off)))
    return OUT_DIR / f"accounting{rule}{fills}.parquet"


def build(fiscal_rule: str = "safe", strict: Strict = PRODUCTION_STRICT) -> pd.DataFrame:
    t0 = time.time()
    off = strict_items(strict)                              # raises on an unknown item before anything is read
    links, funda, caps = load_inputs()
    out = characteristics(links, fiscal_years(funda, off), caps, fiscal_rule)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = out_file(fiscal_rule, off)
    links.to_parquet(LINK_FILE, index=False)
    out.to_parquet(path, index=False)
    print(f"{LINK_FILE}: {len(links):,} security-months, {links.permno.nunique():,} PERMNOs, "
          f"{links.gvkey.isna().mean():.2%} without GVKEY, {links.foreign_filer.mean():.2%} foreign filers")
    print(f"{path}: {len(out):,} rows, {out.month.min()} to {out.month.max()}, fiscal_rule={fiscal_rule}, "
          f"{'no zero-fill for ' + ', '.join(off) if off else 'every zero-fill on'}, {out.fyear.isna().mean():.2%} without a fiscal year "
          f"({time.time() - t0:.0f} s)")
    return out


def _in_universe(out: pd.DataFrame, universe: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """The universe's stock-months from 1990 on, with the columns of `out` (the 22 characteristics by default)."""
    u = universe.loc[universe["month"] >= pd.Period("1990-01", "M"), ["permno", "month"]].astype({"permno": "int32"})
    return u.merge(out[["permno", "month", *(columns or CHARS)]], on=["permno", "month"], how="left")


def universe_missing(out: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """Share of the universe's stock-months (1990 on) in which each characteristic is NaN."""
    return _in_universe(out, universe)[list(CHARS)].isna().mean()


def universe_ties(out: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Per characteristic, over the universe's stock-months (1990 on): `zero`, the share that is exactly 0; `tied`,
    the share whose value another stock of the same month also has; `largest`, the largest group of stocks with one
    value in one month. A zero-fill that makes a whole numerator shows up here (rank ties far from the median)."""
    m = _in_universe(out, universe)
    rows = {}
    for c in CHARS:
        size = m.groupby(["month", c]).size()               # NaN is not a group
        rows[c] = {"zero": (m[c] == 0).mean(), "tied": size[size > 1].sum() / len(m), "largest": int(size.max())}
    return pd.DataFrame(rows).T.astype({"largest": "int64"})


def filing_lags(out: pd.DataFrame, links: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """The filing lag at work in the universe's stock-months (1990 on), for US and foreign filers: `gated`, how many
    use the previous fiscal year because the newer one is too young; the age in days of the accounts at the formation
    date; `at_lag`, how many use accounts in the first month the lag allows, and `at_lag_to_2011`, how many of those
    are fiscal years ending before 2011-12-15. A foreign filer had 6 months to file these last: the deadline is the
    last day of the formation month, the limit described in the module docstring."""
    m = _in_universe(out, universe, ["datadate", "cap_year"]).merge(
        links[["permno", "month", "formation_date", "foreign_filer"]].astype({"permno": "int32"}), on=["permno", "month"], how="left")
    m["filer"] = np.where(m["foreign_filer"], "foreign", "US")
    used = m[m["datadate"].notna()]
    age = (used["month"].dt.year - used["datadate"].dt.year) * 12 + used["month"].dt.month - used["datadate"].dt.month
    at_lag = (age == np.where(used["foreign_filer"], MIN_LAG_MONTHS_FOREIGN, MIN_LAG_MONTHS) + 1).to_numpy()
    t = pd.DataFrame({"filer": used["filer"].to_numpy(), "gated": (used["cap_year"] == june_year(used["month"]) - 2).to_numpy(),
                      "days": (used["formation_date"] - used["datadate"]).dt.days.to_numpy(), "at_lag": at_lag,
                      "to_2011": at_lag & (used["datadate"] < pd.Timestamp("2011-12-15")).to_numpy()})
    g = t.groupby("filer")
    return pd.DataFrame({"stock_months": m.groupby("filer").size(), "with_accounts": g.size(), "gated": g["gated"].sum(),
                         "youngest_days": g["days"].min(), "median_days": g["days"].median(), "oldest_days": g["days"].max(),
                         "at_lag": g["at_lag"].sum(), "at_lag_to_2011": g["to_2011"].sum()})


def security_attributes(out: pd.DataFrame) -> pd.DataFrame:
    """`out` plus year, type (as in afsa.wrds_universe), currency and sector, from the monthly file at formation."""
    from afsa.wrds_data import load

    msf = load(MSF_FILE, columns=["permno", "yyyymm", "sharetype", "issuertype", "usincflg", "siccd"])
    # the row of month m-1 describes the security at the formation date of holding month m
    msf["month"] = pd.PeriodIndex.from_ordinals(((msf["yyyymm"] // 100 - 1970) * 12 + msf["yyyymm"] % 100).to_numpy(), freq="M")
    m = out.merge(msf.astype({"permno": "int32"}).drop(columns="yyyymm"), on=["permno", "month"], how="left")
    m["type"] = np.select([m.sharetype.isna(), m.sharetype == "AD", m.issuertype == "REIT", m.sharetype == "UG", m.usincflg == "N"],
                          ["unknown", "ADR", "REIT", "MLP", "foreign"], "common")
    sic = m["siccd"].astype("float64").to_numpy()
    m["sector"] = np.select([(sic >= 6000) & (sic < 6200), (sic >= 6300) & (sic < 6500), (sic >= 6000) & (sic < 7000)],
                            ["bank", "insurer", "other financial"], "non-financial")
    m["year"] = m["month"].dt.year
    m["years"] = pd.cut(m["year"], [1985, 1994, 2004, 2012, 2016], labels=["1986-1994", "1995-2004", "2005-2012", "2013-2016"]).astype(str)
    m["currency"] = m["curcd"].fillna("none")
    return m


def check(detail: bool = False) -> None:
    """Prints: (1) the panel match of the FORMULAS, with every zero-fill on, under the three fiscal rules; (2) the
    match of the production fills and of the production file; (3) the NaN shares of the universe under the three
    fill conventions, and fill by fill; (4) zeros and ties; (5) the filing lag at work."""
    from afsa.pelger_check import compare

    fmt4, fmt2 = (lambda v: f"{v:.4f}"), (lambda v: f"{v:.2f}")
    links, funda, caps = load_inputs()
    universe = pd.read_parquet(UNIVERSE_FILE, columns=["permno", "month"])
    in_u = _in_universe(links, universe, ["gvkey", "foreign_filer"])
    print(f"link: {len(links):,} security-months of {links.permno.nunique():,} PERMNOs; broad-universe stock-months "
          f"(1990-2025) without GVKEY: {in_u.gvkey.isna().sum()} of {len(in_u):,} ({in_u.gvkey.isna().mean():.3%}); "
          f"foreign filers: {in_u.foreign_filer.sum():,} ({in_u.foreign_filer.mean():.2%})")
    fy = {"all on": fiscal_years(funda, strict=False), "production": fiscal_years(funda), "all off": fiscal_years(funda, strict=True)}
    match = lambda out: compare(out, list(CHARS)).set_index("characteristic")              # noqa: E731

    # (1) the formulas: every fill on (the CPZ convention), under each fiscal rule
    outs = {rule: characteristics(links, fy["all on"], caps, rule) for rule in FISCAL_RULES}
    table = {rule: match(outs[rule]) for rule in FISCAL_RULES}
    c = table["pelger"]
    t = pd.DataFrame({r: table[r]["equal"] for r in FISCAL_RULES}).assign(
        n=c["n"], within1=c["within1"], spearman=c["spearman"], med_rel=c["med_rel"])
    print("\n(1) Formulas, every zero-fill on. Match with the CPZ panel: share of common stock-months equal to 1e-4, by "
          "fiscal rule\n(n, within1, spearman and med_rel under 'pelger'):")
    print(t.to_string(float_format=fmt4))

    # (2) the production fills (under 'pelger', so that only the fills differ from (1)) and the production file
    production = characteristics(links, fy["production"], caps, "safe")
    p, f = match(characteristics(links, fy["production"], caps, "pelger")), match(production)
    hits = lambda x: (x["equal"] * x["n"]).round().astype("int64")                          # noqa: E731
    t = pd.DataFrame({"fills_equal": p["equal"], "fills_n": p["n"], "n_lost": c["n"] - p["n"], "lost_equal": hits(c) - hits(p),
                      "file_equal": f["equal"], "file_n": f["n"]})
    print(f"\n(2) Production: no zero-fill for {', '.join(PRODUCTION_STRICT)}. fills_*: under 'pelger'; n_lost: panel rows "
          "that became NaN, lost_equal: how many\nof them were equal in (1); file_*: the production file ('safe').")
    print(t.to_string(float_format=fmt4))
    if OUT_FILE.exists():
        print(f"{OUT_FILE.name} on disk is this production build: {pd.read_parquet(OUT_FILE).equals(production)}")

    # (3) what is NaN in the universe
    missing = {name: universe_missing(characteristics(links, fy[name], caps, "safe"), universe) for name in fy}
    print("\n(3) Share of broad-universe stock-months (1990-2025) that are NaN, in %, by zero-fill convention ('safe'):")
    print((100 * pd.DataFrame(missing)).to_string(float_format=fmt2))
    rows = {}
    for item in ZERO_FILL_ITEMS:                           # one item off at a time: where that fill is at work
        one = universe_missing(characteristics(links, fiscal_years(funda, strict=(item,)), caps, "safe"), universe)
        rows[item] = 100 * (one - missing["all on"])
    by_item = pd.DataFrame(rows).loc[list(ZERO_FILLED)]
    print(f"\nExtra NaN, in % of universe stock-months, when one zero-fill alone is turned off (against 'all on'; blank = not "
          f"used;\noff in production: {', '.join(PRODUCTION_STRICT)}):")
    used = pd.DataFrame({i: [i in ZERO_FILLED[ch] for ch in by_item.index] for i in by_item.columns}, index=by_item.index)
    print(by_item.where(used).to_string(float_format=fmt2, na_rep=""))

    # (4) zeros and ties: a fill that makes a whole numerator puts many stocks on one value
    ties = {name: universe_ties(characteristics(links, fy[name], caps, "safe"), universe) for name in ("all on", "production")}
    t = pd.concat({name: d.assign(zero=100 * d["zero"], tied=100 * d["tied"]) for name, d in ties.items()}, axis=1)
    print("\n(4) Universe stock-months: exactly 0 (%), sharing their value with another stock of the month (%), and the "
          "largest group\nof equal values in one month:")
    print(t.to_string(float_format=fmt2))

    # (5) the filing lag
    print(f"\n(5) Filing lag of 'safe' in the universe: {MIN_LAG_MONTHS} months, {MIN_LAG_MONTHS_FOREIGN} for foreign filers (ADR or "
          "non-US issuer). Stock-months that use the previous fiscal year\n(gated), age of the accounts at the formation "
          "date in days, stock-months in the first month the lag allows (all, and fiscal years ending before 2011-12-15):")
    print(filing_lags(production, links, universe).to_string(float_format=lambda v: f"{v:.0f}"))

    if detail:
        m = security_attributes(outs["pelger"])
        show = ["AT", "A2ME", "BEME", "Q", "CF", "OP", "RNA", "OA", "PCM"]
        for by in ("years", "type", "currency", "sector"):
            d = compare(m, show, by=by)
            print(f"\nMatch under 'pelger', every zero-fill on, by {by}:")
            print(d.pivot(index=by, columns="characteristic", values="equal")[show].assign(
                n=d[d.characteristic == "AT"].set_index(by)["n"]).to_string(float_format=fmt4))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["build", "check"])
    p.add_argument("--fiscal-rule", default="safe", choices=FISCAL_RULES, help="build: which fiscal year a month uses")
    fills = p.add_mutually_exclusive_group()
    fills.add_argument("--strict", nargs="*", metavar="ITEM", help="build: no zero-fill at all (a missing secondary item "
                       "leaves NaN); with names, none for those and a fill for the rest: " + ", ".join(ZERO_FILL_ITEMS)
                       + ". Default: " + " ".join(PRODUCTION_STRICT))
    fills.add_argument("--all-fills", action="store_true", help="build: every zero-fill on, the CPZ convention")
    p.add_argument("--detail", action="store_true", help="check: also the match by year, security type, currency, sector")
    a = p.parse_args()
    pd.set_option("display.width", 250)
    if a.command == "build":
        build(a.fiscal_rule, False if a.all_fills else PRODUCTION_STRICT if a.strict is None else tuple(a.strict) or True)
    else:
        check(a.detail)


if __name__ == "__main__":
    main()
