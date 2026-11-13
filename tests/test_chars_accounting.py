"""The 22 accounting characteristics: every formula on hand-computed synthetic accounts, the three fiscal rules (with
the no-look-ahead property of "safe" and its longer lag for foreign filers), the link priority, the zero-fill
register, its strict switch and the production default; then, when the WRDS files are on disk, the match rates
against the Chen-Pelger-Zhu panel and the production file."""
import numpy as np
import pandas as pd
import pytest

from afsa import chars_accounting as ca

# One firm, two fiscal years. Every item is there, so every formula can be worked out by hand (see EXPECTED).
ITEMS = {  # item: (fiscal 2000, fiscal 2001)
    "at": (800, 1000), "lt": (500, 600), "seq": (300, 400), "ceq": (290, 380), "pstk": (10, 20), "pstkrv": (12, 25),
    "pstkl": (11, 22), "txditc": (20, 30), "txdb": (10, 15), "che": (80, 100), "ivao": (40, 50), "dlc": (30, 40),
    "dltt": (150, 160), "mib": (4, 5), "act": (400, 500), "lct": (250, 300), "txp": (8, 10), "dp": (50, 60),
    "ppegt": (600, 700), "invt": (100, 120), "sale": (1000, 1200), "revt": (1050, 1250), "cogs": (600, 700),
    "xsga": (180, 200), "xrd": (25, 30), "xad": (8, 10), "oiadp": (120, 150), "tie": (4, 5), "ni": (70, 90),
    "ib": (75, 95), "capx": (70, 80), "wcapch": (9, 12), "csho": (100, 100), "prcc_f": (8, 9),
}
ME_DEC = 2000.0             # the security's market cap of December 2001, USD millions
EXPECTED = {                # fiscal 2001; book equity = 400 - 25 = 375 (no deferred taxes after fiscal 1992)
    "AT": 1000.0,
    "A2ME": 1000 / ME_DEC,
    "BEME": 375 / ME_DEC,
    "C": 100 / 1000,
    "CF": (90 + 60 - 12 - 80) / 375,
    "CF2P": (95 + 60 + 15) / ME_DEC,
    "Q": (1000 + 100 * 9 - 380 - 15) / 1000,
    "Lev": (160 + 40) / (160 + 40 + 400),
    "E2P": 95 / ME_DEC,
    "PROF": (1250 - 700) / 375,
    "CTO": 1200 / 800,
    "FC2Y": (200 + 30 + 10) / 1200,
    "OP": (1250 - 700 - 5 - 200) / 375,
    "PM": 150 / 1200,
    "RNA": 150 / ((800 - 80 - 40) - (800 - 30 - 150 - 4 - 10 - 290)),            # over the NOA of fiscal 2000 = 364
    "D2A": 60 / 1000,
    "Investment": 1000 / 800 - 1,
    "NOA": ((1000 - 100 - 50) - (1000 - 40 - 160 - 5 - 20 - 380)) / 800,          # 455 / 800
    "DPI2A": ((700 - 600) + (120 - 100)) / 800,
    "OA": ((500 - 100 - 300 - 40 - 10 - 60) - (400 - 80 - 250 - 30 - 8 - 50)) / 800,   # (-10 + 18) / 800
    "OL": (700 + 200) / 1000,
    "PCM": (1200 - 700) / 1200,
}


def make_funda(first_fyear: int = 2000, end_month: int = 12, gvkey: str = "001", curcd=("USD", "USD"), **blank) -> pd.DataFrame:
    """The two fiscal years of ITEMS for one firm; `blank={"dp": True}` removes an item from both years."""
    rows = []
    for k in range(2):
        year = first_fyear + k + (end_month <= 5)                 # Compustat: year-ends of January-May belong to fyear-1
        row = {"gvkey": gvkey, "datadate": pd.Timestamp(year, end_month, 1) + pd.offsets.MonthEnd(0),
               "fyear": float(first_fyear + k), "curcd": curcd[k]}
        row.update({item: float(v[k]) for item, v in ITEMS.items()})
        rows.append(row)
    f = pd.DataFrame(rows)[ca.FUNDA_COLUMNS]
    for item in blank:
        f[item] = np.nan
    return f


def make_links(months, permno: int = 1, gvkey: str = "001", foreign: bool = False) -> pd.DataFrame:
    return pd.DataFrame({"permno": np.int32(permno), "month": pd.PeriodIndex(months, freq="M"), "gvkey": gvkey,
                         "foreign_filer": foreign})


def make_caps(years, permno: int = 1, me: float = ME_DEC) -> pd.DataFrame:
    return pd.DataFrame({"permno": np.int32(permno), "cap_year": np.asarray(years, dtype="int64"), "me_dec": me})


def one_row(funda: pd.DataFrame, strict=False, month: str = "2002-07") -> pd.Series:
    """The characteristics of one holding month under the panel's rule (2002-07 uses fiscal 2001 and December 2001).
    `strict=False` is the CPZ convention (every zero-fill on), not the production default."""
    year = pd.Period(month, "M").year
    out = ca.characteristics(make_links([month]), ca.fiscal_years(funda, strict), make_caps(range(year - 3, year + 1)), "pelger")
    return out.iloc[0]


# ---------------------------------------------------------------------------------------------------- formulas
@pytest.mark.parametrize("char", ca.CHARS)
def test_formula(char):
    row = one_row(make_funda())
    assert row["fyear"] == 2001 and row["cap_year"] == 2001
    assert row[char] == pytest.approx(EXPECTED[char], rel=1e-12)


def test_all_22_characteristics_have_an_expected_value():
    assert len(ca.CHARS) == 22 and set(EXPECTED) == set(ca.CHARS)
    assert set(ca.ANNUAL_CHARS) | set(ca.ME_RATIOS) == set(ca.CHARS)


def test_book_equity_rules():
    f = ca.clean_funda(make_funda(1992))                                   # fiscal 1992 and 1993
    assert ca.book_equity(f).tolist() == [300 - 12 + 20, 400 - 25]         # deferred taxes only up to fiscal 1992
    # preferred stock: redemption, else liquidating, else par value, else 0
    assert ca.book_equity(f.assign(pstkrv=np.nan)).tolist() == [300 - 11 + 20, 400 - 22]
    assert ca.book_equity(f.assign(pstkrv=np.nan, pstkl=np.nan)).tolist() == [300 - 10 + 20, 400 - 20]
    assert ca.book_equity(f.assign(pstkrv=np.nan, pstkl=np.nan, pstk=np.nan)).tolist() == [300 + 20, 400]
    # shareholders' equity: SEQ, else CEQ + PSTK, else AT - LT
    assert ca.book_equity(f.assign(seq=np.nan)).tolist() == [290 + 10 - 12 + 20, 380 + 20 - 25]
    assert ca.book_equity(f.assign(seq=np.nan, ceq=np.nan)).tolist() == [800 - 500 - 12 + 20, 1000 - 600 - 25]
    # a year-end in January-May 1993 is fiscal 1992 for Compustat and still gets its deferred taxes (as in the panel)
    g = ca.clean_funda(make_funda(1992, end_month=4))
    assert g.datadate.dt.year.tolist() == [1993, 1994] and ca.book_equity(g).tolist() == [300 - 12 + 20, 400 - 25]


def test_lagged_values_need_the_previous_fiscal_year_in_the_same_currency():
    lagged = ["CTO", "RNA", "Investment", "NOA", "DPI2A", "OA"]
    gap = make_funda()
    gap.loc[0, "fyear"] = 1999.0                                           # the previous row is two years back
    switch = make_funda(curcd=("CAD", "USD"))
    for f in (gap, switch):
        row = ca.fiscal_years(f).iloc[-1]
        assert row[lagged].isna().all()
        assert row[[c for c in ca.ANNUAL_CHARS if c not in lagged]].notna().all()


def test_zero_denominators_and_duplicate_fiscal_years():
    row = ca.fiscal_years(make_funda().assign(sale=0.0)).iloc[-1]
    assert row[["FC2Y", "PM", "PCM"]].isna().all() and row["CTO"] == 0
    f = make_funda()
    empty = f.iloc[[1]].copy()                                             # a second, empty row for fiscal 2001
    empty[[c for c in ITEMS]] = np.nan
    empty["datadate"] = pd.Timestamp("2002-03-31")
    both = pd.concat([f, empty, f.iloc[[1]].assign(fyear=np.nan)], ignore_index=True)
    c = ca.clean_funda(both)
    assert c.fyear.tolist() == [2000, 2001] and c["at"].tolist() == [800, 1000]
    assert c.datadate.tolist() == [pd.Timestamp("2000-12-31"), pd.Timestamp("2001-12-31")]


# ---------------------------------------------------------------------------------------------------- zero-fills
def _blanked(item: str) -> pd.DataFrame:
    """Fiscal 1991-1992 (when deferred taxes still count) with one zero-filled item removed from both years."""
    cols = ("pstkrv", "pstkl", "pstk") if item == "ps" else (item,)
    return make_funda(1991, **{c: True for c in cols})


@pytest.mark.parametrize("item", ca.ZERO_FILL_ITEMS)
def test_zero_fill_register_is_what_the_code_does(item):
    """A missing item listed for a characteristic counts as 0, and leaves it NaN when strict; a missing item that is
    not listed changes nothing between strict and not (the characteristic is NaN both ways or does not use it)."""
    f = _blanked(item)
    filled, strict_all, strict_one = (one_row(f, s, "1993-07") for s in (False, True, (item,)))
    for char in ca.CHARS:
        if item in ca.ZERO_FILLED.get(char, ()):
            assert np.isfinite(filled[char]), (char, item)
            assert np.isnan(strict_all[char]) and np.isnan(strict_one[char]), (char, item)
        else:
            assert (np.isnan(filled[char]) and np.isnan(strict_one[char])) or filled[char] == strict_one[char], (char, item)
    # the production default: the fill is off exactly for the items of PRODUCTION_STRICT
    production = ca.characteristics(make_links(["1993-07"]), ca.fiscal_years(f), make_caps(range(1990, 1994)), "pelger").iloc[0]
    pd.testing.assert_series_equal(production, one_row(f, ca.PRODUCTION_STRICT, "1993-07"))
    pd.testing.assert_series_equal(production, strict_one if item in ca.PRODUCTION_STRICT else filled)
    for char in ca.CHARS:
        assert (item in ca.PRODUCTION_FILLED.get(char, ())) == (np.isfinite(production[char]) and np.isnan(strict_one[char])), (char, item)


def test_production_default_fills_only_what_the_panel_confirms():
    assert ca.PRODUCTION_STRICT == ("xsga", "dp", "cogs", "ceq", "lct") and set(ca.PRODUCTION_STRICT) < set(ca.ZERO_FILL_ITEMS)
    assert ca.strict_items(ca.PRODUCTION_STRICT) == ("dp", "ceq", "xsga", "cogs", "lct")                     # register order
    assert ca.strict_items(True) == ca.ZERO_FILL_ITEMS and ca.strict_items(False) == () and ca.strict_items(()) == ()
    assert {c: set(i) for c, i in ca.PRODUCTION_FILLED.items()} == {
        "BEME": {"ps", "txditc"}, "CF": {"wcapch", "capx", "ps", "txditc"}, "CF2P": {"txdb"}, "Q": {"txdb"},
        "Lev": {"dltt", "dlc"}, "PROF": {"ps", "txditc"}, "FC2Y": {"xrd", "xad"}, "OP": {"tie", "ps", "txditc"},
        "RNA": {"ivao", "dlc", "dltt", "mib", "pstk"}, "NOA": {"ivao", "dlc", "dltt", "mib", "pstk"}, "DPI2A": {"invt"},
        "OA": {"dlc", "txp"}, "OL": set()}
    production = lambda f: ca.fiscal_years(f).iloc[-1]                     # noqa: E731  (fiscal 2001, the default fills)
    cpz = lambda f: ca.fiscal_years(f, strict=False).iloc[-1]              # noqa: E731
    # no selling, research or advertising expense reported: FC2Y is unknown, not 0 (the audit's finding)
    f = make_funda(xsga=True, xrd=True, xad=True)
    assert np.isnan(production(f)[["FC2Y", "OP", "OL"]].astype(float)).all()
    assert cpz(f)["FC2Y"] == 0 and cpz(f)["OL"] == 700 / 1000 and cpz(f)["OP"] == pytest.approx((1250 - 700 - 5) / 375)
    f = make_funda(xrd=True, xad=True)                                     # XSGA alone is enough: the other two count as 0
    assert production(f)["FC2Y"] == pytest.approx(200 / 1200) and production(f)["OP"] == pytest.approx(EXPECTED["OP"])
    # depreciation: CF, CF2P and OA are unknown, like D2A
    f, lagged = make_funda(dp=True), ["CTO", "RNA", "Investment", "NOA", "DPI2A"]
    assert np.isnan(production(f)[["CF", "cfp", "OA", "D2A"]].astype(float)).all()
    assert cpz(f)["CF"] == pytest.approx((90 - 12 - 80) / 375) and cpz(f)["cfp"] == 95 + 15 and np.isfinite(cpz(f)["OA"])
    assert production(f)[lagged].notna().all()
    # common equity (Q, NOA, and RNA through last year's NOA) and current liabilities (OA)
    f = make_funda(ceq=True, lct=True)
    assert np.isnan(production(f)[["Q", "NOA", "RNA", "OA"]].astype(float)).all()
    assert cpz(f)["Q"] == pytest.approx((1000 + 900 - 15) / 1000) and cpz(f)[["NOA", "RNA", "OA"]].notna().all()
    assert np.isnan(production(make_funda(cogs=True))["OP"]) and np.isnan(cpz(make_funda(cogs=True))["PROF"])
    # the confirmed fills stay on in production, with the same values as under the CPZ convention
    f = make_funda(wcapch=True, capx=True, xrd=True, xad=True, tie=True, mib=True, ivao=True, txdb=True, txp=True, invt=True,
                   dltt=True, dlc=True, pstk=True, pstkrv=True, pstkl=True, txditc=True)
    assert production(f)[list(ca.ANNUAL_CHARS)].notna().all()
    pd.testing.assert_series_equal(production(f), cpz(f))
    pd.testing.assert_series_equal(production(make_funda()), cpz(make_funda()))    # complete accounts: no difference


def test_zero_fill_values_and_strict_on_complete_accounts():
    complete = make_funda(1991)
    for strict in (True, ("dp", "xrd"), ca.PRODUCTION_STRICT):             # nothing is missing: strict changes nothing
        pd.testing.assert_series_equal(one_row(complete, strict, "1993-07"), one_row(complete, False, "1993-07"))
    with pytest.raises(ValueError):
        ca.fiscal_years(complete, strict=("at",))                          # not a zero-filled input
    row = one_row(make_funda(xrd=True, xad=True))
    assert row["FC2Y"] == pytest.approx(200 / 1200)
    row = one_row(make_funda(invt=True, txp=True, tie=True, wcapch=True))
    assert row["DPI2A"] == pytest.approx(100 / 800)
    assert row["OA"] == pytest.approx(((500 - 100 - 300 - 40 - 60) - (400 - 80 - 250 - 30 - 50)) / 800)
    assert row["OP"] == pytest.approx((1250 - 700 - 200) / 375)
    assert row["CF"] == pytest.approx((90 + 60 - 80) / 375)
    # main items are never filled
    row = one_row(make_funda(dp=True, cogs=True, act=True, ppegt=True, che=True))
    assert row[["D2A", "PROF", "OL", "PCM", "OA", "DPI2A", "C", "NOA"]].isna().all()
    assert set(ca.ZERO_FILLED) <= set(ca.CHARS) and "AT" not in ca.ZERO_FILLED


# ---------------------------------------------------------------------------------------------------- fiscal rules
def _timing(end_month: int, rule: str, months, foreign: bool = False, **lags) -> pd.DataFrame:
    f = pd.concat([make_funda(1999, end_month), make_funda(2001, end_month)], ignore_index=True)    # fiscal 1999-2002
    return ca.assign_fiscal_year(make_links(months, foreign=foreign), ca.fiscal_years(f), rule, **lags).set_index("month")


def test_fiscal_rule_pelger_uses_compustat_fyear():
    months = pd.period_range("2002-01", "2003-12", freq="M")
    for end_month in (12, 6, 4):
        a = _timing(end_month, "pelger", months)
        assert (a.loc["2002-01":"2002-06", "fyear"] == 2000).all() and (a.loc["2002-01":"2002-06", "cap_year"] == 2000).all()
        assert (a.loc["2002-07":"2003-06", "fyear"] == 2001).all() and (a.loc["2002-07":"2003-06", "cap_year"] == 2001).all()
        assert (a.loc["2003-07":, "fyear"] == 2002).all()
    # the look-ahead of the panel: a year ending on 2002-04-30 is used from July 2002, two months later
    assert _timing(4, "pelger", months).loc["2002-07", "datadate"] == pd.Timestamp("2002-04-30")


def test_fiscal_rule_ff_uses_the_calendar_year_of_the_year_end():
    months = pd.period_range("2002-07", "2003-06", freq="M")
    assert (_timing(12, "ff", months).datadate == pd.Timestamp("2001-12-31")).all()
    assert (_timing(6, "ff", months).datadate == pd.Timestamp("2001-06-30")).all()
    a = _timing(4, "ff", months)                                           # ends in April 2001: Compustat's fiscal 2000
    assert (a.datadate == pd.Timestamp("2001-04-30")).all() and (a.fyear == 2000).all() and (a.cap_year == 2001).all()


def test_fiscal_rule_safe_waits_four_months():
    months = pd.period_range("2002-01", "2003-06", freq="M")
    for end_month in (12, 6, 9, 2):                                        # old enough by July: same as the panel's rule
        pd.testing.assert_frame_equal(_timing(end_month, "safe", months), _timing(end_month, "pelger", months))
    for end_month, first in ((3, "2002-08"), (4, "2002-09"), (5, "2002-10")):
        a, b = _timing(end_month, "safe", months), _timing(end_month, "pelger", months)
        gated = (a.index >= pd.Period("2002-07", "M")) & (a.index < pd.Period(first, "M"))
        assert gated.sum() == end_month - 2
        assert (a.fyear[gated] == 2000).all() and (a.cap_year[gated] == 2000).all()      # what June 2002 used
        assert (a.datadate[gated] == b.loc["2002-06", "datadate"]).all()
        pd.testing.assert_frame_equal(a[~gated], b[~gated])
    # the first fiscal year of a firm is not used early either, and there is nothing to fall back on
    a = _timing(4, "safe", pd.period_range("2000-07", "2000-10", freq="M"))             # fiscal 1999 ends 2000-04-30
    assert a.fyear.isna().tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        _timing(12, "fama", months)


def test_fiscal_rule_safe_waits_six_months_for_foreign_filers():
    """A 20-F was due 6 months after the year-end: an ADR or a non-US issuer waits 6 months, in every year."""
    months = pd.period_range("2002-01", "2003-06", freq="M")
    for end_month in (12, 6, 9, 11):                                       # 6 months old by July: same as the panel's rule
        pd.testing.assert_frame_equal(_timing(end_month, "safe", months, foreign=True), _timing(end_month, "pelger", months))
    # a year ending in January..May 2002 enters in August..December 2002 (a US filer: July, July, August, September, October)
    for end_month, first in ((1, "2002-08"), (2, "2002-09"), (3, "2002-10"), (4, "2002-11"), (5, "2002-12")):
        a, b = _timing(end_month, "safe", months, foreign=True), _timing(end_month, "pelger", months)
        gated = (a.index >= pd.Period("2002-07", "M")) & (a.index < pd.Period(first, "M"))
        assert gated.sum() == end_month
        assert (a.fyear[gated] == 2000).all() and (a.cap_year[gated] == 2000).all()      # what June 2002 used
        assert (a.datadate[gated] == b.loc["2002-06", "datadate"]).all()
        pd.testing.assert_frame_equal(a[~gated], b[~gated])
        us = _timing(end_month, "safe", months)
        assert (us.fyear >= a.fyear).all() and (us.fyear > a.fyear).sum() == min(end_month, 2)   # the US filer is 1 or 2 months ahead
    # the two lags are parameters; with equal lags a foreign filer is treated as a US one
    for end_month in range(1, 13):
        pd.testing.assert_frame_equal(_timing(end_month, "safe", months, foreign=True, min_lag_months_foreign=4),
                                      _timing(end_month, "safe", months))
        pd.testing.assert_frame_equal(_timing(end_month, "safe", months, min_lag_months=6),
                                      _timing(end_month, "safe", months, foreign=True))
    # the flag is read row by row: a security that becomes a foreign filer in October 2002 falls back to fiscal 2000
    links = make_links(months)
    links["foreign_filer"] = links.month >= pd.Period("2002-10", "M")
    f = ca.fiscal_years(pd.concat([make_funda(1999, 4), make_funda(2001, 4)]))                      # year-ends in April
    a = ca.assign_fiscal_year(links, f, "safe").set_index("month")
    assert a.loc["2002-07":"2002-12", "fyear"].tolist() == [2000, 2000, 2001, 2000, 2001, 2001]     # US: from September; foreign: from November
    # without the flag "safe" cannot know the lag and refuses; the other rules do not need it
    bare = make_links(months).drop(columns="foreign_filer")
    with pytest.raises(ValueError, match="foreign_filer"):
        ca.assign_fiscal_year(bare, f, "safe")
    assert ca.assign_fiscal_year(bare, f, "pelger").fyear.notna().all()
    # the foreign lag reaches the values through characteristics() with its default arguments, not only the assignment:
    # an April 2002 year-end is used by a foreign filer from November 2002 (a US filer: from September)
    caps = pd.concat([make_caps([2000], me=1000.0), make_caps([2001])])
    out = ca.characteristics(make_links(["2002-09", "2002-10", "2002-11"], foreign=True), f, caps, "safe")
    assert out.fyear.tolist() == [2000, 2000, 2001] and out.AT.tolist() == [1000.0, 1000.0, 800.0]
    assert ca.characteristics(make_links(["2002-09"]), f, caps, "safe").fyear.tolist() == [2001]
    # the 6 months hold in every year, also after the 20-F deadline became 4 months (fiscal years ending from 2011-12-15)
    late = ca.fiscal_years(pd.concat([make_funda(2011, 4), make_funda(2013, 4)]))                   # ... 2013-04-30, 2014-04-30
    a = ca.assign_fiscal_year(make_links(pd.period_range("2014-09", "2014-11", freq="M"), foreign=True), late, "safe")
    assert a.fyear.tolist() == [2012, 2012, 2013]


def test_foreign_filers_reads_the_monthly_file_at_the_formation_month():
    msf = pd.DataFrame([                       # permno, yyyymm, sharetype, usincflg
        (1, 200001, "NS", "Y"), (1, 200002, "NS", "Y"), (1, 200003, "AD", "Y"), (1, 200005, "NS", "N"), (1, 200006, "NS", "Y"),
        (2, 200003, "AD", "N"), (2, 200004, "AD", "N"),
        (3, 200002, "NS", "Y"), (3, 200003, "NS", None), (3, 200004, None, "N"),
        (5, 200001, "NS", "Y"), (5, 200004, "AD", "N"),
        (9, 200001, "AD", "N"),                # a security that is not in the grid
    ], columns=["permno", "yyyymm", "sharetype", "usincflg"])
    grid = pd.DataFrame({"permno": np.int32([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5]),
                         "month": pd.PeriodIndex(["2000-02", "2000-03", "2000-04", "2000-05", "2000-06", "2000-07", "2000-08",
                                                  "2000-02", "2000-04", "2000-06", "2000-03", "2000-04", "2000-05", "2000-04",
                                                  "2000-04"], freq="M")})
    got = ca.foreign_filers(grid.sample(frac=1, random_state=0), msf.sample(frac=1, random_state=1))    # any row order
    want = pd.Series([False, False,     # permno 1: the rows of January and February (US common)
                      True,             #   an ADR in March
                      True,             #   no row for April: the latest earlier row (March), never a later one
                      True, False,      #   a non-US issuer in May, US again in June
                      False,            #   the row of July does not exist yet: June's
                      True,             # permno 2, holding February: before its first monthly row -> its earliest row
                      True, True,       #   March's row; then May has no row: April's
                      False, False,     # permno 3: missing codes are not foreign
                      True,             #   usincflg N alone is enough
                      False,            # permno 4: not in the monthly file -> a US filer
                      False],           # permno 5, formation in March: January's row (US), not April's (ADR), though it is nearer
                     index=grid.index)
    assert got.dtype == bool and got.tolist() == want[grid.sample(frac=1, random_state=0).index].tolist()
    assert ca.foreign_filers(grid, msf).tolist() == want.tolist()


@pytest.mark.parametrize("min_lag, min_lag_foreign", [(4, 6), (6, 6), (3, 8), (4, 14)])
def test_safe_never_looks_ahead(min_lag, min_lag_foreign):
    """An ADR never gets accounts younger than 6 months, a US stock never younger than 4; both bounds are reached.
    (With 14 months even the previous fiscal year can be too young: then there is no fiscal year at all.)"""
    rng = np.random.default_rng(0)
    parts, links = [], []
    for k in range(48):                                                    # firms with every fiscal year-end month
        gv = f"{k:03d}"
        parts += [make_funda(y, int(rng.integers(1, 13)) if k % 5 == 0 else k % 12 + 1, gvkey=gv) for y in (1996, 1998, 2000)]
        links.append(make_links(pd.period_range("1997-01", "2003-12", freq="M"), permno=k, gvkey=gv, foreign=k >= 24))
    links = pd.concat(links, ignore_index=True)
    links.loc[links.permno % 6 == 5, "foreign_filer"] = rng.random((links.permno % 6 == 5).sum()) < 0.5   # flags that change
    a = ca.assign_fiscal_year(links, ca.fiscal_years(pd.concat(parts)), "safe", min_lag, min_lag_foreign)
    used, foreign = a[a.fyear.notna()], links.foreign_filer[a.fyear.notna()].to_numpy()
    assert foreign.sum() > 1000 and (~foreign).sum() > 1000
    start = used.month.dt.start_time
    assert (used.datadate[foreign] + pd.DateOffset(months=min_lag_foreign) <= start[foreign]).all()
    assert (used.datadate[~foreign] + pd.DateOffset(months=min_lag) <= start[~foreign]).all()
    age = (used.month.dt.year - used.datadate.dt.year) * 12 + used.month.dt.month - used.datadate.dt.month
    assert age[foreign].min() == min_lag_foreign + 1 and age[~foreign].min() == min_lag + 1    # month-ends: lag + 1 day at least
    assert (used.cap_year < used.month.dt.year).all()                      # a December market cap of an earlier year
    out = ca.characteristics(make_links(["2002-07", "2002-09"]), ca.fiscal_years(pd.concat([make_funda(1999, 4), make_funda(2001, 4)])),
                             pd.concat([make_caps([2000], me=1000.0), make_caps([2001])]), "safe")
    assert out.fyear.tolist() == [2000, 2001] and out.AT.tolist() == [1000.0, 800.0]
    assert out.A2ME.tolist() == [1000 / 1000.0, 800 / ME_DEC]              # fiscal 2000 over December 2000, then 2001 over 2001


# ---------------------------------------------------------------------------------------------------- link, caps, grid
def _lnk(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gvkey", "linkprim", "linktype", "lpermno", "linkdt", "linkenddt"]).assign(
        linkdt=lambda d: pd.to_datetime(d.linkdt), linkenddt=lambda d: pd.to_datetime(d.linkenddt))


def test_link_priority_and_dates():
    grid = pd.DataFrame({"permno": np.int32([1, 1, 1, 2, 3, 4, 5]),
                         "month": pd.PeriodIndex(["2000-02", "2000-07", "2001-02", "2000-02", "2000-02", "2000-02", "2000-02"], freq="M")})
    grid["formation_date"] = (grid.month - 1).dt.end_time.dt.normalize()
    lnk = _lnk([
        ("A", "J", "LC", 1, "1990-01-01", None),             # permno 1: secondary link to A, always
        ("B", "P", "LU", 1, "2000-01-31", "2000-06-30"),     #           primary link to B, valid on both of its end dates
        ("C", "C", "LU", 1, "2001-01-01", None),             #           then C (primary by CRSP) beats J
        ("D", "N", "LC", 2, "1990-01-01", None),             # permno 2: N against C -> C
        ("E", "C", "LS", 2, "1990-01-01", None),
        ("F", "P", "LX", 3, "1990-01-01", None),             # permno 3: only an LX link and one that starts too late
        ("G", "P", "LC", 3, "2000-02-01", None),
        ("H", "P", "LU", 4, "1990-01-01", None),             # permno 4: same priority, LC before LU
        ("I", "P", "LC", 4, "1990-01-01", None),
        ("J", "N", "LC", 5, "1990-01-01", None),             # permno 5: a secondary class gets its firm's GVKEY
        ("K", "P", "NU", np.nan, "1990-01-01", None),        # a link without PERMNO
    ])
    out = ca.link(grid, lnk)
    assert len(out) == len(grid) and out[["permno", "month"]].equals(grid[["permno", "month"]])
    assert out.gvkey.tolist()[:2] == ["B", "B"] and out.gvkey[2] == "C"
    assert out.gvkey[3] == "E" and pd.isna(out.gvkey[4]) and out.gvkey[5] == "I" and out.gvkey[6] == "J"
    assert out.linkprim.tolist()[:4] == ["P", "P", "C", "C"] and out.linktype[5] == "LC"


def test_share_classes_share_the_accounts_not_the_cap():
    links = pd.concat([make_links(["2002-07"], permno=1), make_links(["2002-07"], permno=2)])
    caps = pd.concat([make_caps([2001], permno=1, me=2000.0), make_caps([2001], permno=2, me=500.0)])
    out = ca.characteristics(links, ca.fiscal_years(make_funda()), caps, "pelger").set_index("permno")
    assert out.AT.tolist() == [1000.0, 1000.0] and out.Q[1] == out.Q[2]
    assert out.A2ME.tolist() == [0.5, 2.0] and out.BEME.tolist() == [375 / 2000, 375 / 500]


def test_december_caps_and_month_grid():
    days = pd.to_datetime(["2000-11-30", "2000-12-28", "2000-12-29", "2001-01-02", "2001-12-31", "2002-01-31", "2002-02-28"])
    daily = pd.DataFrame({"permno": np.int32(7), "dlycaldt": days, "dlycap": [1e6, 2e6, np.nan, 4e6, 5e6, 6e6, 7e6]})
    other = pd.DataFrame({"permno": np.int32(8), "dlycaldt": pd.to_datetime(["2000-12-29", "2001-01-02"]), "dlycap": [3e6, 3e6]})
    caps = ca.december_caps(pd.concat([daily, other])).set_index(["permno", "cap_year"]).me_dec
    assert caps.to_dict() == {(7, 2000): 2000.0, (7, 2001): 5000.0, (8, 2000): 3000.0}     # last cap of December, in millions
    g = ca.month_grid(pd.concat([daily, other]), first=pd.Period("2000-01", "M"), last=pd.Period("2002-02", "M"))
    assert g[g.permno == 7].month.astype(str).tolist() == [str(m) for m in pd.period_range("2000-12", "2002-02", freq="M")]
    assert g[g.permno == 8].month.astype(str).tolist() == ["2001-01", "2001-02"]
    form = g.set_index(["permno", "month"]).formation_date
    assert form[(7, pd.Period("2001-01", "M"))] == pd.Timestamp("2000-12-29")               # the market's last day of the month
    assert form[(8, pd.Period("2001-02", "M"))] == pd.Timestamp("2001-01-02")
    assert g.permno.dtype == "int32" and not g.duplicated(["permno", "month"]).any()


# ---------------------------------------------------------------------------------------------------- the real data
# Share of common stock-months equal to the CPZ panel (1e-4), measured on 2026-11-13 under "pelger"; see
# analysis\characteristics\accounting.md. What is left is restatements after the panel's vintage and ADR share counts.
MATCH = {"AT": 0.9993, "A2ME": 0.9946, "BEME": 0.9946, "C": 0.9995, "CF": 0.9981, "CF2P": 0.9930, "Q": 0.9992,
         "Lev": 0.9988, "E2P": 0.9944, "PROF": 0.9964, "CTO": 0.9989, "FC2Y": 0.9973, "OP": 0.9963, "PM": 0.9973,
         "RNA": 0.9960, "D2A": 0.9983, "Investment": 0.9991, "NOA": 0.9975, "DPI2A": 0.9982, "OA": 0.9973, "OL": 0.9965,
         "PCM": 0.9963}
MARGIN = 0.002
needs_data = pytest.mark.skipif(
    not all((ca.WRDS_DIR / p).exists() for p in ("raw/crsp_dsf_v2.parquet", "raw/comp_funda.parquet", "raw/ccm_lnkhist.parquet",
                                                 "pelger/characteristics.parquet", "universe/broad.parquet")),
    reason="needs the WRDS downloads and the CPZ panel under Data\\WRDS")


@pytest.fixture(scope="module")
def real():
    """links, December caps, and the fiscal-year table under the CPZ convention ("cpz": every zero-fill on) and
    under the production default."""
    links, funda, caps = ca.load_inputs()
    return links, {"cpz": ca.fiscal_years(funda, strict=False), "production": ca.fiscal_years(funda)}, caps


def _lag_respected(out: pd.DataFrame, foreign: np.ndarray) -> bool:
    """No row uses accounts younger than 4 months (6 for a foreign filer) on the first day of its holding month."""
    lag = (out.datadate + pd.DateOffset(months=ca.MIN_LAG_MONTHS)).mask(foreign, out.datadate + pd.DateOffset(months=ca.MIN_LAG_MONTHS_FOREIGN))
    return bool((lag <= out.month.dt.start_time)[out.datadate.notna()].all())


@needs_data
def test_panel_match(real):
    """The formulas are checked with every zero-fill on, as the panel was built; the production fills only remove rows."""
    from afsa.pelger_check import compare

    links, fy, caps = real
    match = lambda out: compare(out, list(ca.CHARS)).set_index("characteristic")           # noqa: E731
    cpz = ca.characteristics(links, fy["cpz"], caps, "pelger")
    c = match(cpz)
    assert (c.n > 240_000).all()
    for char in ca.CHARS:
        assert c.equal[char] >= MATCH[char] - MARGIN, (char, c.equal[char])
        assert c.spearman[char] > 0.995, (char, c.spearman[char])
    safe = match(ca.characteristics(links, fy["cpz"], caps, "safe"))
    assert (safe.equal > 0.975).all() and (safe.equal < c.equal).all()      # the gated months differ, by design
    ff = match(ca.characteristics(links, fy["cpz"], caps, "ff"))
    assert (ff.equal < 0.90).all()                                          # the written rule is not the panel's
    # production fills, same fiscal rule: a value is either the same or NaN, so no match is lost except through a lost row
    production = ca.characteristics(links, fy["production"], caps, "pelger")
    a, b = cpz[list(ca.CHARS)].to_numpy(), production[list(ca.CHARS)].to_numpy()
    assert not (np.isfinite(b) & (a != b)).any()
    emptied = {char for k, char in enumerate(ca.CHARS) if (np.isfinite(a[:, k]) & np.isnan(b[:, k])).any()}
    assert emptied == {"CF", "CF2P", "Q", "FC2Y", "OP", "RNA", "NOA", "OA", "OL"}
    p = match(production)
    hits = lambda x: (x.equal * x.n).round()                                # noqa: E731
    lost, lost_hits = c.n - p.n, hits(c) - hits(p)
    assert ((lost_hits >= 0) & (lost_hits <= lost)).all() and (lost <= 310).all()   # at most 301 panel rows (OA) became NaN
    assert lost[["FC2Y", "OP", "OL"]].eq(0).all()                           # the panel has no row with XSGA or COGS missing
    assert (p.equal >= c.equal - 1e-12).all()                               # and no match rate falls
    assert (match(ca.characteristics(links, fy["production"], caps, "safe")).equal > 0.975).all()


@needs_data
def test_link_and_universe_coverage(real):
    links, fy, caps = real
    assert not links.duplicated(["permno", "month"]).any()
    assert (links.formation_date.dt.to_period("M") + 1 == links.month).all()
    assert links.foreign_filer.dtype == bool and 0.10 < links.foreign_filer.mean() < 0.15
    universe = pd.read_parquet(ca.UNIVERSE_FILE, columns=["permno", "month", "sharetype", "usincflg"]).astype({"permno": "int32"})
    u = universe[["permno", "month"]]
    m = universe.merge(links, on=["permno", "month"], how="left", indicator=True)
    assert (m._merge == "both").all()                                       # every universe stock-month is in the grid
    assert m.gvkey.isna().mean() < 0.001
    assert (m.foreign_filer == ((m.sharetype == "AD") | (m.usincflg == "N"))).all()   # the universe file's own classification
    out = ca.characteristics(links, fy["production"], caps, "safe")
    missing = ca.universe_missing(out, u)
    assert missing["AT"] < 0.005 and missing["A2ME"] < 0.02
    assert missing.drop(["OA", "DPI2A", "D2A", "FC2Y", "OP", "OL", "CF", "CF2P"]).max() < 0.02
    assert missing[["OA", "FC2Y", "OP", "OL"]].between(0.20, 0.24).all() and missing[["CF", "CF2P", "D2A"]].between(0.05, 0.08).all()
    assert (missing[["CF", "CF2P"]] >= missing["D2A"]).all()                # no cash flow where depreciation is unknown
    cpz = ca.characteristics(links, fy["cpz"], caps, "safe")
    assert (ca.universe_missing(cpz, u) <= missing + 1e-12).all() and ca.universe_missing(cpz, u)[["FC2Y", "OP", "OL", "CF"]].max() < 0.005
    # the audit's finding: with every fill on FC2Y is exactly 0 in a sixth of the universe, one tie group of over 100 stocks
    before, after = ca.universe_ties(cpz, u), ca.universe_ties(out, u)
    assert before.zero["FC2Y"] > 0.15 and before.largest["FC2Y"] > 100
    assert after.zero["FC2Y"] < 0.001 and after.largest["FC2Y"] <= 3 and after.tied["FC2Y"] < 0.02
    assert after.largest.drop("Lev").max() <= 10                            # Lev: firms without debt, a true 0
    assert _lag_respected(out, links.foreign_filer.to_numpy()) and not _lag_respected(out, np.ones(len(out), dtype=bool))
    assert (out.cap_year[out.fyear.notna()] < out.month.dt.year[out.fyear.notna()]).all()
    lags = ca.filing_lags(out, links, u)
    assert lags.youngest_days["US"] > 115 and lags.youngest_days["foreign"] > 175 and (lags.gated > 300).all()


@needs_data
def test_output_file(real):
    if not ca.OUT_FILE.exists():
        pytest.skip("run `python -m afsa.chars_accounting build` first")
    out, links = pd.read_parquet(ca.OUT_FILE), pd.read_parquet(ca.LINK_FILE)
    assert out.permno.dtype == "int32" and str(out.month.dtype) == "period[M]"
    assert (out[list(ca.CHARS)].dtypes == "float64").all() and np.isfinite(out[list(ca.CHARS)].to_numpy(na_value=0.0)).all()
    assert not out.duplicated(["permno", "month"]).any() and len(out) == len(links)
    assert out.month.min() == ca.FIRST_MONTH and out.month.max() == ca.LAST_MONTH
    assert (out.gvkey.fillna("") == links.gvkey.fillna("")).all() and links.foreign_filer.dtype == bool
    assert _lag_respected(out, links.foreign_filer.to_numpy())             # the file is built with "safe"
    fresh_links, fy, caps = real                                            # ... and with the production fills
    pd.testing.assert_frame_equal(links, fresh_links)
    pd.testing.assert_frame_equal(out, ca.characteristics(fresh_links, fy["production"], caps, "safe"))


def test_only_the_default_build_writes_the_production_file(monkeypatch):
    assert ca.out_file() == ca.OUT_FILE and ca.out_file("safe", ca.PRODUCTION_STRICT[::-1]).name == "accounting.parquet"
    assert ca.out_file("safe", False).name == "accounting_allfills.parquet" and ca.out_file("safe", ()) == ca.out_file("safe", False)
    assert ca.out_file("safe", True).name == "accounting_strict.parquet" and ca.out_file("safe", ca.ZERO_FILL_ITEMS) == ca.out_file("safe", True)
    names = {ca.out_file(r, s).name for r in ca.FISCAL_RULES for s in (ca.PRODUCTION_STRICT, False, True, ("lct", "ceq"))}
    assert len(names) == 12 and ca.out_file("pelger", ("lct", "ceq")).name == "accounting_pelger_strict-ceq-lct.parquet"
    assert sum(n == "accounting.parquet" for n in names) == 1 and ca.out_file("pelger").name == "accounting_pelger.parquet"
    with pytest.raises(ValueError):
        ca.build(strict=("at",))                                            # not a zero-filled input; nothing is read or written
    # the command line: the default build is the production one
    calls = []
    monkeypatch.setattr(ca, "build", lambda rule, strict: calls.append((rule, strict)))
    for argv in (["build"], ["build", "--all-fills"], ["build", "--strict"], ["build", "--strict", "ceq", "lct"],
                 ["build", "--fiscal-rule", "pelger"]):
        monkeypatch.setattr("sys.argv", ["chars_accounting", *argv])
        ca.main()
    assert calls == [("safe", ca.PRODUCTION_STRICT), ("safe", False), ("safe", True), ("safe", ("ceq", "lct")),
                     ("pelger", ca.PRODUCTION_STRICT)]
    monkeypatch.setattr("sys.argv", ["chars_accounting", "build", "--strict", "--all-fills"])
    with pytest.raises(SystemExit):
        ca.main()


def test_universe_ties_counts_zeros_and_tie_groups():
    months = pd.PeriodIndex(["2000-01"] * 5 + ["2000-02"] * 5, freq="M")
    universe = pd.DataFrame({"permno": np.int32(list(range(5)) * 2), "month": months})
    out = universe.assign(**{c: np.arange(10.0) for c in ca.CHARS})        # all different: no tie
    out["FC2Y"] = [0, 0, 0, 1, np.nan, 0, 2, 2, np.nan, np.nan]            # January: three at 0; February: two at 2
    out["OL"] = [5.0] * 5 + [6.0, 6.0, 7.0, 8.0, 9.0]
    t = ca.universe_ties(out, universe)
    assert t.loc["FC2Y"].tolist() == [0.4, 0.5, 3] and t.loc["OL"].tolist() == [0.0, 0.7, 5]
    assert t.loc["AT"].tolist() == [0.1, 0.0, 1] and str(t.largest.dtype) == "int64"
    assert ca.universe_missing(out, universe)["FC2Y"] == 0.3
