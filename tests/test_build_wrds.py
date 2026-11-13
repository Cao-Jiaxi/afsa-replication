"""afsa.build_wrds: the daily characteristics, the risk-free rate, the ranking with missing values and the assembly
of the arrays on a small synthetic market (no licensed data), then the arrays on disk (skipped until they are built).

The tests of the arrays on disk fail, by design, on arrays that are older than their sources: rebuild with
`python -m afsa.build_wrds`. They recompute three days and do not compare the arrays with the builder's code."""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from afsa import build_wrds as bw
from afsa.build_window import DAILY, MONTHLY, ranks_and_medians
from afsa.chars_accounting import CHARS as ACCOUNTING_CHARS
from afsa.chars_market import CHARS as MARKET_CHARS
from afsa.config import CHARACTERISTICS, N_CHAR, N_LEVEL, N_SLOTS, TRAIN_DATA_DIR, Config, data_folder

K = {c: k for k, c in enumerate(CHARACTERISTICS)}


# ---------------------------------------------------------------------------------------------------- daily characteristics
def test_daily_characteristics_by_hand():
    r = np.array([[0.01, 0.10], [0.02, -0.10], [-0.03, 0.20], [0.04, 0.00], [0.05, 0.05], [-0.06, 0.01], [0.07, 0.02], [0.08, 0.03]])
    v = np.array([[100.0, 1], [200, 2], [300, 3], [400, 4], [500, 5], [600, 6], [700, 7], [800, 8]])
    d = bw.daily_characteristics(r, v)
    assert set(d) == set(DAILY)
    assert np.isnan(d["Ret_D1"][0]).all() and np.array_equal(d["Ret_D1"][1:], r[:-1])           # yesterday's return
    for c in ("Ret_W1", "STD_W1", "Vol"):
        assert np.isnan(d[c][:5]).all() and not np.isnan(d[c][5:]).any()                        # five earlier days needed
    # day 5 uses days 0..4, day 7 uses days 2..6: written out for the first security
    assert d["Ret_W1"][5, 0] == pytest.approx(1.01 * 1.02 * 0.97 * 1.04 * 1.05 - 1)
    assert d["Ret_W1"][7, 0] == pytest.approx(0.97 * 1.04 * 1.05 * 0.94 * 1.07 - 1)
    mean = (0.01 + 0.02 - 0.03 + 0.04 + 0.05) / 5
    var = sum((x - mean) ** 2 for x in (0.01, 0.02, -0.03, 0.04, 0.05)) / 4                     # sample variance: divisor 4
    assert d["STD_W1"][5, 0] == pytest.approx(var ** 0.5)
    assert d["Vol"][5, 0] == 1500 and d["Vol"][6, 0] == 2000 and d["Vol"][7, 1] == 3 + 4 + 5 + 6 + 7


def test_daily_characteristics_need_all_five_days():
    rng = np.random.default_rng(1)
    r, v = rng.normal(0, 0.02, (14, 3)), rng.uniform(1e3, 1e6, (14, 3))
    r[6, 1] = np.nan                                         # a day without a return
    v[9, 2] = np.nan                                         # a day without a volume (returns are there)
    d = bw.daily_characteristics(r, v)
    assert np.isnan(d["Ret_D1"][7, 1]) and not np.isnan(d["Ret_D1"][8, 1])
    for c in ("Ret_W1", "STD_W1"):
        assert np.isnan(d[c][7:12, 1]).all() and not np.isnan(d[c][12:, 1]).any() and not np.isnan(d[c][5:, [0, 2]]).any()
    assert not np.isnan(d["Vol"][5:, 1]).any()               # the volume of that day exists
    assert np.isnan(d["Vol"][10:, 2]).all() and not np.isnan(d["Vol"][5:10, 2]).any()


def test_daily_characteristics_use_nothing_from_the_day_or_later():
    rng = np.random.default_rng(2)
    r, v = rng.normal(0, 0.02, (40, 4)), rng.uniform(1e3, 1e6, (40, 4))
    base = bw.daily_characteristics(r, v)
    for t in (5, 17, 39):
        r2, v2 = r.copy(), v.copy()
        r2[t:], v2[t:] = rng.normal(0, 0.5, r2[t:].shape), rng.uniform(1, 10, v2[t:].shape)     # day t and everything after
        new = bw.daily_characteristics(r2, v2)
        for c in DAILY:
            assert np.array_equal(new[c][:t + 1], base[c][:t + 1], equal_nan=True)              # rows up to t: bit-identical
            if t + 1 < len(r):
                assert (new[c][t + 1] != base[c][t + 1]).all()                                  # and day t does enter row t+1


def test_weekly_characteristics_tie_whatever_the_order_of_the_operations():
    """Two securities with the same five returns in another order are equal in exact arithmetic: one value, one rank."""
    a = [0.0072, 0.0261, 0.0189, -0.0141, -0.0253]
    b = [0.0072, 0.0189, 0.0261, -0.0141, -0.0253]                          # the same five returns, two of them swapped
    r, v = np.zeros((6, 4)), np.ones((6, 4))
    r[:5, 0], r[:5, 1], r[:5, 2], r[:5, 3] = a, b, [0.01, -0.02, 0.0, 0.03, 0.01], 0.004     # the last one: five equal returns
    raw = bw.daily_characteristics(r, v, decimals=None)
    for c in ("Ret_W1", "STD_W1"):                                          # as computed, the two differ in the last bit
        assert 0 < abs(raw[c][5, 0] - raw[c][5, 1]) < 1e-15
        rk = ranks_and_medians(raw[c][5:6])[0][0]
        assert rk[0] != rk[1]                                               # ... which ranks them apart
    d = bw.daily_characteristics(r, v)
    for c in ("Ret_W1", "STD_W1"):
        assert d[c][5, 0] == d[c][5, 1]                                     # rounded: the same value
        rk = ranks_and_medians(d[c][5:6])[0][0]
        assert rk[0] == rk[1] and len(set(rk)) == 3                         # one average rank; the others keep theirs
        assert np.abs(d[c][5] - raw[c][5]).max() <= 0.5e-12                 # nothing moves by more than half a unit of the 12th decimal
    assert ranks_and_medians(d["Ret_W1"][5:6])[0][0, 0] == pytest.approx((1.5 - 1) / 3 - 0.5)   # the two lowest of four
    assert d["Ret_W1"][5, 0] == pytest.approx(np.prod(1 + np.array(a)) - 1, abs=1e-12)
    assert d["STD_W1"][5, 3] == 0.0                                         # five equal returns: no volatility
    for c in ("Ret_D1", "Vol"):
        assert np.array_equal(d[c], raw[c], equal_nan=True)                 # not rounded


# ---------------------------------------------------------------------------------------------------- risk-free rate
def test_spread_rf_compounds_to_the_monthly_rate():
    dates = pd.bdate_range("2001-01-01", "2001-03-30").to_numpy().astype("datetime64[D]")
    rf_month = pd.Series([0.0054, 0.0, 0.0042], index=pd.period_range("2001-01", "2001-03", freq="M"))
    rf = bw.spread_rf(dates, rf_month)
    months = pd.DatetimeIndex(dates).month
    n_jan = int((months == 1).sum())
    assert n_jan == 23 and np.allclose(rf[months == 1], 1.0054 ** (1 / 23) - 1, rtol=0, atol=1e-18)
    assert (rf[months == 2] == 0).all()                                                         # a zero month stays zero
    for m, rate in zip((1, 2, 3), rf_month):
        assert np.prod(1 + rf[months == m]) - 1 == pytest.approx(rate, abs=1e-13)
    assert rf[months == 3][0] == pytest.approx(1.0042 ** (1 / 22) - 1)                          # March has 22 days: another daily rate
    with pytest.raises(ValueError):
        bw.spread_rf(dates, rf_month.iloc[:2])                                                  # a month without a rate


def test_spread_rf_counts_an_unscheduled_closure_as_a_scheduled_day():
    """September 2001 has 15 trading days; 19 were scheduled when it began (the exchanges were closed on 11-14)."""
    labor_day = pd.DatetimeIndex(["2001-09-03"])                            # a scheduled holiday
    closed = pd.DatetimeIndex(["2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14"])
    assert np.isin(closed.to_numpy().astype("datetime64[D]"), bw.UNSCHEDULED_CLOSURES).all()
    scheduled = pd.bdate_range("2001-08-01", "2001-10-31").difference(labor_day)
    cal = scheduled.difference(closed)
    dates = cal.to_numpy().astype("datetime64[D]")
    rf_month = pd.Series([0.0031, 0.0028, 0.0022], index=pd.period_range("2001-08", "2001-10", freq="M"))
    rf = bw.spread_rf(dates, rf_month)
    sep = np.asarray(cal.month == 9)
    assert sep.sum() == 15 and (bw.scheduled_days(dates)[sep] == 19).all() and (bw.scheduled_days(dates)[~sep] == 23).all()
    assert np.allclose(rf[sep], 1.0028 ** (1 / 19) - 1, rtol=0, atol=1e-18)                     # before and after the closure
    assert 1e4 * (1.0028 ** (1 / 15) - 1 - rf[sep][0]) == pytest.approx(0.392, abs=1e-3)        # the realized days gave 0.39 bp more
    # the rate of a day does not depend on the closure: the same as on a calendar on which the exchanges stayed open
    rf_open = bw.spread_rf(scheduled.to_numpy().astype("datetime64[D]"), rf_month)
    assert np.array_equal(rf, rf_open[np.asarray(scheduled.isin(cal))])
    bm = bw.rf_by_month(dates, rf, rf_month)
    assert list(bm["days"]) == [23, 15, 23] and list(bm["scheduled"]) == [23, 19, 23]
    assert np.allclose(bm["compounded"], bm["target"], rtol=0, atol=1e-13)
    assert bm["compounded"].iloc[0] == pytest.approx(0.0031, abs=1e-13) and bm["compounded"].iloc[2] == pytest.approx(0.0022, abs=1e-13)
    assert bm["compounded"].iloc[1] == pytest.approx(1.0028 ** (15 / 19) - 1, abs=1e-13) and bm["compounded"].iloc[1] < 0.0028
    # the check of the build: passes on these rates, fails on rates spread over the realized days or on a calendar with a closure day
    ok, detail = bw.check_rf_months(dates, rf, rf_month, dates)
    assert ok and "2 months without one" in detail and "2001-09: 15 of 19 days, -5.90 bp" in detail
    realized = np.where(sep, 1.0028 ** (1 / 15) - 1, rf)
    assert not bw.check_rf_months(dates, realized, rf_month, dates)[0]
    assert not bw.check_rf_months(dates, rf, rf_month, scheduled.to_numpy().astype("datetime64[D]"))[0]
    assert not bw.check_rf_months(dates, -rf, rf_month, dates)[0]
    # every listed closure adds a day to its month; a closure known before the month began is not listed
    for month, extra in (("1985-09", 1), ("1994-04", 1), ("2001-09", 4), ("2004-06", 1), ("2012-10", 2), ("2007-01", 0), ("2018-12", 0), ("2025-01", 0)):
        p = pd.Period(month, "M")
        days = pd.bdate_range(p.start_time, p.end_time).to_numpy().astype("datetime64[D]")
        days = days[~np.isin(days, bw.UNSCHEDULED_CLOSURES)]
        assert (bw.scheduled_days(days) == len(days) + extra).all(), month
    assert len(bw.UNSCHEDULED_CLOSURES) == 9


# ---------------------------------------------------------------------------------------------------- ranking
def test_ranks_with_missing_values():
    v = np.array([[3.0, np.nan, 1.0, 2.0, np.nan],        # two without a value: the other three are ranked among themselves
                  [5.0, 5.0, 1.0, np.nan, 9.0],           # a tie takes the average rank
                  [np.nan, np.nan, np.nan, np.nan, np.nan]])
    r, med = ranks_and_medians(v)
    assert np.allclose(r[0], [0.5, 0.0, -0.5, 0.0, 0.0]) and med[0] == 2.0
    assert np.allclose(r[1], [(2.5 - 1) / 3 - 0.5, (2.5 - 1) / 3 - 0.5, -0.5, 0.0, 0.5]) and med[1] == 5.0
    assert (r[2] == 0).all() and med[2] == 0.0
    assert abs(r[0].sum()) < 1e-12 and abs(r[1].sum()) < 1e-12                                  # centred: missing ones sit at the middle


# ---------------------------------------------------------------------------------------------------- a synthetic market
PERMNOS = np.array([10001, 10002, 10005, 10010, 10020, 10033, 10047, 10050])
CAL = pd.bdate_range("1989-10-02", "1990-03-30")
MONTHS = pd.period_range("1989-10", "1990-03", freq="M")


def make_world(seed: int = 0) -> dict:
    """Eight securities, six months of business days, lists of six per month, characteristics with holes."""
    rng = np.random.default_rng(seed)
    dsf = pd.DataFrame([(p, d) for p in PERMNOS for d in CAL], columns=["permno", "dlycaldt"])
    dsf["dlyret"] = rng.normal(0, 0.02, len(dsf))
    dsf["dlyvol"] = rng.integers(1_000, 1_000_000, len(dsf)).astype(float)
    dsf["dlydelflg"] = "N"
    dsf = dsf[~((dsf.permno == 10002) & (dsf.dlycaldt == "1990-01-10"))]        # a day without a row
    dsf = dsf[~((dsf.permno == 10047) & (dsf.dlycaldt < "1989-12-20"))]         # listed late
    dsf = dsf[~((dsf.permno == 10050) & (dsf.dlycaldt > "1990-02-14"))].copy()  # delisted: its last row is the delisting row
    last = (dsf.permno == 10050) & (dsf.dlycaldt == "1990-02-14")
    dsf.loc[last, "dlydelflg"], dsf.loc[last, "dlyvol"] = "Y", np.nan
    dsf.loc[(dsf.permno == 10020) & (dsf.dlycaldt == "1990-01-15"), "dlyret"] = np.nan    # a row without a return
    lists = {m: sorted(set(PERMNOS) - {PERMNOS[k % 8], PERMNOS[(k + 3) % 8]}) for k, m in enumerate(MONTHS[1:])}
    uni = pd.DataFrame([(m, CAL[CAL < m.start_time][-1], p) for m, ps in lists.items() for p in ps], columns=["month", "formation_date", "permno"])
    grid = pd.DataFrame([(p, m) for p in PERMNOS for m in MONTHS], columns=["permno", "month"])
    grid["permno"] = grid["permno"].astype("int32")
    frames = {}
    for name, chars in (("market", MARKET_CHARS), ("accounting", ACCOUNTING_CHARS)):
        df = grid.copy()
        for c in chars:
            x = rng.normal(size=len(df)).round(1)                               # one decimal: ties occur
            x[rng.random(len(df)) < 0.15] = np.nan
            df[c] = x
        frames[name] = df
    frames["market"].loc[frames["market"].permno == 10010, "Beta"] = np.nan     # never has a Beta
    acc = frames["accounting"]
    acc.loc[(acc.permno == 10001) & (acc.month == MONTHS[3]), "OA"] = 9.9       # the largest OA in 1990-01, none in 1990-02
    acc.loc[(acc.permno == 10001) & (acc.month == MONTHS[4]), "OA"] = np.nan
    rf_month = pd.DataFrame({"date": MONTHS.to_timestamp(), "rf": [0.0068, 0.0069, 0.0061, 0.0057, 0.0057, 0.0064]})
    return {"dsf": dsf.reset_index(drop=True), "uni": uni, "market": frames["market"], "accounting": frames["accounting"],
            "rf_month": rf_month, "lists": lists}


def assemble_world(world: dict, tmp_path, monkeypatch, fill: str) -> tuple[dict, dict]:
    """Writes the synthetic files, points afsa.build_wrds at them, and returns (sources, assembled arrays)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    world["uni"].to_parquet(tmp_path / "broad.parquet", index=False)
    world["market"].to_parquet(tmp_path / "market.parquet", index=False)
    world["accounting"].to_parquet(tmp_path / "accounting.parquet", index=False)
    raw = {"crsp_index_daily": pd.DataFrame({"indno": bw.MARKET_INDEX, "dlycaldt": CAL}), "crsp_dsf_v2": world["dsf"],
           "ff_factors_monthly": world["rf_month"]}
    monkeypatch.setattr("afsa.wrds_data.load", lambda name, columns=None: raw[name][columns] if columns else raw[name])
    monkeypatch.setattr(bw, "UNIVERSE_DIR", tmp_path)
    monkeypatch.setattr(bw, "MARKET_FILE", tmp_path / "market.parquet")
    monkeypatch.setattr(bw, "ACCOUNTING_FILE", tmp_path / "accounting.parquet")
    src = bw.load_sources(fill)
    return src, bw.assemble(src, "wrds_broad")


def reference(world: dict, fill: str) -> dict:
    """The same arrays written out with loops and dictionaries, one stock-day at a time."""
    dsf = world["dsf"]
    row = {(int(p), pd.Timestamp(d)): (r, v) for p, d, r, v in zip(dsf.permno, dsf.dlycaldt, dsf.dlyret, dsf.dlyvol)}
    monthly = {}
    for name in ("market", "accounting"):
        df = world[name]
        for c in df.columns[2:]:
            monthly.update({(c, int(p), m): x for p, m, x in zip(df.permno, df.month, df[c])})
    rf_m = dict(zip(pd.PeriodIndex(world["rf_month"]["date"], freq="M"), world["rf_month"]["rf"]))
    n_m = pd.Series(1, index=CAL.to_period("M")).groupby(level=0).sum()
    rf_day = {d: (1 + rf_m[d.to_period("M")]) ** (1 / n_m[d.to_period("M")]) - 1 for d in CAL}

    def daily_value(c, p, i):                               # for target day CAL[i], from the five days before
        if i < 1:
            return np.nan
        if c == "Ret_D1":
            return row.get((p, CAL[i - 1]), (np.nan, np.nan))[0]
        if i < 5:
            return np.nan
        w = [row.get((p, CAL[j]), (np.nan, np.nan)) for j in range(i - 5, i)]
        r, v = np.array([x[0] for x in w]), np.array([x[1] for x in w])
        if c == "Vol":
            return v.sum()
        if np.isnan(r).any():
            return np.nan
        return round(float(np.prod(1 + r) - 1 if c == "Ret_W1" else np.std(r, ddof=1)), 12)

    def value(c, p, i):
        if c in DAILY:
            x = daily_value(c, p, i)
            while fill == "last" and np.isnan(x) and i > 0:
                i -= 1
                x = daily_value(c, p, i)
            return x
        m = CAL[i].to_period("M")
        x = monthly.get((c, p, m), np.nan)
        while fill == "last" and np.isnan(x) and m > MONTHS[0]:
            m -= 1
            x = monthly.get((c, p, m), np.nan)
        return x

    out = {}
    for i, d in enumerate(CAL):
        if d < pd.Timestamp("1989-11-01"):
            continue
        mem = [p for p in world["lists"][d.to_period("M")] if (p, d) in row]
        x = np.array([[value(c, p, i) for c in CHARACTERISTICS] for p in mem])
        ranks, med = np.zeros(x.shape), np.zeros(N_CHAR)
        for k in range(N_CHAR):
            ok = ~np.isnan(x[:, k])
            if ok.sum():
                ranks[ok, k] = (rankdata(x[ok, k]) - 1) / max(ok.sum() - 1, 1) - 0.5
                med[k] = np.median(x[ok, k])
        ret = np.nan_to_num([row[(p, d)][0] for p in mem], nan=0.0)
        out[d] = {"permno": np.array(mem), "ranks": ranks, "median": med, "ret": ret, "rf": rf_day[d], "rf_prev": rf_day[CAL[i - 1]], "x": x}
    return out


@pytest.mark.parametrize("fill", ["none", "last"])
def test_assembled_arrays_equal_a_loop_reference(tmp_path, monkeypatch, fill):
    world = make_world()
    src, a = assemble_world(world, tmp_path, monkeypatch, fill)
    arr, ref = a["arrays"], reference(world, fill)
    dates = pd.DatetimeIndex(arr["dates"])
    assert list(dates) == [d for d in CAL if d >= pd.Timestamp("1989-11-01")] and len(ref) == len(dates)
    assert arr["ranks"].shape == (len(dates), N_SLOTS, N_CHAR) and arr["levels"].shape == (len(dates), N_LEVEL)
    n_missing = 0
    for t, d in enumerate(dates):
        r = ref[d]
        n = len(r["permno"])
        assert np.array_equal(arr["permno"][t, :n], r["permno"]) and (arr["permno"][t, n:] == -1).all()
        assert np.allclose(arr["ranks"][t, :n], r["ranks"], atol=1e-6) and not arr["ranks"][t, n:].any()
        assert np.allclose(arr["levels"][t, :N_CHAR], r["median"], rtol=1e-6, atol=1e-12)
        assert arr["levels"][t, N_CHAR] == np.float32(r["rf_prev"]) and arr["rf"][t] == pytest.approx(r["rf"], abs=1e-15)
        assert np.array_equal(arr["ret"][t, :n], r["ret"].astype(np.float32)) and not arr["ret"][t, n:].any()
        assert np.array_equal(a["n_value"][t], (~np.isnan(r["x"])).sum(0))
        n_missing += int(np.isnan(r["x"]).sum())
    assert n_missing > (300 if fill == "none" else 30)                      # the reference does hold missing values


def test_membership_and_missing_values_in_the_synthetic_market(tmp_path, monkeypatch):
    world = make_world()
    out = {fill: assemble_world(world, tmp_path / fill, monkeypatch, fill)[1]["arrays"] for fill in ("none", "last")}
    arr = out["none"]
    dates = pd.DatetimeIndex(arr["dates"])
    t = {d: i for i, d in enumerate(dates)}

    def slot(day, permno):
        s = np.flatnonzero(arr["permno"][t[pd.Timestamp(day)]] == permno)
        return int(s[0]) if len(s) else None

    # 10002 is on the January list but has no row on 1990-01-10: out that day, back the next, with a new spell
    assert 10002 in world["lists"][pd.Period("1990-01", "M")]
    assert slot("1990-01-09", 10002) is not None and slot("1990-01-10", 10002) is None and slot("1990-01-11", 10002) is not None
    assert arr["run"][t[pd.Timestamp("1990-01-11")], slot("1990-01-11", 10002)] == 1
    # after the gap its weekly characteristics are missing for five days (rank 0 under "none"), then back
    for day, missing in (("1990-01-11", True), ("1990-01-17", True), ("1990-01-18", False)):
        row = arr["ranks"][t[pd.Timestamp(day)], slot(day, 10002)]
        assert (row[[K["Ret_W1"], K["STD_W1"], K["Vol"]]] == 0).all() == missing
    # under "last" the same days carry the value of 1990-01-10: a rank, not the middle
    row = out["last"]["ranks"][t[pd.Timestamp("1990-01-12")], slot("1990-01-12", 10002)]
    assert (row[[K["Ret_W1"], K["STD_W1"], K["Vol"]]] != 0).all()
    # a row without a return: the stock stays, with a return of 0
    assert arr["ret"][t[pd.Timestamp("1990-01-15")], slot("1990-01-15", 10020)] == 0.0
    # the delisting row (1990-02-14) is a day in the universe; afterwards the security is gone although it is on the list
    assert 10050 in world["lists"][pd.Period("1990-02", "M")]
    assert slot("1990-02-14", 10050) is not None and slot("1990-02-15", 10050) is None
    # a monthly characteristic belongs to the day's month: OA of 10001 is the largest in January and missing in February
    for d in CAL[(CAL >= "1990-01-01") & (CAL <= "1990-01-31")]:
        assert arr["ranks"][t[d], slot(d, 10001), K["OA"]] == 0.5
    for d in CAL[(CAL >= "1990-02-01") & (CAL <= "1990-02-28")]:
        assert arr["ranks"][t[d], slot(d, 10001), K["OA"]] == 0.0             # no value: the middle of the scale
        assert out["last"]["ranks"][t[d], slot(d, 10001), K["OA"]] == 0.5     # carried from January
    # never a Beta: rank 0 under both rules
    for a in out.values():
        s = np.argwhere(a["permno"] == 10010)
        assert len(s) > 50 and (a["ranks"][s[:, 0], s[:, 1], K["Beta"]] == 0).all()


def test_list_days_without_a_row_in_the_synthetic_market(tmp_path, monkeypatch):
    world = make_world()
    src, a = assemble_world(world, tmp_path, monkeypatch, "none")
    gone = bw.list_days_without_a_row(src, a)
    rows = set(zip(world["dsf"].permno, world["dsf"].dlycaldt))
    first, last = world["dsf"].groupby("permno").dlycaldt.min(), world["dsf"].groupby("permno").dlycaldt.max()
    want = {"after": 0, "before": 0, "inside": 0}
    for d in CAL[CAL >= "1989-11-01"]:
        for p in world["lists"][d.to_period("M")]:
            if (p, d) not in rows:
                want["after" if d > last[p] else "before" if d < first[p] else "inside"] += 1
    assert gone == {"total": sum(want.values()), **want}
    assert gone["inside"] == 1 and gone["after"] > 5 and gone["before"] > 5    # the gap of 10002, the delisting, the late listing
    assert gone["total"] == int(a["listed"].sum() - a["member"].sum())


def test_direct_recomputation_equals_the_assembled_arrays(tmp_path, monkeypatch):
    """bw.direct_inputs (pandas, from the parquet files) is the build's own check of sample days: the same members,
    returns, ranks and medians as the assembled arrays, on every fifth day of the synthetic market."""
    world = make_world()
    world["market"] = world["market"].assign(formation_date=pd.Timestamp("1989-09-29"))     # the real file has this column
    world["dsf"].to_parquet(tmp_path / "crsp_dsf_v2.parquet", index=False)
    monkeypatch.setattr(bw, "RAW_DIR", tmp_path)
    for fill in ("none", "last"):
        src, a = assemble_world(world, tmp_path, monkeypatch, fill)
        arr = a["arrays"]
        sample = np.arange(0, len(arr["dates"]), 5)
        direct = bw.direct_inputs("wrds_broad", fill, src["dates"], a["days"][sample])
        compared = 0
        for t in sample:
            d = direct[arr["dates"][t]]
            n, keep = len(d["permno"]), ~d["skipped"]
            assert np.array_equal(arr["permno"][t, :n], d["permno"]) and (arr["permno"][t, n:] == -1).all()
            assert np.array_equal(arr["ret"][t, :n], d["ret"].astype(np.float32))
            assert np.abs(arr["ranks"][t, :n][:, keep] - d["ranks"][:, keep]).max() < 2e-6
            assert np.allclose(arr["levels"][t, :N_CHAR][keep], d["median"][keep], rtol=1e-6, atol=1e-12)
            compared += int(keep.sum())
        assert compared > (36 if fill == "none" else 30) * len(sample)       # carried daily values are left out under "last"


def test_no_look_ahead_in_the_assembled_arrays(tmp_path, monkeypatch):
    """Everything dated on or after day D is replaced: X up to and including D and returns before D do not move."""
    world = make_world()
    base = assemble_world(world, tmp_path / "a", monkeypatch, "none")[1]["arrays"]
    rng = np.random.default_rng(5)
    for D in (pd.Timestamp("1989-12-12"), pd.Timestamp("1990-01-31"), pd.Timestamp("1990-02-01")):
        w = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in world.items()}
        late = w["dsf"]["dlycaldt"] >= D
        w["dsf"].loc[late, "dlyret"] = rng.normal(0, 0.3, int(late.sum()))
        w["dsf"].loc[late, "dlyvol"] = rng.uniform(1, 100, int(late.sum()))
        for name in ("market", "accounting"):
            later = w[name]["month"] > D.to_period("M")                         # characteristics formed after D
            w[name].loc[later, w[name].columns[2:]] = rng.normal(size=(int(later.sum()), len(w[name].columns) - 2))
        w["rf_month"].loc[w["rf_month"]["date"] > D, "rf"] = 0.05               # the rates of later months
        new = assemble_world(w, tmp_path / f"b{D:%m%d}", monkeypatch, "none")[1]["arrays"]
        t = int(np.searchsorted(base["dates"], np.datetime64(D, "D")))
        assert np.array_equal(new["permno"], base["permno"])
        assert np.array_equal(new["ranks"][:t + 1], base["ranks"][:t + 1]) and np.array_equal(new["levels"][:t + 1], base["levels"][:t + 1])
        assert np.array_equal(new["ret"][:t], base["ret"][:t]) and np.array_equal(new["rf"][:t + 1], base["rf"][:t + 1])
        assert not np.array_equal(new["ret"][t], base["ret"][t])
        assert not np.array_equal(new["ranks"][t + 1, :, K["Ret_D1"]], base["ranks"][t + 1, :, K["Ret_D1"]])


def test_config_knows_the_wrds_universes():
    cfg = Config(universe="wrds_broad", fill="none")
    assert data_folder(cfg.universe, cfg.fill) == "wrds_broad_fill-none" and data_folder("wrds_common", "last") == "wrds_common"
    assert Config().universe == "broad" and Config().fill == "last" and Config().name() == "attn_K008"      # defaults untouched
    assert set(MONTHLY) == set(MARKET_CHARS) | set(ACCOUNTING_CHARS)


def test_a_missing_folder_names_its_builder(tmp_path, monkeypatch):
    import afsa.panel

    monkeypatch.setattr(afsa.panel, "TRAIN_DATA_DIR", tmp_path)
    for folder, command in ((data_folder("wrds_broad", "last"), "`python -m afsa.build_wrds --universe wrds_broad --fill last`"),
                            (data_folder("wrds_common", "none"), "`python -m afsa.build_wrds --universe wrds_common`"),
                            (data_folder("broad", "last"), "`python -m afsa.build_data --universe broad`"),
                            (data_folder("broad", "none"), "`python -m afsa.build_data --universe broad --fill none`"),
                            ("window_train_fill-none/6y/1996", "`python -m afsa.build_window --rule window_train --fill none`")):
        with pytest.raises(FileNotFoundError, match=re.escape(command)):
            afsa.panel.Panel(folder, "cpu")


# ---------------------------------------------------------------------------------------------------- the arrays on disk
BUILT = [u for u in ("wrds_broad", "wrds_common") if (TRAIN_DATA_DIR / data_folder(u, "none") / "meta.json").exists()]
needs_arrays = pytest.mark.skipif(not BUILT, reason="run `python -m afsa.build_wrds` first")


@pytest.fixture(scope="module", params=BUILT or [None])
def folder(request):
    return TRAIN_DATA_DIR / data_folder(request.param, "none")


@needs_arrays
def test_written_meta(folder):
    meta = json.loads((folder / "meta.json").read_text())
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == []
    passed = " | ".join(c["check"] for c in meta["checks"] if c["ok"] is True)
    for topic in ("source files exist", "lookback days exist", "a list of 500 for every month", "unscheduled closure",
                  "risk-free rate of the previous trading day", "no missing value", "at least 100 members", "direct recomputation",
                  "no look-ahead", "constant within the month"):
        assert topic in passed, topic                                           # checks the arrays must have passed, whatever their number
    assert set(meta["unscheduled_closures"]) >= {"1994-04-27", "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11",
                                                 "2012-10-29", "2012-10-30"}
    assert (meta["n_days"], meta["n_slots"], meta["fill"]) == (9108, 500, "none")
    assert "12 decimals" in meta["daily_characteristics"]["rounding"]          # built with the tie-safe weekly values
    assert (meta["first_day"], meta["last_day"], meta["first_loss_day"]) == ("1989-11-01", "2025-12-31", "1990-01-02")
    if meta["universe"] == "wrds_broad":
        assert meta["n_stock_days_from_1990"] == 4_527_395                      # the stock-days of Ander's universe file
    assert list(meta["rank0_share"]) == list(CHARACTERISTICS)
    # the shares depend on the accounting definitions (training_data.md, section 6): only what must hold is asserted
    assert max(meta["rank0_share"].values()) < 0.5 and meta["rank0_share"]["Ret_D1"] < 1e-3 and meta["rank0_share"]["OA"] > 0.15


@needs_arrays
def test_market_row_reads_the_folder_with_the_fill_suffix(folder):
    """afsa.metrics scores a run against the market of the arrays it trained on. A wrds universe exists only with its
    fill suffix: scoring the first run on wrds_broad --fill none failed here, after training."""
    from afsa.metrics import market_returns
    universe = json.loads((folder / "meta.json").read_text())["universe"]
    market = market_returns(universe, 8, "none")
    assert len(market) == 9108 and np.isfinite(market.to_numpy()).all()


@needs_arrays
def test_arrays_on_disk_are_not_older_than_their_sources(folder):
    """meta.json records the size and the modification time of every source file at the build. A source that has
    changed since (a new market.parquet or accounting.parquet, a new download) means the arrays are stale."""
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["builder"] == "afsa.build_wrds"
    kind = meta["universe"].removeprefix("wrds_")
    assert set(meta["sources"]) == {"raw/crsp_index_daily.parquet", "raw/crsp_dsf_v2.parquet", "raw/ff_factors_monthly.parquet",
                                    "raw/ff_factors_daily.parquet", f"universe/{kind}.parquet", "characteristics/market.parquet",
                                    "characteristics/accounting.parquet"}
    stale = []
    for name, then in meta["sources"].items():
        p = bw.WRDS_DIR / name
        now = {"bytes": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} if p.exists() else None
        if now != then:
            stale.append(f"{name}: {then} at the build, {now} now")
        elif then["modified"] > meta["built"]:
            stale.append(f"{name}: modified {then['modified']}, after the build ({meta['built']})")
    assert not stale, f"{folder.name} is older than its sources, rerun `python -m afsa.build_wrds`: " + "; ".join(stale)


@needs_arrays
def test_written_arrays(folder):
    a = {f.stem: np.load(f, mmap_mode="r") for f in folder.glob("*.npy")}
    meta = json.loads((folder / "meta.json").read_text())
    T, S = len(a["dates"]), len(a["permnos"])
    assert a["ranks"].shape == (T, N_SLOTS, N_CHAR) and a["ranks"].dtype == np.float32
    assert a["levels"].shape == (T, N_LEVEL) and a["levels"].dtype == np.float32 and a["rf"].dtype == np.float64
    assert a["permno"].shape == a["ret"].shape == a["run"].shape == (T, N_SLOTS) and a["slot_of"].shape == (T, S)
    permno = np.asarray(a["permno"])
    member = permno >= 0
    assert member.sum() == meta["n_stock_days"] and 490 <= member.sum(1).min() and member.sum(1).max() <= N_SLOTS
    assert (np.diff(np.where(member, permno, np.iinfo(np.int32).max), axis=1) > 0)[member[:, 1:]].all()     # sorted, filled from slot 0
    assert np.isfinite(a["levels"]).all() and np.isfinite(a["ret"]).all() and (np.asarray(a["ret"])[~member] == 0).all()
    assert (np.asarray(a["run"])[0][member[0]] == 1).all() and (np.asarray(a["rf"]) >= 0).all()
    assert np.array_equal(a["levels"][1:, -1], np.asarray(a["rf"])[:-1].astype(np.float32))                 # X's rf: the previous day's
    for t in (0, 41, 3000, T - 1):
        r = np.asarray(a["ranks"][t])
        assert np.isfinite(r).all() and np.abs(r).max() <= 0.5 and not r[~member[t]].any()
        assert np.abs(r[member[t]].sum(0)).max() < 1e-3                                                     # centred
        share0 = (r[member[t]] == 0).mean(0)                                                                # missing values sit at 0
        assert share0[K["OA"]] > 0.05 and share0[K["Ret_D1"]] < 0.02
    exact = meta["rank_exactly_zero_share"]
    assert all(meta["rank0_share"][c] - 1e-9 <= exact[c] <= meta["rank0_share"][c] + 0.01 for c in CHARACTERISTICS)


SAMPLE_DAYS = ("1990-01-02", "2001-09-17", "2025-12-31")    # the first loss day (also the first day of a month), the first
                                                            # day after the closure of September 2001, the last day


@needs_arrays
def test_ranks_and_medians_on_disk_equal_a_recomputation_from_the_source_files(folder):
    """Three days recomputed here, stock by stock, from the daily file, the month's list, market.parquet and
    accounting.parquet (not with the builder's code): members, returns, the 39 ranks with the missing-value rule,
    the 39 medians and the rf input."""
    meta = json.loads((folder / "meta.json").read_text())
    a = {f.stem: np.load(f, mmap_mode="r") for f in folder.glob("*.npy")}
    dates = np.asarray(a["dates"])
    idx = pd.read_parquet(bw.RAW_DIR / "crsp_index_daily.parquet", columns=["indno", "dlycaldt"])
    cal = np.sort(idx.loc[idx["indno"] == bw.MARKET_INDEX, "dlycaldt"].to_numpy().astype("datetime64[D]"))
    lists = pd.read_parquet(bw.UNIVERSE_DIR / f"{meta['universe'].removeprefix('wrds_')}.parquet", columns=["month", "permno"])
    lists["month"] = pd.PeriodIndex(lists["month"], freq="M")
    monthly = []
    for path in (bw.MARKET_FILE, bw.ACCOUNTING_FILE):
        f = pd.read_parquet(path)
        f["month"] = pd.PeriodIndex(f["month"], freq="M")
        monthly.append(f[f["month"].isin([pd.Period(d[:7], "M") for d in SAMPLE_DAYS])])
    assert set(MONTHLY) <= set(monthly[0].columns) | set(monthly[1].columns)
    wrong = []                                                                  # (day, characteristic, what differs)
    for day in SAMPLE_DAYS:
        d = np.datetime64(day)
        i, t = int(np.searchsorted(cal, d)), int(np.searchsorted(dates, d))
        assert cal[i] == d and dates[t] == d
        week, month = cal[i - 5:i], pd.Period(day[:7], "M")                     # the five trading days before the day
        dsf = pd.read_parquet(bw.RAW_DIR / "crsp_dsf_v2.parquet", columns=["permno", "dlycaldt", "dlyret", "dlyvol"],
                              filters=[("dlycaldt", "in", [pd.Timestamp(x) for x in cal[i - 5:i + 1]])])
        dsf["day"] = dsf["dlycaldt"].to_numpy().astype("datetime64[D]")
        today = dsf[dsf["day"] == d].set_index("permno")
        mem = np.intersect1d(lists.loc[lists["month"] == month, "permno"].to_numpy(dtype=np.int64), today.index.to_numpy())   # sorted
        n = len(mem)
        assert n > 450 and np.array_equal(a["permno"][t, :n], mem) and (np.asarray(a["permno"][t, n:]) == -1).all()
        assert np.array_equal(a["ret"][t, :n], today["dlyret"].reindex(mem).fillna(0.0).to_numpy().astype(np.float32))
        before = dsf[dsf["day"] < d]
        r = before.pivot(index="day", columns="permno", values="dlyret").reindex(index=week, columns=mem).to_numpy()
        v = before.pivot(index="day", columns="permno", values="dlyvol").reindex(index=week, columns=mem).to_numpy()
        x = {"Ret_D1": r[-1], "Ret_W1": np.round(np.prod(1 + r, axis=0) - 1, 12), "STD_W1": np.round(np.std(r, axis=0, ddof=1), 12),
             "Vol": v.sum(axis=0)}                                              # NaN when one of the five days is missing
        for f in monthly:
            rows = f[f["month"] == month].set_index("permno").reindex(mem)
            x.update({c: rows[c].to_numpy(dtype=float) for c in f.columns if c in MONTHLY})
        got = np.asarray(a["ranks"][t])
        assert not got[n:].any()
        for k, c in enumerate(CHARACTERISTICS):
            has = ~np.isnan(x[c])
            assert has.sum() >= 100, (day, c)
            want = np.zeros(n)                                                  # no value: rank 0; the others among themselves
            want[has] = (rankdata(x[c][has]) - 1) / (has.sum() - 1) - 0.5
            if np.abs(got[:n, k] - want).max() >= 2e-6:
                wrong.append((day, c, f"ranks differ by up to {np.abs(got[:n, k] - want).max():.2g}"))
            if a["levels"][t, k] != pytest.approx(np.median(x[c][has]), rel=1e-6, abs=1e-30):
                wrong.append((day, c, f"median {a['levels'][t, k]:.6g} against {np.median(x[c][has]):.6g}"))
        assert a["levels"][t, N_CHAR] == np.float32(a["rf"][t - 1])            # the rate of the day before
    assert not wrong, f"{folder.name} does not hold what its source files give now, rerun `python -m afsa.build_wrds`: {wrong}"


@needs_arrays
def test_rf_on_disk_is_the_monthly_rate_over_the_scheduled_days(folder):
    """Every month recomputed from the Fama-French monthly file, with the unscheduled closures written out again here."""
    dates, rf = np.load(folder / "dates.npy"), np.load(folder / "rf.npy")
    ffm = pd.read_parquet(bw.RAW_DIR / "ff_factors_monthly.parquet", columns=["date", "rf"])
    rate = dict(zip(pd.PeriodIndex(ffm["date"], freq="M"), ffm["rf"].astype(float)))
    closures = {"1994-04": 1, "2001-09": 4, "2004-06": 1, "2012-10": 2}
    months = pd.DatetimeIndex(dates).to_period("M")
    for m in months.unique():
        of_month = np.asarray(months == m)
        days, extra = int(of_month.sum()), closures.get(str(m), 0)
        assert np.allclose(rf[of_month], (1 + rate[m]) ** (1 / (days + extra)) - 1, rtol=0, atol=1e-15), m
        total = np.prod(1 + rf[of_month]) - 1
        assert total == pytest.approx((1 + rate[m]) ** (days / (days + extra)) - 1, abs=1e-12), m
        assert (total < rate[m] - 1e-7) if extra else (abs(total - rate[m]) < 1e-12), m     # less only after a closure
    sep = np.asarray(months == pd.Period("2001-09", "M"))
    assert sep.sum() == 15 and len(set(rf[sep])) == 1                           # the same rate before and after the closure


@needs_arrays
def test_ret_d1_on_disk_is_yesterdays_return(folder):
    a = {f.stem: np.load(f, mmap_mode="r") for f in folder.glob("*.npy")}
    for t in (1, 500, 4321, 9000):
        both = np.flatnonzero((np.asarray(a["slot_of"][t]) >= 0) & (np.asarray(a["slot_of"][t - 1]) >= 0))    # in on t-1 and t
        x = np.asarray(a["ranks"][t])[np.asarray(a["slot_of"][t])[both], K["Ret_D1"]]
        y_prev = np.asarray(a["ret"][t - 1])[np.asarray(a["slot_of"][t - 1])[both]]
        y_same = np.asarray(a["ret"][t])[np.asarray(a["slot_of"][t])[both]]
        assert np.corrcoef(rankdata(x), rankdata(y_prev))[0, 1] > 0.999
        assert abs(np.corrcoef(rankdata(x), rankdata(y_same))[0, 1]) < 0.5


@needs_arrays
def test_panel_loads_and_loss_days_start_in_1990(folder):
    import torch

    from afsa.panel import Panel
    from afsa.train import Task, train_span

    meta = json.loads((folder / "meta.json").read_text())
    cfg = Config(universe=meta["universe"], fill="none")
    assert data_folder(cfg.universe, cfg.fill) == folder.name
    panel = Panel(folder.name, "cpu", first_year=1990, last_year=1997)
    start, stop = train_span(cfg, panel, Task(1998, 0))                        # the first window of the rolling experiment
    assert str(panel.dates[start]) == "1990-01-02" and str(panel.dates[stop - 1]) == "1997-12-31" and stop - start == 2024
    L = cfg.lookback
    b = panel.batch(torch.arange(start - L - 1, start + 10)[None, :], L)
    assert b.X.shape[1:3] == (L + 11, int(b.n_cols[0])) and torch.isfinite(b.X).all()
    assert b.valid[0, L].sum() > 450 and b.valid[0, L + 1].sum() > 450         # the day before the first loss day has its history too
