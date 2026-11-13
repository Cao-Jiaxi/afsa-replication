"""The 13 monthly market characteristics (afsa.chars_market): each formula on small synthetic data with expectations
computed by hand or by a direct numpy calculation, no look-ahead, and (when the files are on disk) the match with
the Chen-Pelger-Zhu panel."""
import numpy as np
import pandas as pd
import pytest

from afsa import chars_market as cm

NAN = np.nan


# ---------------------------------------------------------------------------------------------------- synthetic data
def make_dsf(permno: int, dates, retx, div_yield=0.0, vol=1000.0, shrout=1000.0, p0=10.0, split=None,
             no_trade=(), missing=()) -> pd.DataFrame:
    """Rows of crsp.dsf_v2 for one security from its price returns. The first row has no return (no previous price).

    div_yield: dividend over the previous price per day; split: {date: factor} (a 2-for-1 split has factor 2);
    no_trade: dates priced at the bid-ask midpoint (no high/low); missing: dates without price, return or volume.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    retx = np.broadcast_to(np.asarray(retx, float), (n,)).copy()
    dy = np.broadcast_to(np.asarray(div_yield, float), (n,)).copy()
    fac = np.ones(n)
    for day, f in (split or {}).items():
        fac[dates.get_loc(pd.Timestamp(day))] = f
    prc = np.empty(n)
    prc[0] = p0
    for t in range(1, n):
        prc[t] = prc[t - 1] * (1.0 + retx[t]) / fac[t]
    prev = np.append(NAN, prc[:-1])
    retx[0] = NAN
    cum = np.append(np.cumprod(fac[::-1])[::-1][1:], 1.0)        # CRSP's cumulative factor: the splits still to come
    df = pd.DataFrame({
        "permno": np.int32(permno), "dlycaldt": dates, "dlyret": retx + np.where(np.isnan(retx), NAN, dy), "dlyretx": retx,
        "dlyprc": prc, "dlyprevprc": prev, "dlyfacprc": fac, "dlycumfacpr": cum, "dlyorddivamt": np.nan_to_num(dy * prev),
        "dlynonorddivamt": 0.0, "dlycap": prc * shrout, "shrout": shrout, "dlyvol": np.broadcast_to(np.asarray(vol, float), (n,)).copy(),
        "dlyhigh": prc * 1.01, "dlylow": prc * 0.99, "dlybid": prc * 0.999, "dlyask": prc * 1.001, "dlydelflg": "N"})
    quiet = dates.isin(pd.DatetimeIndex(no_trade))
    df.loc[quiet, ["dlyhigh", "dlylow"]] = NAN
    gone = dates.isin(pd.DatetimeIndex(missing))
    df.loc[gone, ["dlyret", "dlyretx", "dlyprc", "dlycap", "dlyvol", "dlyhigh", "dlylow", "dlybid", "dlyask"]] = NAN
    after = np.append(False, gone[:-1])                          # the day after a gap: CRSP's return spans it; here none
    df.loc[after, ["dlyret", "dlyretx", "dlyprevprc"]] = NAN
    return df


def make_factors(dates, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "mktrf": rng.normal(0.0003, 0.01, n), "smb": rng.normal(0, 0.005, n),
                         "hml": rng.normal(0, 0.005, n), "rf": np.full(n, 0.0002) + 0.0001 * (np.arange(n) // 250 % 2)})


def random_panel(last_day="2006-06-30", seed=1):
    """Three securities from January 2000 to June 2006 (78 months, so holding months to 2006-06): one complete, one
    listed on the 561st day (2002-02-25), one with gaps in its returns."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2000-01-03", last_day)
    fac = make_factors(days, seed)
    frames = []
    for k, (permno, start) in enumerate([(10001, 0), (10002, 560), (10003, 0)]):
        d = days[start:]
        retx = 0.8 * fac.mktrf.to_numpy()[start:] + rng.normal(0, 0.015, len(d))
        dy = np.where(rng.random(len(d)) < 0.01, 0.005, 0.0)
        frames.append(make_dsf(permno, d, retx, dy, vol=rng.integers(1_000, 900_000, len(d)).astype(float),
                               shrout=5_000.0 + 1_000 * k, p0=20.0 + 10 * k, split={d[len(d) // 2]: 2.0},
                               no_trade=d[rng.random(len(d)) < 0.02],
                               missing=d[5:-5][rng.random(len(d) - 10) < (0.03 if permno == 10003 else 0.0)]))
    return pd.concat(frames, ignore_index=True), fac


def value(df: pd.DataFrame, permno: int, month: str, column: str) -> float:
    row = df[(df.permno == permno) & (df.month == pd.Period(month, "M"))]
    assert len(row) == 1
    return float(row[column].iloc[0])


def window(dsf: pd.DataFrame, permno: int, first: str, last: str) -> pd.DataFrame:
    """The daily rows of a security from month `first` to month `last`."""
    month = dsf.dlycaldt.dt.to_period("M")
    return dsf[(dsf.permno == permno) & (month >= pd.Period(first, "M")) & (month <= pd.Period(last, "M"))]


@pytest.fixture(scope="module")
def panel():
    dsf, fac = random_panel()
    return dsf, fac, cm.characteristics(dsf, fac)


# ---------------------------------------------------------------------------------------------------- formulas
def test_monthly_return_does_not_reinvest_dividends():
    # January ends at 10. February: 10 -> 11 -> 11 (dividend 0.22, a yield of 2% on 11) -> 10.45.
    dates = pd.to_datetime(["2001-01-30", "2001-01-31", "2001-02-01", "2001-02-02", "2001-02-05", "2001-03-01"])
    dsf = make_dsf(1, dates, [0, 0, 0.10, 0.0, -0.05, 0.0], div_yield=[0, 0, 0, 0.02, 0, 0])
    g = cm.prepare(dsf, make_factors(dates))
    R = cm.monthly_returns(g)
    assert list(g.months.astype(str)) == ["2001-01", "2001-02", "2001-03"]
    assert R[0, 1] == pytest.approx(10.45 / 10 - 1 + 0.22 / 10, abs=1e-12)      # 0.067: price change plus dividend over the start price
    assert R[0, 1] != pytest.approx(1.10 * 1.02 * 0.95 - 1, abs=1e-5)           # 0.0659: what compounding daily returns gives
    assert R[0, 0] == 0.0 and R[0, 2] == 0.0
    # in the output the return of February belongs to holding month March
    out = cm.characteristics(dsf, make_factors(dates))
    assert list(out.month.astype(str)) == ["2001-02", "2001-03"]                # no holding month after the data
    assert value(out, 1, "2001-03", "ST_Rev") == pytest.approx(0.067, abs=1e-12)
    assert list(out.formation_date) == [pd.Timestamp("2001-01-31"), pd.Timestamp("2001-02-05")]


def test_monthly_return_is_rounded_like_crsp_and_missing_without_returns():
    dates = pd.to_datetime(["2001-01-31", "2001-02-01", "2001-02-02", "2001-03-01", "2001-03-02"])
    dsf = make_dsf(1, dates, [0, 1 / 3, 0, 0, 0], missing=dates[3:])
    R = cm.monthly_returns(cm.prepare(dsf, make_factors(dates)))
    assert R[0, 1] == 0.333333
    assert np.isnan(R[0, 0]) and np.isnan(R[0, 2])               # the first row has no return; March has no price at all


def test_past_return_windows():
    M = 40
    R = np.array([0.01 * (k + 1) * (-1) ** k for k in range(M)])[None, :]
    out = cm.past_returns(R)
    f = 37
    prod = lambda months: np.prod([1 + R[0, k] for k in months]) - 1
    assert out["ST_Rev"][0, f] == R[0, f]
    assert out["r2_1"][0, f] == R[0, f - 1]                                      # the month before last
    assert out["r12_2"][0, f] == pytest.approx(prod(range(f - 11, f)), abs=1e-14)       # 11 months, f-11 .. f-1
    assert out["r12_7"][0, f] == pytest.approx(prod(range(f - 11, f - 5)), abs=1e-14)   # 6 months, f-11 .. f-6
    assert out["r36_13"][0, f] == pytest.approx(prod(range(f - 35, f - 11)), abs=1e-14)  # 24 months, f-35 .. f-12
    # the first formation month with a complete window, and NaN before it
    for name, first in {"ST_Rev": 0, "r2_1": 1, "r12_2": 11, "r12_7": 11, "r36_13": 35}.items():
        assert np.isfinite(out[name][0, first:]).all() and np.isnan(out[name][0, :first]).all(), name
    # a month without a return empties every window that contains it, and only those
    R2 = R.copy()
    R2[0, 20] = NAN
    out2 = cm.past_returns(R2)
    assert list(np.flatnonzero(np.isnan(out2["r12_2"][0, 11:])) + 11) == list(range(21, 32))
    assert list(np.flatnonzero(np.isnan(out2["r12_7"][0, 11:])) + 11) == list(range(26, 32))
    assert list(np.flatnonzero(np.isnan(out2["r36_13"][0, 35:])) + 35) == list(range(35, 40))
    assert np.isnan(out2["ST_Rev"][0, 20]) and np.isnan(out2["r2_1"][0, 21]) and np.isfinite(out2["r2_1"][0, 22])


def test_size_and_turnover():
    dates = pd.to_datetime(["2001-01-30", "2001-01-31", "2001-02-01", "2001-02-02", "2001-02-05", "2001-03-01"])
    # 2,000 thousand shares at 10, then +10% and +10%: 12.10 on 2 February; no price on the 5th, the last day of February
    dsf = make_dsf(1, dates, [0, 0, 0.10, 0.10, 0, 0], shrout=2000.0, vol=[0, 0, 100_000, 23_456, 999, 0], missing=[dates[4]])
    out = cm.characteristics(dsf, make_factors(dates))
    assert value(out, 1, "2001-03", "LME") == pytest.approx(12.10 * 2000 / 1000, rel=1e-12)       # USD millions, last price of the month
    assert value(out, 1, "2001-03", "LTurnover") == pytest.approx(round(123_456 / 100) / 2000, rel=1e-12)   # 1,235 hundreds
    assert value(out, 1, "2001-02", "LME") == pytest.approx(20.0, rel=1e-12)
    assert value(out, 1, "2001-02", "LTurnover") == 0.0


def test_rel_to_high_adjusts_for_splits_and_needs_twelve_months_of_data():
    days = pd.bdate_range("2001-01-01", "2002-03-29")
    retx = np.zeros(len(days))
    up, down = days.get_loc(pd.Timestamp("2001-06-15")), days.get_loc(pd.Timestamp("2001-06-18"))
    retx[up], retx[down] = 0.25, -0.20                           # 24 -> 30 on 15 June 2001 -> back to 24
    dsf = make_dsf(1, days, retx, p0=24.0, split={"2001-09-03": 2.0})    # after the split the price is 12
    out = cm.characteristics(dsf, make_factors(days))
    assert dsf.dlyprc.iloc[-1] == pytest.approx(12.0)
    for month in ("2002-01", "2002-06"[:0] or "2002-02", "2002-03"):
        assert value(out, 1, month, "Rel2High") == pytest.approx(24 / 30, rel=1e-12)     # 12 after the split = 24 before it
    assert out[out.month <= pd.Period("2001-12", "M")].Rel2High.isna().all()            # windows that start before the data
    # once June 2001 has left the 12-month window (formation July 2002 on) the price is at its high
    days2 = pd.bdate_range("2001-01-01", "2002-09-30")
    retx2 = np.zeros(len(days2))
    retx2[up], retx2[down] = 0.25, -0.20
    out2 = cm.characteristics(make_dsf(1, days2, retx2, p0=24.0, split={"2001-09-03": 2.0}), make_factors(days2))
    assert value(out2, 1, "2002-06", "Rel2High") == pytest.approx(0.8) and value(out2, 1, "2002-08", "Rel2High") == pytest.approx(1.0)


def test_spread_is_the_daily_range_with_quotes_on_days_without_a_trade():
    dates = pd.to_datetime(["2001-01-31", "2001-02-01", "2001-02-02", "2001-03-01"])
    dsf = make_dsf(1, dates, 0.0)
    dsf.loc[1, ["dlyhigh", "dlylow"]] = [11.0, 9.0]                              # a trade: (11 - 9) / 10
    dsf.loc[2, ["dlyhigh", "dlylow", "dlybid", "dlyask"]] = [NAN, NAN, 9.9, 10.1]   # no trade: (10.1 - 9.9) / 10
    out = cm.characteristics(dsf, make_factors(dates))
    assert value(out, 1, "2001-03", "Spread") == pytest.approx((0.2 + 0.02) / 2, rel=1e-12)


def test_variance_resid_var_and_suv_against_direct_calculation(panel):
    dsf, fac, out = panel
    for permno, month in [(10001, "2003-05"), (10003, "2004-11"), (10002, "2002-07")]:
        m = pd.Period(month, "M")
        # Variance: daily returns of the two months before the holding month
        w = window(dsf, permno, str(m - 2), str(m - 1)).dropna(subset=["dlyret"])
        assert value(out, permno, month, "Variance") == pytest.approx(np.var(w.dlyret, ddof=1), rel=1e-10)
        # Resid_Var: SSE / (n - 1) of the three-factor regression on the same days
        j = w.merge(fac, left_on="dlycaldt", right_on="date")
        X = np.column_stack([np.ones(len(j)), j.mktrf, j.smb, j.hml])
        y = (j.dlyret - j.rf).to_numpy()
        res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        assert value(out, permno, month, "Resid_Var") == pytest.approx(res @ res / (len(y) - 1), rel=1e-8)
        # SUV: estimated on months m-3 and m-2, evaluated on m-1
        est = window(dsf, permno, str(m - 3), str(m - 2)).dropna(subset=["dlyret", "dlyvol"])
        ev = window(dsf, permno, str(m - 1), str(m - 1)).dropna(subset=["dlyret", "dlyvol"])
        design = lambda r: np.column_stack([np.ones(len(r)), np.maximum(r, 0), np.maximum(-r, 0)])
        coef = np.linalg.lstsq(design(est.dlyret.to_numpy()), est.dlyvol.to_numpy(), rcond=None)[0]
        resid = est.dlyvol.to_numpy() - design(est.dlyret.to_numpy()) @ coef
        expected = (ev.dlyvol.to_numpy() - design(ev.dlyret.to_numpy()) @ coef).mean() / np.sqrt(resid @ resid / (len(resid) - 1))
        assert value(out, permno, month, "SUV") == pytest.approx(expected, rel=1e-7)


def test_suv_with_a_regressor_that_is_zero_in_the_estimation_window():
    # no negative return in the estimation months: the regression is volume on a constant and max(R, 0)
    days = pd.bdate_range("2001-01-01", "2001-04-30")
    rng = np.random.default_rng(3)
    retx = np.abs(rng.normal(0, 0.01, len(days)))
    vol = 1e5 + 5e6 * retx + rng.normal(0, 1e4, len(days))
    dsf = make_dsf(1, days, retx, vol=vol)
    out = cm.characteristics(dsf, make_factors(days))
    est = window(dsf, 1, "2001-01", "2001-02").dropna(subset=["dlyret"])
    ev = window(dsf, 1, "2001-03", "2001-03")
    X = np.column_stack([np.ones(len(est)), est.dlyret])
    coef = np.linalg.lstsq(X, est.dlyvol.to_numpy(), rcond=None)[0]
    resid = est.dlyvol.to_numpy() - X @ coef
    expected = (ev.dlyvol.to_numpy() - coef[0] - coef[1] * ev.dlyret.to_numpy()).mean() / np.sqrt(resid @ resid / (len(resid) - 1))
    assert value(out, 1, "2001-04", "SUV") == pytest.approx(expected, rel=1e-6)


def direct_beta(dsf: pd.DataFrame, fac: pd.DataFrame, permno: int, month: str) -> float:
    m = pd.Period(month, "M")
    j = window(dsf, permno, str(m - 60), str(m - 1)).merge(fac, left_on="dlycaldt", right_on="date").dropna(subset=["dlyret"])
    rm = np.log1p(j.mktrf).to_numpy()
    # volatilities over the last 12 months: the security's log return (rf not subtracted) on its own days, and the
    # market's on every trading day
    last_year = (j.dlycaldt.dt.to_period("M") >= m - 12).to_numpy()
    fac_month = fac.date.dt.to_period("M")
    market = np.log1p(fac.mktrf[(fac_month >= m - 12) & (fac_month <= m - 1)])
    ratio = np.std(np.log1p(j.dlyret.to_numpy())[last_year], ddof=1) / np.std(market, ddof=1)
    # correlation of sums over 3 consecutive observations of ln(1 + R - rf), formed inside the 60-month window
    rc = np.log1p((j.dlyret - j.rf).to_numpy())
    a = rc[2:] + rc[1:-1] + rc[:-2]
    b = rm[2:] + rm[1:-1] + rm[:-2]
    assert last_year.sum() >= 120 and len(a) >= 750
    return np.corrcoef(a, b)[0, 1] * ratio


def test_beta_against_direct_calculation(panel):
    dsf, fac, out = panel
    for permno, month in [(10001, "2005-01"), (10001, "2006-03"), (10003, "2005-07"), (10002, "2006-03")]:
        assert value(out, permno, month, "Beta") == pytest.approx(direct_beta(dsf, fac, permno, month), rel=1e-9)
    # five years of data are needed: the first window that lies inside the data forms holding month 2005-01
    assert out[out.month < pd.Period("2005-01", "M")].Beta.isna().all()
    assert out[(out.month >= pd.Period("2005-01", "M")) & (out.permno != 10002)].Beta.notna().all()
    # the security listed in February 2002 has its 120 days within a year, but its 750 three-day sums only with its
    # 753rd row (752 returns), 2005-01-12: its first Beta is formed at the end of January 2005
    late = out[(out.permno == 10002)].set_index("month").Beta
    assert late[late.notna()].index.min() == pd.Period("2005-02", "M")


def test_beta_takes_rf_out_of_the_correlation_only():
    # a risk-free rate that varies over time moves the correlation series, ln(1 + R - rf); the volatilities ignore it
    dsf, fac = random_panel()
    one = dsf[dsf.permno == 10001]
    fac2 = fac.assign(rf=fac.rf + 0.002 * np.sin(np.arange(len(fac)) / 40.0))
    month = "2006-03"
    b1, b2 = (value(cm.characteristics(one, f, only=["Beta"]), 10001, month, "Beta") for f in (fac, fac2))
    d1, d2 = (direct_beta(one, f, 10001, month) for f in (fac, fac2))
    assert b1 == pytest.approx(d1, rel=1e-9) and b2 == pytest.approx(d2, rel=1e-9)
    assert abs(b2 / b1 - 1) > 1e-4                                # the change in rf did reach Beta


def test_beta_recovers_a_known_loading():
    days = pd.bdate_range("2000-01-03", periods=21 * 70)
    fac = make_factors(days, seed=5)
    fac["rf"] = 0.0
    dsf = make_dsf(1, days, np.expm1(1.5 * np.log1p(fac.mktrf.to_numpy())))      # log return = 1.5 x the market's
    out = cm.characteristics(dsf, fac)
    assert out.Beta.dropna().to_numpy() == pytest.approx(1.5, rel=1e-5)          # CRSP's 6 decimals aside, exactly 1.5


def test_minimum_observations(panel):
    dsf, fac, _ = panel
    one = dsf[dsf.permno == 10001].reset_index(drop=True)
    # only 10 returns in March and April 2003: a variance and an SUV, but no residual variance (20 days)
    month = one.dlycaldt.dt.to_period("M")
    thin = (month >= pd.Period("2003-03", "M")) & (month <= pd.Period("2003-04", "M"))
    keep = np.ones(len(one), bool)
    keep[np.flatnonzero(thin)[10:]] = False
    one.loc[~keep, ["dlyret", "dlyretx"]] = NAN
    out = cm.characteristics(one, fac)
    assert np.isfinite(value(out, 10001, "2003-05", "Variance")) and np.isnan(value(out, 10001, "2003-05", "Resid_Var"))
    assert np.isnan(value(out, 10001, "2003-05", "ST_Rev"))                      # April has no return left
    assert np.isfinite(value(out, 10001, "2003-08", "Resid_Var"))


def test_delisting_rows_are_not_trading_days():
    dates = pd.to_datetime(["2001-01-31", "2001-02-01", "2001-02-02", "2001-03-01", "2001-04-02"])
    dsf = make_dsf(1, dates[:3], [0, 0.01, 0.02])
    gone = dsf.iloc[[-1]].assign(dlycaldt=dates[3], dlyret=-0.5, dlyretx=-0.5, dlyprc=0.0, dlycap=NAN, dlydelflg="Y")
    other = make_dsf(2, dates, 0.0)
    out = cm.characteristics(pd.concat([dsf, gone, other], ignore_index=True), make_factors(dates))
    assert set(out[out.permno == 1].month.astype(str)) == {"2001-02", "2001-03"}     # no row formed in March, its delisting month
    assert value(out, 1, "2001-03", "ST_Rev") == pytest.approx(1.01 * 1.02 - 1, abs=1e-6)


# ---------------------------------------------------------------------------------------------------- no look-ahead
def test_no_look_ahead(panel):
    dsf, fac, base = panel
    rng = np.random.default_rng(7)
    numeric = [c for c in cm.DSF_USED if c not in ("permno", "dlycaldt", "dlydelflg")]
    # at the last two dates every characteristic of the complete security has a value: nothing is compared as NaN = NaN
    assert base[(base.permno == 10001) & (base.month >= pd.Period("2005-02", "M"))][cm.CHARS].notna().all().all()
    for month in ("2002-07", "2005-02", "2006-01"):
        hold = pd.Period(month, "M")
        formation = base.formation_date[base.month == hold].iloc[0]
        assert formation < hold.start_time and formation == dsf.dlycaldt[dsf.dlycaldt < hold.start_time].max()
        d2, f2 = dsf.copy(), fac.copy()
        later = (d2.dlycaldt > formation).to_numpy()
        d2.loc[later, numeric] = d2.loc[later, numeric].to_numpy() * rng.uniform(0.2, 3.0, (later.sum(), len(numeric)))
        d2 = d2[~(later & (rng.random(len(d2)) < 0.1))]                           # and some later rows disappear
        flater = (f2.date > formation).to_numpy()
        f2.loc[flater, cm.FACTORS] = rng.normal(0, 0.02, (flater.sum(), 4))
        changed = cm.characteristics(d2, f2)
        a = base[base.month <= hold].reset_index(drop=True)
        b = changed[changed.month <= hold].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)                    # bit for bit, NaN in the same places
        # the perturbation did bite: one month later, every characteristic that has a value has changed
        after = base[base.month == hold + 1].merge(changed, on=["permno", "month"], suffixes=("", "_c"))
        assert len(after) >= 2
        for c in cm.CHARS:
            if c in ("r2_1", "r12_2", "r12_7", "r36_13"):                        # these skip the newest month
                continue
            both = (after[c].notna() & after[f"{c}_c"].notna()).to_numpy()
            assert both.any() or (c == "Beta" and month == "2002-07"), c         # no five-year window yet in 2002
            assert not np.isclose(after[c][both], after[f"{c}_c"][both], rtol=1e-9).any(), c


def test_no_look_ahead_one_day_at_a_time(panel):
    # the sharpest form: change only the first day after a formation date (security rows and factors). Nothing up to
    # that holding month moves; the next month's values do.
    dsf, fac, base = panel
    hold = pd.Period("2005-06", "M")
    next_day = dsf.dlycaldt[dsf.dlycaldt >= hold.start_time].min()
    d2, f2 = dsf.copy(), fac.copy()
    rows = (d2.dlycaldt == next_day).to_numpy()
    d2.loc[rows, ["dlyret", "dlyretx"]] += 0.05
    d2.loc[rows, ["dlyprc", "dlycap", "dlyhigh", "dlylow", "dlyvol"]] *= 1.5
    f2.loc[f2.date == next_day, cm.FACTORS] += 0.01
    changed = cm.characteristics(d2, f2)
    pd.testing.assert_frame_equal(base[base.month <= hold].reset_index(drop=True),
                                  changed[changed.month <= hold].reset_index(drop=True), check_exact=True)
    a = base[(base.month == hold + 1) & (base.permno == 10001)].iloc[0]
    b = changed[(changed.month == hold + 1) & (changed.permno == 10001)].iloc[0]
    for c in ("ST_Rev", "LTurnover", "Spread", "Variance", "Resid_Var", "SUV", "Beta"):
        assert a[c] != b[c], c


def test_rel_to_high_does_not_depend_on_later_splits():
    # CRSP's cumulative factor on a past day changes when the security splits later; the ratio does not
    days = pd.bdate_range("2001-01-01", "2002-12-31")
    retx = np.random.default_rng(11).normal(0, 0.02, len(days))
    fac = make_factors(days)
    a = cm.characteristics(make_dsf(1, days, retx, p0=50.0), fac, only=["Rel2High"])
    b = cm.characteristics(make_dsf(1, days, retx, p0=50.0, split={"2002-11-15": 3.0}), fac, only=["Rel2High"])
    upto = (a.month <= pd.Period("2002-11", "M")).to_numpy()
    assert a.Rel2High[upto].notna().sum() >= 10
    np.testing.assert_allclose(a.Rel2High[upto], b.Rel2High[upto], rtol=1e-12)


def test_output_layout(panel):
    dsf, fac, out = panel
    assert list(out.columns) == ["permno", "month", "formation_date", *cm.CHARS]
    assert out.permno.dtype == "int32" and str(out.month.dtype) == "period[M]"
    assert all(out[c].dtype == "float64" for c in cm.CHARS)
    assert not out.duplicated(["permno", "month"]).any()
    assert (out.formation_date.dt.to_period("M") + 1 == out.month).all()
    last_day = dsf.groupby(dsf.dlycaldt.dt.to_period("M")).dlycaldt.max()
    assert (out.formation_date.to_numpy() == last_day.reindex(out.month - 1).to_numpy()).all()
    # a security has a row exactly where it has daily rows in the formation month
    rows = dsf.assign(month=dsf.dlycaldt.dt.to_period("M") + 1)[["permno", "month"]].drop_duplicates()
    rows = rows[rows.month <= dsf.dlycaldt.max().to_period("M")]
    assert len(out) == len(rows) == len(out.merge(rows, on=["permno", "month"]))


def test_rows_in_any_order_and_duplicates(panel):
    dsf, fac, out = panel
    shuffled = cm.characteristics(dsf.sample(frac=1.0, random_state=0), fac.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(out, shuffled, check_exact=True)
    with pytest.raises(ValueError, match="duplicate"):
        cm.characteristics(pd.concat([dsf, dsf.iloc[[100]]], ignore_index=True), fac)


def test_read_french(tmp_path):
    text = ("This file was created by CMPT_ME_BEME_RETS_DAILY using the 201611 CRSP database.\n\n"
            "          Mkt-RF     SMB     HML      RF\n19260701    0.10   -0.24   -0.28   0.009\n20161130   -0.25   -0.39    2.23   0.001\n\n"
            "Copyright 2016 Kenneth R. French\n")
    (tmp_path / "f.txt").write_text(text)
    (tmp_path / "f.csv").write_text(text.replace("    ", ",").replace("   ", ","))
    for name in ("f.txt", "f.csv"):
        f = cm.read_french(tmp_path / name)
        assert list(f.columns) == ["date", "mktrf", "smb", "hml", "rf"] and len(f) == 2
        assert f.date.iloc[1] == pd.Timestamp("2016-11-30")
        assert f.iloc[1, 1:].to_numpy(dtype=float) == pytest.approx([-0.0025, -0.0039, 0.0223, 0.00001])


def test_compare_counts_missing_values_and_has_a_purely_relative_rate(monkeypatch):
    """afsa.pelger_check.compare on a synthetic panel: `equal` forgives 1e-6 on a small value, `rel1e4` does not, and
    a value that is missing on our side where the panel has one is counted in `ours_nan` (it lowers no rate)."""
    from afsa import pelger_check

    months = pd.period_range("2000-01", periods=5, freq="M")
    panel = pd.DataFrame({"permno": 1, "date": months.to_timestamp(how="end").normalize(), "month": months,
                          "Variance": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4], "ST_REV": [0.1, 0.2, NAN, 0.4, 0.5]})
    monkeypatch.setattr(pelger_check, "load_pelger", lambda columns=None: panel[["permno", "date", *columns, "month"]].copy())
    ours = pd.DataFrame({"permno": 1, "month": list(months) + [pd.Period("2000-06", "M")], "half": ["a", "a", "a", "b", "b", "b"],
                         # exact; off by 5e-7 (0.25% of the value); off by a relative 5e-5; missing; 50% off; a month the panel lacks
                         "Variance": [1e-4, 2e-4 + 5e-7, 3e-4 * (1 + 5e-5), NAN, 7.5e-4, NAN],
                         "ST_Rev": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    c = pelger_check.compare(ours)
    assert list(c.columns) == ["characteristic", "n", "equal", "within1", "spearman", "med_rel", "rel1e4", "ours_nan"]
    c = c.set_index("characteristic")
    assert (c.n["Variance"], c.equal["Variance"], c.within1["Variance"], c.rel1e4["Variance"], c.ours_nan["Variance"]) == (4, 0.75, 0.75, 0.5, 1)
    assert (c.n["ST_Rev"], c.equal["ST_Rev"], c.rel1e4["ST_Rev"], c.ours_nan["ST_Rev"]) == (4, 1.0, 1.0, 0)   # the panel's own gap is not ours
    by = pelger_check.compare(ours, ["Variance"], by="half").set_index("half")
    assert list(by.columns) == ["characteristic", "n", "equal", "within1", "spearman", "med_rel", "rel1e4", "ours_nan"]
    assert list(by.n) == [3, 1] and list(by.ours_nan) == [0, 1] and by.rel1e4["a"] == pytest.approx(2 / 3) and by.rel1e4["b"] == 0.0
    none = pelger_check.compare(ours.assign(Variance=NAN), ["Variance"])        # nothing to compare: the rates are empty, the count is not
    assert none.n[0] == 0 and none.ours_nan[0] == 5 and none[["equal", "rel1e4"]].isna().all().all()


# ---------------------------------------------------------------------------------------------------- the real files
from afsa.wrds_data import PELGER_DIR, RAW_DIR  # noqa: E402

on_disk = pytest.mark.skipif(not (cm.OUT_FILE.exists() and (PELGER_DIR / "characteristics.parquet").exists()),
                             reason="run `python -m afsa.chars_market build` (needs Data\\WRDS\\raw and the panel)")
VINTAGE = cm.CHAR_DIR / "reference" / "F-F_Research_Data_Factors_daily_201611.txt"

# share of stock-months equal to the panel (afsa.pelger_check.compare), measured 2026-11-13, less a small margin
EQUAL = {"ST_Rev": 0.999, "r2_1": 0.999, "r12_2": 0.997, "r12_7": 0.998, "r36_13": 0.995, "LME": 0.993, "LTurnover": 0.993,
         "Rel2High": 0.999, "Spread": 0.999, "Variance": 0.999, "SUV": 0.985}


@on_disk
def test_file_layout_and_universe_coverage():
    df = pd.read_parquet(cm.OUT_FILE)
    assert list(df.columns) == ["permno", "month", "formation_date", *cm.CHARS]
    assert df.permno.dtype == "int32" and str(df.month.dtype) == "period[M]"
    assert not df.duplicated(["permno", "month"]).any()
    assert df.month.min() <= pd.Period("1986-01", "M") and df.month.max() == pd.Period("2025-12", "M")
    assert (df.formation_date.dt.to_period("M") + 1 == df.month).all()
    first = {c: df.month[df[c].notna()].min() for c in cm.CHARS}
    assert max(first.values()) == first["Beta"] <= pd.Period("1989-11", "M")      # everything is there when the universe starts
    from afsa.wrds_universe import UNIVERSE_DIR
    if (UNIVERSE_DIR / "broad.parquet").exists():
        for kind in ("broad", "common"):
            u = pd.read_parquet(UNIVERSE_DIR / f"{kind}.parquet", columns=["permno", "month", "formation_date", "me"])
            j = u.merge(df, on=["permno", "month"], how="left", suffixes=("", "_c"))
            assert j.LME.notna().all() and (j.formation_date == j.formation_date_c).all()
            assert np.allclose(j.LME * 1000, j.me, rtol=1e-12)                    # the cap that ranks the universe
            cover = j[cm.CHARS].notna().mean()
            assert cover.drop(["Beta", "r36_13", "r12_2", "r12_7"]).min() > 0.995 and cover.min() > 0.93


@on_disk
def test_match_with_the_panel():
    from afsa.pelger_check import compare

    df = pd.read_parquet(cm.OUT_FILE)
    c = compare(df).set_index("characteristic")
    assert c.n.min() > 225_000
    for name, floor in EQUAL.items():
        assert c.equal[name] >= floor, (name, c.equal[name])
    assert (c.spearman.drop(["LME", "LTurnover"]) > 0.9997).all()
    # with today's Fama-French factors the two regression-based ones match the panel in rank and within about 1%
    # (measured: within 1% for 0.7996 and 0.9970 of stock-months; with the factors of 2016 see the next test)
    assert c.within1["Resid_Var"] > 0.78 and c.within1["Beta"] > 0.995
    assert c.spearman["Resid_Var"] > 0.9997 and c.spearman["Beta"] > 0.99995


@on_disk
@pytest.mark.skipif(not (VINTAGE.exists() and (RAW_DIR / "crsp_dsf_v2.parquet").exists()), reason="no archived factor file")
def test_resid_var_and_beta_match_with_the_factors_of_the_panels_time():
    from afsa.pelger_check import compare

    df = cm.from_files(VINTAGE, only=["Resid_Var", "Beta"])
    c = compare(df).set_index("characteristic")
    r = cm.relative_match(df, ["Resid_Var", "Beta"]).set_index("characteristic")
    # measured 2026-11-13: equal 0.9995 and 0.9977; purely relative distance <= 1e-4 for 0.9968 and 0.9976
    assert c.equal["Resid_Var"] > 0.998 and r["rel<=1e-4"]["Resid_Var"] > 0.995
    assert c.equal["Beta"] > 0.996 and r["rel<=1e-4"]["Beta"] > 0.996 and c.within1["Beta"] > 0.999
