"""The 13 monthly market characteristics of AFSA's Table 1, from the daily CRSP file, checked against the CPZ panel.

    r2_1, r12_2, r12_7, r36_13, ST_Rev            past returns
    LME, LTurnover, Rel2High, Resid_Var, Spread, SUV, Variance, Beta

    .venv\\Scripts\\python.exe -m afsa.chars_market build    # -> Data\\WRDS\\characteristics\\market.parquet (15 seconds)
    .venv\\Scripts\\python.exe -m afsa.chars_market check    # match with the Chen-Pelger-Zhu panel, overall and by period
    .venv\\Scripts\\python.exe -m afsa.chars_market check --ff-vintage <file>   # Resid_Var and Beta recomputed with an
                                                             # archived copy of French's daily factor file (.txt or .csv)

Output: one row per (permno, month) with `month` the HOLDING month (Period[M]) and `formation_date` the last trading
day of month - 1. Every value uses only daily rows dated on or before the formation date; a value that cannot be
computed is NaN (no carry-forward, no imputation). A security has a row for every holding month whose formation
month holds at least one of its daily rows (delisting rows aside).

Definitions, with f = month - 1 the formation month. They are the ones that reproduce the CPZ panel
(Data\\WRDS\\pelger\\characteristics.parquet); where the panel contradicts CPZ's written definition, the panel wins
(marked *). The evidence is in analysis\\characteristics\\market.md.

    R(k)        return of calendar month k as legacy CRSP's monthly file has it: price change plus dividends that
                are not reinvested within the month, R = prod(1 + retx_d) - 1 + sum_d (ret_d - retx_d) prod_{s<d}(1 + retx_s),
                over the days of k with a return, rounded to 6 decimals like CRSP's field. retx_d and the dividend
                yield are recomputed from prices where that agrees with CRSP's own 6-decimal daily fields.
    ST_Rev      R(f)
    r2_1      * R(f-1): the month before last, not last month's return
    r12_2       prod_{k=f-11..f-1} (1 + R(k)) - 1     every month of the window must have a return
    r12_7       prod_{k=f-11..f-6} (1 + R(k)) - 1
    r36_13      prod_{k=f-35..f-12} (1 + R(k)) - 1
    LME         the security's own market cap on its last day with a price in f, USD millions (= CIZ `mthcap` / 1000)
    LTurnover   round(shares traded in f / 100) / shares outstanding (thousands) at the end of f: the panel's units,
                because legacy CRSP's monthly volume is in whole hundreds of shares
    Rel2High    split-adjusted close at the end of f over the highest split-adjusted close of months f-11..f
    Spread    * mean over the days of f of (high - low) / ((high + low) / 2), with ask and bid in place of high and
                low on days without a trade (legacy CRSP's ASKHI and BIDLO): a daily range, not a quoted spread
    Variance    sample variance (n - 1) of the daily returns of f-1 and f
    Resid_Var   SSE / (n - 1) of the regression of daily excess returns on a constant and the Fama-French factors
                Mkt-RF, SMB, HML over the days of f-1 and f (at least MIN_OBS_RESID_VAR days)
    SUV         regression of daily share volume on a constant, max(R, 0) and max(-R, 0) over f-2 and f-1; mean
                prediction error over the days of f, divided by sqrt(SSE / (n - 1)) of the estimation
    Beta      * rho * sigma_i / sigma_M (Frazzini and Pedersen), from daily log returns and the Fama-French market.
                sigma over months f-11..f: of ln(1 + R) for the security (not in excess of rf; at least 120 days)
                and of ln(1 + Mkt-RF) for the market on every trading day. rho over months f-59..f (at least 750
                sums): between sums over 3 consecutive observations of the security, of ln(1 + R - rf) and, on
                the same days, of ln(1 + Mkt-RF); the sums are formed inside the window. No shrinkage toward 1.

Short histories. A window that reaches back before the first month of the daily file is NaN (the file would cut
the window of a security listed long before): with data from 1984-01, Beta starts in holding month 1989-01. Inside
the data, a past return needs a return in every month of its window. Rel2High, Variance, Resid_Var, SUV and Beta
take the days a recently listed security has, subject to the minimum numbers of observations below (none for
Rel2High). In its listing month a security's R(k) runs from its first close, as CIZ `mthret` does (legacy CRSP has
no return there). The panel cannot settle these choices: its securities all have three years of returns.

Fama-French factors: the build uses WRDS's current file (ff.factors_daily). French revises past factor returns:
against his file of November 2016, Mkt-RF differs on 19% of the days of 1984-2016, SMB on 83% and HML on 89% (by
more than half a unit of the fourth decimal). So with today's file the two regression-based characteristics are not
equal to the panel: Beta is within 1% of it in 99.7% of stock-months, Resid_Var in 80%. The ranks move little for
Beta (at most 0.011 on the [-0.5, 0.5] scale) and more for Resid_Var: in the broad universe of 1990-2016 its rank
differs from the one the 2016 factors give by 0.0051 on average, by more than 0.01 in 15.5% of stock-months
(analysis\\characteristics\\market.md, section 5). With the file of the panel's time both formulas reproduce the panel:
`check --ff-vintage Data\\WRDS\\characteristics\\reference\\F-F_Research_Data_Factors_daily_201611.txt` (French's public
file as archived in January 2017). The factors, like CRSP itself, are today's vintage and not point-in-time.
Everything else written here derives from CRSP: it stays under Data\\ (git-ignored).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from afsa.config import WRDS_DIR
from afsa.wrds_data import load

CHAR_DIR = WRDS_DIR / "characteristics"
OUT_FILE = CHAR_DIR / "market.parquet"
PAST_RETURNS = ["r2_1", "r12_2", "r12_7", "r36_13", "ST_Rev"]
CHARS = PAST_RETURNS + ["LME", "LTurnover", "Rel2High", "Resid_Var", "Spread", "SUV", "Variance", "Beta"]
DSF_USED = ["permno", "dlycaldt", "dlyret", "dlyretx", "dlyprc", "dlyprevprc", "dlyfacprc", "dlycumfacpr", "dlyorddivamt",
            "dlynonorddivamt", "dlycap", "shrout", "dlyvol", "dlyhigh", "dlylow", "dlybid", "dlyask", "dlydelflg"]
FACTORS = ["mktrf", "smb", "hml", "rf"]

# Minimum numbers of observations. The panel cannot identify them (it holds only securities with years of history), so
# they follow the sources: French's residual-variance portfolios (20 days), CPZ's Beta (120 and 750); 2 for a variance;
# 5 for the volume regression (3 coefficients) is our choice.
MIN_OBS_VARIANCE = 2
MIN_OBS_RESID_VAR = 20
MIN_OBS_SUV = 5                 # days in the estimation months; at least 1 day in the evaluation month
MIN_OBS_BETA_VOL = 120
MIN_OBS_BETA_CORR = 750
RETURN_DECIMALS = 6             # CRSP stores returns with 6 decimals


# ---------------------------------------------------------------------------------------------------- the daily grid
@dataclass
class Daily:
    """Daily rows (sorted by security and day) on a security x month grid. Month k holds the days lo[k]:hi[k]."""
    permnos: np.ndarray                 # P securities
    days: np.ndarray                    # T trading days (datetime64), every day any security or the factor file has
    months: pd.PeriodIndex              # M consecutive calendar months
    lo: np.ndarray
    hi: np.ndarray
    p: np.ndarray                       # per row: security index, day index, month index
    d: np.ndarray
    m: np.ndarray
    cell: np.ndarray                    # per row: index of its security-month in the flattened P x M grid
    x: dict[str, np.ndarray] = field(default_factory=dict)      # per row: the columns of the daily file
    f: dict[str, np.ndarray] = field(default_factory=dict)      # per day: the factors (NaN where the file has no row)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.permnos), len(self.months)


def prepare(dsf: pd.DataFrame, factors: pd.DataFrame) -> Daily:
    """`dsf`: rows of crsp.dsf_v2 with the columns DSF_USED; `factors`: date, mktrf, smb, hml, rf (decimals).

    Delisting rows (dlydelflg = Y: the delisting return, dated after the last trading day, with no price) are
    dropped: they are not trading days, and a delisted security is in no later universe.
    """
    keep = (dsf["dlydelflg"] != "Y").to_numpy() if "dlydelflg" in dsf else np.ones(len(dsf), bool)
    permno = dsf["permno"].to_numpy()[keep]
    dates = dsf["dlycaldt"].to_numpy().astype("datetime64[D]")[keep]
    # The download is ordered by security and day: the columns are then taken as they are (a sorted copy of the 13
    # million rows would double the memory).
    same = permno[1:] == permno[:-1]
    order = None if ((permno[1:] > permno[:-1]) | (same & (dates[1:] > dates[:-1]))).all() else np.lexsort((dates, permno))
    if order is not None:
        permno, dates = permno[order], dates[order]
        if ((permno[1:] == permno[:-1]) & (dates[1:] == dates[:-1])).any():
            raise ValueError("duplicate (permno, dlycaldt) rows")
    factors = factors.sort_values("date")
    fdates = factors["date"].to_numpy().astype("datetime64[D]")
    inside = (fdates >= dates.min()) & (fdates <= dates.max())
    days = np.union1d(dates, fdates[inside])
    months = pd.period_range(pd.Timestamp(days[0]), pd.Timestamp(days[-1]), freq="M")
    starts = months.start_time.to_numpy().astype("datetime64[D]")
    lo = np.searchsorted(days, starts)
    hi = np.append(lo[1:], len(days))
    permnos = np.unique(permno)
    p, d = np.searchsorted(permnos, permno), np.searchsorted(days, dates)
    m = np.searchsorted(hi, d, side="right")
    g = Daily(permnos=permnos, days=days, months=months, lo=lo, hi=hi, p=p, d=d, m=m, cell=p * len(months) + m)
    for c in DSF_USED[2:-1]:
        v = dsf[c].to_numpy(dtype=float)[keep]
        g.x[c] = v if order is None else v[order]
    pos = np.searchsorted(fdates, days)
    found = (pos < len(fdates)) & (fdates[np.minimum(pos, len(fdates) - 1)] == days)
    for c in FACTORS:
        g.f[c] = np.where(found, factors[c].to_numpy(dtype=float)[np.minimum(pos, len(fdates) - 1)], np.nan)
    return g


def _msum(g: Daily, w: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Sum of `w` within each security-month: one value per daily row, or per row of the boolean mask `rows`."""
    P, M = g.shape
    return np.bincount(g.cell if rows is None else g.cell[rows], weights=w, minlength=P * M).reshape(P, M)


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    """Column m of the result is column m - k of `a` (NaN before the first month)."""
    out = np.full(a.shape, np.nan)
    if k < a.shape[1]:
        out[:, k:] = a[:, :a.shape[1] - k]
    return out


def _roll(a: np.ndarray, L: int) -> np.ndarray:
    """Sum over the L months ending at each month; NaN while the window reaches back before the first month."""
    c = np.concatenate([np.zeros(a.shape[:1] + (1,) + a.shape[2:]), np.cumsum(a, axis=1)], axis=1)
    out = np.full(a.shape, np.nan)
    if L <= a.shape[1]:                                   # a sample shorter than the window has no complete window
        out[:, L - 1:] = c[:, L:] - c[:, :a.shape[1] - L + 1]
    return out


def _roll_all(a: np.ndarray, L: int) -> np.ndarray:
    """Like _roll, and NaN unless every month of the window has a value."""
    ok = np.isfinite(a)
    return np.where(_roll(ok.astype(float), L) == L, _roll(np.where(ok, a, 0.0), L), np.nan)


def _month_last(g: Daily, v: np.ndarray) -> np.ndarray:
    """The last non-missing value of `v` within each security-month (as CIZ's monthly price and cap are taken)."""
    out = np.full(g.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(v))
    cell = g.cell[idx]
    last = np.flatnonzero(np.append(cell[1:] != cell[:-1], True))
    out.flat[cell[last]] = v[idx[last]]
    return out


def _month_max(g: Daily, v: np.ndarray) -> np.ndarray:
    out = np.full(g.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(v))
    if len(idx):
        cell = g.cell[idx]
        first = np.flatnonzero(np.append(True, cell[1:] != cell[:-1]))
        out.flat[cell[first]] = np.maximum.reduceat(v[idx], first)
    return out


def _ols(A: np.ndarray, b: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Solves the normal equations A beta = b where `ok`; a singular system (a regressor that is constant or zero in
    the window) gets the minimum-norm least-squares solution, i.e. the redundant regressor drops out."""
    beta = np.full(b.shape, np.nan)
    scale = np.sqrt(np.einsum("...ii->...i", A))
    with np.errstate(divide="ignore", invalid="ignore"):
        det = np.linalg.det(A / (scale[..., :, None] * scale[..., None, :]))
    regular = ok & (det > 1e-10)
    beta[regular] = np.linalg.solve(A[regular], b[regular][..., None])[..., 0]
    singular = ok & ~regular
    if singular.any():
        beta[singular] = np.einsum("...ij,...j->...i", np.linalg.pinv(A[singular], hermitian=True), b[singular])
    return beta


# ---------------------------------------------------------------------------------------------------- characteristics
def monthly_returns(g: Daily) -> np.ndarray:
    """R(k) for every security and calendar month (P x M): CRSP's legacy monthly return, built from daily data."""
    ret, retx, prev = g.x["dlyret"], g.x["dlyretx"], g.x["dlyprevprc"]
    ok = np.isfinite(ret) & np.isfinite(retx)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = g.x["dlyprc"] * g.x["dlyfacprc"] / prev - 1.0                       # CRSP rounds its daily returns to 6 decimals
        rx = np.where(np.abs(exact - retx) <= 6e-7, exact, retx)
        exact = (np.nan_to_num(g.x["dlyorddivamt"]) + np.nan_to_num(g.x["dlynonorddivamt"])) / prev
        income = np.where(np.abs(exact - (ret - retx)) <= 1.1e-6, exact, ret - retx)
    log_rx = np.where(ok, np.log1p(np.where(ok, rx, 0.0)), 0.0)
    income = np.where(ok, income, 0.0)
    # log price relative from the month's first row to the previous row (a sum restarted in every security-month, so
    # that no value depends, even in its last digit, on rows of other months)
    before = pd.Series(log_rx).groupby(g.cell, sort=False).cumsum().to_numpy() - log_rx
    R = np.expm1(_msum(g, log_rx)) + _msum(g, income * np.exp(before))
    return np.where(_msum(g, ok.astype(float)) > 0, np.round(R, RETURN_DECIMALS), np.nan)


def past_returns(R: np.ndarray) -> dict[str, np.ndarray]:
    """The five past-return characteristics by FORMATION month f, from the monthly returns R (P x M)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log1p(R)                                  # NaN wherever a month has no return: the window is then NaN
    cum = lambda first, last: np.expm1(_lag(_roll_all(L, last - first + 1), first))    # months f-last .. f-first
    return {"ST_Rev": R.copy(), "r2_1": _lag(R, 1), "r12_2": cum(1, 11), "r12_7": cum(6, 11), "r36_13": cum(12, 35)}


def size(g: Daily) -> np.ndarray:
    """LME by formation month: the last market cap of the month, USD millions."""
    return _month_last(g, g.x["dlycap"]) / 1000.0


def turnover(g: Daily) -> np.ndarray:
    """LTurnover by formation month, in the panel's units (hundreds of shares per thousand shares outstanding)."""
    vol = g.x["dlyvol"]
    ok = np.isfinite(vol)
    hundreds = np.round(_msum(g, vol[ok], ok) / 100.0)
    shares = _month_last(g, g.x["shrout"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((_msum(g, ok.astype(float)) > 0) & (shares > 0), hundreds / shares, np.nan)


def rel_to_high(g: Daily) -> np.ndarray:
    """Rel2High by formation month. Prices are adjusted with CRSP's cumulative factor; the ratio of two adjusted
    prices dated up to f does not depend on splits after f."""
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.where(g.x["dlyprc"] > 0, g.x["dlyprc"] / g.x["dlycumfacpr"], np.nan)
    high = _month_max(g, adj)
    stack = np.stack([_lag(high, k) for k in range(12)])
    with np.errstate(invalid="ignore"):
        top = np.nanmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    top[:, :11] = np.nan                                 # the window reaches back before the data
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(top), _month_last(g, adj) / top, np.nan)


def spread(g: Daily) -> np.ndarray:
    """Spread by formation month: the mean daily high-low range over its midpoint."""
    high = np.where(np.isfinite(g.x["dlyhigh"]), g.x["dlyhigh"], g.x["dlyask"])
    low = np.where(np.isfinite(g.x["dlylow"]), g.x["dlylow"], g.x["dlybid"])
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (high - low) / ((high + low) / 2.0)
    ok = np.isfinite(s)
    n = _msum(g, ok.astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, _msum(g, s[ok], ok) / n, np.nan)


def variance(g: Daily) -> np.ndarray:
    """Variance by formation month: sample variance of the daily returns of f-1 and f."""
    ok = np.isfinite(g.x["dlyret"])
    r = g.x["dlyret"][ok]
    n, s1, s2 = (_roll(_msum(g, w, ok), 2) for w in (np.ones(len(r)), r, r * r))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n >= MIN_OBS_VARIANCE, (s2 - s1 * s1 / n) / (n - 1.0), np.nan)


def resid_var(g: Daily) -> np.ndarray:
    """Resid_Var by formation month: SSE / (n - 1) of the Fama-French three-factor regression over f-1 and f."""
    F = np.column_stack([np.ones(len(g.days))] + [g.f[c] for c in ("mktrf", "smb", "hml")])[g.d]
    y = g.x["dlyret"] - g.f["rf"][g.d]
    ok = np.isfinite(y) & np.isfinite(F).all(axis=1)
    y, F, k = y[ok], F[ok], F.shape[1]
    A, b = np.empty(g.shape + (k, k)), np.empty(g.shape + (k,))
    for i in range(k):
        b[..., i] = _roll(_msum(g, y * F[:, i], ok), 2)
        for j in range(i, k):
            A[..., i, j] = A[..., j, i] = _roll(_msum(g, F[:, i] * F[:, j], ok), 2)
    n = A[..., 0, 0]
    enough = n >= max(MIN_OBS_RESID_VAR, k + 1)
    coef = _ols(A, b, enough)
    sse = _roll(_msum(g, y * y, ok), 2) - (coef * b).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(enough, np.maximum(sse, 0.0) / (n - 1.0), np.nan)


def suv(g: Daily) -> np.ndarray:
    """SUV by formation month: volume regression estimated on f-2 and f-1, mean standardized surprise in f."""
    r, vol = g.x["dlyret"], g.x["dlyvol"]
    ok = np.isfinite(r) & np.isfinite(vol)
    r, vol = r[ok], vol[ok]
    X = np.column_stack([np.ones(len(r)), np.maximum(r, 0.0), np.maximum(-r, 0.0)])
    A, b = np.empty(g.shape + (3, 3)), np.empty(g.shape + (3,))
    for i in range(3):
        b[..., i] = _msum(g, vol * X[:, i], ok)
        for j in range(i, 3):
            A[..., i, j] = A[..., j, i] = _msum(g, X[:, i] * X[:, j], ok)
    vv = _msum(g, vol * vol, ok)
    est = lambda a: _lag(_roll(a, 2), 1)                 # months f-2 and f-1
    A_est, b_est = est(A), est(b)
    n = A_est[..., 0, 0]
    enough = (n >= MIN_OBS_SUV) & (A[..., 0, 0] >= 1)
    coef = _ols(A_est, b_est, enough)
    sse = np.maximum(est(vv) - (coef * b_est).sum(axis=-1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        surprise = (b[..., 0] - (coef * A[..., 0, :]).sum(axis=-1)) / A[..., 0, 0]     # mean of (volume - prediction) over f
        out = surprise / np.sqrt(sse / (n - 1.0))
    return np.where(enough & np.isfinite(out), out, np.nan)


def beta(g: Daily) -> np.ndarray:
    """Beta by formation month: Frazzini-Pedersen's correlation times the ratio of volatilities.

    The daily series are the panel's (analysis/characteristics/market.md, section 3): the volatilities come from
    ln(1 + R) for the security (not in excess of rf) and from ln(1 + Mkt-RF) on every trading day for the market;
    the correlation from sums over 3 consecutive observations of the security, of ln(1 + R - rf) and, on the same
    days, of ln(1 + Mkt-RF)."""
    P, M = g.shape
    T = len(g.days)
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.log1p(g.x["dlyret"])                                               # for the volatility
        rc = np.log1p(g.x["dlyret"] - g.f["rf"][g.d])                              # for the correlation
        rm = np.log1p(g.f["mktrf"])[g.d]
    ok = np.isfinite(rv) & np.isfinite(rc) & np.isfinite(rm)
    rv, rc, rm, p, d, cell = rv[ok], rc[ok], rm[ok], g.p[ok], g.d[ok], g.cell[ok]

    def msum(w, rows=slice(None)):
        return np.bincount(cell[rows], weights=w, minlength=P * M).reshape(P, M)

    # volatilities over months f-11..f: the security's on the days it has a return, the market's on every trading day
    n, s1, s2 = (_roll(msum(w), 12) for w in (np.ones(len(rv)), rv, rv * rv))
    with np.errstate(divide="ignore", invalid="ignore"):
        mk = np.log1p(g.f["mktrf"])
    day = np.flatnonzero(np.isfinite(mk))
    month_of_day = np.searchsorted(g.hi, day, side="right")
    mn, m1, m2 = (_roll(np.bincount(month_of_day, weights=w, minlength=M)[None, :], 12) for w in (np.ones(len(day)), mk[day], mk[day] ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.sqrt((s2 - s1 * s1 / n) / (n - 1.0) / ((m2 - m1 * m1 / mn) / (mn - 1.0)))

    # sums over 3 consecutive observations of the security, attached to the last of the three
    full = np.zeros(len(rc), bool)
    full[2:] = p[2:] == p[:-2]
    a, b = rc.copy(), rm.copy()
    a[2:] += rc[1:-1] + rc[:-2]
    b[2:] += rm[1:-1] + rm[:-2]
    terms = np.column_stack([np.ones(len(a)), a, b, a * a, b * b, a * b])
    S = np.stack([_roll(msum(terms[full, i], full), 60) for i in range(6)], axis=-1)
    # The panel forms the 3-observation sums inside the 60-month window: the window's first two observations have none.
    start = np.full(M, -1)
    start[59:] = g.lo[:M - 59]                                                     # first day of month f-59
    key = p.astype(np.int64) * T + d
    at = np.searchsorted(key, np.arange(P, dtype=np.int64)[:, None] * T + np.maximum(start, 0)[None, :])
    for step in (0, 1):
        row = np.minimum(at + step, len(key) - 1)
        inside = ((at + step < len(key)) & (p[row] == np.arange(P)[:, None]) & (d[row] < g.hi[None, :]) & full[row]
                  & (start >= 0)[None, :])
        S -= np.where(inside[..., None], terms[row], 0.0)
    N, sa, sb, saa, sbb, sab = np.moveaxis(S, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (sab - sa * sb / N) / np.sqrt((saa - sa * sa / N) * (sbb - sb * sb / N))
        out = rho * ratio
    return np.where((n >= MIN_OBS_BETA_VOL) & (N >= MIN_OBS_BETA_CORR) & np.isfinite(out), out, np.nan)


# ---------------------------------------------------------------------------------------------------- the table
def by_formation_month(g: Daily, only: list[str] | None = None) -> dict[str, np.ndarray]:
    """Every characteristic as a P x M array indexed by formation month."""
    want = set(only or CHARS)
    out = {}
    if want & set(PAST_RETURNS):
        out.update(past_returns(monthly_returns(g)))
    for name, fn in (("LME", size), ("LTurnover", turnover), ("Rel2High", rel_to_high), ("Resid_Var", resid_var),
                     ("Spread", spread), ("SUV", suv), ("Variance", variance), ("Beta", beta)):
        if name in want:
            out[name] = fn(g)
    return {c: out[c] for c in CHARS if c in want}


def table(g: Daily, only: list[str] | None = None) -> pd.DataFrame:
    """permno, month (holding month), formation_date and the characteristics.

    A row exists where the security has a daily row in the formation month, for the holding months up to the last
    month of the data."""
    P, M = g.shape
    values = by_formation_month(g, only)
    present = _msum(g, np.ones(len(g.p))) > 0
    present[:, M - 1] = False                            # its holding month lies after the data
    present[:, g.hi <= g.lo] = False                     # a month without trading days forms nothing
    pi, fi = np.nonzero(present)
    out = pd.DataFrame({"permno": g.permnos[pi].astype("int32"), "month": (g.months + 1)[fi],
                        "formation_date": pd.DatetimeIndex(g.days[np.maximum(g.hi - 1, 0)])[fi]})
    for c, a in values.items():
        out[c] = a[pi, fi].astype("float64")
    return out


def characteristics(dsf: pd.DataFrame, factors: pd.DataFrame, only: list[str] | None = None) -> pd.DataFrame:
    """The table from in-memory frames: `dsf` with the columns DSF_USED, `factors` with date and FACTORS."""
    return table(prepare(dsf, factors), only)


def read_french(path: str | Path) -> pd.DataFrame:
    """Kenneth French's daily file F-F_Research_Data_Factors_daily (.txt or .csv, in percent) as date, mktrf, smb, hml, rf."""
    rows = []
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append((parts[0], *(float(v) / 100.0 for v in parts[1:5])))
    out = pd.DataFrame(rows, columns=["date", *FACTORS])
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    return out


def load_inputs(ff_vintage: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dsf = load("crsp_dsf_v2", columns=DSF_USED)
    factors = read_french(ff_vintage) if ff_vintage else load("ff_factors_daily", columns=["date", *FACTORS])
    _release()
    return dsf, factors


def _release() -> None:
    """Hands the memory that pyarrow keeps after reading a file back to the system (2 GB after the daily file)."""
    import pyarrow

    pyarrow.default_memory_pool().release_unused()


def from_files(ff_vintage: str | Path | None = None, only: list[str] | None = None) -> pd.DataFrame:
    """The table from the files on disk. The daily frame is dropped once its columns are on the grid: the peak is
    about 4 GB of memory."""
    dsf, factors = load_inputs(ff_vintage)
    g = prepare(dsf, factors)
    del dsf
    _release()
    return table(g, only)


# ---------------------------------------------------------------------------------------------------- command line
def build() -> pd.DataFrame:
    t0 = time.time()
    df = from_files()
    CHAR_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"  {OUT_FILE}: {len(df):,} rows, {df.permno.nunique():,} securities, holding months {df.month.min()} to "
          f"{df.month.max()}, {time.time() - t0:.0f} s")
    cover = pd.DataFrame({"first month": [df.month[df[c].notna()].min() for c in CHARS],
                          "share with a value": [df[c].notna().mean() for c in CHARS]}, index=CHARS)
    print(cover.to_string(float_format=lambda v: f"{v:.4f}"))
    return df


def security_type(dsf_types: pd.DataFrame) -> pd.Series:
    """ADR, foreign, REIT, US common or other, from the CIZ classification fields."""
    t = np.where(dsf_types.sharetype == "AD", "ADR", np.where(dsf_types.usincflg == "N", "foreign", np.where(
        dsf_types.issuertype == "REIT", "REIT", np.where(dsf_types.sharetype == "NS", "US common", "other"))))
    return pd.Series(t, index=dsf_types.index)


def relative_match(df: pd.DataFrame, chars: list[str]) -> pd.DataFrame:
    """Shares of stock-months within a purely relative distance of the panel (the standard `equal` has an absolute
    term of 1e-6 that is lenient for values as small as Resid_Var's)."""
    from afsa.pelger_check import PELGER_NAME, load_pelger

    p = load_pelger([PELGER_NAME[c] for c in chars]).rename(columns={PELGER_NAME[c]: f"{c}__p" for c in chars})
    m = df[["permno", "month", *chars]].merge(p.drop(columns="date"), on=["permno", "month"])
    rows = []
    for c in chars:
        a, b = m[c].to_numpy(), m[f"{c}__p"].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
        rel = np.abs(a[ok] / b[ok] - 1.0)
        rows.append({"characteristic": c, "n": int(ok.sum()), "rel<=1e-4": (rel <= 1e-4).mean(), "rel<=1e-3": (rel <= 1e-3).mean(),
                     "rel<=1e-2": (rel <= 1e-2).mean(), "median ours/theirs": np.median(a[ok] / b[ok])})
    return pd.DataFrame(rows)


def check(ff_vintage: str | None = None) -> None:
    from afsa.pelger_check import compare

    fmt = lambda v: f"{v:.4f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
    if ff_vintage:
        print(f"Resid_Var and Beta with the factor file {ff_vintage}")
        df = from_files(ff_vintage, only=["Resid_Var", "Beta"])
        print(compare(df).to_string(index=False, float_format=fmt))
        print(relative_match(df, ["Resid_Var", "Beta"]).to_string(index=False, float_format=fmt))
        df["year"] = df.month.dt.year
        print("\nequal by holding year")
        print(compare(df, by="year").pivot(index="year", columns="characteristic", values="equal").to_string(float_format=fmt))
        return
    df = pd.read_parquet(OUT_FILE)
    print(f"{OUT_FILE.name}: {len(df):,} rows against the CPZ panel (holding months to 2016-12)")
    print(compare(df).to_string(index=False, float_format=fmt))
    print("\npurely relative distance, for the two characteristics that depend on the Fama-French vintage")
    print(relative_match(df, ["Resid_Var", "Beta"]).to_string(index=False, float_format=fmt))
    df["period"] = pd.cut(df.month.dt.year, [1983, 1989, 1997, 2005, 2010, 2016], labels=["1984-89", "1990-97", "1998-05", "2006-10", "2011-16"]).astype(str)
    by = compare(df[df.month.dt.year <= 2016], by="period")
    for stat in ("equal", "within1"):
        print(f"\n{stat} by period")
        print(by.pivot(index="characteristic", columns="period", values=stat).reindex(CHARS).to_string(float_format=fmt))
    # by security type at formation (the classification of the daily file on the security's last day of the month)
    types = load("crsp_dsf_v2", columns=["permno", "dlycaldt", "sharetype", "issuertype", "usincflg"])
    types["month"] = types.dlycaldt.dt.to_period("M") + 1
    types = types.drop_duplicates(["permno", "month"], keep="last")
    types["type"] = security_type(types)
    df = df.merge(types[["permno", "month", "type"]], on=["permno", "month"], how="left")
    by = compare(df[df.month.dt.year <= 2016], by="type")
    print("\nequal by security type")
    print(by.pivot(index="characteristic", columns="type", values="equal").reindex(CHARS).to_string(float_format=fmt))
    print(by.groupby("type").n.max().rename("stock-months").to_frame().T.to_string())
    # LME against the monthly file: the same number that ranks the universe
    msf = load("crsp_msf_v2_198512_202512", columns=["permno", "yyyymm", "mthcap"])
    msf["month"] = pd.PeriodIndex.from_fields(year=msf.yyyymm.to_numpy() // 100, month=msf.yyyymm.to_numpy() % 100, freq="M") + 1
    j = df[["permno", "month", "LME"]].merge(msf, on=["permno", "month"])
    both = j.LME.notna() & j.mthcap.notna()
    print(f"\nLME against mthcap / 1000 of the monthly file: {both.sum():,} stock-months, "
          f"{(np.abs(j.LME - j.mthcap / 1000)[both] <= 1e-9 * j.LME[both]).mean():.6f} equal; "
          f"{(j.LME.isna() != j.mthcap.isna()).sum()} with only one of the two")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["build", "check"])
    p.add_argument("--ff-vintage", help="check only: a copy of French's F-F_Research_Data_Factors_daily file (.txt or .csv)")
    a = p.parse_args()
    if a.command == "build":
        build()
    else:
        check(a.ff_vintage)


if __name__ == "__main__":
    main()
