"""The tools of the gap analysis that work on results: the paired bootstrap, the seed ensemble and the no-trade band."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import afsa.metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "gap"))
import deploy  # noqa: E402
import paired_test  # noqa: E402


def targets(T=120, S=12, seed=0):
    """Random target books of unit gross exposure, a universe that changes, returns and a two-year calendar."""
    rng = np.random.default_rng(seed)
    member = rng.random((T, S)) > 0.15
    member[:, :3] = True
    W = rng.normal(size=(T, S)) * member
    W /= np.abs(W).sum(1, keepdims=True)
    R = rng.normal(scale=0.01, size=(T, S)) * member
    dates = np.concatenate([np.datetime64("2001-01-01") + np.arange(T // 2), np.datetime64("2002-01-01") + np.arange(T - T // 2)])
    return W, R, member, dates


def test_block_bootstrap_draws_whole_blocks_and_pairs_the_two_series():
    idx = paired_test.block_draws(100, block=21, draws=50)
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
    assert (((idx[:, 1:21] - idx[:, :20]) % 100) == 1).all()                 # 21 consecutive days, circular
    assert np.array_equal(idx, paired_test.block_draws(100, block=21, draws=50))      # a fixed stream
    rng = np.random.default_rng(1)
    common = rng.normal(0.0005, 0.01, 2000)                                  # shared noise cancels in the paired difference
    a, b = common + 0.0006 + rng.normal(0, 0.001, 2000), common + rng.normal(0, 0.001, 2000)
    r = paired_test.paired_bootstrap(a, b)
    assert np.isclose(r["delta"], paired_test.sharpe(a) - paired_test.sharpe(b)) and r["lo"] < r["delta"] < r["hi"]
    assert r["lo"] > 0 and r["z"] > 2                                        # a mean 0.6 bp higher on the same path is detected
    alone = paired_test.sharpe(a[paired_test.block_draws(2000)]).std(ddof=1)
    assert r["sd"] < 0.5 * alone                                             # far tighter than either Sharpe ratio on its own
    same = paired_test.paired_bootstrap(a, a)
    assert same["delta"] == 0 and same["lo"] == same["hi"] == 0 and np.isnan(same["z"])


def test_compare_uses_common_days_and_resamples_an_era_within_itself():
    idx = pd.bdate_range("1998-01-01", "2013-12-31")
    rng = np.random.default_rng(2)
    a = pd.DataFrame({"net": rng.normal(0.001, 0.01, len(idx))}, index=idx)
    b = pd.DataFrame({"net": rng.normal(0.0, 0.01, len(idx))}, index=idx)[:-10]
    out = paired_test.compare(a, b, eras=((1998, 2005), (2006, 2013), (2014, 2021)))
    assert out["all"]["days"] == len(idx) - 10 and set(out) == {"all", "1998-05", "2006-13"}   # 2014-21 is not covered
    m = (idx[:-10].year >= 2006)
    assert np.isclose(out["2006-13"]["delta"], paired_test.sharpe(a["net"].to_numpy()[:-10][m]) - paired_test.sharpe(b["net"].to_numpy()[m]))


def test_band_step_moves_only_beyond_the_band_and_keeps_unit_exposure():
    w = np.array([0.5, -0.3, 0.2, 0.0])
    held = np.array([0.4, -0.4, 0.0, 0.2])
    member = np.array([True, True, True, False])
    assert np.allclose(deploy.band_step(held, w, member, 0.0), w)            # no band: the target itself
    h = deploy.band_step(held, w, member, 0.15)
    # gaps of 0.1 stay put, the gap of 0.2 moves by 0.05, the stock that left is closed; then the book is rescaled
    assert np.allclose(h, np.array([0.4, -0.4, 0.05, 0.0]) / 0.85) and np.isclose(np.abs(h).sum(), 1.0)
    assert np.allclose(deploy.band_step(np.zeros(4), w, member, 0.9), w)     # a band wider than every weight, from cash: the target


def test_band_path_trading_and_ensemble():
    W, R, member, dates = targets()
    assert np.allclose(deploy.band_path(W, member, 0.0), W)
    model = deploy.trade(W, R, dates)
    assert np.isclose(model["turnover"].iloc[0], 1.0)                        # bought from cash
    assert np.allclose(model["gross"], (W * R).sum(1)) and np.allclose(model["turnover"].iloc[1:], np.abs(np.diff(W, axis=0)).sum(1))
    assert np.allclose(model["net"], model["gross"] - 5e-4 * model["turnover"] - 1e-4 * model["short"])
    wide = deploy.band_path(W, member, 0.05)
    assert np.allclose(np.abs(wide).sum(1), 1.0) and (wide[~member] == 0).all()
    assert deploy.trade(wide, R, dates)["turnover"].mean() < 0.85 * model["turnover"].mean()  # the band trades less
    # the ensemble nets the seeds' opposite positions and has unit gross exposure
    other = targets(seed=1)[0] * member
    other /= np.abs(other).sum(1, keepdims=True)
    E = deploy.ensemble([W, other])
    assert np.allclose(np.abs(E).sum(1), 1.0)
    assert np.allclose(deploy.ensemble([W, -W + 1e-9 * W]), W)                # what is left after netting, rescaled
    assert np.allclose(deploy.ensemble([W, W]), W)


def test_walk_forward_delta_uses_only_earlier_years():
    W, R, member, dates = targets()
    grid = deploy.band_grid(W, R, member, dates, deltas=(0.0, 0.02, 0.05))
    daily, chosen = deploy.walk_forward(W, R, member, dates, grid)
    assert chosen[2001] == 0.0 and set(chosen) == {2001, 2002}               # nothing to choose from in the first year
    first = pd.DatetimeIndex(dates).year == 2001
    best = max(grid, key=lambda d: deploy.net_sharpe(grid[d][first]))
    assert chosen[2002] == best
    assert np.allclose(daily["gross"][first], deploy.trade(W, R, dates)["gross"][first])
    # other returns in 2002 change neither the choice for 2002 nor the books of that year
    R2 = R.copy()
    R2[~first] = np.random.default_rng(9).normal(scale=0.05, size=R2[~first].shape) * member[~first]
    grid2 = deploy.band_grid(W, R2, member, dates, deltas=(0.0, 0.02, 0.05))
    daily2, chosen2 = deploy.walk_forward(W, R2, member, dates, grid2)
    assert chosen2 == chosen and np.allclose(daily2["turnover"], daily["turnover"]) and np.allclose(daily2["short"], daily["short"])
    # a change of delta at the turn of the year is traded from the book actually held
    held = deploy.band_path(W[first], member[first], 0.0)[-1]
    rest = deploy.trade(deploy.band_path(W[~first], member[~first], chosen[2002], held), R[~first], dates[~first], held)
    assert np.allclose(daily["turnover"][~first], rest["turnover"])


def test_walk_forward_plan_chooses_from_earlier_years_only():
    import scan
    import selection
    rng = np.random.default_rng(3)
    years = np.repeat([2001, 2002, 2003], 200)
    cands = [selection.Candidate(8, s) for s in scan.GRID[:3]]
    nets = {c: rng.normal(0.0002, 0.01, (600, 3)) for c in cands}
    nets[cands[1]][years == 2001] += 0.003                                  # the second setting wins 2001 by far
    nets[cands[2]][years == 2002] += 0.01                                   # the third wins 2002 by even more
    plan = selection.walk_forward_plan(nets, years, start=cands[0])
    assert plan == {2001: cands[0], 2002: cands[1], 2003: cands[2]}
    # the criterion: the average over columns of each column's Sharpe ratio over all earlier days
    before = years < 2003
    assert max(cands, key=lambda c: selection.mean_sharpe(nets[c][before])) == plan[2003]
    # changing 2003 changes no choice
    later = {c: x.copy() for c, x in nets.items()}
    later[cands[0]][years == 2003] += 1.0
    assert selection.walk_forward_plan(later, years, start=cands[0]) == plan
    # one column (a layer) works the same way, and ties go to the first candidate
    flat = {c: (0.001 + np.tile([0.0, 0.002], 300))[:, None] for c in cands}
    assert selection.walk_forward_plan(flat, years, start=cands[2]) == {2001: cands[2], 2002: cands[0], 2003: cands[0]}


def test_lookback_rules_score_only_the_years_they_name(monkeypatch):
    import pytest

    import lookback
    import scan
    import selection
    rng = np.random.default_rng(4)
    years = np.repeat(np.arange(2001, 2013), 120)                            # 12 test years
    cands = [selection.Candidate(8, s) for s in scan.GRID[:3]]
    nets = {c: rng.normal(0.0003, 0.004, (len(years), 1)) for c in cands}
    nets[cands[1]][years <= 2003] += 0.01                                    # the second wins the early years by far
    nets[cands[2]][(years >= 2004) & (years <= 2011)] += 0.0015              # the third is better in the 8 years before 2012
    window, before = (years >= 2004) & (years <= 2011), years < 2012
    every = lookback.make_plan(nets, years, cands[0], "all years")
    assert every == selection.walk_forward_plan(nets, years, cands[0])       # the generic scoring is the current rule
    assert every[2012] == cands[1]                                           # all years still remembers 2001-2003
    last8 = lookback.make_plan(nets, years, cands[0], "last 8 years")
    assert all(last8[y] == every[y] for y in range(2001, 2010))              # at most 8 earlier years: the same rule
    assert last8[2012] == max(cands, key=lambda c: selection.mean_sharpe(nets[c][window])) == cands[2]
    last4 = lookback.make_plan(nets, years, cands[0], "last 4 years")
    assert all(last4[y] == every[y] for y in range(2001, 2006))              # at most 4 earlier years: the same rule
    recent = (years >= 2008) & (years <= 2011)
    assert last4[2012] == max(cands, key=lambda c: selection.mean_sharpe(nets[c][recent]))
    mixed = lookback.make_plan(nets, years, cands[0], "recent return, full risk")
    assert all(mixed[y] == every[y] for y in range(2001, 2010))
    hand = {c: nets[c][window, 0].mean() / nets[c][before, 0].std(ddof=1) for c in cands}
    assert mixed[2012] == max(hand, key=hand.get)
    fade = lookback.make_plan(nets, years, cands[0], "fading memory")
    w = 0.5 ** ((2011 - years[before]) / 8)                                  # last year's weight 1, 8 years earlier 1/2
    def weighted(x):
        m = (w * x).sum() / w.sum()
        return m / np.sqrt((w * x * x).sum() / w.sum() - m * m)
    hand = {c: weighted(nets[c][before, 0]) for c in cands}
    assert fade[2012] == max(hand, key=hand.get)
    monkeypatch.setattr(lookback, "HALF_LIFE", 1e12)                         # no fading: the current rule
    assert lookback.make_plan(nets, years, cands[0], "fading memory") == every
    monkeypatch.undo()
    later = {c: x.copy() for c, x in nets.items()}                           # the last year's results change no choice
    later[cands[0]][years == 2012] += 1.0
    for rule in lookback.WALK + lookback.ADDED:
        assert lookback.make_plan(later, years, cands[0], rule) == lookback.make_plan(nets, years, cands[0], rule)
    with pytest.raises(ValueError):                                          # one series per candidate: a layer, not seeds
        lookback.moments({cands[0]: np.zeros((len(years), 3))}, years)


def test_lookback_swap_trades_one_year_as_the_other_rule():
    import lookback
    days = pd.bdate_range("1998-01-01", "2021-12-31")
    rng = np.random.default_rng(5)
    b = pd.DataFrame({"net": rng.normal(0.0003, 0.004, len(days))}, index=days)
    a = b.copy()
    a.loc[a.index.year == 2008, "net"] += 0.002                              # the two paths differ in 2008 only
    d, sw = lookback.swaps(a, b)
    assert d > 0 and list(sw.index) == list(range(1998, 2022))
    assert sw[2008] == 0.0                                                   # 2008 traded as b: the paths are equal
    assert np.allclose(sw.drop(2008), d)                                     # any other year changes nothing


def test_reality_check_separates_a_real_edge_from_the_best_of_many_draws():
    import scan
    import selection
    rng = np.random.default_rng(4)
    common = rng.normal(0.0004, 0.01, 3000)
    base = common + rng.normal(0, 0.002, 3000)
    noise = {selection.Candidate(8, s): common + rng.normal(0, 0.002, 3000) for s in scan.GRID}
    r = selection.reality_check(noise, base)
    assert r["candidates"] == len(scan.GRID) and r["p"] > 0.05               # the best of 24 copies of the baseline is luck
    assert np.isclose(r["delta"], max(paired_test.sharpe(x) for x in noise.values()) - paired_test.sharpe(base))
    seeds = {c: np.stack([x, x[::-1]], axis=1) for c, x in noise.items()}  # two columns: the criterion averages their Sharpe ratios
    assert np.isclose(selection.reality_check(seeds, np.stack([base, base[::-1]], axis=1))["delta"], r["delta"])
    edge = dict(noise)
    edge[selection.Candidate(8, scan.GRID[5])] = common + 0.0008 + rng.normal(0, 0.002, 3000)
    r = selection.reality_check(edge, base)
    assert r["best"] == selection.Candidate(8, scan.GRID[5]) and r["p"] < 0.01


def test_scan_grid_reuses_the_folders_of_earlier_runs():
    import scan
    assert len(scan.GRID) == len(set(scan.GRID)) == 24
    assert scan.BASELINE.run_dir(8).name == "attn_K008"                      # the baseline values leave the name alone
    assert scan.BOTH.run_dir(30).name == "attn_K030_ridge-0.0001_cost_trade-0.00075"
    assert scan.BASELINE.run_dir(8, 6).name == "attn_K008_train_years-6"
    assert scan.Setting(1e-3, 1e-5, 15).run_dir(8).name == "attn_K008_cost_trade-0.001_epochs-15"


def test_seed_groups_are_disjoint_and_settings_parse_off_the_grid():
    import scan
    import selection
    seeds = tuple(range(10))
    assert selection.seed_groups(seeds, 3) == [(0, 1, 2), (3, 4, 5), (6, 7, 8)]   # the remainder (seed 9) is left out
    assert selection.seed_groups(seeds, 10) == [seeds] and selection.seed_groups(seeds, 1) == [(s,) for s in seeds]
    assert scan.parse_setting("7.5/1e-4/30") == scan.BOTH and scan.parse_setting("7.5/1e-4/30") in scan.GRID
    off = scan.parse_setting("7.5/3e-5/30")                                  # not in the grid: its own folder
    assert off not in scan.GRID
    assert off.run_dir(30, extra={"universe": "wrds_broad", "fill": "none"}).name == "attn_K030_universe-wrds_broad_ridge-3e-05_fill-none_cost_trade-0.00075"
    assert scan.parse_set(["universe=wrds_broad", "fill=none", "epochs=15"]) == {"universe": "wrds_broad", "fill": "none", "epochs": 15}


def test_pilot_grid_reuses_the_rebuilt_history_runs():
    import pilot
    assert len(pilot.GRID) == len(set(pilot.GRID)) == 52 and pilot.REFERENCE in pilot.GRID
    assert sum(s.epochs == 30 for s in pilot.GRID) == len(pilot.COSTS) * len(pilot.RIDGES) == 42
    extra = pilot.parse_set(pilot.EXTRA)                                     # the reference is the rebuilt-history validation run
    assert pilot.REFERENCE.run_dir(30, 6, extra).name == "attn_K030_universe-wrds_broad_train_years-6_entry_rule-history_ridge-0.0001_fill-none"
    assert [pilot.sci(x) for x in (1e-6, 3e-5, 1e-4, 3e-3, 2.5e-4)] == ["1e-6", "3e-5", "1e-4", "3e-3", "2.5e-4"]


def test_pilot_tables_follow_the_grid_they_are_given():
    import pilot
    import scan
    assert pilot.levers(pilot.GRID) == (pilot.COSTS, pilot.RIDGES, pilot.EPOCH_POINTS, pilot.EPOCHS)
    grid = tuple(scan.parse_setting(f"{c}/{r}/30") for c in ("2.5", "5", "7.5", "10") for r in ("1e-5", "3e-5", "1e-4", "3e-4", "1e-3")) \
        + tuple(scan.parse_setting(f"{c}/1e-4/{e}") for e in (20, 50) for c in ("2.5", "5", "7.5", "10"))   # the 28 settings of 2026-11-15
    costs, ridges, points, epochs = pilot.levers(grid)
    assert len(set(grid)) == 28 and costs == (2.5e-4, 5e-4, 7.5e-4, 1e-3) and ridges == (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    assert points == tuple((c, 1e-4) for c in costs) and epochs == (20, 30, 50)
    assert pilot.levers(grid[:20])[2:] == ((), ())                           # no epoch sweep: no epoch table, no epoch edge


def test_market_row_in_log_returns(tmp_path, monkeypatch):
    folder = tmp_path / "broad"
    folder.mkdir()
    permno = np.array([[1, 2, -1], [1, 2, 3], [-1, 2, 3]])
    ret = np.array([[0.10, -0.02, 9.0], [0.03, 0.01, -0.04], [9.0, 0.02, 0.00]])
    rf = np.array([0.0002, 0.0002, 0.0001])
    for name, x in (("permno", permno), ("ret", ret), ("rf", rf), ("dates", np.array(["2001-01-02", "2001-01-03", "2001-01-04"], dtype="datetime64[D]"))):
        np.save(folder / f"{name}.npy", x)
    monkeypatch.setattr(afsa.metrics, "TRAIN_DATA_DIR", tmp_path)
    total = np.array([0.04, 0.0, 0.01])                                      # equal-weighted return of the day's members
    assert np.allclose(afsa.metrics.market_returns("broad"), total - rf)
    assert np.allclose(afsa.metrics.market_returns("broad", log=True), np.log1p(total) - np.log1p(rf))
