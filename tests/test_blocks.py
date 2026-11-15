"""Training on the month blocks (entry_rule history): afsa.panel.BlockPanel and the block forward of afsa.model.

On a universe that never changes, the blocks hold the cross-sections of the day x slot arrays, so training and
evaluating with entry_rule history must give the numbers of entry_rule run. On a changing universe (entrants with a
history, a new listing, a delisting, missing accounts) the block forward is compared with a recomputation that builds
each day's lags one row at a time. Then no look-ahead, models in one batch, a rolling run, the configuration, and the
blocks on disk against the day x slot arrays (skipped until both are built)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import YEAR_DAYS

import afsa.metrics
import afsa.run
from afsa.config import N_CHAR, N_LEVEL, TRAIN_DATA_DIR, Config
from afsa.model import AFSAModel
from afsa.panel import BlockPanel, Panel, build_command, load_panel, panel_folder
from afsa.train import STATS, Task, evaluate, fit, train_span

DT = torch.float64
L = 6                                   # the lookback of the conftest configuration
POOL, SLOTS = 18, 12


def history(cfg: Config, **kw) -> Config:
    return dataclasses.replace(cfg, universe="wrds_broad", entry_rule="history", **kw)


def calendar() -> np.ndarray:
    """The days of conftest (2000: 70, 2001: 70, 2002: 40, each from 1 January) after L days of December 1999, the
    lookback of the first block. Index 6 is 2000-01-01, 66 is 2000-03-01, 76 is 2001-01-01, 107 is 2001-02-01, 135 is
    2001-03-01 and 146 is 2002-01-01."""
    lead = np.datetime64("1999-12-31") - np.arange(L)[::-1]
    return np.concatenate([lead, *(np.datetime64(f"{y}-01-01") + np.arange(n) for y, n in YEAR_DAYS.items())]).astype("datetime64[D]")


def world(seed: int = 0, fixed: bool = False) -> dict:
    """Per-day inputs of a pool of 18 stocks and one list per calendar month.

    fixed: the list is stocks 0..11 every month, and they trade every day and have accounts. Otherwise stocks 0..5, 13
    and 14 are always on the list and three of 6..12 rotate in each month (entrants with a history); 15 is listed on
    day 62 and is on the lists from March 2000, so it has no signal on its first days there; 16 is on the lists of
    January and February 2001 and delists on day 125; 14 has no annual accounts before March 2001, and 13 none at all."""
    rng = np.random.default_rng(seed)
    dates = calendar()
    C = len(dates)
    months = np.unique(dates.astype("datetime64[M]"))
    has_row = np.ones((C, POOL), dtype=bool)
    lists, accounts = {}, {}
    for k, m in enumerate(months):
        acc = np.ones(POOL, dtype=bool)
        if fixed:
            lst = set(range(SLOTS))
        else:
            lst = set(range(6)) | {13, 14} | {6 + (k + j) % 7 for j in range(3)}
            if m >= np.datetime64("2000-03"):
                lst.add(15)
            if np.datetime64("2001-01") <= m <= np.datetime64("2001-02"):
                lst.add(16)
            acc[13], acc[14] = False, m >= np.datetime64("2001-03")
        lists[m], accounts[m] = np.array(sorted(lst)), acc
    if not fixed:
        has_row[:62, 15] = False
        has_row[125:, 16] = False
    beta, f = rng.normal(size=(POOL, 2)), rng.normal(scale=0.01, size=(C, 2))
    idio, shock = np.zeros((C, POOL)), rng.normal(scale=0.01, size=(C, POOL))
    for t in range(1, C):
        idio[t] = -0.5 * idio[t - 1] + shock[t]
    return {"dates": dates, "ids": 20_000 + 3 * np.arange(POOL), "has_row": has_row, "lists": lists, "accounts": accounts,
            "ranks": rng.uniform(-0.5, 0.5, size=(C, POOL, N_CHAR)),
            "levels": rng.normal(size=(C, N_LEVEL)) * np.linspace(0.001, 1000.0, N_LEVEL),
            "ret": f @ beta.T + idio, "rf": np.full(C, 1e-4)}


def blocks(w: dict, lookback: int = L) -> dict:
    """The month blocks of a world, laid out as afsa.build_history writes them (the inputs are taken as given, not
    re-ranked over each list). Months without `lookback` earlier days get no block."""
    dates, has_row = w["dates"], w["has_row"]
    month = dates.astype("datetime64[M]")
    keep = [m for m in np.unique(month) if np.argmax(month == m) >= lookback]
    n_rows = np.array([lookback + int((month == m).sum()) for m in keep])
    M, R, S = len(keep), int(n_rows.max()), max(len(w["lists"][m]) for m in keep)
    listed = np.cumsum(has_row, axis=0)                                      # no gap inside a stock's life here
    out = {"row": np.full((M, R), -1, dtype=np.int32), "permno": np.full((M, S), -1, dtype=np.int32),
           "member": np.zeros((M, R, S), dtype=bool), "ranks": np.zeros((M, R, S, N_CHAR)), "levels": np.zeros((M, R, N_LEVEL)),
           "ret": np.zeros((M, R, S)), "listed": np.zeros((M, R, S), dtype=np.int16), "accounts": np.zeros((M, S), dtype=bool)}
    for b, m in enumerate(keep):
        first, n = int(np.argmax(month == m)), n_rows[b]
        rr, lst = np.arange(first - lookback, first - lookback + n), w["lists"][m]
        k, mem = len(lst), has_row[rr][:, lst]
        out["row"][b, :n] = rr
        out["permno"][b, :k] = w["ids"][lst]
        out["member"][b, :n, :k] = mem
        out["ranks"][b, :n, :k] = np.where(mem[..., None], w["ranks"][rr][:, lst], 0.0)
        out["levels"][b, :n] = w["levels"][rr]
        out["ret"][b, :n, :k] = np.where(mem, w["ret"][rr][:, lst], 0.0)
        out["listed"][b, :n, :k] = np.where(mem, listed[rr][:, lst], 0)
        out["accounts"][b, :k] = w["accounts"][m][lst]
    return {"dates": dates, "rf": w["rf"], "months": np.array(keep), "n_rows": n_rows.astype(np.int16), **out}


def block_panel(w: dict, dtype=DT) -> BlockPanel:
    return BlockPanel.from_arrays(L, "cpu", dtype, **blocks(w))


def day_panel(w: dict, dtype=DT) -> Panel:
    """The day x slot arrays of a fixed world: stocks 0..11 on every day."""
    permno = np.tile(w["ids"][:SLOTS], (len(w["dates"]), 1)).astype(np.int32)
    return Panel.from_arrays(w["dates"], permno, w["ranks"][:, :SLOTS], w["levels"], w["ret"][:, :SLOTS], w["rf"], "cpu", dtype)


def forward(cfg: Config, panel, firsts, n_days: int, seeds=(0, 1)) -> tuple[AFSAModel, object, dict]:
    """One untrained model per seed on the days firsts[m] - 1 .. firsts[m] + n_days - 1 (day 0 = the day before)."""
    model = AFSAModel(cfg, list(seeds), "cpu", DT).eval()
    lev = panel.levels[:20]
    model.set_level_stats(lev.mean(0).expand(len(seeds), -1), lev.std(0).expand(len(seeds), -1))
    days = torch.as_tensor(firsts)[:, None] + torch.arange(-(L + 1), n_days)[None]
    batch = panel.batch(days, L)
    with torch.no_grad():
        return model, batch, model(batch)


@torch.no_grad()
def recompute_day(model: AFSAModel, panel: BlockPanel, c: int) -> dict:
    """Weights, excess returns and own-row residuals of day c by PERMNO, with the lags built one row at a time: the
    factors and residuals of each of the L rows before c in the block of c's month, over that month's list."""
    cfg = model.cfg
    b, r = int(panel.own_block[c]), int(panel.own_row[c])

    def row(s):
        wF = model._attention(panel.ranks[b, s][None, None], panel.block_levels[b, s][None, None], panel.member[b, s][None, None])
        gram = model.gram(wF)
        return wF, gram, model.project(wF, gram, panel.exret[b, s][None, None])[0, 0]

    lags = torch.stack([row(s)[2] for s in range(r - L, r)], dim=-1)          # [S, L], oldest first
    ok = panel.member[b, r] & (panel.listed[b, r] > L) & panel.accounts[b]
    w_port = model.signal(lags[None, None] * cfg.resid_scale)[0, 0] * ok
    wF, gram, eps = row(r)
    w = model.project(wF, gram, w_port[None, None])[0, 0]
    w = w / w.abs().sum()
    mem = panel.member[b, r].numpy()
    p = panel.permno[b][mem]
    return {"w": dict(zip(p, w.numpy()[mem])), "R": dict(zip(p, panel.exret[b, r].numpy()[mem])),
            "eps": dict(zip(p, eps.numpy()[mem])), "signal": dict(zip(p, ok.numpy()[mem]))}


# ---------------------------------------------------------------------------------------------------- equivalence
def test_on_a_fixed_universe_the_blocks_train_exactly_as_the_day_arrays(cfg):
    """The same list every month, every stock trading: each lag row holds the cross-section of its own day, listed
    equals run and every stock has accounts, so entry_rule history must reproduce entry_rule run, dropout included."""
    w = world(fixed=True)
    days, blks, hist = day_panel(w), block_panel(w), history(cfg)
    tasks = [Task(2001, 0), Task(2002, 3)]
    assert train_span(cfg, days, tasks[0]) == train_span(hist, blks, tasks[0]) == (L + 1, L + 70)
    a, history_a = fit(cfg, days, tasks, dtype=DT)
    b, history_b = fit(hist, blks, tasks, dtype=DT)
    assert np.allclose(history_a, history_b, rtol=1e-9, atol=1e-12)
    for name, p in a.named_parameters():
        tol = {"rtol": 0, "atol": 1e-4} if name == "W_level" else {"rtol": 1e-8, "atol": 1e-11}   # W_level: rounding noise
        assert torch.allclose(p, getattr(b, name), **tol), name
    for x, y in zip(evaluate(cfg, days, a, tasks, keep_weights=True), evaluate(hist, blks, b, tasks, keep_weights=True)):
        assert np.array_equal(x["permnos"], y["permnos"]) and np.array_equal(x["dates"], y["dates"])
        for key in ("gross", "turnover", "short", "w_first", "w_last", "weights"):
            assert np.allclose(x[key], y[key], rtol=1e-8, atol=1e-11), key
        assert np.isclose(x["explained_var"], y["explained_var"], rtol=1e-9, atol=1e-12)


def test_block_forward_equals_a_day_by_day_recomputation(cfg):
    """Two models on different months of a changing universe: weights, returns, turnover across month ends and the
    explained variance equal the recomputation with lags built row by row inside each day's own block."""
    hist = history(cfg)
    panel = block_panel(world())
    firsts = (47, 120)                                                       # days 46..76 and 119..149: three blocks each
    model, batch, out = forward(hist, panel, firsts, 30)
    assert batch.X.shape[1] == 3 and (batch.day_block[:, 0] == 0).all() and (batch.day_row >= L).all()
    for m, first in enumerate(firsts):
        ref = [recompute_day(model.select([m]), panel, c) for c in range(first - 1, first + 30)]
        permnos = panel.permnos[batch.cols[m, :int(batch.n_cols[m])].numpy()]
        assert all(set(r["w"]) <= set(permnos) for r in ref)                # every member that day has a column
        W = np.array([[r["w"].get(p, 0.0) for p in permnos] for r in ref])  # [D+1, U]
        Rx = np.array([[r["R"].get(p, 0.0) for p in permnos] for r in ref])
        assert np.allclose(out["w"][m, :, :len(permnos)].numpy(), W, rtol=0, atol=1e-12)
        assert (out["w"][m, :, len(permnos):] == 0).all()                   # padding columns hold nothing
        assert np.allclose(out["ret"][m].numpy(), (W[1:] * Rx[1:]).sum(1), atol=1e-12)
        assert np.allclose(out["turnover"][m].numpy(), np.abs(np.diff(W, axis=0)).sum(1), atol=1e-12)
        assert np.allclose(out["short"][m].numpy(), np.maximum(-W[1:], 0).sum(1), atol=1e-12)
        terms = []
        for p in permnos:                                                    # explained variance over the loss days
            e = np.array([r["eps"][p] for r in ref[1:] if p in r["eps"]])
            x = np.array([r["R"][p] for r in ref[1:] if p in r["R"]])
            if len(x) >= cfg.ev_min_days:
                terms.append(1 - e.var() / x.var())
        assert np.isclose(out["ev"][m].item(), np.mean(terms), rtol=1e-10)
    # the cases of the rule do occur: 15 waits for its 7th day since listing (day 68), 16 leaves on delisting,
    # 14 gets a signal once its accounts enter (March 2001), 13 never, but both stay in the hedge
    ids = world()["ids"]
    days0 = {c: recompute_day(model.select([0]), panel, c) for c in range(46, 77)}
    days1 = {c: recompute_day(model.select([1]), panel, c) for c in range(119, 150)}
    assert [bool(days0[c]["signal"][ids[15]]) for c in (66, 67, 68, 69)] == [False, False, True, True]
    assert ids[16] in days1[124]["w"] and ids[16] not in days1[125]["w"]
    assert not days1[134]["signal"][ids[14]] and days1[135]["signal"][ids[14]] and days1[134]["w"][ids[14]] != 0
    assert any(ids[13] in d["w"] and not d["signal"][ids[13]] and d["w"][ids[13]] != 0 for d in days1.values())


def test_block_weights_use_no_information_from_the_same_or_later_days(cfg):
    """Changing returns from day t0 on, and inputs after t0, in every block, leaves the weights up to t0 unchanged."""
    hist, t0, first = history(cfg), 100, 80
    w = world()
    changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
    rng = np.random.default_rng(1)
    changed["ret"][t0:] = rng.normal(scale=0.02, size=changed["ret"][t0:].shape)
    changed["ranks"][t0 + 1:] = rng.uniform(-0.5, 0.5, size=changed["ranks"][t0 + 1:].shape)
    changed["levels"][t0 + 1:] += 1.0
    _, _, a = forward(hist, block_panel(w), (first,), 40, seeds=(0,))
    _, _, b = forward(hist, block_panel(changed), (first,), 40, seeds=(0,))
    row = t0 - (first - 1)                                                   # day index of t0 in the batch
    assert torch.equal(a["w"][:, :row + 1], b["w"][:, :row + 1])
    assert not torch.allclose(a["w"][:, row + 1], b["w"][:, row + 1])       # the next day does react
    assert torch.equal(a["ret"][:, :row - 1], b["ret"][:, :row - 1])
    assert not torch.allclose(a["ret"][:, row - 1], b["ret"][:, row - 1])   # the return of t0 itself changed


def test_block_training_several_models_together_equals_one_by_one(cfg):
    """Different seeds and windows in one pass (their days touch different blocks), with dropout."""
    hist, panel = history(cfg), block_panel(world())
    tasks = [Task(2002, 0), Task(2001, 0), Task(2002, 5)]
    model, hist_together = fit(hist, panel, tasks, dtype=DT)
    results = evaluate(hist, panel, model, tasks)
    assert np.isfinite(hist_together).all()
    for m, task in enumerate(tasks):
        single, h = fit(hist, panel, [task], dtype=DT)
        for name, p in single.named_parameters():
            tol = {"rtol": 0, "atol": 1e-4} if name == "W_level" else {"rtol": 1e-6, "atol": 1e-9}
            assert torch.allclose(p[0], getattr(model, name)[m], **tol), name
        assert np.allclose(hist_together[:, m], h[:, 0], rtol=1e-6, atol=1e-9)
        res = evaluate(hist, panel, single, [task])[0]
        for key in ("gross", "turnover", "short", "w_first", "w_last"):
            assert np.allclose(results[m][key], res[key], rtol=1e-6, atol=1e-9), key
        assert np.array_equal(results[m]["permnos"], res["permnos"]) and np.array_equal(results[m]["dates"], res["dates"])
    pieces = evaluate(hist, panel, model, tasks, max_bytes=1.0)              # one model at a time
    assert all(np.allclose(x["gross"], y["gross"], atol=1e-12) for x, y in zip(results, pieces))


def test_rolling_run_on_blocks(cfg, tmp_path, monkeypatch):
    hist = history(cfg)
    panel = block_panel(world(), dtype=torch.float32)
    monkeypatch.setattr(afsa.run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(afsa.metrics, "market_returns",
                        lambda universe, train_years, fill="last": pd.Series(panel.market, index=pd.DatetimeIndex(panel.dates)))
    run_dir = afsa.run.run_config(hist, [0, 1], [2001, 2002], panel, max_models=4)
    assert run_dir.name == "attn_K003_universe-wrds_broad_train_years-1_lookback-6_entry_rule-history_d_x-4_d_model-5_ev_min_days-5_epochs-2_batch_days-16"
    assert sorted(p.name for p in run_dir.glob("seed*/*.npz")) == ["2001.npz", "2001.npz", "2002.npz", "2002.npz"]
    table = afsa.metrics.summarize(run_dir)
    assert table.loc["seed0", "days"] == 110 and np.isfinite(table.loc["mean", ["SR", "SR_net", "turnover"]].astype(float)).all()
    z = np.load(run_dir / "seed0" / "2002.npz")
    assert z["history"].shape == (cfg.epochs, len(STATS)) and np.allclose(np.abs(z["w_first"]).sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------------------------------- configuration
def test_history_configuration_folders_and_guards(cfg):
    hist = history(cfg)
    assert "_entry_rule-history" in hist.name() and Config.from_dict(hist.to_dict()) == hist
    with pytest.raises(ValueError, match="wrds_broad"):
        Config(entry_rule="history")                                        # Ander's arrays have no month blocks
    wrds = Config(universe="wrds_broad", fill="none")
    assert panel_folder(wrds) == "wrds_broad_fill-none"
    assert panel_folder(dataclasses.replace(wrds, entry_rule="history")) == "history/wrds_broad_fill-none"
    assert panel_folder(dataclasses.replace(wrds, entry_rule="history", lookback=20)) == "history/wrds_broad_fill-none_lookback-20"
    assert build_command("history/wrds_broad_fill-none") == "python -m afsa.build_history --universe wrds_broad"
    assert build_command("history/wrds_common_lookback-20") == "python -m afsa.build_history --universe wrds_common --fill last --lookback 20"
    blks, days = block_panel(world()), day_panel(world(fixed=True))
    span = (47 + torch.arange(-(L + 1), 10))[None]
    with pytest.raises(ValueError, match="month blocks"):                   # the rule and the sample must match
        AFSAModel(cfg, [0], "cpu", DT)(blks.batch(span, L))
    with pytest.raises(ValueError, match="month blocks"):
        AFSAModel(hist, [0], "cpu", DT)(days.batch(span, L))
    with pytest.raises(ValueError, match="lookback"):
        blks.batch(span, L + 1)
    with pytest.raises(ValueError, match="not in the blocks"):              # day 5 is a lookback day: no block of its own
        blks.batch((5 + torch.arange(-(L + 1), 10))[None].clamp_min(0), L)
    assert not blks.covers(5, 30, L) and blks.covers(6, len(blks.dates), L) and not blks.covers(6, 30, L + 1)


# ---------------------------------------------------------------------------------------------------- the blocks on disk
HISTORY, DAYS = TRAIN_DATA_DIR / "history" / "wrds_broad_fill-none", TRAIN_DATA_DIR / "wrds_broad_fill-none"


@pytest.mark.skipif(not ((HISTORY / "meta.json").exists() and (DAYS / "meta.json").exists()),
                    reason="run `python -m afsa.build_wrds` and `python -m afsa.build_history` first")
def test_blocks_on_disk_serve_the_days_of_the_day_arrays():
    """2008 in both samples: on every day the block batch has the members and excess returns of the day x slot arrays,
    the same own-row residuals from the same model, and the signals of the decided rule as the files hold them.
    In float64: in float32 the residuals differ by up to 3e-6, rounding amplified by the ridge solve of an untrained
    model, whose K x K system has eigenvalues down to 1e-7 against a ridge of 1e-5."""
    cfg = Config(universe="wrds_broad", fill="none", K=30)
    hist = dataclasses.replace(cfg, entry_rule="history")
    blks = load_panel(hist, "cpu", first_year=2008, last_year=2008, dtype=DT)
    days = load_panel(cfg, "cpu", first_year=2008, last_year=2008, dtype=DT)
    assert isinstance(blks, BlockPanel) and blks.lookback == cfg.lookback and blks.n_slots == 500
    (sb, eb), (sd, ed) = blks.year_span(2008, 2008), days.year_span(2008, 2008)
    assert np.array_equal(blks.dates[sb:eb], days.dates[sd:ed]) and eb - sb == 253
    n, Lc = eb - sb, cfg.lookback
    a, b = AFSAModel(cfg, [0], "cpu", DT), AFSAModel(hist, [0], "cpu", DT)   # the same seed: the same parameters
    lev = days.levels[sd - days.offset:ed - days.offset].double()
    for model in (a, b):
        model.set_level_stats(lev.mean(0)[None], lev.std(0)[None])
    bd, bb = days.batch((sd - Lc + torch.arange(Lc + n))[None], Lc), blks.batch((sb - Lc + torch.arange(Lc + n))[None], Lc)
    with torch.no_grad():
        oa, ob = a(bd), b(bb)
    pa, pb = days.permnos[bd.cols[0, :int(bd.n_cols[0])].numpy()], blks.permnos[bb.cols[0, :int(bb.n_cols[0])].numpy()]
    ia = np.searchsorted(pa, pb)
    assert np.isin(pb, pa).all() and not oa["member_d"][0][:, np.setdiff1d(np.arange(len(pa)), ia)].any()
    assert torch.equal(oa["member_d"][0][:, ia], ob["member_d"][0, :, :len(pb)])
    assert torch.equal(oa["R_d"][0][:, ia], ob["R_d"][0, :, :len(pb)])
    assert torch.allclose(oa["eps_d"][0][:, ia], ob["eps_d"][0, :, :len(pb)], rtol=0, atol=1e-12)
    months = np.load(HISTORY / "months.npy")
    own = np.flatnonzero(months.astype("datetime64[Y]").astype(int) + 1970 == 2008)
    member = np.load(HISTORY / "member.npy", mmap_mode="r")[own][:, Lc:]
    listed = np.load(HISTORY / "listed.npy", mmap_mode="r")[own][:, Lc:]
    accounts = np.load(HISTORY / "accounts.npy", mmap_mode="r")[own]
    assert int(bb.signal.sum()) == int((member & (listed > Lc) & accounts[:, None, :]).sum())
    assert int(bb.signal.sum()) > int(bd.valid[0, Lc:].sum())                # the decided rule gives more signals
