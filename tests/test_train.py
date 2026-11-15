"""Batching, the training loop for several models at once, evaluation, and the joining of test years."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch

import afsa.metrics
import afsa.run
from afsa.config import Config
from afsa.metrics import performance, stitch, summarize
from afsa.panel import Panel
from afsa.train import STATS, Task, batch_starts, evaluate, fit, streams, train_span

DT = torch.float64


def test_batches_are_blocks_of_consecutive_days_inside_the_window():
    cfg = Config()                                                           # 16 batches of 126 days
    rng = np.random.default_rng(0)
    for start, stop in ((31, 2024), (2024, 4040), (1000, 3021), (500, 2513)):   # 1993, 2016, 2021 and 2013 loss days
        seen = np.zeros(stop, dtype=int)
        for _ in range(20):
            first = batch_starts(rng, start, stop, cfg)
            assert len(first) == cfg.steps_per_epoch == 16
            assert first.min() >= start and first.max() + cfg.batch_days <= stop
            for f in first:
                seen[f:f + cfg.batch_days] += 1
        slack = max(0, stop - start - 16 * cfg.batch_days)                   # days left over by 16 full batches
        assert seen[start + slack:stop - slack].min() >= 20                  # all other days are used in every epoch
        assert seen[start:stop].min() >= 1 and seen[start:stop].max() <= 40 and seen[:start].sum() == 0
    a = [batch_starts(np.random.default_rng(1), 2024, 4040, cfg) for _ in range(2)]
    assert sorted(a[0]) == sorted(a[1]) and len(set(np.diff(sorted(a[0])))) == 1   # an exact tiling, in random order


def test_first_training_window_starts_once_the_lookback_exists(cfg, panel):
    assert train_span(cfg, panel, Task(2001, 0)) == (cfg.lookback + 1, 70)   # the sample starts in 2000
    assert train_span(cfg, panel, Task(2002, 0)) == (70, 140)
    with pytest.raises(ValueError):
        train_span(cfg, panel, Task(2000, 0))                                # no data before the sample


def test_a_year_the_sample_holds_only_in_part_is_not_a_training_year(cfg, arrays):
    """The arrays of afsa.build_wrds start with 41 days of 1989, the lookback of 1990. A window may not start there."""
    def with_dates(dates):
        return Panel.from_arrays(**{**arrays, "dates": dates}, device="cpu", dtype=DT)

    day = np.timedelta64(1, "D")
    # 41 days at the end of 1999, then 70 days of 2000 and 69 of 2001 (the 180 days of the synthetic sample)
    lead = with_dates(np.concatenate([np.datetime64("1999-11-01") + np.arange(41), np.datetime64("2000-01-01") + np.arange(70),
                                      np.datetime64("2001-01-01") + np.arange(69)]))
    assert lead.year_span(1999, 1999) == (0, 41) and 41 >= cfg.batch_days    # long enough for a batch: it used to be accepted
    with pytest.raises(ValueError, match="does not cover the 1 years before 2000: it starts on 1999-11-01"):
        train_span(cfg, lead, Task(2000, 0))
    two_years = Config(**{**cfg.to_dict(), "train_years": 2})
    with pytest.raises(ValueError, match="does not cover the 2 years before 2001"):
        train_span(two_years, lead, Task(2001, 0))                           # 1999-2000 would start in the partial year too
    assert train_span(cfg, lead, Task(2001, 0)) == (41, 111)                 # 2000 in full, from its first day: the lead is its lookback
    assert str(lead.dates[41]) == "2000-01-01" and train_span(cfg, lead, Task(2002, 0)) == (111, 180)
    # a sample that starts on the year's first trading day is whole, whatever that day is (Ander's arrays: 1990-01-02)
    for first in ("2000-01-01", "2000-01-02", "2000-01-04"):
        start = with_dates(arrays["dates"] + (np.datetime64(first) - arrays["dates"][0]) * (arrays["dates"] < np.datetime64("2001-01-01")))
        assert train_span(cfg, start, Task(2001, 0)) == (cfg.lookback + 1, 70)
    late = with_dates(arrays["dates"] + 20 * day * (arrays["dates"] < np.datetime64("2001-01-01")))      # starts on 2000-01-21
    with pytest.raises(ValueError, match="inside 2000"):
        train_span(cfg, late, Task(2001, 0))


def test_seeds_depend_on_the_task_only():
    assert streams(Task(2005, 1)) == streams(Task(2005, 1))
    assert len({streams(Task(y, s)) for y in (2005, 2006) for s in (0, 1)}) == 4
    assert len(set(streams(Task(2005, 1)))) == 3                             # initialization, batch order, dropout


def test_training_several_models_together_equals_training_them_one_by_one(cfg, panel):
    """Different seeds and different windows in one batch, with dropout: same parameters, history and results."""
    tasks = [Task(2002, 0), Task(2001, 0), Task(2002, 5)]
    model, history = fit(cfg, panel, tasks, dtype=DT)
    results = evaluate(cfg, panel, model, tasks)
    assert history.shape == (cfg.epochs, 3, len(STATS)) and np.isfinite(history).all()
    for m, task in enumerate(tasks):
        single, h = fit(cfg, panel, [task], dtype=DT)
        for name, p in single.named_parameters():
            # W_level multiplies the inert inputs: its gradient is pure rounding noise, which Adam turns into a
            # small drift that differs from run to run
            tol = {"rtol": 0, "atol": 1e-4} if name == "W_level" else {"rtol": 1e-6, "atol": 1e-9}
            assert torch.allclose(p[0], getattr(model, name)[m], **tol), name
        assert np.allclose(history[:, m], h[:, 0], rtol=1e-6, atol=1e-9)
        res = evaluate(cfg, panel, single, [task])[0]
        for key in ("gross", "turnover", "short", "w_first", "w_last"):
            assert np.allclose(results[m][key], res[key], rtol=1e-6, atol=1e-9), key
        assert np.array_equal(results[m]["permnos"], res["permnos"]) and np.array_equal(results[m]["dates"], res["dates"])
    assert not torch.allclose(model.Q[0], model.Q[2])                        # different seeds do differ
    # evaluating in pieces (large K on a small GPU) gives the same numbers
    pieces = evaluate(cfg, panel, model, tasks, max_bytes=1.0)
    assert all(np.allclose(a["gross"], b["gross"], atol=1e-12) for a, b in zip(results, pieces))


def test_evaluation_covers_exactly_the_test_year(cfg, panel):
    tasks = [Task(2001, 0), Task(2002, 0)]                                   # test years of 70 and 40 days
    model, _ = fit(cfg, panel, tasks, dtype=DT)
    for res, n in zip(evaluate(cfg, panel, model, tasks, keep_weights=True), (70, 40)):
        assert len(res["gross"]) == len(res["dates"]) == n and (panel.years[np.isin(panel.dates, res["dates"])] == res["test_year"]).all()
        w = res["weights"]
        assert w.shape == (n, len(res["permnos"])) and np.allclose(np.abs(w).sum(1), 1.0, atol=1e-6)
        assert np.allclose(res["turnover"][1:], np.abs(np.diff(w, axis=0)).sum(1), atol=1e-6)
        assert np.allclose(res["w_first"], w[0], atol=1e-6) and np.allclose(res["w_last"], w[-1], atol=1e-6)
        # the gross return is the weights times the excess returns of the day
        s = int(np.searchsorted(panel.dates, res["dates"][0]))
        col = np.searchsorted(panel.permnos, res["permnos"])
        slot = panel.slot_of[s:s + n][:, col].numpy()
        R = np.where(slot >= 0, np.take_along_axis(panel.exret[s:s + n].numpy(), np.maximum(slot, 0), axis=1), 0.0)
        assert np.allclose(res["gross"], (w * R).sum(1), atol=1e-6)


def test_training_raises_the_objective(panel):
    cfg = Config(K=2, d_x=4, d_model=8, lookback=6, batch_days=16, train_years=1, epochs=8, ev_min_days=5)
    _, history = fit(cfg, panel, [Task(2002, 0), Task(2002, 1)], dtype=DT)
    obj = history[:, :, STATS.index("objective")]
    assert (obj[-1] > obj[0]).all()
    assert (history[-1, :, STATS.index("sharpe_net")] > 0).all()            # the synthetic residuals mean-revert


def _save_year(folder, year, n, permnos, w_first, w_last, turnover0):
    folder.mkdir(parents=True, exist_ok=True)
    dates = (np.datetime64(f"{year}-01-01") + np.arange(n)).astype("datetime64[D]")
    np.savez(folder / f"{year}.npz", dates=dates, gross=np.full(n, 0.001), turnover=np.r_[turnover0, np.full(n - 1, 0.5)],
             short=np.full(n, 0.5), permnos=np.asarray(permnos), w_first=np.asarray(w_first), w_last=np.asarray(w_last))


def test_stitching_pays_the_trade_between_consecutive_models(tmp_path):
    _save_year(tmp_path, 2001, 3, [10, 20, 30], [0.5, -0.5, 0.0], [0.2, -0.3, 0.5], turnover0=9.0)
    _save_year(tmp_path, 2002, 3, [20, 30, 40], [-0.4, 0.1, 0.5], [0.0, 0.5, -0.5], turnover0=9.0)
    daily = stitch(tmp_path)
    assert len(daily) == 6
    assert np.isclose(daily["turnover"].iloc[0], 1.0)                        # bought from cash
    # 10: 0.2 -> 0, 20: -0.3 -> -0.4, 30: 0.5 -> 0.1, 40: 0 -> 0.5
    assert np.isclose(daily["turnover"].iloc[3], 0.2 + 0.1 + 0.4 + 0.5)
    assert np.allclose(daily["cost"], 0.0005 * daily["turnover"] + 0.0001 * 0.5)   # the paper's costs, whatever the run trained with
    assert np.allclose(daily["net"], daily["gross"] - daily["cost"])


def test_performance_metrics():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2001-01-01", periods=500)
    market = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    gross = 0.0006 + 0.3 * market + rng.normal(0, 0.003, 500)
    daily = pd.DataFrame({"gross": gross, "turnover": 0.4, "short": 0.5}, index=idx)
    daily["cost"] = 0.0005 * daily["turnover"] + 0.0001 * daily["short"]
    daily["net"] = daily["gross"] - daily["cost"]
    perf = performance(daily, market)
    assert np.isclose(perf["SR"], gross.mean() / gross.std() * np.sqrt(252))
    assert np.isclose(perf["mu"], gross.mean() * 252 * 100) and np.isclose(perf["sigma"], gross.std() * np.sqrt(252) * 100)
    assert np.isclose(perf["mu"] - perf["mu_net"], 2.5 * 252 / 100)          # 2.5 bp a day of costs
    assert abs(perf["beta"] - 0.3) < 0.05 and np.isclose(perf["cost_bp"], 2.5)


def test_rolling_run_saves_resumes_and_summarizes(cfg, panel, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(afsa.run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(afsa.metrics, "market_returns", lambda universe, train_years, fill="last": pd.Series(panel.market, index=pd.DatetimeIndex(panel.dates)))
    panel32 = type(panel).from_arrays(panel.dates, panel.permno, panel.ranks.numpy(), panel.levels.numpy(),
                                      (panel.exret.numpy() + 1e-4) * (panel.permno >= 0), np.full(panel.n_days, 1e-4))
    run_dir = afsa.run.run_config(cfg, [0, 1], [2001, 2002], panel32, max_models=3)
    assert sorted(p.name for p in run_dir.glob("seed*/*.npz")) == ["2001.npz", "2001.npz", "2002.npz", "2002.npz"]
    assert Config.load(run_dir / "config.json") == cfg and (run_dir / "seed1" / "2002.pt").exists()
    capsys.readouterr()
    afsa.run.run_config(cfg, [0, 1], [2001, 2002], panel32, max_models=3)    # nothing left to do
    assert "0 of 4 models to train" in capsys.readouterr().out
    log = json.loads((run_dir / "provenance.json").read_text())             # the code version, once per call that trained
    assert len(log) == 1 and log[0]["models"] == 4 and len(log[0]["afsa_sha256"]) == 16
    table = summarize(run_dir)
    assert list(table.index) == ["seed0", "seed1", "mean", "sd"] and table.loc["seed0", "days"] == 110
    assert np.isfinite(table.loc["mean", ["SR", "SR_net", "beta", "turnover"]].astype(float)).all()
    other = afsa.run.run_config(Config(**{**cfg.to_dict(), "lr": 0.01}), [0], [2001], panel32, tag="trial")
    assert other != run_dir and other.name.endswith("lr-0.01_trial")         # every changed parameter is in the folder name
    Config(**{**cfg.to_dict(), "ridge": 1e-3}).save(run_dir / "config.json")
    with pytest.raises(ValueError):                                          # a folder never mixes two configurations
        afsa.run.run_config(cfg, [0, 1], [2001, 2002], panel32)


def test_every_parameter_is_a_command_line_option():
    p = argparse.ArgumentParser()
    Config.add_arguments(p)
    args = p.parse_args("--K 30 --epochs 5 --lr 0.001 --lambda_var 10 --drop-groups past_returns value --use_levels false".split())
    c = Config.from_args(args)
    assert (c.K, c.epochs, c.lr, c.lambda_var, c.drop_groups, c.use_levels) == (30, 5, 0.001, 10.0, ("past_returns", "value"), False)
    assert c.d_x == 32 and c.batch_days == 126 and c.steps_per_epoch == 16   # untouched fields keep the paper's values
    assert Config.from_dict(c.to_dict()) == c
    assert c.name() == "attn_K030_use_levels-False_drop-past_returns+value_lambda_var-10.0_epochs-5_lr-0.001"
    assert Config().name() == "attn_K008"
    with pytest.raises(ValueError):
        Config(drop_groups=("momentum",))
