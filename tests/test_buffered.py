"""The buffered universe (config.BUFFERED_UNIVERSES): each month's top 500 plus last month's members that still rank
within 600. The rule on a small ranking, the builders on the synthetic market of tests/test_build_wrds.py with lists of
changing size (600 slots), the configuration, then the list file and the folders on disk (skipped until built)."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch
from test_build_history import L, world_with_accounts
from test_build_history import reference as block_reference
from test_build_wrds import CAL, assemble_world
from test_build_wrds import reference as day_reference

from afsa import build_history as bh
from afsa import build_wrds as bw
from afsa import wrds_universe as wu
from afsa.config import (BUFFERED_UNIVERSES, N_CHAR, N_SLOTS, TRAIN_DATA_DIR, WRDS_BUILDS, WRDS_UNIVERSES, Config,
                         data_folder, n_slots)
from afsa.panel import BlockPanel, build_command, load_panel, panel_folder

NAME = "wrds_broad_buffer600"


# ---------------------------------------------------------------------------------------------------- the rule
def ranking(rows: dict[str, list[int]]) -> pd.DataFrame:
    """A ranked frame as wrds_universe.ranked returns it, from {month: PERMNOs in rank order}."""
    out = [(pd.Period(m, "M"), p, k + 1) for m, ps in rows.items() for k, p in enumerate(ps)]
    return pd.DataFrame(out, columns=["month", "permno", "rank"])


def test_buffer_rule_by_hand():
    """top 3, keep down to rank 5. 4 falls to rank 4 and stays; at rank 6 it goes. A security that delists is gone."""
    r = ranking({"2000-01": [1, 2, 3, 4, 5, 6],
                 "2000-02": [1, 2, 4, 3, 5, 6],        # 3 falls to rank 4: kept
                 "2000-03": [1, 5, 2, 6, 4, 3],        # 3 falls to rank 6: out; 4 (in since February) at rank 5: kept
                 "2000-04": [1, 2, 3, 5, 6]})          # 4 delisted (no row); 3 back in the top 3
    u = wu.buffered(r, keep=5, top=3)
    lists = u.groupby("month")["permno"].apply(lambda s: sorted(s)).to_dict()
    assert lists == {pd.Period("2000-01", "M"): [1, 2, 3], pd.Period("2000-02", "M"): [1, 2, 3, 4],
                     pd.Period("2000-03", "M"): [1, 2, 4, 5], pd.Period("2000-04", "M"): [1, 2, 3, 5]}
    assert u.loc[u["buffered"], ["month", "permno"]].astype(str).values.tolist() == [["2000-02", "3"], ["2000-03", "4"], ["2000-04", "5"]]
    c = wu.churn(u)
    assert c["size"].tolist() == [4, 4, 4] and c["enter"].tolist() == [1, 1, 1] and c["leave"].tolist() == [0, 1, 1]
    assert wu.buffered(r, keep=3, top=3).groupby("month").size().tolist() == [3, 3, 3, 3]     # keep = top: the plain rule
    with pytest.raises(ValueError, match="keep"):
        wu.buffered(r, keep=2, top=3)
    with pytest.raises(ValueError, match="every month"):
        wu.buffered(r[r["month"] != pd.Period("2000-02", "M")], keep=5, top=3)


def test_configuration_knows_the_buffered_universe():
    assert BUFFERED_UNIVERSES[NAME] == ("broad", 600) and NAME in WRDS_BUILDS and NAME not in WRDS_UNIVERSES
    assert n_slots(NAME) == 600 and n_slots("wrds_broad") == N_SLOTS == 500
    cfg = Config(universe=NAME, fill="none", entry_rule="history")
    assert panel_folder(cfg) == f"history/{NAME}_fill-none" and panel_folder(Config(universe=NAME, fill="none")) == f"{NAME}_fill-none"
    assert f"universe-{NAME}" in cfg.name()
    assert build_command(f"history/{NAME}_fill-none") == f"python -m afsa.build_history --universe {NAME}"
    assert build_command(f"{NAME}_fill-none") == f"python -m afsa.build_wrds --universe {NAME}"


# ---------------------------------------------------------------------------------------------------- the builders, 600 slots
def buffered_world() -> dict:
    """The synthetic market of test_build_history with buffered lists: from the second month, each list also keeps the
    first security of last month's list that the plain rotation dropped, so lists hold 6 or 7 securities."""
    w = world_with_accounts()
    months = list(w["lists"])
    lists = {months[0]: w["lists"][months[0]]}
    for prev, m in zip(months[:-1], months[1:]):
        dropped = sorted(set(lists[prev]) - set(w["lists"][m]))
        lists[m] = sorted(set(w["lists"][m]) | set(dropped[:1]))
    w["lists"] = lists
    w["uni"] = pd.DataFrame([(m, CAL[CAL < m.start_time][-1], p) for m, ps in lists.items() for p in ps],
                            columns=["month", "formation_date", "permno"])
    return w


def build_buffered_world(tmp_path, monkeypatch) -> tuple[dict, dict, dict, dict]:
    """(world, its day arrays with 500 and with 600 slots, its blocks with 600 slots)."""
    world = buffered_world()
    src, a = assemble_world(world, tmp_path, monkeypatch, "none")             # the same lists as `broad`, 500 slots
    world["uni"].to_parquet(tmp_path / "broad_buffer600.parquet", index=False)
    return world, a["arrays"], bw.assemble(src, NAME)["arrays"], bh.assemble(src, NAME, L)["arrays"]


def test_day_arrays_with_600_slots_equal_a_loop_reference(tmp_path, monkeypatch):
    world, day500, day, _ = build_buffered_world(tmp_path, monkeypatch)
    assert {len(v) for v in world["lists"].values()} == {6, 7}                 # lists of changing size
    assert day["permno"].shape[1] == day["ret"].shape[1] == day["ranks"].shape[1] == 600
    for name in ("permno", "ret", "run", "ranks"):                             # the first 500 slots are the 500-slot arrays
        assert np.array_equal(day[name][:, :N_SLOTS], day500[name]), name
    assert (day["permno"][:, N_SLOTS:] == -1).all() and not day["ranks"][:, N_SLOTS:].any()
    assert np.array_equal(day["levels"], day500["levels"])
    ref = day_reference(world, "none")
    for t, d in enumerate(pd.DatetimeIndex(day["dates"])):
        r = ref[d]
        n = len(r["permno"])
        assert np.array_equal(day["permno"][t, :n], r["permno"]) and (day["permno"][t, n:] == -1).all()
        assert np.allclose(day["ranks"][t, :n], r["ranks"], atol=1e-6)
        assert np.allclose(day["levels"][t, :N_CHAR], r["median"], rtol=1e-6, atol=1e-12)
        assert np.array_equal(day["ret"][t, :n], r["ret"].astype(np.float32))


def test_blocks_with_600_slots_equal_a_loop_reference(tmp_path, monkeypatch):
    world, _, day, blk = build_buffered_world(tmp_path, monkeypatch)
    assert blk["permno"].shape[1] == blk["member"].shape[2] == blk["ranks"].shape[2] == 600
    v = bh.against_day_arrays(blk, L, day)                                     # own rows = the 600-slot day arrays
    assert v["same_days"] and v["members"] and v["run"] and (v["rank"], v["level"], v["ret"]) == (0, 0, 0)
    ref = block_reference(world, L)
    for b, m in enumerate(pd.PeriodIndex(blk["months"].astype(str), freq="M")):
        r = ref[m]
        n_list = len(r["permno"])
        assert np.array_equal(blk["permno"][b, :n_list], r["permno"]) and (blk["permno"][b, n_list:] == -1).all()
        assert blk["accounts"][b, :n_list].tolist() == r["accounts"] and not blk["accounts"][b, n_list:].any()
        for i, rr in enumerate(r["rows"]):
            mem = blk["member"][b, i]
            assert blk["permno"][b][mem].tolist() == rr["member"] and not mem[n_list:].any()
            assert np.allclose(blk["ranks"][b, i][mem], rr["ranks"], atol=1e-6) and not blk["ranks"][b, i][~mem].any()
            assert np.array_equal(blk["ret"][b, i][mem], rr["ret"].astype(np.float32))
            assert blk["listed"][b, i][mem].tolist() == rr["listed"] and blk["run"][b, i][mem].tolist() == rr["run"]


# ---------------------------------------------------------------------------------------------------- on disk
LIST = wu.UNIVERSE_DIR / "broad_buffer600.parquet"
DAYS = TRAIN_DATA_DIR / data_folder(NAME, "none")
BLOCKS = bh.history_folder(NAME, "none")


@pytest.mark.skipif(not (LIST.exists() and (wu.UNIVERSE_DIR / "broad.parquet").exists()),
                    reason="run `python -m afsa.wrds_universe buffer` first")
def test_buffered_list_on_disk():
    """Every month holds the plain top 500 of broad.parquet; the others rank 501 to 600 and were on last month's list."""
    u, plain = pd.read_parquet(LIST), pd.read_parquet(wu.UNIVERSE_DIR / "broad.parquet")
    for f in (u, plain):
        f["month"] = pd.PeriodIndex(f["month"], freq="M")
    size = u.groupby("month").size()
    assert size.index.min() == pd.Period("1989-11", "M") and size.index.max() == pd.Period("2025-12", "M") and len(size) == 434
    assert size.iloc[0] == 500 and size.between(500, 600).all() and 530 < size.mean() < 560
    assert not u.duplicated(["month", "permno"]).any() and (u.formation_date.dt.to_period("M") + 1 == u.month).all()
    top = u[~u["buffered"]]
    assert set(zip(top.month, top.permno)) == set(zip(plain.month, plain.permno)) and (top["rank"] <= 500).all()
    extra = u[u["buffered"]]
    assert extra["rank"].between(501, 600).all()
    on_list = set(zip(u.month, u.permno))
    assert all((m - 1, p) in on_list for m, p in zip(extra.month, extra.permno))
    assert set(u.permno) <= set(plain.permno)                                 # every buffered security was on a plain list


def meta_of(folder):
    return json.loads((folder / "meta.json").read_text())


def stale(folder) -> list[str]:
    meta = meta_of(folder)
    out = []
    for name, then in meta["sources"].items():
        p = bw.WRDS_DIR / name
        now = {"bytes": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} if p.exists() else None
        if now != then or then["modified"] > meta["built"]:
            out.append(f"{name}: {then} at the build, {now} now")
    return out


@pytest.mark.skipif(not (DAYS / "meta.json").exists(), reason=f"run `python -m afsa.build_wrds --universe {NAME}` first")
def test_buffered_day_arrays_on_disk():
    meta = meta_of(DAYS)
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == [] and not stale(DAYS)
    assert (meta["universe"], meta["fill"], meta["n_slots"], meta["n_days"]) == (NAME, "none", 600, 9108)
    permno = np.load(DAYS / "permno.npy", mmap_mode="r")
    n = (np.asarray(permno) >= 0).sum(1)
    assert permno.shape == (9108, 600) and 495 <= n.min() and n.max() <= 600 and n.mean() > 530


@pytest.mark.skipif(not (BLOCKS / "meta.json").exists(), reason=f"run `python -m afsa.build_history --universe {NAME}` first")
def test_buffered_blocks_on_disk():
    meta = meta_of(BLOCKS)
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == [] and not stale(BLOCKS)
    assert (meta["universe"], meta["lookback"], meta["n_slots"], meta["n_months"]) == (NAME, 30, 600, 434)
    assert any("equal the day x slot arrays" in c["check"] and c["ok"] for c in meta["checks"])
    P = np.load(BLOCKS / "permno.npy")
    n = (P >= 0).sum(1)
    assert P.shape == (434, 600) and n.min() == 500 and n.max() <= 600
    assert all((np.diff(P[b, :n[b]]) > 0).all() and (P[b, n[b]:] == -1).all() for b in range(len(P)))   # sorted, empty slots last
    member = np.load(BLOCKS / "member.npy")
    assert not member[np.broadcast_to((P < 0)[:, None, :], member.shape)].any()             # an empty slot is never a member


@pytest.mark.skipif(not (BLOCKS / "meta.json").exists(), reason=f"run `python -m afsa.build_history --universe {NAME}` first")
def test_buffered_blocks_load_and_run_a_model():
    """The training path: Config, load_panel and the block forward of the model on a month of real data."""
    from afsa.model import AFSAModel

    cfg = Config(universe=NAME, fill="none", entry_rule="history", K=3, d_model=8)
    panel = load_panel(cfg, "cpu", first_year=2008, last_year=2008)
    assert isinstance(panel, BlockPanel) and panel.n_slots == 600
    start, _ = panel.year_span(2008, 2008)
    Lc = cfg.lookback
    batch = panel.batch((start - Lc + torch.arange(Lc + 22))[None], Lc)
    with torch.no_grad():
        out = AFSAModel(cfg, [0], "cpu")(batch)
    assert torch.isfinite(out["w"]).all() and torch.allclose(out["w"].abs().sum(-1), torch.ones(1), atol=1e-5)
    assert int(batch.signal[0, 1].sum()) > 500                                # more than 500 members with a signal on a day
