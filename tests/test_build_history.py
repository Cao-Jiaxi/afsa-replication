"""afsa.build_history: the month blocks of the rebuilt-history entry rule, on the synthetic market of
tests/test_build_wrds.py (no licensed data), then the blocks on disk (skipped until they are built).

The synthetic lists leave out two of eight securities each month, rotating, so every month has two entrants whose
lookback rows are days before they entered. The blocks are compared with a reference written with loops, with the day x
slot arrays of afsa.build_wrds (the months' own rows must equal them), and with themselves after the future is replaced."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from test_build_wrds import CAL, assemble_world, make_world

from afsa import build_history as bh
from afsa import build_wrds as bw
from afsa.build_window import DAILY
from afsa.config import CHARACTERISTICS, N_CHAR, N_LEVEL, N_SLOTS, TRAIN_DATA_DIR, data_folder

L = 6           # lookback of the synthetic blocks: the synthetic calendar has 22 trading days before its first list
K = {c: k for k, c in enumerate(CHARACTERISTICS)}


def world_with_accounts(seed: int = 0) -> dict:
    """make_world plus the fiscal year-end (`datadate`) each holding month uses: five months before the month, which is
    before every security's first row (the file starts on 1989-10-02), so `accounts_post` is false by default. 10010
    has no annual report before 1990-01, then one for a year ending 1989-11-30, after its first row. 10047, listed on
    1989-12-20, has only a fiscal year that ended before its listing until 1990-02, then one ending 1989-12-31."""
    w = make_world(seed)
    acc = w["accounting"]
    acc["datadate"] = (acc["month"].dt.start_time - pd.DateOffset(months=5)).to_numpy()
    acc.loc[(acc.permno == 10010) & (acc.month < pd.Period("1990-01", "M")), "datadate"] = pd.NaT
    acc.loc[(acc.permno == 10010) & (acc.month >= pd.Period("1990-01", "M")), "datadate"] = pd.Timestamp("1989-11-30")
    acc.loc[(acc.permno == 10047) & (acc.month < pd.Period("1990-02", "M")), "datadate"] = pd.Timestamp("1989-06-30")
    acc.loc[(acc.permno == 10047) & (acc.month >= pd.Period("1990-02", "M")), "datadate"] = pd.Timestamp("1989-12-31")
    return w


def build_world(world: dict, tmp_path, monkeypatch, fill: str = "none", lookback: int = L) -> tuple[dict, dict, dict]:
    """(sources, the day x slot arrays of afsa.build_wrds, the assembled blocks) of a synthetic world."""
    src, a = assemble_world(world, tmp_path, monkeypatch, fill)
    return src, a["arrays"], bh.assemble(src, "wrds_broad", lookback)


def reference(world: dict, lookback: int) -> dict:
    """The blocks written out with loops and dictionaries (fill none): {month: {"permno", "rows": [per row: date,
    member list, ranks, medians, rf of the day before, returns, listed, run], "accounts", "accounts_post"}}."""
    dsf = world["dsf"]
    row = {(int(p), pd.Timestamp(d)): (r, v) for p, d, r, v in zip(dsf.permno, dsf.dlycaldt, dsf.dlyret, dsf.dlyvol)}
    monthly = {}
    for name in ("market", "accounting"):
        df = world[name]
        for c in CHARACTERISTICS:
            if c in df.columns:
                monthly.update({(c, int(p), m): x for p, m, x in zip(df.permno, df.month, df[c])})
    rf_m = dict(zip(pd.PeriodIndex(world["rf_month"]["date"], freq="M"), world["rf_month"]["rf"]))
    n_m = pd.Series(1, index=CAL.to_period("M")).groupby(level=0).sum()
    rf_day = {d: (1 + rf_m[d.to_period("M")]) ** (1 / n_m[d.to_period("M")]) - 1 for d in CAL}
    lists = world["lists"]

    def x_of(c, p, i):                                       # what was known at the close of CAL[i - 1]
        if c not in DAILY:
            return monthly.get((c, p, CAL[i].to_period("M")), np.nan)          # the row's own month
        if c == "Ret_D1":
            return row.get((p, CAL[i - 1]), (np.nan, np.nan))[0] if i >= 1 else np.nan
        if i < 5:
            return np.nan
        w = [row.get((p, CAL[j]), (np.nan, np.nan)) for j in range(i - 5, i)]
        r, v = np.array([x[0] for x in w]), np.array([x[1] for x in w])
        if c == "Vol":
            return v.sum()
        return np.nan if np.isnan(r).any() else round(float(np.prod(1 + r) - 1 if c == "Ret_W1" else np.std(r, ddof=1)), 12)

    def streak(p, i, inside):                                # consecutive days j <= i with inside(p, j)
        n = 0
        while i - n >= 0 and inside(p, i - n):
            n += 1
        return n

    has = lambda p, j: (p, CAL[j]) in row                                          # noqa: E731
    in_universe = lambda p, j: CAL[j].to_period("M") in lists and p in lists[CAL[j].to_period("M")] and has(p, j)   # noqa: E731
    first = dsf.groupby("permno").dlycaldt.min()
    acc = world["accounting"]
    out = {}
    for m, slots in lists.items():
        days = [i for i, d in enumerate(CAL) if d.to_period("M") == m]
        rows = []
        for i in range(days[0] - lookback, days[-1] + 1):
            mem = [p for p in slots if has(p, i)]
            x = np.array([[x_of(c, p, i) for c in CHARACTERISTICS] for p in mem])
            ranks, med = np.zeros(x.shape), np.zeros(N_CHAR)
            for k in range(N_CHAR):
                ok = ~np.isnan(x[:, k])
                if ok.sum():
                    ranks[ok, k] = (rankdata(x[ok, k]) - 1) / max(ok.sum() - 1, 1) - 0.5
                    med[k] = np.median(x[ok, k])
            rows.append({"date": CAL[i], "member": mem, "ranks": ranks, "median": med, "rf_prev": rf_day[CAL[i - 1]],
                         "ret": np.nan_to_num([row[(p, CAL[i])][0] for p in mem], nan=0.0),
                         "listed": [streak(p, i, has) for p in mem], "run": [streak(p, i, in_universe) for p in mem]})
        upto = acc[acc["month"] <= m]
        entered = upto[upto["datadate"].notna()]
        out[m] = {"permno": slots, "rows": rows,
                  "accounts": [p in set(entered.permno) for p in slots],
                  "accounts_post": [bool((entered.loc[entered.permno == p, "datadate"] >= first[p]).any()) for p in slots]}
    return out


# ---------------------------------------------------------------------------------------------------- pieces
def test_consecutive_counts_runs_and_restarts():
    present = np.array([[1, 0], [1, 1], [0, 1], [1, 1], [1, 0]], dtype=bool)
    assert bh.consecutive(present).tolist() == [[1, 0], [2, 1], [0, 2], [1, 3], [2, 0]]


def test_history_folder_names():
    assert bh.history_folder("wrds_broad", "none") == TRAIN_DATA_DIR / "history" / "wrds_broad_fill-none"
    assert bh.history_folder("wrds_common", "none", 30) == TRAIN_DATA_DIR / "history" / "wrds_common_fill-none"
    assert bh.history_folder("wrds_broad", "none", 60).name == "wrds_broad_fill-none_lookback-60"


def test_month_rows_layout(tmp_path, monkeypatch):
    src, _, blk = build_world(world_with_accounts(), tmp_path, monkeypatch)
    arr = blk["arrays"]
    months = pd.PeriodIndex(arr["months"].astype(str), freq="M")
    assert list(months.astype(str)) == ["1989-11", "1989-12", "1990-01", "1990-02", "1990-03"]
    for b, m in enumerate(months):
        n = int(arr["n_rows"][b])
        rows = arr["row"][b, :n]
        assert np.array_equal(rows, rows[0] + np.arange(n)) and (arr["row"][b, n:] == -1).all()        # consecutive, then padding
        dates = pd.DatetimeIndex(arr["dates"][rows])
        assert (dates[:L].to_period("M") < m).all() and (dates[L:].to_period("M") == m).all()        # L days before, then the month
        assert n - L == int((CAL.to_period("M") == m).sum())                                          # all of the month's days
    month_index = np.flatnonzero(bw.universe_lists("wrds_broad", src)[0].any(axis=1))
    with pytest.raises(ValueError, match="consecutive"):
        bh.month_rows(src, L, month_index[[0, 2]])
    with pytest.raises(ValueError, match="trading days"):
        bh.month_rows(src, 25, month_index)                                   # 22 days before the first list: too few


# ---------------------------------------------------------------------------------------------------- the blocks
def test_blocks_equal_a_loop_reference(tmp_path, monkeypatch):
    world = world_with_accounts()
    _, _, blk = build_world(world, tmp_path, monkeypatch)
    arr, ref = blk["arrays"], reference(world, L)
    months = pd.PeriodIndex(arr["months"].astype(str), freq="M")
    assert list(months) == list(ref)
    n_entrant_rows = 0
    for b, m in enumerate(months):
        r = ref[m]
        n_list = len(r["permno"])
        assert np.array_equal(arr["permno"][b, :n_list], r["permno"]) and (arr["permno"][b, n_list:] == -1).all()
        assert arr["accounts"][b, :n_list].tolist() == r["accounts"] and arr["accounts_post"][b, :n_list].tolist() == r["accounts_post"]
        assert not arr["member"][b, :, n_list:].any() and not arr["accounts"][b, n_list:].any()
        assert len(r["rows"]) == arr["n_rows"][b]
        for i, rr in enumerate(r["rows"]):
            assert pd.Timestamp(arr["dates"][arr["row"][b, i]]) == rr["date"]
            mem = arr["member"][b, i]
            assert arr["permno"][b][mem].tolist() == rr["member"]
            assert np.allclose(arr["ranks"][b, i][mem], rr["ranks"], atol=1e-6) and not arr["ranks"][b, i][~mem].any()
            assert np.allclose(arr["levels"][b, i, :N_CHAR], rr["median"], rtol=1e-6, atol=1e-12)
            assert arr["levels"][b, i, N_CHAR] == np.float32(rr["rf_prev"])
            assert np.array_equal(arr["ret"][b, i][mem], rr["ret"].astype(np.float32)) and not arr["ret"][b, i][~mem].any()
            assert arr["listed"][b, i][mem].tolist() == rr["listed"] and arr["run"][b, i][mem].tolist() == rr["run"]
            n_entrant_rows += int(i < L and any(run == 0 for run in rr["run"]))
        assert not arr["ranks"][b, arr["n_rows"][b]:].any() and not arr["levels"][b, arr["n_rows"][b]:].any()   # padding
    assert n_entrant_rows >= 20                              # lookback rows on which a member is outside that day's universe


def test_own_rows_equal_the_day_arrays(tmp_path, monkeypatch):
    """On a month's own days the blocks hold exactly the day x slot arrays: members in the same order, ranks, medians,
    rf input, returns and run, bit for bit (also what the build checks on the real arrays)."""
    _, day, blk = build_world(world_with_accounts(), tmp_path, monkeypatch)
    v = bh.against_day_arrays(blk["arrays"], L, day)
    assert v["same_days"] and v["members"] and v["run"] and (v["rank"], v["level"], v["ret"]) == (0, 0, 0)
    assert v["days"] == len(day["dates"]) and v["stock_days"] == int((day["permno"] >= 0).sum())
    broken = {k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in blk["arrays"].items()}
    broken["ranks"][2, L + 3, 0, 5] += 0.01                                    # one rank of one own row
    assert bh.against_day_arrays(broken, L, day)["rank"] == pytest.approx(0.01, abs=1e-6)


def test_an_entrant_has_its_days_before_entry(tmp_path, monkeypatch):
    """10002 and 10020 are on the January list but not on December's: the January block's lookback rows (December
    days) hold their real returns and characteristics, ranked over January's list, while the day x slot arrays have
    nothing for them on those days. On January's first day they have run 1 but listed > L: a signal under the
    decided rule, none under the current one."""
    world = world_with_accounts()
    _, day, blk = build_world(world, tmp_path, monkeypatch)
    arr = blk["arrays"]
    b = int(np.flatnonzero(arr["months"].astype(str) == "1990-01")[0])
    assert {10002, 10020} <= set(world["lists"][pd.Period("1990-01", "M")]) and not {10002, 10020} & set(world["lists"][pd.Period("1989-12", "M")])
    ret = world["dsf"].set_index(["permno", "dlycaldt"])["dlyret"]
    for p in (10002, 10020):
        s = int(np.flatnonzero(arr["permno"][b] == p)[0])
        for i in range(L):                                                     # December days
            d = pd.Timestamp(arr["dates"][arr["row"][b, i]])
            assert d.month == 12 and arr["member"][b, i, s] and arr["run"][b, i, s] == 0
            assert arr["ret"][b, i, s] == np.float32(ret[(p, d)])
            assert (arr["ranks"][b, i, s] != 0).sum() > 25                     # ranked, not left at the middle
            t = int(np.searchsorted(day["dates"], np.datetime64(d, "D")))
            assert p not in day["permno"][t]                                   # absent from that day's universe
        assert arr["run"][b, L, s] == 1 and arr["listed"][b, L, s] > L        # first day in the universe, long listed


def test_lookback_rows_use_their_own_months_characteristics(tmp_path, monkeypatch):
    """OA of 10001 is the largest in 1990-01 and missing in 1990-02. The February block's lookback rows are January
    days: there OA ranks at the top (January's value), on February's own days at 0 (no value). Using the block's month
    on a lookback row would put February's values, formed after those days, into them."""
    _, _, blk = build_world(world_with_accounts(), tmp_path, monkeypatch)
    arr = blk["arrays"]
    b = int(np.flatnonzero(arr["months"].astype(str) == "1990-02")[0])
    s = int(np.flatnonzero(arr["permno"][b] == 10001)[0])
    assert all(arr["member"][b, :arr["n_rows"][b], s])
    assert (arr["ranks"][b, :L, s, K["OA"]] == 0.5).all()
    assert (arr["ranks"][b, L:arr["n_rows"][b], s, K["OA"]] == 0.0).all()


def test_accounts_flags(tmp_path, monkeypatch):
    world = world_with_accounts()
    _, _, blk = build_world(world, tmp_path, monkeypatch)
    arr = blk["arrays"]
    months = arr["months"].astype(str).tolist()

    def flags(p):
        out = {}
        for b, m in enumerate(months):
            s = np.flatnonzero(arr["permno"][b] == p)
            if len(s):
                out[m] = (bool(arr["accounts"][b, s[0]]), bool(arr["accounts_post"][b, s[0]]))
        return out
    f10010 = flags(10010)                                    # no annual report before 1990-01
    assert all(v == (False, False) for m, v in f10010.items() if m < "1990-01") and any(m < "1990-01" for m in f10010)
    assert all(v == (True, True) for m, v in f10010.items() if m >= "1990-01")
    f10047 = flags(10047)                                    # a pre-listing report until 1990-02: accounts, not accounts_post
    assert all(v == (True, False) for m, v in f10047.items() if m < "1990-02") and any(m < "1990-02" for m in f10047)
    assert all(v == (True, True) for m, v in f10047.items() if m >= "1990-02")
    f10001 = flags(10001)                                    # every year-end before the file's first day
    assert f10001 and all(v == (True, False) for v in f10001.values())


def test_no_look_ahead_in_the_blocks(tmp_path, monkeypatch):
    """Everything dated on or after day D is replaced (returns, volumes, the characteristics of later months, the
    rates of later months): every block row before D is unchanged, and so are the inputs of row D itself."""
    world = world_with_accounts()
    base = build_world(world, tmp_path / "a", monkeypatch)[2]["arrays"]
    rng = np.random.default_rng(5)
    for D in (pd.Timestamp("1989-12-12"), pd.Timestamp("1990-01-31"), pd.Timestamp("1990-02-01")):
        w = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in world.items()}
        late = w["dsf"]["dlycaldt"] >= D
        w["dsf"].loc[late, "dlyret"] = rng.normal(0, 0.3, int(late.sum()))
        w["dsf"].loc[late, "dlyvol"] = rng.uniform(1, 100, int(late.sum()))
        for name in ("market", "accounting"):
            later = w[name]["month"] > D.to_period("M")
            chars = [c for c in w[name].columns if c in CHARACTERISTICS]
            w[name].loc[later, chars] = rng.normal(size=(int(later.sum()), len(chars)))
        w["rf_month"].loc[w["rf_month"]["date"] > D, "rf"] = 0.05
        new = build_world(w, tmp_path / f"b{D:%m%d}", monkeypatch)[2]["arrays"]
        assert np.array_equal(new["row"], base["row"]) and np.array_equal(new["permno"], base["permno"])
        when = new["dates"][np.where(new["row"] >= 0, new["row"], 0)]
        before, upto = (when < np.datetime64(D, "D")) & (new["row"] >= 0), (when <= np.datetime64(D, "D")) & (new["row"] >= 0)
        for name in ("member", "ret", "listed", "run"):
            assert np.array_equal(new[name][before], base[name][before]), name
        assert np.array_equal(new["ranks"][upto], base["ranks"][upto]) and np.array_equal(new["levels"][upto], base["levels"][upto])
        on_d = when == np.datetime64(D, "D")
        assert on_d.any() and not np.array_equal(new["ret"][on_d], base["ret"][on_d])           # the day's own return does move
        assert np.array_equal(new["accounts"], base["accounts"])


# ---------------------------------------------------------------------------------------------------- the blocks on disk
BUILT = [u for u in ("wrds_broad", "wrds_common") if (bh.history_folder(u, "none") / "meta.json").exists()]
needs_blocks = pytest.mark.skipif(not BUILT, reason="run `python -m afsa.build_history` first")


@pytest.fixture(scope="module", params=BUILT or [None])
def folder(request):
    return bh.history_folder(request.param, "none")


@needs_blocks
def test_written_meta(folder):
    meta = json.loads((folder / "meta.json").read_text())
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == []
    passed = " | ".join(c["check"] for c in meta["checks"] if c["ok"] is True)
    for topic in ("no gap inside a security's life", "one block per holding month", "equal the day x slot arrays",
                  "recomputation with pandas from the whole table", "its lags exist", "no look-ahead", "never turn false"):
        assert topic in passed, topic
    assert (meta["builder"], meta["fill"], meta["lookback"], meta["n_months"], meta["n_slots"]) == ("afsa.build_history", "none", 30, 434, N_SLOTS)
    assert (meta["first_month"], meta["last_month"], meta["first_loss_day"]) == ("1989-11", "2025-12", "1990-01-02")
    cov = meta["coverage"]
    assert cov["current"] < cov["decided"] < cov["member_stock_days"] and cov["decided_post"] <= cov["decided"]
    if meta["universe"] == "wrds_broad":
        assert cov["member_stock_days"] == 4_527_395                             # the stock-days from 1990-01-02, as in the arrays


@needs_blocks
def test_blocks_on_disk_are_not_older_than_their_sources(folder):
    meta = json.loads((folder / "meta.json").read_text())
    stale = []
    for name, then in meta["sources"].items():
        p = bw.WRDS_DIR / name
        now = {"bytes": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} if p.exists() else None
        if now != then or then["modified"] > meta["built"]:
            stale.append(f"{name}: {then} at the build, {now} now")
    assert not stale, f"{folder.name} is older than its sources, rerun `python -m afsa.build_history`: " + "; ".join(stale)


@needs_blocks
def test_written_blocks(folder):
    meta = json.loads((folder / "meta.json").read_text())
    a = {f.stem: np.load(f, mmap_mode="r") for f in folder.glob("*.npy")}
    assert set(a) == set(bh.ARRAYS)
    for name, spec in meta["arrays"].items():
        assert list(a[name].shape) == spec["shape"] and str(a[name].dtype) == spec["dtype"], name
    M, R = a["row"].shape
    L30 = meta["lookback"]
    assert a["ranks"].shape == (M, R, N_SLOTS, N_CHAR) and a["levels"].shape == (M, R, N_LEVEL) and R == L30 + 23
    permno = np.asarray(a["permno"])
    assert (permno > 0).all() and (np.diff(permno, axis=1) > 0).all()          # 500 securities a month, sorted
    for b in (0, 2, 125, M - 1):
        mem = np.asarray(a["member"][b])
        r = np.asarray(a["ranks"][b])
        assert np.isfinite(r).all() and np.abs(r).max() <= 0.5 and not r[~mem].any()
        assert (np.asarray(a["listed"][b])[mem] >= 1).all() and not np.asarray(a["ret"][b])[~mem].any()


@needs_blocks
def test_own_rows_on_disk_equal_the_day_arrays(folder):
    """Three days, mapped here from the dates (not with the builder's code): each is an own row of the block of its
    month, with the day x slot arrays' members, ranks, levels and returns."""
    meta = json.loads((folder / "meta.json").read_text())
    day_folder = TRAIN_DATA_DIR / data_folder(meta["universe"], "none")
    if not (day_folder / "meta.json").exists():
        pytest.skip(f"{day_folder.name} is not built")
    d = {k: np.load(day_folder / f"{k}.npy", mmap_mode="r") for k in ("dates", "permno", "ranks", "levels", "ret")}
    a = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("dates", "months", "row", "permno", "member", "ranks", "levels", "ret")}
    months = np.asarray(a["months"]).astype(str)
    for day in ("1990-01-02", "2001-09-17", "2025-12-31"):
        b = int(np.flatnonzero(months == day[:7])[0])
        i = int(np.flatnonzero(np.asarray(a["dates"])[np.asarray(a["row"][b]).clip(0)] == np.datetime64(day))[0])
        assert i >= meta["lookback"]                                           # an own row, not a lookback row
        t = int(np.searchsorted(np.asarray(d["dates"]), np.datetime64(day)))
        mem = np.asarray(a["member"][b, i])
        n = int(mem.sum())
        assert np.array_equal(np.asarray(d["permno"][t, :n]), np.asarray(a["permno"][b])[mem])
        assert np.array_equal(np.asarray(d["ranks"][t, :n]), np.asarray(a["ranks"][b, i])[mem])
        assert np.array_equal(np.asarray(d["levels"][t]), np.asarray(a["levels"][b, i]))
        assert np.array_equal(np.asarray(d["ret"][t, :n]), np.asarray(a["ret"][b, i])[mem])


@needs_blocks
def test_a_lookback_row_on_disk_ranks_over_the_blocks_list(folder):
    """The first lookback row of May 2000 (2000-03-17, a March day) recomputed from the whole table with code of its
    own: the members are May's list with a CRSP row that day, and LME and Ret_D1 are ranked among them."""
    table = bw.WRDS_DIR / "table" / "chars_daily_fill-none.parquet"
    if not table.exists():
        pytest.skip("the whole table is not built")
    a = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("dates", "months", "row", "permno", "member", "ranks")}
    b = int(np.flatnonzero(np.asarray(a["months"]).astype(str) == "2000-05")[0])
    day = np.asarray(a["dates"])[int(a["row"][b, 0])]
    assert str(day) == "2000-03-17"
    tab = pd.read_parquet(table, columns=["date", "permno", "LME", "Ret_D1"], filters=[("date", "==", pd.Timestamp(day))])
    P = np.asarray(a["permno"][b])
    tab = tab[tab["permno"].isin(P)].assign(present=1.0).set_index("permno").reindex(P)     # a table row = a CRSP row
    present = tab["present"].notna().to_numpy()
    assert present.sum() > 450 and np.array_equal(present, np.asarray(a["member"][b, 0]))
    for c in ("LME", "Ret_D1"):
        x = tab[c].to_numpy(dtype=float)                                       # NaN off the row and without a value
        ok = ~np.isnan(x)
        want = np.zeros(len(P))
        want[ok] = (rankdata(x[ok]) - 1) / (ok.sum() - 1) - 0.5
        assert np.abs(np.asarray(a["ranks"][b, 0, :, K[c]]) - want).max() < 2e-6, c
