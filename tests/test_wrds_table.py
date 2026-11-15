"""afsa.wrds_table: the whole table of raw characteristics, on the synthetic market of test_build_wrds (no licensed
data), then the table on disk (skipped until `python -m afsa.wrds_table` has written it)."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.stats import rankdata

from afsa import build_wrds as bw
from afsa import wrds_table as wt
from afsa.build_window import DAILY
from afsa.config import CHARACTERISTICS, TRAIN_DATA_DIR, WRDS_UNIVERSES, data_folder
from test_build_wrds import CAL, MONTHS, PERMNOS, make_world

PERMCO = {int(p): 50_000 + int(p) for p in PERMNOS} | {10005: 60_002, 10002: 60_002}     # two share classes of one company
K = {c: k for k, c in enumerate(CHARACTERISTICS)}


def setup_world(tmp_path, monkeypatch) -> tuple[dict, dict, dict]:
    """The synthetic market with a second universe and PERMCOs. Each month one security is on neither list. Points
    afsa.build_wrds at the files and returns (world, sources, the arrays of both universes)."""
    world = make_world()
    world["common"] = {m: sorted(set(PERMNOS) - {PERMNOS[k % 8], PERMNOS[(k + 5) % 8]}) for k, m in enumerate(MONTHS[1:])}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, lists in (("broad", world["lists"]), ("common", world["common"])):
        pd.DataFrame([(m, CAL[CAL < m.start_time][-1], p, PERMCO[int(p)]) for m, ps in lists.items() for p in ps],
                     columns=["month", "formation_date", "permno", "permco"]).to_parquet(tmp_path / f"{name}.parquet", index=False)
    world["market"].to_parquet(tmp_path / "market.parquet", index=False)
    world["accounting"].to_parquet(tmp_path / "accounting.parquet", index=False)
    raw = {"crsp_index_daily": pd.DataFrame({"indno": bw.MARKET_INDEX, "dlycaldt": CAL}), "crsp_dsf_v2": world["dsf"],
           "ff_factors_monthly": world["rf_month"]}
    monkeypatch.setattr("afsa.wrds_data.load", lambda name, columns=None: raw[name][columns] if columns else raw[name])
    monkeypatch.setattr(bw, "UNIVERSE_DIR", tmp_path)
    monkeypatch.setattr(bw, "MARKET_FILE", tmp_path / "market.parquet")
    monkeypatch.setattr(bw, "ACCOUNTING_FILE", tmp_path / "accounting.parquet")
    src = bw.load_sources("none")
    return world, src, {u: bw.assemble(src, u)["arrays"] for u in WRDS_UNIVERSES}


def loop_table(world: dict) -> pd.DataFrame:
    """Every row of the table written out with loops and dictionaries from the synthetic files."""
    dsf = world["dsf"]
    row = {(int(p), pd.Timestamp(d)): (r, v) for p, d, r, v in zip(dsf.permno, dsf.dlycaldt, dsf.dlyret, dsf.dlyvol)}
    monthly = {}
    for name in ("market", "accounting"):
        df = world[name]
        for c in df.columns[2:]:
            monthly.update({(c, int(p), m): x for p, m, x in zip(df.permno, df.month, df[c])})
    rate = dict(zip(pd.PeriodIndex(world["rf_month"]["date"], freq="M"), world["rf_month"]["rf"]))
    n_m = pd.Series(1, index=CAL.to_period("M")).groupby(level=0).sum()
    rf_day = [(1 + rate[d.to_period("M")]) ** (1 / n_m[d.to_period("M")]) - 1 for d in CAL]

    def daily(c, p, i):                                     # for day CAL[i], from the five days before
        w = [row.get((p, CAL[j]), (np.nan, np.nan)) for j in range(i - 5, i)]
        r, v = np.array([x[0] for x in w]), np.array([x[1] for x in w])
        if c == "Ret_D1":
            return r[-1]
        if c == "Vol":
            return v.sum()
        if np.isnan(r).any():
            return np.nan
        return round(float(np.prod(1 + r) - 1 if c == "Ret_W1" else np.std(r, ddof=1)), 12)

    out = []
    for i, d in enumerate(CAL):
        if d < pd.Timestamp("1989-11-01"):
            continue
        m = d.to_period("M")
        for p in sorted(int(x) for x in PERMNOS):
            if (p, d) not in row:
                continue
            rec = {"date": d, "permno": p, "permco": PERMCO[p], "in_broad": p in world["lists"][m], "in_common": p in world["common"][m]}
            rec.update({c: daily(c, p, i) if c in DAILY else monthly.get((c, p, m), np.nan) for c in CHARACTERISTICS})
            rec["rf"] = rf_day[i - 1]
            out.append(rec)
    return pd.DataFrame(out)


def all_rows(src: dict) -> dict:
    cal = src["dates"]
    days = np.flatnonzero((cal >= bw.FIRST_DAY) & (cal <= bw.LAST_DAY))
    return wt.rows(src, days, wt.members(src, days), wt.permco_of(src["permnos"]), np.arange(len(days)))


def test_rows_equal_a_loop_over_the_synthetic_files(tmp_path, monkeypatch):
    world, src, _ = setup_world(tmp_path, monkeypatch)
    tab, ref = all_rows(src), loop_table(world)
    assert len(tab["date"]) == len(ref) and np.array_equal(tab["date"], ref["date"].to_numpy().astype("datetime64[D]"))
    for c in ("permno", "permco", "in_broad", "in_common"):
        assert np.array_equal(tab[c], ref[c].to_numpy()), c
    for c in CHARACTERISTICS:
        if c in ("Ret_W1", "STD_W1"):
            np.testing.assert_allclose(tab[c], ref[c].to_numpy(dtype=float), rtol=0, atol=2e-12, err_msg=c)
        else:
            np.testing.assert_array_equal(tab[c], ref[c].to_numpy(dtype=float), err_msg=c)
    np.testing.assert_allclose(tab["rf"], ref["rf"].to_numpy(dtype=float), rtol=0, atol=1e-15)
    # the cases the synthetic market was made for
    df = pd.DataFrame({c: tab[c] for c in ("date", "permno", "in_broad", "in_common", *CHARACTERISTICS, "rf")})
    of = {p: g.set_index("date") for p, g in df.groupby("permno")}
    assert (~df["in_broad"] & ~df["in_common"]).any() and (df["in_broad"] & df["in_common"]).any()   # rows off both lists, too
    late = of[10047]                                                            # listed on 1989-12-20
    assert late.index[0] == pd.Timestamp("1989-12-20") and np.isnan(late["Ret_D1"].iloc[0])
    assert late[["Ret_W1", "STD_W1", "Vol"]].iloc[:5].isna().all().all() and late[["Ret_W1", "STD_W1", "Vol"]].iloc[5:].notna().all().all()
    gap = of[10002]                                                             # no row on 1990-01-10
    assert pd.Timestamp("1990-01-10") not in gap.index and np.isnan(gap.loc["1990-01-11", "Ret_D1"])
    assert gap.loc["1990-01-11":"1990-01-17", "Ret_W1"].isna().all() and gap.loc["1990-01-18", ["Ret_W1", "Vol"]].notna().all()
    assert np.isnan(of[10020].loc["1990-01-16", "Ret_D1"]) and not np.isnan(of[10020].loc["1990-01-16", "Vol"])   # no return, a volume
    assert of[10050].index[-1] == pd.Timestamp("1990-02-14")                    # the delisting row is the last
    assert of[10010]["Beta"].isna().all()                                       # never has a Beta
    oa = of[10001]["OA"]                                                        # 9.9 in January 1990, none in February: not carried
    assert (oa.loc["1990-01"] == 9.9).all() and oa.loc["1990-02"].isna().all()
    assert (df.groupby([df["date"].dt.to_period("M"), "permno"])[list(set(CHARACTERISTICS) - set(DAILY))].nunique(dropna=False) <= 1).all().all()


def test_build_checks_every_cell_and_writes_a_readable_file(tmp_path, monkeypatch):
    world, src, arrays = setup_world(tmp_path / "src", monkeypatch)
    out = tmp_path / "table"
    assert wt.build(src=src, arrays=arrays, out_dir=out)
    assert not (out / (wt.TABLE_NAME + ".tmp")).exists()
    meta = json.loads((out / "meta.json").read_text())
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == []
    assert sum(c["ok"] is True for c in meta["checks"]) == 7                   # rows, monthly, daily, rf, two universes, read back
    ref = loop_table(world)
    t = pq.read_table(out / wt.TABLE_NAME)
    assert t.schema.equals(wt.SCHEMA) and t.num_rows == len(ref) == meta["n_rows"]
    for c in CHARACTERISTICS:                                                   # empty is null in the file
        assert t.column(c).null_count == int(ref[c].isna().sum()), c
    assert all(t.column(c).null_count == 0 for c in ("date", "permno", "permco", "in_broad", "in_common", "rf"))
    df = pd.read_parquet(out / wt.TABLE_NAME)
    assert list(df.columns) == list(wt.COLUMNS) and df["permno"].dtype == np.int32 and df["in_broad"].dtype == bool
    np.testing.assert_array_equal(df["OA"].to_numpy(), ref["OA"].to_numpy(dtype=float))                  # NaN where empty
    assert set(meta["empty_share"]) == {"all", "in_broad", "in_common"} and meta["empty_share"]["all"]["Beta"] > 0.1   # 10010 has none
    assert meta["n_permnos"] == len(PERMNOS) and sum(y["rows"] for y in meta["rows_by_year"].values()) == len(ref)
    assert meta["n_permcos"] == len(PERMNOS) - 1                                # 10002 and 10005 are one company


def _ffill_by_security(tab, name):
    tab[name] = pd.Series(tab[name]).groupby(tab["permno"]).ffill().to_numpy()


MUTANTS = {
    "rf of day t instead of t-1": lambda tab, src: tab.update(rf=src["rf"][tab["_cal"]]),
    "Ret_D1 of day t (look-ahead)": lambda tab, src: tab.update(Ret_D1=src["ret"][tab["_cal"], np.searchsorted(src["permnos"], tab["permno"])]),
    "a monthly value carried forward": lambda tab, src: _ffill_by_security(tab, "OA"),
    "next month's accounting value": lambda tab, src: tab.update(
        OA=src["monthly"]["OA"][np.minimum(src["month_of_day"][tab["_cal"]] + 1, len(src["monthly"]["OA"]) - 1),
                               np.searchsorted(src["permnos"], tab["permno"])]),
    "a member flag dropped": lambda tab, src: tab["in_common"].__setitem__(np.flatnonzero(tab["in_common"])[-1], False),
    "a row dropped": lambda tab, src: tab.update({k: v[1:] for k, v in tab.items()}),
}


@pytest.mark.parametrize("name", MUTANTS)
def test_the_checks_catch_a_wrong_table(tmp_path, monkeypatch, name):
    _, src, arrays = setup_world(tmp_path, monkeypatch)
    right = wt.rows

    def wrong(*args):
        tab = right(*args)
        MUTANTS[name](tab, src)
        return tab

    monkeypatch.setattr(wt, "rows", wrong)
    assert not wt.build(check_only=True, src=src, arrays=arrays)
    assert wt.build(check_only=True, src=src, arrays=arrays) is False          # and it says so every time


# ---------------------------------------------------------------------------------------------------- the table on disk
TABLE = wt.TABLE_DIR / wt.TABLE_NAME
needs_table = pytest.mark.skipif(not (TABLE.exists() and (wt.TABLE_DIR / "meta.json").exists()), reason="run `python -m afsa.wrds_table` first")


@needs_table
def test_written_table_and_its_meta():
    meta = json.loads((wt.TABLE_DIR / "meta.json").read_text())
    assert meta["builder"] == "afsa.wrds_table" and meta["fill"] == "none"
    assert [c["check"] for c in meta["checks"] if c["ok"] is False] == []
    passed = " | ".join(c["check"] for c in meta["checks"] if c["ok"] is True)
    for topic in ("one row per security and trading day", "35 monthly values", "4 daily values", "rf is the rate of the previous trading day",
                  "rows with in_broad", "rows with in_common", "read back"):
        assert topic in passed, topic
    f = pq.ParquetFile(TABLE)
    assert f.schema_arrow.equals(wt.SCHEMA) and f.metadata.num_rows == meta["n_rows"]
    assert (meta["first_day"], meta["last_day"], meta["n_permnos"]) == ("1989-11-01", "2025-12-31", 2524)
    for u in WRDS_UNIVERSES:                                                    # its member rows are the arrays' stock-days
        assert sum(y[wt.FLAG[u]] for y in meta["rows_by_year"].values()) == json.loads(
            (TRAIN_DATA_DIR / data_folder(u, "none") / "meta.json").read_text())["n_stock_days"]


@needs_table
def test_table_on_disk_is_not_older_than_its_sources():
    meta = json.loads((wt.TABLE_DIR / "meta.json").read_text())
    stale = []
    for name, then in meta["sources"].items():
        p = bw.WRDS_DIR / name
        now = {"bytes": p.stat().st_size, "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} if p.exists() else None
        if now != then or then["modified"] > meta["built"]:
            stale.append(f"{name}: {then} at the build, {now} now")
    assert len(meta["sources"]) == 7 and not stale, "rerun `python -m afsa.wrds_table`: " + "; ".join(stale)


@needs_table
def test_one_day_of_the_table_on_disk_against_the_raw_files_and_the_arrays():
    """2001-09-17, the first day after the closure of September 2001: Ret_D1 is the return of 2001-09-10 for every
    row, the monthly values are those of September in the characteristic files, and ranking the broad rows gives the
    arrays' ranks. Recomputed here, not with the builder's code."""
    day, before = pd.Timestamp("2001-09-17"), pd.Timestamp("2001-09-10")
    df = pd.read_parquet(TABLE, filters=[("date", "==", day)])
    assert len(df) > 1000 and df["permno"].is_monotonic_increasing and (df["rf"] == df["rf"].iloc[0]).all()
    dsf = pd.read_parquet(bw.RAW_DIR / "crsp_dsf_v2.parquet", columns=["permno", "dlycaldt", "dlyret"], filters=[("dlycaldt", "in", [before, day])])
    assert np.array_equal(np.sort(dsf.loc[dsf["dlycaldt"] == day, "permno"].to_numpy()), df["permno"].to_numpy())   # a row each security trading that day
    prev = dsf[dsf["dlycaldt"] == before].set_index("permno")["dlyret"].reindex(df["permno"])
    np.testing.assert_array_equal(df["Ret_D1"].to_numpy(), prev.to_numpy(dtype=float))
    for path, c in ((bw.MARKET_FILE, "LME"), (bw.ACCOUNTING_FILE, "OA")):
        f = pd.read_parquet(path, columns=["permno", "month", c])
        f = f[pd.PeriodIndex(f["month"], freq="M") == pd.Period("2001-09", "M")].set_index("permno")[c]
        np.testing.assert_array_equal(df[c].to_numpy(), f.reindex(df["permno"]).to_numpy(dtype=float), err_msg=c)
    folder = TRAIN_DATA_DIR / data_folder("wrds_broad", "none")
    dates = np.load(folder / "dates.npy")
    t = int(np.searchsorted(dates, np.datetime64("2001-09-17")))
    permno, ranks = np.load(folder / "permno.npy", mmap_mode="r")[t], np.load(folder / "ranks.npy", mmap_mode="r")[t]
    b = df[df["in_broad"]]
    n = len(b)
    assert np.array_equal(permno[:n], b["permno"].to_numpy()) and (permno[n:] == -1).all()
    for c in ("Ret_D1", "LME", "OA", "FC2Y"):
        x = b[c].to_numpy()
        has = ~np.isnan(x)
        want = np.zeros(n)
        want[has] = (rankdata(x[has]) - 1) / (has.sum() - 1) - 0.5             # the missing-value rule: empty -> 0
        assert np.abs(ranks[:n, K[c]] - want).max() < 2e-6, c
