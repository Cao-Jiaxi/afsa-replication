"""The monthly top-500 universes built from our WRDS downloads (skipped until `afsa.wrds_universe build` has run)."""
import pandas as pd
import pytest

from afsa.config import SOURCE_DIR
from afsa.wrds_universe import N_STOCKS, UNIVERSE_DIR

KINDS = {"broad": "universe.parquet", "common": "universe_comun.parquet"}
pytestmark = pytest.mark.skipif(not (UNIVERSE_DIR / "broad.parquet").exists(),
                                reason="run `python -m afsa.wrds_universe download` and `build` first")


@pytest.mark.parametrize("kind", KINDS)
def test_shape_and_timing(kind):
    u = pd.read_parquet(UNIVERSE_DIR / f"{kind}.parquet")
    assert (u.groupby("month").size() == N_STOCKS).all()
    assert not u.duplicated(["month", "permno"]).any()
    # formed at the end of the previous month: no information from the holding month
    assert (u.formation_date.dt.to_period("M") + 1 == u.month).all()
    assert (u.groupby("month")["rank"].agg(["min", "max"]) == [1, N_STOCKS]).all().all()
    ordered = u.sort_values(["month", "rank"])
    assert (ordered.groupby("month").me.diff().dropna() <= 0).all()


@pytest.mark.parametrize("kind", KINDS)
def test_equals_anders_universe(kind):
    path = SOURCE_DIR / KINDS[kind]
    if not path.exists():
        pytest.skip("Ander's files are not here")
    ours = pd.read_parquet(UNIVERSE_DIR / f"{kind}.parquet")
    ander = pd.read_parquet(path)
    assert ours.month.min() == pd.Period("1989-11", "M")             # the lookback months, which Ander's file lacks
    ours = ours[ours.month >= ander.month.min()]
    m = ours.merge(ander[["month", "permno", "me", "rank"]], on=["month", "permno"], how="outer",
                   suffixes=("", "_a"), indicator=True)
    assert (m._merge == "both").all(), f"{(m._merge != 'both').sum()} stock-months differ"
    assert (m["rank"] == m.rank_a).all()
    assert ((m.me - m.me_a).abs() <= 1e-6 * m.me_a).all()
