"""The training arrays as tensors on one device, and the day x stock batches the model consumes.

`Panel` holds the sample in the compact day x slot layout of Data/TrainingData. `Panel.batch` turns a block of
consecutive days into a dense day x stock block whose columns are the stocks that are in the universe on at least
one of those days, so a column is the same stock on every day and time-series operations are plain slices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from .config import TRAIN_DATA_DIR, WINDOW_RULES

ARRAYS = ("dates", "permnos", "permno", "slot_of", "ranks", "levels", "ret", "rf", "run")


def build_command(folder: str) -> str:
    """The command that writes Data/TrainingData/<folder>, for the message of a missing folder. A `wrds_*` folder
    comes from afsa.build_wrds, whose default fill is none (config.data_folder: no suffix = fill last)."""
    name = folder.replace("\\", "/").split("/")[0]                      # window folders are <rule>[_fill-x]/<n>y/<year>
    universe, _, fill = name.partition("_fill-")
    if universe.startswith("wrds_"):
        return f"python -m afsa.build_wrds --universe {universe}" + ("" if fill == "none" else f" --fill {fill or 'last'}")
    extra = f" --fill {fill}" if fill else ""
    if universe in WINDOW_RULES:
        return f"python -m afsa.build_window --rule {universe}{extra}"
    return f"python -m afsa.build_data --universe {universe}{extra}"


def membership(permno: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From the PERMNO of each slot ([T, N], -1 = empty) to the identifiers the model needs.

    permnos [S]: every PERMNO, sorted (stock id = position). slot_of [T, S]: slot of each stock, -1 if absent.
    run [T, N]: consecutive days the slot's stock has been in the universe, up to and including the day.
    """
    T, N = permno.shape
    filled = permno >= 0
    permnos = np.unique(permno[filled]).astype(np.int32)
    day, slot = np.nonzero(filled)
    sid = np.searchsorted(permnos, permno[filled])
    slot_of = np.full((T, len(permnos)), -1, dtype=np.int16)
    slot_of[day, sid] = slot
    present = slot_of >= 0
    spell = np.zeros(present.shape, dtype=np.int32)
    spell[0] = present[0]
    for t in range(1, T):
        spell[t] = np.where(present[t], spell[t - 1] + 1, 0)
    run = np.zeros((T, N), dtype=np.int16)
    run[day, slot] = np.minimum(spell[day, sid], np.iinfo(np.int16).max)
    return permnos, slot_of, run


@dataclass
class Batch:
    """A block of T consecutive days for each of M models; column u of model m is one stock throughout."""
    X: torch.Tensor        # [M, T, U, 39] characteristic ranks known before the day (arbitrary where not member)
    R: torch.Tensor        # [M, T, U]     excess return of the day; 0 where not member
    member: torch.Tensor   # [M, T, U]     in the universe that day
    valid: torch.Tensor    # [M, T, U]     member, and in the universe on each of the `lookback` days before
    levels: torch.Tensor   # [M, T, 40]    39 raw medians and rf of the previous day
    rf: torch.Tensor       # [M, T]        risk-free rate of the day (already subtracted from R)
    cols: torch.Tensor     # [M, U]        stock id of each column (index into Panel.permnos)
    n_cols: torch.Tensor   # [M]           real columns of each model; the remaining ones are padding


class Panel:
    """Sample of one universe on `device`. Day indices are positions in `dates` (the full 1990-2025 calendar)."""

    def __init__(self, universe: str = "broad", device: str | torch.device = "cuda",
                 first_year: int | None = None, last_year: int | None = None, lead_days: int = 64,
                 dtype: torch.dtype = torch.float32):
        folder = TRAIN_DATA_DIR / universe
        if not (folder / "meta.json").exists():
            raise FileNotFoundError(f"{folder} has no training arrays: run `{build_command(universe)}` first")
        self.meta = json.loads((folder / "meta.json").read_text())
        self.universe = universe
        dates = np.load(folder / "dates.npy")
        years = dates.astype("datetime64[Y]").astype(int) + 1970
        # Optionally keep only the days of [first_year, last_year] in memory (plus `lead_days` before, for the
        # residual lookback); day indices stay global.
        lo = 0 if first_year is None else max(0, int(np.searchsorted(years, first_year, "left")) - lead_days)
        hi = len(dates) if last_year is None else int(np.searchsorted(years, last_year, "right"))
        arrays = {"dates": dates, "permnos": np.load(folder / "permnos.npy")}
        for name in ARRAYS[2:]:
            arrays[name] = np.array(np.load(folder / f"{name}.npy", mmap_mode="r")[lo:hi])   # in-memory copy of the slice
        self._setup(arrays, device, dtype, lo)

    @classmethod
    def from_arrays(cls, dates: np.ndarray, permno: np.ndarray, ranks: np.ndarray, levels: np.ndarray, ret: np.ndarray,
                    rf: np.ndarray, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> "Panel":
        """A panel from arrays in memory (tests and small experiments); same layout as the files."""
        self = cls.__new__(cls)
        self.meta, self.universe = {}, "arrays"
        permnos, slot_of, run = membership(permno)
        arrays = {"dates": dates, "permnos": permnos, "permno": permno, "slot_of": slot_of, "ranks": ranks,
                  "levels": levels, "ret": ret, "rf": rf, "run": run}
        self._setup(arrays, device, dtype, 0)
        return self

    def _setup(self, a: dict[str, np.ndarray], device, dtype: torch.dtype, offset: int) -> None:
        self.device = torch.device(device)
        self.dates: np.ndarray = a["dates"].astype("datetime64[D]")              # [T] the full calendar
        self.years: np.ndarray = self.dates.astype("datetime64[Y]").astype(int) + 1970
        self.permnos: np.ndarray = a["permnos"]                                  # [S]
        self.n_days, self.n_stocks = len(self.dates), len(self.permnos)
        self.offset, self.loaded = offset, (offset, offset + len(a["permno"]))
        member = a["permno"] >= 0
        exret = np.where(member, a["ret"].astype(np.float64) - a["rf"][:, None], 0.0)
        self.permno: np.ndarray = a["permno"]                                    # [T, N] -1 = empty slot
        self.market: np.ndarray = exret.sum(axis=1) / member.sum(axis=1)         # equal-weighted universe, excess return
        self.ranks = torch.as_tensor(a["ranks"], dtype=dtype, device=self.device)        # [T, N, 39]
        self.levels = torch.as_tensor(a["levels"], dtype=dtype, device=self.device)      # [T, 40]
        self.exret = torch.as_tensor(exret, dtype=dtype, device=self.device)             # [T, N]
        self.rf = torch.as_tensor(a["rf"], dtype=dtype, device=self.device)              # [T]
        self.run = torch.as_tensor(a["run"], device=self.device)                         # [T, N] int16
        self.slot_of = torch.as_tensor(a["slot_of"], device=self.device)                 # [T, S] int16
        self.n_slots = self.ranks.shape[1]

    # ------------------------------------------------------------------ calendar
    def year_span(self, first_year: int, last_year: int) -> tuple[int, int]:
        """Day indices [start, stop) of the calendar years first_year..last_year."""
        return int(np.searchsorted(self.years, first_year, "left")), int(np.searchsorted(self.years, last_year, "right"))

    def has_days(self, start: int, stop: int) -> bool:
        return self.loaded[0] <= start and stop <= self.loaded[1]

    # ------------------------------------------------------------------ batches
    def batch(self, days: torch.Tensor, lookback: int) -> Batch:
        """days: [M, T] day indices, consecutive along T."""
        d = days.to(self.device) - self.offset
        M, T = d.shape
        slot_of = self.slot_of[d]                                               # [M, T, S]
        present = (slot_of >= 0).any(dim=1)                                     # [M, S] in the universe on some day
        n_cols = present.sum(dim=1)
        U = int(n_cols.max())
        # Present stocks first, in stock-id order: a model's columns do not depend on the other models of the batch.
        cols = torch.argsort(present.to(torch.int8), dim=1, descending=True, stable=True)[:, :U]
        slot = torch.gather(slot_of, 2, cols[:, None, :].expand(M, T, U)).long()   # [M, T, U]; -1 = not member
        member = slot >= 0
        slot = slot.clamp_min(0)
        day = d[:, :, None]
        return Batch(
            X=self.ranks[day, slot],
            R=self.exret[day, slot] * member,
            member=member,
            valid=member & (self.run[day, slot] > lookback),
            levels=self.levels[d],
            rf=self.rf[d],
            cols=cols,
            n_cols=n_cols,
        )
