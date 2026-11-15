"""The training arrays as tensors on one device, and the day x stock batches the model consumes.

`Panel` holds the sample in the compact day x slot layout of Data/TrainingData. `Panel.batch` turns a block of
consecutive days into a dense day x stock block whose columns are the stocks that are in the universe on at least
one of those days, so a column is the same stock on every day and time-series operations are plain slices.

`BlockPanel` holds the month blocks of afsa.build_history (Data/TrainingData/history/), the sample of the
rebuilt-history entry rule (CLAUDE.md section 7): there the lags of a day are the rows before it in the block of its
own month, all ranked over the month's list. `BlockPanel.batch` returns the blocks that a run of days falls in and
where each day sits in them (`BlockBatch`). `load_panel` loads the one a configuration trains on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from .config import N_CHAR, TRAIN_DATA_DIR, WINDOW_RULES, Config, data_folder

ARRAYS = ("dates", "permnos", "permno", "slot_of", "ranks", "levels", "ret", "rf", "run")
BLOCK_ARRAYS = ("row", "n_rows", "permno", "member", "ranks", "levels", "ret", "listed", "accounts")   # one entry per month


def panel_folder(cfg: Config) -> str:
    """Folder under TRAIN_DATA_DIR of the sample a configuration trains on: the month blocks for entry_rule history,
    the day x slot arrays otherwise. Not for the universes fixed per window (one folder per test year: afsa.run)."""
    folder = data_folder(cfg.universe, cfg.fill)
    if cfg.entry_rule != "history":
        return folder
    return f"history/{folder}" + ("" if cfg.lookback == Config().lookback else f"_lookback-{cfg.lookback}")


def load_panel(cfg: Config, device: str | torch.device = "cuda", **kwargs) -> "Panel | BlockPanel":
    """The sample of `panel_folder(cfg)` on `device`; keyword arguments go to the panel (first_year, last_year)."""
    return (BlockPanel if cfg.entry_rule == "history" else Panel)(panel_folder(cfg), device, **kwargs)


def build_command(folder: str) -> str:
    """The command that writes Data/TrainingData/<folder>, for the message of a missing folder. A `wrds_*` folder
    comes from afsa.build_wrds, whose default fill is none (config.data_folder: no suffix = fill last); a `history/`
    folder from afsa.build_history."""
    parts = folder.replace("\\", "/").split("/")
    if parts[0] == "history":                                          # history/<universe>[_fill-x][_lookback-n]
        name, _, lookback = parts[1].partition("_lookback-")
        universe, _, fill = name.partition("_fill-")
        return (f"python -m afsa.build_history --universe {universe}" + ("" if fill == "none" else f" --fill {fill or 'last'}")
                + (f" --lookback {lookback}" if lookback else ""))
    name = parts[0]                                                    # window folders are <rule>[_fill-x]/<n>y/<year>
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

    def covers(self, first: int, stop: int, lookback: int) -> bool:
        """True if the weights of days first..stop-1 can be computed: they and their `lookback` days before are loaded."""
        return self.has_days(first - lookback, stop)

    @staticmethod
    def pass_bytes(cfg: Config, days: int) -> float:
        """Rough bytes of one model's evaluation pass over `days` consecutive days (afsa.train.evaluate)."""
        seq = 4 * cfg.d_model * (cfg.lookback if cfg.n_layers > 1 else 1)
        return (cfg.lookback + 1 + days) * 800 * (N_CHAR + 3 * cfg.K + seq) * 4

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


# ---------------------------------------------------------------------------------------------------- month blocks
@dataclass
class BlockBatch:
    """D+1 consecutive days for each of M models, read from the month blocks they fall in (entry_rule history).

    The model computes factors and residuals on every row of the NB blocks over the block's list (its S slots), takes
    each day's lags from the `lookback` rows before the day in its own block, and gathers the day's weights into one
    column per stock (`cols`), as in `Batch`. Day 0 is the day before the first loss day; its weights are only needed
    for the turnover of day 1. Padding rows of a block repeat its last row, and no day points to them; spare blocks
    (a model whose days touch fewer than NB blocks) are not read either."""
    X: torch.Tensor          # [M, NB, R, S, 39] ranks known before the row's day, over the block's list
    R: torch.Tensor          # [M, NB, R, S]     excess return of the row's day; 0 where not member
    member: torch.Tensor     # [M, NB, R, S]     the list security has a CRSP row that day
    levels: torch.Tensor     # [M, NB, R, 40]    39 raw medians and rf of the previous day
    signal: torch.Tensor     # [M, D+1, S]       may get a signal on the day: member, listed > lookback, accounts entered
    day_block: torch.Tensor  # [M, D+1]          block of each day, 0..NB-1
    day_row: torch.Tensor    # [M, D+1]          row of each day in its block (>= lookback)
    slot: torch.Tensor       # [M, D+1, U]       slot of column u in the day's block; -1 if not a member that day
    rf: torch.Tensor         # [M, D+1]          risk-free rate of the day (already subtracted from R)
    cols: torch.Tensor       # [M, U]            stock id of each column (index into BlockPanel.permnos)
    n_cols: torch.Tensor     # [M]               real columns of each model; the remaining ones are padding


class BlockPanel:
    """Month blocks of one universe on `device` (afsa.build_history, whose docstring defines the arrays).

    Day indices are positions in `dates`, the blocks' calendar, which starts `lookback` trading days before the first
    month; from the first month on, each trading day is an own row of exactly one block, that of its month. The
    calendar methods are those of Panel. `levels` holds the own rows' medians and rf input by day (from `offset`), for
    the training-window statistics; the model reads the blocks' own `block_levels`."""

    def __init__(self, folder: str, device: str | torch.device = "cuda", first_year: int | None = None,
                 last_year: int | None = None, dtype: torch.dtype = torch.float32):
        path = TRAIN_DATA_DIR / folder
        if not (path / "meta.json").exists():
            raise FileNotFoundError(f"{path} has no month blocks: run `{build_command(folder)}` first")
        meta = json.loads((path / "meta.json").read_text())
        months = np.load(path / "months.npy")
        years = months.astype("datetime64[Y]").astype(int) + 1970
        # Optionally keep only the months of [first_year, last_year] and the month before, which holds the day before
        # a window's first loss day; day indices stay global.
        lo = 0 if first_year is None else max(0, int(np.searchsorted(years, first_year, "left")) - 1)
        hi = len(months) if last_year is None else int(np.searchsorted(years, last_year, "right"))
        a = {name: np.array(np.load(path / f"{name}.npy", mmap_mode="r")[lo:hi]) for name in BLOCK_ARRAYS}  # in-memory copies
        a.update(dates=np.load(path / "dates.npy"), rf=np.load(path / "rf.npy"), months=months[lo:hi])
        self._setup(a, int(meta["lookback"]), device, dtype)
        self.meta, self.universe = meta, folder

    @classmethod
    def from_arrays(cls, lookback: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                    **arrays: np.ndarray) -> "BlockPanel":
        """Blocks from arrays in memory (tests), named and laid out as the files: dates, rf, months and BLOCK_ARRAYS."""
        self = cls.__new__(cls)
        self.meta, self.universe = {}, "blocks"
        self._setup({k: np.array(v) for k, v in arrays.items()}, lookback, device, dtype)
        return self

    def _setup(self, a: dict[str, np.ndarray], lookback: int, device, dtype: torch.dtype) -> None:
        self.device, self.lookback = torch.device(device), lookback
        self.dates: np.ndarray = a["dates"].astype("datetime64[D]")             # [C] the blocks' calendar
        self.years: np.ndarray = self.dates.astype("datetime64[Y]").astype(int) + 1970
        self.months: np.ndarray = a["months"]
        self.n_days = len(self.dates)
        row, n_rows = a["row"].astype(np.int64), a["n_rows"].astype(np.int64)
        nB, R = row.shape
        if (n_rows <= lookback).any() or (row[:, :lookback] < 0).any():
            raise ValueError(f"every block needs {lookback} lookback rows and at least one day of its own")
        own_block, own_row = np.full(self.n_days, -1, dtype=np.int64), np.full(self.n_days, -1, dtype=np.int64)
        for b in range(nB):
            r = np.arange(lookback, n_rows[b])
            if (own_block[row[b, r]] >= 0).any():
                raise ValueError("a day is an own row of two blocks")
            own_block[row[b, r]], own_row[row[b, r]] = b, r
        days = np.flatnonzero(own_block >= 0)
        if days[-1] - days[0] + 1 != len(days):
            raise ValueError("the blocks' months must be consecutive")
        # Padding rows repeat the block's last row, so that every row has members (a softmax over no stock is not a
        # number, and its gradient would reach the parameters even though no day reads the row).
        crow = row.copy()
        for b in np.flatnonzero(n_rows < R):
            crow[b, n_rows[b]:] = row[b, n_rows[b] - 1]
            for name in ("member", "ranks", "levels", "ret", "listed"):
                a[name][b, n_rows[b]:] = a[name][b, n_rows[b] - 1]
        member = a["member"]
        exret = np.where(member, a["ret"].astype(np.float64) - a["rf"][crow][:, :, None], 0.0)
        # stocks: every PERMNO on some list, sorted (stock id = position); stock_slot = its slot in each block
        permno = a["permno"]
        used = permno >= 0
        self.permno: np.ndarray = permno                                        # [nB, S] each month's list; -1 = empty slot
        self.permnos: np.ndarray = np.unique(permno[used]).astype(np.int32)       # [n_stocks]
        stock_slot = np.full((nB, len(self.permnos)), -1, dtype=np.int16)
        b_idx, s_idx = np.nonzero(used)
        stock_slot[b_idx, np.searchsorted(self.permnos, permno[used])] = s_idx
        self.n_stocks, self.n_blocks, self.n_rows, self.n_slots = len(self.permnos), nB, R, permno.shape[1]
        self.offset, self.loaded = int(days[0]), (int(days[0]), int(days[-1]) + 1)
        ob, orow = own_block[days], own_row[days]
        self.market: np.ndarray = np.full(self.n_days, np.nan)                  # equal-weighted universe, excess return
        self.market[days] = exret[ob, orow].sum(axis=1) / member[ob, orow].sum(axis=1)

        def tensor(x, dt=None):
            return torch.as_tensor(x, dtype=dt, device=self.device)
        self.ranks = tensor(a["ranks"], dtype)                     # [nB, R, S, 39]
        self.block_levels = tensor(a["levels"], dtype)             # [nB, R, 40]
        self.exret = tensor(exret, dtype)                          # [nB, R, S]
        self.member = tensor(member)                               # [nB, R, S] bool
        self.listed = tensor(a["listed"])                          # [nB, R, S] int16
        self.accounts = tensor(a["accounts"])                      # [nB, S] bool
        self.stock_slot = tensor(stock_slot)                       # [nB, n_stocks] int16
        self.own_block, self.own_row = tensor(own_block), tensor(own_row)        # [C]; -1 = no block in memory
        self.levels = tensor(a["levels"][ob, orow], dtype)         # [days, 40] own rows, by day from `offset`
        self.rf = tensor(a["rf"], dtype)                           # [C]

    # ------------------------------------------------------------------ calendar
    def year_span(self, first_year: int, last_year: int) -> tuple[int, int]:
        """Day indices [start, stop) of the calendar years first_year..last_year."""
        return int(np.searchsorted(self.years, first_year, "left")), int(np.searchsorted(self.years, last_year, "right"))

    def covers(self, first: int, stop: int, lookback: int) -> bool:
        """True if the weights of days first..stop-1 can be computed: each is an own row of a block in memory, whose
        rows before it hold its lags."""
        return lookback == self.lookback and self.loaded[0] <= first and stop <= self.loaded[1]

    def pass_bytes(self, cfg: Config, days: int) -> float:
        """Rough bytes of one model's evaluation pass over `days` consecutive days (afsa.train.evaluate)."""
        n_blocks = (days + 1) // 15 + 2                                          # a month has at least 15 trading days
        seq = 4 * cfg.d_model * (cfg.lookback if cfg.n_layers > 1 else 1)
        return (n_blocks * self.n_rows * (N_CHAR + 3 * cfg.K) + (days + 1) * (cfg.lookback + seq)) * self.n_slots * 4

    # ------------------------------------------------------------------ batches
    def batch(self, days: torch.Tensor, lookback: int) -> BlockBatch:
        """days: [M, lookback + D + 1] consecutive day indices, laid out as for Panel.batch. The first `lookback` are
        not read: the lags of each day are the rows before it in its own month's block."""
        if lookback != self.lookback:
            raise ValueError(f"the blocks were built for a lookback of {self.lookback}, not {lookback}")
        d = days[:, lookback:].to(self.device)
        blk, row = self.own_block[d], self.own_row[d]                           # [M, D+1]
        if bool((blk < 0).any()):
            raise ValueError("a day of the batch is not in the blocks in memory")
        M, D1 = d.shape
        first = blk.min(dim=1).values
        NB = int((blk.max(dim=1).values - first).max()) + 1
        blocks = (first[:, None] + torch.arange(NB, device=self.device)).clamp_max(self.n_blocks - 1)   # [M, NB]
        # Columns: the stocks that are members on at least one of the days, in stock-id order, as in Panel.batch.
        on_list = self.stock_slot[blk].long()                                   # [M, D+1, n_stocks] slot in the day's block
        traded = torch.gather(self.member[blk, row], 2, on_list.clamp_min(0)) & (on_list >= 0)
        slot_of = torch.where(traded, on_list, -1)
        present = traded.any(dim=1)                                             # [M, n_stocks]
        n_cols = present.sum(dim=1)
        U = int(n_cols.max())
        cols = torch.argsort(present.to(torch.int8), dim=1, descending=True, stable=True)[:, :U]
        return BlockBatch(
            X=self.ranks[blocks],
            R=self.exret[blocks],
            member=self.member[blocks],
            levels=self.block_levels[blocks],
            signal=self.member[blk, row] & (self.listed[blk, row] > lookback) & self.accounts[blk],
            day_block=blk - first[:, None],
            day_row=row,
            slot=torch.gather(slot_of, 2, cols[:, None, :].expand(M, D1, U)),
            rf=self.rf[d],
            cols=cols,
            n_cols=n_cols,
        )
