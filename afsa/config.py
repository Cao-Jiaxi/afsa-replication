"""Paths, characteristic names and every tunable parameter of the reproduction.

Defaults are the paper's Table 4 plus the choices the paper leaves open (CLAUDE.md, decision log).
Every field of `Config` is a command-line option of `afsa.run`, e.g. `--K 30 --epochs 10`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Ander" / "data"            # Ander's processed parquet files (read-only input)
TRAIN_DATA_DIR = ROOT / "Data" / "TrainingData"  # model-ready arrays, written by afsa.build_data
RESULTS_DIR = ROOT / "results"
WRDS_DIR = ROOT / "Data" / "WRDS"                # our own WRDS downloads (raw\) and what we build from them

# Universe name -> (X file, membership flag in crsp_daily)
UNIVERSES = {
    "broad": ("X_amplio.parquet", "in_universe"),          # main: top 500 of all NYSE/AMEX/NASDAQ equities
    "common": ("X_comun.parquet", "in_universe_comun"),    # robustness: US common stocks only
}
# Universes fixed within each training window, written by afsa.build_window (one folder per test year):
# "window" needs returns on every day of the training years and of the test year, "window_train" of the training years.
WINDOW_RULES = ("window", "window_train")
# The same monthly top-500 lists with every input rebuilt from our own WRDS downloads, written by afsa.build_wrds
# (the arrays start on 1989-11-01, so the loss days of the first window start on 1990-01-02).
WRDS_UNIVERSES = ("wrds_broad", "wrds_common")
# Buffered monthly lists from the same downloads (afsa.wrds_universe buffer): each month's top 500 plus the securities of
# last month's list that still rank within `keep`, so a list holds 500 to `keep` securities. name -> (the list it
# extends, keep). Written by afsa.build_wrds and afsa.build_history with `keep` slots; trained like the wrds universes.
BUFFERED_UNIVERSES = {"wrds_broad_buffer600": ("broad", 600)}
WRDS_BUILDS = WRDS_UNIVERSES + tuple(BUFFERED_UNIVERSES)          # every universe the WRDS builders write
# Missing characteristics before ranking: "last" carries the stock's last observed value forward (the paper's section
# 4.1, Ander's X); "none" leaves them missing. Either way a stock still without a value gets rank 0 (the median) and
# the others are ranked among themselves.
FILLS = ("last", "none")
# Which members of a day's universe get a LongConv signal (CLAUDE.md section 7). "run" and "member" read the day x slot
# arrays, where each lag is the residual of that day's own universe; "history" (decided 2026-11-14) reads the month
# blocks of afsa.build_history, where the lags of day t are rebuilt over day t's universe (the month's list).
ENTRY_RULES = ("run", "member", "history")


def data_folder(universe: str, fill: str = "last") -> str:
    """Folder under TRAIN_DATA_DIR of a universe's arrays: `broad` holds Ander's X as is, `broad_fill-none` the rebuild,
    `wrds_broad_fill-none` the arrays built from our WRDS downloads under the missing-value policy."""
    return universe if fill == "last" else f"{universe}_fill-{fill}"

# The 39 characteristics of Table 1, in the column order of X. Table 3 drops one group at a time.
CHAR_GROUPS: dict[str, tuple[str, ...]] = {
    "past_returns": ("r2_1", "r12_2", "r12_7", "r36_13", "ST_Rev", "Ret_D1", "Ret_W1", "STD_W1"),
    "trading_frictions": ("AT", "LME", "LTurnover", "Rel2High", "Resid_Var", "Spread", "SUV", "Variance", "Vol", "Beta"),
    "value": ("A2ME", "BEME", "C", "CF", "CF2P", "Q", "Lev", "E2P"),
    "profitability": ("PROF", "CTO", "FC2Y", "OP", "PM", "RNA", "D2A"),
    "investment": ("Investment", "NOA", "DPI2A"),
    "intangibles": ("OA", "OL", "PCM"),
}
CHARACTERISTICS: tuple[str, ...] = tuple(c for g in CHAR_GROUPS.values() for c in g)
N_CHAR = len(CHARACTERISTICS)          # 39 cross-sectional ranks, one value per stock
N_LEVEL = N_CHAR + 1                   # 39 cross-sectional medians + rf, one value per day
LEVEL_NAMES: tuple[str, ...] = tuple(f"med_{c}" for c in CHARACTERISTICS) + ("rf",)
# Sets of characteristics a day's weights can be made orthogonal to, always with the constant (Config.neutral;
# analysis/gap/EXTENSION.md): the saved books carry standing positions along these ranks, not only residual reversal.
_RISK = ("Beta", "Variance", "Resid_Var", "STD_W1", "LME", "LTurnover")
NEUTRAL_SETS: dict[str, tuple[str, ...]] = {
    "dollar": (),                                                          # the constant only
    "risk": _RISK,
    "riskmom": _RISK + ("r2_1", "r12_2", "r12_7", "r36_13", "Rel2High"),
    "all": CHARACTERISTICS,
}
N_SLOTS = 500                          # at most 500 stocks in the universe on any day (a buffered list: n_slots)
TRADING_DAYS = 252


def n_slots(universe: str) -> int:
    """Slots of a universe's arrays: N_SLOTS, or `keep` for a buffered list (the largest size it can reach)."""
    return BUFFERED_UNIVERSES[universe][1] if universe in BUFFERED_UNIVERSES else N_SLOTS


def _f(default, help):  # dataclass field with a help string for the CLI
    return field(default=default, metadata={"help": help})


@dataclass(frozen=True)
class Config:
    # ---- data and protocol -------------------------------------------------------------------
    universe: str = _f("broad", "universe: broad (monthly top 500, main), common (US common stocks only), window / window_train (500 fixed per training window), wrds_broad / wrds_common (rebuilt from WRDS by afsa.build_wrds), or wrds_broad_buffer600 (each month's top 500 plus last month's members down to rank 600)")
    train_years: int = _f(8, "length of the rolling training window, in calendar years")
    lookback: int = _f(30, "number of lagged daily residuals fed to the sequence model")
    entry_rule: str = _f("run", "which members get a signal: run (in the universe on each of the `lookback` days before), member (every member; lags outside the universe are zero residuals) or history (lags rebuilt over the day's universe from the month blocks of afsa.build_history; a signal once listed for more than `lookback` trading days with annual accounts entered)")
    # ---- attention factors -------------------------------------------------------------------
    K: int = _f(8, "number of attention factors")
    d_x: int = _f(32, "dimension of the characteristic embedding and of the queries")
    ridge: float = _f(1e-5, "ridge penalty in the factor loadings")
    q_init_std: float = _f(1.0, "standard deviation of the initial query matrix Q")
    w_init_scale: float = _f(1.0, "multiplies the initial embedding matrix W^K (1 = the default of nn.Linear)")
    use_levels: bool = _f(True, "feed the 39 medians and rf (inert under eq. 1: the softmax over stocks cancels them)")
    fill: str = _f("last", "missing characteristics before ranking: last (the stock's last observed value, paper section 4.1) or none (left at rank 0)")
    drop_groups: tuple[str, ...] = _f((), "characteristic groups removed from the input (Table 3): " + ", ".join(CHAR_GROUPS))
    drop_chars: tuple[str, ...] = _f((), "single characteristics removed from the input, by their name in Table 1")
    # ---- LongConv sequence model --------------------------------------------------------------
    d_model: int = _f(32, "hidden dimension = number of convolutions")
    n_layers: int = _f(1, "number of LongConv blocks")
    dropout: float = _f(0.1, "dropout inside each LongConv block")
    squash: float = _f(1e-3, "strength of the Squash (soft-threshold) operator on the kernel")
    kernel_init_std: float = _f(1.0, "scale of the random draw in the geometric-decay kernel initialization")
    resid_scale: float = _f(100.0, "the residuals enter the sequence model multiplied by this (100 = per cent)")
    cumulative: bool = _f(False, "feed the running sum of the lagged residuals (a residual price path) instead of the residuals")
    # ---- objective ----------------------------------------------------------------------------
    lambda_var: float = _f(100.0, "weight of the explained-variance term")
    sharpe_days: float = _f(252.0, "the Sharpe term is scaled by sqrt(sharpe_days): 252 = annualized, 1 = daily")
    rf_hurdle: bool = _f(False, "Sharpe term on raw returns minus the risk-free rate, as the paper prints it (default: stock-by-stock excess returns)")
    cost_trade: float = _f(5e-4, "cost per unit of turnover in the training objective (results are always scored at the paper's 5 bp: afsa.metrics)")
    cost_short: float = _f(1e-4, "cost per unit of short exposure per day in the training objective (scored at the paper's 1 bp)")
    ev_min_days: int = _f(21, "a stock enters the explained-variance term if it is in the universe on at least this many days of the batch")
    # ---- trading layer (CLAUDE.md section 7; off at the defaults: the book is the target) ----------
    band: float = _f(0.0, "no-trade band inside the model: each day the book skips a stock whose gap to the target is at most `band` and trades the rest of the gap less `band`; return and costs are the book's (0 = no layer)")
    band_eta: float = _f(1.0, "share of the remaining gap the book trades each day (1 = all of it)")
    # ---- learned trading rule (analysis/rl/TRADING_RULE.md; off at the default) -------------------------
    trade_head: bool = _f(False, "learned trading rule: the sequence model's hidden state also gives each stock a band width and a speed for the trading layer, every day; return and costs are the book's (analysis/rl/TRADING_RULE.md)")
    head_band0: float = _f(4e-4, "learned trading rule: the band width where its parameters are zero, and that of members without a signal")
    head_band_max: float = _f(3.2e-3, "learned trading rule: the largest band width")
    head_eta0: float = _f(0.99, "learned trading rule: the speed where its parameters are zero, and that of members without a signal")
    trade_state: bool = _f(False, "learned trading rule (needs trade_head): each day the band width and speed also see the stock's position, its target and whether the trade adds to the position or reduces it, so they are set inside the book's loop (analysis/rl/TRADING_STATE.md)")
    trade_market: bool = _f(False, "learned trading rule (needs trade_head): the band width and speed also see the day's average hidden state over the stocks with a signal, the same for every stock of the day (analysis/rl/TRADING_STATE.md)")
    # ---- neutrality (analysis/gap/EXTENSION.md; off at the defaults) ------------------------------
    signal_center: bool = _f(False, "subtract from the day's signals their mean over the stocks that have one, so the sequence model cannot hold the same position in every stock")
    neutral: str = _f("", "make the day's weights orthogonal to the constant and to a set of the day's characteristic ranks before they are scaled to unit gross exposure: " + ", ".join(NEUTRAL_SETS) + " (empty = off)")
    nosignal_hedge: bool = _f(True, "a member without a signal holds the position the factor hedge gives it (true, as built); false: it holds none, and the weights of the others are scaled to unit gross exposure")
    # ---- optimization -------------------------------------------------------------------------
    epochs: int = _f(30, "passes over the training window")
    batch_days: int = _f(126, "consecutive trading days per batch")
    lr: float = _f(3e-3, "AdamW learning rate")
    lr_schedule: str = _f("constant", "learning-rate schedule over the whole training: constant, or cosine (decays to zero)")
    weight_decay: float = _f(0.05, "AdamW weight decay, applied to the sequence model only")
    solve_dtype: str = _f("float64", "precision of the K x K ridge solve: float64 or float32")

    def __post_init__(self):
        object.__setattr__(self, "drop_groups", tuple(self.drop_groups))
        object.__setattr__(self, "drop_chars", tuple(self.drop_chars))
        unknown = set(self.drop_groups) - set(CHAR_GROUPS)
        if unknown:
            raise ValueError(f"unknown characteristic groups {sorted(unknown)}; choose from {list(CHAR_GROUPS)}")
        unknown = set(self.drop_chars) - set(CHARACTERISTICS)
        if unknown:
            raise ValueError(f"unknown characteristics {sorted(unknown)}; choose from {list(CHARACTERISTICS)}")
        if self.universe not in UNIVERSES and self.universe not in WINDOW_RULES and self.universe not in WRDS_BUILDS:
            raise ValueError(f"unknown universe {self.universe!r}; choose from {list(UNIVERSES) + list(WINDOW_RULES) + list(WRDS_BUILDS)}")
        if self.fill not in FILLS:
            raise ValueError(f"unknown fill rule {self.fill!r}; choose from {list(FILLS)}")
        if self.entry_rule not in ENTRY_RULES:
            raise ValueError(f"entry_rule must be one of {list(ENTRY_RULES)}")
        if self.entry_rule == "history" and self.universe not in WRDS_BUILDS:
            raise ValueError(f"entry_rule history reads the month blocks of afsa.build_history, which exist for {list(WRDS_BUILDS)} only")
        if self.solve_dtype not in ("float64", "float32"):
            raise ValueError("solve_dtype must be float64 or float32")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine")
        if self.band < 0 or not 0 < self.band_eta <= 1:
            raise ValueError("band must be at least 0 and band_eta in (0, 1]")
        if not 0 < self.head_band0 < self.head_band_max or not 0 < self.head_eta0 < 1:
            raise ValueError("the learned trading rule needs 0 < head_band0 < head_band_max and 0 < head_eta0 < 1")
        if self.trade_head and (self.band > 0 or self.band_eta < 1):
            raise ValueError("trade_head learns each stock's band and speed: leave band at 0 and band_eta at 1")
        if (self.trade_state or self.trade_market) and not self.trade_head:
            raise ValueError("trade_state and trade_market extend the learned trading rule: set trade_head true")
        if self.neutral and self.neutral not in NEUTRAL_SETS:
            raise ValueError(f"neutral must be empty or one of {list(NEUTRAL_SETS)}")
        if self.neutral and not self.nosignal_hedge:
            raise ValueError("neutral with nosignal_hedge false is not neutral: the positions of members without a signal "
                             "are dropped after the projection")
        if min(self.K, self.d_x, self.d_model, self.n_layers, self.lookback, self.batch_days, self.epochs) < 1:
            raise ValueError("K, d_x, d_model, n_layers, lookback, batch_days and epochs must be at least 1")

    # ---- derived ------------------------------------------------------------------------------
    @property
    def steps_per_epoch(self) -> int:
        """Batches per epoch: the nominal window length divided by the batch length (16 by default)."""
        return max(1, round(self.train_years * TRADING_DAYS / self.batch_days))

    @property
    def char_mask(self) -> list[bool]:
        """True for the characteristics kept in the input."""
        dropped = {c for g in self.drop_groups for c in CHAR_GROUPS[g]} | set(self.drop_chars)
        return [c not in dropped for c in CHARACTERISTICS]

    def name(self) -> str:
        """Folder name of a run: every non-default field appears in it."""
        parts = [f"attn_K{self.K:03d}"]
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "K" or v == f.default:
                continue
            if f.name == "drop_groups":
                parts.append("drop-" + "+".join(v))
            elif f.name == "drop_chars":
                parts.append("dropchar-" + "+".join(v))
            else:
                parts.append(f"{f.name}-{v}")
        return "_".join(parts)

    # ---- (de)serialization ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["drop_groups"], d["drop_chars"] = list(self.drop_groups), list(self.drop_chars)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- command line ---------------------------------------------------------------------------
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser, skip: tuple[str, ...] = ()) -> None:
        """One `--field` option per tunable parameter."""
        group = parser.add_argument_group("model and training parameters (defaults: paper Table 4)")
        for f in dataclasses.fields(cls):
            if f.name in skip:
                continue
            help_ = f"{f.metadata['help']} [default: {f.default}]"
            if f.name == "drop_groups":
                group.add_argument("--drop_groups", "--drop-groups", nargs="*", default=None, choices=list(CHAR_GROUPS), help=help_)
            elif f.name == "drop_chars":
                group.add_argument("--drop_chars", "--drop-chars", nargs="*", default=None, choices=list(CHARACTERISTICS), metavar="NAME", help=help_)
            elif isinstance(f.default, bool):
                group.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None, metavar="BOOL", help=help_)
            else:
                group.add_argument(f"--{f.name}", type=type(f.default), default=None, help=help_)

    @classmethod
    def from_args(cls, args: argparse.Namespace, **overrides) -> "Config":
        values = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls) if getattr(args, f.name, None) is not None}
        values.update(overrides)
        return cls(**values)
