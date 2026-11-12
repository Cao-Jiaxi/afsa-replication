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

# Universe name -> (X file, membership flag in crsp_daily)
UNIVERSES = {
    "broad": ("X_amplio.parquet", "in_universe"),          # main: top 500 of all NYSE/AMEX/NASDAQ equities
    "common": ("X_comun.parquet", "in_universe_comun"),    # robustness: US common stocks only
}

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
N_SLOTS = 500                          # at most 500 stocks in the universe on any day
TRADING_DAYS = 252


def _f(default, help):  # dataclass field with a help string for the CLI
    return field(default=default, metadata={"help": help})


@dataclass(frozen=True)
class Config:
    # ---- data and protocol -------------------------------------------------------------------
    universe: str = _f("broad", "universe: broad (main) or common (US common stocks only)")
    train_years: int = _f(8, "length of the rolling training window, in calendar years")
    lookback: int = _f(30, "number of lagged daily residuals fed to the sequence model")
    # ---- attention factors -------------------------------------------------------------------
    K: int = _f(8, "number of attention factors")
    d_x: int = _f(32, "dimension of the characteristic embedding and of the queries")
    ridge: float = _f(1e-5, "ridge penalty in the factor loadings")
    q_init_std: float = _f(1.0, "standard deviation of the initial query matrix Q")
    use_levels: bool = _f(True, "feed the 39 medians and rf (inert under eq. 1: the softmax over stocks cancels them)")
    drop_groups: tuple[str, ...] = _f((), "characteristic groups removed from the input (Table 3): " + ", ".join(CHAR_GROUPS))
    # ---- LongConv sequence model --------------------------------------------------------------
    d_model: int = _f(32, "hidden dimension = number of convolutions")
    n_layers: int = _f(1, "number of LongConv blocks")
    dropout: float = _f(0.1, "dropout inside each LongConv block")
    squash: float = _f(1e-3, "strength of the Squash (soft-threshold) operator on the kernel")
    kernel_init_std: float = _f(1.0, "scale of the random draw in the geometric-decay kernel initialisation")
    resid_scale: float = _f(100.0, "the residuals enter the sequence model multiplied by this (100 = per cent)")
    # ---- objective ----------------------------------------------------------------------------
    lambda_var: float = _f(100.0, "weight of the explained-variance term")
    sharpe_days: float = _f(252.0, "the Sharpe term is scaled by sqrt(sharpe_days): 252 = annualised, 1 = daily")
    cost_trade: float = _f(5e-4, "cost per unit of turnover")
    cost_short: float = _f(1e-4, "cost per unit of short exposure per day")
    ev_min_days: int = _f(21, "a stock enters the explained-variance term if it is in the universe on at least this many days of the batch")
    # ---- optimisation -------------------------------------------------------------------------
    epochs: int = _f(30, "passes over the training window")
    batch_days: int = _f(126, "consecutive trading days per batch")
    lr: float = _f(3e-3, "AdamW learning rate")
    weight_decay: float = _f(0.05, "AdamW weight decay, applied to the sequence model only")
    solve_dtype: str = _f("float64", "precision of the K x K ridge solve: float64 or float32")

    def __post_init__(self):
        object.__setattr__(self, "drop_groups", tuple(self.drop_groups))
        unknown = set(self.drop_groups) - set(CHAR_GROUPS)
        if unknown:
            raise ValueError(f"unknown characteristic groups {sorted(unknown)}; choose from {list(CHAR_GROUPS)}")
        if self.universe not in UNIVERSES:
            raise ValueError(f"unknown universe {self.universe!r}; choose from {list(UNIVERSES)}")
        if self.solve_dtype not in ("float64", "float32"):
            raise ValueError("solve_dtype must be float64 or float32")
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
        dropped = {c for g in self.drop_groups for c in CHAR_GROUPS[g]}
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
            else:
                parts.append(f"{f.name}-{v}")
        return "_".join(parts)

    # ---- (de)serialisation ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["drop_groups"] = list(self.drop_groups)
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
            elif isinstance(f.default, bool):
                group.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None, metavar="BOOL", help=help_)
            else:
                group.add_argument(f"--{f.name}", type=type(f.default), default=None, help=help_)

    @classmethod
    def from_args(cls, args: argparse.Namespace, **overrides) -> "Config":
        values = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls) if getattr(args, f.name, None) is not None}
        values.update(overrides)
        return cls(**values)
