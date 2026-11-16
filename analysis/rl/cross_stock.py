"""Cross-stock attention on the signal, against D (CLAUDE.md section 7, rows of 2026-11-16; CROSS_STOCK.md).

    python analysis/rl/cross_stock.py                    -> analysis/rl/CROSS_STOCK_RESULTS.md (CPU, plus a minute of GPU
                                                            for the size of the message; --no-message skips it)
    python analysis/rl/cross_stock.py --validation-only  -> the pre-registered decision only (1996-1997)

Dataset D (buffered list, rebuilt history), K = 30, 5 bp, ridge 1e-4, 30 epochs, seeds 0-2, with `--cross_heads 2`
(heads of 8) against D's stored runs. Objects: the model (one path per seed, statistics averaged) and the seed ensemble,
each also with the band added after training (delta = 4e-4, movers-only). Every path starts from cash on its period's
first day and carries its book across years, at the paper's costs.

The pre-registered decision, on 1996-1997 only: the model against D's model, and the ensemble + band against D's
ensemble + band. The test years are reported and never decide. The message's size is measured by running each saved
test-year model on its year: the norm of the message against the hidden state it is added to, and the share of the
signal's cross-sectional variance that the message carries.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "analysis" / "gap", HERE):
    sys.path.insert(0, str(p))
import band_trained as bt  # noqa: E402
from deploy import band_path, ensemble, market, seed_weights, trade  # noqa: E402
from trading_rule import DELTA, interval  # noqa: E402

from afsa.metrics import stitch  # noqa: E402

CROSS = {"cross_heads": 2, "cross_head_dim": 8}
ERAS = (("1998–2005", (1998, 2005)), ("2006–13", (2006, 2013)), ("2014–21", (2014, 2021)), ("2022–25", (2022, 2025)))


def period(first: int, last: int, train_years: int) -> dict[str, list[pd.DataFrame]]:
    base = dataclasses.replace(bt.BASE, train_years=train_years)
    cross = dataclasses.replace(base, **CROSS)
    dates, permnos, R, member = market(base, first, last)
    seeds = sorted(bt.run_dir(base).glob("seed*"), key=lambda p: int(p.name[4:]))
    W_d = [seed_weights(d, dates, permnos) for d in seeds]
    W_x = [bt.saved(bt.run_dir(cross) / d.name, dates, permnos, "weights") for d in seeds]
    for d, W in zip(seeds, W_d):                                             # the targets reproduce the stored results
        bt.check(stitch(d), trade(W, R, dates), d)
    for d, W in zip(seeds, W_x):
        bt.check(stitch(bt.run_dir(cross) / d.name), trade(W, R, dates), bt.run_dir(cross) / d.name)

    def band(W):
        return band_path(W, member, DELTA, rescale="moved")
    return {"attention model": [trade(W, R, dates) for W in W_x],
            "attention model + band after": [trade(band(W), R, dates) for W in W_x],
            "attention ensemble": [trade(ensemble(W_x), R, dates)],
            "attention ensemble + band after": [trade(band(ensemble(W_x)), R, dates)],
            "D model": [trade(W, R, dates) for W in W_d],
            "D model + band after": [trade(band(W), R, dates) for W in W_d],
            "D ensemble": [trade(ensemble(W_d), R, dates)],
            "D ensemble + band after": [trade(band(ensemble(W_d)), R, dates)]}


def decision(val: dict) -> tuple[str, list[str]]:
    """The pre-registered verdict: the model against D's, the ensemble + band against D's (first test's rule)."""
    lines, signs = [], []
    for a, b in (("attention model", "D model"), ("attention ensemble + band after", "D ensemble + band after")):
        d, lo, hi = interval(bt.pooled(val[a]), bt.pooled(val[b]))
        signs.append(1 if lo > 0 else -1 if hi < 0 else 0)
        lines.append(f"{a} − {b}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    verdict = ("better" if max(signs) == 1 and min(signs) >= 0 else
               "worse" if min(signs) == -1 and max(signs) <= 0 else "inconclusive")
    return verdict, lines


@torch.no_grad()
def message_size() -> pd.DataFrame:
    """Each saved test-year model run on its own year (GPU): the mean, over the stocks with a signal, of |message| /
    |hidden state|, and the share of the day's cross-sectional variance of the signal that the message carries."""
    from afsa.config import Config
    from afsa.model import AFSAModel
    from afsa.panel import load_panel
    run = bt.run_dir(dataclasses.replace(bt.BASE, **CROSS))
    cfg = Config.load(run / "config.json")
    panel = load_panel(cfg, torch.device("cuda"))
    rows = []
    for seed_dir in sorted(run.glob("seed*")):
        for f in sorted(seed_dir.glob("[0-9][0-9][0-9][0-9].pt")):
            year = int(f.stem)
            model = AFSAModel.from_states(cfg, [torch.load(f)], device="cuda")
            seen = {}
            plain = AFSAModel.cross

            def cross(self, z, keys):
                out = plain(self, z, keys)
                seen["z"], seen["msg"], seen["keys"] = z, out - z, keys
                return out
            model.cross = types.MethodType(cross, model)
            s, e = panel.year_span(year, year)
            days = (s + torch.arange(-(cfg.lookback + 1), e - s, device=panel.device)).clamp_max(panel.loaded[1] - 1)[None]
            model(panel.batch(days, cfg.lookback))
            k = seen["keys"]
            ratio = (seen["msg"].norm(dim=-1) / seen["z"].norm(dim=-1).clamp_min(1e-12))[k].mean()
            sig_msg = torch.einsum("mtud,md->mtu", seen["msg"], model.head_w)
            sig = torch.einsum("mtud,md->mtu", seen["z"] + seen["msg"], model.head_w)

            def day_var(v):                                                  # cross-sectional variance on each day
                n = k.sum(-1).clamp_min(1)
                mean = (v * k).sum(-1) / n
                return (((v - mean[..., None]) ** 2) * k).sum(-1) / n
            share = (day_var(sig_msg) / day_var(sig).clamp_min(1e-30))[k.any(-1)].mean()
            rows.append({"seed": int(seed_dir.name[4:]), "year": year, "|message| / |z|": float(ratio),
                         "signal variance from the message": float(share)})
    t = pd.DataFrame(rows)
    out = {}
    for name, (a, b) in ERAS:
        sel = t[(t["year"] >= a) & (t["year"] <= b)]
        out[name] = sel[["|message| / |z|", "signal variance from the message"]].mean()
    return pd.DataFrame(out).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation-only", action="store_true", help="only the pre-registered decision, on 1996-1997")
    p.add_argument("--no-message", action="store_true", help="skip the size of the message (needs the GPU)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    val = period(*bt.VALIDATION, 6)
    verdict, decided = decision(val)
    print(f"Decision: {verdict}", *decided, sep="\n")
    if args.validation_only:
        return
    test = period(bt.TEST[0], bt.HOLDOUT[1], 8)
    levels = bt.table(test, val)
    pairs = [("attention model", "D model"), ("attention model + band after", "D model + band after"),
             ("attention ensemble", "D ensemble"), ("attention ensemble + band after", "D ensemble + band after")]
    c = bt.contrasts(test, val, pairs)
    msg = None if args.no_message else message_size()
    lines = [
        "# Cross-stock attention on the signal: results",
        "",
        "Written by `analysis/rl/cross_stock.py`, as pre-registered in CLAUDE.md §7 (rows of 2026-11-16; the layer is "
        "specified in [CROSS_STOCK.md](CROSS_STOCK.md)). Dataset D (buffered list, rebuilt history), K = 30, 5 bp, ridge "
        "1e-4, 30 epochs, seeds 0–2, with 2 attention heads of 8 after the sequence model, against D's stored runs. Test "
        "years 1998–2021 and 2022–2025 use 8-year windows; the validation years 1996–1997 use 6-year windows. Every path "
        "starts from cash on its period's first day and carries its book across years, scored at the paper's costs (5 bp "
        "per unit traded, 1 bp per unit short a day). *+ band after*: δ = 4e-4, movers-only.",
        "",
        "## Decision (pre-registered: 1996–1997 only)",
        "",
        f"**{verdict.capitalize()}.**",
        "",
        *[f"- {x}" for x in decided],
        "",
        "Pooled seed-mean net series for the models; one series for each ensemble. Paired circular block bootstrap "
        "(21-day blocks, 2,000 draws, the same days for both), 95% interval. Computed before any test-year model trained.",
        "",
        "## Levels",
        "",
        "Model rows are seed averages; ensemble rows are one portfolio. Columns are 1998–2021 unless named; μ_net and σ in "
        "% a year.",
        "",
        bt.md(levels),
        "## Contrasts",
        "",
        "Pooled seed-mean net series, paired circular block bootstrap (21-day blocks, 2,000 draws), 95% interval. Only the "
        "validation column of the rows in the decision decides; everything else is reported.",
        "",
        bt.md(c),
    ]
    if msg is not None:
        lines += [
            "## How much the other stocks matter",
            "",
            "Each saved test-year model run on its own year: the mean over stocks with a signal of the message's norm "
            "against the hidden state it is added to, and the share of each day's cross-sectional variance of the signal "
            "that the message alone carries (its read-out, w_s'm), averaged over the era's models.",
            "",
            bt.md(msg, "{:.3f}"),
        ]
    out = HERE / "CROSS_STOCK_RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(levels.round(2).to_string())
    print(c.to_string())
    if msg is not None:
        print(msg.round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
