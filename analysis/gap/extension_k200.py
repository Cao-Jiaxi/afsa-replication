"""Score batch 5 (K = 200 without no-signal positions, test years 2014-2025) with the harness's own functions,
without editing extension.py while it is being audited: the trial and its first year are added here."""

import pandas as pd

import extension as ex  # noqa: E402

pd.set_option("display.width", 320, "display.max_columns", 40, "display.max_rows", 500)
_first = ex.first_year
ex.first_year = lambda cfg: 2014 if cfg.K == 200 else _first(cfg)
ex.DEV = (2014, 2021)                                                         # the band width of every run here is chosen on 2014-2021
ex.PERIODS = {"2014-21": (2014, 2021), "2014-19": (2014, 2019), "holdout 2022-25": (2022, 2025), "2024-25": (2024, 2025)}
for ref in ("K100_nosig0", "nosig0"):
    ex.TRIALS["K200_nosig0"] = {"K": 200, "cost_trade": 7.5e-4, "ridge": 1e-4, "nosignal_hedge": False, "ref": ref}
    cache = {}
    table = []
    for n in (ref, "K200_nosig0"):
        cache[n] = ex.paths(n)
        mine = cache[n]
        r = []
        for obj, daily in mine.items():
            row = {"trial": n, "object": obj + (f" {daily.attrs['delta']:g}" if "delta" in daily.attrs else "")}
            for p, (a, b) in ex.PERIODS.items():
                x = daily[ex._mask(daily, a, b)]
                row[p] = daily.attrs["seed average"][p] if obj == "model" else ex.sr(x["net"])
                if n != ref:
                    base = cache[ref][obj]
                    days = x.index.intersection(base.index)
                    row[f"d {p}"] = ex.cell(ex.paired_bootstrap(daily.loc[days, "net"].to_numpy(), base.loc[days, "net"].to_numpy()))
            row["turnover 2014-21"] = float(daily.loc[ex._mask(daily, 2014, 2021), "turnover"].mean())
            r.append(row)
        table += r
    print(f"\nK200_nosig0 against {ref} (band width of each run: best on 2014-2021)")
    print(pd.DataFrame(table).set_index(["trial", "object"]).to_string(float_format=lambda x: f"{x:.2f}", na_rep=""))
print("\nnet SR by year")
print(pd.concat({n: ex.by_year(ex.paths(n)).loc[2014:].T for n in ("nosig0", "K100_nosig0", "K200_nosig0")}).round(2).to_string())
for n in ("K200_nosig0",):
    print(f"\n{n}: seed ensemble as tilt + neutral part (gross)\n{ex.split(n).round(2).to_string()}")
