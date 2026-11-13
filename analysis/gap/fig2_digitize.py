"""Reads the paper's Figure 2 back from the PDF: the cumulative out-of-sample return of each model.

    python analysis/gap/fig2_digitize.py            # writes analysis/gap/paper_fig2.csv and prints a summary

The figure is a vector plot (matplotlib), so its line paths hold the plotted values. This script inflates the PDF's
streams with the standard library, finds the plot, calibrates the axes from the tick marks and writes one column
per curve. Findings:

- x is an integer index: the points sit on a grid of 5,880 positions between the "1998" and "2022" ticks, which
  are 25 evenly spaced labels. The paper's out-of-sample series therefore has 5,880 days (24 x 245), against the
  6,040 trading days of 1998-2021, and a tick interval is 245 of its days, not exactly a calendar year.
- y is the growth of one unit: prod(1 + r), before costs (K = 30 ends at 47.8 = exp(5880/252 x (0.1666 - 0.042^2/2))).
- matplotlib drops points that do not change the drawing, so about half the days are stored. The stored points
  are exact; the others are interpolated here, so daily volatility cannot be recovered, but growth over a block can.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "3768292.3770398.pdf"
OUT = Path(__file__).with_name("paper_fig2.csv")
LABELS = ["K=1", "K=3", "K=5", "K=8", "K=15", "K=30", "K=100", "OU K=8", "OU K=30"]      # drawing order = legend order
N_DAYS, N_BLOCKS = 5880, 24


def figure_stream(pdf: Path) -> str:
    """The content stream of the plot: the one with the most line-to operators that also labels a 'Cumulative' axis."""
    data = pdf.read_bytes()
    best = b""
    for m in re.finditer(rb"stream\r?\n", data):
        raw = data[m.end():data.find(b"endstream", m.end())]
        try:
            dec = zlib.decompressobj().decompress(raw)
        except zlib.error:
            continue
        if b"(Cumulative R)" in dec and dec.count(b" l\n") > best.count(b" l\n"):
            best = dec
    if not best:
        raise RuntimeError("Figure 2 not found in the PDF")
    return best.decode("latin-1")


def parse(stream: str) -> tuple[list[np.ndarray], dict[str, float], dict[str, float]]:
    """Curves (thick stroked paths, in drawing order) and the tick positions of both axes, in figure points."""
    lines = stream.split("\n")
    curves, xticks, yticks = [], {}, {}
    width, path, tick = None, [], None
    for i, line in enumerate(lines):
        t = line.split()
        if " w" in line:                                           # line width: 2.8 = a curve, 0.8 = a tick mark
            m = re.search(r"([\d.]+) w(?:\s|$)", line)
            if m:
                width = float(m.group(1))
        if len(t) == 3 and t[2] in ("m", "l"):
            if t[2] == "m":
                path = []
            path.append((float(t[0]), float(t[1])))
        elif t == ["S"] and width == 2.8 and len(path) > 100:
            curves.append(np.array(path))
        elif t == ["B"] and width == 0.8 and len(path) == 2:       # a tick mark: remember where it starts
            tick = path[0]
        elif tick is not None and line.startswith("[ (") and line.endswith("TJ"):
            label = "".join(re.findall(r"\((.*?)\)", line))
            if re.fullmatch(r"(19|20)\d\d", label):
                xticks[label] = tick[0]
            elif re.fullmatch(r"\d+", label):
                yticks[label] = tick[1]
            tick = None
    return curves, xticks, yticks


def digitize() -> pd.DataFrame:
    curves, xticks, yticks = parse(figure_stream(PDF))
    assert len(curves) == len(LABELS), f"expected {len(LABELS)} curves, found {len(curves)}"
    x0, x1 = xticks["1998"], xticks["2022"]
    y0, y1 = yticks["0"], yticks["50"]
    step = (x1 - x0) / (N_DAYS - 1)
    grid = np.arange(N_DAYS)
    out, stored = {}, {}
    for label, c in zip(LABELS, curves):
        idx = (c[:, 0] - x0) / step
        assert np.abs(idx - idx.round()).max() < 0.02, "the points are not on the assumed grid"
        assert idx.round().min() == 0 and idx.round().max() == N_DAYS - 1
        value = (c[:, 1] - y0) / (y1 - y0) * 50
        out[label] = np.interp(grid, idx.round(), value)
        stored[label] = len(c)
    df = pd.DataFrame(out, index=pd.Index(grid, name="day"))
    df.attrs["stored"] = stored
    df.attrs["xticks"] = xticks
    return df


def block_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Log growth of each curve over each of the 24 tick intervals, annualized (% a year: 252 x mean log return)."""
    edges = np.linspace(0, N_DAYS - 1, N_BLOCKS + 1)
    rows = {}
    for b in range(N_BLOCKS):
        lo, hi = edges[b], edges[b + 1]
        start = np.ones(df.shape[1]) if b == 0 else np.array([np.interp(lo, df.index, df[c]) for c in df])
        end = np.array([np.interp(hi, df.index, df[c]) for c in df])
        days = hi - lo + (1 if b == 0 else 0)
        rows[1998 + b] = np.log(end / start) / days * 252 * 100
    return pd.DataFrame(rows, index=df.columns).T


def main() -> None:
    df = digitize()
    ticks = np.array([df.attrs["xticks"][str(y)] for y in range(1998, 2023)])
    print("x ticks evenly spaced:", np.allclose(np.diff(ticks), np.diff(ticks).mean(), atol=1e-4),
          f"({np.diff(ticks).mean():.4f} points = {(N_DAYS - 1) / N_BLOCKS:.2f} days each)")
    print("points stored per curve:", df.attrs["stored"])
    print("first value:", df.iloc[0].round(4).to_dict())
    print("last value: ", df.iloc[-1].round(2).to_dict())
    print("mean log growth, % a year (252 days):", (np.log(df.iloc[-1]) / N_DAYS * 252 * 100).round(2).to_dict())
    df.round(5).to_csv(OUT)
    print(f"written: {OUT}")
    pd.set_option("display.width", 200)
    print("\nAnnualized log growth by tick interval (%):")
    print(block_growth(df).round(1).to_string())


if __name__ == "__main__":
    main()
