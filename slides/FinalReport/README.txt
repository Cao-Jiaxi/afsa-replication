Attention Factors for Statistical Arbitrage -- Final report
Jiaxi Cao, AMS 520: Machine Learning in Quantitative Finance, Fall 2026

main.tex              Beamer presentation in the style of Phase1/main.tex
                      (39 main + 5 backup slides; no speaker notes yet)
assets/sbu_logo.png   Original official SBU horizontal logo (same file as in Phase1)

Compile with pdfLaTeX, twice. In Overleaf's settings choose main.tex as the
main document.

The preamble of main.tex is the Phase I preamble, unchanged apart from the
subtitle and the PDF title; the talk's own roadmap macro follows it. A frame
from the Phase I deck can therefore be pasted in as it is; copy any figure it
uses from Phase1/assets/ into assets/.

Part 1 is the model as we built it. Part 2 is the data (with our WRDS access
against the paper's), the choice of the top
500 (stocks against companies, then the four datasets) and their comparison:
the net Sharpe ratio of the seed ensemble (K = 30), with 1998-2021 the
paper's sample and 2022-2025 the extension. The last part is the scan on
dataset D for the best fixed setting: each K at the setting chosen on
1996-1997, then the whole grid at K = 8 and 30.

The numbers come from the repository: analysis/gap/DATASETS.md,
analysis/gap/SHARE_CLASSES.md, analysis/gap/DATA_ACCESS.md,
analysis/gap/SCAN_BUFFER.md, analysis/gap/SCAN_BUFFER_GRID.md,
analysis/gap/PILOT_BUFFER.md and NOTES.md.
