Attention Factors for Statistical Arbitrage -- Final report
Jiaxi Cao, AMS 520: Machine Learning in Quantitative Finance, Fall 2026

main.tex              Beamer presentation in the style of Phase1/main.tex
                      (47 main + 12 backup slides; no speaker notes yet)
assets/sbu_logo.png   Original official SBU horizontal logo (same file as in Phase1)

Compile with pdfLaTeX, twice. In Overleaf's settings choose main.tex as the
main document.

The preamble of main.tex is the Phase I preamble, unchanged apart from the
subtitle and the PDF title. After it, the source line is redefined in the text
colour (this deck has no grey text). A frame from the Phase I deck can
therefore be pasted in as it is; copy any figure it uses from Phase1/assets/
into assets/. Slide 2 is a table of contents of the sections. Sections 2 to 6
open with a section title page, and each subsection with a title page that
repeats the section's name above its own.

The parts: the model as we built it; the data (with our WRDS access against
the paper's); training and its settings (what we tune and what the paper
leaves open, each explained in turn); the choice of the top
500 (stocks against companies, then the four datasets) and their comparison:
the net Sharpe ratio of the seed ensemble (K = 30), with 1998-2021 the
paper's sample and 2022-2025 the extension. The last part is the scan on
dataset D for the best fixed setting: each K at the setting chosen on
1996-1997, then the whole grid at K = 8 and 30. The backup slides include
the formulas of all 39 characteristics and the risk-free rate, as built.

The numbers come from the repository: analysis/gap/DATASETS.md,
analysis/gap/SHARE_CLASSES.md, analysis/gap/DATA_ACCESS.md,
analysis/gap/SCAN_BUFFER.md, analysis/gap/SCAN_BUFFER_GRID.md,
analysis/gap/PILOT_BUFFER.md, analysis/gap/RESULTS.md (batch size) and
NOTES.md. The formulas come from analysis/characteristics/market.md,
accounting.md and training_data.md.
