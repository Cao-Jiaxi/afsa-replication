Attention Factors for Statistical Arbitrage -- Final report
Jiaxi Cao and Ander Ibarrondo, AMS 520: Machine Learning in Quantitative
Finance, Fall 2026

main.tex              Beamer presentation in the style of Phase1/main.tex
                      (59 main + 13 backup slides; no speaker notes yet)
assets/sbu_logo.png   Original official SBU horizontal logo (same file as in Phase1)
assets/fig2_K030.png  The paper's Figure 2 (K = 30) before and after costs,
                      beside ours by hindsight; drawn at slide size by
                      analysis/gap/fig2_ours.py --deck
assets/net_K030.png   Every curve after costs (the conclusions): the paper,
                      ours by walk-forward and by hindsight, and the band
                      added after or in the loss; drawn by the same command

Compile with pdfLaTeX, twice. In Overleaf's settings choose main.tex as the
main document.

The preamble of main.tex is the Phase I preamble, unchanged apart from the
subtitle, the PDF title and the authors. After it, the source line is redefined
in the text colour (this deck has no grey text). A frame from the Phase I deck
can therefore be pasted in as it is; copy any figure it uses from
Phase1/assets/ into assets/. Slide 2 is a table of contents of the sections.
Every section opens with a title page, and each subsection with a title page
that repeats the section's name above its own; every title page has a Contents
button that links back to slide 2.

The parts: the model as we built it, ending with the paper's Figure 2, which is
before costs, against our estimate of it after costs (Table 2's average cost
spread over the years in proportion to each year's return) and our own curve by
hindsight; the data (with our WRDS access against the paper's), ending with the
choice of the top 500 (stocks against companies, then the four datasets and
their comparison: the net Sharpe ratio of the seed ensemble, K = 30, with
1998-2021 the paper's sample and 2022-2025 the extension); training and its
settings (the three periods, then what we tune and what the paper leaves open,
each explained in turn, with the scan of lambda_Var). Then the results without
the no-trade band under three rules: validation, walk-forward (explained with a
diagram) and hindsight. Then the extensions of the paper's model: the no-trade
band added after training, as a fourth lever (a chart of the three rules with
and without it on 1998-2021, with 2022-2025 in a table below, and walk-forward
by scoring window: since 1998, the last 8 years, the last 4 years), and the no-
trade rule inside the training objective. The conclusions open with every curve
after costs, then the findings, with a critique of the paper's Figure 2, and the
next steps. Cross-references name the section they point to. The backup slides
include the formulas of all 39 characteristics and the risk-free rate, as built,
and every K with the band chosen on 1996-1997.

The numbers come from the repository: analysis/gap/DATASETS.md,
analysis/gap/SHARE_CLASSES.md, analysis/gap/DATA_ACCESS.md,
analysis/gap/SCAN_BUFFER.md, analysis/gap/SCAN_BUFFER_GRID.md,
analysis/gap/PILOT_BUFFER.md, analysis/gap/LAMBDA_VAR.md,
analysis/gap/LOOKBACK.md, analysis/gap/BAND_TRAINED.md,
analysis/gap/BAND_TRAINED_0.0012.md, analysis/gap/RESULTS.md (batch size),
analysis/gap/PAPER_BY_YEAR.md (the paper's curve after costs) and NOTES.md. The
formulas come from analysis/characteristics/market.md, accounting.md and
training_data.md.
