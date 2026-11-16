# Why 2022–2025 is worse, and what was tried

Started 2026-11-15 at Jesse's request: find why the extension years do far worse than 1998–2021, record the reasoning, try fixes in the training strategy or the model (each trial at most 20 minutes of training), and have independent agents challenge the reasoning. Scripts: [extension.py](extension.py) (the trials). Runs are on dataset D (buffered list, rebuilt history; `NOTES.md` §9) unless stated, seeds 0–2, scored at the paper's costs.

How to read this file:
- Each step gives the **claim**, the **evidence**, and **why I reasoned that way**: what the test separates, and what I would have seen if the claim were false.
- A **model** row's level is a seed average (each seed scored alone). Its Δ, and the model rows by year, use the pooled series (the seeds' daily returns averaged), so a Δ is not the difference of two level columns. A **layer** row trades the same models differently (seed ensemble, no-trade band): a deployment result, never a Table 2 row.
- Everything here looks at test years, so it is a diagnostic under CLAUDE.md §8. Nothing in it sets the specification.
- Four years are about 1,000 days: a Sharpe ratio measured on them has a standard error near 0.5. A difference counts only with a paired interval that excludes 0.

## Conclusions

Sections 1–16 below are the record in the order things happened, except where a section says when it was written (5.8, the second half of section 9, 14 and 16), with each prediction written before its result; section 15 is the closing audit. This section is where they end up.

- **The reference** is dataset D at K = 30, trained at 5 bp with ridge 1e-4 for 30 epochs, seeds 0–2. As a model (a seed average) it nets 1.31 on 1998–2021 and −0.34 on 2022–2025. That gap is what this file explains.
- **A member without a signal** is a list member listed for 30 trading days or fewer, or whose first annual accounts have not entered yet (CLAUDE.md §7). The hedge still gives it a position.

**Why 2022–2025 is worse**

1. **Against 1998–2021: about half because 1998–2002 is in it, and half a further fall after 2021.** The net edge broke in spring 2001: the model netted 28% a year until mid-April 2001, 6% from then to the end of 2002 (2002 alone has a net SR of 3.23), and 1.1% a year on 2003–2021. Those five years carry 83% of the reference model's 1998–2021 net return, and its net SR since 2003 is about zero (0.28 on 2003–2021, 0.10 on 2003–2019). 2022–2025 is lower again. With the windfall of point 4 set aside, the model nets 1.32 on 1998–2021, 0.32 on 2003–2021 and −0.82 on 2022–2025. Dropping 1998–2002 accounts for 1.0 of the gap of 2.1, and the fall after 2021 for the other 1.1. That fall clears the noise: 2022–2025 minus 2003–2021 is −1.19 [−2.21, −0.21] (pooled; independent block bootstraps of the two periods), and −1.22 [−2.26, −0.26] for `nosig0`, which holds no such positions. As measured, the split is 1.0 and 0.6, and the fall does not clear the noise (−0.66 [−1.71, +0.33]). The paper's own Figure 2 has no such break. Its gross return falls gradually, from 41% in 2001 to 14–19% a year in 2003–2006, at a gross SR of 7.5–11.6 (`PAPER_BY_YEAR.md`). Ours earns 10–12.5% a year before costs in 2003–2006, at a gross SR of 5.0–6.3, and pays 11–12% a year in costs. So the 2001 break is ours (model, data or universe), not one the paper's figure shows, and 2001–2006 is where we fall short of the paper. (Sections 1, 9, 13)
2. **Against the years just before it: the book had come to lean on exposures, and in 2022–2025 they paid little.** In 2014–2021 the book was 0.20 net long (0.05 in 2003–2013), it held larger slow characteristic exposures, and March 2020 to 2021 was a burst. Depending on what is counted, these earned 2.7–6.7% of the book's 8.4% a year before costs, and between −0.4% and +0.8% a year after 2021. The larger income came from holding more, not from a richer market. Per unit held, the exposures paid about what they paid in 2003–2013: the equal-weighted market returned 13.0% a year against 12.4%, and the tilt's income per unit of its size changed by +1.0 [−4.1, +5.8] points for the risk and momentum ranks and by −0.3 [−4.2, +3.7] for all 39. It is 2022–2025 that departs: the market returned 4.7% a year, and the tilt's income per unit fell by 7.5 [−13.7, −1.3] and 7.7 [−12.4, −2.9] points against 2003–2013 (for the net long position alone and for the six risk ranks the fall is inside the noise). Four years cannot tell whether 2022–2025 was an unlucky draw for these exposures or their end. Either way they are not arbitrage: net of beta × market the model nets −0.25 in 2014–2021. With the windfall of point 4 set aside, gross return fell 7.0 points a year from 2014–2021 to 2022–2025. The tilt is 4.6–6.2 points of that if the momentum ranks count as tilt, but 2.1–2.6 if only the constant and the risk ranks do, and then the rest of the book fell about twice as much. (5.7, 9)
3. **No break in the trading at 2022 that these years can detect.** Three measures, none of them precise:
   - The first-day payoff of a fresh trade has been 1–3 bp per unit traded, against a 5 bp cost, in every era since 2003 (5.8). The day's trade has a rank correlation with the day's returns (stocks with a signal) of 0.74% in 2022–2025, against 0.84% in 2016–2019 (−0.10 [−0.73, +0.51]).
   - At constant turnover, the gross return per unit traded has fallen about 0.2 bp a year since 2003, and 2022–2025 is on that trend (section 9). With the no-signal P&L set aside it is 0.9 bp instead of 1.5, against a forecast of 1.3 bp with an interval of about ±2 bp.
   - The whole book's rank correlation with the day's returns did fall after 2021: 0.32% in 2022–2025 against 1.54% in 2016–2019 (−1.22 [−2.00, −0.41]). Part of that is the tilt of point 2. The neutral part fell less (−0.71 [−1.38, −0.07]) and is on its 2003–2019 trend (0.37% realized, 0.32% forecast). The whole book is below its own trend by −0.66 [−1.56, +0.19].

   Parameter-free reversal on the same stocks has not cleared 5 bp since 2006. (5.8, 9)
4. **The measured 2022–2025 is itself flattered.** About 2% a year came from large hedge shorts in members without a signal that collapsed: Addentax and Figma as new listings, VinFast while it waited for annual accounts. This reading was added after batch 2 was scored (section 9). Without those positions the reference nets −0.80 as a model, −0.65 as an ensemble and −0.33 through the band, and 2022 and 2023 are near zero as well. (9, 13)

**What was tried.** Fifteen trials trained here (each 9 minutes or less), the nine K = 100 runs and three K = 30 runs of the scan (20 epochs, 7.5 and 10 bp), the two runs trained with the band inside (`BAND_TRAINED.md`), and five ideas tested on saved books without training. Net SR differences are against the reference, on 2003–2021 (`win16` from 2006):

| Idea | Trials | Model | Ensemble + band | Verdict |
| --- | --- | --- | --- | --- |
| The sample is stale: a shorter window | `win04` | −1.24 | −0.33 | A shorter window is far worse |
| The fit is noise: a longer window | `win12`, `win16` | +0.26, +0.20 | +0.19, +0.14 | `win12` helps the model, `win16` no clear effect; not the layer |
| Fit less | `drop03`, `wd05`, 20 epochs | +0.17, +0.01, +0.14 | −0.04, −0.05, −0.04 | `drop03` and 20 epochs help the model, `wd05` no effect; not the layer |
| A slower signal: price-path input | `cum` | +0.54 | −0.04 | Helps the model; not the layer |
| A higher training cost | 7.5 and 10 bp | +0.46, +0.54 | −0.03, −0.11 | Helps the model on both periods; not the layer |
| No standing tilt, inside the model | `center`, `neut_dollar`, `neut_risk`, `neut_riskmom` | −0.39 to −0.85 | −0.23 to −0.36 | Lower before 2022, not higher after |
| More factors: K = 100, the scan's block (5 / 7.5 / 10 bp × ridge 1e-4 / 3e-4 / 1e-3) | scan block | −0.59 to +0.57 | −0.16 to +0.13 | Level with the reference through the band before 2022 (5.5) |
| A tighter hedge: K = 100 at 7.5 bp, ridge 1e-4 / 3e-5 / 1e-5 | the scan's 1e-4 cell, batch 3 | −0.03, −0.36, −0.67 | −0.11, −0.27, −0.33 | Its gain after 2021 is the windfall |
| No position without a signal | `nosig0` | +0.10 | +0.08 | Removes the windfall; costs nothing before 2022 |
| More factors: K = 200, against K = 100, both without those positions (2014–2025 only) | batch 5 | −0.13 (2014–2021); −0.71 (2022–2025) | +0.08; −0.42 | Not supported: level on 2014–2021; lower after 2021, which section 2 does not let decide |
| Overlays: inverse-volatility sizing, exposure by turbulence, neutral books | 5.2, 5.3, 5.6 | | −0.57 to −0.01 | None helps |
| A gate by characteristic | 5.8 | | | Not traded: no fifth's first-day payoff clears the 5 bp cost in more than two of five eras |
| Overlay: averaging models across K | 5.4 | | +0.20 | Helps before 2022, not after |
| A multi-day objective: the band trained inside the model, against the same band added after (on disk, `BAND_TRAINED.md`) | δ 4e-4, 1.2e-3 | −0.08, −0.29 | −0.03, −0.19 | Lower on 2022–2025 with the windfall set aside (model −0.25, −0.67; ensemble + band −0.24, −0.55); not higher before |

**What raises the net SR on 2022–2025**

| | 2003–2021 | 2022–2025 as measured | 2022–2025, windfall set aside |
| --- | --- | --- | --- |
| Reference model (trained at 5 bp) | 0.28 | −0.34 | −0.82 |
| Model trained at 10 bp | 0.78 | +0.02 | −0.49 |
| Reference, ensemble + band | 1.32 | +0.13 | −0.37 |
| Trained at 10 bp, ensemble + band | 1.21 | +0.30 | −0.23 |
| K = 100 (7.5 bp, ridge 1e-4), ensemble + band | 1.21 | +0.79 | −0.01 |

1. **Every lever that cuts trading raises it.** These are gains on 2022–2025:
   - a higher training cost: +0.30 to +0.38 for the model
   - the seed ensemble: about +0.15
   - the band: about +0.3 on top of the ensemble

   They are substitutes: all three together add 0.59–0.65, not the sum.
2. **Nothing raises it above zero once the windfall is set aside.** The best is K = 100 (7.5 bp, ridge 1e-4) with the layers, at −0.01 with the windfall's P&L set aside, and −0.15 when trained without those positions (section 13). It was carried into batches 3–5 for its 2022–2025 result and is the highest of the 29 runs on those same years, so −0.01 and its lead are upper estimates: the eleven K = 100 runs with no-signal positions have a median of −0.15. Its lead over the K = 30 reference is 0.36 (0.18 with no position without a signal in either run), but K = 100 trains at 7.5 bp and the reference at 5 bp. At equal training cost the lead is +0.26 (5 bp), +0.24 (7.5 bp) and +0.01 (10 bp), all inside the noise. On 2003–2021 it is level to 0.2 lower.
3. **No change tried here added edge that survives the band, as far as three seeds can tell.** The changes that help the model (a higher training cost, the price-path input, dropout, fewer epochs: +0.14 to +0.54 on 2003–2021, in every seed) do it by trading less. Each has a lower gross SR than the reference on 2003–2021 (1.98–2.59 against 2.69), and through the band each ends at the reference's net SR. The 12-year window is the one change with a higher gross SR (2.79); its layer is +0.19 [−0.02, +0.41]. The changes that reshape the book (neutrality, tighter hedge) give up income that was real before 2022 and get nothing back after. Through the band these trials could see differences of about 0.15–0.55 on 2003–2021. Batch 6 (λ_Var, the one lever on the attention's objective) is unscored.
4. **No change on the trading side can lift it much above zero, because the targets earn too little before costs.** With the windfall set aside, the reference's seed ensemble earns 1.4% a year before costs on 2022–2025 and its banded book 0.4%, while the short fee alone is 1.0–1.1% a year. With trading free and the short fee kept, the ensemble would net +0.10 (K = 30), +0.13 (`nosig0`) or +0.32 (K = 100 at 7.5 bp), and the banded book −0.18: that is the ceiling for any band, smoothing or training cost. A fix has to roughly double what the targets earn before costs. On 2014–2021 the same book broke even at 18 bp per unit traded [10, 25]; on 2022–2025 no positive cost breaks even (as measured, with the windfall, 8 bp). About fifty slower ways of trading the saved targets (partial adjustment, moving averages, each with bands), chosen on 2003–2021, move 2022–2025 by −0.05 to +0.08. (Section 15)

**What I would do.** These are recommendations; the decisions are Jesse's.

1. **Report the extension as a further fall after the break of 2001, with the by-year table.** What the record supports:
   - As a model, the paper's object, our implementation has netted about zero at the paper's 5 bp since 2003 (0.28 on 2003–2021, 0.10 on 2003–2019). 2022–2025 is lower still. As measured (−0.34) it is not detectably different: 2022–2025 minus 2003–2021 is −0.66 [−1.71, +0.33], a test that could not have seen a fall smaller than about 1.4. With the windfall set aside (−0.82) it is: −1.19 [−2.21, −0.21], and −1.22 [−2.26, −0.26] for `nosig0`, which holds no such positions.
   - 1998–2002 carries 83% of the model's 1998–2021 net return.
   - The ensemble + band did cover its costs on 2003–2021 (1.32), but nets about zero on 2022–2025 once the windfall is set aside.
2. **For a deployment row on 2022–2025, use the ensemble + band on the reference setting**, and report it both ways: +0.13 as measured, about −0.35 with the windfall set aside (below the short fee; point 4 above). A higher training cost (7.5–10 bp) and K = 100 (trained at 7.5 bp, so it includes that lever) are 0.03–0.22 lower on 2003–2021 and 0.15–0.36 higher on 2022–2025, all inside the noise, so section 2's rule supports neither for the layer. At equal training cost K = 100 leads by +0.01 to +0.26 on 2022–2025 with the windfall set aside. Using one is a choice to label as such.
3. **Decide whether a member without a signal should hold a hedge position** (`nosignal_hedge`). Dropping it changes §7's entry rule. For: it removes shorts 8 to 15 times the ordinary size in members without a signal (new listings, or members still waiting for their first annual accounts), and it is +0.10 on 2003–2021. Against: it lowers the measured 2022–2025 by 0.46, all of it luck.
4. **Do not adopt** neutrality inside the model, a shorter window, or a ridge below 1e-4.
5. **Use `nosig0` as the reference for any further trial on the extension**, so that no trial is credited with the windfall, and compare K at equal training cost.
6. **What is left to try is on the signal side** (section 15 ranks it): a 60-day look-back for the sequence model, a volume or liquidity input to it, and λ_Var (batch 6). None has a strong prior: the missed-fixes auditor puts each at 5–8% for a gain above 0.3.

**Limits.** Four years; a standard error near 0.5 on every 2022–2025 figure. The development years are test years, so nothing here sets the specification (CLAUDE.md §8). About thirty runs were judged, most at K = 30, all on dataset D with three seeds. Several of my own claims and predictions failed on the way: section 1's claim 2, and claim 1 in half; the first readings of 5.5 and 5.7; section 8's reading of the hedge ordering; batch 2's holdout prediction (section 10); the K = 100 against K = 30 prediction (section 13); and the expectation for K = 200 (section 14). That is a reason to hold the remaining ones loosely. The closing audit (section 15) found no computational error; what it changed is listed there. Batch 6 (section 16) was stopped before anything in it was scored.

## 1. First reading, before any agent or trial

D, K = 30, reference setting (5 bp, ridge 1e-4, 30 epochs). Net SR by year:

| | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | 6.07 | 5.46 | 9.32 | 1.94 | 3.23 | −0.09 | 0.33 | 0.04 | −0.40 | −1.49 | 1.17 | 1.17 | −0.71 | −0.78 |
| Ensemble + band 1.2e-3 | 6.68 | 5.22 | 8.71 | 3.00 | 4.50 | 2.46 | 2.95 | 2.48 | 1.79 | 0.78 | 0.89 | 3.34 | 0.95 | 0.41 |

| | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | 0.02 | 2.16 | 0.74 | 0.24 | −1.25 | 0.52 | −0.30 | 0.02 | 1.55 | 0.76 | 0.09 | 0.53 | −0.96 | −1.24 |
| Ensemble + band 1.2e-3 | 1.43 | 2.81 | 1.60 | 0.74 | −0.01 | 1.24 | 0.34 | 0.77 | 2.55 | 1.45 | 0.84 | 1.12 | −0.72 | −0.65 |

The model row is the pooled seed series; the band's width is the best on 2003–2021, so 2022–2025 did not choose it.

| D, K = 30, reference model | 1998–2005 | 2006–2013 | 2014–2021 | 2022–2025 |
| --- | --- | --- | --- | --- |
| Gross SR | 6.90 | 3.15 | 1.71 | 0.76 |
| Gross return, % a year | 25.3 | 10.9 | 7.8 | 3.3 |
| Cost, % a year | 12.2 | 10.0 | 6.2 | 4.8 |
| Net SR | 3.56 | 0.26 | 0.36 | −0.34 |
| Turnover a day | 0.87 | 0.70 | 0.41 | 0.30 |
| Gross return per unit traded, bp | 11.5 | 6.2 | 7.6 | 4.4 |
| Beta | 0.01 | 0.05 | 0.18 | 0.13 |
| In-sample net SR, last epoch | 13.3 | 5.8 | 2.6 | 2.8 |

Three claims, and why I read the table that way:

1. **For the model, 2022–2025 is not a new failure.** Its net SR is 0.28 on 2003–2021 and −0.34 on 2022–2025; the 1.31 of 1998–2021 comes from 1998–2002. *Why I read it so:* a break in 2022 would show as years near the 1998–2021 average until 2021 and a drop after. The years 2003–2021 instead scatter around zero (ten of nineteen at or below 0.05), which is what one regime since 2003 looks like. *What would overturn it:* a formal break-date search that prefers 2022, or a 2022–2025 minus 2003–2021 difference whose interval excludes 0. **Half overturned later, by the second test:** the break search prefers 2001 (section 9), but with the no-signal P&L set aside (section 9) the model's 2022–2025 minus 2003–2021 is −1.19 [−2.21, −0.21], and `nosig0`'s, which needs no adjustment, is −1.22 [−2.26, −0.26] (section 13). As measured it is −0.66 [−1.71, +0.33]. So the edge broke in 2001, and 2022–2025 is a further fall, not the same level.
2. **For the layer, the problem is 2024 and 2025.** The ensemble + band nets 1.32 on 2003–2021, with one year of nineteen at or below zero. It nets 0.84 and 1.12 in 2022 and 2023, then −0.72 and −0.65. *Why:* two years in a row far below every earlier year but one is not what the 2003–2021 scatter produces, so here I do suspect a change, in the last two years only. *What would overturn it:* model-free reversal portfolios that are as healthy in 2024–2025 as before (then the loss is ours, not the market's), or a loss concentrated in a handful of days (then it is an accident, not a regime). **Overturned later, by something I had not listed:** the layer's 2022 and 2023 were carried by a windfall (sections 9 and 13). Market exposure only lifted its 2003–2021: net of beta × market (beta fitted in each era), 1.32 becomes 1.16, and 2014–2021's 1.11 becomes 0.84. For the model it carries all of it: 0.30 becomes −0.01 on 2003–2021 (pooled seeds; section 9).
3. **The loss is in the return before costs, not in the costs.** Costs fell with turnover (12.2% → 4.8% a year), but the gross return fell faster (25.3% → 3.3%). In 2024 and 2025 the model earned 1.6% and 0.35% before costs, about 2 and 0.5 bp per unit traded, against a cost of 5 bp plus 1 bp a day on the short side (about 1% a year whatever the trading). *Why this matters for a fix:* a trading layer can only remove cost. Where the targets earn nothing before costs, no band rescues them; the fix has to raise what the targets earn.

A fourth observation I could not explain at this point: the two models that fail in 2024 and 2025 have the **highest in-sample fit since the 2015 model** (net SR 3.2 and 3.5, against 2.1–2.3 for the 2020–2023 models). Their windows (2016–2023 and 2017–2024) hold 2020, a year that paid 16.7% before costs out of sample. A model can fit such a year well and learn little that lasts. **Corrected later:** the windows of the 2021, 2022 and 2023 models hold 2020 too, and their fit is 2.1–2.3, so 2020 does not separate the two models. Section 9 (item 3) tries the standing tilt instead; the books do not support it.

## 2. The trial protocol, fixed before any trial ran

- **Reference:** D, K = 30, 5 bp, ridge 1e-4, 30 epochs, seeds 0–2. A trial changes one thing and trains every test year its window allows (84 models from 1998; about 4 minutes).
- **Development years: 2003–2021.** A trial is judged on them. **Holdout: 2022–2025**, reported once per trial, never used to choose or tune.
- **Objects:** model; seed ensemble; ensemble + band, the width being the best of the 9 on that run's own development years.
- **Kept:** development net SR above the reference with a paired 95% interval excluding 0, for the model or for the ensemble + band, not clearly below for the other, and a holdout difference that is not negative. Anything else is *not supported*, whatever its holdout says.
- A window other than 8 years keeps the reference's 480 optimizer steps, so the trial changes the sample and not the amount of fitting.
- *Added after the closing audit.* **Code.** The reference trained on 2026-11-14 (`afsa` hash 60009b0d in its `provenance.json`), as did 20 epochs, 7.5 bp, 10 bp and K = 8. Batch 1 and the K = 100 block trained on the code committed as 7a67f12 (3020a785), which holds the trading layer. Batches 2 and 3 trained on e612786e and batch 4 on b130ecfa, which add the three model flags. Every trial − reference difference assumes that the later code trains the reference setting as the earlier code did. Unit tests show that the layer and the flags are skipped at their defaults (`tests/test_book.py`, `tests/test_neutral.py`), but no reference model was retrained or replayed under the later code.

Why this protocol:
- **Why not judge on 2022–2025 directly?** With a standard error near 0.5, ten trials judged on four years would hand back the luckiest one. The goal is a higher net SR in 2022–2025 that would also hold in 2026.
- **Why 2003–2021 and not 1996–1997?** The validation years net 8 and reward fast trading that every later year punishes (`REPORT.md` §10.7). They are the wrong regime for this question. 2003–2021 is the regime the extension continues (claim 1), with 19 years.
- **Why keep the holdout at all?** Claim 2 says 2024–2025 may differ from 2003–2021. A fix that works on the development years and not on the holdout is evidence for that.
- **Cost of the choice:** the development years are test years. Whatever passes is a labelled diagnostic, not a specification (§8).
- **Multiple testing:** about a dozen trials at 95% give roughly one false pass in two batches. A pass by a small margin is weak; a finalist is retrained with new seeds.

## 3. What the runs already on disk say under this protocol

Scored with `extension.py report` before any new training. Net SR; differences are trial − reference, paired block bootstrap.

| Trial | Object | 2003–2021 | Δ 2003–2021 | 2022–2025 | Δ 2022–2025 | Turnover 2003–21 / 2022–25 |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | model | 0.28 | | −0.34 | | 0.60 / 0.30 |
| | ensemble | 0.61 | | −0.18 | | 0.56 / 0.26 |
| | ensemble + band 1.2e-3 | 1.32 | | +0.13 | | 0.27 / 0.07 |
| 20 epochs | model | 0.42 | +0.14 [+0.05, +0.23] | −0.27 | +0.08 [−0.13, +0.28] | 0.55 / 0.26 |
| | ensemble + band 1.2e-3 | 1.28 | −0.04 [−0.15, +0.06] | +0.03 | −0.11 [−0.39, +0.16] | 0.24 / 0.05 |
| Trained at 7.5 bp | model | 0.72 | +0.46 [+0.30, +0.63] | −0.06 | +0.30 [+0.09, +0.51] | 0.40 / 0.18 |
| | ensemble + band 1.2e-3 | 1.29 | −0.03 [−0.19, +0.14] | +0.28 | +0.15 [−0.10, +0.39] | 0.15 / 0.03 |
| Trained at 10 bp | model | 0.78 | +0.54 [+0.30, +0.80] | +0.02 | +0.38 [+0.09, +0.69] | 0.28 / 0.14 |
| | ensemble + band 8e-4 | 1.21 | −0.11 [−0.35, +0.16] | +0.30 | +0.17 [−0.14, +0.49] | 0.12 / 0.03 |
| K = 100 (7.5 bp, ridge 3e-4) | model | 0.53 | +0.29 [+0.09, +0.49] | −0.11 | +0.25 [−0.23, +0.71] | 0.48 / 0.23 |
| | ensemble + band 1.2e-3 | 1.29 | −0.03 [−0.33, +0.25] | +0.32 | +0.19 [−0.32, +0.69] | 0.20 / 0.04 |

1. **Every lever that makes the model trade less helps the model, on both periods.** A higher training cost is the clearest (+0.46 and +0.54 on the development years, +0.30 and +0.38 on the holdout, all four intervals above 0).
2. **None of them helps once the band is on.** Through the ensemble + band all five runs land at 1.21–1.32 on the development years. A higher training cost, fewer epochs, more factors and a band remove trades of no net value: with the band on, each still lowers turnover (0.27 a day on 2003–2021 for the reference, 0.12–0.24 with one of them added) but leaves the net SR where it was.
3. **Every run loses in 2024 and in 2025**, as model and as layer (2024: −0.04 to −0.96; 2025: −0.37 to −1.40). Trading less shrinks the loss and never turns it.

*Why this matters for what to try:* the levers already scanned all act on cost. They converge to the same layer result and leave 2024–2025 negative. A trial worth running has to change what the targets earn before costs: the sample the model learns from, how much noise it fits, or what the signal can see.

## 4. Batch 1: trials that need no new code

Written before they ran. Each has a reason, a prediction, and what its failure would mean.

| Trial | Change | Reason | Prediction | If it fails |
| --- | --- | --- | --- | --- |
| `win04` | 4-year window, 60 epochs | **Stale sample.** The window is 8 years while the edge shrinks every year, so the model is fitted to a richer market than it will trade (in-sample net SR 2–3.5, out-of-sample near 0). A short window sees only the recent market | Better if the cause is staleness: less turnover, higher net SR, most in 2024–2025, whose windows would drop 2020 by 2025 | Staleness is not the cause; a short window fits more noise |
| `win12`, `win16` | 12- and 16-year windows, 20 and 15 epochs | **Estimation noise.** The opposite reading of the same gap: with a weak signal, 8 years are too few and the fit is mostly noise. More years average it out | Better if the cause is noise: higher net SR on the development years, seeds agree more. `win16` starts in 2006 | Old years carry patterns that no longer pay, which outweighs the lower noise |
| `drop03` | dropout 0.3 (0.1) | **Noise fitting.** If the fit is mostly noise, a stronger regularizer should lose less. 20 epochs already gives the model +0.14 | Model up by 0.1–0.2 on the development years; little through the band | The in-sample excess is not what costs money out of sample |
| `wd05` | weight decay 0.5 (0.05) | The same, through the size of the sequence model's weights | As `drop03` | As `drop03` |
| `cum` | the sequence model sees the residual price path | **Horizon.** The reference reacts to yesterday's residual; a price path makes the signal depend on a dislocation accumulated over days, which is slower. If the edge that is left lives at a longer horizon, this input finds it. On Ander's arrays it gained +0.24 at K = 30, all in 2006–2013; validation rejected it, but validation rewards speed | Turnover down to about 0.45; model up on the development years; through the band unclear | The band already extracts the slow part of the fast signal |

The window trials are the pair I care most about: the two explanations of the in-sample gap predict opposite signs, so the pair decides between them whatever happens.

## 5. Pre-checks on the saved books, while batch 1 trained

No training: each idea is first tried as an overlay on the reference's saved daily weights. *Why pre-check:* an overlay takes a minute and a model change takes an hour to code and test, so an idea that shows nothing as an overlay has to earn its hour another way.

**5.1 Where the money is lost, by characteristic.** The reference ensemble's P&L by fifths of a stock's trailing 63-day volatility (net, % a year, paper's costs):

| | 2003–2013 | 2014–2021 | 2022–2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- |
| Most volatile fifth | +1.0 | −0.8 | +0.5 | −4.0 | −1.1 |
| The other four fifths | +1.2 | +3.5 | +1.3 | +1.35 | −3.1 |
| Gross return per unit traded, bp: volatile fifth / rest | 7.7 / 7.0 | 4.7 / 13.0 | 7.5 / 10.6 | −8.1 / 10.2 | 2.3 / −0.6 |

Multi-year columns are averages of the yearly figures.

1. **2024 and 2025 fail differently.** 2024's loss is all in the most volatile fifth (−4.0%), while the rest earns +1.35%. 2025's loss is broad: the other four fifths earn nothing before costs (−0.6 bp per unit traded, against 6–33 bp in 2019–2024).
2. **The book trades the volatile fifth hardest**: 25–32% of its trading until 2010 and 32–45% since, for 20% of the stocks. In 2014–2021 that fifth earns 4.7 bp per unit traded, just below the 5 bp it costs to trade; the rest earns 13.0 bp. The same holds by `Variance`, `Spread`, `STD_W1`, `LTurnover` and `Beta`, which are largely the same stocks.
3. *Why it could be so:* the sequence model reads raw residuals in per cent, so a volatile stock's ordinary day looks like a large dislocation and gets a large position that flips often. *Why I do not trust it yet:* since 2003 the volatile fifth beat the rest in 9 of 23 years by net P&L (2008, 2022 and 2025 among them; 2016 a tie) and in 10 by gross return per unit traded (2013 among them), and by year its P&L swings from −4.8% (2019) to +8.2% (2008). A pattern this noisy can come and go.

**Corrected later (sections 9 and 13):** the row of the other four fifths also holds the five or so members a day without 21 days of history, and with them 92% of the stock-days of members without a signal, whose hedge shorts were lucky after 2021. With the P&L of those positions set aside, that row nets −1.8% a year in 2022–2023 (+1.3 as measured) and earns 2.7 bp per unit traded (10.6). 2022 alone earns −4.5 bp, so point 1's 6–33 bp in 2019–2024 becomes −4.5 to 33 bp (2025: −1.4 bp set aside, −0.6 as measured). The most volatile fifth held some of those positions in 2024: set aside, its loss that year is −5.3% (−4.0 as measured). Point 1's contrast between 2024 and 2025 stands.

**5.2 Sizing by inverse volatility, as an overlay** (target ÷ trailing volatility^p, rescaled to unit gross exposure):

| | 2003–2021 | Δ | 2014–2021 | 2022–2025 | Δ | 2024–2025 |
| --- | --- | --- | --- | --- | --- | --- |
| Ensemble + band, as is | 1.32 | | 1.11 | +0.13 | | −0.68 |
| p = 0.5, 63 days | 1.30 | −0.01 [−0.18, +0.16] | 1.16 | +0.23 | +0.09 [−0.23, +0.42] | −0.41 |
| p = 1, 63 days | 1.23 | −0.09 [−0.28, +0.13] | 1.20 | +0.21 | +0.08 [−0.43, +0.61] | −0.23 |

Not supported: nothing on the development years (it loses in 2003–2013 what it gains in 2014–2021), and the holdout gain is inside the noise. An in-model version could still differ, because the network would re-fit around it.

**5.3 Exposure that follows market turbulence.** The book is always fully invested, so it trades even when the edge is below costs. I scaled its exposure by lagged cross-sectional dispersion relative to its trailing median (nine variants). Ensemble: +0.01 to +0.03 on the development years. Ensemble + band: −0.03 to −0.22, several intervals below 0. *Why it fails:* the return per day before costs does rise with turbulence (7.7 bp in the top state against 3–4.5 bp), but so does the book's volatility, so the Sharpe ratio per day before costs does not rise (4.4 / 2.7 / 2.9 / 2.1 / 3.0 from the calmest state to the top one, 2003–2021). After costs the ensemble's is highest in the top state (1.66 against −0.11 to 0.48), which is why it gains a little. The band book's is as high in the calmest state as in the top one (1.9 and 1.9, against 0.9–1.2 between), and cutting exposure in calm states gives that up. Letting exposure rise to 2 or 3 in turbulent states does not change this: ensemble +0.02 to +0.11 on 2003–2021 (no interval excludes 0), ensemble + band −0.14 to −0.41. **Dropped.**

**5.4 Averaging different models, not only seeds** (seed ensembles of several runs averaged into one book):

| Book, through its band | 2003–2021 | Δ | 2022–2025 | Δ | 2024–2025 |
| --- | --- | --- | --- | --- | --- |
| K = 30 reference | 1.32 | | +0.13 | | −0.68 |
| K = 8 + 30 + 100 | 1.52 | +0.20 [+0.04, +0.36] | +0.15 | +0.01 [−0.21, +0.22] | −0.74 |
| K = 30 + 100 | 1.41 | +0.10 [−0.09, +0.28] | +0.46 | +0.33 [+0.00, +0.63] | −0.52 |
| Ten models (three K, three training costs, three ridges) | 1.51 | +0.19 [+0.01, +0.37] | +0.14 | +0.01 [−0.27, +0.27] | −0.64 |

More averaging raises the development years (clearly for three K together) and does nothing for 2024–2025. K = 30 + 100 does gain on 2022–2025 (+0.33, the interval just clear of 0), most of it in 2022–2023 (+0.44 [−0.01, +0.90]). *What this tells me:* the last two years' loss is shared by every model and every combination, and each of the ten models loses in 2024 and in 2025 through its own band. It is not noise specific to one model, which averaging removes. It belongs to the kind of signal all these models learn.

**Corrected later (section 9):** most of K = 30 + 100's gain on 2022–2025 is the windfall, which grows with the hedge. With the P&L of positions in members without a signal set aside it is +0.12 [−0.12, +0.38].

**5.5 The K = 100 block under this protocol** (`SCAN_BUFFER_BLOCK.md`: training cost 5 / 7.5 / 10 bp × ridge 1e-4 / 3e-4 / 1e-3, 30 epochs). Ensemble + band, band chosen on 2003–2021; Δ against the K = 30 reference:

| Ensemble + band | 2003–2021 | Δ | 2022–2025 | Δ |
| --- | --- | --- | --- | --- |
| K = 8, 5 bp, ridge 1e-4 | 1.38 | +0.06 [−0.16, +0.29] | −0.40 | −0.53 [−0.91, −0.15] |
| K = 30, 5 bp, ridge 1e-4 (reference) | 1.32 | | +0.13 | |
| K = 100, ridge 1e-3: 5 / 7.5 / 10 bp | 1.45 / 1.32 / 1.26 | +0.13 / +0.01 / −0.06 | +0.01 / +0.06 / +0.13 | −0.12 / −0.07 / −0.00 |
| K = 100, ridge 3e-4: 5 / 7.5 / 10 bp | 1.33 / 1.29 / 1.34 | +0.01 / −0.03 / +0.03 | +0.24 / +0.32 / +0.41 | +0.11 / +0.19 / +0.27 |
| K = 100, ridge 1e-4: 5 / 7.5 / 10 bp | 1.16 / 1.21 / 1.15 | −0.16 / −0.11 / −0.16 | +0.72 / +0.79 / +0.58 | +0.58 [−0.01, +1.13] / +0.65 [+0.11, +1.13] / +0.45 [−0.20, +1.01] |

1. **On the development years the number of factors and the ridge do not matter through the band**: every K = 100 cell is within its interval of the reference.
2. **On the holdout the results line up by how tightly the factors are hedged, with one exception**: K = 8 worst (−0.53 [−0.91, −0.15]), then the nine K = 100 cells in order of the ridge. Every 1e-3 cell (+0.01 to +0.13 on 2022–2025) is below every 3e-4 cell (+0.24 to +0.41), and every 3e-4 cell is below every 1e-4 cell (+0.58 to +0.79). The exception is K = 30 (+0.13): it is level with K = 100 at ridge 1e-3 (Δ −0.12 to −0.00), not below it. Two differences are clear: K = 8, and K = 100 at 7.5 bp with ridge 1e-4 (+0.65 [+0.11, +1.13]). *Why I take the ordering seriously although most cells are inside the noise:* ten of the eleven runs fall in one order along one mechanical axis, and K and the ridge move that axis independently. *Why I do not take it as a fix yet:* the development years show nothing, so the rule of section 2 cannot choose K = 100 at ridge 1e-4; and no K = 100 model clearly earns its costs alone (−0.47 to +0.13; two of the nine are above 0, by 0.09 and 0.13).
3. As models, the K = 100 settings follow the first reading: a higher training cost helps on both periods (10 bp, ridge 3e-4: +0.49 [+0.14, +0.86] on the development years).

**Corrected later (sections 9, 11 and 13):** much of this ordering is a windfall that grows with the hedge. With it set aside the ridge no longer orders the K = 100 cells, and K = 100 leads K = 30 by 0.18–0.36, inside the noise.

**5.6 Is it factor exposure? Neutralizing the saved book.** Each day the target was projected off the day's characteristic ranks and rescaled:

| K = 30 reference, ensemble + band | 2003–2021 | Δ | 2022–2025 | Δ | 2024–2025 |
| --- | --- | --- | --- | --- | --- |
| As saved | 1.32 | | +0.13 | | −0.68 |
| Dollar-neutral | 0.94 | −0.38 [−0.72, −0.02] | −0.11 | −0.24 [−1.14, +0.58] | −1.21 |
| + six risk characteristics | 0.94 | −0.38 [−0.67, −0.08] | −0.02 | −0.15 [−1.03, +0.62] | −1.02 |
| + risk and momentum (11) | 0.74 | −0.57 [−0.91, −0.25] | +0.30 | +0.16 [−0.79, +1.00] | −0.73 |

| Gross return, % a year (projection on the constant and the eleven risk and momentum ranks) | 2003–2013 | 2014–2021 | 2022–2023 | 2024–2025 |
| --- | --- | --- | --- | --- |
| K = 30: the part the projection removes | +1.8 | +3.2 | −0.9 | −1.1 |
| K = 30: the neutral book | 10.5 | 5.2 | 6.7 | 2.1 |
| K = 100 (5 bp, ridge 1e-4): the part removed | +1.2 | +1.6 | −0.1 | −0.1 |
| K = 100: the neutral book | 10.2 | 6.5 | 8.3 | 2.7 |

1. **The book carries standing tilts, and they grew.** By 2022–2025 the K = 30 ensemble is short high-volatility and wide-spread stocks (exposure to the `Variance` rank −0.012 in 2003–2013, −0.055 in 2014–2021, −0.095 in 2022–2025), long cash-rich and high-Q stocks, short high book-to-market. In 2022–2025 K = 100 carries about half of the volatility and spread exposures (0.49–0.52 of K = 30's), 0.57 of the book-to-market one, 0.69 of the Q one and 0.85 of the cash one.
2. **The tilts earned 1.8–3.2% a year until 2021 and lost about 1% a year after.** That swing is the part of K = 100's holdout advantage I can account for: its tilts lost 0.1% a year. *Corrected after the audit (section 15):* I first wrote that this also explains 5.1: that the 2024 loss in the volatile fifth is the short side of this tilt, in a year when volatile stocks rallied. Neither half holds. In 2024 the fifth lost 2.3% a year before costs (longs +1.9%, shorts −4.1%). The exact split of 5.7 puts 0.4% of that in the tilt and 1.9% in the neutral part, whose short positions lost 3.9%. Under 5.7's other definitions the tilt's part is between −0.8% and +0.6%. With the positions in members without a signal set aside (section 9), the tilt's part is 0.5% of 3.6%. The tilt as a whole earned +0.4% in 2024 and lost in 2025 (−2.1%). And volatile stocks barely outran the rest in 2024: the fifth returned 10.8% a year against 9.0% for the other members (equal-weighted), and a unit long-short along the `Variance` rank lost 1.2%. They rallied in 2023 and 2025 (31.3% and 29.1% against 7.4% and 7.8%), and in those years the fifth earned +0.6% and +0.7% before costs.
3. **Removing the tilts is not a fix.** Every projection loses on the development years (intervals below 0), because the tilts paid there, and none helps the holdout beyond noise. Dollar-neutrality makes 2024–2025 worse (−1.21): the book is net long and the market rose.
4. **From 2003–2013 to 2024–2025 the larger part of the decline is in the neutral book itself:** 10.5% → 5.2% → 6.7% → 2.1% a year before costs over the four columns, against +1.8% → +3.2% → −0.9% → −1.1% for the part removed. That is the reversal edge proper, and no exposure accounting touches it. From 2014–2021 to 2022–2025 it is the other way round: the neutral book falls 0.8 points (5.2% to 4.4% a year) and the part removed 4.2 (3.2% to −1.0%).

**Corrected in 5.7:** the first row of the second table is the saved book minus the rescaled neutral book, so it is not the tilt alone. Split exactly, without rescaling, the tilt on the constant and the eleven risk and momentum ranks earned 2.4% a year in 2003–2013 and 4.0% in 2014–2021, then 0.0% in 2022–2023 and −0.8% in 2024–2025. That is about −0.4% a year after 2021, not the −1% of point 2. K = 100's tilt earned 0.2% and 0.0%.

*Where this leaves the reasoning:* two things happened. The pure residual-reversal edge kept shrinking, as it has since 2003; and from 2022 the slow exposures the model had come to rely on stopped paying. 5.7 measures both more carefully, and corrects how much weight each can bear.

**5.7 The exact split.** The table in 5.6 rescales the neutral book, which blurs the sizes. Without any rescaling, each day's saved weights are the sum of a *tilt* (their projection on the constant and the eleven risk and momentum ranks) and a *neutral part* (the rest). Gross return of each, % a year:

| K = 30 reference, seed ensemble | 1998–2002 | 2003–2013 | 2014–2021 | 2022–2023 | 2024–2025 |
| --- | --- | --- | --- | --- | --- |
| Tilt | 3.1 | 2.4 | 4.0 | 0.0 | −0.8 |
| of which the net long position | 0.0 | 0.7 | 2.7 | −0.3 | 1.7 |
| Neutral part | 32.8 | 9.9 | 4.4 | 5.8 | 1.9 |
| Gross exposure of the tilt (the book's is 1) | 0.24 | 0.35 | 0.52 | 0.54 | 0.43 |
| Gross exposure of the neutral part | 0.97 | 0.94 | 0.85 | 0.85 | 0.92 |
| K = 100 (5 bp, ridge 1e-4): tilt / neutral part | 1.9 / 30.5 | 1.5 / 9.9 | 1.9 / 6.2 | 0.2 / 8.0 | 0.0 / 2.6 |

**A correction I made after writing the first version of this section.** I first read the table above as "4.4 of the 5.0-point drop is the tilt". That number depends on which ranks define the tilt, so I recomputed the split for four definitions (K = 30 reference, gross % a year):

| The tilt is the projection on the constant and | Tilt: 2014–2021 | 2022–2023 | 2024–2025 | Neutral part: 2014–2021 | 2022–2023 | 2024–2025 |
| --- | --- | --- | --- | --- | --- | --- |
| nothing else (the net long position) | 2.7 | −0.3 | 1.7 | 5.8 | 6.2 | −0.6 |
| six risk ranks | 3.2 | 1.0 | 0.6 | 5.3 | 4.9 | 0.4 |
| risk and momentum ranks (11; the table above) | 4.0 | 0.0 | −0.8 | 4.4 | 5.8 | 1.9 |
| all 39 ranks | 6.7 | 1.9 | −0.4 | 1.8 | 4.0 | 1.5 |

What holds under every definition, and what does not:

1. **By 2014–2021 a large part of the return is not short-horizon reversal.** The net long position alone earns 2.7 of the 8.4% a year, in a rising market. With characteristic exposures added the tilt earns 3.2–6.7%.
2. **The tilt's income falls after 2021 under every definition, by 2.0 to 5.9 points a year** of the 5.0-point drop in gross return from 2014–2021 to 2022–2025: 2.0 for the net long position alone, 2.4 with the risk ranks, 4.4 with momentum added, 5.9 with all 39. So between 40% and all of the measured drop is the tilt; the second correction below redoes this without the windfall. The part common to every definition is the 2.0 points of net long income, and even that is not statistically firm: [−2.1, +6.0] (the 5.0-point drop itself: [+0.0, +10.0]).
3. **2022–2023 were good years for the neutral part under every definition** (4.0–6.2% a year, level with or above 2014–2021), and **2024–2025 were its worst two-year stretch under three of the four definitions** (−0.6 to +1.9%; with all 39 ranks, 2016–2017 and 2019–2020 were lower, at 0.6% and 1.0%). At the layer's cost of about 1.9% a year (0.07 of turnover a day, and 1% for the short side), none of those is positive after costs.
4. **The momentum ranks are the swing, and mostly the slow ones.** Leaving them in the neutral part moves 2024–2025 from +1.9 to between −0.6 and +0.4. Added to the six risk ranks, `r12_2`, `r12_7` and `r36_13` move the neutral part's 2024–2025 from 0.4 to 1.5% a year (`r12_7` alone to 1.1); `r2_1` and `Rel2High` together move it to 0.6. The book turned short 7–12 month winners after 2021 (exposure to the `r12_7` rank +0.054 in 2014–2021, −0.005 in 2022–2023, −0.020 in 2024–2025), and in 2024–2025 they kept winning: the long-short portfolio on that rank earned 6.0% a year at unit gross exposure. The book is not short one-month winners: its exposure to `r2_1` is +0.020 in 2014–2021 and +0.003 in 2024–2025, and that rank's portfolio lost 0.4% a year in 2024–2025. These are slow ranks (the book's exposure to `r12_7` has a 21-day autocorrelation of 0.82 in 2024–2025), so the swing belongs with the standing tilts of 5.6, not with reversal at a monthly horizon.
5. **K = 100 with a tight hedge (7.5 bp, ridge 1e-4) carries about half the tilt** under every definition (1.1–3.7% in 2014–2021), and its neutral part shows the same contrast between 2022–2023 (6.4–8.4%) and 2024–2025 (1.1–2.5%). For it 2024–2025 is not the worst two-year stretch: 2018–2019 is lower under three of the four definitions. The K = 100 row of the first table is the 5 bp run (tilt 1.0–4.8% in 2014–2021; neutral part 6.2–8.3% and 1.1–2.6%).

**A second correction, from section 13.** The neutral part of 2022–2025 in these tables contains the windfall of section 9. In the model without positions in no-signal members it earns 3.1% a year in 2022–2023 and 0.7% in 2024–2025 (4.3% in 2014–2021), so point 3's "good years" were good by about 3 points of luck. The windfall also changes point 2, whose shares are of the measured drop. In that model (`nosig0`) gross return falls 7.0 points a year, from 8.5% in 2014–2021 to 1.5% in 2022–2025. The tilt is 2.1, 2.6, 4.6 and 6.2 points of that fall under the four definitions (30%, 37%, 66% and 89%), never all of it, and the neutral part 4.9, 4.4, 2.4 and 0.8. So with the constant alone, or with the risk ranks, the rest of the book fell about twice as much as the tilt (2.3 and 1.7 times). Under those two definitions only the neutral part's fall has an interval that excludes 0: [+0.9, +8.8] and [+0.9, +7.9], against [−2.4, +6.4] and [−1.1, +6.1] for the tilt (21-day blocks, each period resampled separately). Most of the fall is the tilt only when the momentum ranks count as tilt. Setting the no-signal P&L aside in the reference itself gives the same picture: a 7.2-point fall, of which the tilt is 27%, 36%, 64% and 91%.

*Why I still lean on the split:* it is an identity (the parts add up to the book each day), with no rescaling and no band. *What it cannot do:* say what "the tilt" is without a choice of ranks, and show that removing a tilt raises the Sharpe ratio. The overlay of 5.6 says removal does not help on the development years, because there the tilt paid.

**5.8 Choosing where to trade** (run later, after batch 2). If fresh trades still paid more than they cost in some part of the universe, a gate on the signal could move the trading there. First-day payoff of the day's trade, relative to the universe, in bp per unit traded (reference ensemble, stocks with a signal):

| | 2003–2013 | 2014–2019 | 2020–2021 | 2022–2023 | 2024–2025 |
| --- | --- | --- | --- | --- | --- |
| All stocks with a signal | 2.9 | 2.2 | 1.9 | 1.2 | 3.1 |
| Range over the fifths of `Variance` | 0.8 to 4.4 | 0.0 to 3.6 | −0.4 to 3.6 | −1.5 to 4.0 | −1.3 to 6.3 |
| Range over the fifths of `LME` (size) | 1.3 to 4.1 | 0.9 to 3.1 | −0.6 to 4.3 | −3.5 to 4.4 | −1.9 to 8.0 |

A trade costs 5 bp. Of the 40 fifths (volatility, size, turnover, spread, distance from the high, momentum, volume and unexplained volume), none is above 5 bp in more than two of the five eras. The most volatile fifth is the best of the `Variance` fifths in four of the five eras (the worst in 2022–2023), but it earns 3.6–4.4 bp until 2021 and clears 5 bp only in 2024–2025 (6.3 bp [−2.9, +14.3]). **Dropped: no part of the universe reliably pays a fresh trade its cost.** The table also shows that 2024–2025 is not a collapse of the fresh trades (3.1 bp [−0.7, +6.6], against 2.9 [+2.1, +3.7] in 2003–2013; each two-year figure is uncertain by about ±3 bp): the trading engine is what it has been since 2003, a few basis points short of its cost.

## 6. Batch 1: results

Net SR; Δ is trial − reference, paired block bootstrap. `win12` starts in 2002 and `win16` in 2006.

| Trial | Object | 2003–2021 | Δ 2003–2021 | 2022–2025 | Δ 2022–2025 | 2024 / 2025 |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | model | 0.28 | | −0.34 | | −0.96 / −1.24 |
| | ensemble + band | 1.32 | | +0.13 | | −0.72 / −0.65 |
| `win04` | model | −0.84 | −1.24 [−1.56, −0.94] | −1.49 | −1.31 [−2.18, −0.53] | −2.78 / −2.56 |
| | ensemble + band | 0.99 | −0.33 [−0.65, +0.01] | +0.21 | +0.07 [−0.67, +0.83] | −0.87 / −0.21 |
| `win12` | model | 0.54 | +0.26 [+0.09, +0.44] | −0.30 | +0.05 [−0.39, +0.54] | −0.22 / −1.58 |
| | ensemble + band | 1.51 | +0.19 [−0.02, +0.41] | −0.04 | −0.17 [−0.69, +0.35] | −0.52 / −1.17 |
| `win16` | model | (2006–21) | +0.20 [−0.03, +0.43] | −0.30 | +0.06 [−0.47, +0.62] | −0.47 / −1.46 |
| | ensemble + band | (2006–21) | +0.14 [−0.09, +0.38] | −0.12 | −0.25 [−0.88, +0.40] | −0.91 / −1.36 |
| `drop03` | model | 0.45 | +0.17 [+0.06, +0.29] | −0.18 | +0.18 [−0.03, +0.38] | −0.53 / −1.24 |
| | ensemble + band | 1.27 | −0.04 [−0.16, +0.07] | +0.19 | +0.05 [−0.18, +0.28] | −0.38 / −0.99 |
| `wd05` | model | 0.30 | +0.01 [−0.05, +0.08] | −0.35 | −0.00 [−0.12, +0.12] | −0.73 / −1.23 |
| | ensemble + band | 1.26 | −0.05 [−0.18, +0.08] | +0.12 | −0.02 [−0.22, +0.18] | −0.61 / −0.67 |
| `cum` | model | 0.81 | +0.54 [+0.32, +0.77] | −0.20 | +0.15 [−0.25, +0.58] | −0.44 / −0.99 |
| | ensemble + band | 1.28 | −0.04 [−0.28, +0.25] | −0.02 | −0.15 [−0.56, +0.27] | −0.27 / −0.95 |

1. **The stale-sample explanation is refuted.** A 4-year window is far worse (model −1.24 on the development years and −1.31 on the holdout, both intervals below 0). It trades more (0.79 a day against 0.60), not less: a short window fits its own noise.
2. **The estimation-noise explanation is supported, weakly.** Twelve years help the model (+0.26, interval above 0) and sixteen point the same way; the order is 4 ≪ 8 < 12 ≈ 16. Through the band the gain is +0.19 with an interval that touches 0, and on the holdout the longer windows are lower (−0.17 and −0.25, inside the noise), with a worse 2025.
3. **Dropout 0.3 and the price-path input help the model and nothing else**, as predicted: +0.17 and +0.54 on the development years, 0 through the band. Weight decay does nothing.
4. **Verdicts under section 2's rule.** Kept, for the model only: `win12` (weak: holdout +0.05), `drop03`, `cum`. Kept for the ensemble + band: none. Refuted: `win04`. No effect: `wd05`, `win16` (interval includes 0). Section 2's retraining of a finalist with new seeds was not done for the three kept for the model.
5. **Not one trial turns 2024 or 2025.** Fourteen objects, all negative in both years.

*What batch 1 settles:* the gap between fit and out-of-sample result is noise, not staleness, and dropout, fewer epochs and the price-path input are substitutes for the band: their model gains vanish through it. The longer windows are the exception: `win12` keeps +0.19 [−0.02, +0.41] of its +0.26 through the band, and `win16` +0.14 of +0.20, both inside the noise. So batch 1 adds nothing for the layer, and section 5.7 says why: what failed in 2022–2025 is not something more careful fitting touches.

## 7. Batch 2: a model that cannot hold the tilt

Written before it ran. Two changes to the model itself (`afsa.config.Config.signal_center`, `Config.neutral`; both off by default, tested in `tests\test_neutral.py`):

| Trial | Change | Reason | Prediction |
| --- | --- | --- | --- |
| `center` | each day the signals are centred over the stocks that have one | The sequence model can output the same number for every stock. After the hedge that is "long everything, short the factor portfolios", a standing position. Centring removes that channel and nothing else | Net long position and beta fall. Development years: slightly lower (it gives up the 0.7–2.7% a year the net long position earned). Holdout: about the same |
| `neut_dollar` | the day's weights are made orthogonal to the constant, inside the model | Exact dollar neutrality, as a check on `center` | As `center`. As an overlay it lost 0.38 on the development years |
| `neut_risk` | orthogonal to the constant and six risk ranks (Beta, Variance, Resid_Var, STD_W1, LME, LTurnover) | 5.7: the tilt is a bet that the last window's premia continue. If the network is trained on the neutral book, its objective rewards only what it adds beyond the tilt, and the capacity and the gross exposure the tilt used go to reversal | Development years: lower than the reference by 0.2–0.5 through the band (the overlay lost 0.38). Holdout: higher by 0.1–0.3. 2024–2025: still at or below zero, because the neutral part itself earned 1.9% before costs |
| `neut_riskmom` | the same with five momentum ranks added (r2_1, r12_2, r12_7, r36_13, Rel2High) | The full set of 5.7 | As `neut_risk`, more so (the overlay: −0.57 and +0.16) |

**How these will be read.** Section 2's rule stands, and by it I expect all four to fail: the development years include 2014–2021, when the tilt earned half the return. So I state now what else I will look at, and why it is not a second chance at passing:
- the *neutral part* of each trial's book (the split of 5.7) against the reference's neutral part, on both periods. If training on the neutral book makes the reversal part better, this rises. If it only removes the tilt, this is unchanged, and the change is worth nothing beyond the overlay.
- the holdout by year.

A trial that is lower on the development years and higher on the holdout is **a trade-off for Jesse to judge, not a fix that passed**: it gives up income that was real until 2021 for robustness to its end.

## 8. Batch 3: does the ordering by hedge tightness extend?

Written before it ran. The split of 5.7 across runs already on disk (tilt = the constant and the eleven risk and momentum ranks; 5.7's caution about that choice applies), gross return of the neutral part in % a year:

| Seed ensemble | Tilt's share of exposure, 2014–2021 | Neutral part, 2003–2021 | 2022–2023 | 2024–2025 | 2022–2025 |
| --- | --- | --- | --- | --- | --- |
| K = 8, 5 bp, ridge 1e-4 | 0.65 | 7.2 | 3.2 | −1.4 | 0.9 |
| K = 30, 5 bp, ridge 1e-4 (reference) | 0.52 | 7.6 | 5.8 | 1.9 | 3.9 |
| K = 30, 10 bp, ridge 1e-4 | 0.61 | 5.3 | 5.6 | 0.9 | 3.2 |
| K = 100, 10 bp, ridge 1e-3 | 0.56 | 6.8 | 3.4 | −0.7 | 1.4 |
| K = 100, 7.5 bp, ridge 1e-4 | 0.35 | 7.7 | 8.2 | 2.5 | 5.3 |

On the development years the neutral part earns about the same whatever the hedge (5.3–7.7%). On the holdout it earns more the tighter the hedge: 0.9% at K = 8, 5.3% at K = 100 with ridge 1e-4, and K = 100 with ridge 1e-3 behaves like a small K. *My reading:* the eleven ranks do not span everything a loose hedge leaves in the book. What they miss (industries, combinations the attention factors pick up) also stopped paying after 2021, and only a tight hedge removes it.

**The test.** K = 100, 7.5 bp, 30 epochs, with ridge 3e-5 and 1e-5: tighter than any run on D at K = 100. About 9 minutes each.

**Prediction, if the ordering is a mechanism:** through the band the holdout is at least the 1e-4 cell's (+0.79), or at any rate clearly above the reference's +0.13; the neutral part earns at least 5.3% a year in 2022–2025; the development years stay within noise of the 1e-4 cell's 1.21. **If it is an accident of eleven runs:** the two new cells fall back toward the reference, and K = 100 at ridge 1e-4 was a lucky cell.

*Why this is a fair test:* the two settings have never been trained; they extend the axis beyond every result I have seen; and the expected sign is written here first. *What could spoil it:* at a small ridge the model trades more (REPORT.md §9.4), so its model row may fall. The prediction is about the ensemble + band and the neutral part.

**Added after the second audit:** not the fair test I called it. The hypothesis was formed on the 2022–2025 results above, and the new cells were scored on the same four years, with the same collapsed listings in their books, so their holdout cannot confirm it independently. Section 2 keeps the holdout out of choosing trials; batches 3, 4 and 5 were chosen from it. Their K = 100 setting (7.5 bp, ridge 1e-4) is 7th of the nine K = 100 cells on 2003–2021 through the band (1.21) and 1st on 2022–2025 (+0.79).

## 9. What nine independent analysts found, and what their skeptics changed

A workflow ran beside the trials (CPU only, on the saved runs): nine analysts each tested one hypothesis, and each finding went to a skeptic told to refute it by reading the scripts, recomputing the headline numbers by another code path, and naming the strongest rival explanation. The analysts started from my first reading (section 1) and were told they could overturn it. **No skeptic found a computational error that changes a number. Seven headlines were corrected for claiming more than the data carry; two were partly refuted.** What stands after the skeptics:

| Question | What stands | What the skeptic struck |
| --- | --- | --- |
| When did the edge go? | The largest break is spring 2001 (2001-04-17, interval 2000-11 to 2001-07). 1998–2002 carries 83% of the reference model's 1998–2021 net return. Its net SR since 2003 is about zero: 0.10 for 2003–2019, 0.28 with the burst of March 2020 to 2021 | The analyst's "second break in mid-2021". With turnover held constant (models trained at 2.5 bp), the gross return per unit traded falls about 0.2 bp a year from 2003: 4.3 bp in 2003–2013, 2.7 in 2014 to March 2020, 6.6 in the COVID burst, 1.5 in 2022–2025, which is on the pre-COVID trend (on the trend fit's own measure, which divides by the average turnover of 1998–2025 rather than 2022–2025's: forecast 1.30, realized 1.27; with the no-signal P&L set aside, 0.9 bp per unit traded instead of 1.5) |
| Is the opportunity itself gone? | Parameter-free reversal books on the same universe lost 60–85% of their gross return between 1998–2005 and 2014–2021. None has cleared 5 bp per unit traded since 2006, even through a band | "The benchmarks did not fall further after 2021": the intervals are too wide to say. Net of its market exposure the K = 30 ensemble had no edge in 2014–2021 either (net SR 0.01, then −0.40; change −0.41 [−1.69, +0.95]) |
| What does the book hold? | It changed from a fast reversal book into a slow one that carries the market: its cosine with itself 21 days earlier went from 0.17 (2006–2013) to 0.49 (2014–2021) and 0.66–0.71 (2022–2025). Market plus slow characteristics earned +3.3% a year in 2014–2021 and about zero after; half of that is the market's own return through a beta of 0.14–0.19. The fast part did not change. Trading has run at break-even against its 5 bp cost since 2014 | "The 2024–2025 loss is failed trades": it cannot be assigned. A two-year window as bad as 2024–2025 appears in 38% of stationary resamples |
| How fast does the alpha decay? | A one-day-old target keeps 48–73% of the gross return and a five-day-old one 16–23%. Of about 50 slower ways to trade the same targets, none nets more than 0.27 (K = 30) or 0.35 (K = 100) in 2022–2025. A trade pays 2–3 bp per unit of turnover on its first day against a 5 bp cost, as since 2014 (15 bp in 1998–2002) | "Do not spend training on speed" holds for trading layers, not for a model built with a longer horizon, which nothing on disk tests. *Later: wrong. The two runs trained with the band inside (`BAND_TRAINED.md`) are such a model, and both are lower than the same band added after on 2022–2025, with the windfall set aside too (Conclusions)* |
| Is it fitting noise? | Fitting noise through trading costs 0.3–0.5 of net SR, did not grow after 2021, and is zero at a training cost of 7.5–10 bp | The analyst's premise that noise can only be fitted by trading. A standing tilt is fitted with no turnover: the model for 2024 gained 0.85 of in-sample gross SR between epochs 20 and 30 while its turnover rose by 0.003. No lever on disk regularizes the tilt, and a higher training cost enlarges its share of the book (at K = 30 in 2014–2021, from 5 to 10 bp: 0.40 to 0.64 of the weights' cross-sectional variance by the skeptic's slow-tilt measure; 0.52 to 0.61 of gross exposure by the split of section 8) |
| Where is the money lost? | Not in a few days, one leg, one security type, one size bucket, one industry or trades against large moves; none of 12 filters raises the 2022–2025 net SR | "The loss is broad". Against 2014–2021 the shortfall is in the systematic legs (net market exposure and characteristic tilts: +4.8 to +6.7% a year, then about zero), while the stock-specific positions earn +2.35%, then +3.22%, and are below their costs in both periods. **Corrected later:** 1.6 points of the 3.22 are the no-signal shorts described below the table. With them set aside the stock-specific positions earn +2.6%, then +1.6%, a fall of 1.0 [−3.8, +1.9], inside the noise |
| Market exposure and cost floor | 2022–2025: net long 0.18, beta 0.135. Beta × market added +0.64% a year, so the exposure propped the result up. Financing at the risk-free rate costs 0.71% a year; the short fee 1.03% a year, the only fixed floor (0.24 of SR in every era) | "Do not add a neutrality constraint": overlays cannot tell what a network trained under neutrality would do. Batch 2 answers that |
| Which existing levers help? | Only the levers that cut trading carry over to 2022–2025: training cost 5 → 10 bp, +0.35 [+0.14, +0.56] for the model (averaged over K = 8, 30 and 100 at ridges 1e-4, 3e-4 and 1e-3; at K = 30, ridge 1e-4 alone +0.38, section 3); ensemble + band, +0.49 [+0.26, +0.74]. They are substitutes: with the layers on, the training cost adds +0.15 [−0.11, +0.40]. A setting chosen on 2003–2021 would have scored −0.31 as a model and +0.01 with the layers, no better than the reference | "The hedge levers reverse". A larger K raises the *gross* SR in every four-year window since 1998 (+0.50 in 2003–2021, +0.73 [+0.34, +1.10] in 2022–2025) and a larger ridge lowers it. The *net* ranking flipped only because the cost a loose hedge used to save shrank with turnover. The tight-hedge corner (K = 100, ridge ≤ 1e-4, with the layers) is "motivated, not proven" |
| What is known elsewhere? | The decay is documented (decimalization in 2001, algorithmic trading). The paper's own Figure 2 earns 64–69% of its 24-year gross return in 1998–2005 and 5.7% a year in 2014–2021 at K = 30, but it shows no 2001 step: its gross SR stays at 7.5–11.6 through 2006 (`PAPER_BY_YEAR.md`), so decimalization is not shown to be the cause of ours. The public large-stock reversal factor's gross Sharpe ratio is 0.89 / 0.31 / 0.18 / 0.08 by era | "It keeps falling through 2022–2025": no detectable further fall since 2006. The literature predicts a net SR near zero for the extension, not −0.3 |

One finding I had not looked for (the skeptic of *Where is the money lost?*): **the measured 2022–2025 result is flattered by luck.** The hedge puts large short positions (in the reference ensemble 1.6% of the book each in 2022–2023 and 2.8% in 2024–2025, about 0.4 stocks a day) in members that have no signal yet, mostly new listings. In 2022–2025 these earned +2.2% a year [+0.6, +4.2] relative to the list's equal-weighted return, against zero before 2022: Addentax fell 95% on its first day on the list, and VinFast collapsed. Without that relative P&L the reference ensemble's 2022–2025 net SR is −0.72, not −0.18; the table below sets aside their raw P&L instead (+2.05% a year, net SR −0.67). Nothing in the model chose these stocks, and the same positions earned about zero before 2022, so the gain is not one the strategy can count on.

**I checked the windfall myself, and it reaches into section 5.5.** Written after batch 2 was scored and before batch 3 was. Seed ensembles, 2022–2025:

| | No-signal positions: size each | Their P&L, % a year | Net SR | Net SR with their P&L set aside |
| --- | --- | --- | --- | --- |
| K = 8, 5 bp, ridge 1e-4 | 1.7% | +1.7 | −0.64 | −0.98 |
| K = 30, 5 bp, ridge 1e-4 (reference) | 2.2% | +2.05 | −0.18 | −0.67 |
| K = 100, 7.5 bp, ridge 1e-4 (tight hedge) | 2.8% | +3.2 | +0.38 | −0.45 |
| K = 100, 10 bp, ridge 1e-3 (loose hedge) | 1.4% | +1.6 | −0.00 | −0.29 |

An ordinary position is 0.18% of the book. These are always short, on 0.43 stocks a day; before 2022 they earned about zero (−0.2% a year). They are the hedge of the part of the signals common to all stocks (section 10, after point 4): `center`, which removes that part, holds 0.15% in each.

- **The tighter the hedge, the larger these shorts, and the larger the windfall.** That is a good part of 5.5's ordering "by hedge tightness". With the windfall set aside, K still orders the runs (8 < 30 < 100), but the ridge no longer does: the loose-hedge K = 100 is ahead of the tight one.
- **From here on every trial is read both ways:** as pre-registered, and with the P&L of no-signal positions set aside (`extension.py report --ex-nosignal`; the positions and their costs stay, only what they earned is removed). The band width is chosen again on the adjusted 2003–2021. Of the band figures this file reports that way, only the reference's width changes, from 1.2e-3 to 1.6e-3: its −0.37 is for the wider band (−0.38 at 1.2e-3), and the differences against it move by at most 0.01. *Why add a reading after the fact:* three stocks' collapse is not something a fix can claim. *What it is not:* the luck set aside. Over 2022–2025 it nets 18 names: the three collapses (+8.9% of the book) and ten that lost (−1.7%, chiefly Circle and Kaspi, shorted while they rose). Single-name luck elsewhere in the book went the other way. Through the band, the three worst positions with a signal (shorts in Super Micro, AppLovin and CoreWeave) lost 1.2% of the book a year, against 0.5–0.8% for the three worst in each earlier four-year block since 2002. Super Micro and AppLovin were held for about two years at three times an ordinary position, so they are the model's own tilt (5.7), and this reading keeps them. Setting aside the largest names whatever their sign gives −0.21 to −0.29 through the band on 2022–2025 (3, 5 or 10 names), and 1.10–1.20 instead of 1.32 on 2003–2021. *Why keep the pre-registered one:* it is what was promised, and it is what the stored results say.
- **This bears on batch 3's prediction.** A tighter ridge will hold larger shorts in the same lucky names, so the pre-registered measure would rise for the wrong reason. The test of section 8 is fair only on the second reading.

**What this changes in my reasoning.**

1. **Claim 1 of section 1 stands, with the date corrected**: the large break is 2001, and the model's net edge has been near zero since 2003. Three analysts and their skeptics reached it by different routes (break search, free benchmarks, constant-turnover probe).
2. **Claim 2 is weakened.** I wrote that for the ensemble + band "I do suspect a change, in the last two years only". The skeptics give two reasons to doubt it. The layer's 2003–2021 result is lifted by market exposure (the reference layer's +0.13 on 2022–2025 is +0.01 once beta × market is removed, and its 2014–2021 is flattered the same way, 1.11 → 0.84; over 2003–2021 it is 1.32 → 1.16). And 2024–2025 was picked because it is the worst pair; allowing for that, it is an ordinary draw. The skeptic of *Is the opportunity itself gone?* summed it up as "2014–2021 was flattered rather than 2022–2025 broken". Its evidence was the market-hedged comparison (ensemble net SR 0.01 in 2014–2021, −0.40 in 2022–2025; change −0.41 [−1.69, +0.95]), which shows only that the two periods cannot be told apart once the market is removed. Against 2003–2013, the exposures paid their usual rate per unit in 2014–2021 and less in 2022–2025 (Conclusions, point 2).
3. **The fourth observation of section 1 stays unexplained.** A standing tilt costs no turnover, so no cost term restrains it, and the 2024 model gained 0.85 of in-sample gross SR between epochs 20 and 30 while its turnover rose by 0.003. That makes the tilt a candidate, but the books these two models traded do not support it. The tilt's share of gross exposure is 0.43 in 2024–2025 against 0.54 in 2022–2023 (5.7's table), and 2024 and 2025 are each below 2022 and 2023 under all four tilt definitions of 5.7, in every seed. A gain in fit at little added turnover is not special to these models either: the 2013 model gained 1.22 of gross SR while its turnover rose by 0.017. Their in-sample books were not replayed, so this check is indirect.
4. **Section 5.7's split is confirmed by four independent decompositions** (stale books, regression on ranks, position age, signal against hedge). All carry the same caution: each component is inside its own noise, and 2014–2021 is the tilt's best era, most of it 2019–2021.
5. **The edge per unit traded is on a 20-year downward line that reaches zero around 2030 at K = 30.** A linear extrapolation is a warning, not a forecast. But it means a cost-side fix faces a falling ceiling.

*Why I give the skeptics' corrections this much weight:* each recomputed the numbers independently before judging them, and their corrections go in one direction, against reading four years too finely, which is the error I am most exposed to here.

## 10. Batch 2: results

| Trial | Object | 2003–2021 | Δ 2003–2021 | 2022–2025 | Δ 2022–2025 | Neutral part, gross % a year: 2003–2021 / 2022–2025 |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | model | 0.28 | | −0.34 | | 7.6 / 3.9 |
| | ensemble + band | 1.32 | | +0.13 | | |
| `center` | model | −0.26 | −0.59 [−0.90, −0.31] | −1.06 | −0.91 [−1.88, +0.09] | 9.7 / 4.9 |
| | ensemble + band | 0.97 | −0.34 [−0.66, −0.05] | −0.47 | −0.60 [−1.69, +0.52] | |
| `neut_dollar` | model | −0.08 | −0.39 [−0.65, −0.11] | −0.56 | −0.23 [−0.95, +0.40] | 8.1 / 4.1 |
| | ensemble + band | 1.09 | −0.23 [−0.52, +0.08] | −0.04 | −0.18 [−0.97, +0.52] | |
| `neut_risk` | model | −0.20 | −0.52 [−0.84, −0.20] | −0.84 | −0.54 [−1.32, +0.19] | 8.5 / 3.9 |
| | ensemble + band | 1.06 | −0.26 [−0.54, +0.04] | −0.22 | −0.36 [−1.20, +0.40] | |
| `neut_riskmom` | model | −0.48 | −0.85 [−1.23, −0.49] | −0.63 | −0.32 [−1.17, +0.44] | 8.8 / 3.6 |
| | ensemble + band | 0.96 | −0.36 [−0.71, −0.04] | +0.02 | −0.12 [−1.01, +0.67] | |

1. **The development years are lower, as predicted** (−0.23 to −0.36 through the band; the overlays lost 0.38–0.57).
2. **The holdout is not higher. My prediction of +0.1 to +0.3 was wrong:** all four are lower (−0.12 to −0.60), inside the noise. This table was scored before the windfall check of section 9, so it has only the pre-registered reading; the second reading is given after point 4.
3. **Training on the neutral book does not improve the reversal part after 2021.** Its gross return in 2022–2025 is 3.6–4.1% a year for the three neutral models against the reference's 3.9%. On 2003–2021, the other period section 7 named, it is higher (8.1–8.8% against 7.6%), but by less than the tilt it replaces earned (3.1% a year in the reference, 0–1.6% in the neutral models), so the book earns less before costs. The network found a little more reversal before 2022 once the tilt was closed to it, and none after. (`center` shows 9.7% and 4.9%, but it trades 0.80 a day in 2022–2025, against the reference's 0.30, and nets less: −1.06 against −0.34 as measured, −1.04 against −0.82 with the no-signal P&L set aside.)
4. **Verdict: not supported, and no trade-off to offer either.** Neutrality inside the model removes income that was real until 2021 and gives nothing back after. It remains a risk control: the neutral models carry no net position and no exposure to the chosen ranks, which is what the paper's Table 2 betas (0.05–0.07) describe. It does not raise the Sharpe ratio in any period.

**Read again with the P&L of no-signal positions set aside** (section 9's second reading, added after the closing audit; each band width chosen again on 2003–2021):

| Trial | Model, 2022–2025 | Δ | Ensemble + band, 2022–2025 | Δ |
| --- | --- | --- | --- | --- |
| Reference | −0.82 | | −0.37 | |
| `center` | −1.04 | −0.39 [−1.34, +0.56] | −0.46 | −0.09 [−1.21, +1.02] |
| `neut_dollar` | −1.17 | −0.38 [−1.05, +0.25] | −0.63 | −0.26 [−1.04, +0.44] |
| `neut_risk` | −1.43 | −0.67 [−1.43, +0.03] | −0.81 | −0.44 [−1.23, +0.31] |
| `neut_riskmom` | −1.23 | −0.47 [−1.28, +0.28] | −0.59 | −0.22 [−1.10, +0.58] |

All four are still lower, so the verdict stands, and the 2003–2021 differences move by 0.06 or less. But most of `center`'s shortfall as measured (−0.91 for the model, −0.60 through the band) is the windfall it does not hold, and section 7's "removes that channel and nothing else" was wrong: centring the signals removes 87% (2003–2021) and 93% (2022–2025) of the ensemble's gross weight in members without a signal, and in 2022–2025 those positions earn −0.08% a year against the reference's +2.05%. A member without a signal holds only the hedge of the other stocks' signals, so the large short positions of section 9 (85% of them short in the reference's ensemble, 47% in `center`'s) are mostly the hedge of the signals' common level. The three neutral models keep them (0.86–0.92% of the book a day in 2022–2025, against the reference's 0.95%) and their windfall (+1.9 to +2.0% a year). The neutral-part column above holds that windfall too: without it the neutral part earns 2.0% a year in 2022–2025 for the reference, 1.7–2.4% for the three neutral models and 5.2% for `center`.

*Why `center` drops them:* a member without a signal has no ω^port of its own, so its position is only the hedge of the other stocks' signals. A part common to all signals is a long position in every stock that has a signal. The hedge offsets it, and on a stock without a signal only the offset is left: a short. Centring removes that common part, and with it most of the book's net long position (0.19 for the reference's seed ensemble in 2022–2025, 0.03 for `center`'s). The neutral models remove the constant from the weights after the hedge, which leaves the shorts in place.

*Why this result is informative although negative:* it closes the last reading under which the tilt was the cause and its removal the cure. Sections 5.6, 5.7 and 9 said the tilt *accounts* for the drop from 2014–2021. Batch 2 says that removing it leaves the reversal book as it was, and section 9 says that book has traded at break-even against a 5 bp cost since 2014.

## 11. Batch 3: results

K = 100, 7.5 bp, 30 epochs; ensemble + band. Δ against the K = 30 reference.

| Ridge | 2003–2021 | Δ | 2022–2025, as pre-registered | Δ | 2022–2025, no-signal P&L set aside | Δ | Neutral part, gross % a year: 2022–2023 / 2024–2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K = 30 reference | 1.32 | | +0.13 | | −0.37 | | 5.8 / 1.9 |
| 1e-4 (on disk before) | 1.21 | −0.11 [−0.42, +0.15] | +0.79 | +0.65 [+0.11, +1.13] | −0.01 | +0.36 [−0.16, +0.88] | 8.2 / 2.5 |
| 3e-5 | 1.05 | −0.27 [−0.61, +0.06] | +0.91 | +0.77 [−0.04, +1.47] | −0.11 | +0.26 [−0.46, +0.94] | 10.9 / 2.5 |
| 1e-5 | 0.99 | −0.33 [−0.74, +0.05] | +0.86 | +0.73 [−0.13, +1.53] | −0.10 | +0.28 [−0.50, +1.07] | 10.7 / 1.9 |

1. **On the pre-registered measure the prediction holds:** the two tighter cells reach +0.91 and +0.86, above the 1e-4 cell's +0.79; the neutral part earns 6.3–6.7% a year in 2022–2025; and on 2003–2021 they stay within noise of the 1e-4 cell, at its edge (−0.16 [−0.31, +0.00] and −0.22 [−0.47, +0.04]).
2. **On the fair reading it does not.** With the P&L of the no-signal shorts set aside (the reading I fixed before scoring, section 9), the three cells are −0.01, −0.11 and −0.10. A tighter ridge buys nothing more; what rose on the pre-registered measure was the windfall, which grows with the hedge.
3. **The tighter hedge costs the development years**: 1.21, 1.05, 0.99, the last two close to clearly below the reference. As models they are worse still (2003–2021: 0.24, −0.05, −0.30), because a small ridge trades more.
4. **What survives is K, not the ridge, and less of it than these Δ say:** K = 100 at 7.5 bp is +0.26 to +0.36 above the K = 30 reference (5 bp) on the holdout with the luck set aside, inside the noise, and level with it on the development years at ridge 1e-4. Part of that is the training cost: K = 30 at 7.5 bp is itself +0.12 [−0.13, +0.38] above the reference. At equal cost and ridge 1e-4, K = 100 leads by +0.26 [−0.29, +0.81] at 5 bp, +0.24 [−0.25, +0.75] at 7.5 bp and +0.01 [−0.50, +0.53] at 10 bp. In 2024–2025 the neutral part earns 1.9–2.5% a year at every ridge, as at K = 30.
5. **Verdict:** the ordering by hedge tightness of section 5.5 was, for the ridge, an artefact of three stocks. It does not extend. K = 100 at ridge 1e-4 stays the highest on disk for 2022–2025, at about zero; it is a maximum picked on those years, so an upper estimate.

*What I take from having been half right:* the pre-registered prediction came true for a reason I had not written down, and only the skeptic's finding let me see it. A prediction that holds is not yet a mechanism confirmed; here the check that mattered was the one added between the prediction and its scoring.

## 12. Batch 4: no position without a signal

Written before it ran. One more change to the model itself: `Config.nosignal_hedge = False`. A member without a signal holds no position, and the others are scaled to unit gross exposure. It still enters the factors and the hedge of the other stocks.

*Reason:* section 9's windfall. As built, the hedge gives a new listing a short of 1.4–2.8% of the book, 8 to 15 times an ordinary position, in a stock with a few days of history. *Added after the closing audit:* not only new listings. Every member without a signal (a new listing, or a member whose first annual accounts have not entered) gets such a position. In 2022–2025 these were all short, at 1.4–2.8% of the book in the four runs of section 9, 8 to 15 times an ordinary position; in 2003–2021 the reference's were 7 times an ordinary position, 96% of them short. Of the reference's 431 such stock-days in 2022–2025, 304 were in stocks listed 30 trading days or fewer; the other 127 were VinFast, short at 2.1% of the book while it waited for accounts (2023-09-27 to 2024-03-28), which earned a fifth of the windfall (+0.44 of +2.05% a year). In 2022–2025 three of them collapsed and paid; nothing in the model chose them. A position of that size in an unseasoned stock is a risk the strategy does not intend, and it makes the measured extension depend on three names. The "set aside" reading keeps the positions and removes their P&L; this trial removes the positions, so the network is trained and scored without them.

| Trial | Change | Prediction |
| --- | --- | --- |
| `nosig0` | the reference without those positions (about 3 minutes) | Development years: unchanged within 0.05 (the positions earned about −0.2% a year before 2022). Holdout: about 0.5 below the reference as measured, near the set-aside figure of −0.37 for the ensemble + band. By section 2's rule that is a failure, and it is the point: it measures the extension without the luck |
| `K100_nosig0` | K = 100, 7.5 bp, ridge 1e-4, without those positions (about 8 minutes), compared with `nosig0` | The clean comparison of K = 100 with K = 30. From the set-aside figures (−0.01 against −0.37) I expect about +0.35 on the holdout through the band, inside the noise, and no difference on the development years |

## 13. Batch 4: results

| Trial | Object | 2003–2021 | Δ 2003–2021 | 2022–2025 | Δ 2022–2025 | 2022 / 2023 / 2024 / 2025 |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | model | 0.28 | | −0.34 | | 0.09 / 0.53 / −0.96 / −1.24 |
| | ensemble + band | 1.32 | | +0.13 | | 0.84 / 1.12 / −0.72 / −0.65 |
| `nosig0` | model | 0.38 | +0.10 [+0.02, +0.18] | −0.80 | −0.47 [−1.01, −0.02] | −0.56 / −0.21 / −1.15 / −1.60 |
| | ensemble | 0.69 | +0.08 [−0.01, +0.16] | −0.65 | −0.46 [−0.99, −0.02] | −0.43 / −0.04 / −0.88 / −1.35 |
| | ensemble + band | 1.39 | +0.08 [−0.03, +0.20] | −0.33 | −0.46 [−1.04, +0.05] | 0.15 / 0.19 / −0.90 / −0.84 |
| `K100_nosig0`, against `nosig0` | model | 0.32 | −0.04 [−0.41, +0.31] | −0.80 | −0.01 [−0.50, +0.43] | −0.70 / −0.72 / −1.31 / −0.86 |
| | ensemble + band | 1.17 | −0.22 [−0.60, +0.09] | −0.15 | +0.18 [−0.32, +0.66] | −0.05 / 0.39 / −0.61 / −0.35 |

1. **`nosig0` lands where predicted on the holdout** (−0.33 through the band, against the set-aside figure of −0.37), so at K = 30 the set-aside reading was a fair stand-in for a model without those positions. At K = 100 the two differ by 0.14 (`K100_nosig0` −0.15 against −0.01 set aside on 2022–2025; 1.17 against 1.31 on 2003–2021).
2. **It is slightly better, not unchanged, on the development years**: +0.10 for the model with an interval above 0, +0.08 through the band. On 1998–2021 the model goes from 1.31 to 1.38. Dropping the positions costs nothing before 2022.
3. **K = 100 against K = 30 without the luck: +0.18 on the holdout through the band, inside the noise, and −0.22 on the development years.** I predicted about +0.35 and no difference. The models are level (−0.80 both). K = 100 is not reliably better. Nor is this the clean comparison section 12 called it: `K100_nosig0` trains at 7.5 bp and `nosig0` at 5 bp, and no K = 30 run at 7.5 bp without those positions exists to separate the two.
4. **2022 and 2023 were not good years either.** Without those positions (`nosig0`) the ensemble + band earns 0.15 and 0.19 in 2022 and 2023, against the reference's 0.84 and 1.12; the reference with their P&L set aside earns 0.19 and 0.36 (band 1.6e-3 for both; the reference as measured 1.2e-3). So section 1's claim 2 loses its last support: the layer did not hold up until 2023 and then fail; it was carried in 2022–2023 by two collapsed listings.
5. **The neutral part without the windfall** (split of 5.7, gross % a year, 2003–2013 / 2014–2021 / 2022–2023 / 2024–2025): 9.9 / 4.3 / 3.1 / 0.7 for `nosig0`, against 9.9 / 4.4 / 5.8 / 1.9 for the reference. This corrects 5.7 once more: with the eleven ranks the reversal part did not hold at its 2014–2021 level in 2022–2023 (4.3% against 3.1% a year, a fall inside the noise: 1.2 points [−2.8, +5.0]; with all 39 ranks it held, 1.7% and 1.5%). It fell from 4.3% to 1.9% a year over 2022–2025, and the tilt from 4.2% to −0.4%.
6. **Verdicts.** As measured, `nosig0` passes section 2's development leg for the model (+0.10 [+0.02, +0.18]) and fails on the holdout (model −0.47, ensemble + band −0.46), as intended: it removes luck. On section 9's second reading, with the reference's no-signal P&L set aside, it is kept for the model: +0.06 [+0.02, +0.10] on 2003–2021, and +0.03 (model) and +0.04 (ensemble + band) on 2022–2025. As a change to the model it is the one I would keep, for a reason the rule does not measure: it removes positions 8 to 15 times the ordinary size in stocks with a few days of history or without annual accounts yet (VinFast for seven months on the list), at no cost before 2022.

## 14. Batch 5: does K extend beyond 100?

Written before it ran, while the audit of section 15 was reading the sections above. After batches 3 and 4 one lever is left standing: with the windfall out, K = 100 leads K = 30 by 0.18–0.36 on 2022–2025 through the band, inside the noise, and K = 8 trails both. The ridge axis failed the same kind of test in batch 3; this is that test for K.

**The trial.** `K200_nosig0`: K = 200, 7.5 bp, ridge 1e-4, 30 epochs, no position without a signal, test years 2014–2025 only (36 models), so that it stays under 20 minutes. Compared with `K100_nosig0` and `nosig0` on 2014–2021 and on 2022–2025.

**Prediction.** If K is a real lever beyond 100: through the band, 2022–2025 above `K100_nosig0`'s −0.15, and 2014–2021 not clearly below its 0.64. **What I expect:** no clear difference in either period (within ±0.3), because the step from 30 to 100 was already inside the noise and cost 0.22 on the development years. A K = 200 that is clearly worse on 2014–2021 would close the K axis as the ridge axis was closed.

*What this cannot show:* twelve test years instead of 28, so the development period is 2014–2021 only, the years section 9 says were flattered by market exposure. I will read it beside the split.

**Results** (36 models in 9 minutes). Every run here has no position without a signal; each band width is the best on 2014–2021.

| Through the band | 2014–2021 | 2022–2025 | 2024–2025 | Neutral part, gross % a year: 2014–2021 / 2022–2025 |
| --- | --- | --- | --- | --- |
| K = 30 (`nosig0`) | 1.12 | −0.33 | −0.85 | 4.3 / 1.9 |
| K = 100 (`K100_nosig0`) | 0.66 | −0.20 | −0.39 | 4.1 / 1.8 |
| K = 200 | 0.74 | −0.62 | −0.83 | 3.0 / 0.8 |
| K = 200 − K = 100 | +0.08 [−0.47, +0.70] | −0.42 [−0.93, +0.08] | −0.44 [−1.15, +0.28] | |
| K = 200 − K = 30 | −0.38 [−0.81, +0.06] | −0.29 [−1.00, +0.37] | +0.02 [−1.00, +1.05] | |

As models, K = 200 is clearly worse than K = 100 on 2022–2025 (−1.42 against −0.80; −0.71 [−1.28, −0.14]) and than K = 30 on 2014–2021 (−0.02 against 0.42; −0.46 [−0.89, −0.09]), and level with K = 100 on 2014–2021 (−0.13 [−0.67, +0.45]).

1. **My expectation of no clear difference was wrong on 2022–2025.** K = 200 is lower than K = 100 there for every object, clearly so for the model. The condition I wrote for closing the axis, clearly worse on 2014–2021, is not met against K = 100 (model −0.13 [−0.67, +0.45], band +0.08 [−0.47, +0.70]); it is met only for the model against K = 30, which trains at 5 bp (−0.46 [−0.89, −0.09]). The comparators' band widths were chosen again on 2014–2021, which moves `K100_nosig0` from the −0.15 and 0.64 named above to −0.20 and 0.66; against the widths as named, K = 200 through the band is +0.10 [−0.45, +0.71] on 2014–2021 and −0.47 [−1.02, +0.04] on 2022–2025. The tilt is smaller (a quarter of the book's exposure against a third), but the neutral part earns less too (0.8% a year against 1.8%).
2. **The K axis peaks at or below 100, and the peak is inside the noise.** K = 30, 100 and 200 are within 0.5 of one another through the band in both periods (0.46 apart on 2014–2021, 0.42 on 2022–2025), and K = 30 is the best of the three on 2014–2021.
3. **Verdict: not supported.** A larger K does not extend the small lead K = 100 had after 2021. With the ridge axis closed by batch 3, no setting of K or the ridge is left that raises 2022–2025 once the windfall is set aside.

## 15. The closing audit

**What it was.** A workflow of nine auditors, each told to refute one part of this file with its own code. The parts:
- the model flags
- the harness
- the numbers of the trials, of the pre-checks and of the split
- the protocol
- the logic
- missed fixes
- a reader's view (Jesse, Ander, Prof. Polak)

Every serious issue was to go to a verifier.

**The first audit (2026-11-15).** It was stopped at Jesse's wrap-up, after six reports and before any verifier finished. The six were the model flags, the harness, the three number lenses and the reader. Its corrections to the Conclusions were applied the same day, each checked against this file's own tables:
- the seed ensemble alone is worth about 0.15, not 0.3–0.5;
- the ensemble + band did cover its costs on 2003–2021;
- the recommendation of a 7.5–10 bp model under the layer contradicted the file's own verdict;
- "the trading itself did not change" joined two measures;
- the Conclusions lacked definitions.

**The second audit (2026-11-16)** finished the job: 19 agents, CPU only.
- **Who ran.** A verifier for each of the seven unapplied major findings; one agent per auditor for the forty minor ones; and the protocol, logic and missed-fixes lenses, each followed by a verifier of its serious issues.
- **The bar.** A finding counted only once the verifier had reproduced it with its own code.
- **Outcome.** Of 60 verdicts, 57 needed a fix (7 major) and 3 were already fixed; none was rejected. All 57 are applied, merged where two touched one sentence.
- **Pre-registered text** (sections 8, 12 and 16) got appended notes, never rewrites.
- **Journals.** On this machine only, under `%USERPROFILE%\.claude\projects\c--Claude-AFSA\fb5f607a-4449-4f09-bac6-7c5ed9e481fc\subagents\workflows\` (`wf_14482bfd-547`, `wf_502f1e5e-87c`).

**What both confirmed.**
- **No computational error.** About 450 cells of the trial tables reproduce by an independent code path within 0.01, and no interval changes side of zero. The pre-check tables of 5.1–5.6, the split tables and section 9's windfall table reproduce cell by cell. The split is an exact identity: tilt + neutral = gross to 1e-17.
- **The three model flags are sound.** With them off, results are byte-identical to the committed code.
- **The order of events is clean.** Each prediction was written before its batch launched and not edited afterwards.

**What the second audit changed in the conclusions.**
1. **2022–2025 is a further fall, not four more years like 2003–2021.** With the windfall set aside, 2022–2025 minus 2003–2021 is −1.19 [−2.21, −0.21] for the model and −1.22 [−2.26, −0.26] for `nosig0`. Section 1 had named this test as the way to overturn its claim 1, and never ran it. As measured the difference is −0.66 [−1.71, +0.33].
2. **"2014–2021 was flattered" is withdrawn.** Against 2003–2013, the exposures paid their usual rate per unit held in 2014–2021; the book simply held more of them. It is 2022–2025 that departs (Conclusions, point 2).
3. **"No break in the trading" is qualified.** The whole book's rank correlation with the day's returns fell after 2021: −1.22 [−2.00, −0.41]. The neutral part's is on its 2003–2019 trend.
4. **The 2001 break is ours.** The paper's Figure 2 shows none: its gross SR stays at 7.5–11.6 through 2006 (`PAPER_BY_YEAR.md`).
5. **The holdout chose trials.** K = 100 at 7.5 bp with ridge 1e-4 was carried into batches 3–5 for its 2022–2025 result. It is 7th of the nine K = 100 cells on 2003–2021 and 1st on 2022–2025, so its −0.01 is an upper estimate. Its lead over K = 30 also mixes K with the training cost: at equal cost it is +0.01 to +0.26.
6. **`nosig0` is kept for the model on the second reading:** +0.06 [+0.02, +0.10]. `ep20`, `drop03` and `win12` pass the rule for the model by similar margins and were never retrained with new seeds, as section 2 promised.
7. **K = 200 does not close the K axis.** It is level with K = 100 on 2014–2021, so the pre-registered closing condition was not met.
8. **Batch 2 is read again without the windfall** (section 10). Centring removes most of the no-signal positions, which explains most of `center`'s deficit as measured.
9. **The multi-day objective had been tested.** The two runs trained with the band inside (`BAND_TRAINED.md`) are such a model, and both are lower on 2022–2025 with the windfall set aside.
10. **The ceiling for trading-side changes** (Conclusions, "What raises", point 4). I recomputed it from the saved books: without the windfall, the reference's banded book earns 0.39% a year before costs on 2022–2025, against a short fee of 1.07%.
11. **Smaller corrections** in sections 1, 2 (which code trained what), 3, 5.1, 5.3–5.8, 6, 9, 11, 13 and 14, each marked where it stands.

**What the missed-fixes lens ranks as left to try**, by expected value per GPU minute. The percentage is its prior that the change raises the 2022–2025 ensemble + band by more than 0.3 with the windfall set aside.
1. Score `lv10`: free.
2. A 60-day look-back for the sequence model, without no-signal positions: about 7 minutes of GPU, 8%.
3. Momentum neutrality without the constant, inside the model: about 4 minutes, 5%. This idea is informed by the test years (5.7).
4. A volume or liquidity input to the sequence model: about 5 minutes, 5%.
5. `lv1000`: about 6 minutes, 5%.
6. The largest 300 within D: 2%.
7. `band_eta` 0.5 with no band: 2%.
8. Another normalization of exposure: 3%.
9. Seeds 3–9: 2%. They are needed for the `nosig0` decision, not for the goal.
10. Quarterly retraining: 2%.
11. An earnings-date flag: 5%, but it needs a WRDS download.

**Raised but not verified** (minor; for the next session).
- **Protocol: the rule's verdicts are not stated in full.** Applied mechanically, it keeps ten runs for the model, including four K = 100 cells.
- **Protocol: multiple testing.** No 2022–2025 interval survives a Holm or Benjamini–Hochberg correction. Several development-year intervals the record leans on, `nosig0`'s +0.10 among them, do not survive Holm.
- **Protocol: two recommendations sit uneasily with CLAUDE.md §7 and §8.** These are the entry-rule change and the deployment band width.
- **Logic: the window trials are confounded with the number of passes over each day.** A window other than 8 years keeps 480 optimizer steps, so each day is seen more or less often.

## 16. Batch 6: the weight of the factor objective, and more seeds

Written before it ran, while the audit of section 15 was still running. Every run here has no position without a signal (`nosignal_hedge` false), K = 30, 5 bp, ridge 1e-4, 30 epochs, 1998–2025, and is compared with `nosig0`; section 2's rule applies, each band width chosen on 2003–2021.

| Trial | Change | Models | Reason | Prediction |
| --- | --- | --- | --- | --- |
| `lv10` | λ_Var 10 (default 100) | 84 | The objective is a batch's annualized net Sharpe plus λ_Var × explained variance. At 100 the second term is most of it: the 2023 model ends at 47.1 = 2.1 (net SR) + 100 × 0.45 (explained variance). So the attention mostly explains variance, and only the sequence model serves trading alone. The weight has never been varied here. A smaller weight hands the factors to trading, which the paper credits for its edge over PCA + LongConv | The factors fit trading in-sample and lose it out of sample (batch 1: the gap is noise). Model lower on 2003–2021; through the band within noise |
| `lv1000` | λ_Var 1000 | 84 | The other direction: factors closer to statistical ones, a regularizer on the attention that no trial so far has touched (dropout and weight decay act on the sequence model) | Explained variance up, in-sample net SR down; model up by 0.1–0.2 on 2003–2021, like `drop03`; through the band within noise, because fitting less has always been a substitute for the band |
| `nosig0`, seeds 3–9 | seven new seeds, in a separate folder | 196 | Seeds disagree: two seeds' trades correlate at about 0.6, so a third or more of each seed's trading is its own noise. Averaging seeds removes it from the target itself, not just from how it is traded; on `wrds_broad` ten seeds beat three by 0.05–0.15 (SCAN_RIDGE.md). This is the one lever on the table that should raise the edge per unit traded | 10-seed ensemble + band above the 3-seed one by 0.05–0.15 on 2003–2021, and on 2022–2025 still below zero (about −0.2 against −0.33) |
| reference, seeds 3–9 | the same for the reference | 196 | Section 2 promised that a trial passing by a small margin is retrained with new seeds. `nosig0` passed by +0.10 [+0.02, +0.18] for the model; seeds 3–9 of both runs re-measure that margin on independent models | `nosig0` − reference on seeds 3–9: model +0.05 to +0.15 on 2003–2021, the same sign as seeds 0–2 |

*What would change my conclusions:* a 10-seed ensemble + band above zero on 2022–2025 would make the seed count the first lever to clear the windfall-free bar; a λ_Var run that passes through the band would be the first change to the factor model to add edge the band cannot. *What I expect:* neither. The seed gain is real but small against a gap of 0.3, and the λ_Var runs act on how much is fitted, which every earlier trial found to be a substitute for the band.

Training: about 20 minutes in all, one GPU job, by [extension_batch6_train.py](extension_batch6_train.py), scored by [extension_batch6_score.py](extension_batch6_score.py). Both add the trials to `extension.TRIALS` in memory, because `extension.py` was under audit.

**Status: stopped at Jesse's wrap-up, before anything was scored.**
- **On disk:**
  - `lv10` is complete with weights: 84 models, replayed exactly.
  - `lv1000` has 36 of its 84 models (seed 0 every year, seed 1 to 2005) and no weights.
  - The seed runs had not started.
- **Checks:** every file written was loaded back, and none was broken.
- **To finish:** run `extension_batch6_train.py`, which skips finished models (about 17 minutes of GPU). Then run `extension_batch6_score.py` on the CPU, with argument A, B or C for one part.
- **The predictions above stand as written.**

**Note added after the second audit.** The reason given for the reference's seeds 3–9 says `nosig0` "passed"; it passed section 2's development leg for the model, not the rule (13.6). Three trials the rule keeps for the model by similar margins under both readings were not retrained: `ep20` (+0.14 [+0.05, +0.23]), `drop03` (+0.17 [+0.06, +0.29]) and `win12` (+0.26 [+0.09, +0.44]). Section 2's promise of new seeds was applied to `nosig0` alone.
