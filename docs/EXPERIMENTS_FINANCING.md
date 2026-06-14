# Experiment log: closing the financing-coefficient gap

**Question.** The reproduced Cox survival model matches the thesis on human capital and
competitive advantage, but the **financing** hazard ratios diverge (thesis: angels 0.28,
FFF 0.55, debt 0.13, govt 3.62, VC 1.03). The thesis states all financing variables are
**binary**, so the lever must be *which* variables / *which* measurement window — or the
data source. These experiments isolate it.

Reproduce with: `uv run python src/experiments/financing_grid.py`

## Target (thesis first-year 2004 counts)

| | angel | company | govt | VC | FFF | debt |
|---|---|---|---|---|---|---|
| Thesis | 69 | 44 | 18 | 20 | 129 | 2,011 |
| Public-use (non-imputed) | 163 | 69 | 31 | 31 | 231 | ~1,194 |
| Logically-imputed public | 152 | 66 | 25 | 40 | 185 | — |

The raw flags are ~2× the thesis in **both** public files — a difference imputation does
not close. This is the first sign the gap is in the data, not the recipe.

## Experiment 1 — financing construction grid (6 definitions)

Varied measurement window (baseline-only vs ever-over-panel), FFF (spouse+parents equity
only vs +family loans), and debt (business loans vs `tot_debt_bus_r` vs `tot_debt_r`).
Re-fit the main Cox model each time. **Result: financing HRs barely move.**

| Construction | angel | FFF | debt | govt | VC |
|---|---|---|---|---|---|
| current (ever, FFF+loans) | 1.26 | 1.03 | 0.67 | 1.08 | 1.83 |
| ever, FFF=equity, busloans | 1.27 | 1.00 | 0.68 | 1.07 | 1.83 |
| ever, FFF=equity, tot_debt | 1.25 | 1.00 | 0.54 | 1.11 | 1.78 |
| baseline, FFF=equity, tot_debt_bus | 1.23 | 1.24 | 0.96 | 1.08 | 1.52 |
| baseline, FFF=equity, busloans | 1.23 | 1.24 | 0.98 | 1.06 | 1.51 |
| **thesis** | **0.28** | **0.55** | **0.13** | **3.62** | **1.03** |

Cleaning FFF (dropping family loans) and broadening debt nudge those two toward the
thesis, but **angels stay ~1.25 (harmful) and government stays ~1.1 — never close.**

## Experiment 2 — survival definition

Re-ran the grid under the `competing_risks` survival coding. Financing HRs shift but
still miss: government flips to **protective** (0.63–0.72) — the *opposite* of the thesis's
3.62 — and angels stay harmful (~1.15).

## Experiment 3 — unconditional Kaplan–Meier (model-free)

| | 8-yr survival if funded | if not | univariate HR | thesis HR |
|---|---|---|---|---|
| angel | 44% | 42% | 0.92 (≈neutral) | 0.28 |
| VC | 35% | 42% | 1.23 (harmful) | 1.03 |
| **govt** | 47% | 42% | **0.79 (protective)** | **3.62 (harmful)** |
| FFF | 47% | 40% | 0.80 (protective) | 0.55 |
| debt | 50% | 36% | 0.64 (protective) | 0.13 |

Even with **no covariates and no modeling choices**, government-funded firms survive
*better* in the public-use data — the reverse of the thesis. Debt and FFF are protective
in the right direction but far weaker.

## Experiment 4 — sample-cleaning filters

Tested the hypothesis that the thesis dropped samples during cleaning
(`src/experiments/cleaning_filters.py`). Applied complete-case filters on the raw equity
section, on equity + human capital, on having a reported revenue, etc.

| Filter | N (model) | angel n | govt n | angel HR | govt HR |
|---|---|---|---|---|---|
| none | 4,521 | 292 | 59 | 1.31 | 1.10 |
| complete equity section | 2,956 | 280 | 59 | 1.28 | 1.07 |
| complete equity + human capital | 2,956 | 279 | 57 | 1.28 | 1.07 |
| + reported revenue | 2,934 | 278 | 56 | 1.29 | 1.05 |
| **thesis** | **~3,768** | **69** | **18** | **0.28** | **3.62** |

Requiring complete equity responses **does** shrink the sample toward the thesis's ~3,768
— so cleaning plausibly explains the *sample size*. But the financing **HRs do not move**
(angel stays ~1.28, government ~1.07) and the financing counts stay ~4× the thesis. Sample
cleaning is not the lever for the coefficient gap.

## Experiment 5 — time-varying financing (long counting-process panel)

The thesis built a firm-YEAR panel in SAS. Tested whether time-varying financing
(`src/experiments/timevarying_cox.py`) flips the signs — a firm taking government money
just before failing looks harmful contemporaneously but protective as an ever-flag.

| Timing | angel | FFF | debt | govt | VC |
|---|---|---|---|---|---|
| contemporaneous | 1.05 | 1.30 | 0.72 | **0.75** | 2.30 |
| cumulative | 1.37 | 1.21 | 0.90 | **1.15** | 1.80 |
| thesis | 0.28 | 0.55 | 0.13 | **3.62** | 1.03 |

Government stays protective/neutral; no flip to the published 3.62.

## Experiment 6 — alternative "government funding" variables

| Definition (ever) | n | univariate HR |
|---|---|---|
| govt equity `f3e` | 59 | 0.79 |
| govt loan `f11a_bus_loans_govt` | 92 | 0.65 |
| govt equity OR loan | 135 | 0.74 |
| thesis | — | **3.62 (harmful)** |

*Every* government-funding variable in the public file is protective. There is no public
variable under which government funding predicts failure.

## Government: the sharpest discriminator (all specifications)

| Specification | govt HR |
|---|---|
| firm-level, ever | 1.10 |
| firm-level, baseline | 1.08 |
| competing-risks survival | 0.63–0.72 |
| cleaning filters | 1.05–1.10 |
| time-varying contemporaneous | 0.75 |
| time-varying cumulative | 1.15 |
| unconditional Kaplan–Meier | 0.79 |
| alternative govt variables | 0.65–0.79 |
| **thesis** | **3.62** |

## Conclusion

The financing gap is **not** explained by any of: financing-variable construction,
measurement window, FFF/debt definition, imputation, survival-variable coding, sample
cleaning, time-varying vs. fixed panel structure, or the choice of government variable.
In the public-use data, government funding is **protective or neutral under all eight
specifications** — the opposite of the thesis's 3.62 — and angels are neutral-to-harmful
vs. the thesis's protective 0.28. Debt and FFF reproduce in *direction* (protective) but
roughly half the magnitude.

**Data source — now confirmed.** The author located the exact `.dta` used for the thesis
(recovered from a Feb 2017 email backup, `Kaufman Firm Survey Data.dta`). Its MD5 is
**identical** to the public-use file used in this reproduction
(`3b92fb9e9848c2e399e13e3ba24f866e`). So the confidential-enclave hypothesis is **ruled
out** — the thesis used the same public-use data, byte for byte.

The original SAS/Stata code did **not** survive (the recovered archives contain the data,
the literature PDFs, and theory/lit-review notes, but no `.sas`/`.do` files). Given
identical data and that the published financing estimates are unreproducible under every
specification tried here, the financing results most plausibly stem from a
**variable-construction or data-preparation choice in the original (now-lost) SAS panel**
— the thesis describes downloading the non-panel file and manually transposing/matching by
firm id, which is error-prone for rare-category flags (government n≈18, VC n≈20). The
implausible magnitudes (government as the single "strongest predictor of failure," debt at
0.13) and the author's own caveats ("small sample size... grain of salt") fit fragile
estimates from a small-n, possibly-misaligned construction rather than a robust signal.

**Bottom line:** the paper's central contribution — human/social capital and competitive
advantage, and their interactions — reproduces on the public data. The financing
magnitudes do not, and (given public-use data) most plausibly reflect the original panel
build rather than a definition we can recover. Confirming this would require the original
SAS do-files.
