# Extensions: competing risks, multiple imputation, team capital, ML interactions

Two methodological extensions beyond the reproduction, both on the **same public-use KFS
data**. Built on `data/processed/extended.parquet` (competing-risks event types + team
features). Reproduce with `uv run python src/run_extensions.py`.

## Extension 2a — Competing risks (don't drop the acquisitions)

The thesis built a binary survival variable and **dropped firms that sold or merged**. But
acquisition is often a *success*; lumping it with censoring biases the failure hazards. We
instead model events properly: cause-specific Cox for failure, a separate one for M&A, and
Aalen–Johansen cumulative incidence.

Cumulative incidence by year 8: **P(failure) = 0.448, P(sale/merger) = 0.063.**

Failure hazard ratios — competing-risks vs. the original (M&A-dropped) model:

| Variable | Competing-risks HR | Original binary HR |
|---|---|---|
| debtfin | 0.68 | 0.68 |
| univcompadv | 0.57 | 0.57 |
| compcompadv | 0.42 | 0.35 |
| patentcompadv | 0.56 | 0.55 |
| foundered / workexp / age | ~0.98–0.99 | ~0.98 |
| **eqangels** | **1.05** | **1.31** |
| **eqvc** | **1.10** | **1.85** |
| **eqgovt** | **0.78** | **1.10** |

**Finding:** the human-capital and competitive-advantage hazards are unchanged, but the
**equity-financing** hazards move sharply toward neutral once M&A is modeled as its own
event. Equity-funded firms (angels, VC, government) disproportionately exit via
*acquisition*; dropping those firms (as the thesis did) left their failures behind and made
equity look artificially harmful. This is a concrete way the original's data-handling choice
biased exactly the coefficients that proved hardest to reproduce.

## Extension 2b — Multiple imputation for missing financing

The baseline equity questions are ~23% missing; the reproduction (and, inferred, the
thesis) coded missing as 0. We instead impute with MICE (`IterativeImputer`, m=10) and pool
with Rubin's rules.

| Financing | HR missing-as-0 | HR MICE-pooled |
|---|---|---|
| eqangels | 1.09 | 1.09 |
| eqgovt | 0.69 | 0.72 |
| eqvc | 1.00 | 0.95 |
| eqfff | 1.27 | 1.27 |
| debtfin | 0.85 | 0.84 |

**Finding:** proper imputation yields essentially identical estimates to missing-as-0. So
**missingness was *not* the driver** of the financing discrepancy — which, combined with 2a,
narrows the original problem to the M&A-dropping and panel-construction choices, not the
missing-data coding.

## Extension 4a — Team-level founder capital

The thesis used only the **primary** founder. We aggregated human capital over *all*
founders (team size, max/mean education and experience, gender/racial mix, etc.) and tested
whether the team adds predictive signal.

Out-of-sample concordance (failure):

| Feature set | Cox (linear) | Random Survival Forest | Gradient-Boosted Cox |
|---|---|---|---|
| primary-founder HC | 0.684 | 0.692 | 0.696 |
| + team HC | 0.686 | 0.692 | 0.697 |

**Finding (a null worth reporting):** team-level features add almost nothing beyond the
primary founder (+0.001–0.002 C). Of the team features, only mean team work-experience has
non-trivial permutation importance. The primary founder's characteristics capture nearly all
the founder-side signal in this data.

## Extension 4b — Data-driven interaction discovery (Friedman's H-statistic)

The thesis hand-picked its interaction terms. We ranked pairwise interaction strength with
Friedman's H² on the gradient-boosted survival model (background sample n=150).

Strongest interactions the data favours:

| Interaction | H² |
|---|---|
| team work-exp × primary-founder work-exp | 0.068 |
| university × company partnership | 0.048 |
| company partnership × patent advantage | 0.025 |
| debt × company partnership | 0.020 |

Where the thesis's hand-picked interactions land (of 30 evaluated):

| Thesis term | Definition | Rank | H² |
|---|---|---|---|
| `netfwex_2` | work-exp × company partnership | 6 / 30 | 0.010 |
| `ednet` | education × university partnership | 27 / 30 | ~0.000 |
| `indnet` | relevant-experience × company partnership | 29 / 30 | ~0.000 |

**Finding:** only one of the thesis's three interactions (`netfwex_2`) shows meaningful
interaction strength in a flexible model. The two others — including **`indnet`, which the
thesis reported as one of its strongest results** (HR 0.15, "85% less likely to fail") —
show essentially *zero* interaction in the GBM. The data instead favours different
combinations: **stacked competitive advantages** (university+company, company+patent) and
**team-vs-primary experience**. The broad thesis ("interactions/networks matter, especially
around competitive advantage and experience") survives; the *specific* hand-picked terms
largely do not, and ML surfaces better-supported ones.

## Caveats

- All interactions are weak in absolute terms (H² < 0.07); the GBM only modestly beats Cox
  (C 0.696 vs 0.684), so nonlinearity/interactions are real but secondary to main effects.
- Competing-risks cause-specific hazards answer a different question than the original
  all-cause model; the M&A model uses a ridge penalty (rare event, n≈281).
- H-statistic is computed on one GBM fit with a 150-row background; ranks are indicative,
  not inferential.
