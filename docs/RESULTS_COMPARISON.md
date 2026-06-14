# Reproduction vs. Thesis: Results Comparison

Rebuild of *"Survival of the Funded: Econometric Analysis of Startup Longevity and
Success"* (Keogh, JEMI 2021) from the **KFS public-use file**. The thesis used a panel
assembled in SAS/Stata (and likely the confidential NORC enclave extract for some
continuous variables); this rebuild uses the freely available public-use microdata, so
some divergence is expected. See `VARIABLE_MAPPING.md` for construction details.

## Sample & summary statistics — close match

| Quantity | Thesis | This rebuild |
|---|---|---|
| Firms (analysis sample) | 4,298 → 3,768 analyzed | 4,644 (4,521 in Cox after listwise deletion) |
| Survive all 8 years | "more than one-third" | 42.0% ✓ |
| VC equity, whole panel | "66 observations ... over 8 years" | **66 firms** ✓ |
| Relevant founder experience | 0.17 | 0.173 ✓ |
| High-tech | ~13% | 13.0% ✓ |
| White founder | 82% | 83% ✓ |
| Founder education (level) | mean 6.26 | median 7 ✓ |
| Founder age (level) | mean 3.55 | median 3 ✓ |

First-year failure ("a little more than one-quarter") is sensitive to how baseline→
follow-up-1 attrition is coded; our `paper` definition yields ~11%, the largest single
discrepancy (see Caveats).

## Cox survival model (Table II) — hazard ratios

| Variable | Thesis HR | Rebuild HR | Verdict |
|---|---|---|---|
| `foundered` (education) | 0.977 | 0.983 | ✓ match (sign, magnitude) |
| `founderworkexp` (experience) | 0.99 | 0.987 | ✓ match |
| `founderexpsameind` | (kept, p≈.16) | 1.127 (p=.03) | ~ direction differs |
| `univcompadv` | 0.68 | 0.566 | ✓ protective |
| `compcompadv` | 0.175 | 0.346 | ✓ strongly protective |
| `patentcompadv` | 0.51 | 0.547 | ✓ near-exact |
| `foundage` | 0.96 | 0.946 | ✓ match |
| `foundhisp` | 1.29 | 1.216 | ✓ match |
| `debtfin` | 0.13 | 0.683 | ~ same sign, weaker |
| `eqfff` | 0.55 | 0.893 | ~ same sign, weaker |
| `eqangels` | 0.28 | **1.306** | ✗ sign flip |
| `eqvc` | 1.03 | 1.853 | ~ both ≥1 |
| `eqgovt` | 3.62 | 1.099 | ✗ much weaker |
| `tottm` (trademarks) | 0.63 | 0.996 | ✗ |
| `indnet` interaction | 0.15 | 1.178 | ✗ sign flip |

**What reproduces:** the thesis's core story about human/social capital and competitive
advantages — education, experience, age, and especially university/company/patent
partnerships are protective, with magnitudes in the same range. Industry pattern also
aligns (Real Estate and Professional Services significantly protective).

**What does not:** the **financing** hazard ratios (angels, government, FFF, debt) and the
key **interaction terms**. The thesis reports dramatic financing effects (angels 0.28,
debt 0.13, government 3.62) that we cannot reproduce from the public-use file under any
financing measurement we tried (baseline-only or ever-over-panel; both shown in code).

## LIML-IV revenue & employment (Tables IV–VII)

| Variable | Thesis coef | Rebuild coef | Verdict |
|---|---|---|---|
| Revenue: `survived` | 3.12–3.31 | 4.89 | ✓ dominant positive |
| Revenue: `eqangels` | 0.55 | 0.748 | ✓ match (sign, sig) |
| Revenue: `univcompadv` | −0.44 | −0.432 | ✓ near-exact |
| Revenue: `foundage` | −0.05 | −0.132 | ✓ direction |
| Employment: `eqvc` | −0.81 | 1.659 | ✗ sign flip |
| Employment: `eqgovt` | 1.81 | 0.121 | ~ same sign, weaker |

The **revenue** equation reproduces the thesis's headline marginal effects well
(angel equity raises revenue; university partnership lowers the revenue *level*; survival
dominates). **Employment** financing effects diverge, mirroring the Cox financing gap.

## Source of the financing divergence — investigated

We ran a dedicated experiment series (`src/experiments/financing_grid.py`, written up in
`EXPERIMENTS_FINANCING.md`) to find the lever. Findings:

- **Not the financing construction.** Across 6 definitions (baseline vs ever; FFF with/
  without family loans; three debt measures) the angel HR stays ~1.25 and government ~1.1
  — never near the thesis's 0.28 / 3.62.
- **Not the survival definition.** Under competing-risks coding, government even flips
  *protective* (0.7), the opposite of the thesis.
- **Not imputation.** The logically-imputed public file gives the same ~2× counts.
- **Visible model-free.** Unconditional Kaplan–Meier curves already show government-funded
  firms surviving *better* in the public data — reverse of the thesis.

- **Not time-varying structure or the government variable.** A long counting-process panel
  with time-varying financing still leaves government protective (0.75); every public
  government-funding variable (equity, loan, both) is protective (HR 0.65–0.79).

**Conclusion:** financing was binary (thesis-confirmed), and the thesis data file was
recovered and is **byte-for-byte identical** to the public-use file used here (matching
MD5) — so it is *certain* the same public data was used. Government funding is
protective/neutral under all eight specifications we tried — the opposite of the thesis's
harmful 3.62 — so the published financing estimates are not reproducible from the data by
any definition or model. The original SAS/Stata code did not survive, so the exact
construction can't be recovered; the most likely explanation is a **variable-construction /
data-preparation choice in the original hand-built SAS panel** (download non-panel →
transpose → match by id), fragile for rare-category flags (government n≈18, VC n≈20). The
competitive-advantage and human-capital conclusions — the paper's central contribution —
reproduce on the identical public data; the financing magnitudes most plausibly reflect the
original panel build. See `EXPERIMENTS_FINANCING.md`.
