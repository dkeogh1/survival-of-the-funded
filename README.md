# Survival of the Funded — reproduction

A reproducible Python rebuild of the empirical analysis in **"Survival of the Funded:
Econometric Analysis of Startup Longevity and Success"** (Daniel Keogh, *Journal of
Entrepreneurship, Management and Innovation*, vol. 17 issue 4, 2021; senior thesis ~2017).

The thesis studies how startup **financing strategy** interacts with founder **human/social
capital** and firm **competitive advantages** to predict three outcomes — **survival**,
**revenue**, and **employment** — using the [Kauffman Firm Survey](https://www.kauffman.org/entrepreneurship/research/kauffman-firm-survey/)
(KFS), an 8-wave panel (2004 baseline → 2011) of ~4,900 U.S. firms founded in 2004.

## Methods reproduced

- **Cox proportional hazards** model of firm survival (thesis Tables II–III)
- **LIML instrumental-variables** regressions of revenue & employment, with survival
  predicted from the Cox model used as the instrument to correct selection bias
  (thesis Tables IV–VII)
- The thesis's interaction terms (human capital × competitive advantage, and human
  capital × financing)

## Data

Uses the **freely downloadable KFS public-use file** (the thesis additionally may have
used the paid NORC confidential enclave for some continuous variables — see
`docs/RESULTS_COMPARISON.md`). The data file is ~171 MB and is **not** committed.

```bash
mkdir -p data/raw
curl -L -o data/raw/KFS8_PublicUse.dta \
  "https://kauffman-firm-survey.s3-us-west-2.amazonaws.com/KFS8---PublicUse-101413---STATA.dta"
```

## Quick start

```bash
uv sync                              # install dependencies (pandas, lifelines, linearmodels, ...)
uv run python src/run_all.py         # build dataset + fit all models
# or step by step:
uv run python src/build_dataset.py paper     # or: competing_risks
uv run python src/survival_cox.py
uv run python src/liml_models.py
```

Result tables are written to `output/tables/*.csv`.

## Repository layout

| Path | Purpose |
|---|---|
| `src/config.py` | KFS column names, wave structure, NAICS→industry map, interactions |
| `src/build_dataset.py` | Builds the firm-level analysis panel (survival + covariates) |
| `src/survival_cox.py` | Cox PH models (Tables II–III) |
| `src/liml_models.py` | LIML-IV revenue & employment models (Tables IV–VII) |
| `src/run_all.py` | End-to-end pipeline |
| `docs/VARIABLE_MAPPING.md` | Thesis variable → KFS column crosswalk + construction rules |
| `docs/RESULTS_COMPARISON.md` | Side-by-side of reproduced vs. published results |
| `docs/DanielKeogh_FinalThesis.pdf` | The thesis (reference) |

## Two survival definitions

`build_dataset.py` supports two codings of the (manufactured) survival variable:

- **`paper`** (default): terminal survey non-response counts as failure, matching the
  thesis text; only firms responding in the final wave are censored; interior response
  gaps are dropped.
- **`competing_risks`**: the KFS codebook convention — failure only on permanent
  out-of-business; non-response is right-censored.

## What reproduces, and what doesn't

Summary statistics, the survival/competitive-advantage/human-capital results, and the
revenue equation reproduce closely. The **financing** hazard ratios and several
interaction terms diverge from the published magnitudes — most likely due to data source
(public-use vs. confidential) and survival-variable construction details. The full
accounting is in [`docs/RESULTS_COMPARISON.md`](docs/RESULTS_COMPARISON.md).
