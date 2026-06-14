# Variable Mapping: Thesis → Kauffman Firm Survey (KFS) Public-Use File

Source data: `KFS8_PublicUse.dta` (KFS 8-wave public-use microdata, baseline 2004 → 7th
follow-up 2011). 4,928 firms × 4,492 columns. Variables are stored **wide**, one column per
wave with suffix `_0` (baseline) … `_7` (7th follow-up). Owner-level variables carry an
additional `_owner_0N` index; the **primary founder is `owner_01`**.

The thesis glossary names are on the left; KFS columns on the right.

## Outcome / dependent variables

| Thesis var | Meaning | KFS column(s) | Construction |
|---|---|---|---|
| (survival) | Firm still in business | `final_status_code_0.._7`, `a10_out_of_business_1.._7` | See "Survival construction" below |
| `lrevenues` | Revenue level (1–9 scale, Appendix A) | `tot_revenue_r_0.._7` | Categorical $ level; lagged within firm |
| `lemp` | Total employment | `c5_num_employees_0.._7` | Lagged within firm |

## Predictors — Human & Social Capital (primary founder, `owner_01`)

| Thesis var | Meaning | KFS column (baseline) |
|---|---|---|
| `foundered` | Education level (1–10, Appendix B) | `g9_education_owner_01_0` |
| `founderworkexp` | Years of professional experience | `g2_work_exp_owner_01_0` |
| `founderexp` | # businesses previously founded | `g3a_oth_bus_owner_01_0` *(dropped in final model, p=0.95)* |
| `founderexpsameind` | Founded a business in a relevant industry | `g3b_bus_same_ind_owner_01_0` (NaN→0) |

## Predictors — Venture Financing (binary; "ever received over observed waves")

| Thesis var | Meaning | KFS column(s) |
|---|---|---|
| `eqangels` | Angel equity | `f3c_eq_invest_angels_*` |
| `eqcompanies` | Company equity | `f3d_eq_invest_companies_*` |
| `eqvc` | Venture-capital equity | `f3f_eq_invest_vent_cap_*` |
| `eqgovt` | Government equity | `f3e_eq_invest_govt_*` |
| `eqfff` | Friends/Family/Fools | `f3a_eq_invest_spouse_*`, `f3b_eq_invest_parents_*`, family loans `f11a_bus_loans_fam_*`, `f7a_pers_loan_fam_*`, `f9a_pers_loan_fam_*` |
| `debtfin` | Any debt financing | any `f11a_bus_loans_*` (business loans) OR personal loans `f7a_*`/`f9a_*` taken for the business |

## Predictors — Competitive Advantage (binary)

Subtype channels only exist from 2007 (wave 3). Thesis back-fills 2004–2006 if a 2007
channel is reported and a competitive advantage (`d2_comp_advantage`) was claimed then.

| Thesis var | Meaning | KFS column(s) |
|---|---|---|
| `univcompadv` | University partnership | `d2a_compadv_univ_reason_3.._7` |
| `compcompadv` | Company partnership | `d2a_compadv_comp_reason_3.._7` |
| `patentcompadv` | Patent advantage | `d2a_compadv_patents_reason_3.._7` |
| `govlabcompadv` | Govt lab / research center | `d2a_compadv_govlab_reason_3.._7` |
| (overall flag) | Has any competitive advantage | `d2_comp_advantage_0.._7` |

## Predictors — Intellectual Property (counts)

| Thesis var | KFS column |
|---|---|
| `totcr` | `total_copyrights_0` |
| `tottm` | `total_trademarks_0` |
| `totpatents` | `total_patents_0` |

## Controls — Founder demographics (primary founder, baseline)

| Thesis var | KFS column | Notes |
|---|---|---|
| `foundhisp` | `g5_hisp_origin_owner_01_0` | |
| `foundamind` | `g6_race_amind_owner_01_0` | |
| `foundasian` | `g6_race_asian_owner_01_0` | |
| `foundblack` | `g6_race_black_owner_01_0` | |
| `foundwhite` | `g6_race_white_owner_01_0` | base = "other" |
| `foundmale` | `g10_gender_owner_01_0` | (1=male) — individually insignificant, p=0.876 |
| `foundage` | `age_owner_01_r_0` | age level 1–7 (Appendix C) |

## Controls — Industry (NAICS 2-digit → 14 dummies; base = Other Services / NAICS 81)

| Thesis var | NAICS 2-digit |
|---|---|
| `mining` | 21 |
| `ut` | 22 |
| `con` | 23 |
| `manu` | 31, 32, 33 |
| `tnw` | 48, 49 |
| `inf` | 51 |
| `finser` | 52 |
| `re` | 53 |
| `profser` | 54 |
| `management` | 55 |
| `wm` | 56 |
| `eduser` | 61 |
| `rec` | 71 |
| `food` | 72 |
| `hightech` | `hightech_0` (KFS-derived flag, not NAICS) |

NAICS via `naics_code_0` (stored as string). Sectors not in the 14 named buckets
(11 agriculture, 42 wholesale, 44/45 retail, 62 health care, 81 other services, 92 public
admin) fall into the reference group. **Interpretation point** — the thesis names "other
services" (81) as the base but does not enumerate where 42/44/45/62 go; we fold all
unnamed sectors into the reference and flag this.

## Interaction terms

| Thesis var | Definition |
|---|---|
| `ednet` | `foundered` × `univcompadv` |
| `indnet` | `founderexpsameind` × `compcompadv` |
| `netfwex_2` | `founderworkexp` × `compcompadv` |
| `netang_si` | `founderexpsameind` × `eqangels` |
| `netvc_si` | `founderexpsameind` × `eqvc` |
| `netcomp_si` | `founderexpsameind` × `eqcompanies` |

## Survival construction (from KFS codebook Ch. 2 & 5)

For each firm, walk waves 1→7 using `final_status_code_w`:

- `10` / `30` = completed the survey that wave → **alive / at risk**.
- `463` = "No Longer in Business"; `431` = "Temporarily Stopped". At that wave, read
  `a10_out_of_business_w`: `1`=Sold, `2`=Merged, `3`/`4`=Out of Business, `5`=Temp stopped.
- `200/209/210/219/220/229/330/401/430/590` = unit non-response (dropout).
- `465` = started before 2004 (ineligible) and `468` = duplicate → **drop firm**.

Coding:
- **event = 1 (failure)** at the first wave the firm is permanently Out of Business
  (`463`/`431` with `a10` ∈ {3,4}, or `463` with missing `a10`). `duration` = that wave #.
- **Sold/Merged** (`a10` ∈ {1,2}) → **dropped** (thesis removes merged/acquired firms).
- **Censored (event = 0)**: firm completes through wave 7, or drops out — `duration` = last
  wave observed alive. (Thesis alternative: consecutive non-response = non-survival; offered
  as a sensitivity option, not the default.)

Financing & competitive-advantage indicators are "ever observed across waves"; IP counts,
human capital and demographics are taken at **baseline (wave 0)**.
