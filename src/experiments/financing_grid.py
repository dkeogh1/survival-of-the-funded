"""Experiment: how does the financing-variable construction move the Cox HRs?

The thesis reports binary financing variables and specific first-year (2004) counts:
    angel 69, company 44, govt 18, VC 20, FFF 129, debt ~2011.
and dramatic survival hazard ratios:
    eqangels 0.28, eqfff 0.55, debtfin 0.13, eqgovt 3.62, eqvc 1.03.

This script swaps in several candidate financing constructions (window x FFF x debt),
re-uses the survival outcome + non-financing covariates from analysis.parquet, re-fits
the main Cox model, and prints baseline counts and financing HRs side by side so we can
see which definition best reproduces the published numbers.

Run: uv run python src/experiments/financing_grid.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import survival_cox as S

WAVES = C.WAVES
THESIS_COUNT = {"eqangels": 69, "eqcompanies": 44, "eqgovt": 18, "eqvc": 20,
                "eqfff": 129, "debtfin": 2011}
THESIS_HR = {"eqangels": 0.28, "eqfff": 0.55, "debtfin": 0.13,
             "eqgovt": 3.62, "eqvc": 1.03, "eqcompanies": None}

BUS_LOAN = ["f11a_bus_loans_bank", "f11a_bus_loans_nonbank", "f11a_bus_loans_owner",
            "f11a_bus_loans_fam", "f11a_bus_loans_govt", "f11a_bus_loans_emp",
            "f11a_bus_loans_other_bus", "f11a_bus_loans_other_ind"]
PERS_LOAN = ["f7a_pers_loan_bank", "f7a_pers_loan_fam", "f7a_pers_loan_other",
             "f9a_pers_loan_bank", "f9a_pers_loan_fam", "f9a_pers_loan_other"]
EQUITY = {"eqangels": "f3c_eq_invest_angels", "eqcompanies": "f3d_eq_invest_companies",
          "eqgovt": "f3e_eq_invest_govt", "eqvc": "f3f_eq_invest_vent_cap"}
FFF_EQUITY = ["f3a_eq_invest_spouse", "f3b_eq_invest_parents"]


def _load_raw() -> pd.DataFrame:
    meta = pyreadstat.read_dta(str(C.RAW_DTA), metadataonly=True)[1]
    have = set(meta.column_names)
    stems = list(EQUITY.values()) + FFF_EQUITY + BUS_LOAN + PERS_LOAN
    cols = ["mprid"]
    for s in stems:
        cols += [f"{s}_{w}" for w in WAVES if f"{s}_{w}" in have]
    cols += [f"tot_debt_bus_r_{w}" for w in WAVES if f"tot_debt_bus_r_{w}" in have]
    cols += [f"tot_debt_r_{w}" for w in WAVES if f"tot_debt_r_{w}" in have]
    return pyreadstat.read_dta(str(C.RAW_DTA), usecols=list(dict.fromkeys(cols)))[0]


def _ind(raw, stem, waves, how="one"):
    cols = [f"{stem}_{w}" for w in waves if f"{stem}_{w}" in raw.columns]
    if not cols:
        return pd.Series(0, index=raw.index)
    vals = raw[cols].apply(pd.to_numeric, errors="coerce")
    hit = vals.eq(1) if how == "one" else vals.gt(0)
    return hit.any(axis=1).astype(int)


def _any(series_list):
    out = series_list[0]
    for s in series_list[1:]:
        out = out | s
    return out.astype(int)


def build_scheme(raw: pd.DataFrame, window: str, fff: str, debt: str) -> pd.DataFrame:
    waves = [0] if window == "baseline" else WAVES
    out = pd.DataFrame({"mprid": raw["mprid"]})
    for name, stem in EQUITY.items():
        out[name] = _ind(raw, stem, waves)
    if fff == "equity_only":
        out["eqfff"] = _any([_ind(raw, s, waves) for s in FFF_EQUITY])
    else:  # equity + family loans
        out["eqfff"] = _any([_ind(raw, s, waves) for s in FFF_EQUITY]
                            + [_ind(raw, "f11a_bus_loans_fam", waves),
                               _ind(raw, "f7a_pers_loan_fam", waves),
                               _ind(raw, "f9a_pers_loan_fam", waves)])
    if debt == "busloans":
        out["debtfin"] = _any([_ind(raw, s, waves) for s in BUS_LOAN + PERS_LOAN])
    elif debt == "tot_debt_bus":
        out["debtfin"] = _ind(raw, "tot_debt_bus_r", waves, how="pos")
    else:  # tot_debt (broadest: any outstanding debt incl. owner/personal)
        out["debtfin"] = _ind(raw, "tot_debt_r", waves, how="pos")
    return out


def fit_financing(analysis: pd.DataFrame, fin: pd.DataFrame) -> tuple[pd.Series, int]:
    base = analysis.drop(columns=[c for c in fin.columns if c != "mprid"], errors="ignore")
    df = base.merge(fin, on="mprid")
    covars = [c for c in S.MAIN if c in df.columns]
    d = df[["duration", "event"] + covars].apply(pd.to_numeric, errors="coerce").dropna()
    nz = [c for c in covars if d[c].nunique() > 1]
    cph = CoxPHFitter().fit(d[["duration", "event"] + nz], "duration", "event")
    return np.exp(cph.summary["coef"]), d.shape[0]


def main() -> None:
    raw = _load_raw()
    analysis = pd.read_parquet(C.PROCESSED / "analysis.parquet")

    schemes = {
        "current(ever, FFF+loans, busloans)": ("ever", "equity_loans", "busloans"),
        "ever, FFF=equity, busloans":          ("ever", "equity_only", "busloans"),
        "ever, FFF=equity, tot_debt":          ("ever", "equity_only", "tot_debt"),
        "baseline, FFF=equity, tot_debt_bus":  ("baseline", "equity_only", "tot_debt_bus"),
        "baseline, FFF=equity, tot_debt":      ("baseline", "equity_only", "tot_debt"),
        "baseline, FFF=equity, busloans":      ("baseline", "equity_only", "busloans"),
    }

    fin_vars = ["eqangels", "eqcompanies", "eqgovt", "eqvc", "eqfff", "debtfin"]
    print(f"\nTHESIS first-year counts: {THESIS_COUNT}")
    print(f"THESIS hazard ratios    : "
          f"{ {k:v for k,v in THESIS_HR.items() if v} }\n")

    for label, (window, fff, debt) in schemes.items():
        fin = build_scheme(raw, window, fff, debt)
        merged = fin.merge(analysis[["mprid"]], on="mprid")  # restrict to analysis sample
        counts = {v: int(merged[v].sum()) for v in fin_vars}
        hr, n = fit_financing(analysis, fin)
        print(f"### {label}   (N={n})")
        print("    counts :", {v: counts[v] for v in fin_vars})
        print("    HR     :", {v: round(float(hr.get(v, np.nan)), 3) for v in fin_vars})
        print()


if __name__ == "__main__":
    main()
