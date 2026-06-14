"""Experiment: does a data-cleaning / sample filter reproduce the thesis financing HRs?

Hypothesis (author): the thesis dropped samples during cleaning, shrinking 4,928 -> ~3,768
and changing the financing coefficients. We apply several plausible filters, then report
sample size, angel/government counts, and Cox financing HRs for each.

Run: uv run python src/experiments/cleaning_filters.py
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

RAW_F3 = ["f3c_eq_invest_angels_0", "f3d_eq_invest_companies_0", "f3e_eq_invest_govt_0",
          "f3f_eq_invest_vent_cap_0", "f3a_eq_invest_spouse_0", "f3b_eq_invest_parents_0"]
RAW_HC = ["g9_education_owner_01_0", "g2_work_exp_owner_01_0", "age_owner_01_r_0"]
THESIS_HR = {"eqangels": 0.28, "eqfff": 0.55, "debtfin": 0.13, "eqgovt": 3.62, "eqvc": 1.03}


def fit(df: pd.DataFrame) -> tuple[pd.Series, int, int]:
    covars = [c for c in S.MAIN if c in df.columns]
    d = df[["duration", "event"] + covars].apply(pd.to_numeric, errors="coerce").dropna()
    nz = [c for c in covars if d[c].nunique() > 1]
    cph = CoxPHFitter().fit(d[["duration", "event"] + nz], "duration", "event")
    return np.exp(cph.summary["coef"]), d.shape[0], int(d.event.sum())


def report(label, df, raw):
    hr, n, ev = fit(df)
    m = df.merge(raw[["mprid"]], on="mprid")
    ang = int(m["eqangels"].sum()); gov = int(m["eqgovt"].sum())
    fin = {v: round(float(hr.get(v, np.nan)), 2) for v in THESIS_HR}
    print(f"### {label}")
    print(f"    N(model)={n}  events={ev}  angel_n={ang}  govt_n={gov}")
    print(f"    HR: {fin}\n")


def main():
    raw, _ = pyreadstat.read_dta(str(C.RAW_DTA), usecols=["mprid"] + RAW_F3 + RAW_HC)
    analysis = pd.read_parquet(C.PROCESSED / "analysis.parquet")
    df = analysis.merge(raw, on="mprid")

    print(f"thesis HR target: {THESIS_HR}")
    print(f"thesis sample: ~3,768 analyzed (angel 69, govt 18)\n")

    report("F0: no extra filter (full analysis sample)", df, raw)

    # F1: complete cases on the raw equity section (drop firms not asked / missing)
    f1 = df[df[RAW_F3].notna().all(axis=1)]
    report("F1: complete raw equity section (f3 non-missing)", f1, raw)

    # F2: complete cases on equity + human capital raw inputs
    f2 = df[df[RAW_F3 + RAW_HC].notna().all(axis=1)]
    report("F2: complete equity + human-capital raw inputs", f2, raw)

    # F3: firms that ever reported revenue (a real operating record)
    f3 = df[pd.to_numeric(df["rev_first"], errors="coerce").notna()]
    report("F3: has a reported revenue level", f3, raw)

    # F4: drop firms that received equity but with missing flags -> keep only firms where
    # the equity question was answered at baseline (f3c non-missing)
    f4 = df[df["f3c_eq_invest_angels_0"].notna()]
    report("F4: equity question answered at baseline", f4, raw)

    # F5: intersection F2 + F3 (strictest "clean" sample)
    f5 = df[df[RAW_F3 + RAW_HC].notna().all(axis=1)
            & pd.to_numeric(df["rev_first"], errors="coerce").notna()]
    report("F5: complete inputs AND reported revenue", f5, raw)


if __name__ == "__main__":
    main()
