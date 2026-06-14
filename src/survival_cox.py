"""Cox proportional-hazards models of startup survival (thesis Tables II & III).

Fits three specifications and writes hazard-ratio tables to output/tables/.
  A: main effects only (non-interactive)
  B: + competitive-advantage x human-capital interactions (ednet, indnet, netfwex_2)
  C: + financing x relevant-experience interactions (netang_si, netvc_si, netcomp_si)

Run: uv run python src/survival_cox.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import config as C

# Covariate groups for the final survival model (founderexp dropped per thesis, p=0.95)
HUMAN_CAP = ["foundered", "founderexpsameind", "founderworkexp"]
FINANCING = ["eqfff", "eqgovt", "eqangels", "eqvc", "debtfin"]
COMPADV = ["univcompadv", "compcompadv", "patentcompadv", "govlabcompadv"]
IP = ["totcr", "tottm", "totpatents"]
DEMOG = ["foundage", "foundhisp", "foundamind", "foundasian", "foundblack",
         "foundwhite", "foundmale"]
INDUSTRY = C.INDUSTRY_DUMMIES + ["hightech"]

MAIN = HUMAN_CAP + FINANCING + COMPADV + IP + DEMOG + INDUSTRY


def add_interactions(df: pd.DataFrame, names: list[str]) -> list[str]:
    for name in names:
        a, b = C.INTERACTIONS[name]
        df[name] = df[a] * df[b]
    return names


def fit(df: pd.DataFrame, covars: list[str], label: str) -> pd.DataFrame:
    cols = ["duration", "event"] + covars
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    # drop zero-variance columns (e.g. an industry with no firms in the estimation set)
    nz = [c for c in covars if d[c].nunique() > 1]
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(d[["duration", "event"] + nz], duration_col="duration", event_col="event")
    print(f"\n=== Model {label}: N={d.shape[0]}, events={int(d.event.sum())}, "
          f"concordance={cph.concordance_index_:.3f} ===")
    s = cph.summary
    out = pd.DataFrame({
        "hazard_ratio": np.exp(s["coef"]),
        "coef": s["coef"],
        "se": s["se(coef)"],
        "z": s["z"],
        "p": s["p"],
    })
    out.index.name = "variable"
    return out.round(4)


def main() -> None:
    df = pd.read_parquet(C.PROCESSED / "analysis.parquet")
    (C.OUTPUT / "tables").mkdir(parents=True, exist_ok=True)

    res_a = fit(df, MAIN, "A (main effects)")

    df_b = df.copy()
    bvars = add_interactions(df_b, ["ednet", "indnet", "netfwex_2"])
    res_b = fit(df_b, MAIN + bvars, "B (+ compadv x human capital)")

    df_c = df.copy()
    cvars = add_interactions(df_c, ["netang_si", "netvc_si", "netcomp_si"])
    res_c = fit(df_c, MAIN + cvars, "C (+ financing x relevant experience)")

    for name, res in [("cox_A_main", res_a), ("cox_B_compadv_x_hc", res_b),
                      ("cox_C_financing_x_exp", res_c)]:
        res.to_csv(C.OUTPUT / "tables" / f"{name}.csv")

    print("\n----- Model A hazard ratios (key variables vs. thesis) -----")
    targets = {"eqangels": 0.28, "debtfin": 0.13, "eqvc": 1.03, "eqgovt": 3.62,
               "eqfff": 0.55, "univcompadv": 0.68, "compcompadv": 0.175,
               "patentcompadv": 0.51, "foundered": 0.977, "foundage": 0.96,
               "tottm": 0.63, "foundhisp": 1.29}
    show = res_a.loc[[v for v in targets if v in res_a.index],
                     ["hazard_ratio", "z", "p"]].copy()
    show["thesis_HR"] = [targets[v] for v in show.index]
    print(show.to_string())
    if "indnet" in res_b.index:
        print(f"\nindnet (relevant-exp x company partnership) HR = "
              f"{res_b.loc['indnet','hazard_ratio']:.3f}  (thesis ~0.15)")


if __name__ == "__main__":
    main()
