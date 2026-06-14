"""LIML-IV models of revenue and employment (thesis Tables IV-VII).

The thesis corrects for selection bias by entering a `survived` term in the
revenue/employment equations and instrumenting it with the firm's Cox-predicted
survival, rather than restricting to survivors only. We follow that design with
linearmodels' IVLIML estimator.

  endogenous : survived (1 = firm alive in final wave)
  instrument : Cox predicted survival probability from the survival model (Model A)
  outcomes   : rev_last (revenue level, Appendix A 1-9), emp_last (total employment)
  controls   : lagged outcome, financing, human capital, competitive advantage,
               IP, demographics, industry, and the thesis interaction terms.

Run: uv run python src/liml_models.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from linearmodels.iv import IVLIML

import config as C
import survival_cox as S


def cox_instrument(df: pd.DataFrame) -> pd.Series:
    """Predicted survival probability S(t=8) from the main survival model, per firm.

    The thesis instruments the (endogenous) survival term with survival predicted
    from the Cox model. The *probability* (a nonlinear transform of the linear risk
    index via the baseline hazard) carries variation independent of the included
    linear controls, which the raw log partial hazard does not.
    """
    cols = ["duration", "event"] + S.MAIN
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    nz = [c for c in S.MAIN if d[c].nunique() > 1]
    cph = CoxPHFitter().fit(d[["duration", "event"] + nz], "duration", "event")
    surv = cph.predict_survival_function(d[nz])      # rows = times, cols = firms
    prob = surv.iloc[-1].to_numpy()                  # S at the last event time
    return pd.Series(prob, index=d.index, name="cox_survprob")


def run_liml(df: pd.DataFrame, outcome: str, lag: str, exog: list[str],
             label: str) -> pd.DataFrame:
    d = df.copy()
    d["survived"] = (d["event"] == 0).astype(float)
    d = d[[outcome, lag, "survived", "cox_survprob"] + exog].apply(
        pd.to_numeric, errors="coerce").dropna()
    exog_nz = [c for c in exog if d[c].nunique() > 1]

    dep = d[outcome]
    endog = d[["survived"]]
    instr = d[["cox_survprob"]]
    ex = d[[lag] + exog_nz].assign(const=1.0)

    mod = IVLIML(dep, ex, endog, instr).fit(cov_type="robust")
    print(f"\n=== {label}: N={d.shape[0]}, R2={mod.rsquared:.3f} ===")
    out = pd.DataFrame({"coef": mod.params, "se": mod.std_errors,
                        "z": mod.tstats, "p": mod.pvalues})
    out.index.name = "variable"
    return out.round(4)


def main() -> None:
    df = pd.read_parquet(C.PROCESSED / "analysis.parquet")
    (C.OUTPUT / "tables").mkdir(parents=True, exist_ok=True)

    df = df.join(cox_instrument(df), how="left")
    df = df.dropna(subset=["cox_survprob"])

    # thesis interaction terms
    S.add_interactions(df, ["ednet", "indnet", "netfwex_2",
                            "netang_si", "netvc_si", "netcomp_si"])

    base_exog = (S.FINANCING + S.COMPADV + S.HUMAN_CAP + S.IP
                 + S.DEMOG + S.INDUSTRY)
    inter = ["ednet", "indnet", "netfwex_2", "netang_si", "netvc_si", "netcomp_si"]

    rev_main = run_liml(df, "rev_last", "rev_first", base_exog, "Revenue (Table IV, main)")
    emp_main = run_liml(df, "emp_last", "emp_first", base_exog, "Employment (Table V, main)")
    rev_int = run_liml(df, "rev_last", "rev_first", base_exog + inter,
                       "Revenue (Tables VI, interactions)")
    emp_int = run_liml(df, "emp_last", "emp_first", base_exog + inter,
                       "Employment (Table VII, interactions)")

    for name, res in [("liml_revenue_main", rev_main), ("liml_employment_main", emp_main),
                      ("liml_revenue_interactions", rev_int),
                      ("liml_employment_interactions", emp_int)]:
        res.to_csv(C.OUTPUT / "tables" / f"{name}.csv")

    print("\n----- Revenue main coefficients vs. thesis -----")
    rtv = {"survived": 3.2, "eqangels": 0.55, "univcompadv": -0.44, "foundage": -0.05}
    print(rev_main.loc[[v for v in rtv if v in rev_main.index], ["coef", "z", "p"]]
          .assign(thesis=lambda x: [rtv[v] for v in x.index]).to_string())
    print("\n----- Employment main coefficients vs. thesis -----")
    etv = {"eqvc": -0.81, "eqgovt": 1.81}
    print(emp_main.loc[[v for v in etv if v in emp_main.index], ["coef", "z", "p"]]
          .assign(thesis=lambda x: [etv[v] for v in x.index]).to_string())


if __name__ == "__main__":
    main()
