"""Competing-risks survival analysis (extension #2).

The original thesis hand-built a binary survival variable and *dropped* firms that
sold or merged. But M&A is often a success, not a failure, so lumping it with
censoring biases the failure hazards. Here we model the two events properly:

  - Cause-specific Cox for FAILURE (out of business), treating sold/merged as censored
  - Cause-specific Cox for EXIT via SALE/MERGER, treating failure as censored
  - Cumulative incidence functions (Aalen-Johansen) for each event

and compare the failure hazards to the original binary-survival model.

Run: uv run python src/competing_risks.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, AalenJohansenFitter

import config as C
import survival_cox as S

COVARS = S.MAIN   # same specification as the original Cox Model A, for comparability


def cause_specific(df: pd.DataFrame, cause: int, label: str) -> pd.DataFrame:
    d = df.copy()
    d["event"] = (d["event_type"] == cause).astype(int)   # other causes -> censored
    cols = ["duration", "event"] + COVARS
    d = d[cols].apply(pd.to_numeric, errors="coerce").dropna()
    nz = [c for c in COVARS if d[c].nunique() > 1]
    # small ridge penalty: the M&A event is rare (n=281) and separates on sparse
    # industry dummies, so an unpenalized fit fails to converge.
    cph = CoxPHFitter(penalizer=0.1).fit(d[["duration", "event"] + nz],
                                         "duration", "event")
    print(f"\n=== Cause-specific Cox: {label}  (N={len(d)}, events={int(d.event.sum())}, "
          f"C={cph.concordance_index_:.3f}) ===")
    out = pd.DataFrame({"hazard_ratio": np.exp(cph.summary["coef"]),
                        "z": cph.summary["z"], "p": cph.summary["p"]})
    out.index.name = "variable"
    return out.round(4)


def main() -> None:
    df = pd.read_parquet(C.PROCESSED / "extended.parquet")
    (C.OUTPUT / "tables").mkdir(parents=True, exist_ok=True)

    fail = cause_specific(df, 1, "FAILURE (out of business)")
    mna = cause_specific(df, 2, "EXIT via sale/merger")
    fail.to_csv(C.OUTPUT / "tables" / "cox_causespecific_failure.csv")
    mna.to_csv(C.OUTPUT / "tables" / "cox_causespecific_mna.csv")

    # cumulative incidence (Aalen-Johansen) for each event, overall
    print("\n=== Cumulative incidence at t=8 (Aalen-Johansen) ===")
    for cause, name in [(1, "failure"), (2, "sale/merger")]:
        ajf = AalenJohansenFitter(calculate_variance=False)
        ajf.fit(df["duration"], df["event_type"], event_of_interest=cause)
        cif = ajf.cumulative_density_.iloc[-1, 0]
        print(f"  P({name} by year 8) = {cif:.3f}")

    # how does treating M&A as a competing event (vs dropping it) move failure HRs?
    print("\n=== Failure hazards: competing-risks vs original (dropped M&A) ===")
    try:
        orig = pd.read_csv(C.OUTPUT / "tables" / "cox_A_main.csv").set_index("variable")
        keys = ["debtfin", "eqangels", "eqgovt", "eqvc", "eqfff",
                "univcompadv", "compcompadv", "patentcompadv",
                "foundered", "founderworkexp", "foundage"]
        cmp = pd.DataFrame({
            "HR_competing_risks": fail["hazard_ratio"].reindex(keys),
            "HR_original_binary": orig["hazard_ratio"].reindex(keys),
        })
        print(cmp.round(3).to_string())
    except FileNotFoundError:
        print("  (run survival_cox.py first to populate the original-model comparison)")


if __name__ == "__main__":
    main()
