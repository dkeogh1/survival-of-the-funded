"""Multiple imputation (MICE) for the missing financing data (extension #2).

The public-use equity questions (f3*) are ~34% missing. The reproduction (and, we
infer, the thesis) coded missing as 0 -- "did not receive" -- which conflates
"answered no" with "didn't answer" and biases the rare-event financing hazards.

Here we instead impute the missing *baseline* financing flags with IterativeImputer
(MICE), generate m completed datasets, fit a Cox model on each, and pool the
estimates with Rubin's rules. We compare the financing hazard ratios to the
missing-as-0 baseline on the same baseline-financing specification.

Run: uv run python src/imputation.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter

import config as C
import survival_cox as S

# baseline raw financing flags (retain NaN) -> model financing variables
RAW_FIN = {
    "eqangels": ["f3c_eq_invest_angels_0"],
    "eqgovt":   ["f3e_eq_invest_govt_0"],
    "eqvc":     ["f3f_eq_invest_vent_cap_0"],
    "eqfff":    ["f3a_eq_invest_spouse_0", "f3b_eq_invest_parents_0"],
    "debtfin":  ["f11a_bus_loans_bank_0", "f11a_bus_loans_nonbank_0",
                 "f11a_bus_loans_owner_0", "f11a_bus_loans_govt_0"],
}
FIN = list(RAW_FIN)
OTHER = [c for c in S.MAIN if c not in FIN]   # non-financing covariates (from parquet)
M = 10                                         # number of imputations


def load() -> pd.DataFrame:
    raw_cols = ["mprid"] + sorted({c for cc in RAW_FIN.values() for c in cc})
    raw, _ = pyreadstat.read_dta(str(C.RAW_DTA), usecols=raw_cols)
    for c in raw_cols[1:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    ext = pd.read_parquet(C.PROCESSED / "extended.parquet")
    keep = ["mprid", "duration", "event"] + OTHER
    return ext[keep].merge(raw, on="mprid")


def financing_from_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, cols in RAW_FIN.items():
        out[name] = (out[cols] >= 0.5).any(axis=1).astype(int)
    return out


def fit_cox(df: pd.DataFrame):
    cov = [c for c in FIN + OTHER if c in df.columns]
    d = df[["duration", "event"] + cov].apply(pd.to_numeric, errors="coerce").dropna()
    nz = [c for c in cov if d[c].nunique() > 1]
    cph = CoxPHFitter(penalizer=0.01).fit(d[["duration", "event"] + nz],
                                          "duration", "event")
    return cph.summary["coef"], cph.summary["se(coef)"] ** 2


def main() -> None:
    df = load()
    raw_fin_cols = sorted({c for cc in RAW_FIN.values() for c in cc})
    miss = df[raw_fin_cols].isna().mean().mean()
    print(f"baseline financing missingness: {miss:.1%} of cells\n")

    # --- baseline: missing-as-0 ---
    base = df.copy()
    base[raw_fin_cols] = base[raw_fin_cols].fillna(0)
    coef0, _ = fit_cox(financing_from_flags(base))

    # --- MICE: m imputations, pool with Rubin's rules ---
    impute_cols = raw_fin_cols + OTHER
    coefs, varis = [], []
    for m in range(M):
        imp = IterativeImputer(max_iter=10, sample_posterior=True, random_state=m)
        X = df[impute_cols].apply(pd.to_numeric, errors="coerce")
        filled = pd.DataFrame(imp.fit_transform(X), columns=impute_cols, index=df.index)
        dd = df.copy()
        dd[raw_fin_cols] = filled[raw_fin_cols]            # imputed continuous -> threshold
        dd[OTHER] = filled[OTHER]
        c, v = fit_cox(financing_from_flags(dd))
        coefs.append(c); varis.append(v)

    coefs = pd.concat(coefs, axis=1)
    varis = pd.concat(varis, axis=1)
    pooled = coefs.mean(axis=1)
    W = varis.mean(axis=1)                                  # within-imputation var
    Bv = coefs.var(axis=1, ddof=1)                          # between-imputation var
    T = W + (1 + 1 / M) * Bv                                # total variance (Rubin)
    z = pooled / np.sqrt(T)

    print(f"=== Financing hazard ratios: missing-as-0  vs  MICE-pooled (m={M}) ===\n")
    res = pd.DataFrame({
        "HR_missing_as_0": np.exp(coef0.reindex(FIN)),
        "HR_MICE_pooled":  np.exp(pooled.reindex(FIN)),
        "z_MICE":          z.reindex(FIN),
    }).round(3)
    print(res.to_string())
    (C.OUTPUT / "tables").mkdir(parents=True, exist_ok=True)
    res.to_csv(C.OUTPUT / "tables" / "mice_financing.csv")


if __name__ == "__main__":
    main()
