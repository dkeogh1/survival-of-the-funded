"""Build the EXTENDED analysis dataset for the methodological extensions.

Adds two things the original lacked:
  1. Competing-risks outcome: event_type 0=censored, 1=failure (out of business),
     2=sold/merged -- instead of dropping M&A firms.
  2. Team-level human capital aggregated over ALL founders (the thesis used only
     the primary founder), plus the primary-founder covariates and everything else
     from build_dataset.

Writes data/processed/extended.parquet. Run: uv run python src/build_extended.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyreadstat

import config as C
import build_dataset as B

# owners present at baseline (1..10); later indices only appear in follow-up waves
TEAM_OWNERS = [f"{i:02d}" for i in range(1, 11)]
TEAM_VARS = {  # KFS stem (without _ownerNN_0) -> short name
    "g9_education_owner":  "educ",
    "g2_work_exp_owner":   "workexp",
    "g3b_bus_same_ind_owner": "sameind",
    "g10_gender_owner":    "gender",     # 1=male, 2=female
    "age_owner":           "age",        # NOTE actually age_owner_NN_r
    "g6_race_white_owner": "white",
}


def _team_cols() -> list[str]:
    cols = []
    for stem in TEAM_VARS:
        for o in TEAM_OWNERS:
            name = (f"age_owner_{o}_r_0" if stem == "age_owner"
                    else f"{stem}_{o}_0")
            cols.append(name)
    return cols


def resolve_competing(states: list[str]) -> tuple[float, int, bool]:
    """(duration, event_type, drop). 0=censored, 1=failure, 2=sold/merged."""
    for k, st in enumerate(states, start=1):
        if st in ("out_of_business", "sold_merged"):
            return float(k), (1 if st == "out_of_business" else 2), False
        if st == "ineligible":
            return np.nan, 0, True
    last_alive = max([k for k, st in enumerate(states, start=1) if st == "alive"],
                     default=0)
    if states[-1] == "alive":
        return 8.0, 0, False                      # survived whole panel
    return float(max(last_alive, 1)), 0, False    # dropout -> right-censored


def build_event_type(df: pd.DataFrame) -> pd.DataFrame:
    status = {w: B._num(df, f"final_status_code_{w}") for w in C.WAVES}
    a10 = {w: B._num(df, f"a10_out_of_business_{w}") for w in C.FOLLOWUP_WAVES}
    durs, etypes, drops = [], [], []
    for i in range(len(df)):
        if status[0].iat[i] in C.DROP_FIRM_CODES:
            durs.append(np.nan); etypes.append(0); drops.append(True); continue
        states = [B._classify_wave(status[w].iat[i],
                                   a10[w].iat[i] if w in a10 else np.nan)
                  for w in C.FOLLOWUP_WAVES]
        d, e, drop = resolve_competing(states)
        durs.append(d); etypes.append(e); drops.append(drop)
    for w in C.FOLLOWUP_WAVES:
        df.drop(columns=[f"a10_out_of_business_{w}"], errors="ignore", inplace=True)
    df["duration"] = np.clip(durs, 1, 8)
    df["event_type"] = etypes
    df["event"] = (df["event_type"] == 1).astype(int)   # failure indicator
    df["_drop"] = drops
    return df


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    def stack(short):
        cols = []
        for o in TEAM_OWNERS:
            c = (f"age_owner_{o}_r_0" if short == "age" else
                 f"{[k for k,v in TEAM_VARS.items() if v==short][0]}_{o}_0")
            if c in df.columns:
                cols.append(B._num(df, c))
        return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=df.index)

    educ, workexp = stack("educ"), stack("workexp")
    sameind, gender = stack("sameind"), stack("gender")
    age, white = stack("age"), stack("white")

    df["team_size"] = educ.notna().sum(axis=1).clip(lower=1)   # # owners w/ data
    df["team_educ_max"] = educ.max(axis=1)
    df["team_educ_mean"] = educ.mean(axis=1)
    df["team_workexp_mean"] = workexp.mean(axis=1)
    df["team_workexp_max"] = workexp.max(axis=1)
    df["team_any_sameind"] = (sameind.fillna(0).eq(1).any(axis=1)).astype(int)
    df["team_age_mean"] = age.mean(axis=1)
    # gender diversity: has both a male (1) and a female (2) founder
    df["team_gender_mixed"] = ((gender.eq(1).any(axis=1)) &
                               (gender.eq(2).any(axis=1))).astype(int)
    df["team_share_female"] = gender.eq(2).sum(axis=1) / df["team_size"]
    # racial diversity: any non-white founder (white race code is 5; 0 = not white)
    df["team_any_nonwhite"] = (white.fillna(0).eq(0).any(axis=1) &
                               educ.notna().any(axis=1)).astype(int)
    return df


def main() -> None:
    cols = B._cols_for_load() + _team_cols()
    _, meta = pyreadstat.read_dta(str(C.RAW_DTA), metadataonly=True)
    have = set(meta.column_names)
    cols = [c for c in dict.fromkeys(cols) if c in have]
    df, _ = pyreadstat.read_dta(str(C.RAW_DTA), usecols=cols)
    print(f"loaded {len(df)} firms, {len(cols)} columns")

    df = build_event_type(df)
    df = B.build_covariates(df)
    df = build_team_features(df)
    df = B.build_panel_outcomes(df)

    ext = df[~df["_drop"]].copy()
    n = len(ext)
    print(f"analysis firms (incl. M&A): {n}")
    print("  event_type breakdown:",
          {0: int((ext.event_type == 0).sum()),
           1: int((ext.event_type == 1).sum()),
           2: int((ext.event_type == 2).sum())})
    print(f"  failures={int((ext.event_type==1).sum())} "
          f"({(ext.event_type==1).mean():.1%}), "
          f"sold/merged={int((ext.event_type==2).sum())} "
          f"({(ext.event_type==2).mean():.1%})")
    print(f"  median team size={ext.team_size.median():.0f}, "
          f"gender-mixed teams={ext.team_gender_mixed.mean():.1%}")

    keep = (["mprid", "duration", "event", "event_type"]
            + list(C.HC) + list(C.FINANCING) + list(C.COMPADV) + list(C.IP)
            + list(C.DEMOG) + C.INDUSTRY_DUMMIES + ["hightech"]
            + ["team_size", "team_educ_max", "team_educ_mean", "team_workexp_mean",
               "team_workexp_max", "team_any_sameind", "team_age_mean",
               "team_gender_mixed", "team_share_female", "team_any_nonwhite"]
            + ["rev_first", "emp_first", "rev_last", "emp_last"])
    keep = [c for c in keep if c in ext.columns]
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    ext[keep].to_parquet(C.PROCESSED / "extended.parquet", index=False)
    print(f"wrote {C.PROCESSED/'extended.parquet'}  shape={ext[keep].shape}")


if __name__ == "__main__":
    main()
