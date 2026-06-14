"""Experiment: time-varying financing in a long (counting-process) panel.

The thesis built a firm-YEAR panel in SAS. If financing entered the Cox model as a
time-varying covariate, the sign can flip versus a firm-level "ever received" flag:
a firm that takes government money the year before it fails looks *harmful*
contemporaneously, but *protective* as an ever-flag (it survived long enough to get it).

We expand the analysis sample to one row per firm-year at risk, attach time-varying
financing (cumulative-to-date or contemporaneous), keep the other covariates time-fixed,
and fit lifelines' CoxTimeVaryingFitter.

Run: uv run python src/experiments/timevarying_cox.py [cumulative|contemporaneous]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from lifelines import CoxTimeVaryingFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import survival_cox as S

FIN_STEMS = {
    "eqangels": ["f3c_eq_invest_angels"],
    "eqvc":     ["f3f_eq_invest_vent_cap"],
    "eqgovt":   ["f3e_eq_invest_govt"],
    "eqfff":    ["f3a_eq_invest_spouse", "f3b_eq_invest_parents"],
    "debtfin":  ["f11a_bus_loans_bank", "f11a_bus_loans_nonbank", "f11a_bus_loans_owner",
                 "f11a_bus_loans_govt", "f11a_bus_loans_emp", "f11a_bus_loans_other_bus",
                 "f11a_bus_loans_other_ind"],
}
TIMEFIXED = [c for c in S.MAIN if c not in FIN_STEMS]   # human cap, compadv, IP, demog, industry
THESIS_HR = {"eqangels": 0.28, "eqfff": 0.55, "debtfin": 0.13, "eqgovt": 3.62, "eqvc": 1.03}


def per_wave_financing() -> dict[str, pd.DataFrame]:
    meta = pyreadstat.read_dta(str(C.RAW_DTA), metadataonly=True)[1]
    have = set(meta.column_names)
    cols = ["mprid"]
    for stems in FIN_STEMS.values():
        for s in stems:
            cols += [f"{s}_{w}" for w in C.WAVES if f"{s}_{w}" in have]
    raw = pyreadstat.read_dta(str(C.RAW_DTA), usecols=list(dict.fromkeys(cols)))[0]
    out = {}
    for name, stems in FIN_STEMS.items():
        wave_flags = {}
        for w in C.WAVES:
            cc = [f"{s}_{w}" for s in stems if f"{s}_{w}" in raw.columns]
            wave_flags[w] = (raw[cc].apply(pd.to_numeric, errors="coerce").eq(1)
                             .any(axis=1).astype(int)) if cc else pd.Series(0, index=raw.index)
        out[name] = pd.DataFrame(wave_flags, index=raw.index).assign(mprid=raw["mprid"])
    return out


def build_long(analysis: pd.DataFrame, timing: str) -> pd.DataFrame:
    fin = per_wave_financing()
    fin = {k: v.set_index("mprid") for k, v in fin.items()}
    rows = []
    fixed_cols = ["mprid", "duration", "event"] + [c for c in TIMEFIXED if c in analysis.columns]
    a = analysis[fixed_cols].copy()
    for r in a.itertuples(index=False):
        rec = dict(zip(fixed_cols, r))
        mid, dur, ev = rec["mprid"], int(rec["duration"]), int(rec["event"])
        for t in range(1, dur + 1):                       # intervals (t-1, t]
            row = {k: rec[k] for k in fixed_cols if k not in ("duration", "event")}
            row["start"], row["stop"] = t - 1, t
            row["event"] = ev if t == dur else 0
            for name in FIN_STEMS:
                waves = fin[name].columns
                if timing == "cumulative":                # received by start of interval
                    val = int(fin[name].loc[mid, [w for w in range(0, t) if w in waves]].max())
                else:                                     # contemporaneous: wave at interval start
                    w = min(t - 1, max(waves))
                    val = int(fin[name].loc[mid, w]) if w in waves else 0
                row[name] = val
            rows.append(row)
    return pd.DataFrame(rows)


def main(timing: str = "cumulative"):
    analysis = pd.read_parquet(C.PROCESSED / "analysis.parquet")
    long = build_long(analysis, timing)
    covars = [c for c in (list(FIN_STEMS) + TIMEFIXED) if c in long.columns]
    d = long[["mprid", "start", "stop", "event"] + covars].apply(
        pd.to_numeric, errors="coerce").dropna()
    nz = [c for c in covars if d[c].nunique() > 1]

    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(d[["mprid", "start", "stop", "event"] + nz],
            id_col="mprid", event_col="event", start_col="start", stop_col="stop")
    hr = np.exp(ctv.summary["coef"])
    print(f"\n=== Time-varying Cox ({timing} financing): "
          f"{d.mprid.nunique()} firms, {len(d)} firm-year rows, {int(d.event.sum())} events ===\n")
    comp = pd.DataFrame({"HR_timevarying": hr.reindex(THESIS_HR).round(3),
                         "thesis_HR": pd.Series(THESIS_HR)})
    print(comp.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cumulative")
