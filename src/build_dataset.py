"""Build the firm-level analysis dataset from the KFS public-use file.

Produces data/processed/analysis.parquet with one row per firm:
  - survival outcome: duration (1-8) + event (1=failed/out of business)
  - baseline covariates: human capital, IP, demographics, industry
  - "ever observed" indicators: financing, competitive advantage
  - panel outcomes for LIML: first/last revenue & employment, lagged values

Run: uv run python src/build_dataset.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyreadstat

import config as C


def _cols_for_load() -> list[str]:
    cols = ["mprid", C.NAICS_COL, C.HIGHTECH_COL]
    cols += C.STATUS + C.OUT_OF_BUSINESS + C.REVENUE + C.EMPLOYMENT
    cols += [f"{stem}_0" for stem in C.HC.values()]
    cols += list(C.IP.values())
    cols += list(C.DEMOG.values())
    # financing stems across all waves
    for stems in C.FINANCING.values():
        for s in stems:
            cols += [f"{s}_{w}" for w in C.WAVES]
    # competitive-advantage subtypes (waves 3-7) + overall flag (all waves)
    for stem in C.COMPADV.values():
        cols += [f"{stem}_{w}" for w in range(3, 8)]
    cols += [f"{C.COMPADV_OVERALL}_{w}" for w in C.WAVES]
    # keep only columns that actually exist in the file
    _, meta = pyreadstat.read_dta(str(C.RAW_DTA), metadataonly=True)
    have = set(meta.column_names)
    seen, out = set(), []
    for c in cols:
        if c in have and c not in seen:
            seen.add(c); out.append(c)
    return out


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def _classify_wave(status: float, a10: float) -> str:
    """Map (final_status_code, a10_out_of_business) for one wave to a state."""
    if status in C.ALIVE_CODES:
        return "alive"
    if status in C.DROP_FIRM_CODES:
        return "ineligible"
    if status in (C.PERMANENT_STOP_CODE, C.TEMP_STOP_CODE):
        if a10 in (C.A10_SOLD, C.A10_MERGED):
            return "sold_merged"
        return "out_of_business"           # a10 in {3,4}, NaN, or temp-stop
    return "nonresponse"                    # refusals, retired, unlocatable, etc.


def build_survival(df: pd.DataFrame, definition: str = "paper") -> pd.DataFrame:
    """Construct (duration, event) plus a drop flag per firm.

    definition="paper":  terminal survey attrition counts as failure; only firms
        alive in the final wave are censored. Interior response gaps -> dropped.
    definition="competing_risks":  KFS-codebook convention -- failure only on
        permanent out-of-business; non-response is right-censored.
    """
    status = {w: _num(df, f"final_status_code_{w}") for w in C.WAVES}
    a10 = {w: _num(df, f"a10_out_of_business_{w}") for w in C.FOLLOWUP_WAVES}

    durations, events, drops = [], [], []
    for i in range(len(df)):
        if status[0].iat[i] in C.DROP_FIRM_CODES:
            durations.append(np.nan); events.append(0); drops.append(True); continue

        states = [_classify_wave(status[w].iat[i],
                                 a10[w].iat[i] if w in a10 else np.nan)
                  for w in C.FOLLOWUP_WAVES]            # index 0 -> wave 1 ... 6 -> wave 7

        duration, event, drop = _resolve_firm(states, definition)
        durations.append(duration); events.append(event); drops.append(drop)

    for w in C.FOLLOWUP_WAVES:
        df.drop(columns=[f"a10_out_of_business_{w}"], errors="ignore", inplace=True)
    df["duration"] = np.clip(durations, 1, 8)
    df["event"] = events
    df["_drop"] = drops
    return df


def _resolve_firm(states: list[str], definition: str) -> tuple[float, int, bool]:
    """Reduce a wave-by-wave state sequence to (duration, event, drop)."""
    term_wave = term_kind = None
    for k, st in enumerate(states, start=1):     # k = wave number (1..7)
        if st in ("out_of_business", "sold_merged"):
            term_wave, term_kind = k, st
            break
        if st == "ineligible":
            return np.nan, 0, True

    last_alive = max([k for k, st in enumerate(states, start=1) if st == "alive"],
                     default=0)

    if term_kind == "sold_merged":
        return float(term_wave), 0, True
    if term_kind == "out_of_business":
        return float(term_wave), 1, False        # failure during follow-up term_wave

    alive_final = states[-1] == "alive"          # responded in final wave (7)
    if alive_final:
        return 8.0, 0, False                     # survived the whole panel

    if definition == "paper":
        # all firms are alive at baseline (wave 0); terminal non-response is failure.
        if last_alive == 0:
            return 1.0, 1, False                 # failed by first follow-up (year 1)
        gap = any(states[j] != "alive" for j in range(last_alive)) \
              and any(states[j] == "alive" for j in range(last_alive, len(states)))
        if gap:
            return float(last_alive), 0, True    # interior gap -> remove
        return float(last_alive), 1, False       # terminal attrition == failure

    return float(max(last_alive, 1)), 0, False   # competing_risks: censor


def build_covariates(df: pd.DataFrame) -> pd.DataFrame:
    # ---- human & social capital (baseline)
    for name, stem in C.HC.items():
        df[name] = _num(df, f"{stem}_0")
    df["founderexpsameind"] = df["founderexpsameind"].fillna(0).clip(0, 1)

    # ---- financing: ever == 1 across observed waves (any contributing stem)
    for name, stems in C.FINANCING.items():
        ever = pd.Series(0, index=df.index)
        for s in stems:
            wide = [f"{s}_{w}" for w in C.WAVES if f"{s}_{w}" in df.columns]
            if wide:
                ever = ever | (df[wide].apply(pd.to_numeric, errors="coerce")
                               .eq(1).any(axis=1).astype(int))
        df[name] = ever.astype(int)

    # ---- competitive advantage: ever reported the channel (waves 3-7)
    for name, stem in C.COMPADV.items():
        wide = [f"{stem}_{w}" for w in range(3, 8) if f"{stem}_{w}" in df.columns]
        df[name] = (df[wide].apply(pd.to_numeric, errors="coerce")
                    .eq(1).any(axis=1).astype(int)) if wide else 0

    # ---- IP counts (baseline)
    for name, col in C.IP.items():
        df[name] = _num(df, col).fillna(0)

    # ---- demographics (baseline). Race fields store the race-code as the "yes"
    # value (white=5, black=4, asian=3, amind=1; 0=no) -> binarize as != 0.
    for name in ("foundwhite", "foundblack", "foundasian", "foundamind"):
        df[name] = _num(df, C.DEMOG[name]).gt(0).astype("Int64")
    df["foundhisp"] = _num(df, C.DEMOG["foundhisp"]).eq(1).astype("Int64")
    df["foundmale"] = _num(df, C.DEMOG["foundmale"]).eq(1).astype("Int64")  # 1=male,2=female
    df["foundage"] = _num(df, C.DEMOG["foundage"])                          # age level 1-7

    # ---- industry dummies from NAICS 2-digit
    naics2 = df[C.NAICS_COL].astype("string").str.replace(r"\D", "", regex=True).str[:2]
    industry = naics2.map(C.NAICS_TO_INDUSTRY)
    for dummy in C.INDUSTRY_DUMMIES:
        df[dummy] = (industry == dummy).astype(int)
    df["hightech"] = _num(df, C.HIGHTECH_COL).fillna(0).astype(int)

    return df


def build_panel_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """First/last observed revenue & employment among waves the firm was alive."""
    rev = df[[c for c in C.REVENUE if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    emp = df[[c for c in C.EMPLOYMENT if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    df["rev_first"] = rev.bfill(axis=1).iloc[:, 0]
    df["emp_first"] = emp.bfill(axis=1).iloc[:, 0]
    df["rev_last"] = rev.ffill(axis=1).iloc[:, -1]
    df["emp_last"] = emp.ffill(axis=1).iloc[:, -1]
    return df


def main(definition: str = "paper") -> None:
    cols = _cols_for_load()
    df, _ = pyreadstat.read_dta(str(C.RAW_DTA), usecols=cols)
    print(f"loaded {len(df)} firms, {len(cols)} columns  [survival definition: {definition}]")

    df = build_survival(df, definition=definition)
    df = build_covariates(df)
    df = build_panel_outcomes(df)

    analysis = df[~df["_drop"]].copy()
    print(f"after dropping sold/merged/ineligible: {len(analysis)} firms")
    print(f"  failures (event=1): {int(analysis['event'].sum())} "
          f"({analysis['event'].mean():.1%})")
    print(f"  first-year failures (duration==1 & event): "
          f"{int(((analysis.duration==1) & (analysis.event==1)).sum())}")
    print(f"  survived all 8 (duration==8 & event==0): "
          f"{int(((analysis.duration==8) & (analysis.event==0)).sum())} "
          f"({((analysis.duration==8)&(analysis.event==0)).mean():.1%})")

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    keep = (["mprid", "duration", "event"]
            + list(C.HC) + list(C.FINANCING) + list(C.COMPADV)
            + list(C.IP) + list(C.DEMOG) + C.INDUSTRY_DUMMIES + ["hightech"]
            + ["rev_first", "emp_first", "rev_last", "emp_last"])
    keep = [c for c in keep if c in analysis.columns]
    out = analysis[keep]
    out.to_parquet(C.PROCESSED / "analysis.parquet", index=False)
    print(f"wrote {C.PROCESSED/'analysis.parquet'}  shape={out.shape}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "paper")
