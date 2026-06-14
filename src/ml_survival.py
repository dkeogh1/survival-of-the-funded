"""ML survival models + data-driven interaction discovery (extension #4).

Two questions:
  (a) Do flexible ML survival models (Random Survival Forest, Gradient-Boosted Cox)
      predict failure better than the linear Cox -- and do TEAM-level founder
      features (all founders, not just the primary one) add signal?
  (b) The thesis hand-picked its interaction terms. Which interactions does the data
      actually favour? We rank pairwise interaction strength with Friedman's
      H-statistic on the gradient-boosted model and check where the thesis's chosen
      interactions (ednet, indnet, netfwex_2) land.

Run: uv run python src/ml_survival.py
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

import config as C
import survival_cox as S

PRIMARY_HC = S.HUMAN_CAP
TEAM_HC = ["team_size", "team_educ_max", "team_educ_mean", "team_workexp_mean",
           "team_workexp_max", "team_any_sameind", "team_age_mean",
           "team_gender_mixed", "team_share_female", "team_any_nonwhite"]
BASE_FEATURES = (S.FINANCING + S.COMPADV + S.IP + ["foundage", "foundhisp",
                 "foundmale", "hightech"] + C.INDUSTRY_DUMMIES)
THESIS_PAIRS = {  # name -> (feat_a, feat_b)
    "ednet": ("foundered", "univcompadv"),
    "indnet": ("founderexpsameind", "compcompadv"),
    "netfwex_2": ("founderworkexp", "compcompadv"),
}


def load_xy(features):
    df = pd.read_parquet(C.PROCESSED / "extended.parquet")
    d = df[["duration", "event"] + features].apply(pd.to_numeric, errors="coerce").dropna()
    X = d[features].astype(float)
    y = Surv.from_arrays(event=d["event"].astype(bool), time=d["duration"].astype(float))
    return X, y


def cindex(model, X, y):
    risk = model.predict(X)
    return concordance_index_censored(y["event"], y["time"], risk)[0]


# ---------------------------------------------------------------- H-statistic
def _centered_pd1(model, Xbg, j, grid):
    """Centered 1-D partial dependence of `model` on feature j, at each grid value."""
    out = np.empty(len(grid))
    base = Xbg.values.copy()
    jj = Xbg.columns.get_loc(j)
    for g, v in enumerate(grid):
        base[:, jj] = v
        out[g] = model.predict(base).mean()
    return out - out.mean()


def _centered_pd2(model, Xbg, j, k, pts):
    """Centered 2-D partial dependence at the observed (j,k) points `pts`."""
    out = np.empty(len(pts))
    base = Xbg.values.copy()
    jj, kk = Xbg.columns.get_loc(j), Xbg.columns.get_loc(k)
    for i, (vj, vk) in enumerate(pts):
        base[:, jj] = vj
        base[:, kk] = vk
        out[i] = model.predict(base).mean()
    return out - out.mean()


def h_statistic(model, Xbg, j, k):
    """Friedman's H^2 for the (j,k) interaction, evaluated on the background sample."""
    pdj = pd.Series(_centered_pd1(model, Xbg, j, Xbg[j].values), index=Xbg.index)
    pdk = pd.Series(_centered_pd1(model, Xbg, k, Xbg[k].values), index=Xbg.index)
    pts = list(zip(Xbg[j].values, Xbg[k].values))
    pjk = _centered_pd2(model, Xbg, j, k, pts)
    num = np.sum((pjk - pdj.values - pdk.values) ** 2)
    den = np.sum(pjk ** 2)
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    (C.OUTPUT / "tables").mkdir(parents=True, exist_ok=True)

    # ---- (a) predictive comparison: primary-founder vs + team features
    print("=== Predictive performance (out-of-sample concordance) ===")
    for label, feats in [("primary-founder HC", PRIMARY_HC + BASE_FEATURES),
                         ("+ team HC", PRIMARY_HC + TEAM_HC + BASE_FEATURES)]:
        X, y = load_xy(feats)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, ytr)
        rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=30,
                                   max_features="sqrt", random_state=0, n_jobs=-1).fit(Xtr, ytr)
        gbm = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                               max_depth=3, subsample=0.8,
                                               random_state=0).fit(Xtr, ytr)
        print(f"\n[{label}]  (n={len(X)}, {len(feats)} features)")
        print(f"    Cox (linear)            C = {cindex(cox, Xte, yte):.3f}")
        print(f"    Random Survival Forest  C = {cindex(rsf, Xte, yte):.3f}")
        print(f"    Gradient-Boosted Cox    C = {cindex(gbm, Xte, yte):.3f}")

    # ---- team-feature importance (does the team add signal beyond primary founder?)
    feats = PRIMARY_HC + TEAM_HC + BASE_FEATURES
    X, y = load_xy(feats)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    gbm = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                           max_depth=3, subsample=0.8,
                                           random_state=0).fit(Xtr, ytr)
    imp = permutation_importance(gbm, Xte, yte, n_repeats=8, random_state=0,
                                 scoring=lambda m, Xv, yv: cindex(m, Xv, yv))
    impser = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    print("\n=== Top 12 features by permutation importance (GBM) ===")
    print(impser.head(12).round(4).to_string())
    print("\n  team features rank:")
    print(impser[[f for f in TEAM_HC]].round(4).to_string())

    # ---- (b) interaction discovery via Friedman's H-statistic
    Xbg = Xtr.sample(min(150, len(Xtr)), random_state=0)
    cand = list(impser.head(8).index)
    pairs = set(map(frozenset, itertools.combinations(cand, 2)))
    for a, b in THESIS_PAIRS.values():
        if a in X.columns and b in X.columns:
            pairs.add(frozenset((a, b)))
    rows = []
    for p in pairs:
        j, k = tuple(p)
        rows.append((j, k, h_statistic(gbm, Xbg, j, k)))
    H = pd.DataFrame(rows, columns=["feat_a", "feat_b", "H2"]).sort_values(
        "H2", ascending=False).reset_index(drop=True)
    H.to_csv(C.OUTPUT / "tables" / "interaction_Hstat.csv", index=False)
    print("\n=== Strongest interactions by Friedman's H^2 (top 10) ===")
    print(H.head(10).round(4).to_string(index=False))

    print("\n=== Where the thesis's hand-picked interactions rank ===")
    def rank_of(a, b):
        m = H[((H.feat_a == a) & (H.feat_b == b)) | ((H.feat_a == b) & (H.feat_b == a))]
        return (int(m.index[0]) + 1, float(m.H2.iloc[0])) if len(m) else (None, None)
    for name, (a, b) in THESIS_PAIRS.items():
        r, h = rank_of(a, b)
        print(f"  {name:10s} ({a} x {b}): "
              + (f"rank {r}/{len(H)}, H^2={h:.4f}" if r else "not evaluated"))


if __name__ == "__main__":
    main()
