"""GPU survival model: XGBoost Accelerated Failure Time + exact SHAP interactions.

Two payoffs that justify the GPU on this small dataset:
  1. A 4th survival model (GPU-trained XGBoost-AFT) for the concordance comparison.
  2. EXACT SHAP pairwise interaction values, computed on-device via XGBoost's CUDA
     predictor -- no `shap`/`numba` (which clash with our numpy). This is an
     independent second opinion on the interaction-discovery result from
     ml_survival.py (Friedman's H-statistic): does SHAP also rank the thesis's
     hand-picked interactions near zero?

Run: uv run python src/gpu_survival.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split

import config as C
import ml_survival as ML   # reuse feature lists

FEATURES = ML.PRIMARY_HC + ML.TEAM_HC + ML.BASE_FEATURES
THESIS_PAIRS = ML.THESIS_PAIRS


def _device() -> str:
    """Use CUDA if a GPU is present, else fall back to CPU."""
    try:
        d = xgb.DMatrix(np.zeros((4, 2), "float32"), label=np.arange(4))
        xgb.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


PARAMS = dict(objective="survival:aft", eval_metric="aft-nloglik",
              aft_loss_distribution="normal", aft_loss_distribution_scale=1.0,
              tree_method="hist", device=_device(), learning_rate=0.05,
              max_depth=3, subsample=0.8, min_child_weight=10)


def make_dmatrix(X: pd.DataFrame, y) -> xgb.DMatrix:
    lower = y["time"].astype(float)
    upper = np.where(y["event"], y["time"], np.inf).astype(float)
    d = xgb.DMatrix(X.values.astype("float32"), feature_names=list(X.columns))
    d.set_float_info("label_lower_bound", lower)
    d.set_float_info("label_upper_bound", upper)
    return d


def main() -> None:
    X, y = ML.load_xy(FEATURES)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    dtr, dte = make_dmatrix(Xtr, ytr), make_dmatrix(Xte, yte)

    booster = xgb.train(PARAMS, dtr, num_boost_round=400,
                        evals=[(dtr, "train"), (dte, "test")],
                        early_stopping_rounds=40, verbose_eval=False)

    # AFT predicts survival TIME -> risk = -time (higher = fails sooner)
    risk = -booster.predict(dte)
    c = concordance_index_censored(yte["event"], yte["time"], risk)[0]
    print(f"=== XGBoost-AFT [{PARAMS['device']}]  (n={len(X)}, {len(FEATURES)} features, "
          f"best_iter={booster.best_iteration}) ===")
    print(f"    out-of-sample concordance C = {c:.3f}")
    print("    (compare: Cox 0.686, RSF 0.692, GBM 0.697 from ml_survival.py)\n")

    # ---- exact SHAP pairwise interactions on GPU ----
    inter = booster.predict(dte, pred_interactions=True)   # (n, F+1, F+1) incl. bias
    F = len(FEATURES)
    strength = np.abs(inter[:, :F, :F]).mean(axis=0)       # mean |interaction| per pair
    np.fill_diagonal(strength, 0.0)
    pairs = [(FEATURES[i], FEATURES[j], strength[i, j])
             for i in range(F) for j in range(i + 1, F)]
    H = pd.DataFrame(pairs, columns=["feat_a", "feat_b", "shap_interaction"]) \
        .sort_values("shap_interaction", ascending=False).reset_index(drop=True)
    (C.OUTPUT / "tables").mkdir(parents=True, exist_ok=True)
    H.to_csv(C.OUTPUT / "tables" / "shap_interactions_gpu.csv", index=False)

    print("=== Strongest interactions by exact SHAP (top 10) ===")
    print(H.head(10).round(5).to_string(index=False))

    print("\n=== Where the thesis's hand-picked interactions rank (SHAP) ===")
    def rank_of(a, b):
        m = H[((H.feat_a == a) & (H.feat_b == b)) | ((H.feat_a == b) & (H.feat_b == a))]
        return (int(m.index[0]) + 1, float(m.shap_interaction.iloc[0])) if len(m) else (None, None)
    for name, (a, b) in THESIS_PAIRS.items():
        r, v = rank_of(a, b)
        print(f"  {name:10s} ({a} x {b}): "
              + (f"rank {r}/{len(H)}, SHAP={v:.5f}" if r else "not evaluated"))

    # cross-check vs Friedman's H if available
    try:
        hf = pd.read_csv(C.OUTPUT / "tables" / "interaction_Hstat.csv")
        top_h = set(map(frozenset, hf.head(5)[["feat_a", "feat_b"]].values))
        top_s = set(map(frozenset, H.head(5)[["feat_a", "feat_b"]].values))
        overlap = top_h & top_s
        print(f"\n  top-5 overlap between SHAP and Friedman's H: {len(overlap)}/5")
        for p in overlap:
            print("    both rank highly:", " x ".join(p))
    except FileNotFoundError:
        print("\n  (run ml_survival.py to enable the SHAP-vs-H cross-check)")


if __name__ == "__main__":
    main()
