"""
ml/train.py
===========
Retrains the three twin models on telemetry from `ml.dataset` (the same
physics the live demo runs, randomized over many flights). Writes artifacts
to ``models/`` where `ml.models.TwinModels` looks first.

    python ml/train.py --flights 140

Models
------
  anomaly_detector.joblib : IsolationForest, healthy rows only
  fault_classifier.joblib : {model, labels} -- XGBoost multi-class
  rul_regressor.joblib    : XGBoost regressor, degradation rows only
  feature_columns.joblib  : the 36-feature ordering
  metrics.json            : held-out (split-by-flight) scores
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)

from ml.dataset import make_training_frame  # noqa: E402


def split_by_flight(df, test_frac=0.2, seed=42):
    ids = df["flight_id"].unique()
    rng = np.random.default_rng(seed)
    sh = rng.permutation(ids)
    n_test = max(1, int(len(sh) * test_frac))
    test = set(sh[:n_test])
    return df[~df.flight_id.isin(test)].copy(), df[df.flight_id.isin(test)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20240)
    ap.add_argument("--outdir", type=str, default=str(_REPO_ROOT / "models"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.flights} randomized flights ...")
    df, feature_cols = make_training_frame(args.flights, seed=args.seed)
    print(f"  {len(df):,} rows / {df.flight_id.nunique()} flights")
    print(df["fault_label"].value_counts().to_string())

    train_df, test_df = split_by_flight(df, seed=args.seed)
    print(f"  train {train_df.flight_id.nunique()} flights / "
          f"test {test_df.flight_id.nunique()} flights")

    metrics: dict = {}

    # -- anomaly detector ------------------------------------------------
    print("\n[1/3] anomaly detector (IsolationForest, healthy only)")
    healthy = train_df[train_df.fault_label == "healthy"]
    anom = IsolationForest(n_estimators=250, contamination=0.03,
                           random_state=42, n_jobs=-1)
    anom.fit(healthy[feature_cols])
    scores = -anom.score_samples(test_df[feature_cols])
    y_true = (test_df.fault_label != "healthy").astype(int).values
    auc = float(roc_auc_score(y_true, scores))
    flagged = (anom.predict(test_df[feature_cols]) == -1)
    metrics["anomaly_detector"] = {
        "roc_auc": auc,
        "recall_on_faults": float(flagged[y_true == 1].mean()),
        "false_positive_rate": float(flagged[y_true == 0].mean()),
    }
    print(f"  ROC-AUC {auc:.3f}  recall {metrics['anomaly_detector']['recall_on_faults']:.2f}"
          f"  FPR {metrics['anomaly_detector']['false_positive_rate']:.2f}")

    # -- fault classifier ---------------------------------------------
    print("\n[2/3] fault classifier (XGBoost multi-class)")
    labels = sorted(train_df.fault_label.unique())
    l2i = {l: i for i, l in enumerate(labels)}
    clf = xgb.XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, objective="multi:softprob",
        num_class=len(labels), random_state=42, n_jobs=-1, eval_metric="mlogloss",
    )
    clf.fit(train_df[feature_cols], train_df.fault_label.map(l2i))
    y_true_c = test_df.fault_label.map(l2i).values
    y_pred_c = clf.predict(test_df[feature_cols])
    print(classification_report(y_true_c, y_pred_c, target_names=labels, zero_division=0))
    macro_f1 = classification_report(y_true_c, y_pred_c, target_names=labels,
                                     zero_division=0, output_dict=True)["macro avg"]["f1-score"]
    metrics["fault_classifier"] = {
        "macro_f1": float(macro_f1),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true_c, y_pred_c).tolist(),
    }

    # -- RUL regressor ----------------------------------------------
    print("[3/3] RUL regressor (XGBoost, degradation rows only)")
    rtr = train_df[train_df.remaining_useful_life.notna()]
    rte = test_df[test_df.remaining_useful_life.notna()]
    rul = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
        random_state=42, n_jobs=-1,
    )
    rul.fit(rtr[feature_cols], rtr.remaining_useful_life)
    if len(rte):
        pred = rul.predict(rte[feature_cols])
        mae = float(mean_absolute_error(rte.remaining_useful_life, pred))
        r2 = float(r2_score(rte.remaining_useful_life, pred))
        metrics["rul_regressor"] = {"mae_seconds": mae, "r2": r2}
        print(f"  MAE {mae:.0f}s ({mae/60:.1f} min)   R2 {r2:.3f}")

    # -- save ---------------------------------------------------------
    joblib.dump(anom, outdir / "anomaly_detector.joblib")
    joblib.dump({"model": clf, "labels": labels}, outdir / "fault_classifier.joblib")
    joblib.dump(rul, outdir / "rul_regressor.joblib")
    joblib.dump(feature_cols, outdir / "feature_columns.joblib")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {outdir}/")


if __name__ == "__main__":
    main()
