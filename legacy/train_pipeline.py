"""
Digital Twin ML Training Pipeline
==================================
Trains three models on the engine simulator dataset:
  1. Anomaly detector (Isolation Forest, trained on healthy data only)
  2. Fault classifier (XGBoost multi-class)
  3. RUL regressor (XGBoost regression, only on degradation-fault rows)

Plus SHAP explainability wired to the classifier, so predictions come with
a "why" -- not just a black-box label.

Usage (Kaggle or local):
    python train_pipeline.py --data /path/to/uav_training_data.csv --outdir ./models
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import xgboost as xgb
import shap
import joblib

warnings.filterwarnings("ignore")

SENSOR_COLS = ["rpm", "cht", "egt", "oil_pressure", "oil_temp", "fuel_flow", "vibration", "battery_voltage"]
ROLLING_WINDOW = 6  # ~1 minute of history at 10s sampling


# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Defensive fix: a stray shell command has been seen prepended to the
    # first column's header in some exports (e.g. a Kaggle setup cell's
    # output leaking into the CSV). If the first column name doesn't look
    # like 'flight_id', assume this happened and rename it.
    if df.columns[0] != "flight_id":
        df = df.rename(columns={df.columns[0]: "flight_id"})
    df = df.sort_values(["flight_id", "t_s"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Physics-informed expected values (mirrors engine_simulator_v2's
#    recalibrated + turbocharged physics, steady-state only -- matches what
#    the Digital Twin Core computes live, so training and live inference
#    use the same reference model)
# ---------------------------------------------------------------------------

IDLE_RPM, MAX_RPM, CRUISE_RPM = 1400.0, 5800.0, 5000.0
MAP_IDLE, MAP_MAX = 12.0, 29.9
CHT_IDLE, CHT_CRUISE, CHT_REDLINE = 90.0, 125.0, 150.0
EGT_IDLE, EGT_CRUISE, EGT_REDLINE = 420.0, 800.0, 900.0
OIL_P_MIN, OIL_P_MAX = 0.8, 5.0
OIL_T_IDLE, OIL_T_CRUISE = 60.0, 100.0
FUEL_FLOW_MAX = 25.0
VIB_BASELINE = 1.0
BATTERY_NOMINAL = 13.8
TURBO_CRITICAL_ALT_M = 5000.0
CRUISE_LOAD_REF = 0.7


def expected_engine_values(throttle_pct: np.ndarray, altitude_m: np.ndarray, ambient_c: np.ndarray) -> pd.DataFrame:
    """Vectorized steady-state physics expectation, matching the simulator's
    recalibrated + turbocharged model. Used to compute residual features."""
    thr = throttle_pct / 100.0
    temp_k = ambient_c + 273.15
    pressure_pa = 101325 * (1 - 2.25577e-5 * altitude_m) ** 5.25588

    above_critical = altitude_m > TURBO_CRITICAL_ALT_M
    excess_alt = np.clip(altitude_m - TURBO_CRITICAL_ALT_M, 0, None)
    falloff = np.clip((1 - 2.25577e-5 * excess_alt) ** 5.25588, 0.3, None)
    turbo_map_ceiling = np.where(above_critical, MAP_MAX * falloff, MAP_MAX)
    turbo_density_ratio = np.where(above_critical, falloff, 1.0)

    map_expected = MAP_IDLE + thr * (turbo_map_ceiling - MAP_IDLE)

    rpm_expected = np.where(
        thr <= 0.5,
        IDLE_RPM + (thr / 0.5) * (CRUISE_RPM - IDLE_RPM),
        CRUISE_RPM + ((thr - 0.5) / 0.5) * (MAX_RPM - CRUISE_RPM),
    )

    load_factor = np.clip((map_expected / MAP_MAX) * (rpm_expected / MAX_RPM), 0, None)

    frac_low = load_factor / CRUISE_LOAD_REF
    frac_high = np.clip((load_factor - CRUISE_LOAD_REF) / (1.0 - CRUISE_LOAD_REF), None, 1.2)
    egt_expected = np.where(load_factor <= CRUISE_LOAD_REF,
                             EGT_IDLE + frac_low * (EGT_CRUISE - EGT_IDLE),
                             EGT_CRUISE + frac_high * (EGT_REDLINE - EGT_CRUISE))
    cht_expected = np.where(load_factor <= CRUISE_LOAD_REF,
                             CHT_IDLE + frac_low * (CHT_CRUISE - CHT_IDLE),
                             CHT_CRUISE + frac_high * (CHT_REDLINE - CHT_CRUISE))

    ambient_offset = ambient_c - 15.0
    ram_air_cooling = thr * 15.0
    cht_expected = cht_expected + ambient_offset - ram_air_cooling

    oil_temp_expected = OIL_T_IDLE + (OIL_T_CRUISE - OIL_T_IDLE) * load_factor * 1.2
    oil_temp_expected = oil_temp_expected + ambient_offset * 0.5 - ram_air_cooling * 0.5

    rpm_factor = rpm_expected / MAX_RPM
    oil_pressure_expected = OIL_P_MIN + (OIL_P_MAX - OIL_P_MIN) * rpm_factor
    oil_pressure_expected = np.clip(oil_pressure_expected, None, 7.0)

    vibration_expected = VIB_BASELINE * (0.5 + 0.5 * load_factor)
    fuel_flow_expected = FUEL_FLOW_MAX * load_factor * turbo_density_ratio
    battery_expected = np.where(rpm_expected < 2000, BATTERY_NOMINAL - 0.6, BATTERY_NOMINAL)

    return pd.DataFrame({
        "rpm_expected": rpm_expected, "cht_expected": cht_expected, "egt_expected": egt_expected,
        "oil_pressure_expected": oil_pressure_expected, "oil_temp_expected": oil_temp_expected,
        "fuel_flow_expected": fuel_flow_expected, "vibration_expected": vibration_expected,
        "battery_voltage_expected": battery_expected,
    })


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected = expected_engine_values(df["throttle_pct"].values, df["altitude_m"].values, df["ambient_c"].values)
    for col in SENSOR_COLS:
        df[f"{col}_residual"] = df[col].values - expected[f"{col}_expected"].values

    # Rolling stats + rate of change, computed PER FLIGHT so history never
    # leaks across flight boundaries.
    grouped = df.groupby("flight_id", sort=False)
    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = grouped[col].transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
        df[f"{col}_roll_std"] = grouped[col].transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
        df[f"{col}_rate"] = grouped[col].transform(lambda s: s.diff().fillna(0))

    # "Time spent in an anomalous state" -- a single snapshot's residual
    # magnitude alone doesn't tell you how long a trend has been developing,
    # which matters a lot for RUL (how much life is left depends on the
    # RATE of decline, not just the current value). This is computed purely
    # from residual thresholds -- NOT from ground-truth fault labels -- so
    # it's available identically at real inference time on live data.
    is_anomalous = (
        (df["oil_pressure_residual"].abs() > 0.5) |
        (df["cht_residual"].abs() > 15) |
        (df["egt_residual"].abs() > 50) |
        (df["vibration_residual"].abs() > 0.5)
    )
    df["_is_anomalous"] = is_anomalous
    df["time_in_anomaly_state"] = df.groupby("flight_id", group_keys=False)["_is_anomalous"].apply(
        lambda s: s.groupby((~s).cumsum()).cumcount().where(s, 0)
    )
    df = df.drop(columns=["_is_anomalous"])

    return df


def get_feature_columns() -> list:
    cols = []
    for col in SENSOR_COLS:
        cols += [f"{col}_residual", f"{col}_roll_mean", f"{col}_roll_std", f"{col}_rate"]
    cols += ["throttle_pct", "altitude_m", "ambient_c", "time_in_anomaly_state"]
    return cols


# ---------------------------------------------------------------------------
# 4. Split by flight (never split by row -- avoids leakage)
# ---------------------------------------------------------------------------

def split_by_flight(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    flight_ids = df["flight_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(flight_ids)
    n_test = max(1, int(len(shuffled) * test_frac))
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return df[df.flight_id.isin(train_ids)].copy(), df[df.flight_id.isin(test_ids)].copy()


# ---------------------------------------------------------------------------
# 5. Model 1: Anomaly detector
# ---------------------------------------------------------------------------

def train_anomaly_detector(train_df: pd.DataFrame, feature_cols: list):
    healthy = train_df[train_df.fault_label == "healthy"]
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    model.fit(healthy[feature_cols])
    return model


def evaluate_anomaly_detector(model, test_df: pd.DataFrame, feature_cols: list):
    scores = -model.score_samples(test_df[feature_cols])  # higher = more anomalous
    y_true = (test_df["fault_label"] != "healthy").astype(int).values
    auc = roc_auc_score(y_true, scores)
    preds = model.predict(test_df[feature_cols])  # -1 = anomaly, 1 = normal
    pred_anomaly = (preds == -1).astype(int)
    print(f"Anomaly detector ROC-AUC: {auc:.3f}")
    print(f"Anomaly detector flagged {pred_anomaly.sum()} / {len(pred_anomaly)} rows as anomalous "
          f"(true fault rate: {y_true.mean():.1%})")
    return {"roc_auc": auc}


# ---------------------------------------------------------------------------
# 6. Model 2: Fault classifier
# ---------------------------------------------------------------------------

def train_fault_classifier(train_df: pd.DataFrame, feature_cols: list):
    labels = sorted(train_df["fault_label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = train_df["fault_label"].map(label_to_idx).values

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(labels),
        random_state=42, n_jobs=-1, eval_metric="mlogloss",
    )
    model.fit(train_df[feature_cols], y)
    return model, labels


def evaluate_fault_classifier(model, labels, test_df: pd.DataFrame, feature_cols: list):
    label_to_idx = {l: i for i, l in enumerate(labels)}

    def _report(subset_df, title):
        if len(subset_df) == 0:
            return
        y_true = subset_df["fault_label"].map(label_to_idx).values
        y_pred = model.predict(subset_df[feature_cols])
        print(f"--- {title} ({len(subset_df):,} rows) ---")
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    _report(test_df, "All rows (including post-failure, where root cause is ambiguous by design)")

    # Post-failure rows (RPM ~0) are inherently ambiguous: once the engine has
    # fully died, oil_pressure/fuel_flow/etc. all crash to 0 regardless of
    # which fault caused it -- a single snapshot can't recover root cause at
    # that point (and it's not actionable for predictive maintenance anyway,
    # since the mission has already ended). Evaluate the pre-failure window
    # separately, since that's the window that actually matters.
    pre_failure = test_df[test_df["rpm"] > 100]
    _report(pre_failure, "Pre-failure only (the actionable predictive-maintenance window)")

    y_true_all = test_df["fault_label"].map(label_to_idx).values
    y_pred_all = model.predict(test_df[feature_cols])
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("Confusion matrix, all rows (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    return {"confusion_matrix": cm.tolist()}


# ---------------------------------------------------------------------------
# 7. Model 3: RUL regressor (only on rows with a real countdown)
# ---------------------------------------------------------------------------

def train_rul_regressor(train_df: pd.DataFrame, feature_cols: list):
    rul_rows = train_df[train_df["remaining_useful_life"].notna()]
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    model.fit(rul_rows[feature_cols], rul_rows["remaining_useful_life"])
    return model


def evaluate_rul_regressor(model, test_df: pd.DataFrame, feature_cols: list):
    rul_rows = test_df[test_df["remaining_useful_life"].notna()]
    if len(rul_rows) == 0:
        print("No RUL-labeled rows in test set -- skipping RUL evaluation")
        return {}
    y_true = rul_rows["remaining_useful_life"].values
    y_pred = model.predict(rul_rows[feature_cols])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RUL regressor -- MAE: {mae:.1f}s ({mae/60:.1f} min)  RMSE: {rmse:.1f}s  R2: {r2:.3f}")
    return {"mae_seconds": mae, "rmse_seconds": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# 8. SHAP explainability
# ---------------------------------------------------------------------------

def explain_prediction(classifier_model, labels, row_features: pd.DataFrame, top_n: int = 3) -> str:
    """Returns a human-readable explanation string for a single prediction,
    e.g. 'Misfire predicted: 61% due to vibration_residual, 39% due to egt_rate'"""
    explainer = shap.TreeExplainer(classifier_model)
    shap_values = explainer.shap_values(row_features)

    pred_idx = int(np.argmax(classifier_model.predict_proba(row_features)[0]))
    pred_label = labels[pred_idx]

    if isinstance(shap_values, list):
        values_for_class = shap_values[pred_idx][0]
    else:
        values_for_class = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]

    feature_names = row_features.columns.tolist()
    contrib = sorted(zip(feature_names, values_for_class), key=lambda x: -abs(x[1]))[:top_n]
    total_abs = sum(abs(v) for _, v in contrib) or 1e-9
    parts = [f"{abs(v)/total_abs:.0%} due to {name}" for name, v in contrib]
    return f"{pred_label} predicted: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="./models")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data...")
    df = load_and_clean(args.data)
    print(f"{len(df):,} rows, {df.flight_id.nunique()} flights")

    print("\nEngineering features...")
    df = add_features(df)
    feature_cols = get_feature_columns()

    print("\nSplitting by flight (train/test)...")
    train_df, test_df = split_by_flight(df, test_frac=args.test_frac, seed=args.seed)
    print(f"Train: {train_df.flight_id.nunique()} flights, {len(train_df):,} rows")
    print(f"Test:  {test_df.flight_id.nunique()} flights, {len(test_df):,} rows")

    print("\n=== Training anomaly detector ===")
    anomaly_model = train_anomaly_detector(train_df, feature_cols)
    anomaly_metrics = evaluate_anomaly_detector(anomaly_model, test_df, feature_cols)

    print("\n=== Training fault classifier ===")
    classifier_model, labels = train_fault_classifier(train_df, feature_cols)
    classifier_metrics = evaluate_fault_classifier(classifier_model, labels, test_df, feature_cols)

    print("\n=== Training RUL regressor ===")
    rul_model = train_rul_regressor(train_df, feature_cols)
    rul_metrics = evaluate_rul_regressor(rul_model, test_df, feature_cols)

    print("\n=== Example SHAP explanation (one faulted test row) ===")
    faulted_rows = test_df[test_df.fault_label != "healthy"]
    if len(faulted_rows) > 0:
        sample_row = faulted_rows.iloc[[len(faulted_rows) // 2]]
        explanation = explain_prediction(classifier_model, labels, sample_row[feature_cols])
        print(f"Ground truth: {sample_row['fault_label'].values[0]}")
        print(f"Explanation:  {explanation}")

    print("\nSaving models...")
    joblib.dump(anomaly_model, f"{args.outdir}/anomaly_detector.joblib")
    joblib.dump({"model": classifier_model, "labels": labels}, f"{args.outdir}/fault_classifier.joblib")
    joblib.dump(rul_model, f"{args.outdir}/rul_regressor.joblib")
    joblib.dump(feature_cols, f"{args.outdir}/feature_columns.joblib")

    with open(f"{args.outdir}/metrics.json", "w") as f:
        json.dump({
            "anomaly_detector": anomaly_metrics,
            "fault_classifier": classifier_metrics,
            "rul_regressor": rul_metrics,
        }, f, indent=2)

    print(f"\nAll models saved to {args.outdir}/")


if __name__ == "__main__":
    main()
