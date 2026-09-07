"""
Digital Twin Core -- v2
========================
Everything v1 did (physics health index, ML inference, SQLite logging,
enriched MQTT republish) PLUS:

  1. Multi-UAV support: telemetry records carry a "uav_id" field. Run
     multiple mqtt_publisher.py processes with different --uav-id, all
     publishing to the same topic -- this one core process demuxes them
     and tracks each independently.
  2. Session tracking: every start of this script stamps a new session_id,
     so later you can compare "this mission" against past ones.
  3. Persistent alerts_log table: alerts are no longer print-only.
  4. SHAP explainability: when the ML model flags an anomaly, computes a
     human-readable "why" (reusing train_pipeline.explain_prediction).
  5. Maintenance advisory: rule-based (fault, severity, RUL) -> recommended
     action + urgency.
  6. Warm-up-aware ML anomaly flag: a cold engine reads like a big
     deviation from every "expected steady-state" baseline, so the raw
     anomaly detector will very often fire in the first ~400s of a flight.
     That's expected physics, not a fault -- so `ml_anomaly` is now
     suppressed (forced False, with a note) until WARMUP_GRACE_S, matching
     how physics-based alerts were already being suppressed. Nothing about
     the model changes; only what we call a state versus noise before the
     engine is at temperature.

Usage:
    python digital_twin_core.py --broker-host localhost --topic uav/engine/telemetry
"""

import argparse
import json
import math
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
import shap
import paho.mqtt.client as mqtt

from engine_simulator_v2 import EngineSpecs
from train_pipeline import (
    expected_engine_values as ml_expected_engine_values,
    SENSOR_COLS,
    ROLLING_WINDOW as ML_ROLLING_WINDOW,
    get_feature_columns,
)

DEFAULT_TOPIC = "uav/engine/telemetry"
DEFAULT_ENRICHED_TOPIC = "uav/engine/enriched"
DEFAULT_DB = "digital_twin_history.db"
DEFAULT_UAV_ID = "UAV-01"
ROLLING_WINDOW = 30
MODEL_DIR = "."
WARMUP_GRACE_S = 400.0


def expected_healthy_values(specs: EngineSpecs, throttle_pct: float, altitude_m: float, ambient_c: float) -> dict:
    thr = throttle_pct / 100.0
    temp_k = ambient_c + 273.15
    pressure_pa = 101325 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    ambient_pressure_inHg = pressure_pa / 3386.39
    density_ratio = pressure_pa / (287.05 * temp_k) / 1.225

    max_available_map = min(specs.map_max, ambient_pressure_inHg)
    map_expected = specs.map_idle + thr * (max_available_map - specs.map_idle) * density_ratio

    if thr <= 0.5:
        rpm_expected = specs.idle_rpm + (thr / 0.5) * (specs.cruise_rpm - specs.idle_rpm)
    else:
        rpm_expected = specs.cruise_rpm + ((thr - 0.5) / 0.5) * (specs.max_rpm - specs.cruise_rpm)

    engine_load_factor = max(0.0, (map_expected / specs.map_max) * (rpm_expected / specs.max_rpm))

    egt_expected = specs.egt_idle + (specs.egt_cruise - specs.egt_idle) * engine_load_factor * 1.5
    cht_expected = specs.cht_idle + (specs.cht_cruise - specs.cht_idle) * engine_load_factor * 1.3
    ambient_offset = ambient_c - 15.0
    ram_air_cooling = thr * 15.0
    cht_expected += ambient_offset - ram_air_cooling

    oil_temp_expected = specs.oil_temp_idle + (specs.oil_temp_cruise - specs.oil_temp_idle) * engine_load_factor * 1.2
    oil_temp_expected += ambient_offset * 0.5 - ram_air_cooling * 0.5

    rpm_factor = rpm_expected / specs.max_rpm
    oil_pressure_expected = specs.oil_pressure_min + (specs.oil_pressure_max - specs.oil_pressure_min) * rpm_factor

    vibration_expected = specs.vibration_baseline * (0.5 + 0.5 * engine_load_factor)
    fuel_flow_expected = specs.fuel_flow_max * engine_load_factor * density_ratio
    battery_expected = specs.battery_voltage_nominal - (0.6 if rpm_expected < 2000 else 0)

    return {
        "rpm": rpm_expected, "cht": cht_expected, "egt": egt_expected,
        "oil_pressure": oil_pressure_expected, "oil_temp": oil_temp_expected,
        "fuel_flow": fuel_flow_expected, "vibration": vibration_expected,
        "battery_voltage": battery_expected,
    }


TOLERANCE = {
    "rpm": 200.0, "cht": 12.0, "egt": 40.0, "oil_pressure": 0.5,
    "oil_temp": 8.0, "fuel_flow": 1.5, "vibration": 0.3, "battery_voltage": 0.4,
}
SUBSYSTEM_PARAMS = {
    "thermal_health": ["cht", "egt"],
    "lubrication_health": ["oil_pressure", "oil_temp"],
    "combustion_health": ["vibration", "rpm"],
    "electrical_health": ["battery_voltage"],
}
ML_ANOMALY_THRESHOLDS = {"oil_pressure": 0.5, "cht": 15.0, "egt": 50.0, "vibration": 0.5}

ADVISORY_RULES = {
    "misfire": {"action": "Inspect ignition system (plugs, leads, magneto timing) and injector "
                           "spray pattern on next ground check.", "base_urgency": "medium"},
    "cooling_degradation": {"action": "Inspect cooling fins/ducting for obstruction or damage. "
                                       "Reduce throttle / consider RTB if CHT keeps rising.", "base_urgency": "high"},
    "lubrication_degradation": {"action": "Inspect oil system for leaks, pump wear, or filter blockage. "
                                           "Monitor oil pressure closely; prepare to abort mission.", "base_urgency": "high"},
    "sensor_drift_cht": {"action": "Cross-check CHT sensor against a secondary probe on next ground check; "
                                    "sensor may need recalibration.", "base_urgency": "low"},
    "healthy": {"action": "No action required. Continue normal monitoring.", "base_urgency": "none"},
}


def get_advisory(fault_prediction: str, overall_health: float, rul_seconds) -> dict:
    rule = ADVISORY_RULES.get(fault_prediction, {
        "action": "Unrecognized condition -- flag for manual engineering review.", "base_urgency": "medium"})
    urgency = rule["base_urgency"]
    if urgency != "none":
        if (rul_seconds is not None and rul_seconds < 300) or overall_health < 30:
            urgency = "critical"
        elif (rul_seconds is not None and rul_seconds < 900) or overall_health < 50:
            urgency = "high" if urgency != "critical" else urgency
    return {"urgency": urgency, "action": rule["action"]}


def residual_to_score(residual: float, tolerance: float) -> float:
    ratio = abs(residual) / max(tolerance, 1e-6)
    return max(0.0, min(100.0, 100.0 * math.exp(-0.5 * ratio ** 2 / 4.0)))


@dataclass
class FlightState:
    history: dict = field(default_factory=lambda: {p: deque(maxlen=ROLLING_WINDOW) for p in TOLERANCE})
    anomaly_run_length: int = -1
    last_alert_state: dict = field(default_factory=dict)


class DigitalTwinCore:
    def __init__(self, db_path: str = DEFAULT_DB, model_dir: str = MODEL_DIR):
        self.specs = EngineSpecs()
        self.fleet: dict = {}
        self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.conn = sqlite3.connect(db_path)
        self._init_db()

        self.anomaly_model = joblib.load(f"{model_dir}/anomaly_detector.joblib")
        clf_bundle = joblib.load(f"{model_dir}/fault_classifier.joblib")
        self.classifier_model = clf_bundle["model"]
        self.fault_labels = clf_bundle["labels"]
        self.rul_model = joblib.load(f"{model_dir}/rul_regressor.joblib")
        self.feature_cols = joblib.load(f"{model_dir}/feature_columns.joblib")

        if self.feature_cols != get_feature_columns():
            raise RuntimeError(
                "feature_columns.joblib doesn't match train_pipeline.get_feature_columns() -- "
                "retrain (python train_pipeline.py --data <csv> --outdir .) before running this."
            )
        self.shap_explainer = shap.TreeExplainer(self.classifier_model)
        print(f"Session started: {self.session_id}")

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, uav_id TEXT,
                received_at REAL, t_s REAL, phase TEXT,
                throttle_pct REAL, altitude_m REAL, ambient_c REAL,
                rpm REAL, cht REAL, egt REAL,
                oil_pressure REAL, oil_temp REAL,
                fuel_flow REAL, vibration REAL, battery_voltage REAL,
                fault_label TEXT, remaining_useful_life REAL,
                thermal_health REAL, lubrication_health REAL,
                combustion_health REAL, electrical_health REAL, overall_health REAL,
                ml_anomaly INTEGER, ml_fault_prediction TEXT,
                ml_fault_confidence REAL, ml_rul_seconds REAL, warming_up INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, uav_id TEXT, received_at REAL, t_s REAL,
                severity TEXT, source TEXT, subsystem TEXT, message TEXT,
                explanation TEXT, recommended_action TEXT, rul_seconds REAL
            )
        """)
        self.conn.commit()

    def _get_state(self, uav_id: str) -> FlightState:
        if uav_id not in self.fleet:
            self.fleet[uav_id] = FlightState()
        return self.fleet[uav_id]

    def _build_ml_features(self, state: FlightState, record: dict) -> pd.DataFrame:
        expected_df = ml_expected_engine_values(
            np.array([record["throttle_pct"]]), np.array([record["altitude_m"]]), np.array([record["ambient_c"]]))
        expected = {col: expected_df[f"{col}_expected"].iloc[0] for col in SENSOR_COLS}

        residuals, feat = {}, {}
        for col in SENSOR_COLS:
            residual = record[col] - expected[col]
            residuals[col] = residual
            recent = [v for (_, v) in state.history[col]][-ML_ROLLING_WINDOW:]
            feat[f"{col}_residual"] = residual
            feat[f"{col}_roll_mean"] = float(np.mean(recent)) if recent else record[col]
            feat[f"{col}_roll_std"] = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
            feat[f"{col}_rate"] = (recent[-1] - recent[-2]) if len(recent) >= 2 else 0.0

        is_anomalous = (
            abs(residuals["oil_pressure"]) > ML_ANOMALY_THRESHOLDS["oil_pressure"]
            or abs(residuals["cht"]) > ML_ANOMALY_THRESHOLDS["cht"]
            or abs(residuals["egt"]) > ML_ANOMALY_THRESHOLDS["egt"]
            or abs(residuals["vibration"]) > ML_ANOMALY_THRESHOLDS["vibration"]
        )
        state.anomaly_run_length = state.anomaly_run_length + 1 if is_anomalous else 0
        feat["throttle_pct"] = record["throttle_pct"]
        feat["altitude_m"] = record["altitude_m"]
        feat["ambient_c"] = record["ambient_c"]
        feat["time_in_anomaly_state"] = state.anomaly_run_length
        return pd.DataFrame([feat])[self.feature_cols]

    def _explain(self, X: pd.DataFrame, fault_idx: int) -> str:
        shap_values = self.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            values_for_class = shap_values[fault_idx][0]
        else:
            values_for_class = shap_values[0, :, fault_idx] if shap_values.ndim == 3 else shap_values[0]
        contrib = sorted(zip(X.columns.tolist(), values_for_class), key=lambda x: -abs(x[1]))[:3]
        total_abs = sum(abs(v) for _, v in contrib) or 1e-9
        return ", ".join(f"{abs(v)/total_abs:.0%} due to {name}" for name, v in contrib)

    def _run_ml_inference(self, state: FlightState, record: dict, warming_up: bool) -> dict:
        X = self._build_ml_features(state, record)
        raw_is_anomaly = bool(self.anomaly_model.predict(X)[0] == -1)
        proba = self.classifier_model.predict_proba(X)[0]
        fault_idx = int(np.argmax(proba))
        fault_prediction = self.fault_labels[fault_idx]
        fault_confidence = float(proba[fault_idx])

        # Suppress the anomaly flag during warm-up: a cold engine legitimately
        # deviates from the steady-state baseline the models were trained
        # against. This mirrors the physics-alert warm-up grace already in
        # place, applied consistently to the ML flag shown in the UI.
        is_anomaly = raw_is_anomaly and not warming_up

        rul_seconds = None
        if fault_prediction in ("cooling_degradation", "lubrication_degradation"):
            rul_seconds = float(self.rul_model.predict(X)[0])

        explanation = self._explain(X, fault_idx) if is_anomaly else ""

        return {
            "ml_anomaly": is_anomaly,
            "ml_fault_prediction": fault_prediction,
            "ml_fault_confidence": round(fault_confidence, 3),
            "ml_rul_seconds": round(rul_seconds, 1) if rul_seconds is not None else None,
            "ml_explanation": explanation,
            "warming_up": warming_up,
        }

    def process(self, record: dict) -> dict:
        uav_id = record.get("uav_id", DEFAULT_UAV_ID)
        state = self._get_state(uav_id)
        warming_up = record["t_s"] < WARMUP_GRACE_S

        expected = expected_healthy_values(self.specs, record["throttle_pct"], record["altitude_m"], record["ambient_c"])
        residuals = {p: record[p] - expected[p] for p in TOLERANCE if p in record}

        subsystem_scores = {}
        for subsystem, params in SUBSYSTEM_PARAMS.items():
            param_scores = [residual_to_score(residuals[p], TOLERANCE[p]) for p in params if p in residuals]
            subsystem_scores[subsystem] = sum(param_scores) / len(param_scores) if param_scores else 100.0
        overall_health = sum(subsystem_scores.values()) / len(subsystem_scores)

        for p, val in record.items():
            if p in state.history:
                state.history[p].append((record["t_s"], val))

        ml_results = self._run_ml_inference(state, record, warming_up)
        advisory = get_advisory(ml_results["ml_fault_prediction"], overall_health, ml_results["ml_rul_seconds"])

        enriched = {
            "uav_id": uav_id, "session_id": self.session_id,
            **record, **subsystem_scores,
            "overall_health": round(overall_health, 1),
            **ml_results,
            "advisory_urgency": advisory["urgency"],
            "advisory_action": advisory["action"],
        }

        self._log_to_db(enriched)
        if not warming_up:
            self._check_alerts(state, uav_id, subsystem_scores, ml_results, advisory, record)
        return enriched

    def _log_to_db(self, e: dict):
        self.conn.execute("""
            INSERT INTO telemetry_history (
                session_id, uav_id, received_at, t_s, phase, throttle_pct, altitude_m, ambient_c,
                rpm, cht, egt, oil_pressure, oil_temp, fuel_flow, vibration, battery_voltage,
                fault_label, remaining_useful_life,
                thermal_health, lubrication_health, combustion_health, electrical_health, overall_health,
                ml_anomaly, ml_fault_prediction, ml_fault_confidence, ml_rul_seconds, warming_up
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            e["session_id"], e["uav_id"], time.time(), e["t_s"], e["phase"], e["throttle_pct"],
            e["altitude_m"], e["ambient_c"], e["rpm"], e["cht"], e["egt"], e["oil_pressure"],
            e["oil_temp"], e["fuel_flow"], e["vibration"], e["battery_voltage"],
            e["fault_label"], e["remaining_useful_life"],
            e["thermal_health"], e["lubrication_health"], e["combustion_health"],
            e["electrical_health"], e["overall_health"],
            int(e["ml_anomaly"]), e["ml_fault_prediction"], e["ml_fault_confidence"], e["ml_rul_seconds"],
            int(e["warming_up"]),
        ))
        self.conn.commit()

    def _log_alert(self, uav_id, t_s, severity, source, subsystem, message, explanation, action, rul_seconds):
        self.conn.execute("""
            INSERT INTO alerts_log (session_id, uav_id, received_at, t_s, severity, source, subsystem,
                message, explanation, recommended_action, rul_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (self.session_id, uav_id, time.time(), t_s, severity, source, subsystem,
              message, explanation, action, rul_seconds))
        self.conn.commit()

    def _check_alerts(self, state, uav_id, subsystem_scores, ml_results, advisory, record):
        ALERT_THRESHOLD = 60.0
        for subsystem, score in subsystem_scores.items():
            was_degraded = state.last_alert_state.get(subsystem, False)
            is_degraded = score < ALERT_THRESHOLD
            if is_degraded and not was_degraded:
                msg = f"{subsystem} dropped to {score:.1f}/100"
                print(f"  !! ALERT (physics) [{uav_id}]: {msg} at t={record['t_s']:.0f}s "
                      f"(ground-truth fault={record['fault_label']})")
                self._log_alert(uav_id, record["t_s"], "medium", "physics", subsystem, msg, "", "", None)
            state.last_alert_state[subsystem] = is_degraded

        was_ml = state.last_alert_state.get("_ml_anomaly", False)
        if ml_results["ml_anomaly"] and not was_ml:
            msg = (f"anomaly detected, predicted fault={ml_results['ml_fault_prediction']} "
                   f"(confidence={ml_results['ml_fault_confidence']:.0%})")
            print(f"  !! ALERT (ML) [{uav_id}]: {msg} at t={record['t_s']:.0f}s. "
                  f"Why: {ml_results['ml_explanation']}. Advisory: {advisory['urgency']} -- {advisory['action']}")
            self._log_alert(uav_id, record["t_s"], advisory["urgency"], "ml", ml_results["ml_fault_prediction"],
                             msg, ml_results["ml_explanation"], advisory["action"], ml_results["ml_rul_seconds"])
        state.last_alert_state["_ml_anomaly"] = ml_results["ml_anomaly"]


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to broker (reason_code={reason_code}). Subscribing to '{userdata['topic']}'...")
    client.subscribe(userdata["topic"])


def on_message(client, userdata, msg):
    core: DigitalTwinCore = userdata["core"]
    try:
        record = json.loads(msg.payload)
    except json.JSONDecodeError:
        print(f"  !! Skipped malformed message on {msg.topic}")
        return
    try:
        enriched = core.process(record)
    except (KeyError, TypeError) as e:
        print(f"  !! Skipped record missing expected fields: {e}")
        return
    userdata["pub_client"].publish(userdata["enriched_topic"], json.dumps(enriched), qos=0)
    userdata["count"] += 1
    if userdata["count"] % 20 == 0:
        print(f"[{userdata['count']:6d}] uav={enriched['uav_id']:8s} t={enriched['t_s']:8.1f}s  "
              f"overall={enriched['overall_health']:5.1f}  ml_anomaly={enriched['ml_anomaly']}  "
              f"ml_fault={enriched['ml_fault_prediction']}"
              f"{'  [WARMING UP]' if enriched['warming_up'] else ''}")


def main():
    parser = argparse.ArgumentParser(description="Digital Twin Core -- fleet-aware physics+ML inference engine")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--enriched-topic", default=DEFAULT_ENRICHED_TOPIC)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    core = DigitalTwinCore(db_path=args.db, model_dir=args.model_dir)

    pub_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="digital-twin-core-pub")
    pub_client.connect(args.broker_host, args.broker_port)
    pub_client.loop_start()

    userdata = {"core": core, "topic": args.topic, "count": 0,
                "pub_client": pub_client, "enriched_topic": args.enriched_topic}

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="digital-twin-core", userdata=userdata)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to broker at {args.broker_host}:{args.broker_port} ...")
    client.connect(args.broker_host, args.broker_port)
    print(f"Logging enriched telemetry + ML predictions to '{args.db}'")
    print(f"Republishing enriched records to '{args.enriched_topic}' for the dashboard")
    print("Waiting for messages... (Ctrl+C to stop)\n")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. Processed {userdata['count']} messages.")
    finally:
        pub_client.loop_stop()
        pub_client.disconnect()


if __name__ == "__main__":
    main()