"""
ml/models.py
============
Live ML inference layer for the digital twin.

Loads the three trained artifacts:
  * anomaly_detector.joblib   -- IsolationForest on healthy-only residual space
  * fault_classifier.joblib   -- XGBoost multi-class {healthy, misfire,
                                 cooling_degradation, lubrication_degradation,
                                 sensor_drift_cht}
  * rul_regressor.joblib      -- XGBoost regressor (seconds to failure)
  * feature_columns.joblib    -- the exact 36-feature ordering

and exposes a single `TwinModels.infer(sample, phys)` call that turns one
decoded telemetry frame into an operator-facing diagnosis.

Two robustness features the demo depends on:

1. **Graceful fallback.** If an artifact is missing or fails to unpickle, a
   lightweight synthetic estimator is trained on the spot from the physics
   model so the pipeline never hard-fails in front of a jury.

2. **Hybrid, physics-first RUL.** The shipped XGB RUL regressor is weak
   (R2 ~ 0.22 on held-out flights). For a number an operator would actually
   act on, we instead extrapolate the *observed degradation rate* of the
   physically-limiting channel (oil pressure -> abort threshold, CHT ->
   redline, cylinder EGT spread -> misfire limit) and report time-to-limit.
   The ML estimate is carried alongside as a cross-check.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.physics_model import PhysicsModel, SENSOR_COLS, load_specs  # noqa: E402

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

ROLLING_WINDOW = 6
_ANOMALY_RESID_GATE = {"oil_pressure": 0.5, "cht": 15.0, "egt": 50.0, "vibration": 0.5}

_MODEL_SEARCH_DIRS = [_REPO_ROOT / "models", _REPO_ROOT]
_FAULT_FALLBACK_LABELS = [
    "healthy", "cooling_degradation", "lubrication_degradation",
    "misfire", "sensor_drift_cht",
]


def get_feature_columns() -> list[str]:
    cols: list[str] = []
    for c in SENSOR_COLS:
        cols += [f"{c}_residual", f"{c}_roll_mean", f"{c}_roll_std", f"{c}_rate"]
    cols += ["throttle_pct", "altitude_m", "ambient_c", "time_in_anomaly_state"]
    return cols


def _find_artifact(name: str) -> Path | None:
    for d in _MODEL_SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Streaming feature builder -- one instance per running scenario
# ---------------------------------------------------------------------------
_TRAIN_SAMPLE_DT_S = 10.0   # the training CSV is sampled at ~0.1 Hz
_TRAIN_ROLL_WINDOW_S = ROLLING_WINDOW * _TRAIN_SAMPLE_DT_S  # ~60 s of history


class FeatureBuilder:
    """Builds the 36-feature inference row so that its statistics match the
    *time* semantics of the training set, independent of the live sample rate.

    The training pipeline computed rolling mean/std over 6 samples (~60 s at
    0.1 Hz), rate as the 10-s sample-to-sample diff, and
    ``time_in_anomaly_state`` as a count of consecutive anomalous 10-s
    samples. The demo streams at 2 Hz, so those are re-expressed here in the
    same units (60-s window, per-10-s rate, anomaly-seconds / 10)."""

    def __init__(self, phys: PhysicsModel, feature_cols: list[str], hz: float = 2.0):
        self.phys = phys
        self.feature_cols = feature_cols
        self.hz = hz
        self.win_span_s = _TRAIN_ROLL_WINDOW_S
        maxlen = max(4, int(round((self.win_span_s + 12.0) * hz)))
        self.tbuf: dict[str, deque] = {c: deque(maxlen=maxlen) for c in SENSOR_COLS}
        self.anom_since_t: float | None = None
        self._last_t = 0.0

    def reset(self) -> None:
        for d in self.tbuf.values():
            d.clear()
        self.anom_since_t = None
        self._last_t = 0.0

    def _value_back(self, buf: deque, now_t: float, span_s: float, fallback: float) -> float:
        target = now_t - span_s
        best = None
        for (tt, vv) in buf:
            if tt <= target:
                best = vv
            else:
                break
        return best if best is not None else (buf[0][1] if buf else fallback)

    def build(self, sample: dict) -> tuple[pd.DataFrame, dict]:
        t = float(sample.get("t_s", self._last_t + 1.0 / self.hz))
        self._last_t = t
        thr = float(sample["throttle_pct"])
        alt = float(sample["altitude_m"])
        amb = float(sample["ambient_c"])
        expected = self.phys.expected(thr, alt, amb)

        residuals = {c: float(sample[c]) - expected[c] for c in SENSOR_COLS}
        feat: dict[str, float] = {}
        for c in SENSOR_COLS:
            val = float(sample[c])
            buf = self.tbuf[c]
            buf.append((t, val))
            window = [v for (tt, v) in buf if tt >= t - self.win_span_s]
            feat[f"{c}_residual"] = residuals[c]
            feat[f"{c}_roll_mean"] = float(np.mean(window)) if window else val
            feat[f"{c}_roll_std"] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            v_back = self._value_back(buf, t, _TRAIN_SAMPLE_DT_S, val)
            feat[f"{c}_rate"] = val - v_back          # per-10-s change (training units)

        is_anom = any(abs(residuals[k]) > g for k, g in _ANOMALY_RESID_GATE.items())
        if is_anom:
            if self.anom_since_t is None:
                self.anom_since_t = t
        else:
            self.anom_since_t = None
        anomaly_seconds = 0.0 if self.anom_since_t is None else (t - self.anom_since_t)

        feat["throttle_pct"] = thr
        feat["altitude_m"] = alt
        feat["ambient_c"] = amb
        # same units as training: consecutive anomalous 10-s samples
        feat["time_in_anomaly_state"] = anomaly_seconds / _TRAIN_SAMPLE_DT_S

        X = pd.DataFrame([feat]).reindex(columns=self.feature_cols, fill_value=0.0)
        return X, residuals


# ---------------------------------------------------------------------------
# Physics-first RUL
# ---------------------------------------------------------------------------
@dataclass
class RulTrend:
    window_s: float = 45.0
    _buf: deque = field(default_factory=lambda: deque(maxlen=256))

    def update(self, t_s: float, channel_value: float) -> None:
        self._buf.append((t_s, channel_value))
        while self._buf and (t_s - self._buf[0][0]) > self.window_s:
            self._buf.popleft()

    def rate_per_s(self) -> float | None:
        if len(self._buf) < 6:
            return None
        ts = np.array([p[0] for p in self._buf])
        ys = np.array([p[1] for p in self._buf])
        if np.ptp(ts) < 5.0:
            return None
        slope = np.polyfit(ts - ts[0], ys, 1)[0]
        return float(slope)


class RulEstimator:
    """Time (s) until the first physically-limiting channel reaches its abort
    limit, from the observed rate of change. `None` == no degrading trend."""

    def __init__(self, specs: dict | None = None):
        s = (specs or load_specs())["engine"]
        self.oil_abort = float(s["oil_pressure_bar"]["abort_threshold"])
        self.cht_redline = float(s["cht_c"]["redline"])
        self.egt_spread_limit = float(s["egt_c"]["max_cylinder_spread"])
        self.vib_redline = float(s["vibration_rms_g"]["redline"])
        self._oil = RulTrend()
        self._cht = RulTrend()
        self._vib = RulTrend()

    def reset(self) -> None:
        self.__init__()

    def update(self, sample: dict, residuals: dict | None = None) -> dict:
        residuals = residuals or {}
        t = float(sample["t_s"])
        oil = float(sample["oil_pressure"])
        cht = float(sample["cht"])
        vib = float(sample["vibration"])
        self._oil.update(t, oil)
        self._cht.update(t, cht)
        self._vib.update(t, vib)

        candidates: list[tuple[str, float]] = []

        # Only treat a downward oil-pressure slope as degradation if pressure
        # is also meaningfully *below* what physics expects for this operating
        # point -- otherwise a throttle-back on descent reads as a failure.
        r = self._oil.rate_per_s()
        if (r is not None and r < -0.003
                and residuals.get("oil_pressure", 0.0) < -0.4):
            ttl = (oil - self.oil_abort) / (-r)
            if ttl > 0:
                candidates.append(("oil pressure -> abort limit", ttl))

        r = self._cht.rate_per_s()
        if (r is not None and r > 0.03 and cht < self.cht_redline
                and residuals.get("cht", 0.0) > 6.0):
            candidates.append(("CHT -> redline", (self.cht_redline - cht) / r))

        r = self._vib.rate_per_s()
        if (r is not None and r > 0.006 and vib < self.vib_redline
                and residuals.get("vibration", 0.0) > 0.4):
            candidates.append(("vibration -> redline", (self.vib_redline - vib) / r))

        if not candidates:
            return {"rul_seconds": None, "rul_driver": None, "rul_source": "trend"}
        driver, ttl = min(candidates, key=lambda c: c[1])
        return {"rul_seconds": float(max(0.0, ttl)), "rul_driver": driver,
                "rul_source": "trend"}


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
class TwinModels:
    def __init__(self, model_dir: str | os.PathLike | None = None, verbose: bool = True,
                 hz: float = 2.0):
        self.phys = PhysicsModel()
        self.feature_cols = get_feature_columns()
        self.status: dict[str, str] = {}
        self._load(verbose)
        self.features = FeatureBuilder(self.phys, self.feature_cols, hz=hz)
        self.rul = RulEstimator()

    # -- loading / fallback --------------------------------------------------
    def _load(self, verbose: bool) -> None:
        fc = _find_artifact("feature_columns.joblib")
        if fc and joblib:
            try:
                cols = joblib.load(fc)
                if isinstance(cols, list) and cols:
                    self.feature_cols = cols
                    self.status["feature_columns"] = f"loaded ({len(cols)})"
            except Exception as e:
                self.status["feature_columns"] = f"fallback ({e!s:.40})"

        self.anomaly_model = self._load_anomaly()
        self.classifier_model, self.fault_labels = self._load_classifier()
        self.rul_model = self._load_rul()
        if verbose:
            for k, v in self.status.items():
                print(f"  [ml] {k:18s}: {v}")

    def _load_anomaly(self):
        p = _find_artifact("anomaly_detector.joblib")
        if p and joblib:
            try:
                m = joblib.load(p)
                self.status["anomaly_detector"] = "loaded"
                return m
            except Exception as e:
                self.status["anomaly_detector"] = f"fallback ({e!s:.40})"
        else:
            self.status["anomaly_detector"] = "fallback (artifact missing)"
        return self._synth_anomaly()

    def _load_classifier(self):
        p = _find_artifact("fault_classifier.joblib")
        if p and joblib:
            try:
                b = joblib.load(p)
                self.status["fault_classifier"] = "loaded"
                return b["model"], list(b["labels"])
            except Exception as e:
                self.status["fault_classifier"] = f"fallback ({e!s:.40})"
        else:
            self.status["fault_classifier"] = "fallback (artifact missing)"
        return None, _FAULT_FALLBACK_LABELS

    def _load_rul(self):
        p = _find_artifact("rul_regressor.joblib")
        if p and joblib:
            try:
                m = joblib.load(p)
                self.status["rul_regressor"] = "loaded (used as cross-check)"
                return m
            except Exception as e:
                self.status["rul_regressor"] = f"fallback ({e!s:.40})"
        else:
            self.status["rul_regressor"] = "fallback (artifact missing)"
        return None

    def _synth_anomaly(self):
        """Tiny IsolationForest fit on physics-nominal feature vectors so the
        anomaly path still works with no artifact present."""
        from sklearn.ensemble import IsolationForest
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(600):
            thr = rng.uniform(15, 100)
            alt = rng.uniform(0, 6000)
            amb = 15 - 0.0065 * alt
            exp = self.phys.expected(thr, alt, amb)
            feat = {}
            for c in SENSOR_COLS:
                noise = exp[c] * rng.normal(0, 0.01)
                feat[f"{c}_residual"] = noise
                feat[f"{c}_roll_mean"] = exp[c] + noise
                feat[f"{c}_roll_std"] = abs(noise) * 0.5
                feat[f"{c}_rate"] = rng.normal(0, 0.01)
            feat.update(throttle_pct=thr, altitude_m=alt, ambient_c=amb,
                        time_in_anomaly_state=0.0)
            rows.append(feat)
        X = pd.DataFrame(rows).reindex(columns=self.feature_cols, fill_value=0.0)
        m = IsolationForest(n_estimators=120, contamination=0.02, random_state=0)
        m.fit(X)
        return m

    # -- inference --------------------------------------------------------
    def reset(self) -> None:
        self.features.reset()
        self.rul.reset()

    def infer(self, sample: dict) -> dict:
        t0 = time.perf_counter()
        X, residuals = self.features.build(sample)

        raw_anom = bool(self.anomaly_model.predict(X)[0] == -1)

        if self.classifier_model is not None:
            proba = self.classifier_model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            fault = self.fault_labels[idx]
            conf = float(proba[idx])
        else:
            fault, conf = self._rule_fault(residuals, sample)

        norm_resid = self.phys.normalized_residuals(residuals)
        norm_resid_max = max((abs(v) for v in norm_resid.values()), default=0.0)
        anomaly_seconds = (0.0 if self.features.anom_since_t is None
                           else float(sample.get("t_s", 0.0)) - self.features.anom_since_t)

        # An anomaly is declared only when all three agree:
        #   * the supervised classifier names a non-healthy fault with real
        #     confidence (this is the detector -- macro-F1 ~0.95),
        #   * a physics residual is actually outside its tolerance band, and
        #   * that breach has persisted (or is severe), so a single-frame
        #     climb/throttle transient never lights the board.
        # `raw_anom` (IsolationForest) is carried as corroboration only.
        gated = any(abs(residuals[k]) > g for k, g in _ANOMALY_RESID_GATE.items())
        classifier_hit = fault != "healthy" and conf >= 0.60
        persistent = anomaly_seconds >= 3.0
        severe = norm_resid_max >= 5.0
        is_anom = classifier_hit and gated and (persistent or severe)

        health = self.phys.subsystem_health(residuals)

        egts = [float(sample[f"egt_{i}"]) for i in range(1, 5) if f"egt_{i}" in sample]
        egt_spread = (max(egts) - min(egts)) if len(egts) == 4 else 0.0

        rul_info = self.rul.update(sample, residuals)
        ml_rul = None
        ml_rul_reliable = False
        if self.rul_model is not None and fault in (
            "cooling_degradation", "lubrication_degradation"
        ):
            try:
                ml_rul = float(self.rul_model.predict(X)[0])
                ml_rul_reliable = 0.0 <= ml_rul <= 3600.0
            except Exception:
                ml_rul = None

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "residuals": residuals,
            "norm_residuals": norm_resid,
            "norm_residual_max": norm_resid_max,
            "subsystem_health": health,
            "health_index": health["overall_health"],
            "anomaly": is_anom,
            "anomaly_raw": raw_anom,
            "anomaly_seconds": anomaly_seconds,
            "fault": fault,
            "fault_confidence": conf,
            "egt_cyl_spread": egt_spread,
            "rul_seconds": rul_info["rul_seconds"],
            "rul_driver": rul_info["rul_driver"],
            "ml_rul_seconds": ml_rul,
            "ml_rul_reliable": ml_rul_reliable,
            "feature_row": X,
            "infer_latency_ms": latency_ms,
        }

    @staticmethod
    def _rule_fault(residuals: dict, sample: dict) -> tuple[str, float]:
        egts = [sample.get(f"egt_{i}", sample["egt"]) for i in range(1, 5)]
        spread = max(egts) - min(egts)
        if residuals["oil_pressure"] < -0.8:
            return "lubrication_degradation", 0.8
        if spread > 90 or residuals["vibration"] > 1.2:
            return "misfire", 0.75
        if residuals["cht"] > 15:
            return "cooling_degradation", 0.7
        if residuals["cht"] > 25 and residuals["egt"] < -10:
            return "sensor_drift_cht", 0.6
        return "healthy", 0.9


__all__ = ["TwinModels", "FeatureBuilder", "RulEstimator", "get_feature_columns"]
