"""
ml/explainability.py
====================
Low-latency, real-time SHAP root-cause attribution.

The training notebook (cell 18/19) computed a one-off SHAP explanation on a
single held-out row. That logic is operationalised here for the live
dashboard:

  * A single `shap.TreeExplainer` is built once at startup over the XGBoost
    fault classifier, primed with a cached background distribution of 50
    representative feature vectors (drawn from the training CSV if present,
    otherwise synthesised from the physics model).
  * `get_live_explanation(feature_row)` returns the top-3 signed feature
    contributions for the *currently predicted* fault class, formatted for an
    operator ("Oil_Pressure_residual  -3.8  (64% of attribution)").
  * Target budget < 80 ms/call. Measured latency is returned with every
    result and the explainer is only invoked when an anomaly is flagged, so
    the UI frame-rate is never gated on SHAP.

If SHAP or the tree model is unavailable, it degrades to a signed-residual
ranking that carries the same operator-facing shape, so the panel is never
blank.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.physics_model import PhysicsModel, SENSOR_COLS  # noqa: E402

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

_PRETTY = {
    "oil_pressure": "Oil Pressure", "oil_temp": "Oil Temp", "cht": "CHT",
    "egt": "EGT", "rpm": "RPM", "vibration": "Vibration RMS",
    "fuel_flow": "Fuel Flow", "battery_voltage": "Battery V",
    "throttle_pct": "Throttle", "altitude_m": "Altitude", "ambient_c": "Ambient T",
    "time_in_anomaly_state": "Time-in-anomaly",
}


def _pretty(feature: str) -> str:
    for raw, nice in _PRETTY.items():
        if feature.startswith(raw):
            suffix = feature[len(raw):].lstrip("_")
            return f"{nice} {suffix}".strip() if suffix else nice
    return feature


class LiveExplainer:
    def __init__(self, classifier_model, fault_labels: list[str],
                 feature_cols: list[str], background: pd.DataFrame | None = None,
                 n_background: int = 50, verbose: bool = True):
        self.model = classifier_model
        self.labels = list(fault_labels)
        self.feature_cols = list(feature_cols)
        self.n_background = n_background
        self.status = "disabled"
        self._explainer = None
        self._bg = self._prepare_background(background)
        self._init_explainer(verbose)

    # -- setup ----------------------------------------------------------
    def _prepare_background(self, background: pd.DataFrame | None) -> pd.DataFrame:
        if background is not None and len(background):
            bg = background.reindex(columns=self.feature_cols, fill_value=0.0)
            if len(bg) > self.n_background:
                bg = bg.sample(self.n_background, random_state=42)
            return bg.reset_index(drop=True)
        return self._synth_background()

    def _synth_background(self) -> pd.DataFrame:
        phys = PhysicsModel()
        rng = np.random.default_rng(7)
        rows = []
        for _ in range(self.n_background):
            thr = rng.uniform(15, 100); alt = rng.uniform(0, 6000)
            amb = 15 - 0.0065 * alt
            exp = phys.expected(thr, alt, amb)
            feat = {}
            for c in SENSOR_COLS:
                feat[f"{c}_residual"] = exp[c] * rng.normal(0, 0.01)
                feat[f"{c}_roll_mean"] = exp[c]
                feat[f"{c}_roll_std"] = abs(exp[c]) * 0.01
                feat[f"{c}_rate"] = rng.normal(0, 0.02)
            feat.update(throttle_pct=thr, altitude_m=alt, ambient_c=amb,
                        time_in_anomaly_state=0.0)
            rows.append(feat)
        return pd.DataFrame(rows).reindex(columns=self.feature_cols, fill_value=0.0)

    def _init_explainer(self, verbose: bool) -> None:
        if shap is None or self.model is None:
            self.status = "fallback (residual ranking)"
            if verbose:
                print(f"  [shap] {self.status}")
            return

        # Two construction strategies, tried in order:
        #   1. interventional perturbation against the cached 50-row background
        #      (the textbook "SHAP against a reference distribution" setup)
        #   2. tree_path_dependent -- no background needed; the standard fast
        #      TreeSHAP path, and the one that tolerates models trained with
        #      categorical handling.
        attempts = [
            ("interventional+bg",
             lambda: shap.TreeExplainer(self.model, data=self._bg,
                                        feature_perturbation="interventional")),
            ("tree_path_dependent",
             lambda: shap.TreeExplainer(self.model)),
        ]
        for name, make in attempts:
            try:
                expl = make()
                t0 = time.perf_counter()
                expl.shap_values(self._bg.iloc[[0]], check_additivity=False)
                warm_ms = (time.perf_counter() - t0) * 1000.0
                self._explainer = expl
                self.status = f"ready [{name}] (warmup {warm_ms:.0f} ms, bg={len(self._bg)})"
                break
            except Exception as e:
                self.status = f"fallback ({name}: {e!s:.40})"
        if verbose:
            print(f"  [shap] {self.status}")

    # -- inference -----------------------------------------------------
    def get_live_explanation(self, feature_row: pd.DataFrame,
                             predicted_fault: str | None = None,
                             top_n: int = 3) -> dict:
        t0 = time.perf_counter()
        X = feature_row.reindex(columns=self.feature_cols, fill_value=0.0)

        if self._explainer is not None:
            try:
                res = self._explain_shap(X, predicted_fault, top_n)
                res["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                res["method"] = "TreeSHAP"
                return res
            except Exception:
                pass

        res = self._explain_residual(X, top_n)
        res["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        res["method"] = "residual-rank"
        return res

    def _explain_shap(self, X: pd.DataFrame, predicted_fault, top_n) -> dict:
        if predicted_fault is not None and predicted_fault in self.labels:
            cls = self.labels.index(predicted_fault)
        else:
            cls = int(np.argmax(self.model.predict_proba(X)[0]))

        sv = self._explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            vals = np.asarray(sv[cls])[0]
        else:
            sv = np.asarray(sv)
            vals = sv[0, :, cls] if sv.ndim == 3 else sv[0]

        order = np.argsort(-np.abs(vals))[:top_n]
        total = float(np.sum(np.abs(vals))) or 1e-9
        drivers = []
        for i in order:
            col = self.feature_cols[i]
            drivers.append({
                "feature": col,
                "label": _pretty(col),
                "value": float(X.iloc[0, i]),
                "shap": float(vals[i]),
                "direction": "raises" if vals[i] > 0 else "lowers",
                "impact_pct": round(100.0 * abs(vals[i]) / total, 1),
            })
        return {"class": self.labels[cls], "drivers": drivers}

    def _explain_residual(self, X: pd.DataFrame, top_n) -> dict:
        resid_cols = [c for c in self.feature_cols if c.endswith("_residual")]
        tol = PhysicsModel().specs["residual_tolerance"]
        scored = []
        for c in resid_cols:
            base = c[:-len("_residual")]
            v = float(X.iloc[0][c])
            scored.append((c, v, abs(v) / max(float(tol.get(base, 1.0)), 1e-6)))
        scored.sort(key=lambda s: -s[2])
        top = scored[:top_n]
        total = sum(s[2] for s in top) or 1e-9
        drivers = [{
            "feature": c, "label": _pretty(c), "value": v, "shap": v,
            "direction": "raises" if v > 0 else "lowers",
            "impact_pct": round(100.0 * s / total, 1),
        } for c, v, s in top]
        return {"class": None, "drivers": drivers}


def make_background(feature_cols: list[str], n: int = 50) -> pd.DataFrame | None:
    """Cache a background distribution of representative *healthy* feature
    vectors, generated from the same scenario physics the demo runs and
    passed through the same feature transform used at inference time
    (`ml.dataset.build_features_batch`). Consistent with `FeatureBuilder`
    by construction; no external CSV needed."""
    try:
        from ml.dataset import generate_flights, build_features_batch
        raw = generate_flights(6, seed=999)
        feats = build_features_batch(raw)
        healthy = feats[feats["fault_label"] == "healthy"]
        if healthy.empty:
            healthy = feats
        take = healthy.sample(min(len(healthy), n), random_state=1)
        return take.reindex(columns=feature_cols, fill_value=0.0).reset_index(drop=True)
    except Exception:
        return None


# Back-compat alias (older callers passed a csv path first arg, now ignored).
def load_background_from_csv(_path, feature_cols: list[str], n: int = 50):
    return make_background(feature_cols, n=n)


__all__ = ["LiveExplainer", "make_background", "load_background_from_csv"]
