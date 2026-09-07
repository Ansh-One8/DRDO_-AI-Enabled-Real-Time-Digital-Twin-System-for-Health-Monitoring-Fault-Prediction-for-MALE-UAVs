"""
core/physics_model.py
=====================
The **Physics-Informed Digital Twin baseline**.

A first-principles, steady-state thermodynamic model of the Rotax-914-class
piston engine. Given the *commanded* operating point (throttle, altitude,
ambient temperature) it predicts what every sensor *should* read on a healthy
engine:

    EGT_nominal        = f(throttle, load, altitude)
    CHT_nominal        = f(rpm, airspeed/ram-cooling, ambient)
    OilPressure_nominal= f(rpm, oil_temp)
    ...

The live **residual vector**

    R(t) = X_measured(t) - X_physics(t)

is what the dashboard's "Twin Cognition" view renders: the twin is not
threshold-checking raw numbers, it is tracking *structural divergence from
physics*. A fault shows up as a residual growing outside its tolerance band
long before the raw value hits a redline.

This module is the single source of truth for the expected-value model. Both
the offline training pipeline (`train_pipeline.add_features`) and the live
inference path (`ml/models.py`) import `expected_engine_values` from here, so
the residual features a model was trained on are computed by the exact same
code that runs in the demo.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "engine_specs.yaml"

SENSOR_COLS = [
    "rpm", "cht", "egt", "oil_pressure", "oil_temp",
    "fuel_flow", "vibration", "battery_voltage",
]


@functools.lru_cache(maxsize=1)
def load_specs(path: str | os.PathLike | None = None) -> dict:
    """Parsed config/engine_specs.yaml (cached)."""
    with open(path or _CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Numeric constants -- pulled from YAML once, kept as module scalars so the
# vectorized math below stays readable and fast. These MUST stay numerically
# identical to what the shipped models were trained against.
# ---------------------------------------------------------------------------
_S = load_specs()["engine"]
IDLE_RPM = float(_S["rpm"]["idle"])
CRUISE_RPM = float(_S["rpm"]["cruise"])
MAX_RPM = float(_S["rpm"]["max"])
MAP_IDLE = float(_S["manifold_pressure_inHg"]["idle"])
MAP_MAX = float(_S["manifold_pressure_inHg"]["max"])
CHT_IDLE = float(_S["cht_c"]["idle"])
CHT_CRUISE = float(_S["cht_c"]["cruise"])
CHT_REDLINE = float(_S["cht_c"]["redline"])
EGT_IDLE = float(_S["egt_c"]["idle"])
EGT_CRUISE = float(_S["egt_c"]["cruise"])
EGT_REDLINE = float(_S["egt_c"]["redline"])
OIL_P_MIN = float(_S["oil_pressure_bar"]["min"])
OIL_P_MAX = float(_S["oil_pressure_bar"]["max"])
OIL_T_IDLE = float(_S["oil_temp_c"]["idle"])
OIL_T_CRUISE = float(_S["oil_temp_c"]["cruise"])
FUEL_FLOW_MAX = float(_S["fuel_flow_lph"]["max"])
VIB_BASELINE = float(_S["vibration_rms_g"]["baseline"])
BATTERY_NOMINAL = float(_S["battery_voltage_v"]["nominal"])
TURBO_CRITICAL_ALT_M = float(_S["turbo"]["critical_altitude_m"])
CRUISE_LOAD_REF = 0.7  # engine-load fraction that corresponds to "cruise" temps

_ATM = load_specs()["atmosphere"]
ISA_T0_C = float(_ATM["isa_sea_level_temp_c"])

_TOL = load_specs()["residual_tolerance"]
_SUBSYS = load_specs()["subsystems"]


# ---------------------------------------------------------------------------
# Steady-state expected-value model (vectorized; accepts scalars or arrays)
# ---------------------------------------------------------------------------
def expected_engine_values(throttle_pct, altitude_m, ambient_c) -> pd.DataFrame:
    """Steady-state physics expectation for a *healthy* engine.

    Turbocharged: boost holds rated MAP up to the turbo critical altitude,
    then falls off. Two-segment temperature interpolation anchored to the
    real idle / cruise / redline spec values (a single multiplier overshoots
    the narrow real gap between cruise and max EGT).

    Returns a DataFrame with one ``<sensor>_expected`` column per SENSOR_COLS.
    """
    thr = np.asarray(throttle_pct, dtype=float) / 100.0
    altitude_m = np.asarray(altitude_m, dtype=float)
    ambient_c = np.asarray(ambient_c, dtype=float)

    pressure_pa = 101325.0 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    _ = pressure_pa  # kept for readability / future density-based terms

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
    egt_expected = np.where(
        load_factor <= CRUISE_LOAD_REF,
        EGT_IDLE + frac_low * (EGT_CRUISE - EGT_IDLE),
        EGT_CRUISE + frac_high * (EGT_REDLINE - EGT_CRUISE),
    )
    cht_expected = np.where(
        load_factor <= CRUISE_LOAD_REF,
        CHT_IDLE + frac_low * (CHT_CRUISE - CHT_IDLE),
        CHT_CRUISE + frac_high * (CHT_REDLINE - CHT_CRUISE),
    )

    ambient_offset = ambient_c - ISA_T0_C
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
        "rpm_expected": np.atleast_1d(rpm_expected),
        "cht_expected": np.atleast_1d(cht_expected),
        "egt_expected": np.atleast_1d(egt_expected),
        "oil_pressure_expected": np.atleast_1d(oil_pressure_expected),
        "oil_temp_expected": np.atleast_1d(oil_temp_expected),
        "fuel_flow_expected": np.atleast_1d(fuel_flow_expected),
        "vibration_expected": np.atleast_1d(vibration_expected),
        "battery_voltage_expected": np.atleast_1d(battery_expected),
    })


class PhysicsModel:
    """Object wrapper around the expected-value model, plus residual scoring
    and subsystem-health roll-up. One instance is created at dashboard start
    and reused every frame."""

    def __init__(self, specs: dict | None = None):
        self.specs = specs or load_specs()
        self.tol = self.specs["residual_tolerance"]
        self.subsystems = self.specs["subsystems"]

    # -- expected + residual ------------------------------------------------
    def expected(self, throttle_pct: float, altitude_m: float, ambient_c: float) -> dict:
        row = expected_engine_values(throttle_pct, altitude_m, ambient_c).iloc[0]
        return {c: float(row[f"{c}_expected"]) for c in SENSOR_COLS}

    def residuals(self, measured: dict, throttle_pct: float,
                  altitude_m: float, ambient_c: float) -> dict:
        """R(t) = measured - physics, per sensor (only sensors present in
        ``measured`` are returned)."""
        exp = self.expected(throttle_pct, altitude_m, ambient_c)
        return {c: float(measured[c]) - exp[c] for c in SENSOR_COLS if c in measured}

    def normalized_residuals(self, residuals: dict) -> dict:
        """Residual divided by its tolerance band -- 0 = on model, 1 = at the
        edge of 'healthy', >1 = physics-breaking. This is what the Twin
        Cognition radar plots."""
        return {k: v / max(float(self.tol.get(k, 1.0)), 1e-6) for k, v in residuals.items()}

    # -- health roll-up ---------------------------------------------------
    @staticmethod
    def _residual_to_score(residual: float, tolerance: float) -> float:
        ratio = abs(residual) / max(tolerance, 1e-6)
        return max(0.0, min(100.0, 100.0 * np.exp(-0.5 * ratio ** 2 / 4.0)))

    def subsystem_health(self, residuals: dict) -> dict:
        """0-100 health index per subsystem + a combined 'overall'."""
        scores = {}
        for name, params in self.subsystems.items():
            per = [self._residual_to_score(residuals[p], float(self.tol[p]))
                   for p in params if p in residuals]
            scores[name] = float(np.mean(per)) if per else 100.0
        scores["overall_health"] = float(np.mean(list(scores.values()))) if scores else 100.0
        return scores


__all__ = ["PhysicsModel", "expected_engine_values", "load_specs", "SENSOR_COLS"]
