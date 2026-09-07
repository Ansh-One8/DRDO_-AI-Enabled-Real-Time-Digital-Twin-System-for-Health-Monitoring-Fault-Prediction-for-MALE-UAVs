"""
core/scenario_runner.py
=======================
Deterministic **flight-scenario generator** -- the "unseen dynamic data" the
twin is evaluated against.

The old pipeline replayed the same CSV the models were trained on. This module
instead *generates* fresh telemetry every run from the physics simulator, with
a fixed seed per scenario so a demo is byte-for-byte reproducible but is never
a row the model has seen.

Each scenario is a compressed 3-minute (180 s) mission that walks the full
lifecycle:

    TAXI (0-15s) -> CLIMB (15-50s) -> CRUISE (50-95s)
                 -> LOITER (95-150s) -> EMERGENCY RECOVERY (150-180s)

Four scenarios:

  1. nominal_loiter          -- everything in band under ISA conditions.
  2. altitude_cooling_degradation -- climb pushed higher than planned; thinning
     air + partial duct blockage -> CHT drifts up, combustion gets rough.
  3. lubrication_loss         -- oil-pump / bearing failure: oil pressure
     collapses, vibration spikes, RUL falls fast toward mission abort.
  4. injector_misfire         -- one injector clogs mid-cruise: a single
     cylinder's EGT diverges hard from the other three -> instant anomaly.

Telemetry parameters emitted (engineering units):
    RPM, CHT (mean + CHT_Cyl1..4), EGT (mean + EGT_Cyl1..4), Oil_Pressure,
    Oil_Temp, Fuel_Flow_LPH, Manifold_Pressure, Vibration_RMS,
    Altitude_ft, Airspeed_kts, Battery_Voltage, plus mission_stage / t_s /
    fault_label / remaining_useful_life (ground truth, for validation only).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

# Allow running as a bare script (`python core/scenario_runner.py`) as well as
# an imported package module -- engine_simulator_v2 lives at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engine_simulator_v2 import (
    EngineSpecs, EnginePhysicsModel, FaultInjector, FaultScenario, FaultType,
)

# ---------------------------------------------------------------------------
# Compressed 3-minute mission lifecycle
# ---------------------------------------------------------------------------
MISSION_DURATION_S = 180.0
STAGES = [
    ("TAXI", 0.0, 15.0),
    ("CLIMB", 15.0, 50.0),
    ("CRUISE", 50.0, 95.0),
    ("LOITER", 95.0, 150.0),
    ("EMERGENCY_RECOVERY", 150.0, 180.0),
]


def mission_stage(t: float) -> str:
    for name, lo, hi in STAGES:
        if lo <= t < hi:
            return name
    return STAGES[-1][0]


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, u))


@dataclass
class MissionShape:
    """Commanded operating point vs. mission time for the compressed profile.
    Scenario-specific overrides (e.g. an over-altitude climb) are layered on
    top via `alt_bias_fn`."""
    cruise_altitude_m: float = 4200.0
    hot_weather: bool = False
    alt_bias_fn: Callable[[float], float] = field(default=lambda t: 0.0)
    throttle_bias_fn: Callable[[float], float] = field(default=lambda t: 0.0)

    def throttle_at(self, t: float) -> float:
        stage = mission_stage(t)
        base = {
            "TAXI": 15.0, "CLIMB": 88.0, "CRUISE": 66.0,
            "LOITER": 55.0, "EMERGENCY_RECOVERY": 35.0,
        }[stage]
        # small deterministic breathing so charts aren't dead-flat
        base += 1.2 * math.sin(t / 7.0)
        return float(np.clip(base + self.throttle_bias_fn(t), 0.0, 100.0))

    def altitude_at(self, t: float) -> float:
        stage = mission_stage(t)
        if stage == "TAXI":
            alt = 0.0
        elif stage == "CLIMB":
            alt = _lerp(0.0, self.cruise_altitude_m, (t - 15.0) / 35.0)
        elif stage in ("CRUISE", "LOITER"):
            alt = self.cruise_altitude_m + 25.0 * math.sin(t / 18.0)
        else:  # EMERGENCY_RECOVERY -- descend
            alt = _lerp(self.cruise_altitude_m, self.cruise_altitude_m * 0.55,
                        (t - 150.0) / 30.0)
        return max(0.0, alt + self.alt_bias_fn(t))

    def ambient_at(self, t: float) -> float:
        sl = 40.0 if self.hot_weather else 15.0
        return sl - 0.0065 * self.altitude_at(t)

    def airspeed_at(self, t: float) -> float:
        stage = mission_stage(t)
        base = {"TAXI": 0.0, "CLIMB": 78.0, "CRUISE": 95.0,
                "LOITER": 72.0, "EMERGENCY_RECOVERY": 88.0}[stage]
        return float(base + 2.0 * math.sin(t / 5.0))


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
@dataclass
class ScenarioSpec:
    key: str
    title: str
    blurb: str
    seed: int
    shape: MissionShape
    faults: list                      # list[FaultScenario]
    # optional deterministic post-shaping of raw sensor readings
    shaping_fn: Callable[[float, dict, np.random.Generator], dict] | None = None


def _misfire_shaping(t: float, r: dict, rng: np.random.Generator) -> dict:
    """From CRUISE onward, progressively clog injector #3: depress its EGT and
    lean-misfire it, so the cylinder-to-cylinder EGT spread blows past the
    90 C limit and vibration climbs."""
    if t < 60.0:
        return r
    p = min(1.0, (t - 60.0) / 40.0)
    drop = 140.0 * p
    r["egt_3"] = r["egt_3"] - drop
    r["cht_3"] = r["cht_3"] - 18.0 * p          # unburnt charge -> cooler head
    r["egt"] = float(np.mean([r["egt_1"], r["egt_2"], r["egt_3"], r["egt_4"]]))
    r["cht"] = float(np.mean([r["cht_1"], r["cht_2"], r["cht_3"], r["cht_4"]]))
    r["vibration"] = r["vibration"] + 1.9 * p + 0.15 * math.sin(t * 3.0)
    r["rpm"] = r["rpm"] - 60.0 * p
    r["fuel_flow"] = r["fuel_flow"] * (1.0 - 0.06 * p)
    return r


def _cooling_shaping(t: float, r: dict, rng: np.random.Generator) -> dict:
    """Blocked cooling duct + thin air: heat that the fins can no longer shed
    shows up as a rising CHT above the physics baseline, a hotter EGT from
    the leaning mixture, and progressively rougher running."""
    if t < 45.0:
        return r
    p = min(1.0, (t - 45.0) / 80.0)
    cht_rise = 34.0 * p
    for i in range(1, 5):
        r[f"cht_{i}"] = r[f"cht_{i}"] + cht_rise + rng.normal(0, 0.6)
    r["cht"] = float(np.mean([r[f"cht_{i}"] for i in range(1, 5)]))
    r["egt"] = r["egt"] + 22.0 * p              # leaner burn runs hotter EGT
    r["vibration"] = r["vibration"] + 0.9 * p
    return r


_COOLING_ALT_BIAS = lambda t: 0.0 if t < 20.0 else min(2600.0, (t - 20.0) * 42.0)

SCENARIOS: dict[str, ScenarioSpec] = {
    "nominal_loiter": ScenarioSpec(
        key="nominal_loiter",
        title="Nominal Loiter",
        blurb="Stable ISR loiter under ISA conditions. All subsystems in band -- "
              "baseline for the twin.",
        seed=101,
        shape=MissionShape(cruise_altitude_m=4200.0),
        faults=[],
    ),
    "altitude_cooling_degradation": ScenarioSpec(
        key="altitude_cooling_degradation",
        title="High-Altitude Cooling Degradation",
        blurb="Climb overshoots planned ceiling; thinning air + partial cooling-duct "
              "blockage. CHT drifts up, combustion turns rough.",
        seed=202,
        shape=MissionShape(cruise_altitude_m=4600.0, alt_bias_fn=_COOLING_ALT_BIAS),
        faults=[FaultScenario(fault_type=FaultType.COOLING_DEGRADATION,
                              onset_s=45.0, ramp_duration_s=110.0, total_life_s=520.0)],
        shaping_fn=_cooling_shaping,
    ),
    "lubrication_loss": ScenarioSpec(
        key="lubrication_loss",
        title="Lubrication Loss & Bearing Failure",
        blurb="Oil-pump drive failure at loiter. Oil pressure collapses, friction "
              "heat and vibration spike, RUL falls toward mission abort.",
        seed=303,
        shape=MissionShape(cruise_altitude_m=4000.0),
        faults=[FaultScenario(fault_type=FaultType.LUBRICATION_DEGRADATION,
                              onset_s=72.0, ramp_duration_s=70.0, total_life_s=150.0)],
    ),
    "injector_misfire": ScenarioSpec(
        key="injector_misfire",
        title="Injector Clog / Cylinder Misfire",
        blurb="Injector #3 clogs mid-cruise. That cylinder's EGT diverges hard from "
              "the other three -- an immediate, unambiguous anomaly.",
        seed=404,
        shape=MissionShape(cruise_altitude_m=4200.0),
        faults=[FaultScenario(fault_type=FaultType.MISFIRE,
                              onset_s=60.0, ramp_duration_s=40.0)],
        shaping_fn=_misfire_shaping,
    ),
}

M_PER_FT = 0.3048


class ScenarioRunner:
    """Steps one scenario forward at a fixed internal rate and yields decoded
    telemetry dicts. Deterministic: same scenario key -> identical stream."""

    def __init__(self, scenario_key: str | None = None, hz: float = 2.0,
                 spec: ScenarioSpec | None = None):
        if spec is None:
            if scenario_key not in SCENARIOS:
                raise KeyError(f"unknown scenario '{scenario_key}'. "
                               f"Options: {list(SCENARIOS)}")
            spec = SCENARIOS[scenario_key]
        self.spec = spec
        self.hz = hz
        self.dt = 1.0 / hz
        self.rng = np.random.default_rng(self.spec.seed)

        # A MALE UAV is never launched on a cold engine, and the 3-minute
        # window is a slice of a multi-hour ISR sortie -- so the engine starts
        # already at operating temperature. We also tighten the thermal time
        # constants from the batch-dataset defaults (tuned for 10 s sampling
        # over many-hour flights) to values appropriate for a 2 Hz, 3-minute
        # window, so CHT/oil temp track the commanded operating point instead
        # of lagging the whole scenario.
        self.specs = EngineSpecs(tau_cht=35.0, tau_egt=5.0, tau_oil=90.0)
        self.physics = EnginePhysicsModel(self.specs, self.rng)
        self.physics.state = {"cht": 95.0, "egt": 440.0, "oil_temp": 92.0}
        self.injector = FaultInjector(list(self.spec.faults), rng=self.rng)
        self._t = 0.0

    # -- introspection used by the dashboard header ----------------------
    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def blurb(self) -> str:
        return self.spec.blurb

    def reset(self) -> None:
        self.__init__(hz=self.hz, spec=self.spec)

    def step(self) -> dict | None:
        """Advance one tick. Returns None once the 180 s mission is over."""
        t = self._t
        if t > MISSION_DURATION_S:
            return None

        thr = self.spec.shape.throttle_at(t)
        alt_m = self.spec.shape.altitude_at(t)
        amb = self.spec.shape.ambient_at(t)
        ias = self.spec.shape.airspeed_at(t)

        mods = self.injector.get_physical_modifiers(t)
        readings = self.physics.update(self.dt, thr, alt_m, amb, mods)
        readings = self.injector.apply_sensor_faults(t, readings)
        if self.spec.shaping_fn is not None:
            readings = self.spec.shaping_fn(t, readings, self.rng)

        label, min_rul = self.injector.get_labels(t)

        rec = {
            "t_s": round(t, 2),
            "mission_stage": mission_stage(t),
            "progress": round(t / MISSION_DURATION_S, 4),
            "throttle_pct": round(thr, 2),
            "altitude_m": round(alt_m, 1),
            "altitude_ft": round(alt_m / M_PER_FT, 0),
            "airspeed_kts": round(ias, 1),
            "ambient_c": round(amb, 2),
            # --- primary engine telemetry ---
            "rpm": round(readings["rpm"], 1),
            "manifold_pressure": round(readings["map_inHg"], 2),
            "cht": round(readings["cht"], 2),
            "egt": round(readings["egt"], 2),
            "oil_pressure": round(readings["oil_pressure"], 3),
            "oil_temp": round(readings["oil_temp"], 2),
            "fuel_flow": round(readings["fuel_flow"], 3),
            "fuel_flow_lph": round(readings["fuel_flow"], 3),
            "vibration": round(max(0.0, readings["vibration"]), 3),
            "battery_voltage": round(readings["battery_voltage"], 3),
            # --- per-cylinder ---
            **{f"cht_{i}": round(readings[f"cht_{i}"], 2) for i in range(1, 5)},
            **{f"egt_{i}": round(readings[f"egt_{i}"], 2) for i in range(1, 5)},
            # --- ground truth (validation overlay only) ---
            "fault_label": label,
            "remaining_useful_life": round(min_rul, 1) if min_rul is not None else None,
        }
        self._t += self.dt
        return rec

    def stream(self) -> Iterator[dict]:
        while True:
            rec = self.step()
            if rec is None:
                return
            yield rec


def list_scenarios() -> list[dict]:
    return [{"key": s.key, "title": s.title, "blurb": s.blurb} for s in SCENARIOS.values()]


# ---------------------------------------------------------------------------
# Randomized flight factory -- used only to build the ML training set. Same
# physics config as the four demo scenarios, but randomized operating point,
# fault type, onset, severity and seed, so the demo scenarios themselves are
# always out-of-sample.
# ---------------------------------------------------------------------------
def random_flight_spec(rng: np.random.Generator, p_fault: float = 0.62) -> ScenarioSpec:
    seed = int(rng.integers(1, 2_000_000_000))
    cruise_alt = float(rng.uniform(3000, 7200))
    hot = bool(rng.random() < 0.28)
    over_alt = rng.random() < 0.35
    shape = MissionShape(
        cruise_altitude_m=cruise_alt,
        hot_weather=hot,
        alt_bias_fn=(_COOLING_ALT_BIAS if over_alt else (lambda t: 0.0)),
        throttle_bias_fn=(lambda t, b=float(rng.uniform(-4, 4)): b),
    )
    if rng.random() > p_fault:
        return ScenarioSpec(key=f"rand_healthy_{seed}", title="rand", blurb="",
                            seed=seed, shape=shape, faults=[])

    _ftypes = [FaultType.COOLING_DEGRADATION, FaultType.LUBRICATION_DEGRADATION,
               FaultType.MISFIRE, FaultType.SENSOR_DRIFT_CHT]
    ftype = _ftypes[int(rng.integers(len(_ftypes)))]
    onset = float(rng.uniform(25.0, 120.0))
    shaping = None
    if ftype in (FaultType.COOLING_DEGRADATION, FaultType.LUBRICATION_DEGRADATION):
        ramp = float(rng.uniform(45.0, 130.0))
        life = ramp * float(rng.uniform(1.6, 4.0))
        if ftype == FaultType.COOLING_DEGRADATION:
            shaping = _cooling_shaping
    elif ftype == FaultType.MISFIRE:
        ramp = float(rng.uniform(25.0, 55.0)); life = None
        shaping = _misfire_shaping
    else:  # SENSOR_DRIFT_CHT
        ramp = float(rng.uniform(60.0, 150.0)); life = None

    fs = FaultScenario(fault_type=ftype, onset_s=onset, ramp_duration_s=ramp,
                       total_life_s=life)
    return ScenarioSpec(key=f"rand_{ftype.value}_{seed}", title="rand", blurb="",
                        seed=seed, shape=shape, faults=[fs], shaping_fn=shaping)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print one scenario's telemetry stream")
    ap.add_argument("scenario", choices=list(SCENARIOS), nargs="?", default="lubrication_loss")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=10, help="print every Nth sample")
    args = ap.parse_args()

    runner = ScenarioRunner(args.scenario, hz=args.hz)
    print(f"# {runner.title}: {runner.blurb}")
    for i, rec in enumerate(runner.stream()):
        if i % args.every == 0:
            print(json.dumps({k: rec[k] for k in
                              ("t_s", "mission_stage", "rpm", "cht", "egt",
                               "oil_pressure", "vibration", "fault_label",
                               "remaining_useful_life")}))
