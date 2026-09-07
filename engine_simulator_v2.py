"""
Aero Piston Engine Digital Twin - Bulletproof Production Simulator
====================================================
Generates physics-informed synthetic telemetry for a MALE UAV piston engine.
Features exact ODE thermal integration, phugoid flight dynamics, and robust fault states.
"""

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class EngineSpecs:
    """Calibrated against published Rotax 912/914 operator manual data --
    the engine family actually used in real MALE UAVs (IAI Heron uses a
    turbocharged Rotax 914; IAI Searcher uses a similar flat-four Limbach).
    Real MALE altitude band is 3,000-9,000m, which requires turbocharging
    to maintain rated power -- naturally aspirated engines lose too much
    power above ~3,000-4,000m to be usable for MALE profiles."""

    idle_rpm: float = 1400
    max_rpm: float = 5800          # takeoff, 5 min max (real spec)
    cruise_rpm: float = 5000       # below max continuous (5500)
    map_idle: float = 12.0
    map_max: float = 29.9

    cht_idle: float = 90.0
    cht_cruise: float = 125.0      # normal cruise CHT, leaves headroom under redline
    cht_redline: float = 150.0     # real spec: CHT max 150C (300F)

    egt_idle: float = 420.0
    egt_cruise: float = 800.0      # real spec: EGT normal ~800C (1472F)
    egt_redline: float = 900.0     # real spec: EGT max ~900C (1652F)

    oil_pressure_min: float = 0.8     # real spec: min 0.8 bar below 3500 RPM
    oil_pressure_max: float = 5.0     # real spec: normal max 5.0 bar above 3500 RPM
    oil_pressure_coldstart_spike: float = 7.0  # real spec: cold-start transient up to 7 bar

    oil_temp_idle: float = 60.0
    oil_temp_cruise: float = 100.0
    oil_temp_max_safe: float = 130.0

    fuel_flow_max: float = 25.0
    vibration_baseline: float = 1.0
    battery_voltage_nominal: float = 13.8
    injection_timing_nominal: float = 22.0

    turbo_critical_altitude_m: float = 5000.0  # altitude up to which turbo holds rated MAP
    n_cylinders: int = 4  # real Rotax 912/914 is a flat-four -- per-cylinder CHT/EGT modeled

    tau_cht: float = 120.0
    tau_egt: float = 5.0
    tau_oil: float = 240.0


class FlightPhase(str, Enum):
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    LOITER = "loiter"
    DESCENT = "descent"
    LANDING = "landing"


class MissionProfile:
    def __init__(self, duration_hours: float = 6.0, cruise_altitude_m: float = 4500,
                 hot_weather: bool = False, seed: Optional[int] = None):
        self.duration_s = duration_hours * 3600
        self.cruise_altitude_m = cruise_altitude_m
        self.hot_weather = hot_weather
        self.rng = np.random.default_rng(seed)

        self.t_taxi_end = 60                     
        self.t_takeoff_end = self.t_taxi_end + 90  
        self.t_climb_end = self.t_takeoff_end + 900  
        self.t_cruise_end = self.duration_s - 600      
        self.t_descent_end = self.duration_s - 120    

    def phase_at(self, t: float) -> FlightPhase:
        if t < self.t_taxi_end: return FlightPhase.TAXI
        if t < self.t_takeoff_end: return FlightPhase.TAKEOFF
        if t < self.t_climb_end: return FlightPhase.CLIMB
        if t < self.t_cruise_end: return FlightPhase.CRUISE if int(t) % 1800 < 1200 else FlightPhase.LOITER
        if t < self.t_descent_end: return FlightPhase.DESCENT
        return FlightPhase.LANDING

    def throttle_at(self, t: float) -> float:
        phase = self.phase_at(t)
        base = {FlightPhase.TAXI: 15, FlightPhase.TAKEOFF: 100, FlightPhase.CLIMB: 85,
                FlightPhase.CRUISE: 65, FlightPhase.LOITER: 55, FlightPhase.DESCENT: 30,
                FlightPhase.LANDING: 20}[phase]
        return float(np.clip(base + self.rng.normal(0, 1.0), 0, 100))

    def altitude_at(self, t: float) -> float:
        phase = self.phase_at(t)
        if phase == FlightPhase.TAXI: return 0.0
        if phase == FlightPhase.TAKEOFF: return 50 * (t - self.t_taxi_end) / 90
        if phase == FlightPhase.CLIMB:
            return 50 + ((t - self.t_takeoff_end) / 900) * (self.cruise_altitude_m - 50)
        if phase in (FlightPhase.CRUISE, FlightPhase.LOITER):
            # FIXED: Smooth phugoid oscillation instead of violent random teleportation
            return self.cruise_altitude_m + (math.sin(t / 20.0) * 5.0) + self.rng.normal(0, 0.5)
        if phase == FlightPhase.DESCENT:
            return self.cruise_altitude_m * (1 - (t - self.t_cruise_end) / (self.t_descent_end - self.t_cruise_end))
        return max(0.0, 50 * (1 - (t - self.t_descent_end) / max(1, self.duration_s - self.t_descent_end)))

    def ambient_temp_at(self, t: float) -> float:
        sea_level_temp = 40.0 if self.hot_weather else 15.0 
        return sea_level_temp - (6.5 / 1000) * self.altitude_at(t)


class FaultType(str, Enum):
    NONE = "healthy"
    MISFIRE = "misfire"
    COOLING_DEGRADATION = "cooling_degradation"
    LUBRICATION_DEGRADATION = "lubrication_degradation"
    SENSOR_DRIFT_CHT = "sensor_drift_cht"


@dataclass
class FaultScenario:
    fault_type: FaultType
    onset_s: float              
    ramp_duration_s: float = 600  
    total_life_s: Optional[float] = None 


class FaultInjector:
    def __init__(self, scenarios: list[FaultScenario], rng: np.random.Generator):
        self.scenarios = scenarios
        self.rng = rng

    def _get_progress(self, fault: FaultScenario, t: float) -> float:
        if t < fault.onset_s:
            return 0.0
        # FIXED: Zero-division safeguard for instant faults
        safe_ramp = max(1e-9, fault.ramp_duration_s)
        return min(1.0, (t - fault.onset_s) / safe_ramp)

    def get_physical_modifiers(self, t: float) -> dict:
        mods = {
            "cooling_health": 1.0,      
            "lubrication_health": 1.0, 
            "combustion_health": 1.0,
            "is_failed": False
        }
        
        _, min_rul = self.get_labels(t)
        if min_rul is not None and min_rul <= 0.0:
            mods["is_failed"] = True
            return mods

        for fault in self.scenarios:
            p = self._get_progress(fault, t)
            if p == 0: continue

            if fault.fault_type == FaultType.COOLING_DEGRADATION:
                mods["cooling_health"] = 1.0 - (0.7 * p) 

            elif fault.fault_type == FaultType.LUBRICATION_DEGRADATION:
                mods["lubrication_health"] = 1.0 - (0.6 * p) 

            elif fault.fault_type == FaultType.MISFIRE:
                if self.rng.random() < (0.2 * p): 
                    mods["combustion_health"] = self.rng.uniform(0.1, 0.5)

        return mods

    def apply_sensor_faults(self, t: float, readings: dict) -> dict:
        for fault in self.scenarios:
            p = self._get_progress(fault, t)
            if p == 0: continue

            if fault.fault_type == FaultType.SENSOR_DRIFT_CHT:
                readings["cht"] += (40.0 * p)
                
        return readings

    def get_labels(self, t: float) -> tuple[str, Optional[float]]:
        active = []
        min_rul = None

        for fault in self.scenarios:
            if t >= fault.onset_s:
                active.append(fault.fault_type.value)
                if fault.total_life_s is not None:
                    rul = max(0.0, fault.total_life_s - (t - fault.onset_s))
                    min_rul = rul if min_rul is None else min(min_rul, rul)

        label = ",".join(active) if active else FaultType.NONE.value
        return label, min_rul


class EnginePhysicsModel:
    def __init__(self, specs: EngineSpecs, rng: np.random.Generator):
        self.specs = specs
        self.rng = rng
        self.state = None  # initialized on first update() call using real ambient temp
        # Fixed per-cylinder manufacturing variance (set once, persists whole flight) --
        # mirrors real engines where e.g. cylinder 3 always runs a bit hotter than cylinder 1.
        self.cyl_bias_cht = rng.normal(0, 3.0, size=specs.n_cylinders)
        self.cyl_bias_egt = rng.normal(0, 20.0, size=specs.n_cylinders)

    def update(self, dt: float, throttle: float, altitude_m: float, ambient_c: float, mods: dict) -> dict:
        if self.state is None:
            # Cold engine starts at ambient temperature, not a hardcoded placeholder
            self.state = {"cht": ambient_c, "egt": ambient_c, "oil_temp": ambient_c}
        thr = throttle / 100.0
        temp_k = ambient_c + 273.15
        pressure_pa = 101325 * (1 - 2.25577e-5 * altitude_m)**5.25588
        ambient_pressure_inHg = pressure_pa / 3386.39
        density_ratio = pressure_pa / (287.05 * temp_k) / 1.225

        # Turbocharger compensation: real MALE UAV engines (e.g. Rotax 914 in the
        # IAI Heron) are turbocharged specifically to hold rated power up to a
        # "critical altitude" (~5000m here), unlike naturally aspirated engines
        # that lose power immediately as altitude increases. Above the critical
        # altitude, the turbo can no longer fully compensate and power falls off.
        if altitude_m <= self.specs.turbo_critical_altitude_m:
            turbo_map_ceiling = self.specs.map_max
            turbo_density_ratio = 1.0  # boost fully compensates for thin air
        else:
            excess_alt = altitude_m - self.specs.turbo_critical_altitude_m
            falloff = max(0.3, (1 - 2.25577e-5 * excess_alt) ** 5.25588)
            turbo_map_ceiling = self.specs.map_max * falloff
            turbo_density_ratio = falloff

        # --- TERMINAL FAILURE STATE ---
        if mods["is_failed"]:
            rpm_actual = 0.0
            map_actual = ambient_pressure_inHg
            engine_load_factor = 0.0
            fuel_flow = 0.0
            vibration = 0.0
            target_egt = ambient_c
            target_cht = ambient_c
            target_oil_t = ambient_c
            oil_pressure = 0.0
            # FIXED: Added baseline sensor noise so the data line doesn't unnaturally flatline
            battery_voltage = self.specs.battery_voltage_nominal - 1.2 + self.rng.normal(0, 0.05)
        
        # --- NORMAL OPERATION STATE ---
        else:
            map_target = self.specs.map_idle + thr * (turbo_map_ceiling - self.specs.map_idle)
            map_actual = max(0.0, map_target + self.rng.normal(0, 0.2))
            
            # Smooth, continuous throttle -> RPM curve (no formula seams):
            # idle_rpm at thr=0, cruise_rpm at thr=0.5, max_rpm at thr=1.0
            if thr <= 0.5:
                rpm_target = self.specs.idle_rpm + (thr / 0.5) * (self.specs.cruise_rpm - self.specs.idle_rpm)
            else:
                rpm_target = self.specs.cruise_rpm + ((thr - 0.5) / 0.5) * (self.specs.max_rpm - self.specs.cruise_rpm)
            rpm_actual = max(0.0, rpm_target + self.rng.normal(0, 15))

            engine_load_factor = max(0.0, (map_actual / self.specs.map_max) * (rpm_actual / self.specs.max_rpm))
            engine_load_factor *= mods["combustion_health"]

            # Two-segment linear interpolation anchored to real spec values:
            # idle (load=0) -> cruise (load=0.7) -> redline (load=1.0).
            # A single multiplier overshoots the real redline at full throttle,
            # since real engines have only a narrow ~100C gap between normal
            # cruise EGT (800C) and max EGT (900C) despite a much bigger jump
            # in throttle/load between the two.
            CRUISE_LOAD_REF = 0.7
            if engine_load_factor <= CRUISE_LOAD_REF:
                frac = engine_load_factor / CRUISE_LOAD_REF
                target_egt = self.specs.egt_idle + frac * (self.specs.egt_cruise - self.specs.egt_idle)
                target_cht = self.specs.cht_idle + frac * (self.specs.cht_cruise - self.specs.cht_idle)
            else:
                frac = (engine_load_factor - CRUISE_LOAD_REF) / (1.0 - CRUISE_LOAD_REF)
                frac = min(1.2, frac)  # allow slight overshoot above 100% load, but bounded
                target_egt = self.specs.egt_cruise + frac * (self.specs.egt_redline - self.specs.egt_cruise)
                target_cht = self.specs.cht_cruise + frac * (self.specs.cht_redline - self.specs.cht_cruise)
            
            friction_heat = (1.0 - mods["lubrication_health"]) * 40.0
            target_oil_t = self.specs.oil_temp_idle + (self.specs.oil_temp_cruise - self.specs.oil_temp_idle) * engine_load_factor * 1.2 + friction_heat
            
            ambient_offset = (ambient_c - 15.0) 
            ram_air_cooling = thr * 15.0 * mods["cooling_health"]
            
            target_cht = target_cht + ambient_offset - ram_air_cooling
            target_oil_t = target_oil_t + (ambient_offset * 0.5) - (ram_air_cooling * 0.5)

            oil_viscosity_factor = max(0.5, (100 / max(1, self.state["oil_temp"]))**0.5)
            rpm_factor = rpm_actual / self.specs.max_rpm
            oil_pressure = (self.specs.oil_pressure_min + (self.specs.oil_pressure_max - self.specs.oil_pressure_min) * rpm_factor * oil_viscosity_factor)
            # Real spec: cold, thick oil can spike pressure transiently, but only up
            # to a hard ceiling (relief valve opens above this) -- cap it there.
            oil_pressure = min(oil_pressure, self.specs.oil_pressure_coldstart_spike)
            oil_pressure *= mods["lubrication_health"] 

            vib_misfire_spike = (1.0 - mods["combustion_health"]) * 3.0
            vibration = max(0.0, self.specs.vibration_baseline * (0.5 + 0.5 * engine_load_factor) + vib_misfire_spike + self.rng.normal(0, 0.05))
            
            fuel_flow = max(0.0, self.specs.fuel_flow_max * engine_load_factor * turbo_density_ratio)
            battery_voltage = self.specs.battery_voltage_nominal - (0.6 if rpm_actual < 2000 else 0) + self.rng.normal(0, 0.05)

        # FIXED: Exact analytical ODE solution (unconditionally stable at any dt)
        self.state["egt"] = target_egt + (self.state["egt"] - target_egt) * math.exp(-dt / self.specs.tau_egt)
        self.state["cht"] = target_cht + (self.state["cht"] - target_cht) * math.exp(-dt / self.specs.tau_cht)
        self.state["oil_temp"] = target_oil_t + (self.state["oil_temp"] - target_oil_t) * math.exp(-dt / self.specs.tau_oil)

        # Per-cylinder CHT/EGT: real 4-cylinder engines report each cylinder
        # separately (this is how real engine monitors catch a single bad
        # cylinder). Each cylinder = shared thermal state + fixed manufacturing
        # bias + independent sensor noise.
        cyl_cht = [float(self.state["cht"] + self.cyl_bias_cht[i] + self.rng.normal(0, 0.5))
                   for i in range(self.specs.n_cylinders)]
        cyl_egt = [float(self.state["egt"] + self.cyl_bias_egt[i] + self.rng.normal(0, 2.0))
                   for i in range(self.specs.n_cylinders)]

        result = {
            "rpm": float(rpm_actual),
            "map_inHg": float(map_actual),
            "cht": float(np.mean(cyl_cht)),   # fleet/summary value (backward compatible)
            "egt": float(np.mean(cyl_egt)),   # fleet/summary value (backward compatible)
            "oil_pressure": float(oil_pressure + (self.rng.normal(0, 0.05) if not mods["is_failed"] else 0)),
            "oil_temp": float(self.state["oil_temp"] + self.rng.normal(0, 0.2)),
            "fuel_flow": float(fuel_flow + (self.rng.normal(0, 0.1) if not mods["is_failed"] else 0)),
            "vibration": float(vibration),
            "battery_voltage": float(battery_voltage),
        }
        for i in range(self.specs.n_cylinders):
            result[f"cht_{i+1}"] = round(cyl_cht[i], 3)
            result[f"egt_{i+1}"] = round(cyl_egt[i], 3)
        return result


class EngineSimulator:
    def __init__(self, specs: EngineSpecs, mission: MissionProfile, injector: FaultInjector):
        self.specs = specs
        self.mission = mission
        self.injector = injector
        self.rng = self.mission.rng 
        self.physics = EnginePhysicsModel(specs, self.rng)

    def step(self, t: float, dt: float) -> dict:
        throttle = self.mission.throttle_at(t)
        altitude = self.mission.altitude_at(t)
        ambient = self.mission.ambient_temp_at(t)
        
        physical_mods = self.injector.get_physical_modifiers(t)
        readings = self.physics.update(dt, throttle, altitude, ambient, physical_mods)
        readings = self.injector.apply_sensor_faults(t, readings)
        label, min_rul = self.injector.get_labels(t)

        record = {
            "t_s": round(t, 1),
            "phase": self.mission.phase_at(t).value,
            "throttle_pct": round(throttle, 1),
            "altitude_m": round(altitude, 1),
            "ambient_c": round(ambient, 1),
            **{k: round(v, 3) for k, v in readings.items()},
            "fault_label": label,
            "remaining_useful_life": round(min_rul, 1) if min_rul is not None else None
        }
        return record

    def run(self, hz: float = 1.0):
        dt = 1.0 / hz
        step = 0
        
        while step * dt <= self.mission.duration_s:
            t = step * dt
            yield self.step(t, dt)
            step += 1


# ---------------------------------------------------------------------------
# Random fault scenario generator (for building a varied, labeled dataset)
# ---------------------------------------------------------------------------

def random_fault_scenarios(mission: MissionProfile, rng: "random.Random",
                            p_fault_flight: float = 0.6) -> list:
    """With probability p_fault_flight, injects a fault into an otherwise
    healthy flight, at a random onset after the climb phase completes."""
    if rng.random() > p_fault_flight:
        return []

    candidate_types = [f for f in FaultType if f != FaultType.NONE]
    fault_type = rng.choice(candidate_types)
    onset = rng.uniform(mission.t_climb_end, mission.t_cruise_end * 0.8)

    kwargs = dict(fault_type=fault_type, onset_s=onset)

    if fault_type in (FaultType.COOLING_DEGRADATION, FaultType.LUBRICATION_DEGRADATION):
        kwargs["ramp_duration_s"] = rng.uniform(300, 900)
        # total_life_s correlated with ramp speed: a fault progressing quickly
        # (short ramp) realistically means less time remains until failure,
        # not an unrelated random number. Preserves flight-to-flight variety
        # via the random multiplier, while making RUL learnable from the
        # observable rate of degradation -- matching real engine behavior.
        kwargs["total_life_s"] = kwargs["ramp_duration_s"] * rng.uniform(2.5, 6.0)
    elif fault_type == FaultType.SENSOR_DRIFT_CHT:
        kwargs["ramp_duration_s"] = rng.uniform(600, 1800)
    else:  # MISFIRE
        kwargs["ramp_duration_s"] = rng.uniform(120, 400)

    return [FaultScenario(**kwargs)]


# ---------------------------------------------------------------------------
# Batch dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_flights: int, hz: float = 0.1, seed: int = 42,
                      min_duration_h: float = 8.0, max_duration_h: float = 30.0):
    """Yields one dict per row across n_flights simulated flights, each with
    a flight_id column added. Returns a list of dicts (convert to DataFrame
    or write straight to CSV).

    Duration/altitude ranges reflect real MALE UAV operational envelopes:
    MALE is defined as 3,000-9,000m altitude, missions typically 18-52 hours
    (e.g. IAI Heron: up to 52h at 10,500m; IAI Searcher: up to 18h, 1,500-6,400m).
    Default range here (8-30h, 3,000-9,000m) covers a representative band
    without generating unmanageable file sizes."""
    import random as _random
    rng = _random.Random(seed)
    all_records = []

    for flight_id in range(n_flights):
        duration_h = rng.uniform(min_duration_h, max_duration_h)
        hot_weather = rng.random() < 0.25
        cruise_alt = rng.uniform(3000, 9000)

        mission = MissionProfile(duration_hours=duration_h, cruise_altitude_m=cruise_alt,
                                  hot_weather=hot_weather, seed=seed + flight_id)
        scenarios = random_fault_scenarios(mission, rng)
        injector = FaultInjector(scenarios, rng=mission.rng)
        sim = EngineSimulator(EngineSpecs(), mission, injector)

        for record in sim.run(hz=hz):
            record["flight_id"] = flight_id
            all_records.append(record)

    return all_records


def write_csv(records: list, path: str):
    if not records:
        raise ValueError("No records to write")
    fieldnames = ["flight_id"] + [k for k in records[0].keys() if k != "flight_id"]
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_demo():
    """Original single-flight console demo -- unchanged behavior."""
    seed = 42
    mission = MissionProfile(duration_hours=1.0, seed=seed)

    faults = [
        FaultScenario(
            fault_type=FaultType.LUBRICATION_DEGRADATION,
            onset_s=900.0,
            ramp_duration_s=600.0,
            total_life_s=2700.0
        )
    ]

    injector = FaultInjector(faults, rng=mission.rng)
    sim = EngineSimulator(EngineSpecs(), mission, injector)

    print("Simulating 1-hour flight. Lubrication failure starts at 15m. Engine death at 60m.")

    hz = 1.0
    step_count = 0
    for record in sim.run(hz=hz):
        if step_count % int(300 * hz) == 0:
            print(json.dumps(record, indent=2))
        step_count += 1


def run_stream(hz: float, duration_hours: float, seed: int, fault: Optional[str]):
    import random as _random
    import time as _time

    mission = MissionProfile(duration_hours=duration_hours, seed=seed)
    rng = _random.Random(seed)

    if fault:
        ftype = FaultType(fault)
        kwargs = dict(fault_type=ftype, onset_s=min(900.0, mission.t_climb_end + 60))
        if ftype in (FaultType.COOLING_DEGRADATION, FaultType.LUBRICATION_DEGRADATION):
            kwargs["ramp_duration_s"] = 600.0
            kwargs["total_life_s"] = 2700.0
        elif ftype == FaultType.SENSOR_DRIFT_CHT:
            kwargs["ramp_duration_s"] = 1200.0
        else:
            kwargs["ramp_duration_s"] = 300.0
        scenarios = [FaultScenario(**kwargs)]
    else:
        scenarios = random_fault_scenarios(mission, rng)

    injector = FaultInjector(scenarios, rng=mission.rng)
    sim = EngineSimulator(EngineSpecs(), mission, injector)

    print(f"Streaming at {hz} Hz -- injected faults: "
          f"{[s.fault_type.value for s in scenarios] or 'none'}", flush=True)
    for record in sim.run(hz=hz):
        print(json.dumps(record), flush=True)
        _time.sleep(1.0 / hz)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MALE UAV piston engine digital twin simulator (v2)")
    parser.add_argument("--mode", choices=["demo", "batch", "stream"], default="demo",
                         help="demo: original single-flight console printout. "
                              "batch: generate a labeled CSV dataset. "
                              "stream: live JSON output, one line per timestep.")
    parser.add_argument("--flights", type=int, default=60, help="number of flights (batch mode)")
    parser.add_argument("--hz", type=float, default=0.1, help="sample rate in Hz")
    parser.add_argument("--output", type=str, default="engine_dataset_v2.csv", help="CSV path (batch mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-hours", type=float, default=6.0, help="flight duration (stream mode)")
    parser.add_argument("--fault", type=str, default=None,
                         choices=[f.value for f in FaultType if f != FaultType.NONE],
                         help="force a specific fault (stream mode; default: random)")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()

    elif args.mode == "batch":
        print(f"Generating {args.flights} simulated flights at {args.hz} Hz ...")
        records = generate_dataset(args.flights, hz=args.hz, seed=args.seed)
        write_csv(records, args.output)
        fault_flights = len({r["flight_id"] for r in records if r["fault_label"] != FaultType.NONE.value})
        print(f"Wrote {len(records):,} records ({args.flights} flights, "
              f"{fault_flights} with an injected fault) -> {args.output}")

    else:  # stream
        run_stream(args.hz, args.duration_hours, args.seed, args.fault)

