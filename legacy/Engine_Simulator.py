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
    idle_rpm: float = 1400
    max_rpm: float = 5800
    cruise_rpm: float = 5000
    map_idle: float = 12.0
    map_max: float = 29.9
    cht_idle: float = 90.0
    cht_cruise: float = 135.0
    egt_idle: float = 350.0
    egt_cruise: float = 720.0
    oil_pressure_max: float = 4.5
    oil_pressure_min: float = 2.0
    oil_temp_idle: float = 60.0
    oil_temp_cruise: float = 100.0
    fuel_flow_max: float = 25.0
    vibration_baseline: float = 1.0
    battery_voltage_nominal: float = 13.8 
    injection_timing_nominal: float = 22.0

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

    def update(self, dt: float, throttle: float, altitude_m: float, ambient_c: float, mods: dict) -> dict:
        if self.state is None:
            # Cold engine starts at ambient temperature, not a hardcoded placeholder
            self.state = {"cht": ambient_c, "egt": ambient_c, "oil_temp": ambient_c}
        thr = throttle / 100.0
        temp_k = ambient_c + 273.15
        pressure_pa = 101325 * (1 - 2.25577e-5 * altitude_m)**5.25588
        ambient_pressure_inHg = pressure_pa / 3386.39
        density_ratio = pressure_pa / (287.05 * temp_k) / 1.225 

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
            max_available_map = min(self.specs.map_max, ambient_pressure_inHg)
            map_target = self.specs.map_idle + thr * (max_available_map - self.specs.map_idle) * density_ratio
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
            
            target_egt = self.specs.egt_idle + (self.specs.egt_cruise - self.specs.egt_idle) * engine_load_factor * 1.5
            target_cht = self.specs.cht_idle + (self.specs.cht_cruise - self.specs.cht_idle) * engine_load_factor * 1.3
            
            friction_heat = (1.0 - mods["lubrication_health"]) * 40.0
            target_oil_t = self.specs.oil_temp_idle + (self.specs.oil_temp_cruise - self.specs.oil_temp_idle) * engine_load_factor * 1.2 + friction_heat
            
            ambient_offset = (ambient_c - 15.0) 
            ram_air_cooling = thr * 15.0 * mods["cooling_health"]
            
            target_cht = target_cht + ambient_offset - ram_air_cooling
            target_oil_t = target_oil_t + (ambient_offset * 0.5) - (ram_air_cooling * 0.5)

            oil_viscosity_factor = max(0.5, (100 / max(1, self.state["oil_temp"]))**0.5)
            rpm_factor = rpm_actual / self.specs.max_rpm
            oil_pressure = (self.specs.oil_pressure_min + (self.specs.oil_pressure_max - self.specs.oil_pressure_min) * rpm_factor * oil_viscosity_factor)
            oil_pressure *= mods["lubrication_health"] 

            vib_misfire_spike = (1.0 - mods["combustion_health"]) * 3.0
            vibration = max(0.0, self.specs.vibration_baseline * (0.5 + 0.5 * engine_load_factor) + vib_misfire_spike + self.rng.normal(0, 0.05))
            
            fuel_flow = max(0.0, self.specs.fuel_flow_max * engine_load_factor * density_ratio)
            battery_voltage = self.specs.battery_voltage_nominal - (0.6 if rpm_actual < 2000 else 0) + self.rng.normal(0, 0.05)

        # FIXED: Exact analytical ODE solution (unconditionally stable at any dt)
        self.state["egt"] = target_egt + (self.state["egt"] - target_egt) * math.exp(-dt / self.specs.tau_egt)
        self.state["cht"] = target_cht + (self.state["cht"] - target_cht) * math.exp(-dt / self.specs.tau_cht)
        self.state["oil_temp"] = target_oil_t + (self.state["oil_temp"] - target_oil_t) * math.exp(-dt / self.specs.tau_oil)

        return {
            "rpm": float(rpm_actual),
            "map_inHg": float(map_actual),
            "cht": float(self.state["cht"] + self.rng.normal(0, 0.5)),
            "egt": float(self.state["egt"] + self.rng.normal(0, 2.0)),
            "oil_pressure": float(oil_pressure + (self.rng.normal(0, 0.05) if not mods["is_failed"] else 0)),
            "oil_temp": float(self.state["oil_temp"] + self.rng.normal(0, 0.2)),
            "fuel_flow": float(fuel_flow + (self.rng.normal(0, 0.1) if not mods["is_failed"] else 0)),
            "vibration": float(vibration),
            "battery_voltage": float(battery_voltage),
        }


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
            "altitude_m": round(altitude, 1),
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


if __name__ == "__main__":
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
        