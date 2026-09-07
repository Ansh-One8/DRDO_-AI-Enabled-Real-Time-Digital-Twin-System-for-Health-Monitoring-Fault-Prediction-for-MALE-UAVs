"""
core/twin_pipeline.py
=====================
End-to-end **digital-twin pipeline**, wired once and stepped frame by frame:

    ScenarioRunner  ->  CanTelemetryBus (pack)  ->  (decode)
                    ->  PhysicsModel residuals
                    ->  TwinModels  (anomaly / fault / health / physics-RUL)
                    ->  LiveExplainer (TreeSHAP, only when an anomaly is live)
                    ->  advisory + enriched frame

One `TwinPipeline` instance backs the dashboard; `run()` also works headless
for the CLI fallback in `run_demo.sh`.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.can_telemetry import CanTelemetryBus
from core.scenario_runner import ScenarioRunner, SCENARIOS, MISSION_DURATION_S
from ml.models import TwinModels
from ml.explainability import LiveExplainer, make_background


# ---------------------------------------------------------------------------
# Operator advisory rules -- (fault, urgency) -> concrete crew action
# ---------------------------------------------------------------------------
def build_advisory(fault: str, health: float, rul_s: float | None,
                   anomaly: bool, egt_spread: float) -> dict:
    if not anomaly and fault == "healthy":
        return {"urgency": "nominal",
                "text": "All subsystems within physics tolerance. Continue mission."}

    imminent = rul_s is not None and rul_s < 90
    near = rul_s is not None and rul_s < 300

    table = {
        "lubrication_degradation": (
            "Oil system integrity degrading. Throttle back to 55%, minimise load. "
            "RUL below abort threshold -- execute RTL / nearest-recovery NOW.",
            "Oil pressure trending down. Reduce power to 60%, prepare divert, "
            "watch for bearing rumble on vibration channel."),
        "cooling_degradation": (
            "Cylinder-head cooling lost. Reduce throttle to 60%, increase airspeed "
            "for ram cooling, descend to denser air. Abort if CHT keeps climbing.",
            "CHT drifting above physics baseline. Enrich mixture, lower cruise power "
            "to 62%, monitor trend."),
        "misfire": (
            "Single-cylinder misfire confirmed (EGT spread {spread:.0f} C). Expect "
            "power loss and rising vibration -- plan precautionary landing.",
            "Combustion roughness on one cylinder. Log event, monitor EGT spread and "
            "vibration; no immediate action if power stable."),
        "sensor_drift_cht": (
            "CHT sensor drift suspected -- cross-check against secondary probe before "
            "trusting the thermal channel.",
            "Possible CHT sensor bias. Flag for maintenance; corroborate with EGT and "
            "oil-temp trends."),
    }
    hi, lo = table.get(fault, ("Unclassified anomaly -- hand to engineering review.",
                               "Unclassified deviation -- monitor."))
    txt = (hi if (imminent or near or health < 55) else lo).format(spread=egt_spread)
    if imminent:
        urg = "critical"
    elif near or health < 55:
        urg = "warning"
    else:
        urg = "caution"
    return {"urgency": urg, "text": txt}


class TwinPipeline:
    def __init__(self, scenario_key: str = "nominal_loiter", hz: float = 2.0,
                 buffer_len: int = 600, verbose: bool = True):
        self.hz = hz
        self.verbose = verbose
        self.models = TwinModels(verbose=verbose, hz=hz)
        bg = make_background(self.models.feature_cols, n=50)
        self.explainer = LiveExplainer(
            self.models.classifier_model, self.models.fault_labels,
            self.models.feature_cols, background=bg, verbose=verbose)
        self.buffer: deque = deque(maxlen=buffer_len)
        self.frame_idx = 0
        self.last_explanation: dict | None = None
        self._bus: CanTelemetryBus | None = None
        self._runner: ScenarioRunner | None = None
        self.scenario_key = scenario_key
        self.reset(scenario_key)

    # -- lifecycle ------------------------------------------------------
    def reset(self, scenario_key: str | None = None) -> None:
        if scenario_key:
            self.scenario_key = scenario_key
        if self._bus is not None:
            self._bus.shutdown()
        self._bus = CanTelemetryBus(channel=f"twin_{self.scenario_key}_{int(time.time()*1e3)}",
                                    verbose=self.verbose)
        self._runner = ScenarioRunner(self.scenario_key, hz=self.hz)
        self.models.reset()
        self.buffer.clear()
        self.frame_idx = 0
        self.last_explanation = None

    @property
    def scenario_title(self) -> str:
        return SCENARIOS[self.scenario_key].title

    @property
    def scenario_blurb(self) -> str:
        return SCENARIOS[self.scenario_key].blurb

    @property
    def done(self) -> bool:
        return self._runner is not None and self._runner._t > MISSION_DURATION_S

    # -- one frame ---------------------------------------------------
    def step(self) -> dict | None:
        rec = self._runner.step()
        if rec is None:
            return None

        n_frames = self._bus.publish(rec)
        decoded = self._bus.poll()
        if decoded is None:
            decoded = dict(rec)
        # carry fields not on the wire (stage label + ground truth overlay)
        decoded["t_s"] = rec["t_s"]
        decoded["mission_stage"] = rec["mission_stage"]
        decoded["progress"] = rec["progress"]
        decoded["fault_label"] = rec["fault_label"]
        decoded["remaining_useful_life"] = rec["remaining_useful_life"]

        out = self.models.infer(decoded)

        explanation = self.last_explanation
        if out["anomaly"]:
            explanation = self.explainer.get_live_explanation(
                out["feature_row"], out["fault"])
            self.last_explanation = explanation

        adv = build_advisory(out["fault"], out["health_index"], out["rul_seconds"],
                             out["anomaly"], out["egt_cyl_spread"])

        frame = {
            "frame": self.frame_idx,
            "t_s": rec["t_s"],
            "mission_stage": rec["mission_stage"],
            "progress": rec["progress"],
            # commanded / raw telemetry (decoded from CAN)
            "throttle_pct": decoded["throttle_pct"],
            "altitude_ft": decoded.get("altitude_ft", rec["altitude_ft"]),
            "altitude_m": decoded.get("altitude_m", rec["altitude_m"]),
            "airspeed_kts": decoded.get("airspeed_kts", rec["airspeed_kts"]),
            "rpm": decoded["rpm"],
            "cht": decoded["cht"],
            "egt": decoded["egt"],
            "oil_pressure": decoded["oil_pressure"],
            "oil_temp": decoded["oil_temp"],
            "fuel_flow_lph": decoded.get("fuel_flow_lph", rec["fuel_flow_lph"]),
            "manifold_pressure": decoded.get("manifold_pressure", rec["manifold_pressure"]),
            "vibration": decoded["vibration"],
            "battery_voltage": decoded["battery_voltage"],
            **{f"cht_{i}": decoded.get(f"cht_{i}", rec[f"cht_{i}"]) for i in range(1, 5)},
            **{f"egt_{i}": decoded.get(f"egt_{i}", rec[f"egt_{i}"]) for i in range(1, 5)},
            # twin outputs
            "residuals": out["residuals"],
            "norm_residuals": out["norm_residuals"],
            "subsystem_health": out["subsystem_health"],
            "health_index": out["health_index"],
            "anomaly": out["anomaly"],
            "anomaly_raw": out["anomaly_raw"],
            "fault": out["fault"],
            "fault_confidence": out["fault_confidence"],
            "egt_cyl_spread": out["egt_cyl_spread"],
            "rul_seconds": out["rul_seconds"],
            "rul_driver": out["rul_driver"],
            "ml_rul_seconds": out["ml_rul_seconds"],
            "ml_rul_reliable": out["ml_rul_reliable"],
            "infer_latency_ms": out["infer_latency_ms"],
            "explanation": explanation,
            "advisory": adv,
            # bus + ground truth
            "can_frames": n_frames,
            "can_fps": self._bus.fps,
            "can_status": self._bus.status(),
            "gt_fault": rec["fault_label"],
            "gt_rul": rec["remaining_useful_life"],
        }
        self.buffer.append(frame)
        self.frame_idx += 1
        return frame

    def shutdown(self) -> None:
        if self._bus is not None:
            self._bus.shutdown()

    # -- headless run --------------------------------------------------
    def run(self, realtime: bool = True, speed: float = 4.0):
        dt = (1.0 / self.hz) / max(speed, 1e-6)
        while True:
            f = self.step()
            if f is None:
                return
            flag = "  <<< ANOMALY" if f["anomaly"] else ""
            rul = f["rul_seconds"]
            rul_s = f"{rul/60:4.1f}min" if rul is not None else "  --  "
            print(f"t={f['t_s']:6.1f}s {f['mission_stage']:<18} "
                  f"H={f['health_index']:5.1f} rpm={f['rpm']:6.0f} "
                  f"CHT={f['cht']:5.1f} oilP={f['oil_pressure']:4.2f} "
                  f"fault={f['fault']:<24} RUL={rul_s} "
                  f"CANfps={f['can_fps']:5.1f}{flag}")
            if realtime:
                time.sleep(dt)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Headless digital-twin pipeline")
    ap.add_argument("scenario", nargs="?", default="lubrication_loss",
                    choices=list(SCENARIOS))
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--no-realtime", action="store_true")
    args = ap.parse_args()

    pipe = TwinPipeline(args.scenario)
    print(f"\n# {pipe.scenario_title}: {pipe.scenario_blurb}\n")
    try:
        pipe.run(realtime=not args.no_realtime, speed=args.speed)
    finally:
        pipe.shutdown()
