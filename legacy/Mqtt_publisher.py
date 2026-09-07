"""
MQTT Publisher for Engine Simulator
====================================
Wraps engine_simulator_v2's streaming mode and publishes each reading to
an MQTT broker topic, so any number of subscribers (Digital Twin Core,
dashboard, logger, etc.) can consume the same live feed independently.

Requires a running MQTT broker. Easiest zero-install option (pure Python,
no admin rights needed on Windows):

    pip install amqtt
    python -m amqtt.scripts.broker_script

...leave that running in its own terminal, then run this publisher in a
second terminal. (If you'd rather use real Mosquitto, that works too --
just point --broker-host/--broker-port at it instead.)

Usage:
    python mqtt_publisher.py --hz 5 --duration-hours 6 --fault lubrication_degradation
"""

import argparse
import json
import random
import time

import paho.mqtt.client as mqtt

from engine_simulator_v2 import (
    EngineSpecs, MissionProfile, FaultInjector, FaultScenario, FaultType,
    EngineSimulator, random_fault_scenarios,
)

DEFAULT_TOPIC = "uav/engine/telemetry"


def build_scenarios(mission: MissionProfile, seed: int, forced_fault: str | None):
    if forced_fault:
        ftype = FaultType(forced_fault)
        kwargs = dict(fault_type=ftype, onset_s=min(900.0, mission.t_climb_end + 60))
        if ftype in (FaultType.COOLING_DEGRADATION, FaultType.LUBRICATION_DEGRADATION):
            kwargs["ramp_duration_s"] = 600.0
            kwargs["total_life_s"] = 2700.0
        elif ftype == FaultType.SENSOR_DRIFT_CHT:
            kwargs["ramp_duration_s"] = 1200.0
        else:
            kwargs["ramp_duration_s"] = 300.0
        return [FaultScenario(**kwargs)]
    rng = random.Random(seed)
    return random_fault_scenarios(mission, rng)


def main():
    parser = argparse.ArgumentParser(description="Publish simulated engine telemetry to MQTT")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--uav-id", type=str, default="UAV-01",
                         help="tag published records with this UAV identifier -- run multiple "
                              "publisher processes with different --uav-id (same topic) to "
                              "simulate a fleet")
    parser.add_argument("--hz", type=float, default=5.0, help="publish rate (sim-time compressed for demo)")
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fault", type=str, default=None,
                         choices=[f.value for f in FaultType if f != FaultType.NONE],
                         help="force a specific fault (default: random, may be none)")
    parser.add_argument("--sim-hz", type=float, default=0.1,
                         help="simulator's internal sample rate (sim-seconds per sample)")
    parser.add_argument("--speed", type=float, default=50.0,
                         help="playback speed multiplier vs real time (higher = faster demo)")
    args = parser.parse_args()

    mission = MissionProfile(duration_hours=args.duration_hours, seed=args.seed)
    scenarios = build_scenarios(mission, args.seed, args.fault)
    injector = FaultInjector(scenarios, rng=mission.rng)
    sim = EngineSimulator(EngineSpecs(), mission, injector)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="engine-publisher")
    print(f"Connecting to broker at {args.broker_host}:{args.broker_port} ...")
    client.connect(args.broker_host, args.broker_port)
    client.loop_start()

    print(f"Publishing to topic '{args.topic}' -- injected faults: "
          f"{[s.fault_type.value for s in scenarios] or 'none'}")
    print(f"Simulated flight duration: {args.duration_hours}h, "
          f"played back at {args.speed}x speed\n")

    sleep_per_sample = (1.0 / args.sim_hz) / args.speed  # real seconds to wait between publishes

    count = 0
    try:
        for record in sim.run(hz=args.sim_hz):
            record["uav_id"] = args.uav_id
            payload = json.dumps(record)
            client.publish(args.topic, payload, qos=0)
            count += 1
            if count % 20 == 0:
                print(f"[{count:6d}] t={record['t_s']:8.1f}s  phase={record['phase']:8s}  "
                      f"fault={record['fault_label']}")
            time.sleep(sleep_per_sample)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"Done. Published {count} messages.")


if __name__ == "__main__":
    main()