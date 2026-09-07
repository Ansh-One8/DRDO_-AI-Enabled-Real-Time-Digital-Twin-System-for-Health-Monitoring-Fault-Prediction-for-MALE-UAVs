"""
core/can_telemetry.py
=====================
Virtual **CAN / SocketCAN telemetry layer** -- the demo's stand-in for the
ECU/FADEC bus a real MALE UAV engine sits on.

Uses `python-can`'s in-process `virtual` backend. The scenario runner's
decoded telemetry is *packed into raw CAN frames* on a fixed message set
(arbitration IDs 0x100-0x104, 8-byte payloads, big-endian scaled uint16
signals) and the digital-twin side *reads and decodes those frames back to
engineering units*, exactly as it would off a real bus. This makes the
"ingest ECU/FADEC telemetry over CAN" claim concrete and inspectable
(`--dump` prints the hex frames).

Frame map (5 x 8 bytes = 20 signals):

  0x100  ENGINE_CORE   rpm | manifold_pressure_inHg | oil_pressure_bar | oil_temp_c
  0x101  CHT_BANK      cht_1 | cht_2 | cht_3 | cht_4
  0x102  EGT_BANK      egt_1 | egt_2 | egt_3 | egt_4
  0x103  FLOW_DYNAMICS fuel_flow_lph | vibration_rms_g | altitude_ft | airspeed_kts
  0x104  AUX_STATE     battery_voltage_v | throttle_pct | ambient_c | mission_t_s

If `python-can` is not importable, a byte-compatible in-process loopback is
used instead so the pipeline still runs (mode is reported at construction).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

try:
    import can  # python-can
    _HAVE_CAN = True
except Exception:  # pragma: no cover
    can = None
    _HAVE_CAN = False


# ---------------------------------------------------------------------------
# Signal definitions: name -> (scale, offset). raw_uint16 = (value/scale) - offset
# decode: value = (raw_uint16 + offset) * scale
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sig:
    """A scaled unsigned-16 CAN signal.

        raw_u16 = round(value / scale) + offset      # wire form, 0..65535
        value   = (raw_u16 - offset) * scale         # engineering units

    ``offset`` is a raw-count bias that lets a signed engineering quantity
    (e.g. -50 C ambient) map into the unsigned wire range.
    """
    name: str
    scale: float
    offset: int = 0

    def encode(self, value: float) -> int:
        raw = int(round(value / self.scale)) + self.offset
        return max(0, min(0xFFFF, raw))

    def decode(self, raw: int) -> float:
        return (raw - self.offset) * self.scale


FRAME_MAP: dict[int, tuple[str, list[Sig]]] = {
    0x100: ("ENGINE_CORE", [
        Sig("rpm", 0.2),                 # 0 .. 13107 rpm
        Sig("manifold_pressure", 0.001),  # 0 .. 65.5 inHg
        Sig("oil_pressure", 0.0002),      # 0 .. 13.1 bar
        Sig("oil_temp", 0.005, offset=10000),   # -50 .. 277 C
    ]),
    0x101: ("CHT_BANK", [
        Sig("cht_1", 0.005, offset=10000),
        Sig("cht_2", 0.005, offset=10000),
        Sig("cht_3", 0.005, offset=10000),
        Sig("cht_4", 0.005, offset=10000),
    ]),
    0x102: ("EGT_BANK", [
        Sig("egt_1", 0.02, offset=2500),   # -50 .. 1260 C
        Sig("egt_2", 0.02, offset=2500),
        Sig("egt_3", 0.02, offset=2500),
        Sig("egt_4", 0.02, offset=2500),
    ]),
    0x103: ("FLOW_DYNAMICS", [
        Sig("fuel_flow_lph", 0.001),      # 0 .. 65.5 L/h
        Sig("vibration", 0.0002),         # 0 .. 13.1 g
        Sig("altitude_ft", 1.0),          # 0 .. 65535 ft
        Sig("airspeed_kts", 0.01),        # 0 .. 655 kts
    ]),
    0x104: ("AUX_STATE", [
        Sig("battery_voltage", 0.001),    # 0 .. 65.5 V
        Sig("throttle_pct", 0.01),        # 0 .. 655 %
        Sig("ambient_c", 0.005, offset=10000),  # -50 .. 277 C
        Sig("t_s", 0.05),                 # 0 .. 3276 s mission time
    ]),
}

BASE_ID = min(FRAME_MAP)
TOP_ID = max(FRAME_MAP)
DERIVED_ON_DECODE = ("cht", "egt")  # mean of the per-cylinder banks


def _pack_frame(sigs: list[Sig], sample: dict) -> bytes:
    raws = [s.encode(float(sample.get(s.name, 0.0))) for s in sigs]
    return struct.pack(">4H", *raws)


def _unpack_frame(sigs: list[Sig], data: bytes) -> dict:
    raws = struct.unpack(">4H", data.ljust(8, b"\x00")[:8])
    return {s.name: s.decode(r) for s, r in zip(sigs, raws)}


# ---------------------------------------------------------------------------
# Loopback fallback (no python-can): same public surface, same byte framing
# ---------------------------------------------------------------------------
class _LoopMsg:
    __slots__ = ("arbitration_id", "data", "timestamp", "dlc")

    def __init__(self, arbitration_id: int, data: bytes):
        self.arbitration_id = arbitration_id
        self.data = data
        self.timestamp = time.time()
        self.dlc = len(data)


class _LoopbackBus:
    def __init__(self):
        from collections import deque
        self._q = deque(maxlen=512)

    def send(self, msg):
        self._q.append(msg)

    def recv(self, timeout=0.0):
        return self._q.popleft() if self._q else None

    def shutdown(self):
        self._q.clear()


# ---------------------------------------------------------------------------
# Public bus wrapper
# ---------------------------------------------------------------------------
class CanTelemetryBus:
    """Broadcast side and receive side share one virtual channel
    (`receive_own_messages=True`), which is all a single-process demo needs.
    For a real deployment, point this at a SocketCAN channel instead
    (`interface='socketcan', channel='can0'`)."""

    def __init__(self, channel: str = "uav_engine", *, interface: str = "virtual",
                 verbose: bool = True):
        self.channel = channel
        self.frames_sent = 0
        self.frames_recv = 0
        self._last_publish_ts = None
        self._fps_ema = 0.0

        if _HAVE_CAN and interface == "virtual":
            self._bus = can.interface.Bus(
                channel=channel, interface="virtual", receive_own_messages=True)
            self.mode = f"python-can {can.__version__} [{interface}:{channel}]"
        elif _HAVE_CAN:
            self._bus = can.interface.Bus(channel=channel, interface=interface,
                                          receive_own_messages=True)
            self.mode = f"python-can {can.__version__} [{interface}:{channel}]"
        else:
            self._bus = _LoopbackBus()
            self.mode = "in-process loopback (python-can not installed)"
        if verbose:
            print(f"  [can] {self.mode}")

    # -- broadcast ------------------------------------------------------
    def publish(self, sample: dict) -> int:
        """Pack one telemetry sample into the 5-frame message set and send.
        Returns the number of frames written."""
        n = 0
        for arb_id, (_name, sigs) in FRAME_MAP.items():
            data = _pack_frame(sigs, sample)
            msg = (can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
                   if _HAVE_CAN else _LoopMsg(arb_id, data))
            self._bus.send(msg)
            n += 1
        self.frames_sent += n

        now = time.perf_counter()
        if self._last_publish_ts is not None:
            dt = now - self._last_publish_ts
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0 else 0.8 * self._fps_ema + 0.2 * inst
        self._last_publish_ts = now
        return n

    # -- ingest -------------------------------------------------------
    def poll(self, timeout: float = 0.0, max_frames: int = 16) -> dict | None:
        """Drain currently-queued frames and reassemble one telemetry dict.
        Returns None if no complete set arrived. Adds `_can_*` metadata."""
        got: dict[int, dict] = {}
        raw_hex: dict[int, str] = {}
        deadline = time.perf_counter() + timeout
        reads = 0
        while reads < max_frames:
            msg = self._bus.recv(timeout=0.0)
            if msg is None:
                if timeout and time.perf_counter() < deadline:
                    time.sleep(0.001)
                    continue
                break
            reads += 1
            arb = msg.arbitration_id
            if arb in FRAME_MAP:
                _name, sigs = FRAME_MAP[arb]
                got[arb] = _unpack_frame(sigs, bytes(msg.data))
                raw_hex[arb] = bytes(msg.data).hex()
                self.frames_recv += 1

        if not got:
            return None

        sample: dict = {}
        for d in got.values():
            sample.update(d)
        for base in DERIVED_ON_DECODE:
            parts = [sample[f"{base}_{i}"] for i in range(1, 5) if f"{base}_{i}" in sample]
            if parts:
                sample[base] = sum(parts) / len(parts)
        sample["altitude_m"] = sample.get("altitude_ft", 0.0) * 0.3048
        sample["fuel_flow"] = sample.get("fuel_flow_lph", 0.0)
        sample["_can_frames"] = sorted(got)
        sample["_can_frame_count"] = len(got)
        sample["_can_hex"] = raw_hex
        sample["_can_rx_ts"] = time.time()
        return sample

    @property
    def fps(self) -> float:
        return round(self._fps_ema, 1)

    def status(self) -> dict:
        return {
            "mode": self.mode, "channel": self.channel,
            "frames_sent": self.frames_sent, "frames_recv": self.frames_recv,
            "publish_fps": self.fps,
            "id_range": f"0x{BASE_ID:03X}-0x{TOP_ID:03X}",
        }

    def shutdown(self) -> None:
        try:
            self._bus.shutdown()
        except Exception:
            pass


def decode_frame(arbitration_id: int, data: bytes) -> dict:
    """Standalone helper: decode a single raw frame to engineering units."""
    if arbitration_id not in FRAME_MAP:
        raise KeyError(f"unknown arbitration id 0x{arbitration_id:X}")
    _name, sigs = FRAME_MAP[arbitration_id]
    return _unpack_frame(sigs, data)


if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.scenario_runner import ScenarioRunner

    ap = argparse.ArgumentParser(description="Stream a scenario over the virtual CAN bus")
    ap.add_argument("scenario", nargs="?", default="lubrication_loss")
    ap.add_argument("--dump", action="store_true", help="print raw hex frames")
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()

    bus = CanTelemetryBus()
    runner = ScenarioRunner(args.scenario, hz=2.0)
    print(f"# {runner.title} -> CAN {bus.status()['id_range']}")
    for i, rec in enumerate(runner.stream()):
        bus.publish(rec)
        decoded = bus.poll()
        if decoded is None:
            continue
        if args.dump and i % args.every == 0:
            for arb in decoded["_can_frames"]:
                print(f"  0x{arb:03X}  {decoded['_can_hex'][arb]}")
        if i % args.every == 0:
            print(f"t={decoded['t_s']:6.1f}s  rpm={decoded['rpm']:7.1f}  "
                  f"cht={decoded['cht']:6.1f}  egt={decoded['egt']:6.1f}  "
                  f"oilP={decoded['oil_pressure']:5.2f}  vib={decoded['vibration']:4.2f}  "
                  f"frames={decoded['_can_frame_count']}  fps={bus.fps}")
    bus.shutdown()
