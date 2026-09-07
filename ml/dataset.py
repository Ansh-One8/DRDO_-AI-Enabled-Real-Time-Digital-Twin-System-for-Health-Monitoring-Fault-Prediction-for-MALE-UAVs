"""
ml/dataset.py
=============
Training-set generation for the twin's ML layer.

Produces telemetry from the **same physics configuration the live demo runs**
(`core.scenario_runner`), but over many randomized flights -- random operating
point, fault type, onset, severity and seed. The four demo scenarios are never
in this set, so evaluating on them is a genuine generalization test.

`build_features_batch` is the vectorized twin of the streaming
`ml.models.FeatureBuilder`: identical residual definition, identical 60-s
rolling window, identical per-10-s rate, identical anomaly-seconds/10 encoding
for `time_in_anomaly_state`. Training on features from this function and
serving with `FeatureBuilder` are therefore consistent by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.physics_model import expected_engine_values, SENSOR_COLS  # noqa: E402
from core.scenario_runner import ScenarioRunner, random_flight_spec  # noqa: E402
from ml.models import (  # noqa: E402
    get_feature_columns, _TRAIN_SAMPLE_DT_S, _TRAIN_ROLL_WINDOW_S,
    _ANOMALY_RESID_GATE,
)

HZ = 2.0


# ---------------------------------------------------------------------------
# Flight generation
# ---------------------------------------------------------------------------
def generate_flights(n_flights: int, seed: int = 20240, hz: float = HZ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for fid in range(n_flights):
        spec = random_flight_spec(rng)
        runner = ScenarioRunner(hz=hz, spec=spec)
        rows = list(runner.stream())
        df = pd.DataFrame(rows)
        df["flight_id"] = fid
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Vectorized feature builder (mirrors ml.models.FeatureBuilder)
# ---------------------------------------------------------------------------
def build_features_batch(df: pd.DataFrame, hz: float = HZ) -> pd.DataFrame:
    df = df.sort_values(["flight_id", "t_s"]).reset_index(drop=True)
    win = max(2, int(round(_TRAIN_ROLL_WINDOW_S * hz)))         # 60 s
    back = max(1, int(round(_TRAIN_SAMPLE_DT_S * hz)))          # 10 s

    exp = expected_engine_values(df["throttle_pct"].values,
                                 df["altitude_m"].values,
                                 df["ambient_c"].values)
    for c in SENSOR_COLS:
        df[f"{c}_residual"] = df[c].values - exp[f"{c}_expected"].values

    g = df.groupby("flight_id", sort=False)
    for c in SENSOR_COLS:
        df[f"{c}_roll_mean"] = g[c].transform(
            lambda s: s.rolling(win, min_periods=1).mean())
        df[f"{c}_roll_std"] = g[c].transform(
            lambda s: s.rolling(win, min_periods=1).std().fillna(0.0))
        df[f"{c}_rate"] = g[c].transform(lambda s: s - s.shift(back)).fillna(0.0)

    gate = (
        (df["oil_pressure_residual"].abs() > _ANOMALY_RESID_GATE["oil_pressure"]) |
        (df["cht_residual"].abs() > _ANOMALY_RESID_GATE["cht"]) |
        (df["egt_residual"].abs() > _ANOMALY_RESID_GATE["egt"]) |
        (df["vibration_residual"].abs() > _ANOMALY_RESID_GATE["vibration"])
    )
    df["_gate"] = gate
    # consecutive anomalous run length in *samples*, then -> seconds/10
    run = df.groupby("flight_id", group_keys=False)["_gate"].apply(
        lambda s: s.groupby((~s).cumsum()).cumcount().where(s, 0))
    df["time_in_anomaly_state"] = run.values * (1.0 / hz) / _TRAIN_SAMPLE_DT_S
    df = df.drop(columns=["_gate"])

    # normalize primary fault label: collapse comma-joined multi-labels to the
    # first, and NaN RUL stays NaN
    df["fault_label"] = df["fault_label"].fillna("healthy").astype(str).str.split(",").str[0]
    return df


def make_training_frame(n_flights: int = 90, seed: int = 20240,
                        hz: float = HZ) -> tuple[pd.DataFrame, list[str]]:
    raw = generate_flights(n_flights, seed=seed, hz=hz)
    feats = build_features_batch(raw, hz=hz)
    return feats, get_feature_columns()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20240)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    feats, cols = make_training_frame(args.flights, seed=args.seed)
    print(f"{len(feats):,} rows / {feats.flight_id.nunique()} flights")
    print(feats["fault_label"].value_counts())
    print("RUL-labeled rows:", feats["remaining_useful_life"].notna().sum())
    if args.out:
        feats.to_csv(args.out, index=False)
        print("wrote", args.out)
