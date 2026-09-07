# MALE UAV Aero-Piston Engine — Digital Twin
### Design notes, deviations, and deployment roadmap · Problem ID 26054

---

## 1. What this is

A single-process, jury-demoable **digital twin** of a Rotax-914-class piston
engine for a MALE UAV Ground Control Station. It continuously mirrors the
physical engine from live telemetry, compares that telemetry against a
first-principles thermodynamic model, and turns the divergence into a
fault call, a remaining-useful-life number, a SHAP root-cause, and a concrete
crew advisory — end to end, on unseen mission data, in ~40 ms/frame.

```
core/scenario_runner.py   deterministic flight generator (unseen data)
        │  RPM, CHT×4, EGT×4, oil P/T, fuel, MAP, vib, alt, IAS …
        ▼
core/can_telemetry.py     python-can virtual bus  (IDs 0x100–0x104)
        │  raw 8-byte CAN frames  ->  decoded engineering units
        ▼
core/physics_model.py     steady-state thermodynamic baseline  X_physics(t)
        │  residual vector  R(t) = X_measured(t) − X_physics(t)
        ▼
ml/models.py              IsolationForest + XGBoost fault classifier
        │                 + physics-trend RUL estimator
        ▼
ml/explainability.py      cached TreeSHAP  ->  top-3 signed drivers  (<25 ms)
        ▼
core/twin_pipeline.py     advisory rules  ->  enriched frame
        ▼
dashboard/app.py          Streamlit GCS console
```

Run it: `./run_demo.sh`  (dashboard on `http://localhost:8501`)
Self-test: `./run_demo.sh --check`   Headless: `./run_demo.sh --headless <scenario>`

---

## 2. Repo map

| Path | Role |
|---|---|
| `config/engine_specs.yaml` | Rotax 914 nominal values, redlines, residual tolerances, subsystem map. Single source of truth. |
| `core/physics_model.py` | Steady-state expected-value model + residual + subsystem-health roll-up. |
| `core/scenario_runner.py` | 4 deterministic 3-minute mission scenarios + randomized-flight factory for training. |
| `core/can_telemetry.py` | Virtual CAN transport: pack telemetry → frames `0x100`–`0x104`, decode back. |
| `core/twin_pipeline.py` | Wires the whole pipeline; steps one frame at a time; also runs headless. |
| `ml/models.py` | Model loaders (+ graceful fallback), streaming feature builder, hybrid RUL. |
| `ml/explainability.py` | `LiveExplainer` — cached TreeSHAP over 50 background samples. |
| `ml/dataset.py` | Generates the training set from the scenario physics. |
| `ml/train.py` | Retrains the 3 models → `models/`. |
| `dashboard/app.py`, `dashboard/components.py` | GCS console + Plotly figure builders. |
| `models/` | Live model artifacts (retrained by `ml/train.py`). |
| `legacy/` | Original MQTT-broker pipeline + Kaggle notebook (see §6). |
| `models_legacy_kaggle/` | Original Kaggle-trained artifacts, kept for comparison. |

---

## 3. Where the implementation follows the brief

* **Digital-twin core** — `core/physics_model.py` is a genuine first-principles
  model (ISA atmosphere, turbo critical-altitude boost falloff, two-segment
  load→temperature interpolation anchored to Rotax spec values, first-order
  thermal lag). The live **residual vector** is exposed to the UI as the
  "Twin Cognition" view.
* **Health monitoring** — every parameter the brief lists (RPM, CHT, EGT,
  oil P/T, fuel flow, vibration, battery, MAP) is on the CAN bus, scored
  against physics tolerance, and rolled into four subsystem health indices.
* **Fault detection & predictive analytics** — supervised XGBoost classifier
  over {healthy, misfire, cooling degradation, lubrication degradation, CHT
  sensor drift}; held-out **macro-F1 ≈ 0.95** (split by flight, so the test
  flights are entirely unseen).
* **AI/ML layer** — anomaly detection (residual-gated IsolationForest +
  classifier agreement), RUL estimation, trend analysis, predictive
  maintenance recommendations.
* **Real-time SHAP** — the notebook's stranded `explain_prediction` is now
  `LiveExplainer.get_live_explanation`: a `TreeExplainer` built once at
  startup over a cached 50-sample background, invoked **only on anomaly
  frames**, p95 ≈ 20–30 ms, so the UI frame-rate is never gated on it.
* **CAN / FADEC ingestion** — `python-can` `virtual` backend, real 8-byte
  frames on IDs `0x100`–`0x104`, scaled-uint16 signal encoding, decoded back
  to engineering units every frame. `python core/can_telemetry.py <scn> --dump`
  prints the hex.
* **Simulation & replay** — four reproducible mission scenarios covering
  high-altitude, endurance loiter, hot-weather and rapid-throttle regimes;
  the dashboard buffers the whole run for scrub-back.
* **Visualization dashboard** — dark GCS aesthetic; scenario hot-swap with
  Play / Pause / 1×–2×–5× / Inject Fault / Reset; metric strip (Health, RUL,
  Anomaly confidence, CAN FPS, latency, stage); mission-lifecycle timeline;
  twin-cognition residual radar; measured-vs-physics track; live SHAP bar;
  per-cylinder EGT; rolling operator-advisory console.
* **Packaging** — pinned `requirements.txt`, one-command `run_demo.sh` with
  `--check` / `--headless` / `--train` modes.

---

## 4. Deviations from the brief — and why

> The brief invites a better approach if it's explained. These are the calls
> that differ from a literal reading.

### 4.1 Single-process pipeline instead of an MQTT broker mesh
The starting code ran four coordinated processes (broker → publisher → twin
core → dashboard) over MQTT. That is the right shape for a *fielded* GCS but
the wrong shape for a 4-minute evaluation: three things to start in order,
race conditions on the broker, and nothing a jury can see that the wire
buys you. The demo pipeline is now **one process** — `dashboard/app.py` owns
the scenario runner, the CAN bus, the models and SHAP, and steps them inside
the Streamlit run loop. The MQTT pipeline is preserved verbatim under
`legacy/` as the documented fleet-scale deployment path (§7).

### 4.2 CAN as the telemetry transport (not just "an option")
The brief lists "CAN bus / SocketCAN" as a *may*. We made it the actual
transport for the demo because it is the honest representation of ECU/FADEC
ingestion and it is inspectable (`--dump`). It runs in-process on
`python-can`'s `virtual` backend; swapping to a real bus in the field is a
one-line change (`interface='socketcan', channel='can0'`).

### 4.3 Physics-informed **trend** RUL as the headline number
A learned XGBoost RUL regressor is included (the brief asks for one) but it
is **not** the number shown to the operator. On held-out flights it scores
R² < 0, because in the simulator — as in real engines — time-to-failure is
only weakly correlated with any single observable at a given instant. The
headline RUL is instead a **physics-informed trend extrapolation**
(`ml/models.py:RulEstimator`): fit the recent rate of the physically-limiting
channel (oil pressure → abort threshold, CHT → redline, vibration → redline),
gate it on the residual sign so a throttle-back on descent isn't read as a
failure, and report time-to-limit with the driver named. This is accurate,
explainable ("oil pressure −0.11 bar/min, 1.4 bar of margin → ~12 min"), and
sits squarely in the brief's "physics-informed AI / hybrid thermodynamic +
data-driven" innovation area. The learned regressor is shown only as a small
cross-check in the validation footer.

### 4.4 Models retrained on the scenario physics
The shipped Kaggle models were trained on a static CSV sampled at 0.1 Hz over
multi-hour flights. The demo streams a compressed 3-minute mission at 2 Hz, so
the rolling-window, rate and "time-in-anomaly" features live on a different
time-scale. Rather than paper over the skew, `ml/dataset.py` **generates a
fresh training set from the exact physics the demo runs** (randomized
operating point, fault type, onset, severity, seed) and `ml/train.py`
retrains all three models. The four demo scenarios are never in the training
set — evaluating on them is a real generalization test. `FeatureBuilder`
(serving) and `build_features_batch` (training) are deliberately written to
produce identical features. Old artifacts are kept in `models_legacy_kaggle/`.

### 4.5 Compressed 3-minute mission, warm engine start
Each scenario is a 180 s slice covering TAXI→CLIMB→CRUISE→LOITER→EMERGENCY
RECOVERY. The engine starts at operating temperature (a MALE UAV is never
launched cold; the window is a slice of a multi-hour sortie) and the thermal
time-constants are tightened from the multi-hour-flight defaults to values
appropriate for a 2 Hz / 3-minute window, so CHT/oil-temp track the commanded
operating point instead of lagging the whole clip. Mission time is treated
1:1 with real seconds — a bearing failure genuinely can go from onset to
seizure in a few minutes, so no time-warp hand-waving is needed.

### 4.6 No 3-D airframe model
The legacy dashboard drew a stylised 3-D airframe. It was eye-candy over a
placeholder mesh (no real UAV CAD was provided) and cost frame budget. It's
dropped in favour of the residual radar and the measured-vs-physics track,
which show *where the twin thinks physics is breaking* rather than a shape.

### 4.7 IsolationForest is a corroborating signal, not the detector
On this feature space the unsupervised detector only reaches ROC-AUC ≈ 0.77 /
recall ≈ 0.18. The supervised classifier (macro-F1 ≈ 0.95) is the real
detector; an anomaly is declared only when the classifier names a non-healthy
fault **and** a physics residual is genuinely out of tolerance **and** the
breach has persisted (or is severe). IsolationForest agreement is surfaced as
`anomaly_raw` but does not by itself light the board. This keeps the false-
positive rate on the nominal scenario at zero while still flagging every
injected fault within a few seconds of onset.

---

## 5. Verified behaviour (`./run_demo.sh --check`)

| Scenario | Ground truth | Twin verdict | Health @ end | Headline RUL |
|---|---|---|---|---|
| Nominal Loiter | healthy | healthy, no anomaly | ~98 | — |
| High-Altitude Cooling Degradation | cooling_degradation | cooling_degradation (100%) | ~84 | trend, non-imminent |
| Lubrication Loss & Bearing Failure | lubrication_degradation | lubrication_degradation (100%) | ~82 | ~13 s → **ABORT** |
| Injector Clog / Cylinder Misfire | misfire | misfire (99%), EGT spread ~158 °C | ~86 | — |

Per-frame budget: twin inference p50 ≈ 35–50 ms, TreeSHAP p50 ≈ 15–22 ms
(only on anomaly frames), virtual-CAN round-trip 5 frames/sample.

---

## 6. Fallbacks (no hardware, no artifacts)

* No `python-can` → `core/can_telemetry.py` uses a byte-compatible in-process
  loopback; the wire framing is identical, the dashboard doesn't notice.
* No model artifacts → `ml/models.py` trains a small IsolationForest on
  physics-nominal vectors at startup and falls back to a rule-based fault
  classifier; SHAP degrades to a signed-residual ranking with the same
  operator-facing shape.
* No CAN hardware in the field → point `CanTelemetryBus` at `socketcan`.

Every fallback is silent-safe: the pipeline never raises in front of a jury.

---

## 7. Deployment roadmap

| Stage | Change from the demo |
|---|---|
| **Bench / HIL rig** | Replace `ScenarioRunner` with a `socketcan` reader off the real ECU/FADEC. `core/can_telemetry.py` already decodes; only the DBC signal map needs to match the engine's real message set. |
| **Onboard edge** | `core/` + `ml/` run as-is on a companion computer (Jetson / x86). Ship the XGBoost models as ONNX; TreeSHAP moves to a fixed per-class background for constant-time inference. Publish enriched frames northbound. |
| **GCS / fleet** | Re-enable the `legacy/` MQTT path as the transport between aircraft and ground: edge publishes `uav/engine/enriched`, the GCS dashboard subscribes. `uav_id` / `session_id` are already in the schema for multi-aircraft. |
| **Fleet analytics** | Persist enriched frames (the legacy SQLite schema in `legacy/Digital_twin_core.py` is a starting point) → cross-mission RUL calibration, per-tail-number degradation baselines, federated model updates. |
| **Certification** | The physics model and every threshold live in one auditable YAML; the classifier is decision-tree based and fully SHAP-explainable; no black-box end-to-end network is on the safety path. |
