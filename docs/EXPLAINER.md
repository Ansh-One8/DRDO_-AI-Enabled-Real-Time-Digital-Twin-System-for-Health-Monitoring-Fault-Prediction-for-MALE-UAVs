# MALE UAV Aero-Piston Engine Digital Twin — Complete Explainer

**Smart India Hackathon (SIH) — Problem Statement ID 26054**
Prepared as a study + jury-defence guide. Read top to bottom once; after that use the
section index to jump.

---

## Table of contents

1. [The problem in plain language](#1-the-problem-in-plain-language)
2. [Glossary — every abbreviation, expanded](#2-glossary)
3. [Domain primer — how the engine works and what each sensor tells you](#3-domain-primer)
4. [System architecture — the whole pipeline](#4-system-architecture)
5. [Module deep-dives](#5-module-deep-dives)
   - 5.1 [`config/engine_specs.yaml`](#51-configengine_specsyaml)
   - 5.2 [`core/physics_model.py` — the physics-informed baseline](#52-corephysics_modelpy)
   - 5.3 [`core/scenario_runner.py` — deterministic flight generation](#53-corescenario_runnerpy)
   - 5.4 [`core/can_telemetry.py` — the virtual CAN/FADEC bus](#54-corecan_telemetrypy)
   - 5.5 [`ml/dataset.py` + `ml/train.py` — data generation and training](#55-mldatasetpy--mltrainpy)
   - 5.6 [`ml/models.py` — features, models, anomaly logic, hybrid RUL](#56-mlmodelspy)
   - 5.7 [`ml/explainability.py` — real-time SHAP](#57-mlexplainabilitypy)
   - 5.8 [`core/twin_pipeline.py` — wiring + operator advisories](#58-coretwin_pipelinepy)
   - 5.9 [`dashboard/` — the GCS console](#59-dashboard)
6. [The four scenarios, explained](#6-the-four-scenarios-explained)
7. [Design decisions and trade-offs — every "why" and "why not"](#7-design-decisions-and-trade-offs)
8. [Numbers you must be able to defend](#8-numbers-you-must-be-able-to-defend)
9. [Deployment roadmap](#9-deployment-roadmap)
10. [Jury Q&A bank](#10-jury-qa-bank)
11. [4-minute demo script](#11-4-minute-demo-script)
12. [Known limitations and honest future work](#12-known-limitations-and-honest-future-work)

---

## 1. The problem in plain language

### What is a "digital twin"?

A **digital twin** is a live, continuously updated virtual model of a physical
asset. It is not a one-off simulation — it is *synchronised* with the real thing
by a stream of sensor data, so at every moment the virtual model reflects the
current state of the physical engine. On top of that synchronised model you run
analytics: *is it healthy? what's failing? how long until it can't fly?*

Three ingredients, all present in this project:

| Ingredient | Here |
|---|---|
| **A model of the physical system** | `core/physics_model.py` — first-principles thermodynamics of the engine |
| **A live data link to the real asset** | `core/can_telemetry.py` — the CAN bus carrying ECU/FADEC telemetry |
| **Analytics that turn state into decisions** | `ml/` — anomaly detection, fault classification, remaining-useful-life, explanation |

### Why does a MALE UAV need this?

- **MALE** = **M**edium **A**ltitude **L**ong **E**ndurance — a class of drone that
  flies for 18–50+ hours at 3–9 km altitude, doing surveillance and reconnaissance.
- These aircraft use a **piston engine** (like a car engine: pistons, cylinders,
  spark, crankshaft), typically a turbocharged **Rotax 914** family engine.
- If that single engine fails mid-mission, the aircraft is lost or has to abort —
  expensive and operationally damaging.
- Today's engine monitors are **threshold-based and reactive**: they warn you
  *after* a temperature crosses a redline. By then the damage is done.
- We want **predictive**: detect the *trend* toward a failure, estimate *how long*
  you have, and tell the operator *what to do* — while there's still time to act.

### What the brief asked for (and what we built)

The brief lists deliverables A–F: a digital-twin core, a health-monitoring system,
fault detection + predictive analytics, an AI/ML layer, simulation & replay, and a
visualisation dashboard. Everything in this repo maps to one of those. Where we
deviated, [Section 7](#7-design-decisions-and-trade-offs) explains why — the brief
explicitly invites a better approach if it's justified.

---

## 2. Glossary

Every abbreviation used in the code, the UI, or this document.

### Engine parameters

| Short | Full form | What it is | Why the twin watches it |
|---|---|---|---|
| **RPM** | Revolutions Per Minute | Crankshaft rotational speed | Primary measure of how hard the engine is working; every other expected value is a function of RPM and throttle |
| **CHT** | Cylinder Head Temperature | Temperature of the aluminium head casting around each combustion chamber (°C) | Redline 150 °C. A rising CHT means heat isn't being carried away — blocked cooling, wrong mixture, or detonation. Sustained overheat cracks the head |
| **EGT** | Exhaust Gas Temperature | Temperature of the burnt gas leaving each cylinder (°C), ~800 °C at cruise | Tells you about the *combustion* itself. A lean mixture runs hotter; a dead cylinder runs cold. Comparing the four cylinders' EGTs catches a single bad one |
| **MAP** | Manifold Absolute Pressure (also "manifold pressure") | Air pressure inside the intake manifold, measured in inches of mercury (inHg) | How much air the engine is breathing. A turbocharger *raises* MAP above ambient to restore power at altitude. Falling MAP at constant throttle = induction leak or turbo failure |
| **Oil Pressure** | — | Pressure the oil pump develops to push oil through the bearings (bar) | The single most safety-critical parameter. Below ~1.5 bar in flight, bearings start to fail; total loss = seizure within minutes |
| **Oil Temp** | Oil Temperature | Bulk temperature of the engine oil (°C) | Rises with load and with friction. A climbing oil temp *with* falling oil pressure = the lubrication system is losing the fight |
| **Fuel Flow** | — | Rate the engine is consuming fuel, litres per hour (LPH) | Proportional to power. If fuel flow doesn't match the power the engine is making, the fuel metering (injectors, pump) is suspect |
| **Vibration RMS** | Vibration, Root-Mean-Square | A single number summarising vibration amplitude, in units of g (1 g = 9.81 m/s²). RMS = √(mean of squared samples) over a short window | Mechanical health. A misfire, an imbalanced prop, or a failing bearing all raise vibration |
| **Battery Voltage** | — | Voltage of the aircraft electrical bus (V), nominal 13.8 V | If it sags below ~12.4 V the alternator/charging system has failed; the ECU and avionics are now on a countdown |
| **Injection timing** | — | Crank angle before top-dead-centre at which fuel is injected (degrees BTDC) | Wrong timing = rough running, lost power, high EGT. Listed in the brief; modelled as a nominal + tolerance in the spec file |
| **IAS** | Indicated AirSpeed | Airspeed as read from pitot pressure, in knots (kts; 1 kt ≈ 0.514 m/s) | Faster airflow = more **ram-air cooling** over the cylinders. The physics model subtracts a cooling term proportional to throttle/airspeed |
| **BTDC** | Before Top Dead Centre | Crank position reference for ignition/injection timing | — |
| **TDC** | Top Dead Centre | Piston at the very top of its stroke | — |

### Atmosphere & flight

| Short | Full form | What it is |
|---|---|---|
| **ISA** | International Standard Atmosphere | An agreed reference model: 15 °C and 101 325 Pa at sea level, temperature dropping 6.5 °C per 1000 m. Used so "expected" values are defined against a standard day |
| **Altitude** | — | Height above sea level (m or ft; 1 ft = 0.3048 m). Higher = thinner air = less oxygen = less power and less cooling |
| **Density ratio** | — | Local air density ÷ sea-level ISA density. Multiplies the power/fuel terms |
| **Critical altitude** | — | The altitude up to which a turbocharger can still maintain full rated manifold pressure (~5000 m here). Above it, boost — and power — fall off |
| **Phase / stage** | — | Which part of the flight: taxi, climb, cruise, loiter, descent/recovery |
| **Loiter** | — | Flying a holding pattern over a target area — the bulk of an ISR mission |
| **ISR** | Intelligence, Surveillance, Reconnaissance | The mission these UAVs fly |

### Systems & protocols

| Short | Full form | What it is | Role here |
|---|---|---|---|
| **UAV** | Unmanned Aerial Vehicle | A drone | The platform |
| **MALE** | Medium Altitude Long Endurance | UAV class: ~3–9 km, 18–50 h | The specific platform class |
| **GCS** | Ground Control Station | The ground operator's console and radios | Where this dashboard would run |
| **ECU** | Engine Control Unit | The engine's embedded computer | Source of all telemetry |
| **FADEC** | Full Authority Digital Engine Control | An ECU that has *complete* control of the engine (throttle, mixture, ignition, boost) with no mechanical backup | The real-world telemetry source we emulate |
| **CAN** | Controller Area Network | A robust 2-wire differential serial bus, invented for cars, universal in aerospace/automotive. Messages = an 11-bit ID + up to 8 data bytes, broadcast to all nodes | Our telemetry transport |
| **SocketCAN** | — | The Linux kernel's CAN networking stack; lets you treat a CAN interface like a network socket | The one-line change to go from our virtual bus to real hardware |
| **DBC** | Database CAN (file format) | A text file that defines which signal lives in which byte of which CAN message, with scale/offset | Our `FRAME_MAP` is a hand-coded DBC equivalent |
| **MQTT** | Message Queuing Telemetry Transport | A lightweight publish/subscribe messaging protocol over TCP | The *old* pipeline's transport; kept in `legacy/` for fleet deployment |
| **RTL / RTB** | Return To Launch / Return To Base | Abort the mission and fly home | The advisory the twin issues on a critical fault |
| **HIL** | Hardware In the Loop | A test rig where real hardware runs against a simulated environment | A deployment stage |

### AI / ML / stats

| Short | Full form | What it is | Role here |
|---|---|---|---|
| **ML** | Machine Learning | Models that learn patterns from data rather than being explicitly programmed | The fault classifier, anomaly detector, RUL regressor |
| **RUL** | Remaining Useful Life | Estimated time until the asset can no longer perform its function — here, time to mission abort | A headline number on the dashboard |
| **Anomaly detection** | — | Deciding whether the current state is "not normal" *without* being told what the fault is | `IsolationForest` + our residual gate |
| **Isolation Forest** | — | An unsupervised anomaly algorithm. It builds random trees; points that get "isolated" (cut off) with very few splits are outliers. Trained only on healthy data | Corroborating anomaly signal |
| **XGBoost** | eXtreme Gradient Boosting | A high-performance library that builds an *ensemble of decision trees*, each new tree correcting the previous ensemble's errors (gradient boosting). Fast, accurate on tabular data, and — crucially — explainable | Fault classifier and the (cross-check) RUL regressor |
| **Gradient boosting** | — | Train trees sequentially; each fits the *residual error* of the running total. Sum of all trees = prediction | How XGBoost works internally |
| **Classifier** | — | A model that outputs a category (here: one of 5 fault labels) plus a probability per category | `fault_classifier.joblib` |
| **Regressor** | — | A model that outputs a continuous number (here: seconds of RUL) | `rul_regressor.joblib` (kept as cross-check only) |
| **Softmax** | — | Turns a vector of raw model scores into probabilities that sum to 1 | Gives the "Anomaly 100%" confidence number |
| **Feature** | — | One input number to the model (e.g. `cht_residual`, `vibration_roll_std`) | We build 36 per frame |
| **Residual** | — | measured − expected. The core quantity in this project | Physics subtracts the operating point; ML works on what's left |
| **Rolling window / rolling statistic** | — | A statistic (mean, std) computed over the last *N* seconds of a signal, recomputed every step | Captures trend and noisiness, not just the instantaneous value |
| **RMS** | Root Mean Square | √(mean of squares). A magnitude summary that doesn't cancel out like a plain mean would | Vibration channel |
| **SHAP** | SHapley Additive exPlanations | A method from cooperative game theory. It fairly distributes a prediction among its input features by asking "how much did each feature contribute vs. its average?" Contributions sum exactly to (prediction − baseline) | The "root cause" bar chart |
| **Shapley value** | — | The game-theory quantity SHAP computes: the average marginal contribution of a player (feature) across all possible coalitions (feature subsets) | — |
| **TreeSHAP** | — | An algorithm that computes exact SHAP values for tree ensembles in polynomial time (naive SHAP is exponential). This is why real-time SHAP is even possible | Makes per-frame explanation feasible (~20 ms) |
| **Background / reference distribution** | — | The set of "typical" data points SHAP compares against to define "average". We cache 50 healthy feature vectors | `ml/explainability.py:make_background()` |
| **tree_path_dependent** | — | A TreeSHAP mode that uses the tree's own training-data coverage as the reference, so it needs no external background and tolerates categorical splits | The mode we actually run |
| **ROC-AUC** | Receiver Operating Characteristic — Area Under Curve | A score from 0.5 (random) to 1.0 (perfect) measuring how well a detector *ranks* anomalies above normals across all thresholds | Anomaly detector: 0.77 |
| **F1 score** | — | Harmonic mean of precision and recall for one class. Balances "don't cry wolf" against "don't miss it" | Per fault class |
| **Macro-F1** | — | The plain average of the per-class F1 scores (every class counts equally, regardless of how common it is) | Fault classifier: **0.95** |
| **Precision** | — | Of the times the model said "fault X", how often was it right |
| **Recall** | — | Of the actual "fault X" cases, how many did the model catch |
| **MAE** | Mean Absolute Error | Average of |predicted − actual|. Same units as the target (seconds, for RUL) |
| **R²** | Coefficient of determination | 1.0 = perfect, 0 = no better than predicting the mean, **negative = worse than predicting the mean** | RUL regressor: −1.7 → why we don't use it as the headline |
| **Split by flight** | — | When making train/test sets, keep whole flights together — never let rows from one flight land in both. Prevents the model from "memorising" a flight it will be tested on (data leakage) | How `ml/train.py` splits |
| **Data leakage** | — | When information from the test set sneaks into training, inflating scores dishonestly | Avoided by split-by-flight and by never using ground-truth labels as features |
| **PINN / physics-informed** | Physics-Informed (Neural Network / model) | An ML approach where physical laws are baked into the model or its inputs, so it needs less data and generalises better | Our residual approach is a lightweight version of this |
| **Determinism** | — | Same inputs → byte-identical outputs every run. Achieved with fixed random seeds | Every scenario is reproducible for the demo |

### Software / infra

| Short | Full form | What it is |
|---|---|---|
| **API** | Application Programming Interface | The set of functions a module exposes to the rest of the code |
| **ODE** | Ordinary Differential Equation | An equation relating a quantity to its rate of change. The thermal lag `dT/dt = (T_target − T)/τ` is a first-order ODE |
| **EMA** | Exponential Moving Average | A running average that weights recent samples more: `ema ← α·new + (1−α)·ema`. Used for the CAN FPS readout |
| **FPS** | Frames Per Second | Here: telemetry samples processed per second |
| **YAML** | YAML Ain't Markup Language | A human-readable config file format |
| **venv** | Virtual environment | An isolated Python install (`uav_env/`) so dependencies don't clash with the system |
| **joblib** | — | A Python library for saving/loading trained models to disk (`.joblib` files) |
| **Streamlit** | — | A Python framework that turns a script into a web app; re-runs the whole script on every interaction |
| **Plotly** | — | The charting library used for every graph |
| **struct** | — | Python's module for packing/unpacking binary data — used to build the CAN byte frames |
| **uint16** | unsigned 16-bit integer | A whole number 0–65535, two bytes. Each CAN signal is encoded as one |
| **big-endian** | — | Byte order: most-significant byte first. The CAN convention we use (`>` in `struct`) |

---

## 3. Domain primer

You will be asked "do you actually understand the engine?" Here's enough to answer with confidence.

### 3.1 How a 4-stroke piston aero-engine makes power

1. **Intake:** piston moves down, drawing an air+fuel mixture into the cylinder through the intake **manifold**. A **turbocharger** (driven by exhaust gas) compresses the incoming air first, so even in thin high-altitude air the cylinder gets a sea-level-like charge. The pressure in the manifold is **MAP**.
2. **Compression:** piston moves up, squeezing the mixture.
3. **Combustion (power):** a spark ignites the mixture near **TDC** (top dead centre). The burning gas expands violently, driving the piston down — this is the only stroke that produces power. Peak gas temperature here shows up downstream as **EGT**.
4. **Exhaust:** piston moves up again, pushing burnt gas out past the exhaust valve and over the turbo.

Four cylinders fire in sequence, turning the **crankshaft**, whose speed is **RPM**. The crank drives the **propeller**. Engine power ≈ how much air it can pump (MAP × RPM) × how efficiently it burns it.

### 3.2 Where the heat goes (and why CHT matters)

Only ~⅓ of the fuel's energy becomes propeller thrust. Roughly another third leaves as hot exhaust (**EGT**), and the last third has to be removed from the engine metal or it melts. On these engines that's done by:
- **Air cooling** over the cylinder heads — proportional to airspeed and to the cooling-duct airflow (**ram-air cooling**), and
- **Oil** carrying heat from the bearings and piston undersides to an oil cooler.

**CHT** is the temperature of the head metal. If cooling airflow drops (blocked duct, thin air, low airspeed) or combustion runs too hot (lean mixture, detonation), CHT climbs toward its 150 °C redline. That's the **cooling-degradation** failure mode.

### 3.3 Why oil pressure is the scariest gauge

The crankshaft and connecting-rod bearings are plain bearings — metal surfaces separated only by a pressurised film of oil. Lose the pressure and the metal contacts, generates heat, welds, and the engine seizes — often within a couple of minutes. So a **falling oil pressure trend** is the highest-priority thing a twin can catch early. That's the **lubrication-loss** failure mode, and it's the only scenario with a hard RUL countdown to seizure.

### 3.4 What a misfire looks like

If one cylinder stops burning properly (a clogged injector starves it of fuel, or a weak spark), that cylinder:
- produces **little or no power** → the engine loses ~25% power and the crank speed becomes uneven → **vibration** rises,
- exhausts **cooler gas** (or unburnt fuel) → **that cylinder's EGT drops** well below the other three.

A single mean EGT can't see this — you need **per-cylinder EGT**. When `max(EGT) − min(EGT)` across the four cylinders exceeds ~90 °C, that's a misfire signature. This is exactly how real engine analysers (JPI, Electronics International) work.

### 3.5 Sensor faults vs. engine faults

Not every bad reading means a sick engine. A **CHT sensor drift** — the thermocouple slowly develops a bias — makes the gauge read 40 °C high while the engine is perfectly fine. A naive threshold monitor screams "overheat!" and the operator needlessly aborts. A good twin notices that CHT rose *but EGT, oil temp and everything else stayed normal* — inconsistent with real overheating — and reports **sensor drift**, not engine fault. This is the `sensor_drift_cht` scenario and it's the subtlest one.

---

## 4. System architecture

### 4.1 The pipeline (one process, top to bottom)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  core/scenario_runner.py        Deterministic flight generator               │
│    • picks a 180-second mission: TAXI→CLIMB→CRUISE→LOITER→EMERGENCY RECOVERY  │
│    • drives engine_simulator_v2's physics with throttle/altitude/ambient     │
│    • injects the scenario's fault at its onset time                          │
│    • emits a telemetry dict: RPM, CHT×4, EGT×4, oil P/T, fuel, MAP, vib,     │
│      altitude, airspeed, battery … + ground-truth label (validation only)    │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  one sample every 0.5 s of engine time (2 Hz)
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  core/can_telemetry.py          Virtual CAN / FADEC bus (python-can)         │
│    • PACK: 20 signals → 5 CAN frames, IDs 0x100–0x104, 8 bytes each,         │
│      each value scaled to a uint16 and struct-packed big-endian              │
│    • the twin side only ever sees data that survived pack → wire → unpack    │
│    • DECODE: frames → engineering-units dict; recompute mean CHT/EGT         │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  decoded telemetry
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  core/physics_model.py          Physics-informed baseline  X_physics(t)      │
│    • from throttle, altitude, ambient ONLY, compute what a HEALTHY engine    │
│      should read: ISA atmosphere → turbo model → load → temps → pressures    │
│    • residual  R(t) = X_measured(t) − X_physics(t)   per sensor              │
│    • normalised residual  = R / tolerance   (drives the "cognition" radar)   │
│    • subsystem health indices 0–100 from the residuals                       │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  residual vector + health
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ml/models.py                   ML inference layer                           │
│    • FeatureBuilder: 36 features (per sensor: residual, 60-s rolling mean &  │
│      std, per-10-s rate) + throttle/alt/ambient + time-in-anomaly            │
│    • IsolationForest → raw anomaly flag (corroboration)                      │
│    • XGBoost classifier → fault label + confidence   (the real detector)     │
│    • RulEstimator → physics trend RUL (headline)                             │
│    • XGBoost RUL regressor → cross-check number only                         │
│    • anomaly = classifier_hit AND residual-gate AND (persistent OR severe)   │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  fault, confidence, health, RUL, feature row
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ml/explainability.py           Real-time TreeSHAP   (only on anomaly frames)│
│    • one TreeExplainer built at startup, primed on 50 cached healthy vectors │
│    • get_live_explanation() → top-3 signed feature contributions, ~20 ms     │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  top-3 drivers
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  core/twin_pipeline.py          Wiring + operator advisory rules             │
│    • build_advisory(fault, health, RUL, egt_spread) → concrete crew action  │
│      + urgency tier (nominal / caution / warning / critical)                 │
│    • assembles the "enriched frame" and appends it to a 600-frame buffer     │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  enriched frame
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  dashboard/app.py + components.py   Streamlit GCS console                    │
│    • scenario hot-swap, Play/Pause, 1×/2×/5×, Inject Fault, Reset            │
│    • metric strip, mission-lifecycle timeline, twin-cognition radar,         │
│      measured-vs-physics track, live SHAP bar, per-cylinder EGT,             │
│      operator advisory console, validation footer                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Why one process (and not the original 4-process MQTT design)

The starting code ran **four separate programs** talking over an **MQTT** broker:
`broker` ← `mqtt_publisher.py` (sim) → `Digital_twin_core.py` (inference) → `dashboard.py`.
That is a good architecture for a *fielded* system where the engine and the ground
station are physically separate. For a **4-minute jury demo** it is a liability:
you have to start four things in the right order, the broker can race, and there's
nothing on screen that the extra moving parts justify.

So the demo is **one process**: `dashboard/app.py` owns the scenario runner, the
CAN bus, the models and SHAP, and steps them inside its own run loop. The MQTT
version is preserved **unchanged** in `legacy/` and is the documented path for
fleet deployment ([Section 9](#9-deployment-roadmap)).

**This is a strength, not a shortcut** — you kept the distributed design *and*
made a demo that starts with one command. Say exactly that if asked.

### 4.3 Repo map

| Path | Full form / role |
|---|---|
| `config/engine_specs.yaml` | Single source of truth: Rotax-914 nominal values, redlines, residual tolerances, subsystem map |
| `core/physics_model.py` | Steady-state thermodynamic expected-value model + residual + health |
| `core/scenario_runner.py` | 4 deterministic 3-minute scenarios + randomised-flight factory for training |
| `core/can_telemetry.py` | Virtual CAN transport: pack telemetry → frames 0x100–0x104, decode back |
| `core/twin_pipeline.py` | Wires the whole pipeline; steps one frame; also runs headless |
| `engine_simulator_v2.py` | The underlying physics simulator (pre-existing, reused): ODE thermal integration, per-cylinder modelling, fault injection |
| `ml/dataset.py` | Generates the training set from the scenario physics |
| `ml/train.py` | Trains the 3 models → `models/` |
| `ml/models.py` | Model loaders (+ fallback), streaming feature builder, hybrid RUL |
| `ml/explainability.py` | `LiveExplainer` — cached TreeSHAP |
| `dashboard/app.py` | Streamlit GCS console + run loop |
| `dashboard/components.py` | Plotly figure builders (no Streamlit calls) |
| `models/` | Live model artifacts (produced by `ml/train.py`) |
| `models_legacy_kaggle/` | Original Kaggle-trained artifacts, kept for comparison |
| `legacy/` | Original MQTT pipeline + training notebook + old dashboard |
| `run_demo.sh` | One-command launcher: dashboard / `--headless` / `--check` / `--train` |
| `DESIGN_NOTES.md` | Short architecture + deviation summary |
| `docs/EXPLAINER.md` | This document |

---

## 5. Module deep-dives

### 5.1 `config/engine_specs.yaml`

**Why a config file at all?** Two reasons a jury will appreciate:
1. **Auditability.** A propulsion engineer can open one file and check every number
   against the Rotax operator manual — no hunting through code.
2. **Certification.** For a real defence system, the safety-relevant constants
   (redlines, tolerances) must be reviewable and change-controlled. Keeping them
   out of code is a step toward that.

**What's in it:**
- `engine:` nominal + redline values for RPM, MAP, CHT, EGT, oil pressure/temp,
  fuel flow, vibration, battery, injection timing; turbo critical altitude;
  cylinder count.
- `thermal_time_constants_s:` how fast CHT (120 s), EGT (5 s), oil temp (240 s)
  respond to a change — used for the lag model.
- `residual_tolerance:` per-sensor "how far from physics is still healthy" band
  (RPM ±200, CHT ±12 °C, EGT ±40 °C, oil pressure ±0.5 bar, …). **These set the
  sensitivity of the whole twin.** Tighten them → more sensitive, more false
  alarms. They were tuned so the nominal scenario sits at ~97–99 % health.
- `subsystems:` which sensors roll up into `thermal_health`, `lubrication_health`,
  `combustion_health`, `electrical_health`.
- `atmosphere:` ISA sea-level reference (15 °C, 101325 Pa, 6.5 °C/km lapse rate).

Loaded once and cached (`functools.lru_cache`).

### 5.2 `core/physics_model.py`

**This is the "physics-informed" half of "physics-informed AI." Understand this section cold.**

#### The idea

A pure data-driven monitor has to learn, from data, that "cruise at 4 km on a
−12 °C day should have CHT ≈ 90 °C." That's a lot to learn, it needs data covering
every altitude/temperature/throttle combination, and it breaks on conditions it
never saw. Instead we **compute** the expected value from first principles, then
let the ML work only on the **residual** (`measured − expected`). The operating
point is subtracted out; the model only has to recognise "fault-shaped" residuals.

#### The equations (all in `expected_engine_values()`)

Inputs: `throttle_pct`, `altitude_m`, `ambient_c`. Everything below is vectorised
(works on arrays for training, scalars for live).

**1. Atmosphere** — the barometric formula (standard atmosphere pressure vs height):
```
pressure_pa = 101325 · (1 − 2.25577e-5 · altitude_m) ^ 5.25588
```
Air gets exponentially thinner with height. (Density then follows from the ideal
gas law, `ρ = p / (R·T)`.)

**2. Turbocharger model** — a turbo restores sea-level manifold pressure up to a
**critical altitude** (~5000 m), then can't keep up:
```
if altitude ≤ 5000 m:   turbo_map_ceiling = MAP_MAX,   density_ratio_effective = 1.0
else:
   excess = altitude − 5000
   falloff = clip( (1 − 2.25577e-5 · excess) ^ 5.25588 ,  0.3, 1.0 )
   turbo_map_ceiling = MAP_MAX · falloff
   density_ratio_effective = falloff
```
This is why a MALE UAV uses a *turbocharged* engine — a naturally-aspirated one
would lose too much power above 3–4 km.

**3. Manifold pressure and RPM from throttle:**
```
MAP_expected = MAP_IDLE + throttle · (turbo_map_ceiling − MAP_IDLE)

RPM_expected = { IDLE + (thr/0.5)·(CRUISE − IDLE)          if throttle ≤ 50%
              { CRUISE + ((thr−0.5)/0.5)·(MAX − CRUISE)    if throttle > 50%
```
Two straight-line segments anchored at the three spec points (idle 1400, cruise
5000, max 5800). Simple, monotonic, no discontinuity at the join.

**4. Engine load** — the single number everything thermal depends on:
```
load_factor = (MAP_expected / MAP_MAX) · (RPM_expected / RPM_MAX)     [0 … ~1]
```
Physically: air pumped per unit time ÷ max air pumped per unit time.

**5. Temperatures** — two-segment interpolation anchored to real spec values,
because a real engine has only a *narrow* gap between cruise and redline EGT
(~100 °C) despite a big throttle change, so a single linear multiplier overshoots:
```
if load ≤ 0.7 (the "cruise reference" load):
    frac = load / 0.7
    EGT_expected = 420 + frac·(800 − 420)
    CHT_expected =  90 + frac·(125 −  90)
else:
    frac = clip((load − 0.7) / 0.3, 0, 1.2)
    EGT_expected = 800 + frac·(900 − 800)
    CHT_expected = 125 + frac·(150 − 125)
```
Then CHT is corrected for the day and the cooling airflow:
```
CHT_expected += (ambient_c − 15)          # hotter day → hotter head
CHT_expected -= throttle · 15             # more airspeed/duct flow → more cooling
```

**6. Oil, fuel, vibration, battery** — each a simple monotonic function of load
or RPM:
```
OilTemp_expected  = 60 + 40 · load · 1.2  + 0.5·(ambient−15) − 0.5·(throttle·15)
OilPress_expected = 0.8 + (5.0 − 0.8) · (RPM_expected / RPM_MAX)     capped at 7.0
Vibration_expected = 1.0 · (0.5 + 0.5 · load)
FuelFlow_expected  = 25 · load · density_ratio_effective
Battery_expected   = 13.8   (13.2 if RPM < 2000, i.e. alternator barely spinning)
```

#### From expected values to health

```
residual(sensor)            = measured − expected
normalised_residual(sensor) = residual / tolerance(sensor)      # 0 = perfect, 1 = edge of healthy
score(sensor)               = 100 · exp( −0.5 · ratio² / 4 )    # ratio = |normalised residual|
subsystem_health            = mean( score  over that subsystem's sensors )
health_index                = mean( the four subsystem healths )
```
The Gaussian `score` curve means: residual 0 → 100; at 1× tolerance → ~94;
at 2× → ~78; at 3× → ~50; at 4× → ~28. Gentle near zero, punishing far out.

#### What about transients?

The *steady-state* physics above assumes temperatures have settled. Real CHT lags
throttle changes by ~2 minutes. The **simulator** applies that lag (see 5.3); the
physics baseline does not. To keep the two comparable inside a 180-second window
we (a) start the simulated engine already warm and (b) shorten the simulator's
thermal time-constants for the compressed timeline. Result: on the nominal
scenario the residuals stay small (health 97–99 %). This is discussed openly in
[Section 7](#7-design-decisions-and-trade-offs) and [Section 12](#12-known-limitations-and-honest-future-work).

### 5.3 `core/scenario_runner.py`

**Purpose:** generate *fresh, unseen* telemetry every run — not replay a CSV the
model trained on. This directly answers the brief's "replace self-looping training
datasets with dynamic generation."

#### The compressed mission

Every scenario is **180 seconds** and walks the full lifecycle:

| Stage | Time | Throttle | What's happening |
|---|---|---|---|
| **TAXI** | 0–15 s | 15 % | On the ground, engine idling |
| **CLIMB** | 15–50 s | 88 % | Full-power climb to cruise altitude |
| **CRUISE** | 50–95 s | 66 % | Transit to the operating area |
| **LOITER** | 95–150 s | 55 % | Holding pattern over the target (the bulk of an ISR mission) |
| **EMERGENCY RECOVERY** | 150–180 s | 35 % | Descent / divert |

`MissionShape` defines `throttle_at(t)`, `altitude_at(t)`, `ambient_at(t)`,
`airspeed_at(t)` as smooth functions of time, with a small `1.2·sin(t/7)` breathing
term so the charts aren't dead-flat. Scenario-specific twists (e.g. an
over-altitude climb) are layered on as bias functions.

#### How a scenario runs

`ScenarioRunner.step()` each tick:
1. Look up commanded throttle / altitude / ambient / airspeed for time `t`.
2. Ask `FaultInjector` for the physical modifiers active at `t`
   (`cooling_health`, `lubrication_health`, `combustion_health`, `is_failed`).
3. Call `EnginePhysicsModel.update(dt, throttle, altitude, ambient, mods)` — the
   pre-existing simulator (`engine_simulator_v2.py`) that does the *actual*
   physics with a proper **ODE thermal integration**
   `T ← T_target + (T − T_target)·e^(−dt/τ)`, per-cylinder manufacturing bias,
   turbo, and a coherent terminal-failure state.
4. Apply sensor faults (e.g. CHT drift adds a growing bias to the *reading*).
5. Apply the scenario's `shaping_fn` (e.g. push cylinder 3's EGT down for misfire).
6. Emit the telemetry dict + the ground-truth `fault_label` and `remaining_useful_life`.

#### Determinism

Each scenario has a fixed integer seed (101/202/303/404). All randomness
(sensor noise, cylinder bias, misfire timing) flows from that one seed, so a
given scenario produces **byte-identical** telemetry every run — essential for a
reproducible demo, and it means the four scenarios are always *out-of-sample*
relative to the training set (which uses different seeds).

#### `random_flight_spec()`

For training only: builds a randomised `ScenarioSpec` — random cruise altitude
(3–7.2 km), random hot-weather flag, ~62 % chance of a fault, random fault type /
onset / ramp / severity / seed. `ml/dataset.py` calls this 140 times.

### 5.4 `core/can_telemetry.py`

**Purpose:** make "ingest ECU/FADEC telemetry over CAN" concrete and inspectable,
not a bullet point.

#### What CAN actually is

**CAN** (Controller Area Network) is the serial bus that engine ECUs, avionics
and actuators use. Physically it's two twisted wires. Logically, each **message**
is:
- an **arbitration ID** (11 bits — also sets priority), and
- **0–8 bytes** of payload,
broadcast to every node on the bus. There's no "connection" — nodes just publish
messages with their ID and everyone who cares listens. A **DBC** file says which
physical quantity lives in which bits of which message, and how to scale it.

We use **`python-can`** with its `virtual` backend — a fully in-process bus with
the real `python-can` API. Swapping to hardware is one line:
`interface='virtual'` → `interface='socketcan', channel='can0'`.

#### Our message set (a hand-coded DBC)

`FRAME_MAP` — 5 messages, IDs `0x100`–`0x104`, 8 bytes each = **4 signals per
message** (`struct` format `>4H` = big-endian, four unsigned 16-bit ints):

```
0x100  ENGINE_CORE    rpm | manifold_pressure | oil_pressure | oil_temp
0x101  CHT_BANK       cht_1 | cht_2 | cht_3 | cht_4
0x102  EGT_BANK       egt_1 | egt_2 | egt_3 | egt_4
0x103  FLOW_DYNAMICS  fuel_flow_lph | vibration | altitude_ft | airspeed_kts
0x104  AUX_STATE      battery_voltage | throttle_pct | ambient_c | mission_time
```
That's 20 signals, and it *includes per-cylinder CHT and EGT* — which is how a
real engine analyser reports, and what the misfire detection needs.

#### Encoding — how a float becomes bytes

Each signal has a `scale` and an integer `offset`:
```
raw_uint16 = round(value / scale) + offset            clamped to 0 … 65535   (the wire form)
value      = (raw_uint16 − offset) · scale                                    (decode)
```
- `scale` sets resolution and range. RPM uses `scale = 0.2` → 0–13107 RPM in
  0.2-RPM steps. Oil pressure uses `0.0002` → 0–13.1 bar in fine steps.
- `offset` lets a **signed** quantity fit an **unsigned** field. Ambient
  temperature can be −40 °C; with `scale = 0.005, offset = 10000`,
  −12.3 °C → `round(−12.3/0.005) + 10000 = 7540` on the wire → decodes back to
  −12.3 °C. (Getting this backwards was a real bug during the build — the twin saw
  +50 °C ambient and every CHT residual blew up. Fixing the encoder fixed it.
  Good story for "how do you know it works?" — because a scaling error is
  *visible*.)

#### Decoding

`bus.poll()` drains all queued frames, `struct.unpack(">4H", data)` each one,
inverts the scale/offset, and merges into one telemetry dict. Mean CHT/EGT are
recomputed from the four cylinder values. Metadata is attached:
`_can_frames` (which IDs arrived), `_can_hex` (raw hex per frame),
`_can_frame_count`, `_can_rx_ts`. A rolling **EMA** of the publish interval gives
the "CAN FPS" readout.

Try it: `python core/can_telemetry.py injector_misfire --dump` prints the hex.

#### Fallback

If `python-can` isn't installed, `_LoopbackBus` is used — same public methods,
same byte framing (`struct` is stdlib), so the pipeline never notices.

### 5.5 `ml/dataset.py` + `ml/train.py`

#### Why retrain at all?

The models shipped in the repo were trained (on Kaggle) on a **static CSV**
sampled at **0.1 Hz** (one row per 10 s) over **multi-hour** flights. The demo
streams a **2 Hz**, **3-minute** mission. Several features live on a *time scale*:
- rolling mean/std over "6 samples" is 60 s at 0.1 Hz but only 3 s at 2 Hz,
- `rate` as a "sample-to-sample diff" is a per-10-s change vs a per-0.5-s change,
- `time_in_anomaly_state` counted samples, so its numeric range differs 20×.

Feeding 2 Hz features to a 0.1 Hz-trained model is a **train/serve skew** and it
showed (classifier collapsed to "healthy" for everything). Two ways to fix it:
paper over the skew, or **retrain on data that matches how the twin actually
runs**. We did the second — it's the honest fix and it's what the brief asks for
("dynamic generation").

#### `ml/dataset.py`

- `generate_flights(n)` — runs `n` randomised `ScenarioRunner` flights (via
  `random_flight_spec`), concatenates their telemetry with a `flight_id`.
- `build_features_batch(df)` — the **vectorised twin of the streaming
  `FeatureBuilder`**: identical residual definition, identical 60-s rolling
  window (`int(60·hz)` rows), identical per-10-s rate (`shift(int(10·hz))`),
  identical `time_in_anomaly_state = run_length · (1/hz) / 10`. Because training
  features and serving features come from code that is deliberately kept
  line-for-line equivalent, **there is no train/serve skew by construction.**

#### `ml/train.py`

1. Generate 140 randomised flights → ~50 000 rows.
2. **Split by flight**, 80/20 — whole flights go to train *or* test, never both.
   No row-level leakage; the test flights are genuinely unseen.
3. Train three models:
   - **Anomaly** — `IsolationForest(n_estimators=250, contamination=0.03)` on the
     **healthy rows only**. It learns the shape of the healthy residual cloud.
   - **Fault classifier** — `XGBClassifier(n_estimators=350, max_depth=6,
     learning_rate=0.08, subsample=0.85, colsample_bytree=0.85)`,
     `objective="multi:softprob"` over the 5 labels.
   - **RUL regressor** — `XGBRegressor(n_estimators=400, max_depth=5,
     learning_rate=0.06)` on the rows that have a real countdown (degradation
     faults only).
4. Save to `models/`, write `models/metrics.json`.

Runs in ~20 s. `./run_demo.sh --train` reruns it.

### 5.6 `ml/models.py`

#### The 36 features (`get_feature_columns()`)

For each of the 8 sensors `[rpm, cht, egt, oil_pressure, oil_temp, fuel_flow,
vibration, battery_voltage]`:

| Feature | Meaning | Why it helps |
|---|---|---|
| `<s>_residual` | measured − physics-expected | The core signal — operating point removed |
| `<s>_roll_mean` | mean of the sensor over the last 60 s | Smooths noise; captures the current level |
| `<s>_roll_std` | std dev over the last 60 s | **Roughness** — jumps when a misfire or bearing problem starts |
| `<s>_rate` | change over the last 10 s | **Trend direction and speed** — a slow drift vs a fast collapse |

Plus 4 context features: `throttle_pct`, `altitude_m`, `ambient_c`,
`time_in_anomaly_state` (how long, in 10-s units, a residual has been outside its
gate — a proxy for "how developed is this trend").

**8 × 4 + 4 = 36.** Note: **no ground-truth label is ever a feature** — that would
be leakage.

#### The three models and how they're combined

```
raw_anomaly   = IsolationForest.predict(X) == −1          # unsupervised, corroboration
proba         = XGBClassifier.predict_proba(X)            # 5 probabilities
fault, conf   = argmax(proba)                             # the supervised call

residual_gate = any |residual| beyond its gate value      # oil>0.5, cht>15, egt>50, vib>0.5
classifier_hit= fault != "healthy"  AND  conf ≥ 0.60
persistent    = the gate has held ≥ 3 s
severe        = max normalised residual ≥ 5×

ANOMALY  =  classifier_hit  AND  residual_gate  AND  (persistent OR severe)
```

**Why this AND-logic?** Three independent checks must agree:
1. the *supervised* model (macro-F1 0.95) names a specific fault with confidence,
2. a *physics* residual is genuinely out of tolerance (not just model noise),
3. it has *lasted* (or is extreme) — so a one-frame throttle-slam transient during
   the climb never lights the board.

`raw_anomaly` from the IsolationForest is carried in the output as `anomaly_raw`
for corroboration but **does not by itself trigger an alert** — on this feature
space its recall is only 0.18 (see [Section 8](#8-numbers-you-must-be-able-to-defend)),
so the supervised classifier is the real detector and the forest is a sanity check.

#### Hybrid, physics-first RUL (`RulEstimator`)

The learned RUL regressor scores **R² = −1.7** on held-out flights — worse than
guessing the mean. That's not a bug we can train away: in the simulator (as in
real engines) the *total life* after a fault onset is only weakly related to any
instantaneous reading, on purpose, to give flight-to-flight variety. So a
snapshot regressor can't learn it.

Instead the headline RUL **measures the actual rate toward a known limit**:
```
for each limiting channel (oil pressure, CHT, vibration):
    slope = least-squares fit over the last ~45 s of that channel
    if the channel is degrading   AND   its residual confirms it's abnormal:
        RUL_channel = (current_value − limit) / (−slope)      # or (limit − value)/slope
RUL = min(RUL_channel over all channels),  and name which channel drove it
```
- Limits from the spec file: oil pressure → 1.5 bar abort threshold; CHT → 150 °C
  redline; vibration → 4 g redline.
- The **residual guard** (`residual < −0.4 bar` for oil, `> +6 °C` for CHT) is
  what stops a throttle-back on descent — which legitimately lowers oil pressure —
  from being read as a failure.
- This is genuinely **physics-informed AI**: it's an algorithm in the ML layer,
  it's adaptive to the observed data, and it's fully explainable
  ("oil pressure −0.11 bar/min, 1.4 bar of margin → ~12 min"). It sits in the
  brief's "hybrid thermodynamic + data-driven" innovation area.
- The learned regressor's number is still computed and shown as a small
  cross-check in the validation footer, flagged unreliable if outside 0–3600 s.

#### Graceful fallback

If a `.joblib` file is missing or won't unpickle:
- anomaly → a small `IsolationForest` trained at startup on physics-nominal
  vectors,
- classifier → a rule-based function (`_rule_fault`: big negative oil residual →
  lubrication; EGT spread > 90 or high vibration → misfire; high CHT residual →
  cooling; …),
- so the pipeline **never raises**.

### 5.7 `ml/explainability.py`

#### What SHAP does, briefly

Given a trained model and one input, **SHAP** answers "how much did each feature
push this specific prediction away from the model's average prediction?" It comes
from cooperative game theory: treat features as players, the prediction as the
payout, and compute each player's fair share (its **Shapley value**) by averaging
its marginal contribution over all possible feature subsets. The shares sum
exactly to `(this prediction − baseline prediction)` — that additivity is why
it's trustworthy.

Naive SHAP is exponential in the number of features. **TreeSHAP** computes it
*exactly* for tree ensembles in polynomial time by walking the trees — that's the
only reason per-frame SHAP is feasible here.

#### How we operationalised it

The training notebook (cell 18/19) had a one-off `explain_prediction` that built a
fresh explainer for a single row. That's the "stranded SHAP logic" the brief
mentions. `LiveExplainer`:
1. **At startup**: build **one** `shap.TreeExplainer` over the fault classifier.
   Prime it with a **cached background of 50 representative healthy feature
   vectors** (`make_background()` generates 6 short flights and samples 50 healthy
   rows through the *same* feature transform). Warm it up with one throwaway call
   so the tree paths are JIT-compiled.
2. Two construction strategies, tried in order:
   `interventional` + the cached background (textbook "SHAP vs a reference set"),
   then `tree_path_dependent` (no background needed; tolerates the categorical
   splits XGBoost created). We run on the second.
3. **Per frame, only when an anomaly is live**: `get_live_explanation()` computes
   SHAP values for the *predicted class*, takes the **top 3 by magnitude**, and
   returns each with its raw value, sign, and share of the total attribution.
   Measured latency (`time.perf_counter()`): **p50 ≈ 15–22 ms, p95 ≈ 30 ms**,
   well under the 80 ms budget, and it's off the critical path on healthy frames.
4. **Fallback**: if SHAP is unavailable, rank the residual features by
   `|value| / tolerance` — same operator-facing shape, never a blank panel.

### 5.8 `core/twin_pipeline.py`

The glue. `TwinPipeline`:
- **`__init__`**: loads `TwinModels` (heavy — the joblibs), builds the SHAP
  background, constructs `LiveExplainer`. Done once.
- **`reset(scenario)`**: spins up a fresh CAN bus and `ScenarioRunner`, clears the
  feature builder's history and the 600-frame ring buffer.
- **`step()`**: one full frame — `runner.step()` → `bus.publish()` →
  `bus.poll()` (decode) → re-attach the stage label + ground truth (not on the
  wire) → `models.infer()` → if anomaly, `explainer.get_live_explanation()` →
  `build_advisory()` → assemble the enriched frame → append to the buffer.
- **`run()`**: a headless loop for `./run_demo.sh --headless`.

**`build_advisory(fault, health, rul_s, anomaly, egt_spread)`** — rule-based,
deliberately *not* ML, because a crew action must be traceable:
- `imminent = rul_s < 90 s`, `near = rul_s < 300 s`.
- Per fault there's a **high-urgency** and a **low-urgency** message; pick high if
  `imminent or near or health < 55`.
- Urgency tier: `critical` if imminent; `warning` if near or health < 55;
  else `caution`; `nominal` if no anomaly.
- Example (lubrication, imminent): *"Oil system integrity degrading. Throttle back
  to 55 %, minimise load. RUL below abort threshold — execute RTL / nearest-
  recovery NOW."*

### 5.9 `dashboard/`

**`dashboard/app.py`** — the Streamlit app. Streamlit's model is "re-run the whole
script on every interaction," so:
- The heavy pipeline is created once via `@st.cache_resource` (shared across
  reruns).
- `st.session_state` holds `playing`, `speed` (1/2/5), `scenario`, and the rolling
  advisory list.
- **The run loop**: if `playing` and the mission isn't over, call `pipe.step()`
  `speed` times, render everything, `time.sleep(0.12)`, `st.rerun()`. So the app
  advances 1/2/5 simulated frames per ~0.12 s screen refresh.
- Header controls: scenario `selectbox`, Play / Pause, speed `radio`,
  **Inject Fault**, Reset.
- **Inject Fault** (`_inject_fault`): on the nominal scenario it hot-swaps to
  `lubrication_loss` and plays; on a fault scenario it resets and silently
  replays up to `(fault_onset − 6 s)` so the rolling features are populated, then
  plays — the degradation starts within a couple of seconds instead of after 60+.

**`dashboard/components.py`** — pure Plotly figure builders, dark palette, no
Streamlit calls (so they're unit-testable). The six figures are described in
[Section 6](#6-the-four-scenarios-explained) context and in the chat explanation
you already have; the key ones:
- `mission_lifecycle_fig` — time series with stage bands, throttle, altitude,
  health, anomaly markers.
- `twin_cognition_fig` — polar bar chart of `|residual| / tolerance` per sensor,
  with the 1× tolerance ring.
- `residual_track_fig` — measured (CAN) vs physics-baseline for the current worst
  channel, with the gap shaded.
- `shap_bar_fig` — horizontal bar of the top-3 SHAP drivers.
- `per_cylinder_egt_fig` — the four cylinder EGT lines.
- `fmt_rul` — turns RUL seconds into `MM:SS to limit` / `NNs — ABORT` /
  `ABORT NOW` with a severity tag.

---

## 6. The four scenarios, explained

Run any of them from the dashboard dropdown, or `./run_demo.sh --headless <key>`.

### `nominal_loiter` — the baseline
Stable ISR loiter, ISA conditions, no fault. Everything should track physics:
health 97–99 %, anomaly CLEAR, RUL "—". **This is the control** — it proves the
twin doesn't cry wolf. If a jury only remembers one thing, it should be that the
green line stays flat here.

### `altitude_cooling_degradation` — Cylinder-Head Cooling Degradation
- **Injected**: `FaultInjector` ramps `cooling_health` down from 1.0 (onset 45 s,
  110 s ramp) — the cooling airflow is progressively lost. The scenario also
  pushes the climb ~2.6 km higher than planned (thinner air, less cooling), and
  `_cooling_shaping` adds a direct CHT rise (heat that the fins can no longer
  shed) plus rougher running.
- **Signature**: CHT climbs 30–40 °C above the physics baseline; EGT rises
  (leaner burn); vibration creeps up. Oil pressure, RPM, fuel all normal.
- **Twin should say**: `cooling_degradation`, ~100 % confidence; health ~72–84;
  advisory *"reduce throttle, increase airspeed for ram cooling, descend to
  denser air."* RUL usually "no imminent hard limit" because CHT, while high,
  isn't racing the 150 °C redline within 3 minutes.
- **The radar**: CHT and Vibration bars way past the ring; everything else inside.

### `lubrication_loss` — Lubrication Loss & Bearing Failure
- **Injected**: `lubrication_health` ramps down (onset 72 s, 70 s ramp), which in
  the simulator scales oil pressure down and injects friction heat; the scenario
  carries `total_life_s = 150 s` — a real countdown to seizure.
- **Signature**: oil pressure collapses from ~5 bar toward 1.5; oil temp rises;
  vibration spikes (bearing rumble).
- **Twin should say**: `lubrication_degradation`, ~100 %; **headline RUL counting
  down to ~10–15 s → "ABORT NOW"**, driver *"oil pressure → abort limit"*;
  advisory *"execute RTL / nearest-recovery NOW."*
- **SHAP**: `Oil Pressure residual` dominates (~60 % of attribution).
- **This is the money scenario** — it shows prediction *before* failure, a
  concrete time budget, and an unambiguous action.

### `injector_misfire` — Injector Clog / Cylinder Misfire
- **Injected**: `FaultInjector` drops `combustion_health` randomly (onset 60 s);
  `_misfire_shaping` progressively depresses **cylinder 3's EGT** by up to 140 °C,
  drops its CHT, and adds vibration and a small RPM/fuel loss.
- **Signature**: on the **per-cylinder EGT** chart, cylinder 3's line peels away
  from the other three; `max−min` spread blows past 90 °C; vibration jumps; a
  slight power loss.
- **Twin should say**: `misfire`, ~99 %; health ~85; advisory *"single-cylinder
  misfire confirmed (EGT spread N °C) … plan a precautionary landing."*
- **SHAP**: `Vibration RMS roll_std` (the roughness feature) leads.

*(A fifth fault type, `sensor_drift_cht`, exists in the model and the training
set but isn't one of the four demo buttons — mention it as "the twin can also
distinguish a lying sensor from a real overheat," see [Section 3.5](#35-sensor-faults-vs-engine-faults).)*

---

## 7. Design decisions and trade-offs

Every one of these is a "why did you / why didn't you" a jury can ask.

### Why physics-informed residuals instead of raw-signal deep learning?
- **Data efficiency**: the model doesn't have to learn the entire operating
  envelope; physics removes it. We train on 140 short flights, not millions of
  hours.
- **Generalisation**: a raw-signal model trained on temperate low-altitude data
  would flag a hot-day high-altitude cruise as anomalous. Residuals are
  operating-point-invariant.
- **Explainability & certification**: the physics is auditable YAML + closed-form
  equations; the classifier is decision trees with exact SHAP. Nothing on the
  safety path is a black box.
- **Trade-off**: the physics model must be reasonably accurate or the residuals
  carry a systematic bias. We accept a modelling burden in exchange for the
  above.

### Why XGBoost and not a neural network?
- Tabular data with 36 features and ~50 k rows — **gradient-boosted trees are the
  known best-in-class** here; a neural net would need far more data to match it.
- **TreeSHAP** gives *exact* per-prediction explanations in milliseconds. Neural
  explanations (integrated gradients, KernelSHAP) are approximate and slower.
- Trees are **inspectable** and quantise/deploy trivially to an edge computer
  (ONNX, or even hand-rolled).
- Trade-off: trees don't extrapolate beyond the training range — mitigated
  because the *physics* handles the operating-point range and the trees only see
  bounded residuals.

### Why a single process for the demo, when the brief implies a distributed GCS?
Covered in [4.2](#42-why-one-process-and-not-the-original-4-process-mqtt-design).
Short version: the distributed **MQTT** design is preserved in `legacy/` and is
the fleet-deployment path; the demo collapses it to one process so it starts with
one command and has no broker race. You didn't throw the architecture away — you
have both.

### Why is the headline RUL a physics trend and not the trained regressor?
The regressor scores **R² < 0**. In the simulator (and in reality) time-to-failure
isn't a function of an instantaneous snapshot — it depends on the unobserved
severity of the initiating fault. So we **measure the rate toward a known limit**
instead, which is accurate, explainable, and adaptive. The regressor's output is
still shown as a cross-check. This is squarely in the brief's "hybrid
thermodynamic + data-driven" innovation area.

### Why does the IsolationForest have low recall — is the anomaly detection weak?
The **unsupervised** detector alone gets ROC-AUC 0.77 / recall 0.18 on this
feature space — unsupervised methods struggle when faults and healthy transients
overlap in feature space. But the **detector isn't the IsolationForest** — it's
the *supervised classifier* (macro-F1 0.95) plus the physics residual gate plus a
persistence check, all of which must agree. The forest is a corroborating sanity
check. Net result: **zero false alarms on the nominal scenario, every injected
fault caught within a few seconds of onset** (verified by `./run_demo.sh --check`).

### Why not a 3-D model of the airframe?
The legacy dashboard drew a stylised 3-D aircraft. No real UAV CAD was provided,
so it was decoration over a placeholder mesh, and it cost frame budget. It's
replaced by the **residual radar** and the **measured-vs-physics track**, which
show *where the twin believes physics is breaking* — information, not ornament.

### Why compress the mission to 180 seconds, and start the engine warm?
A real ISR sortie is 18–50 h; you can't demo that. 180 s covers the full
lifecycle. The engine starts warm because (a) a MALE UAV is never launched cold —
the window is a slice of a long sortie — and (b) the steady-state physics baseline
would otherwise disagree with a still-warming simulator for the whole clip. The
simulator's thermal time-constants are shortened to match the compressed
timeline. This is a **modelling choice for the demo**, stated openly; the physics
*equations* are unchanged.

### Why keep the old files instead of deleting them?
`legacy/` (MQTT pipeline, training notebook, old dashboard) and
`models_legacy_kaggle/` (original artifacts) are kept so the evolution is visible
and the fleet-deployment path is documented, not lost.

### Why Streamlit?
Fastest path to a real-time dashboard in pure Python — no separate frontend
build, no JS. Trade-off: Streamlit re-runs the whole script per interaction, so
we drive the sim in the run loop and cache the heavy objects. For a production
GCS you'd move to a proper event-driven frontend (the `legacy/` design already
separates concerns for that).

### Why simulated data and not a real engine dataset?
No instrumented MALE UAV engine dataset is publicly available, and the brief
explicitly permits "simulated or real engine datasets." The simulator is
physics-based (ODE thermal integration, turbo model, per-cylinder effects), and
the twin's physics baseline is an *independent* implementation, so we're not just
checking a model against itself. Swapping in a real CAN feed is a one-line
transport change ([Section 9](#9-deployment-roadmap)).

---

## 8. Numbers you must be able to defend

From `models/metrics.json` (held-out, split by flight):

| Metric | Value | How to talk about it |
|---|---|---|
| **Fault classifier macro-F1** | **0.95** | "Averaged equally over all five classes, on flights the model never saw. Per-class F1 ranges ~0.91 (misfire) to ~0.97 (cooling)." |
| Classifier accuracy | ~0.95 | Secondary to macro-F1 because the classes are imbalanced (healthy dominates) |
| **Anomaly detector ROC-AUC** | 0.77 | "The *unsupervised* layer alone. It's a corroborating signal — the supervised classifier plus the physics gate is the actual detector." |
| Anomaly detector recall / FPR | 0.18 / 0.03 | "Low standalone recall is expected for unsupervised detection when faults and transients overlap; that's *why* it's not used alone." |
| **RUL regressor R²** | **−1.7** | "Deliberately not the headline number — time-to-failure isn't learnable from a snapshot here. The physics trend estimator is the headline; this is a logged cross-check." |
| RUL regressor MAE | ~91 s | Same caveat |
| **End-to-end scenario check** | **4/4 PASS** | `./run_demo.sh --check` — nominal stays clear, all three faults are caught and correctly classified |
| Twin inference latency | p50 ~35–55 ms | `time.perf_counter()` around `models.infer()` — feature build + both models + health + RUL |
| SHAP latency | p50 ~15–22 ms, p95 ~30 ms | Only on anomaly frames; budget was 80 ms |
| CAN | 5 frames / sample, IDs 0x100–0x104 | Real pack→unpack every frame |

**Rule for the jury:** never oversell. "The classifier is strong (0.95), the
unsupervised anomaly layer is a weaker corroborator (0.77) which is why it's not
the sole trigger, and the learned RUL is poor which is why we use a physics trend
instead." Owning the weak numbers *builds* credibility.

---

## 9. Deployment roadmap

| Stage | What changes from the demo |
|---|---|
| **Bench / HIL rig** | Replace `ScenarioRunner` with a `socketcan` reader on the real ECU/FADEC bus. `core/can_telemetry.py` already decodes — only the signal map (`FRAME_MAP`) needs to match the engine's real DBC. |
| **Onboard edge** | `core/` + `ml/` run unchanged on a companion computer (NVIDIA Jetson / small x86). Export the XGBoost models to **ONNX** for portable, fast inference; move TreeSHAP to a fixed per-class background for constant-time explanation. The edge publishes *enriched* frames northbound. |
| **GCS / fleet** | Re-enable the `legacy/` **MQTT** path as the air-to-ground transport: the edge publishes `uav/engine/enriched`, the ground dashboard subscribes. `uav_id` / `session_id` are already in the data model for multi-aircraft. |
| **Fleet analytics** | Persist enriched frames (the legacy SQLite schema is a starting point) → cross-mission RUL calibration, per-tail-number degradation baselines, and **federated learning** (each aircraft contributes model updates without shipping raw telemetry). |
| **Certification** | The physics model and every threshold are in one auditable YAML; the classifier is tree-based and exactly SHAP-explainable; no black-box end-to-end network sits on the safety path. This is a defensible basis for airworthiness review. |

---

## 10. Jury Q&A bank

### Conceptual

**Q: What exactly is the "digital twin" here — the simulator or the analytics?**
The twin is the *synchronised virtual model plus its analytics*. Concretely:
`core/physics_model.py` is the virtual model, it's kept in step with the physical
engine by the CAN telemetry stream, and the `ml/` layer turns the model↔reality
difference into health, faults, RUL and advice. The scenario simulator stands in
for the physical engine during the demo.

**Q: How is this different from a normal threshold alarm?**
A threshold alarm fires when a raw value crosses a line — always *after* the
problem. We compare against a *physics prediction* (so we catch a deviation long
before it reaches a redline), we track the *rate* of deviation (so we can predict
time-to-failure), and we *classify* the fault and *explain* it (so the operator
knows what to do, not just that something's wrong).

**Q: Why should I trust the fault call?**
Three independent things must agree: a supervised classifier at 95 % macro-F1, a
physics residual actually outside tolerance, and persistence over time. Plus SHAP
shows you *which* sensor deviations drove the call, and the validation footer
shows the simulator's ground truth next to our verdict.

### Physics / domain

**Q: Walk me through your physics model.**
See [5.2](#52-corephysics_modelpy). Key beats: barometric atmosphere → turbo
model with a 5 km critical altitude → engine load = (MAP/MAPmax)·(RPM/RPMmax) →
two-segment temperature interpolation anchored at idle/cruise/redline spec values
→ CHT corrected for ambient and ram-air cooling → oil/fuel/vibration as monotonic
functions of load. Then residual = measured − expected.

**Q: Why turbocharged? Why does critical altitude matter?**
A naturally-aspirated engine loses power roughly with air density and would be
useless above 3–4 km. A turbo compresses the intake air to hold sea-level
manifold pressure up to its *critical altitude* (~5 km here); above that, boost
and power fall off. MALE UAVs cruise at 3–9 km, so turbocharging is mandatory,
and the model has to represent the falloff.

**Q: How do you detect a misfire specifically?**
Per-cylinder EGT. A dead/weak cylinder exhausts cooler gas, so its EGT drops well
below the other three. When `max−min` across the four exceeds ~90 °C, combined
with a vibration rise, that's a misfire. A single averaged EGT can't see it —
that's why the CAN frame set carries all four cylinders.

**Q: A sensor could just be faulty. Do you handle that?**
Yes — `sensor_drift_cht`. If CHT rises but EGT, oil temperature and everything
else stay normal, that's *inconsistent with real overheating*, so the classifier
reports sensor drift, not an engine fault. It's the subtlest of the five classes
and it's in the training set.

### ML

**Q: How big is your training set and where's it from?**
140 randomised 3-minute flights (~50 000 rows at 2 Hz), generated by
`ml/dataset.py` from the *same physics the demo runs* but with randomised
operating point, fault type, onset, severity and seed. The four demo scenarios
use different seeds, so they're always out-of-sample.

**Q: How do you avoid data leakage / overfitting?**
Split **by flight**, not by row — whole flights go to train or test, never both.
No ground-truth label is ever a feature. Held-out macro-F1 is 0.95.

**Q: Your anomaly detector's AUC is only 0.77.**
Correct, and that's the *unsupervised* layer in isolation. Unsupervised anomaly
detection struggles when healthy transients (a hard climb) and real faults sit
near each other in feature space. That's exactly why it isn't the trigger — the
supervised classifier plus the physics gate plus persistence is. The forest is a
corroborating check. Net: zero false alarms on nominal, all faults caught.

**Q: Your RUL regressor has negative R². Isn't that broken?**
It can't be trained away: after a fault onset, the time to failure depends on the
severity of the initiating fault, which isn't observable in a single snapshot —
by design, for flight-to-flight variety. So we don't use it. The headline RUL is
a physics trend extrapolation (`RulEstimator`): fit the recent rate of the
limiting channel, divide the remaining margin by that rate, gate it on the
residual sign. Accurate, explainable, adaptive. The regressor is a logged
cross-check only.

**Q: Why not deep learning / an LSTM / a transformer?**
36 tabular features, 50 k rows: gradient-boosted trees are the state of the art
for this shape of problem and would beat a neural net without far more data. We
also need **exact** millisecond explanations (TreeSHAP) and trivial edge
deployment (ONNX / quantised trees). An LSTM would add latency, opacity and data
hunger for no accuracy gain here. If we later fuse raw vibration *spectra* (not
just RMS), a small 1-D CNN on that channel would be worth it — that's future work.

**Q: How does SHAP work and why is it fast enough?**
SHAP assigns each feature its game-theoretic fair share of
`(prediction − baseline)`; the shares sum exactly to that difference. Naive SHAP
is exponential; **TreeSHAP** computes it exactly for tree ensembles in polynomial
time by walking the trees. We build one explainer at startup, prime it on 50
cached healthy vectors, and only call it on anomaly frames — p50 ~18 ms.

**Q: What are your 36 features?**
Per sensor (8 of them): physics residual, 60-s rolling mean, 60-s rolling std,
10-s rate. Plus throttle, altitude, ambient, and time-spent-in-anomaly. Residual
= the deviation, std = roughness, rate = trend speed. No labels as features.

### Software / systems

**Q: Is the CAN bus real or a mock?**
It's `python-can` with the `virtual` backend — a real in-process CAN bus with the
real API and real byte framing. Every telemetry sample is packed into 5
8-byte frames (IDs 0x100–0x104), sent, received, and decoded — the twin never
sees data that didn't survive the wire format. Going to hardware is one line:
`interface='virtual'` → `interface='socketcan', channel='can0'`. Run
`python core/can_telemetry.py <scenario> --dump` to see the hex.

**Q: What's your latency budget and are you within it?**
At 2 Hz the budget is 500 ms/frame. Twin inference is ~40–55 ms, SHAP ~20 ms
(anomaly frames only). ~10× headroom — comfortably real-time and edge-deployable.

**Q: What happens if a model file is missing, or CAN isn't installed, or SHAP fails?**
Every stage has a fallback: synthetic anomaly model, rule-based classifier,
byte-compatible loopback bus, residual-ranking "SHAP." The pipeline never raises.
`./run_demo.sh --check` verifies the happy path end to end.

**Q: How is the demo reproducible?**
Fixed seeds per scenario → byte-identical telemetry every run. `--check` is a
deterministic pass/fail over all four scenarios.

### Application / operations

**Q: Who uses this and how?**
Three audiences (as in the brief): the **UAV operator** watches the health index,
RUL and the advisory console during a mission; the **propulsion engineer** uses
the residual radar and SHAP to understand *why*; **maintenance** uses the
post-mission replay and the advisory log to plan work. In the field it runs at
the GCS, fed by a downlinked CAN-over-radio telemetry stream.

**Q: What decision does it actually drive?**
Continue / throttle-back / divert / abort. The advisory console turns the fault +
RUL + health into one of those, with numbers ("throttle to 60 %", "RTL now, RUL
< 12 min").

**Q: What's the false-alarm cost and how do you manage it?**
A false abort wastes a sortie. We manage it with the three-way AND (classifier +
physics gate + persistence) and by tuning the residual tolerances so nominal sits
at 97–99 % health. Verified: zero alarms on the nominal scenario.

**Q: Could this run onboard, not just on the ground?**
Yes — `core/` + `ml/` are ~a few hundred KB of models and pure-Python/NumPy. On a
Jetson-class board with ONNX models it's well under real-time. Onboard analytics +
a compressed health summary downlink is the "Edge AI for UAV" innovation the brief
asks for.

### "Gotcha" questions

**Q: You trained and you test on the same simulator — isn't that circular?**
Two defences. (1) The training flights and the demo scenarios use different seeds,
altitudes, fault types and severities — the demo is genuinely out-of-sample, and
we split by flight. (2) The twin's physics baseline is an *independent*
re-implementation of the expected-value model, not the simulator's own code — so
the residual isn't a model checking itself. The honest limitation is that both
share a physics *worldview*; validating against real engine data is the next step,
and the architecture is built for that swap.

**Q: The health index dips at the start even on the nominal run — bug?**
That's the throttle slam from taxi (15 %) to climb (88 %) at t≈15 s. The physics
baseline reacts instantly; the simulated engine's temperatures lag. The residual
spikes for a few seconds, the health index dips, and then they re-converge — and
critically **no anomaly fires**, because the persistence + classifier checks
reject a short transient. It's a visible demonstration that the twin is sensitive
but not trigger-happy.

**Q: Your RUL number jumps around.**
It's a live trend extrapolation: `RUL = margin / rate`, and both the margin (noisy
sensor near a threshold) and the rate (least-squares slope over 45 s) fluctuate
each frame, so near the abort point small noise = big percentage swings, and the
residual guard can briefly flip it to "—". That's how raw trend prognostics
behave; production systems smooth it with a Kalman filter. We can add a 5–10 s
smoother if a steadier readout is preferred — kept it raw so the mechanism is
visible.

**Q: Why five fault classes and not more?**
They cover the brief's list (misfire, cooling degradation, lubrication issues,
sensor drift) and the highest-severity real failure modes. The framework is
class-agnostic — adding detonation, fuel-pump decay, ignition-module failure, etc.
is a matter of adding scenarios to `scenario_runner.py` and retraining; no
architecture change.

**Q: What breaks this system?**
(1) A physics-model error introduces a systematic residual bias — mitigated by the
auditable spec file and by calibrating against real data. (2) A fault mode not in
the training set would be caught by the *anomaly* layer as "unknown deviation" but
mis-labelled by the classifier — the SHAP panel would still show the operator the
real driver. (3) Simultaneous multi-fault conditions aren't explicitly modelled
yet. All three are named in [Section 12](#12-known-limitations-and-honest-future-work).

---

## 11. 4-minute demo script

> Rehearse this. Timings assume 5× speed.

**0:00 — Frame it (20 s).**
"MALE UAV, single piston engine, 30-hour missions. If that engine fails, the
mission and often the aircraft are lost. Today's monitors are threshold alarms —
they warn you after the redline. This is a digital twin that predicts the failure
and tells the operator what to do. One process, one command, live telemetry over
a virtual CAN bus."

**0:20 — Nominal (40 s).** Select *Nominal Loiter*, Play at 5×.
"Left: the mission lifecycle — taxi, climb, cruise, loiter. The green line is the
health index, computed by comparing every sensor against a first-principles
physics model. It stays at 98 %. Right: the twin-cognition radar — every bar is
`|sensor − physics| ÷ tolerance`; all green, all inside the ring. No false
alarms. This is the control."

**1:00 — Inject lubrication failure (90 s).** Switch to *Lubrication Loss*, hit
**Inject**.
"I've injected an oil-pump failure at loiter. Watch the oil-pressure track — the
orange sensor line is peeling away from the dotted physics baseline; the red gap
is the residual. Top strip: anomaly confidence just went to 100 %, classified as
*lubrication degradation*. And the RUL — remaining useful life — is counting down:
'ABORT in 40 seconds', driver 'oil pressure to abort limit'. The advisory console:
'execute Return-To-Launch now.' Bottom right, live SHAP: the call is 60 % driven
by the oil-pressure residual — the twin is telling the engineer *why*, in 18
milliseconds, every frame."

**2:30 — Misfire (60 s).** Switch to *Injector Misfire*, Play at 5×.
"Different failure. The per-cylinder EGT chart — three cylinders bunched, cylinder
3 diverging by 150 °C. That spread plus the vibration rise is an unambiguous
single-cylinder misfire. SHAP leads with the vibration-roughness feature. Advisory:
precautionary landing."

**3:30 — Close (30 s).**
"Physics-informed: the ML works on residuals from an auditable thermodynamic
model, so it needs 140 training flights, not millions of hours, and it
generalises. Classifier macro-F1 0.95 on unseen flights. Everything's tree-based
and SHAP-explainable — nothing on the safety path is a black box. It runs at
40 milliseconds a frame, so it deploys to an onboard edge computer. The
distributed MQTT ground-station path is already in the repo. Thank you."

---

## 12. Known limitations and honest future work

A jury respects a team that knows the edges of its own system.

| Limitation | Why it exists | Path forward |
|---|---|---|
| **Validated on simulated data only** | No public instrumented MALE-engine dataset | Bench/HIL run against a real ECU; recalibrate the spec-file tolerances from real healthy data |
| **Twin physics and simulator share a worldview** | Same modelling assumptions on both sides | The physics baseline is already an independent implementation; real-data validation closes the loop |
| **Compressed 180-s mission, warm-start** | Can't demo 30 h; steady-state baseline vs a warming engine | For an onboard build the baseline becomes stateful (its own thermal lag) so full cold-start is handled |
| **RUL readout is jittery near the threshold** | Raw `margin / rate` with a noisy margin | Kalman-filter the slope estimate; report a confidence band, not a point |
| **Unsupervised anomaly layer is weak alone (AUC 0.77)** | Faults and transients overlap in feature space | Better residual features (vibration *spectra*, not just RMS); a small autoencoder on the residual vector |
| **Single-fault scenarios only** | Simpler to model and label | Add compound-fault scenarios; move from single-label to multi-label classification |
| **Learned RUL regressor is not usable (R² < 0)** | Time-to-failure isn't in a snapshot | Sequence model (small temporal CNN/GRU) over the residual history, trained with survival-analysis loss |
| **5 fault classes** | Covers the brief + top severities | Framework is class-agnostic — add scenarios + retrain |
| **Streamlit re-runs the whole script** | Fastest path to a Python dashboard | Production GCS uses an event-driven frontend over the `legacy/` MQTT feed |
| **Fixed CAN signal map** | Hand-coded for the demo | Load a real DBC file at startup |

---

*End of explainer. Keep `DESIGN_NOTES.md` for the short version, this for depth,
and `./run_demo.sh --check` to prove it runs.*
