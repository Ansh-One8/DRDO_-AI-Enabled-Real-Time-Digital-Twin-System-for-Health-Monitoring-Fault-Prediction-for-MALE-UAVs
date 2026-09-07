# AI-Enabled Real-Time Digital Twin — MALE UAV Aero-Piston Engine

**Smart India Hackathon · Problem Statement ID 26054**

A single-process, real-time **digital twin** of a Rotax-914-class piston engine
for a MALE (Medium Altitude Long Endurance) UAV. It mirrors the engine from live
CAN/FADEC telemetry, compares every sensor against a first-principles
thermodynamic model, and turns the divergence into a fault classification, a
remaining-useful-life (RUL) estimate, a SHAP root-cause, and a concrete crew
advisory — end to end, on unseen mission data, in ~40 ms per frame.

```
core/scenario_runner.py   deterministic flight generator (unseen data)
        │  RPM, CHT×4, EGT×4, oil P/T, fuel, MAP, vibration, altitude, airspeed …
        ▼
core/can_telemetry.py     python-can virtual bus  (IDs 0x100–0x104)
        │  raw 8-byte CAN frames  →  decoded engineering units
        ▼
core/physics_model.py     steady-state thermodynamic baseline  X_physics(t)
        │  residual vector  R(t) = X_measured(t) − X_physics(t)
        ▼
ml/models.py              IsolationForest + XGBoost fault classifier
        │                 + physics-trend RUL estimator
        ▼
ml/explainability.py      cached TreeSHAP  →  top-3 signed drivers  (<25 ms)
        ▼
core/twin_pipeline.py     operator-advisory rules  →  enriched frame
        ▼
dashboard/app.py          Streamlit Ground Control Station console
```

## Quick start

```bash
python -m venv uav_env && . uav_env/Scripts/activate   # or uav_env/bin/activate on Linux/Mac
pip install -r requirements.txt
python ml/train.py --flights 140       # trains the 3 models into models/  (~20 s)
streamlit run dashboard/app.py         # dashboard on http://localhost:8501
```

Or, with the bundled launcher (Git Bash / Linux / Mac):

```bash
./run_demo.sh            # launch the dashboard
./run_demo.sh --check    # end-to-end self-test of all four scenarios
./run_demo.sh --headless lubrication_loss   # run the pipeline in the terminal
```

In the dashboard: pick a scenario, press **▶ PLAY**, or hit **⚠ INJECT** to jump
straight into the lubrication failure. Use **5×** speed for a full run in ~35 s.

## Scenarios

| Key | What it demonstrates |
|---|---|
| `nominal_loiter` | Baseline — health stays ~98 %, zero false alarms |
| `altitude_cooling_degradation` | CHT drifts above physics baseline; classified as cooling degradation |
| `lubrication_loss` | Oil pressure collapses; **RUL counts down to ABORT**; RTL advisory |
| `injector_misfire` | One cylinder's EGT diverges >90 °C; misfire classified from vibration + spread |

## What's here

| Path | Role |
|---|---|
| `config/engine_specs.yaml` | Single source of truth: Rotax-914 nominal values, redlines, tolerances |
| `core/physics_model.py` | First-principles expected-value model + residuals + subsystem health |
| `core/scenario_runner.py` | Four deterministic 3-minute mission scenarios + randomized-flight factory |
| `core/can_telemetry.py` | Virtual CAN/FADEC bus — pack telemetry → frames `0x100`–`0x104`, decode back |
| `core/twin_pipeline.py` | Wires the pipeline; also runs headless |
| `ml/dataset.py`, `ml/train.py` | Training-set generation + model training → `models/` |
| `ml/models.py` | Feature builder, model loaders (+ fallback), hybrid physics-trend RUL |
| `ml/explainability.py` | `LiveExplainer` — cached real-time TreeSHAP |
| `dashboard/` | Streamlit GCS console + Plotly figure builders |
| `legacy/` | Original 4-process MQTT pipeline — the documented fleet-deployment path |

## Documentation

- **[DESIGN_NOTES.md](DESIGN_NOTES.md)** — architecture, every deviation from the brief and why, deployment roadmap.
- **[docs/EXPLAINER.md](docs/EXPLAINER.md)** — full teaching guide: glossary, domain primer, module deep-dives, metrics, and a jury Q&A bank. Also available as `docs/EXPLAINER.docx` / `.html`.

## Model performance

Held-out (split by flight), from `models/metrics.json`:

| Metric | Value |
|---|---|
| Fault classifier — macro-F1 | **0.95** |
| Anomaly detector — ROC-AUC | 0.77 (corroborating signal; the classifier + physics gate is the detector) |
| RUL regressor — R² | −1.7 → **not used as the headline**; RUL comes from physics-trend extrapolation |
| End-to-end scenario check | 4 / 4 PASS |

## Requirements

Python 3.11–3.14. Key deps (pinned in `requirements.txt`): `numpy`, `pandas`,
`scikit-learn`, `xgboost`, `shap`, `python-can`, `streamlit`, `plotly`, `pyyaml`.
