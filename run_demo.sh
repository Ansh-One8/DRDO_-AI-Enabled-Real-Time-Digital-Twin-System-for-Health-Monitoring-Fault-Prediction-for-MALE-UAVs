#!/usr/bin/env bash
# =============================================================================
# MALE UAV Aero-Piston Engine Digital Twin -- one-command demo launcher
# =============================================================================
# Brings up the full single-process pipeline and the GCS dashboard:
#
#   scenario runner -> virtual CAN (python-can) -> physics-informed twin
#                   -> ML (anomaly / fault / physics-RUL) -> live TreeSHAP
#                   -> operator advisory  ->  Streamlit console
#
# No MQTT broker, no external services. Ctrl-C to stop.
#
# Usage:
#   ./run_demo.sh                 # launch the dashboard (default)
#   ./run_demo.sh --headless      # run the pipeline in the terminal, no UI
#   ./run_demo.sh --train         # regenerate the training set + retrain models
#   ./run_demo.sh --check         # self-test all four scenarios end to end
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- pick a Python interpreter -----------------------------------------------
PY=""
for c in "./uav_env/Scripts/python.exe" "./uav_env/bin/python" "./.venv/Scripts/python.exe" \
         "./.venv/bin/python" "python3" "python"; do
  if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "No Python interpreter found."; exit 1; }
echo "[run_demo] python: $PY"

# --- dependencies -----------------------------------------------------------
if ! "$PY" -c "import streamlit, can, shap, xgboost, plotly, yaml" >/dev/null 2>&1; then
  echo "[run_demo] installing requirements..."
  "$PY" -m pip install -r requirements.txt
fi

# --- models: train on first run if artifacts are absent --------------------
if [ ! -f "models/fault_classifier.joblib" ] || [ "${1:-}" = "--train" ]; then
  echo "[run_demo] training models (randomized flights from the scenario physics)..."
  "$PY" ml/train.py --flights 140
  [ "${1:-}" = "--train" ] && exit 0
fi

MODE="${1:-dashboard}"
case "$MODE" in
  --headless)
    exec "$PY" core/twin_pipeline.py "${2:-lubrication_loss}" --speed "${3:-6}"
    ;;
  --check)
    exec "$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from core.twin_pipeline import TwinPipeline
from core.scenario_runner import SCENARIOS
ok = True
for key in SCENARIOS:
    p = TwinPipeline(key, verbose=False)
    last = None
    while True:
        f = p.step()
        if f is None: break
        last = f
    p.shutdown()
    hit = last["fault"] if last["anomaly"] else "healthy"
    gt = last["gt_fault"]
    good = (gt == "healthy" and not last["anomaly"]) or (gt != "healthy" and last["anomaly"])
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {key:32s} gt={gt:24s} twin={hit:24s} "
          f"H={last['health_index']:.0f} RUL={last['rul_seconds']}")
sys.exit(0 if ok else 1)
PYEOF
    ;;
  dashboard|"")
    echo "[run_demo] starting GCS dashboard on http://localhost:8501 ..."
    exec "$PY" -m streamlit run dashboard/app.py \
      --server.headless true --server.port 8501 --browser.gatherUsageStats false
    ;;
  *)
    echo "unknown option: $MODE"; exit 2 ;;
esac
