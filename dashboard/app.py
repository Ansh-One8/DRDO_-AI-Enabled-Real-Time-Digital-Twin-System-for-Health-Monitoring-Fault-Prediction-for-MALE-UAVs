"""
dashboard/app.py
================
MALE UAV Aero-Piston Engine -- Digital Twin GCS console.

Single-process: this app owns the whole pipeline
(scenario -> virtual CAN -> physics twin -> ML -> SHAP -> advisory) and
steps it inside the Streamlit run loop. No broker, no external services.

    streamlit run dashboard/app.py

Header controls: scenario hot-swap, Play / Pause, speed (1x / 2x / 5x),
Inject Fault, Reset. Body: metric strip, mission-lifecycle timeline,
twin-cognition residual radar, measured-vs-physics track, live SHAP
root-cause bar, per-cylinder EGT, and a rolling operator advisory console.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.scenario_runner import SCENARIOS, MISSION_DURATION_S
from core.twin_pipeline import TwinPipeline
from dashboard import components as C

st.set_page_config(page_title="UAV Engine Digital Twin — GCS", layout="wide",
                   initial_sidebar_state="collapsed")

_CSS = """
<style>
  .stApp { background:#0a0e14; }
  section.main > div { padding-top: 0.6rem; }
  h1,h2,h3,h4 { font-family: ui-monospace, Menlo, monospace !important;
                letter-spacing:.04em; color:#c8d3e0; }
  .twin-band { border:1px solid #1e2733; background:#111720; border-radius:8px;
               padding:10px 14px; margin-bottom:6px; }
  .kpi { border:1px solid #1e2733; background:#111720; border-radius:8px;
         padding:8px 12px; height:100%; }
  .kpi .lab { font:600 10px ui-monospace,monospace; color:#7f8ea3;
              letter-spacing:.14em; text-transform:uppercase; }
  .kpi .val { font:700 22px ui-monospace,monospace; color:#e6edf5; margin-top:2px; }
  .kpi .sub { font:500 10px ui-monospace,monospace; color:#7f8ea3; margin-top:1px; }
  .sev-critical { color:#ff4d4d !important; }
  .sev-warning  { color:#ffb020 !important; }
  .sev-caution  { color:#ffd23f !important; }
  .sev-nominal  { color:#3ddc84 !important; }
  .adv-row { font:500 12px ui-monospace,monospace; padding:5px 8px;
             border-left:3px solid #37c0e0; margin:3px 0; background:#0d131c; }
  .adv-critical { border-left-color:#ff4d4d; color:#ffd7d7; }
  .adv-warning  { border-left-color:#ffb020; color:#ffe9c7; }
  .adv-caution  { border-left-color:#ffd23f; color:#fff3cf; }
  .adv-nominal  { border-left-color:#3ddc84; color:#c9f5dd; }
  [data-testid="stMetricValue"] { font-family: ui-monospace, monospace; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline (heavy: models + SHAP). One per server process.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading twin models + SHAP explainer…")
def get_pipeline() -> TwinPipeline:
    return TwinPipeline(scenario_key="nominal_loiter", hz=2.0, verbose=False)


pipe = get_pipeline()

ss = st.session_state
ss.setdefault("playing", False)
ss.setdefault("speed", 2)
ss.setdefault("scenario", "nominal_loiter")
ss.setdefault("advisories", [])


def _switch_scenario(key: str):
    pipe.reset(key)
    ss.scenario = key
    ss.playing = False
    ss.advisories = []


def _inject_fault():
    """Demo helper: if on the nominal scenario, hot-swap to the lubrication
    failure; otherwise jump the clock to just before this scenario's fault
    onset so the degradation begins immediately."""
    spec = SCENARIOS[ss.scenario]
    if not spec.faults:
        _switch_scenario("lubrication_loss")
        ss.playing = True
        return
    onset = min(f.onset_s for f in spec.faults)
    target = max(0.0, onset - 6.0)
    # replay from t=0 up to target so history/rolling features are populated
    pipe.reset(ss.scenario)
    while pipe._runner._t < target:
        if pipe.step() is None:
            break
    ss.playing = True


# ---------------------------------------------------------------------------
# Header / controls
# ---------------------------------------------------------------------------
st.markdown("## ▮ MALE-UAV AERO-PISTON ENGINE · DIGITAL TWIN — GROUND CONTROL")
keys = list(SCENARIOS)
titles = [SCENARIOS[k].title for k in keys]

c = st.columns([3.2, 0.9, 0.9, 1.5, 0.9, 1.0])
with c[0]:
    sel = st.selectbox("SCENARIO", titles, index=keys.index(ss.scenario),
                       label_visibility="collapsed")
    if keys[titles.index(sel)] != ss.scenario:
        _switch_scenario(keys[titles.index(sel)])
        st.rerun()
with c[1]:
    if st.button("▶ PLAY" if not ss.playing else "▶ PLAYING", use_container_width=True,
                 disabled=ss.playing or pipe.done):
        ss.playing = True
        st.rerun()
with c[2]:
    if st.button("⏸ PAUSE", use_container_width=True, disabled=not ss.playing):
        ss.playing = False
        st.rerun()
with c[3]:
    ss.speed = st.radio("SPEED", [1, 2, 5], index=[1, 2, 5].index(ss.speed),
                        horizontal=True, format_func=lambda x: f"{x}×",
                        label_visibility="collapsed")
with c[4]:
    if st.button("⚠ INJECT", use_container_width=True):
        _inject_fault()
        st.rerun()
with c[5]:
    if st.button("⟲ RESET", use_container_width=True):
        _switch_scenario(ss.scenario)
        st.rerun()

st.caption(f"**{pipe.scenario_title}** — {pipe.scenario_blurb}")

# ---------------------------------------------------------------------------
# Advance the sim (synchronous, inside the run loop)
# ---------------------------------------------------------------------------
if ss.playing and not pipe.done:
    for _ in range(int(ss.speed)):
        f = pipe.step()
        if f is None:
            ss.playing = False
            break
        adv = f["advisory"]
        stamp = time.strftime("%H:%M:%S")
        if not ss.advisories or ss.advisories[-1][2] != adv["text"]:
            ss.advisories.append((stamp, adv["urgency"], adv["text"], f["t_s"]))
            ss.advisories = ss.advisories[-14:]

buf = list(pipe.buffer)
df = pd.DataFrame(buf) if buf else pd.DataFrame()
latest = buf[-1] if buf else None

# ---------------------------------------------------------------------------
# Metric strip
# ---------------------------------------------------------------------------
def kpi(col, lab, val, sub="", sev="nominal"):
    col.markdown(
        f'<div class="kpi"><div class="lab">{lab}</div>'
        f'<div class="val sev-{sev}">{val}</div>'
        f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


m = st.columns(6)
if latest:
    hi = latest["health_index"]
    hsev = "nominal" if hi >= 80 else "caution" if hi >= 60 else "warning" if hi >= 45 else "critical"
    kpi(m[0], "Operational Health", f"{hi:0.0f}%", "physics residual index", hsev)

    rtxt, rsev = C.fmt_rul(latest["rul_seconds"])
    drv = latest["rul_driver"] or "no degrading trend"
    kpi(m[1], "Remaining Useful Life", rtxt, drv, rsev)

    if latest["anomaly"]:
        kpi(m[2], "Anomaly", f"{latest['fault_confidence']*100:0.0f}%",
            latest["fault"].replace("_", " "), "critical")
    else:
        kpi(m[2], "Anomaly", "CLEAR", f"cls: {latest['fault'].replace('_',' ')}", "nominal")

    cs = latest["can_status"]
    kpi(m[3], "CAN / FADEC Bus", f"{latest['can_fps']:0.0f} fps",
        f"{cs['id_range']} · {cs['frames_recv']} frm", "nominal")

    kpi(m[4], "Inference Latency", f"{latest['infer_latency_ms']:0.0f} ms",
        "twin + SHAP per frame", "nominal" if latest["infer_latency_ms"] < 90 else "caution")

    stage = latest["mission_stage"].replace("_", " ")
    kpi(m[5], "Mission Stage", stage, f"t+{latest['t_s']:0.0f}s / {MISSION_DURATION_S:0.0f}s",
        "nominal")
else:
    for i, lab in enumerate(["Operational Health", "Remaining Useful Life", "Anomaly",
                             "CAN / FADEC Bus", "Inference Latency", "Mission Stage"]):
        kpi(m[i], lab, "—", "press ▶ PLAY", "nominal")

st.progress(min(1.0, (latest["progress"] if latest else 0.0)))

# ---------------------------------------------------------------------------
# Primary visuals
# ---------------------------------------------------------------------------
r1 = st.columns([1.55, 1.0])
r1[0].plotly_chart(C.mission_lifecycle_fig(df, MISSION_DURATION_S),
                   use_container_width=True, config={"displayModeBar": False})
r1[1].plotly_chart(
    C.twin_cognition_fig(latest["norm_residuals"] if latest else {}),
    use_container_width=True, config={"displayModeBar": False})

r2 = st.columns([1.0, 1.0])
r2[0].plotly_chart(C.residual_track_fig(df), use_container_width=True,
                   config={"displayModeBar": False})
r2[1].plotly_chart(C.shap_bar_fig(latest["explanation"] if latest else None),
                   use_container_width=True, config={"displayModeBar": False})

r3 = st.columns([1.0, 1.0])
r3[0].plotly_chart(C.per_cylinder_egt_fig(df), use_container_width=True,
                   config={"displayModeBar": False})

with r3[1]:
    st.markdown("#### ▸ OPERATOR ADVISORY CONSOLE")
    if latest:
        a = latest["advisory"]
        st.markdown(
            f'<div class="adv-row adv-{a["urgency"]}">[{a["urgency"].upper()}] {a["text"]}</div>',
            unsafe_allow_html=True)
    for stamp, urg, text, t_s in reversed(ss.advisories[:-1]):
        st.markdown(
            f'<div class="adv-row adv-{urg}">[{stamp} · t+{t_s:0.0f}s] {text}</div>',
            unsafe_allow_html=True)
    if not ss.advisories:
        st.caption("no advisories — start a scenario")

# ---------------------------------------------------------------------------
# Validation footer (ground truth is for the jury, not the operator)
# ---------------------------------------------------------------------------
if latest:
    gt = latest["gt_fault"]
    gtr = latest["gt_rul"]
    ml_rul = latest["ml_rul_seconds"]
    ml_txt = (f"{ml_rul:0.0f}s" if ml_rul is not None else "n/a")
    st.caption(
        f"validation overlay · sim ground-truth fault = **{gt}** · "
        f"sim RUL = {gtr if gtr is not None else '—'}s · "
        f"twin classifier = **{latest['fault']}** ({latest['fault_confidence']*100:0.0f}%) · "
        f"physics-RUL headline vs learned-regressor cross-check = {ml_txt} · "
        f"SHAP: {pipe.explainer.status}")

# ---------------------------------------------------------------------------
# Drive the loop
# ---------------------------------------------------------------------------
if ss.playing and not pipe.done:
    time.sleep(0.12)
    st.rerun()
elif pipe.done and ss.playing:
    ss.playing = False
    st.rerun()
