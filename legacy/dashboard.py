"""
UAV Digital Twin -- Ground Control Station Dashboard (v2)
============================================================
New in this version:
  - Mission Control tab: start/stop the broker, publisher, and twin core
    from buttons in the browser instead of separate terminals.
  - A live 3D visualization of the airframe, color-coded per subsystem
    health (this is what makes it an actual "digital twin" visually,
    not just a chart of numbers).
  - Clearly labeled, titled charts (Plotly) instead of bare line charts.
  - A persistent alerts log (from alerts_log table) with severity + the
    SHAP-based "why" + the recommended maintenance action.
  - Warm-up awareness: matches the twin core's warm-up grace period, so
    the UI doesn't show a scary anomaly banner while the engine is still
    cold (which is physically normal, not a fault).

Still queued for a later pass: multi-UAV fleet overview screen, PDF
export, cross-mission benchmarking. The plumbing for fleet support
(uav_id, session_id) is already in the data model so that's addable
without another schema change.

Run:
    streamlit run dashboard.py
"""

import json
import sqlite3
import subprocess
import sys
import socket
import time
from collections import deque

import pandas as pd
import paho.mqtt.client as mqtt
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_ENRICHED_TOPIC = "uav/engine/enriched"
DEFAULT_TELEMETRY_TOPIC = "uav/engine/telemetry"
DEFAULT_DB = "digital_twin_history.db"
BUFFER_LEN = 300
WARMUP_GRACE_S = 400.0

st.set_page_config(page_title="UAV Digital Twin GCS", layout="wide")


# ---------------------------------------------------------------------------
# Live data buffer (MQTT)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_buffer():
    return deque(maxlen=BUFFER_LEN)


@st.cache_resource
def get_mqtt_client():
    """Created exactly once per server process (that's what cache_resource
    is for), but connecting is handled separately in ensure_connected() so
    a broker that wasn't up yet on the first run gets retried on every
    subsequent rerun instead of being cached as a permanent failure."""
    buf = get_buffer()

    def on_message(client, userdata, msg):
        try:
            buf.append(json.loads(msg.payload.decode()))
        except Exception:
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="gcs-dashboard")
    client.on_message = on_message
    client._loop_started = False
    return client


def ensure_mqtt_connected(broker_host: str, broker_port: int, topic: str):
    client = get_mqtt_client()
    if not client.is_connected():
        try:
            client.connect(broker_host, broker_port)
            client.subscribe(topic)
        except Exception:
            return client  # broker still not reachable -- try again next rerun
    if not client._loop_started:
        try:
            client.loop_start()
            client._loop_started = True
        except Exception:
            pass
    return client


def port_is_open(host: str, port: int, timeout=0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Mission Control -- start/stop backend processes as subprocesses so the
# person never has to open a separate terminal for the broker/publisher/
# twin core. Handles live only as long as this Streamlit server process
# lives; that's an acceptable limitation for a demo/prototype, not
# something to rely on for a real deployment.
# ---------------------------------------------------------------------------
def _proc_running(key: str) -> bool:
    proc = st.session_state.get(key)
    return proc is not None and proc.poll() is None


def _start_proc(key: str, cmd: list):
    if _proc_running(key):
        return
    st.session_state[key] = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


def _stop_proc(key: str):
    proc = st.session_state.get(key)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    st.session_state[key] = None


def render_control_tab(broker_host: str, broker_port: int, topic: str):
    st.markdown("### Mission Control")
    st.caption("Start/stop the simulated engine and the digital twin core without touching a terminal. "
               "The MQTT broker still needs to be reachable -- start it here too if it isn't already running "
               "elsewhere.")

    broker_up = port_is_open(broker_host, broker_port)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**1. MQTT Broker**")
        st.write("🟢 Running" if broker_up else "🔴 Not reachable")
        bc1, bc2 = st.columns(2)
        if bc1.button("Start broker", disabled=broker_up):
            _start_proc("proc_broker", [sys.executable, "-m", "amqtt.scripts.broker_script"])
            time.sleep(1.0)
            st.rerun()
        if bc2.button("Stop broker", disabled=not _proc_running("proc_broker")):
            _stop_proc("proc_broker")
            st.rerun()

    with c2:
        st.markdown("**2. Simulated Engine**")
        st.write("🟢 Running" if _proc_running("proc_publisher") else "⚪ Stopped")
        fault = st.selectbox("Inject fault", ["none", "misfire", "cooling_degradation",
                                               "lubrication_degradation", "sensor_drift_cht"])
        uav_id = st.text_input("UAV ID", "UAV-01")
        speed = st.slider("Playback speed", 1.0, 100.0, 20.0)
        pc1, pc2 = st.columns(2)
        if pc1.button("Start engine sim", disabled=_proc_running("proc_publisher") or not broker_up):
            cmd = [sys.executable, "Mqtt_publisher.py", "--hz", "2", "--speed", str(speed),
                   "--uav-id", uav_id, "--broker-host", broker_host, "--broker-port", str(broker_port),
                   "--topic", topic]
            if fault != "none":
                cmd += ["--fault", fault]
            _start_proc("proc_publisher", cmd)
            time.sleep(0.5)
            st.rerun()
        if pc2.button("Stop engine sim", disabled=not _proc_running("proc_publisher")):
            _stop_proc("proc_publisher")
            st.rerun()

    with c3:
        st.markdown("**3. Digital Twin Core**")
        st.write("🟢 Running" if _proc_running("proc_twin") else "⚪ Stopped")
        tc1, tc2 = st.columns(2)
        if tc1.button("Start twin core", disabled=_proc_running("proc_twin") or not broker_up):
            _start_proc("proc_twin", [sys.executable, "digital_twin_core.py",
                                       "--broker-host", broker_host, "--broker-port", str(broker_port),
                                       "--topic", topic])
            time.sleep(0.5)
            st.rerun()
        if tc2.button("Stop twin core", disabled=not _proc_running("proc_twin")):
            _stop_proc("proc_twin")
            st.rerun()

    st.divider()
    if st.button("🛑 Stop everything"):
        _stop_proc("proc_publisher")
        _stop_proc("proc_twin")
        _stop_proc("proc_broker")
        st.rerun()

    if not (broker_up and _proc_running("proc_publisher") and _proc_running("proc_twin")):
        st.info("Start all three, in order (broker → engine sim → twin core), then switch to the **Live** tab.")
    else:
        st.success("All three components are running. Switch to the **Live** tab to see telemetry.")


# ---------------------------------------------------------------------------
# 3D digital twin visualization
# ---------------------------------------------------------------------------
def health_to_hex(score: float) -> str:
    """0-100 health -> red/yellow/green hex, matching the same bands used
    for the alert thresholds elsewhere in the app."""
    score = max(0.0, min(100.0, score))
    if score >= 80:
        r, g, b = 0x2e, 0xcc, 0x71   # green
    elif score >= 50:
        t = (score - 50) / 30
        r = int(0xf1 + t * (0x2e - 0xf1))
        g = int(0xc4 + t * (0xcc - 0xc4))
        b = int(0x0e + t * (0x71 - 0x0e))
    else:
        t = score / 50
        r = int(0xe7 + t * (0xf1 - 0xe7))
        g = int(0x4c + t * (0xc4 - 0x4c))
        b = int(0x3c + t * (0x0e - 0x3c))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_3d_twin(thermal, lubrication, combustion, electrical, rpm, phase):
    """Stylized (not photorealistic -- no real UAV CAD model was provided)
    3D airframe: each labeled part is colored by the subsystem health it
    represents, so the shape you see IS the health state, not a separate
    reading you have to cross-reference against a legend."""
    prop_speed = max(0.0, min(rpm, 6000)) / 6000.0 * 40
    html = f"""
    <div id="twin-canvas" style="width:100%;height:420px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.js"></script>
    <script>
      const mount = document.getElementById('twin-canvas');
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0e1117);
      const camera = new THREE.PerspectiveCamera(45, mount.clientWidth/420, 0.1, 100);
      camera.position.set(6, 4, 8);
      const renderer = new THREE.WebGLRenderer({{antialias:true}});
      renderer.setSize(mount.clientWidth, 420);
      mount.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      const dl = new THREE.DirectionalLight(0xffffff, 0.8);
      dl.position.set(5,10,5);
      scene.add(dl);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      function labeledBox(w,h,d,color,x,y,z) {{
        const geo = new THREE.BoxGeometry(w,h,d);
        const mat = new THREE.MeshStandardMaterial({{color:color}});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x,y,z);
        scene.add(mesh);
        return mesh;
      }}

      labeledBox(4, 0.8, 0.8, 0x555b66, 0, 0, 0);
      labeledBox(1, 1, 1, "{health_to_hex(thermal)}", 2.2, 0.1, 0);
      labeledBox(0.8, 0.4, 0.6, "{health_to_hex(lubrication)}", 2.2, -0.6, 0);
      labeledBox(1.2, 0.5, 0.6, "{health_to_hex(electrical)}", -1.8, -0.3, 0);

      const propGroup = new THREE.Group();
      const bladeMat = new THREE.MeshStandardMaterial({{color: "{health_to_hex(combustion)}"}});
      for (let i=0;i<2;i++) {{
        const blade = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1.6, 0.1), bladeMat);
        blade.rotation.z = i * Math.PI/2;
        propGroup.add(blade);
      }}
      propGroup.position.set(2.9, 0.1, 0);
      scene.add(propGroup);

      function animate() {{
        requestAnimationFrame(animate);
        propGroup.rotation.x += {prop_speed} * 0.01;
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    </script>
    <div style="display:flex;gap:16px;font-size:12px;color:#aaa;margin-top:4px;flex-wrap:wrap;">
      <span>Engine block = Thermal ({thermal:.0f})</span>
      <span>Oil sump = Lubrication ({lubrication:.0f})</span>
      <span>Propeller = Combustion ({combustion:.0f})</span>
      <span>Battery = Electrical ({electrical:.0f})</span>
      <span style="margin-left:auto;">Phase: {phase}</span>
    </div>
    """
    components.html(html, height=460)


# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------
def labeled_chart(df, cols, title, y_title, colors=None):
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(x=df["t_s"], y=df[col], mode="lines", name=col,
                                  line=dict(color=(colors[i] if colors else None))))
    fig.update_layout(title=title, xaxis_title="Mission time (s)", yaxis_title=y_title,
                       height=260, margin=dict(l=40, r=20, t=40, b=30),
                       template="plotly_dark", legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, use_container_width=True)


def render_alerts_panel(db_path: str, session_id: str, uav_id: str):
    st.markdown("### Alerts & Advisory Log")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT * FROM alerts_log WHERE session_id=? AND uav_id=? ORDER BY id DESC LIMIT 20",
            conn, params=(session_id, uav_id))
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        st.caption("No alerts logged yet this session.")
        return

    severity_icon = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟦", "none": "⬜"}
    for _, row in df.iterrows():
        icon = severity_icon.get(row["severity"], "⬜")
        with st.expander(f"{icon} t={row['t_s']:.0f}s — {row['message']}", expanded=False):
            if row["explanation"]:
                st.write(f"**Why:** {row['explanation']}")
            if row["recommended_action"]:
                st.write(f"**Recommended action:** {row['recommended_action']}")
            if row["rul_seconds"] is not None and not pd.isna(row["rul_seconds"]):
                st.write(f"**Predicted time to failure:** {row['rul_seconds']:.0f}s")


def render_live_tab(broker_host, broker_port, topic, db_path):
    ensure_mqtt_connected(broker_host, broker_port, topic)
    buf = get_buffer()

    if not buf:
        st.info("Awaiting live telemetry. Go to **Mission Control** and confirm all three "
                 "components are running.")
        return

    latest = buf[-1]
    warming_up = latest.get("warming_up", latest["t_s"] < WARMUP_GRACE_S)

    st.markdown(f"## {latest['uav_id']} — {latest['phase'].upper()}"
                + (" 🧊 *warming up*" if warming_up else ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RPM", f"{latest['rpm']:.0f}")
    c2.metric("CHT", f"{latest['cht']:.1f} °C")
    c3.metric("EGT", f"{latest['egt']:.1f} °C")
    c4.metric("Oil Pressure", f"{latest['oil_pressure']:.1f} bar")
    c5.metric("Altitude", f"{latest['altitude_m']:.0f} m")

    st.markdown("### 🚁 Live Digital Twin")
    render_3d_twin(latest["thermal_health"], latest["lubrication_health"],
                    latest["combustion_health"], latest["electrical_health"],
                    latest["rpm"], latest["phase"])

    st.markdown("### Subsystem Health")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Overall", f"{latest['overall_health']:.0f}/100")
    h2.metric("Thermal", f"{latest['thermal_health']:.0f}/100")
    h3.metric("Lubrication", f"{latest['lubrication_health']:.0f}/100")
    h4.metric("Combustion", f"{latest['combustion_health']:.0f}/100")
    h5.metric("Electrical", f"{latest['electrical_health']:.0f}/100")

    st.markdown("### AI Diagnostics")
    if warming_up:
        st.info("🧊 **Engine warming up** — readings are expected to look unusual for the first "
                 f"{WARMUP_GRACE_S:.0f}s of a flight while the engine reaches operating temperature. "
                 "Fault detection is suppressed until warm-up completes to avoid false alarms.")
    elif latest["ml_anomaly"]:
        st.error(f"🚨 **ANOMALY DETECTED** — predicted fault: "
                 f"**{latest['ml_fault_prediction'].upper().replace('_',' ')}** "
                 f"(confidence {latest['ml_fault_confidence']:.0%})")
        if latest.get("ml_explanation"):
            st.write(f"**Why:** {latest['ml_explanation']}")
        if latest.get("ml_rul_seconds") is not None:
            st.warning(f"⏳ Predicted time to failure: {latest['ml_rul_seconds']:.0f}s "
                       f"(~{latest['ml_rul_seconds']/60:.1f} min)")
        if latest.get("advisory_action"):
            st.write(f"**Recommended action** ({latest.get('advisory_urgency','').upper()}): "
                     f"{latest['advisory_action']}")
    else:
        st.success("✅ **STATUS: NOMINAL OPERATION**")

    df = pd.DataFrame(buf)
    st.markdown("### Telemetry Trends")
    labeled_chart(df, ["rpm"], "Engine Speed", "RPM")
    labeled_chart(df, ["cht", "egt"], "Temperatures", "°C")
    labeled_chart(df, ["oil_pressure"], "Oil Pressure", "bar")
    labeled_chart(df, ["overall_health", "thermal_health", "lubrication_health",
                        "combustion_health", "electrical_health"], "Subsystem Health", "score /100")

    render_alerts_panel(db_path, latest.get("session_id", ""), latest["uav_id"])

    st.caption(f"Ground-truth fault label (sim only, for validation): {latest.get('fault_label')} | "
               f"session: {latest.get('session_id','?')} | buffer: {len(buf)}/{BUFFER_LEN} samples")


def render_replay_tab(db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        n = pd.read_sql("SELECT COUNT(*) as n FROM telemetry_history", conn)["n"].iloc[0]
    except Exception as e:
        st.info(f"No mission history yet ({e}).")
        return
    if n == 0:
        st.info("No mission history logged yet.")
        return

    df = pd.read_sql("SELECT * FROM telemetry_history ORDER BY id", conn)
    st.write(f"Logged samples: {len(df)}")

    labeled_chart(df, ["overall_health", "thermal_health", "lubrication_health",
                        "combustion_health", "electrical_health"], "Full Mission — Health", "score /100")
    labeled_chart(df, ["rpm"], "Full Mission — RPM", "RPM")
    labeled_chart(df, ["cht", "egt"], "Full Mission — Temperatures", "°C")

    st.markdown("### Fault / anomaly timeline")
    changes = df[df["fault_label"].ne(df["fault_label"].shift())]
    st.dataframe(changes[["t_s", "phase", "fault_label", "ml_anomaly", "ml_fault_prediction",
                           "ml_fault_confidence", "ml_rul_seconds"]], use_container_width=True)

    st.download_button("Download full log as CSV", df.to_csv(index=False), file_name="mission_log.csv")


def main():
    st.title("🚁 MALE UAV Digital Twin — Ground Control Station")

    with st.sidebar:
        st.markdown("### Connection")
        broker_host = st.text_input("MQTT broker host", "localhost")
        broker_port = st.number_input("MQTT broker port", value=1883)
        telemetry_topic = st.text_input("Raw telemetry topic", DEFAULT_TELEMETRY_TOPIC)
        enriched_topic = st.text_input("Enriched topic", DEFAULT_ENRICHED_TOPIC)
        db_path = st.text_input("SQLite log path", DEFAULT_DB)

    tab_control, tab_live, tab_replay = st.tabs(["Mission Control", "Live", "Mission Replay"])

    with tab_control:
        render_control_tab(broker_host, int(broker_port), telemetry_topic)
    with tab_replay:
        render_replay_tab(db_path)
    with tab_live:
        render_live_tab(broker_host, int(broker_port), enriched_topic, db_path)
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()