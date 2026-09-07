"""
dashboard/components.py
=======================
Plotly figure builders for the GCS dashboard. Dark, high-contrast,
military/aerospace palette. Every function takes plain data and returns a
`go.Figure` (or an HTML string) -- no Streamlit calls in here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -- palette --------------------------------------------------------------
BG = "#0a0e14"
PANEL = "#111720"
GRID = "#1e2733"
FG = "#c8d3e0"
AMBER = "#ffb020"
GREEN = "#3ddc84"
RED = "#ff4d4d"
CYAN = "#37c0e0"
VIOLET = "#a78bfa"
STAGE_COLORS = {
    "TAXI": "#1b2430", "CLIMB": "#132a1f", "CRUISE": "#0f2436",
    "LOITER": "#241f10", "EMERGENCY_RECOVERY": "#2c1414",
}

_SENSOR_UNITS = {
    "rpm": "RPM", "cht": "°C", "egt": "°C", "oil_pressure": "bar",
    "oil_temp": "°C", "fuel_flow": "L/h", "vibration": "g",
    "battery_voltage": "V",
}
_SENSOR_LABEL = {
    "rpm": "RPM", "cht": "CHT", "egt": "EGT", "oil_pressure": "Oil Press",
    "oil_temp": "Oil Temp", "fuel_flow": "Fuel Flow", "vibration": "Vib RMS",
    "battery_voltage": "Battery",
}


def _base_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=FG, family="ui-monospace, SFMono-Regular, Menlo, monospace", size=12),
        margin=dict(l=54, r=18, t=38 if title else 14, b=32),
        title=dict(text=title, font=dict(size=13, color=FG)) if title else None,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Mission lifecycle timeline
# ---------------------------------------------------------------------------
def mission_lifecycle_fig(df: pd.DataFrame, mission_duration_s: float = 180.0) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, 300, "MISSION LIFECYCLE")

    t = df["t_s"].to_numpy()

    # stage bands
    seen = []
    for stg in df["mission_stage"]:
        if not seen or seen[-1][0] != stg:
            seen.append([stg, None, None])
    idx = 0
    for i, stg in enumerate(df["mission_stage"]):
        if seen[idx][1] is None:
            seen[idx][1] = df["t_s"].iloc[i]
        seen[idx][2] = df["t_s"].iloc[i]
        if idx + 1 < len(seen) and seen[idx + 1][0] == df["mission_stage"].iloc[min(i + 1, len(df) - 1)]:
            idx += 1
    for stg, lo, hi in seen:
        fig.add_vrect(x0=lo, x1=max(hi, lo + 0.1), fillcolor=STAGE_COLORS.get(stg, "#1b2430"),
                      opacity=0.55, line_width=0, layer="below",
                      annotation_text=stg.replace("_", " "), annotation_position="top left",
                      annotation=dict(font=dict(size=9, color="#8fa0b4")))

    alt = df["altitude_ft"].to_numpy()
    alt_norm = 100.0 * (alt - alt.min()) / max(np.ptp(alt), 1.0) if len(alt) else alt
    fig.add_trace(go.Scatter(x=t, y=df["throttle_pct"], name="Throttle %", mode="lines",
                             line=dict(color=CYAN, width=1.6)))
    fig.add_trace(go.Scatter(x=t, y=alt_norm, name="Altitude (norm)", mode="lines",
                             line=dict(color=VIOLET, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=df["health_index"], name="Predicted Health", mode="lines",
                             line=dict(color=GREEN, width=2.6)))

    anom = df[df["anomaly"]]
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom["t_s"], y=anom["health_index"], name="Anomaly",
                                 mode="markers", marker=dict(color=RED, size=5, symbol="x")))

    fig.add_hline(y=55, line=dict(color=RED, width=1, dash="dash"), opacity=0.5)
    fig.update_xaxes(range=[0, mission_duration_s], title_text="mission time  (s)")
    fig.update_yaxes(range=[0, 105], title_text="%  /  health  /  alt(norm)")
    return _base_layout(fig, 300, "MISSION LIFECYCLE  —  TAXI → CLIMB → CRUISE → LOITER → RECOVERY")


# ---------------------------------------------------------------------------
# 2. Digital-twin cognition -- normalized residual radar
# ---------------------------------------------------------------------------
def twin_cognition_fig(norm_residuals: dict) -> go.Figure:
    order = ["rpm", "cht", "egt", "oil_pressure", "oil_temp", "fuel_flow",
             "vibration", "battery_voltage"]
    if not norm_residuals:
        norm_residuals = {k: 0.0 for k in order}
    order = [k for k in order if k in norm_residuals] or order
    mags = [min(abs(norm_residuals.get(k, 0.0)), 4.0) for k in order]
    labels = [_SENSOR_LABEL.get(k, k) for k in order]
    colors = [RED if m > 1.0 else (AMBER if m > 0.6 else GREEN) for m in mags]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=mags, theta=labels, marker_color=colors,
                              marker_line_color=BG, marker_line_width=1.5, opacity=0.9,
                              hovertemplate="%{theta}: %{r:.2f}× tolerance<extra></extra>"))
    # tolerance ring at r = 1.0
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(labels) + 1), theta=labels + [labels[0]], mode="lines",
        line=dict(color=AMBER, width=1.4, dash="dash"), name="tolerance",
        hoverinfo="skip"))
    fig.update_layout(
        template="plotly_dark", height=300, paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=FG, family="ui-monospace, monospace", size=11),
        margin=dict(l=30, r=30, t=42, b=30), showlegend=False,
        title=dict(text="DIGITAL-TWIN COGNITION  —  |sensor − physics| / tolerance",
                   font=dict(size=13, color=FG)),
        polar=dict(
            bgcolor=PANEL,
            radialaxis=dict(range=[0, 4], gridcolor=GRID, tickvals=[1, 2, 3, 4],
                            ticktext=["1×", "2×", "3×", "≥4×"],
                            tickfont=dict(size=9, color="#7f8ea3")),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10, color=FG)),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Twin residual over time -- measured vs physics baseline for one channel
# ---------------------------------------------------------------------------
def _worst_channel(df: pd.DataFrame) -> str:
    if df.empty:
        return "cht"
    last = df.iloc[-1]["norm_residuals"]
    if not isinstance(last, dict) or not last:
        return "cht"
    return max(last, key=lambda k: abs(last[k]))


def residual_track_fig(df: pd.DataFrame, channel: str | None = None) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, 300, "TWIN RESIDUAL")
    ch = channel or _worst_channel(df)
    t = df["t_s"].to_numpy()
    meas = df[ch].to_numpy() if ch in df.columns else np.zeros(len(df))
    resid = np.array([r.get(ch, 0.0) if isinstance(r, dict) else 0.0 for r in df["residuals"]])
    physics = meas - resid

    fig.add_trace(go.Scatter(x=t, y=physics, name="physics baseline", mode="lines",
                             line=dict(color=CYAN, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=meas, name="sensor (CAN)", mode="lines",
                             line=dict(color=AMBER, width=2.0),
                             fill="tonexty", fillcolor="rgba(255,77,77,0.16)"))
    unit = _SENSOR_UNITS.get(ch, "")
    return _base_layout(fig, 300,
                        f"TWIN RESIDUAL  —  {_SENSOR_LABEL.get(ch, ch).upper()}  "
                        f"(Δ shaded, {unit})")


# ---------------------------------------------------------------------------
# 4. Live SHAP attribution bar
# ---------------------------------------------------------------------------
def shap_bar_fig(explanation: dict | None) -> go.Figure:
    fig = go.Figure()
    if not explanation or not explanation.get("drivers"):
        _base_layout(fig, 300, "ROOT-CAUSE ATTRIBUTION  —  awaiting anomaly")
        fig.add_annotation(text="no anomaly active", showarrow=False,
                           font=dict(color="#7f8ea3", size=12))
        return fig
    drivers = explanation["drivers"][::-1]
    labels = [d["label"] for d in drivers]
    impact = [d["impact_pct"] * (1 if d["shap"] >= 0 else -1) for d in drivers]
    text = [f"{d['value']:+.2f}  ({d['impact_pct']:.0f}%)" for d in drivers]
    colors = [RED if d["shap"] >= 0 else CYAN for d in drivers]
    fig.add_trace(go.Bar(x=[abs(v) for v in impact], y=labels, orientation="h",
                         marker_color=colors, text=text, textposition="outside",
                         textfont=dict(color=FG, size=11),
                         hovertemplate="%{y}: %{x:.0f}% of attribution<extra></extra>"))
    method = explanation.get("method", "TreeSHAP")
    lat = explanation.get("latency_ms", 0.0)
    fig.update_xaxes(range=[0, max(abs(v) for v in impact) * 1.35 + 5])
    return _base_layout(fig, 300,
                        f"ROOT-CAUSE ATTRIBUTION  —  {method}  ({lat:.0f} ms)")


# ---------------------------------------------------------------------------
# 5. Per-cylinder EGT (misfire signature)
# ---------------------------------------------------------------------------
def per_cylinder_egt_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _base_layout(fig, 260, "PER-CYLINDER EGT")
    t = df["t_s"].to_numpy()
    palette = [GREEN, CYAN, RED, VIOLET]
    for i in range(1, 5):
        col = f"egt_{i}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df[col], name=f"Cyl {i}", mode="lines",
                                     line=dict(color=palette[i - 1], width=1.7)))
    return _base_layout(fig, 260, "PER-CYLINDER EGT  —  spread flags injector / misfire")


# ---------------------------------------------------------------------------
# 6. Small helpers for the metric strip
# ---------------------------------------------------------------------------
def fmt_rul(rul_seconds: float | None) -> tuple[str, str]:
    """(display string, severity) for the RUL metric."""
    if rul_seconds is None:
        return "—", "nominal"
    if rul_seconds <= 20:
        return "ABORT NOW", "critical"
    if rul_seconds < 90:
        return f"{int(rul_seconds)}s — ABORT", "critical"
    m, s = divmod(int(rul_seconds), 60)
    sev = "warning" if rul_seconds < 300 else "caution"
    return f"{m:02d}:{s:02d} to limit", sev
