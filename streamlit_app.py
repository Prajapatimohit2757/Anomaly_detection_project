"""
SkyGuard AI — Streamlit dashboard (operations-center design)

Runs the real-time simulation, model scoring, and dashboard all in ONE
process — no separate FastAPI/uvicorn server needed.

Run with:
    streamlit run streamlit_app.py

Place this file directly in your project root: C:\\Users\\hp\\sih_2026\\streamlit_app.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "models/skyguard_rf_model.pkl"
FEATURES_PATH = "models/feature_cols_v4.pkl"
DATA_PATH = "data/processed/skyguard_full_featured.csv"
THRESHOLD = 0.524
TICK_SECONDS = 1
HISTORY_LIMIT = 300

# Real offline validation metrics — from the actual train.py run (see conversation
# history / terminal output). NOT live production accuracy — labeled as such below.
OFFLINE_METRICS = {
    "accuracy": 0.9084,
    "precision": 0.723,
    "recall": 0.412,
    "f1": 0.525,
    "roc_auc": 0.766,
}
DETECTION_BY_TYPE = {
    "Spike": 0.7113,
    "Frozen": 0.5700,
    "Jump": 0.4908,
    "Drift": 0.2505,
}

LABEL_MAP = {
    'temp_9am': 'morning temperature', 'temp_3pm': 'afternoon temperature',
    'temp_min': 'minimum temperature', 'temp_max': 'maximum temperature',
    'humidity_9am': 'morning humidity', 'humidity_3pm': 'afternoon humidity',
    'pressure_9am': 'morning pressure', 'pressure_3pm': 'afternoon pressure',
    'temp_avg': 'average temperature', 'humidity_avg': 'average humidity',
    'pressure_avg': 'average pressure',
}

st.set_page_config(page_title="SkyGuard AI", page_icon="\U0001F6F0\uFE0F", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Design system — dark navy / cyan operations-center theme
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0A0E1A",
    "panel": "#101827",
    "elevated": "#161F32",
    "border": "#22304A",
    "text": "#E7ECF5",
    "text_dim": "#8492AC",
    "accent": "#22D3EE",
    "healthy": "#22C55E",
    "warning": "#EAB308",
    "high": "#F97316",
    "critical": "#EF4444",
}

STATUS_COLOR = {
    "healthy": COLORS["healthy"],
    "warning": COLORS["warning"],
    "high": COLORS["high"],
    "critical": COLORS["critical"],
}
STATUS_LABEL = {"healthy": "Healthy", "warning": "Warning", "high": "High Risk", "critical": "Critical"}

st.markdown(f"""
<style>
    .stApp {{
        background: {COLORS['bg']};
    }}
    [data-testid="stSidebar"] {{
        background: {COLORS['panel']};
        border-right: 1px solid {COLORS['border']};
    }}
    .sg-card {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        padding: 18px 20px;
    }}
    .sg-kpi-value {{
        font-size: 32px;
        font-weight: 700;
        color: {COLORS['text']};
        line-height: 1.1;
        margin: 4px 0 2px 0;
    }}
    .sg-kpi-label {{
        font-size: 13px;
        color: {COLORS['text_dim']};
    }}
    .sg-kpi-sub {{
        font-size: 12px;
        margin-top: 6px;
    }}
    .sg-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .sg-evidence-bar-bg {{
        background: {COLORS['border']};
        border-radius: 6px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }}
    .sg-evidence-bar-fill {{
        border-radius: 6px;
        height: 8px;
    }}
    .sg-live-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {COLORS['healthy']};
        margin-right: 6px;
        box-shadow: 0 0 6px {COLORS['healthy']};
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)


def badge_html(status_key, text=None):
    color = STATUS_COLOR[status_key]
    label = text or STATUS_LABEL[status_key]
    return f'<span class="sg-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{label}</span>'


def kpi_card(label, value, sub=None, sub_color=None):
    sub_html = f'<div class="sg-kpi-sub" style="color:{sub_color or COLORS["text_dim"]};">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="sg-card">
            <div class="sg-kpi-label">{label}</div>
            <div class="sg-kpi-value">{value}</div>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Explanation helpers (self-contained — no src package dependency)
# ---------------------------------------------------------------------------
def humanize_reason(feature_name, actual_value):
    if 'max_abs_zscore' in feature_name:
        return f"Reading is {actual_value:.1f}\u03c3 away from this station's own recent history"
    if 'max_abs_spatial_zscore' in feature_name:
        return f"Reading disagrees sharply with nearby stations ({actual_value:.1f}\u03c3)"
    if 'max_abs_diff' in feature_name:
        return f"Reading jumped by up to {actual_value:.1f} from the previous day, across variables"
    if 'unchanged_streak' in feature_name:
        if actual_value < 3:
            return None
        return f"Value has stayed identical for {int(actual_value)} consecutive readings \u2014 possible frozen sensor"
    if 'roll_std3' in feature_name:
        return f"Almost no variation in the last 3 readings ({actual_value:.2f}) \u2014 possible frozen sensor"
    if 'spatial_dev' in feature_name and 'zscore' in feature_name:
        base = feature_name.replace('_spatial_dev_zscore', '')
        label = LABEL_MAP.get(base, base)
        direction = "higher than" if actual_value > 0 else "lower than"
        return f"{label.capitalize()} is {direction} nearby stations ({abs(actual_value):.1f}\u03c3)"
    if 'spatial_dev' in feature_name:
        base = feature_name.replace('_spatial_dev', '')
        label = LABEL_MAP.get(base, base)
        direction = "higher than" if actual_value > 0 else "lower than"
        return f"{label.capitalize()} is {direction} nearby stations by {abs(actual_value):.1f}"
    if 'mean_gap' in feature_name:
        base = feature_name.replace('_mean_gap', '')
        label = LABEL_MAP.get(base, base)
        direction = "rising" if actual_value > 0 else "falling"
        return f"{label.capitalize()} has been {direction} over the last week (short vs long-term average gap: {actual_value:.1f})"
    if 'zscore' in feature_name:
        base = feature_name.replace('_zscore', '')
        label = LABEL_MAP.get(base, base)
        return f"{label.capitalize()} is {actual_value:.1f} standard deviations from normal for this station"
    if 'diff1' in feature_name:
        base = feature_name.replace('_diff1', '')
        label = LABEL_MAP.get(base, base)
        return f"{label.capitalize()} changed by {actual_value:.1f} from the previous reading"
    if 'swing' in feature_name:
        return f"Unusual {feature_name.replace('_', ' ')}: {actual_value:.1f}"
    if 'ratio' in feature_name:
        return f"Unusual temperature-to-humidity ratio: {actual_value:.2f}"
    if 'range' in feature_name:
        return f"Unusually wide daily temperature range: {actual_value:.1f}\u00b0C"
    return f"{feature_name}: {actual_value:.2f}"


def categorize_reason(reason_text):
    """Buckets a human-readable reason into one of the 4 evidence types for the
    'Why was this flagged?' panel, matching the design spec's evidence categories."""
    t = reason_text.lower()
    if 'frozen sensor' in t or 'variation in the last 3' in t:
        return "Frozen sensor behavior"
    if 'nearby stations' in t:
        return "Spatial inconsistency"
    if 'temperature-to-humidity' in t or 'daily temperature range' in t:
        return "Multivariate inconsistency"
    return "Temporal deviation"


def explain_row(X_row, explainer, feature_cols, top_n=5):
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_row = shap_values[0, :, 1]
    else:
        shap_row = shap_values[0]

    contributions = pd.DataFrame({
        'feature': feature_cols,
        'actual_value': X_row.iloc[0].values,
        'shap_value': shap_row,
    })
    top = contributions[contributions['shap_value'] > 0].sort_values(
        'shap_value', ascending=False
    ).head(top_n)

    reasons = []
    for _, r in top.iterrows():
        reason = humanize_reason(r['feature'], r['actual_value'])
        if reason:
            reasons.append(reason)
    return reasons


# ---------------------------------------------------------------------------
# Load model + data once, cached across reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer


@st.cache_data
def load_stream_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


model, feature_cols, explainer = load_model_artifacts()
stream_df = load_stream_data()


def score_row(row):
    X_row = row[feature_cols].to_frame().T.astype(float)
    prob = float(model.predict_proba(X_row)[0, 1])
    is_anomaly = prob >= THRESHOLD
    result = {
        "station_id": row["station_id"],
        "date": str(row["date"].date()),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "temperature": round(float(row["temp_avg"]), 1),
        "humidity": round(float(row["humidity_avg"]), 1),
        "pressure": round(float(row["pressure_avg"]), 1),
        "probability": round(prob, 3),
        "is_anomaly": bool(is_anomaly),
        "reasons": [],
    }
    if is_anomaly:
        result["reasons"] = explain_row(X_row, explainer, feature_cols)
    return result


def status_for(reading):
    p = reading["probability"]
    if p >= THRESHOLD:
        return "critical"
    if p >= 0.35:
        return "high"
    if p >= 0.20:
        return "warning"
    return "healthy"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "row_idx": 0, "stations": {}, "station_history": {}, "alerts": [],
    "running": True, "page": "Overview",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def advance_simulation():
    if not st.session_state.running:
        return
    row = stream_df.iloc[st.session_state.row_idx]
    reading = score_row(row)
    sid = reading["station_id"]

    st.session_state.stations[sid] = reading
    hist = st.session_state.station_history.setdefault(sid, [])
    hist.append(reading)
    if len(hist) > HISTORY_LIMIT:
        del hist[0]

    if reading["is_anomaly"]:
        st.session_state.alerts.insert(0, reading)
        st.session_state.alerts = st.session_state.alerts[:60]
    st.session_state.row_idx = (st.session_state.row_idx + 1) % len(stream_df)


def compute_health(sid):
    """Real health score computed from THIS session's accumulated model
    predictions for the station — not a fake number. anomaly_rate is the rate
    at which the MODEL flagged this station, not ground truth (a real
    deployment wouldn't have ground truth either)."""
    hist = st.session_state.station_history.get(sid, [])
    n = len(hist)
    if n == 0:
        return None
    anomaly_rate = sum(1 for r in hist if r["is_anomaly"]) / n
    recent = hist[-20:]
    recent_rate = sum(1 for r in recent if r["is_anomaly"]) / len(recent)
    # weight recent behavior more heavily — a station trending worse should
    # score lower than one with a single old blip
    blended_rate = 0.4 * anomaly_rate + 0.6 * recent_rate
    health = round(100 * (1 - blended_rate))
    health = max(0, min(100, health))

    reason_tags = {}
    for r in hist:
        if r["is_anomaly"]:
            for reason in r["reasons"]:
                tag = categorize_reason(reason)
                reason_tags[tag] = reason_tags.get(tag, 0) + 1

    if health >= 85:
        band = "healthy"
    elif health >= 70:
        band = "warning"  # spec calls this "Good" but reuses yellow
    elif health >= 50:
        band = "high"
    else:
        band = "critical"

    return {
        "station_id": sid,
        "health": health,
        "readings": n,
        "anomaly_rate": anomaly_rate,
        "band": band,
        "top_reasons": sorted(reason_tags.items(), key=lambda x: -x[1])[:3],
    }


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""<div style="padding:4px 0 20px 0;">
            <div style="font-size:20px;font-weight:700;color:{COLORS['text']};">
                \U0001F6F0\uFE0F SkyGuard AI
            </div>
            <div style="font-size:12px;color:{COLORS['text_dim']};">Weather Intelligence</div>
        </div>""",
        unsafe_allow_html=True,
    )

    pages = ["Overview", "Stations", "Live Alerts", "Analytics", "Sensor Health & Maintenance"]
    icons = ["\U0001F4CA", "\U0001F5FA\uFE0F", "\U0001F6A8", "\U0001F4C8", "\U0001F527"]
    labels = [f"{icon}  {p}" for icon, p in zip(icons, pages)]
    choice = st.radio("nav", labels, label_visibility="collapsed",
                       index=pages.index(st.session_state.page))
    st.session_state.page = pages[labels.index(choice)]

    st.divider()

    if st.button("\u23F8\uFE0F Pause simulation" if st.session_state.running else "\u25B6\uFE0F Resume simulation",
                  width="stretch"):
        st.session_state.running = not st.session_state.running

    st.markdown(
        f"""<div style="position:fixed; bottom:20px; width:200px;">
            <div style="font-size:11px;color:{COLORS['text_dim']};margin-bottom:4px;">SYSTEM STATUS</div>
            <div style="font-size:13px;color:{COLORS['healthy']};">
                <span class="sg-live-dot"></span>All systems operational
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header (shared across pages)
# ---------------------------------------------------------------------------
def render_header(title, subtitle):
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"<h1 style='margin-bottom:0;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COLORS['text_dim']};margin-top:-8px;'>{subtitle}</div>",
                     unsafe_allow_html=True)
    with h2:
        is_live = st.session_state.running
        dot_color = COLORS["healthy"] if is_live else COLORS["warning"]
        latest_date = stream_df.iloc[st.session_state.row_idx]["date"].date() if len(stream_df) else "\u2014"
        st.markdown(
            f"""<div style="text-align:right;padding-top:8px;">
                <span style="color:{dot_color};">\u25CF</span>
                <span style="color:{COLORS['text']};font-size:13px;"> {'System Live' if is_live else 'Paused'}</span><br/>
                <span style="color:{COLORS['text_dim']};font-size:12px;">Simulated date: {latest_date}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border-color:%s;margin-top:8px;'>" % COLORS["border"], unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    render_header("Weather Station Intelligence", "Real-time AWS anomaly monitoring")

    stations = st.session_state.stations
    total = len(stations)
    counts = {"healthy": 0, "warning": 0, "high": 0, "critical": 0}
    for r in stations.values():
        counts[status_for(r)] += 1

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Stations", total, "Across monitored network")
    with c2:
        pct = f"{counts['healthy']/total:.0%}" if total else "\u2014"
        kpi_card("Healthy", counts["healthy"], pct, COLORS["healthy"])
    with c3:
        pct = f"{(counts['warning']+counts['high'])/total:.0%}" if total else "\u2014"
        kpi_card("Warning", counts["warning"] + counts["high"], pct, COLORS["warning"])
    with c4:
        pct = f"{counts['critical']/total:.0%}" if total else "\u2014"
        kpi_card("Critical", counts["critical"], pct, COLORS["critical"])

    st.write("")
    col_map, col_alerts = st.columns([2, 1])

    with col_map:
        st.markdown("**Live AWS Network**")
        st.caption("Real-time station health and anomaly status")
        fmap = folium.Map(
            location=[22.9, 78.9], zoom_start=4.4,
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr='&copy; OpenStreetMap contributors &copy; CARTO',
        )
        for sid, reading in stations.items():
            status = status_for(reading)
            color = STATUS_COLOR[status]
            folium.CircleMarker(
                location=[reading["lat"], reading["lon"]],
                radius=9 if status in ("critical", "high") else 6,
                color=color, fill=True, fill_color=color, fill_opacity=0.85, weight=2,
                popup=folium.Popup(
                    f"<b>{sid}</b><br>{reading['date']}<br>"
                    f"T={reading['temperature']}\u00b0C H={reading['humidity']}% P={reading['pressure']}hPa<br>"
                    f"Status: {STATUS_LABEL[status]}",
                    max_width=220,
                ),
                tooltip=sid,
            ).add_to(fmap)
        st_folium(fmap, width=None, height=460, returned_objects=[])

    with col_alerts:
        st.markdown("**Live Alerts**")
        alerts = st.session_state.alerts
        if not alerts:
            st.caption("No anomalies detected yet \u2014 watching the stream\u2026")
        for a in alerts[:8]:
            status = status_for(a)
            st.markdown(
                f"""<div class="sg-card" style="margin-bottom:8px;padding:12px 14px;">
                    {badge_html(status)}<br/>
                    <b>{a['station_id']}</b> \u2022 {a['date']}<br/>
                    <span style="color:{COLORS['text_dim']};font-size:12px;">{a['reasons'][0] if a['reasons'] else ''}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        if len(alerts) > 8:
            st.caption(f"+{len(alerts) - 8} more \u2014 see Live Alerts page")


# ---------------------------------------------------------------------------
# Page: Stations
# ---------------------------------------------------------------------------
def page_stations():
    render_header("Weather Stations", "Search, filter, and inspect individual stations")

    stations = st.session_state.stations
    if not stations:
        st.caption("Waiting for the first reading\u2026")
        return

    filter_choice = st.radio("Filter", ["All", "Healthy", "Warning", "High Risk", "Critical"],
                              horizontal=True, label_visibility="collapsed")
    filter_map = {"Healthy": "healthy", "Warning": "warning", "High Risk": "high", "Critical": "critical"}

    rows = []
    for sid, r in stations.items():
        status = status_for(r)
        if filter_choice != "All" and status != filter_map.get(filter_choice):
            continue
        rows.append({
            "Station": sid, "Temperature": f"{r['temperature']}\u00b0C",
            "Pressure": f"{r['pressure']} hPa", "Humidity": f"{r['humidity']}%",
            "AI Score": f"{r['probability']:.0%}", "_prob": r["probability"],
            "Status": STATUS_LABEL[status], "Last Updated": r["date"],
        })
    table_df = pd.DataFrame(rows)
    if table_df.empty:
        st.info(f"No stations currently match the '{filter_choice}' filter.")
    else:
        table_df = table_df.sort_values("_prob", ascending=False).drop(columns=["_prob"])
        st.dataframe(table_df, width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Station intelligence")
    selected = st.selectbox("Select a station to inspect", sorted(stations.keys()))
    reading = stations[selected]
    status = status_for(reading)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"#### {selected}")
        st.markdown(badge_html(status), unsafe_allow_html=True)
        st.write("")
        s1, s2, s3 = st.columns(3)
        s1.metric("Temperature", f"{reading['temperature']}\u00b0C")
        s2.metric("Pressure", f"{reading['pressure']} hPa")
        s3.metric("Humidity", f"{reading['humidity']}%")

        st.markdown("**AI Anomaly Analysis**")
        st.markdown(f"Anomaly Score: **{reading['probability']:.0%}**")
        st.progress(min(reading["probability"], 1.0))

        if reading["is_anomaly"] and reading["reasons"]:
            st.markdown("**Why was this flagged?**")
            tag_counts = {}
            for reason in reading["reasons"]:
                tag_counts[categorize_reason(reason)] = tag_counts.get(categorize_reason(reason), 0) + 1
            for reason in reading["reasons"]:
                tag = categorize_reason(reason)
                st.markdown(f"**{tag}**")
                st.caption(reason)
        else:
            st.success("No active evidence of anomaly for this reading.")

    with right:
        history = st.session_state.station_history.get(selected, [])
        if history:
            hist_df = pd.DataFrame(history)
            hist_df["date"] = pd.to_datetime(hist_df["date"])
            hist_df = hist_df.sort_values("date")
            st.markdown("**Sensor timeline (this session)**")
            tab_t, tab_h, tab_p = st.tabs(["Temperature", "Humidity", "Pressure"])
            with tab_t:
                st.line_chart(hist_df.set_index("date")["temperature"])
            with tab_h:
                st.line_chart(hist_df.set_index("date")["humidity"])
            with tab_p:
                st.line_chart(hist_df.set_index("date")["pressure"])


# ---------------------------------------------------------------------------
# Page: Live Alerts
# ---------------------------------------------------------------------------
def page_alerts():
    render_header("Alerts & Anomalies", "Incident view of every anomaly detected this session")

    alerts = st.session_state.alerts
    st.markdown(f"**{len(alerts)} alerts** this session")

    if not alerts:
        st.caption("No anomalies detected yet.")
        return

    for a in alerts:
        status = status_for(a)
        with st.container(border=True):
            top = st.columns([1, 3, 1])
            with top[0]:
                st.markdown(badge_html(status), unsafe_allow_html=True)
            with top[1]:
                st.markdown(f"**{a['station_id']}** \u2014 {a['date']}")
            with top[2]:
                st.markdown(f"AI Score: **{a['probability']:.0%}**")
            for reason in a["reasons"]:
                st.markdown(f"- {reason}")


# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------
def page_analytics():
    render_header("AI Analytics", "Model performance and explainability")

    st.markdown("#### Offline validation performance")
    st.caption("Measured on a held-out time-based test split with injected, labeled anomalies. "
               "This is NOT live production accuracy \u2014 real-world performance depends on how "
               "well injected fault patterns match genuine sensor failures.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{OFFLINE_METRICS['accuracy']:.1%}")
    c2.metric("Precision", f"{OFFLINE_METRICS['precision']:.1%}")
    c3.metric("Recall", f"{OFFLINE_METRICS['recall']:.1%}")
    c4.metric("F1 Score", f"{OFFLINE_METRICS['f1']:.1%}")
    c5.metric("ROC-AUC", f"{OFFLINE_METRICS['roc_auc']:.3f}")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Detection rate by anomaly type**")
        st.caption("Spike and jump faults are easiest to catch; slow drift is inherently "
                   "harder \u2014 it looks similar to natural weather variation day-to-day.")
        det_df = pd.DataFrame({"Anomaly type": list(DETECTION_BY_TYPE.keys()),
                                "Detection rate": list(DETECTION_BY_TYPE.values())})
        st.bar_chart(det_df.set_index("Anomaly type"))

    with col2:
        st.markdown("**Alerts this session, by severity**")
        alerts = st.session_state.alerts
        sev_counts = {"Critical": 0, "High Risk": 0, "Warning": 0}
        for a in alerts:
            s = status_for(a)
            if s == "critical":
                sev_counts["Critical"] += 1
            elif s == "high":
                sev_counts["High Risk"] += 1
            else:
                sev_counts["Warning"] += 1
        if sum(sev_counts.values()) > 0:
            st.bar_chart(pd.DataFrame({"Severity": list(sev_counts.keys()),
                                        "Count": list(sev_counts.values())}).set_index("Severity"))
        else:
            st.caption("No alerts yet this session.")

    st.divider()
    st.markdown("#### How SkyGuard reaches a decision")
    st.caption("Evidence from four independent signals is combined before a single anomaly score is produced.")
    st.markdown(
        f"""<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
        {"".join([
            f'<div class="sg-card" style="flex:1;min-width:140px;text-align:center;padding:14px 10px;">{label}</div>'
            + ('<div style="color:%s;font-size:20px;">&#8595;</div>' % COLORS['text_dim'] if False else '')
            for label in ["Temporal evidence", "Spatial evidence", "Multivariate evidence", "ML prediction"]
        ])}
        </div>
        <div style="text-align:center;color:{COLORS['text_dim']};margin:10px 0;">\u2193 evidence fusion \u2193</div>
        <div class="sg-card" style="text-align:center;padding:14px;">Anomaly score \u2192 Severity \u2192 Root cause \u2192 Sensor health \u2192 Maintenance priority</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Sensor Health & Maintenance
# ---------------------------------------------------------------------------
def page_sensor_health():
    render_header("Sensor Health & Maintenance",
                  "Stations ranked by recent anomaly behavior to prioritize inspection")

    station_ids = sorted(st.session_state.station_history.keys())
    if not station_ids:
        st.caption("Waiting for readings\u2026")
        return

    health_records = [compute_health(sid) for sid in station_ids]
    health_records = [h for h in health_records if h is not None]

    network_health = round(sum(h["health"] for h in health_records) / len(health_records))
    band_counts = {"healthy": 0, "warning": 0, "high": 0, "critical": 0}
    for h in health_records:
        band_counts[h["band"]] += 1

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Network health", f"{network_health}%")
    c2.metric("Healthy", band_counts["healthy"])
    c3.metric("Good", band_counts["warning"])
    c4.metric("Warning", band_counts["high"])
    c5.metric("Critical", band_counts["critical"])

    st.caption(
        "Health = 100 \u2212 (weighted anomaly-flag rate for this station this session, "
        "recent readings weighted more heavily than older ones). This is computed from the "
        "model's own live predictions, not from injected ground truth \u2014 the same signal a "
        "real deployment would have available."
    )

    st.divider()
    st.markdown("#### Sensor health table")
    table_rows = []
    for h in health_records:
        table_rows.append({
            "Station": h["station_id"], "Health Score": h["health"],
            "Readings seen": h["readings"], "Anomaly rate": f"{h['anomaly_rate']:.0%}",
            "Status": STATUS_LABEL[h["band"]],
        })
    st.dataframe(
        pd.DataFrame(table_rows).sort_values("Health Score"),
        width="stretch", hide_index=True,
    )

    st.divider()
    st.markdown("#### Maintenance priority")
    st.caption("Stations are ranked using recent anomaly behavior and sensor-health indicators "
               "to prioritize inspection. This does not claim to know the exact cause or date of "
               "a physical failure \u2014 it flags where inspection is most worthwhile.")

    priority_order = sorted(health_records, key=lambda h: h["health"])
    tiers = [("HIGH PRIORITY", [h for h in priority_order if h["band"] == "critical"]),
             ("MEDIUM PRIORITY", [h for h in priority_order if h["band"] == "high"]),
             ("LOW PRIORITY", [h for h in priority_order if h["band"] in ("warning", "healthy") and h["health"] < 95])]

    any_shown = False
    for tier_name, items in tiers:
        if not items:
            continue
        any_shown = True
        st.markdown(f"**{tier_name}**")
        for h in items[:5]:
            with st.container(border=True):
                cols = st.columns([1, 2, 1])
                cols[0].markdown(f"**{h['station_id']}**")
                cols[1].markdown(f"Health Score: **{h['health']}**")
                cols[2].markdown(badge_html(h["band"]), unsafe_allow_html=True)
                if h["top_reasons"]:
                    st.caption("Reasons: " + ", ".join(f"{tag} ({count}x)" for tag, count in h["top_reasons"]))
                    top_tag = h["top_reasons"][0][0]
                    action = {
                        "Frozen sensor behavior": "Inspect temperature sensor and communication link.",
                        "Spatial inconsistency": "Cross-check against nearby stations; inspect calibration.",
                        "Multivariate inconsistency": "Inspect humidity/pressure sensor pairing.",
                        "Temporal deviation": "Review recent readings for gradual drift; schedule inspection.",
                    }.get(top_tag, "Schedule a routine inspection.")
                    st.markdown(f"Recommended action: *{action}*")
                else:
                    st.caption("Health declining but no specific fault pattern dominant yet.")
    if not any_shown:
        st.success("All stations currently in good health \u2014 no maintenance priorities this session.")


# ---------------------------------------------------------------------------
# Main render loop — ticks the simulation, then draws whichever page is active
# ---------------------------------------------------------------------------
@st.fragment(run_every=TICK_SECONDS if st.session_state.running else None)
def app():
    advance_simulation()
    page = st.session_state.page
    if page == "Overview":
        page_overview()
    elif page == "Stations":
        page_stations()
    elif page == "Live Alerts":
        page_alerts()
    elif page == "Analytics":
        page_analytics()
    elif page == "Sensor Health & Maintenance":
        page_sensor_health()


app()