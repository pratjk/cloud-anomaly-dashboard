"""
Self federated Cloud Anomaly Detection — Edge Client Anomaly Injector UI

A premium Streamlit application that acts as an edge client.
Allows targeting a remote central server and injecting simulated
real-world anomaly payloads with a professional industrial interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import simulator.anomaly_injector as injector

# ============================================================================
# Configuration
# ============================================================================
REGIONS = {
    "North America (San Francisco)":  (37.7749,  -122.4194),
    "Europe (Frankfurt)":             (50.1109,    8.6821),
    "Asia (Tokyo)":                   (35.6762,  139.6503),
    "South America (Sao Paulo)":     (-23.5505,  -46.6333),
    "Australia (Sydney)":            (-33.8688,  151.2093),
    "Eastern Europe (Moscow)":        (55.7558,   37.6173),
    "Southeast Asia (Singapore)":      (1.3521,  103.8198),
    "Middle East (Dubai)":            (25.2048,   55.2708),
}

PAYLOADS = [
    {
        "id": "memory_leak",
        "label": "Memory Leak",
        "icon": "🧠",
        "color": "#a78bfa",
        "accent": "rgba(167,139,250,0.12)",
        "border": "rgba(167,139,250,0.35)",
        "severity": "HIGH",
        "severity_color": "#f59e0b",
        "description": "Simulates progressive heap exhaustion. Memory climbs steadily while CPU stays moderate — a classic leak signature.",
        "metrics": "MEM: 98.5% · CPU: 45% · NET: 300 Mbps",
        "fn": injector.generate_memory_leak,
        "btn_label": "Inject Memory Leak",
    },
    {
        "id": "ddos",
        "label": "DDoS / Network Flood",
        "icon": "🌐",
        "color": "#38bdf8",
        "accent": "rgba(56,189,248,0.08)",
        "border": "rgba(56,189,248,0.3)",
        "severity": "CRITICAL",
        "severity_color": "#ef4444",
        "description": "Simulates a massive influx of unauthorized network packets overwhelming bandwidth — volumetric DDoS fingerprint.",
        "metrics": "NET: 1000 Mbps · CPU: 85% · MEM: 70%",
        "fn": injector.generate_ddos_attack,
        "btn_label": "Inject DDoS Attack",
    },
    {
        "id": "cpu_spike",
        "label": "Crypto-Mining / CPU Spike",
        "icon": "⚡",
        "color": "#fb923c",
        "accent": "rgba(251,146,60,0.08)",
        "border": "rgba(251,146,60,0.3)",
        "severity": "HIGH",
        "severity_color": "#f59e0b",
        "description": "Unauthorized heavy processor workload with elevated network activity — typical cryptomining or ransomware CPU signature.",
        "metrics": "CPU: 99.9% · NET: 950 Mbps · MEM: 60%",
        "fn": injector.generate_crypto_mining,
        "btn_label": "Inject CPU Spike",
    },
    {
        "id": "disk_failure",
        "label": "Disk Failure",
        "icon": "💾",
        "color": "#4ade80",
        "accent": "rgba(74,222,128,0.07)",
        "border": "rgba(74,222,128,0.25)",
        "severity": "MEDIUM",
        "severity_color": "#38bdf8",
        "description": "Simulates sudden I/O failure — disk read/write velocity collapses to near-zero, triggering storage degradation alert.",
        "metrics": "DISK: 0 MB/s · CPU: 20% · MEM: 45%",
        "fn": injector.generate_disk_failure,
        "btn_label": "Inject Disk Failure",
    },
]

# ============================================================================
# CSS — Industrial Premium Theme (matches dashboard)
# ============================================================================
INJECTOR_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #05080f !important;
    color: #e2e8f0 !important;
}

div[data-testid="stAppViewContainer"] {
    background-color: #05080f;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.08) 0%, transparent 60%),
        linear-gradient(rgba(99,102,241,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.025) 1px, transparent 1px);
    background-size: 100% 100%, 40px 40px, 40px 40px;
    min-height: 100vh;
}

div[data-testid="stHeader"] {
    background: rgba(5, 8, 15, 0.9) !important;
    border-bottom: 1px solid rgba(99,102,241,0.12);
    backdrop-filter: blur(20px);
}

div[data-testid="block-container"] {
    padding: 1.5rem 3rem 3rem !important;
    max-width: 960px;
    margin: 0 auto;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(14, 20, 35, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 14px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    transition: all 0.25s ease !important;
}

div[data-testid="metric-container"]:hover {
    border-color: rgba(99,102,241,0.45) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.1) !important;
}

div[data-testid="stMetricLabel"] > div {
    color: #64748b !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

div[data-testid="stMetricValue"] > div {
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: rgba(14,20,35,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.25rem !important;
    width: 100%;
    transition: all 0.25s ease !important;
    letter-spacing: 0.3px !important;
}

.stButton > button:hover {
    background: rgba(99,102,241,0.2) !important;
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.2) !important;
    color: #c7d2fe !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(14, 20, 35, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* Label */
.stTextInput label, .stSelectbox label {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(99,102,241,0.1) !important;
    margin: 1.5rem 0 !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Custom Components ── */

.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.12);
}

.page-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.page-subtitle {
    font-size: 12px;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.35rem;
}

.config-panel {
    background: rgba(14, 20, 35, 0.7);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.75rem;
    backdrop-filter: blur(12px);
}

.config-panel-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6366f1;
    font-family: 'Inter', sans-serif;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    font-family: 'Inter', sans-serif;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}

.payload-card {
    background: rgba(14, 20, 35, 0.75);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(16px);
}

.payload-card:hover {
    transform: translateY(-2px);
}

.payload-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.payload-icon {
    font-size: 1.5rem;
    line-height: 1;
}

.payload-label {
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
    margin-bottom: 2px;
}

.payload-desc {
    font-size: 12px;
    color: #64748b;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}

.payload-metrics {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    background: rgba(5,8,15,0.5);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 6px;
    padding: 5px 10px;
    margin-bottom: 1rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.5px;
}

.status-strip {
    background: rgba(14,20,35,0.6);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1.5rem;
}

.status-strip-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #475569;
    font-family: 'Inter', sans-serif;
    margin-bottom: 0.75rem;
}

.target-url-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #6366f1;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 8px 12px;
    word-break: break-all;
}

.live-dot {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    color: #4ade80;
    font-family: 'Inter', sans-serif;
}

.live-dot::before {
    content: '';
    width: 7px;
    height: 7px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 8px #4ade80;
    animation: blink 1.5s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.result-success {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    border-left: 3px solid #22c55e;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #86efac;
    margin-top: 0.5rem;
}

.result-error {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #fca5a5;
    margin-top: 0.5rem;
}

.injection-log-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.06);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
}

.injection-log-item:last-child {
    border-bottom: none;
}
</style>
"""

# ============================================================================
# Page Setup
# ============================================================================
st.set_page_config(
    page_title="Edge Client Injector",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(INJECTOR_STYLES, unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================
st.markdown("""
<div class="page-header">
    <div class="page-title">⚡ Edge Client Injector</div>
    <div class="page-subtitle">Simulate real-world anomaly payloads and transmit to the central inference engine</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Connection Config Panel
# ============================================================================
st.markdown('<div class="config-panel">', unsafe_allow_html=True)
st.markdown('<div class="config-panel-title">⚙ Connection Configuration</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    device_id = st.text_input("Device ID", "Client-Laptop-1", help="Identifier for this client machine")
with col_b:
    default_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
    target_url = st.text_input("Central Server API URL", default_url, help="Set via API_URL in your .env file")

col_c, col_d = st.columns(2)
with col_c:
    region = st.selectbox("Simulated Geo-Origin", list(REGIONS.keys()))
with col_d:
    lat, lon = REGIONS[region]
    st.markdown(f"""
    <div style="padding-top:1.75rem;">
        <div style="font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:1.2px; color:#475569; font-family:'Inter',sans-serif; margin-bottom:4px;">Coordinates</div>
        <div style="font-size:13px; font-family:'JetBrains Mono',monospace; color:#6366f1;">{lat:.4f}°, {lon:.4f}°</div>
    </div>
    """, unsafe_allow_html=True)

# Connection target preview
st.markdown(f"""
<div style="margin-top:1rem;">
    <div style="font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:#475569; font-family:'Inter',sans-serif; margin-bottom:6px;">Target Endpoint</div>
    <div class="target-url-display">{target_url}</div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# Session state for injection log
# ============================================================================
if "injection_log" not in st.session_state:
    st.session_state.injection_log = []

# ============================================================================
# Payload Grid
# ============================================================================
st.markdown('<div class="section-title">Select Anomaly Payload</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
cols = [col1, col2, col1, col2]

for i, payload in enumerate(PAYLOADS):
    with cols[i]:
        # Severity badge colors
        sev_bg = {
            "CRITICAL": "rgba(239,68,68,0.12)", 
            "HIGH": "rgba(245,158,11,0.12)", 
            "MEDIUM": "rgba(56,189,248,0.12)"
        }.get(payload["severity"], "rgba(99,102,241,0.12)")
        sev_fg = {
            "CRITICAL": "#f87171", 
            "HIGH": "#fbbf24", 
            "MEDIUM": "#38bdf8"
        }.get(payload["severity"], "#a5b4fc")

        st.markdown(f"""
        <div class="payload-card" style="border:1px solid {payload['border']}; background:{payload['accent']};">
            <div class="payload-card-top">
                <div>
                    <div class="payload-label">{payload['icon']} {payload['label']}</div>
                </div>
                <span class="badge" style="background:{sev_bg}; border:1px solid {sev_fg}; color:{sev_fg};">{payload['severity']}</span>
            </div>
            <div class="payload-desc">{payload['description']}</div>
            <div class="payload-metrics">▶ {payload['metrics']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(payload["btn_label"], key=f"btn_{payload['id']}", use_container_width=True):
            with st.spinner(f"Transmitting {payload['label']} payload..."):
                p = payload["fn"](device_id, latitude=lat, longitude=lon)
                result = injector.request_anomaly(p, url=target_url)

            if result:
                st.markdown(f"""
                <div class="result-success">
                    ✓ {payload['label']} payload successfully delivered to inference engine
                </div>""", unsafe_allow_html=True)
                st.toast(f"Payload delivered: {payload['label']}", icon="✅")
                st.session_state.injection_log.append({
                    "icon": "✓",
                    "color": "#4ade80",
                    "msg": f"{payload['label']} → DELIVERED",
                    "device": device_id,
                    "region": region,
                })
            else:
                st.markdown(f"""
                <div class="result-error">
                    ✗ Delivery failed — Is the API online at {target_url}?
                </div>""", unsafe_allow_html=True)
                st.toast(f"Delivery failed: {payload['label']}", icon="❌")
                st.session_state.injection_log.append({
                    "icon": "✗",
                    "color": "#f87171",
                    "msg": f"{payload['label']} → FAILED",
                    "device": device_id,
                    "region": region,
                })

# ============================================================================
# Session Status Strip
# ============================================================================
st.markdown("<hr>", unsafe_allow_html=True)

c_stat1, c_stat2, c_stat3 = st.columns(3)
total = len(st.session_state.injection_log)
success = sum(1 for x in st.session_state.injection_log if x["icon"] == "✓")
failed = total - success

with c_stat1:
    st.metric("Total Injections", total)
with c_stat2:
    st.metric("Successful", success)
with c_stat3:
    st.metric("Failed", failed)

# ============================================================================
# Injection Log
# ============================================================================
if st.session_state.injection_log:
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Session Injection Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-strip">', unsafe_allow_html=True)
    for entry in reversed(st.session_state.injection_log[-10:]):
        st.markdown(f"""
        <div class="injection-log-item">
            <span style="color:{entry['color']}; font-weight:700;">{entry['icon']}</span>
            <span style="color:#94a3b8;">{entry['msg']}</span>
            <span style="color:#334155; margin-left:auto;">{entry['device']} · {entry['region']}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Clear Log", use_container_width=False):
        st.session_state.injection_log = []
        st.rerun()

# ============================================================================
# Footer
# ============================================================================
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; border-top:1px solid rgba(99,102,241,0.08); margin-top:2rem;">
    <div style="font-size:11px; color:#1e293b; font-family:'JetBrains Mono',monospace;">
        Cloud Anomaly Detection Platform · Edge Client Module · v2.0
    </div>
</div>
""", unsafe_allow_html=True)
