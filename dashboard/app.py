"""
Cloud Infrastructure Anomaly Detection Dashboard
Monitors system metrics and anomalies in real-time with visualization and alerting.
"""

import logging
import time
import os
from datetime import datetime
from typing import Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
# streamlit_autorefresh replaced with smooth session_state-based polling

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.forecast import calculate_time_to_breach

# ============================================================================
# Configuration
# ============================================================================
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_PROTOCOL = os.getenv("API_PROTOCOL", "http")
API_BASE_URL = f"{API_PROTOCOL}://{API_HOST}:{API_PORT}"
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "5"))
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
API_RETRY_DELAY = int(os.getenv("API_RETRY_DELAY", "1"))

REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "5000"))
RECORDS_TO_DISPLAY = int(os.getenv("RECORDS_TO_DISPLAY", "10"))
RETENTION_RECORDS_THRESHOLD = int(os.getenv("RETENTION_RECORDS_THRESHOLD", "50"))

MODEL_VERSION_FILE = Path(os.getenv("MODEL_VERSION_FILE", "ml/model_version.txt"))
LAST_RETRAIN_FILE = Path(os.getenv("LAST_RETRAIN_FILE", "ml/last_retrain.txt"))

ANOMALY_CRITICAL_THRESHOLD = float(os.getenv("ANOMALY_CRITICAL_THRESHOLD", "0.7"))
ANOMALY_WARNING_THRESHOLD = float(os.getenv("ANOMALY_WARNING_THRESHOLD", "0.4"))
ANOMALY_INFO_THRESHOLD = float(os.getenv("ANOMALY_INFO_THRESHOLD", "0.2"))

CPU_MEMORY_MAX = 100
METRIC_MIN = 0

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Clean Light Theme CSS — White Background, Professional Enterprise Style
# ============================================================================
DASHBOARD_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* ── App background — pure white with very faint grid ── */
div[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
    background-image:
        linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px);
    background-size: 40px 40px, 40px 40px;
    min-height: 100vh;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* ── Header bar ── */
div[data-testid="stHeader"] {
    background: rgba(255,255,255,0.95) !important;
    border-bottom: 1px solid #e2e8f0 !important;
    backdrop-filter: blur(12px);
}

/* ── Block container ── */
div[data-testid="block-container"] {
    padding: 1.75rem 2.5rem 3rem !important;
    max-width: 1600px;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 1px 6px rgba(15,23,42,0.06) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}

div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

div[data-testid="metric-container"]:hover::before { opacity: 1; }

div[data-testid="metric-container"]:hover {
    border-color: #bfdbfe !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.1) !important;
    transform: translateY(-2px) !important;
}

/* Metric label */
div[data-testid="stMetricLabel"] > div {
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Metric value */
div[data-testid="stMetricValue"] > div {
    color: #0f172a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.875rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}

/* Metric delta */
div[data-testid="stMetricDelta"] > div {
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #e2e8f0 !important;
    gap: 3px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    padding: 9px 18px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #475569 !important;
    background: #ffffff !important;
}

.stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    background: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.08) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #3b82f6 !important;
    border: 1px solid #2563eb !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 1px 3px rgba(59,130,246,0.3) !important;
}

.stButton > button:hover {
    background: #2563eb !important;
    border-color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.05) !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.04) !important;
}

details[data-testid="stExpanderDetails"] {
    background: #f8fafc !important;
}

/* ── Code blocks ── */
.stCode, code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e40af !important;
}

/* ── Input fields ── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiSelect > div > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}

h1 { font-size: 1.875rem !important; }
h2 { font-size: 1.375rem !important; }
h3 { font-size: 1.125rem !important; }

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Captions ── */
div[data-testid="stCaption"] p {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #3b82f6 !important;
    border-right-color: transparent !important;
}


/* ══════════════════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════════════════ */

/* KPI card */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 6px rgba(15,23,42,0.06);
}

.kpi-card:hover {
    border-color: #bfdbfe;
    box-shadow: 0 6px 24px rgba(59,130,246,0.1);
    transform: translateY(-2px);
}

.kpi-card-accent {
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    border-radius: 14px 0 0 14px;
}

.kpi-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 1.875rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    color: #0f172a;
}

.kpi-sub {
    font-size: 11px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.25rem;
}

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #f1f5f9;
}

.section-header-icon {
    width: 34px;
    height: 34px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
}

.section-header-text {
    font-size: 0.9375rem;
    font-weight: 700;
    color: #0f172a;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.2px;
}

.section-header-sub {
    font-size: 11px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.4px;
}

.badge-success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
}

.badge-danger {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
}

.badge-warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #d97706;
}

.badge-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
}

/* Live dot */
.live-dot {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #16a34a;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.4px;
}

.live-dot::before {
    content: '';
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 0 2px #dcfce7;
    animation: live-blink 1.5s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes live-blink {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 2px #dcfce7; }
    50% { opacity: 0.5; box-shadow: 0 0 0 4px #dcfce7; }
}

/* Alert banners */
.alert-critical {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}

.alert-warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

.alert-success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

.alert-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

/* Countermeasure card */
.countermeasure-card {
    background: linear-gradient(135deg, #faf5ff, #eff6ff);
    border: 1px solid #ddd6fe;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}

.countermeasure-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0%, rgba(139,92,246,0.04) 50%, transparent 100%);
    animation: rotate-bg 8s linear infinite;
}

@keyframes rotate-bg {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Anomaly record */
.anomaly-record {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.anomaly-record:hover {
    background: #fee2e2;
    border-color: #fca5a5;
}

.normal-record {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* Jail card */
.jail-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}

.jail-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.jail-device-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9375rem;
    font-weight: 700;
    color: #dc2626;
}

/* Spinner override */
.stSpinner > div {
    border-color: #3b82f6 !important;
    border-right-color: transparent !important;
}

/* Page title area */
.page-title-area {
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e2e8f0;
}

.page-title {
    font-size: 1.625rem;
    font-weight: 800;
    color: #0f172a;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.page-subtitle {
    font-size: 12px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.25rem;
}

.page-meta {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}

/* Metric strip */
.metric-strip {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.875rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(15,23,42,0.04);
}

.metric-strip-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.metric-strip-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
}

.metric-strip-value {
    font-size: 13px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #1e293b;
}
</style>
"""


# ============================================================================
# HELPER: Plotly Light Theme
# ============================================================================
PLOTLY_DARK_THEME = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,250,252,0.5)",
    font=dict(family="Inter, sans-serif", color="#64748b", size=12),
    xaxis=dict(
        gridcolor="rgba(148,163,184,0.15)",
        linecolor="#e2e8f0",
        tickcolor="#cbd5e1",
        zerolinecolor="#e2e8f0",
    ),
    yaxis=dict(
        gridcolor="rgba(148,163,184,0.15)",
        linecolor="#e2e8f0",
        tickcolor="#cbd5e1",
        zerolinecolor="#e2e8f0",
    ),
    margin=dict(l=16, r=16, t=32, b=16),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
    ),
)


# ============================================================================
# Helper Functions
# ============================================================================

def load_file_content(file_path: Path) -> Optional[str]:
    try:
        if file_path.exists():
            return file_path.read_text().strip()
        return None
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None


def validate_dataframe(df: pd.DataFrame) -> bool:
    required_columns = {"prediction", "cpu_usage", "memory_usage", "disk_io", "network_traffic"}
    return all(col in df.columns for col in required_columns)


@st.cache_data(ttl=30, show_spinner=False)
def check_api_health() -> Tuple[bool, str]:
    """Cached API health check — result is reused for up to 30 s across re-runs."""
    health_url = f"{API_BASE_URL}/health"
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            logger.info(f"Checking API health at {health_url} (attempt {attempt}/{API_RETRY_ATTEMPTS})")
            response = requests.get(health_url, timeout=API_TIMEOUT)
            response.raise_for_status()
            health_data = response.json()
            status = health_data.get("status", "unknown")
            if status in ["healthy", "degraded"]:
                logger.info(f"API health check passed: {status}")
                return True, f"API healthy ({status})"
            else:
                logger.warning(f"API health check failed: {status}")
                return False, f"API unhealthy ({status})"
        except requests.exceptions.Timeout as e:
            error_msg = f"Health check timeout after {API_TIMEOUT}s (attempt {attempt})"
            logger.warning(error_msg)
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY)
                continue
            return False, error_msg
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to API (attempt {attempt})"
            logger.warning(error_msg)
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY)
                continue
            return False, error_msg
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    return False, f"API health check failed after {API_RETRY_ATTEMPTS} attempts"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_predictions_data() -> Optional[pd.DataFrame]:
    """Cached predictions fetch — result is reused for up to 30 s across re-runs."""
    predictions_url = f"{API_BASE_URL}/predictions"
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            logger.info(f"Fetching predictions from {predictions_url} (attempt {attempt}/{API_RETRY_ATTEMPTS})")
            response = requests.get(predictions_url, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "predictions" in data:
                predictions = data["predictions"]
            elif isinstance(data, list):
                predictions = data
            else:
                logger.warning(f"Unexpected API response format: {type(data)}")
                predictions = []
            if not predictions:
                logger.warning("API returned empty predictions list")
                return pd.DataFrame()
            df = pd.DataFrame(predictions)
            if not validate_dataframe(df):
                logger.error("DataFrame validation failed")
                return pd.DataFrame()
            logger.info(f"Successfully fetched {len(df)} records on attempt {attempt}")
            return df
        except requests.exceptions.Timeout:
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY)
                continue
            return None
        except requests.exceptions.ConnectionError:
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY)
                continue
            return None
        except Exception as e:
            logger.error(f"Failed to fetch predictions: {str(e)}")
            return None
    return None


@st.cache_data(ttl=5)
def calculate_metrics(df_len: int, anomaly_count: int) -> Tuple[float, float]:
    anomaly_ratio = anomaly_count / df_len if df_len > 0 else 0
    stability_score = round((1 - anomaly_ratio) * 100, 2)
    confidence_score = round(100 - (abs(50 - stability_score) * 1.2), 2)
    confidence_score = max(0, confidence_score)
    return stability_score, confidence_score


def get_status_info(anomaly_ratio: float) -> Tuple[str, str, str]:
    if anomaly_ratio > ANOMALY_CRITICAL_THRESHOLD:
        return "CRITICAL", "#ef4444", "Immediate attention required — critical anomaly rate detected"
    elif anomaly_ratio > ANOMALY_WARNING_THRESHOLD:
        return "WARNING", "#f59e0b", "Moderate anomaly activity — monitor closely"
    elif anomaly_ratio > ANOMALY_INFO_THRESHOLD:
        return "ELEVATED", "#38bdf8", "Increased anomaly pattern observed"
    else:
        return "HEALTHY", "#22c55e", "System operating within normal parameters"


def _prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if 'cause' not in df.columns:
        df['cause'] = "—"
    else:
        df['cause'] = df['cause'].fillna("—")
    if 'remediation' not in df.columns:
        df['remediation'] = "None"
    else:
        df['remediation'] = df['remediation'].fillna("None")
    column_order = ['id', 'device_id', 'prediction', 'cause', 'remediation', 'cpu_usage', 'memory_usage', 'disk_io', 'network_traffic']
    available_cols = [col for col in column_order if col in df.columns]
    if 'timestamp' in df.columns:
        available_cols.append('timestamp')
    df = df[available_cols]
    rename_mapping = {
        'id': 'ID', 'device_id': 'Device', 'prediction': 'Status',
        'cause': 'Cause', 'remediation': 'Remediation',
        'cpu_usage': 'CPU %', 'memory_usage': 'Memory %',
        'disk_io': 'Disk I/O', 'network_traffic': 'Network',
        'timestamp': 'Timestamp'
    }
    df = df.rename(columns={k: v for k, v in rename_mapping.items() if k in df.columns})
    return df


def _style_anomalies(row):
    if row.get('Status') == 'Anomaly':
        return ['background-color: #fef2f2; color: #b91c1c; font-weight: 600;'] * len(row)
    else:
        return ['background-color: #f0fdf4; color: #166534;'] * len(row)


def generate_mock_llm_post_mortem(row: pd.Series) -> str:
    cause = row.get("cause", "Unknown Anomaly")
    get_val = lambda key, default: row.get(key, default) if hasattr(row, 'get') else getattr(row, key, default)
    cpu = get_val("cpu_usage", 0)
    ram = get_val("memory_usage", 0)
    disk = get_val("disk_io", 0)
    net = get_val("network_traffic", 0)
    remediation = get_val("remediation", "None")
    device_id = get_val("device_id", "Unknown Device")
    timestamp = get_val("timestamp", "Unknown Time")
    cpu_status = "(Critical)" if cpu > 85 else "(Nominal)"
    ram_status = "(Heap Overflow / Memory Leak)" if ram > 85 else "(Nominal)"
    net_status = "(Potential DDoS Flood)" if net > 800 else "(Nominal)"
    disk_status = "(Hardware Degradation)" if disk > 400 else "(Nominal)"
    return f"""
**Executive Summary:**
At `{timestamp}`, the Anomaly Detection ML engine intercepted a major infrastructural irregularity originating from `{device_id}`. The signature correlates with a **[{cause}]** threat vector.

**Telemetry Analysis:**
- **CPU Load:** `{cpu}%` {cpu_status}
- **Memory Saturation:** `{ram}%` {ram_status}
- **Network Ingress:** `{net} Mbps` {net_status}
- **Disk I/O:** `{disk} MB/s` {disk_status}

**Automated Remediation Log:**
> `{remediation}`

**Recommended Engineering Steps:**
1. Investigate access logs for `{device_id}` at the incident timestamp.
2. Review codebase for resource leaks or missing auto-scaling policies.
3. Adjust Isolation Forest confidence bounds if this is a false positive.

*Report dynamically generated by local GenAI Engine.*
"""


# ============================================================================
# Render: Page Header
# ============================================================================
def render_page_header(df: Optional[pd.DataFrame] = None):
    st.markdown(DASHBOARD_STYLES, unsafe_allow_html=True)

    now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    version = load_file_content(MODEL_VERSION_FILE) or "v1.0"

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"""
        <div class="page-title-area">
            <div class="page-title">🛡️ Self Federated Cloud Anomaly Detection System</div>
            <div class="page-subtitle">Collaborative ML-powered cloud-native security intelligence</div>
            <div class="page-meta">
                <span class="live-dot">SECURE NODE ACTIVE</span>
                <span class="badge badge-info">Model {version}</span>
                <span style="font-size:11px; color:#94a3b8; font-family:'JetBrains Mono',monospace;">{now}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if df is not None and not df.empty:
            total = len(df)
            anomaly_count = (df["prediction"] == "Anomaly").sum()
            rate = round(anomaly_count / total * 100, 1) if total > 0 else 0
            _, color, _ = get_status_info(anomaly_count / total if total > 0 else 0)
            st.markdown(f"""
            <div style="text-align:right; padding-top:1rem;">
                <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; font-family:'Inter',sans-serif;">System Health</div>
                <div style="font-size:2.5rem; font-weight:800; color:{color}; font-family:'JetBrains Mono',monospace; line-height:1;">{100-rate:.0f}%</div>
                <div style="font-size:11px; color:#94a3b8; font-family:'JetBrains Mono',monospace;">stability score</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Render: KPI Strip
# ============================================================================
def render_kpi_strip(df: pd.DataFrame):
    total = len(df)
    anomaly_count = int((df["prediction"] == "Anomaly").sum())
    normal_count = total - anomaly_count
    anomaly_rate = round(anomaly_count / total * 100, 1) if total > 0 else 0
    avg_cpu = round(df["cpu_usage"].mean(), 1)
    avg_mem = round(df["memory_usage"].mean(), 1)

    cols = st.columns(5)
    kpi_data = [
        ("Total Events", f"{total:,}", "#2563eb", "All telemetry records"),
        ("Anomalies", f"{anomaly_count:,}", "#dc2626", f"{anomaly_rate}% of traffic"),
        ("Normal", f"{normal_count:,}", "#16a34a", "Clean signals"),
        ("Avg CPU", f"{avg_cpu}%", "#7c3aed", "Last batch"),
        ("Avg Memory", f"{avg_mem}%", "#ea580c", "Last batch"),
    ]
    for col, (label, value, color, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-card-accent" style="background: linear-gradient(180deg, {color}, transparent);"></div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color: {color};">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Render: Status Alert Banner
# ============================================================================
def render_status_banner(anomaly_ratio: float):
    status, color, message = get_status_info(anomaly_ratio)
    if anomaly_ratio > ANOMALY_CRITICAL_THRESHOLD:
        css_cls = "alert-critical"
        icon = "🔴"
    elif anomaly_ratio > ANOMALY_WARNING_THRESHOLD:
        css_cls = "alert-warning"
        icon = "🟡"
    elif anomaly_ratio > ANOMALY_INFO_THRESHOLD:
        css_cls = "alert-info"
        icon = "🔵"
    else:
        css_cls = "alert-success"
        icon = "🟢"

    st.markdown(f"""
    <div class="{css_cls}" style="margin: 1rem 0;">
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <span style="font-size:1.2rem;">{icon}</span>
            <div>
                <div style="font-weight:700; font-size:13px; font-family:'Inter',sans-serif; color:{'#b91c1c' if anomaly_ratio > ANOMALY_CRITICAL_THRESHOLD else '#0f172a'};">SYSTEM STATUS: {status}</div>
                <div style="font-size:12px; color:#64748b; font-family:'JetBrains Mono',monospace; margin-top:2px;">{message}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Render: Stability Gauge
# ============================================================================
def render_stability_gauge(stability_score: float):
    color_val = "#22c55e" if stability_score >= 70 else "#f59e0b" if stability_score >= 40 else "#ef4444"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stability_score,
        number={"font": {"size": 52, "color": color_val, "family": "JetBrains Mono"}, "suffix": "%"},
        title={"text": "AI Stability Score", "font": {"size": 13, "color": "#64748b", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#cbd5e1", "tickfont": {"color": "#94a3b8", "size": 10}},
            "bar": {"color": color_val, "thickness": 0.2},
            "bgcolor": "rgba(248,250,252,0.0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.08)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.07)"},
                {"range": [70, 100], "color": "rgba(34,197,94,0.07)"},
            ],
            "threshold": {
                "line": {"color": color_val, "width": 3},
                "thickness": 0.8,
                "value": stability_score,
            },
        }
    ))
    gauge.update_layout(**PLOTLY_DARK_THEME, height=240)
    st.plotly_chart(gauge, use_container_width=True)


# ============================================================================
# Render: Metric Charts
# ============================================================================
def render_analysis_charts(df: pd.DataFrame, max_chart_records: int = 150):
    df_chart = df.tail(max_chart_records).copy().reset_index(drop=True)
    df_chart['index'] = range(len(df_chart))
    df_chart['is_anomaly'] = (df_chart['prediction'] == 'Anomaly').astype(int)

    c1, c2 = st.columns(2)

    # CPU + Memory
    with c1:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📊</div>
            <div>
                <div class="section-header-text">CPU & Memory Trend</div>
                <div class="section-header-sub">Last 150 records</div>
            </div>
        </div>""", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart['index'], y=df_chart['cpu_usage'],
            name='CPU %', line=dict(color='#2563eb', width=2),
            fill='tozeroy', fillcolor='rgba(37,99,235,0.07)',
        ))
        fig.add_trace(go.Scatter(
            x=df_chart['index'], y=df_chart['memory_usage'],
            name='Memory %', line=dict(color='#7c3aed', width=2, dash='dot'),
            fill='tozeroy', fillcolor='rgba(124,58,237,0.04)',
        ))
        fig.update_layout(**PLOTLY_DARK_THEME, height=240, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Network + Disk
    with c2:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📡</div>
            <div>
                <div class="section-header-text">Network & Disk I/O</div>
                <div class="section-header-sub">Last 150 records</div>
            </div>
        </div>""", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_chart['index'], y=df_chart['network_traffic'],
            name='Network', line=dict(color='#ea580c', width=2),
            fill='tozeroy', fillcolor='rgba(234,88,12,0.06)',
        ))
        fig2.add_trace(go.Scatter(
            x=df_chart['index'], y=df_chart['disk_io'],
            name='Disk I/O', line=dict(color='#34d399', width=1.5, dash='dot'),
        ))
        fig2.update_layout(**PLOTLY_DARK_THEME, height=240, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    # Anomaly distribution
    with c3:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🎯</div>
            <div>
                <div class="section-header-text">Detection Distribution</div>
                <div class="section-header-sub">Anomaly vs. Normal ratio</div>
            </div>
        </div>""", unsafe_allow_html=True)
        counts = df['prediction'].value_counts()
        fig3 = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.6,
            marker=dict(
                colors=['#ef4444', '#22c55e'] if 'Anomaly' in counts.index else ['#22c55e'],
                line=dict(color='rgba(14,20,35,0.8)', width=2),
            ),
            textfont=dict(family='Inter', size=12, color='#cbd5e1'),
        ))
        fig3.update_layout(**PLOTLY_DARK_THEME, height=240, showlegend=True)
        fig3.add_annotation(
            text=f"{len(df):,}<br><span style='font-size:10px'>events</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color='#f1f5f9', family='JetBrains Mono'),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Anomaly over time
    with c4:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📈</div>
            <div>
                <div class="section-header-text">Anomaly Pulse</div>
                <div class="section-header-sub">Occurrence over time</div>
            </div>
        </div>""", unsafe_allow_html=True)
        fig4 = go.Figure(go.Bar(
            x=df_chart['index'],
            y=df_chart['is_anomaly'],
            marker=dict(
                color=df_chart['is_anomaly'].apply(lambda v: 'rgba(239,68,68,0.7)' if v == 1 else 'rgba(34,197,94,0.3)'),
                line=dict(width=0),
            ),
            name='Anomaly',
        ))
        fig4.update_layout(**PLOTLY_DARK_THEME, height=240, showlegend=False)
        fig4.update_yaxes(tickvals=[0, 1], ticktext=["Normal", "Anomaly"])
        st.plotly_chart(fig4, use_container_width=True)


# ============================================================================
# Render: Countermeasures
# ============================================================================
def render_countermeasures_status(df: pd.DataFrame):
    anomalies = df[df['prediction'] == 'Anomaly']
    if anomalies.empty:
        st.markdown("""
        <div class="alert-success">
            <div style="font-weight:600; font-size:13px; font-family:'Inter',sans-serif; color:#4ade80;">Active Countermeasures System</div>
            <div style="font-size:12px; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:4px;">SCANNING... — No active threats detected. All systems nominal.</div>
        </div>""", unsafe_allow_html=True)
        return
    latest = anomalies.iloc[-1]
    remediation = latest.get("remediation", "None") or "Monitoring escalated"
    cause = latest.get("cause", "Unknown") or "Unknown threat vector"
    device = latest.get("device_id", "Unknown")
    st.markdown(f"""
    <div class="countermeasure-card">
        <div style="position:relative; z-index:1;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.75rem;">
                <span class="badge badge-danger">⚡ COUNTERMEASURE ACTIVE</span>
            </div>
            <div style="font-size:14px; font-weight:700; color:#c4b5fd; font-family:'JetBrains Mono',monospace; margin-bottom:0.5rem;">{remediation}</div>
            <div style="font-size:11px; color:#475569; font-family:'JetBrains Mono',monospace;">Triggered by: <span style="color:#94a3b8;">{cause}</span> — Device: <span style="color:#94a3b8;">{device}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Render: AI Prognosis
# ============================================================================
def render_ai_prognosis_panel(df: pd.DataFrame):
    st.markdown("""
    <div class="section-header" style="margin-top:1.5rem;">
        <div class="section-header-icon">🔮</div>
        <div>
            <div class="section-header-text">AI Predictive Prognosis</div>
            <div class="section-header-sub">Estimated time-to-failure based on current trajectory</div>
        </div>
    </div>""", unsafe_allow_html=True)

    cpu_vals = df['cpu_usage'].tail(10).tolist()
    cpu_time = calculate_time_to_breach(cpu_vals)
    ram_vals = df['memory_usage'].tail(10).tolist()
    ram_time = calculate_time_to_breach(ram_vals)

    c1, c2 = st.columns(2)
    with c1:
        if cpu_time is not None:
            if cpu_time == 0:
                st.markdown("""<div class="alert-critical"><b style="color:#f87171">CPU LIMIT BREACHED</b> — Critical processing saturation detected. Immediate action required.</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="alert-warning"><b>CPU WARNING</b> — Estimated breach in ~<b>{cpu_time:.0f}s</b> at current trajectory.</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-success"><b style="color:#4ade80">CPU STABLE</b> — Trajectory is healthy. No breach forecast.</div>""", unsafe_allow_html=True)
    with c2:
        if ram_time is not None:
            if ram_time == 0:
                st.markdown("""<div class="alert-critical"><b style="color:#f87171">MEMORY SATURATED</b> — System RAM has reached critical threshold.</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="alert-warning"><b>MEMORY WARNING</b> — Estimated exhaustion in ~<b>{ram_time:.0f}s</b>.</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-success"><b style="color:#4ade80">MEMORY STABLE</b> — No memory exhaustion forecast detected.</div>""", unsafe_allow_html=True)


# ============================================================================
# Render: Recent Records Table
# ============================================================================
def render_recent_records(df: Optional[pd.DataFrame] = None):
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📋</div>
        <div>
            <div class="section-header-text">Inference Feed</div>
            <div class="section-header-sub">Most recent telemetry records with predictions</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if df is None or df.empty:
        st.markdown("""<div class="alert-info">Waiting for data from the API...</div>""", unsafe_allow_html=True)
        return

    display_df = df.tail(RECORDS_TO_DISPLAY).copy()
    display_df = _prepare_display_dataframe(display_df)
    styled_df = display_df.style.apply(_style_anomalies, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=320)

    anomaly_mask = df["prediction"] == "Anomaly"
    anomaly_count = anomaly_mask.sum()

    if anomaly_count > 0:
        st.markdown("""
        <div class="section-header" style="margin-top:2rem;">
            <div class="section-header-icon">🚨</div>
            <div>
                <div class="section-header-text">Anomaly Detail Roll-Up</div>
                <div class="section-header-sub">3 most recent detections — AI analysis available</div>
            </div>
        </div>""", unsafe_allow_html=True)

        anomalies = df[anomaly_mask].iloc[-3:]
        for idx, row in anomalies.iterrows():
            cause = row.get("cause", "") or "Anomaly detected based on learned patterns"
            if pd.isna(cause) or cause == "":
                cause = "Anomaly detected based on learned patterns"
            rec_id = row.get('id', idx)

            with st.expander(f"Anomaly #{rec_id} — {cause}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("CPU", f"{row['cpu_usage']:.1f}%")
                with c2:
                    st.metric("Memory", f"{row['memory_usage']:.1f}%")
                with c3:
                    st.metric("Disk I/O", f"{row['disk_io']:.1f}")
                with c4:
                    st.metric("Network", f"{row['network_traffic']:.1f}")

                if 'timestamp' in row and not pd.isna(row['timestamp']):
                    st.caption(f"Detected at: {row['timestamp']}")

                if st.button("Generate AI Incident Report", key=f"btn_ai_{rec_id}"):
                    st.session_state.ai_report_active = True
                    st.session_state.active_report_id = rec_id
                    with st.spinner("AI Analysis Engine parsing telemetry signature..."):
                        time.sleep(1.0)
                        report = generate_mock_llm_post_mortem(row)
                        st.markdown(f"""
                        <div class="alert-info">
                            <pre style="white-space:pre-wrap; font-family:'JetBrains Mono',monospace; font-size:12px; color:#94a3b8; margin:0;">{report}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                
                # If we are looking at this expander and it's expanded, and report was clicked,
                # we are in "AI Insight" mode. 
                # Note: Streamlit buttons are transient. We'll use the flag set above.


# ============================================================================
# Render: Threat Map
# ============================================================================
def render_geographical_threat_map(df: pd.DataFrame):
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🌍</div>
        <div>
            <div class="section-header-text">Global Threat Intelligence</div>
            <div class="section-header-sub">Real-time geo-tracking of anomaly origin vectors</div>
        </div>
    </div>""", unsafe_allow_html=True)

    anomalies = df[(df['prediction'] == 'Anomaly') & (df['latitude'] != 0) & (df['longitude'] != 0)].copy()

    if anomalies.empty:
        st.markdown("""<div class="alert-info">Scanning for external threat signatures — no active geo-tagged anomalies detected.</div>""", unsafe_allow_html=True)
        view_state = pdk.ViewState(latitude=20, longitude=0, zoom=0.8, pitch=45, bearing=0)
        st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/dark-v10', initial_view_state=view_state))
        return

    anomalies['radius'] = 120000
    anomalies['elevation'] = 50000

    scatter_layer = pdk.Layer(
        "ScatterplotLayer", anomalies,
        get_position=["longitude", "latitude"],
        get_color="[239, 68, 68, 180]",
        get_radius="radius",
        pickable=True,
        stroked=True,
        line_width_min_pixels=2,
        get_line_color="[248, 113, 113, 255]",
    )
    column_layer = pdk.Layer(
        "ColumnLayer", anomalies,
        get_position=["longitude", "latitude"],
        get_elevation="elevation",
        elevation_scale=0.5,
        radius=60000,
        get_fill_color="[239, 68, 68, 120]",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=anomalies['latitude'].mean(),
        longitude=anomalies['longitude'].mean(),
        zoom=1.5, pitch=50, bearing=10,
    )

    r = pdk.Deck(
        layers=[column_layer, scatter_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={"text": "Device: {device_id}\nThreat: {cause}\nAlert ID: #{id}"},
    )
    st.pydeck_chart(r)

    st.markdown("""
    <div class="section-header" style="margin-top:1.5rem;">
        <div class="section-header-icon">📍</div>
        <div>
            <div class="section-header-text">Active Breach Locations</div>
        </div>
    </div>""", unsafe_allow_html=True)
    display = anomalies[['id', 'device_id', 'latitude', 'longitude', 'cause']].rename(
        columns={'id': 'Alert ID', 'device_id': 'Device', 'cause': 'Threat Vector'})
    st.dataframe(display, use_container_width=True)


# ============================================================================
# Render: Cyber Jail
# ============================================================================
def render_cyber_jail():
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">⛓️</div>
        <div>
            <div class="section-header-text">Zero-Trust Cyber Jail</div>
            <div class="section-header-sub">Automatically quarantined devices — review and release</div>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        response = requests.get(f"{API_BASE_URL}/quarantine", timeout=5)
        if response.status_code == 200:
            jailed = response.json()
            if not jailed:
                st.markdown("""
                <div class="alert-success">
                    <div style="font-weight:600; font-size:13px; font-family:'Inter',sans-serif; color:#4ade80;">Cyber Jail is Empty</div>
                    <div style="font-size:12px; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:4px;">No devices are currently quarantined. Internal security perimeter is clean.</div>
                </div>""", unsafe_allow_html=True)
                return

            st.markdown(f"""
            <div style="margin-bottom:1rem;">
                <span class="badge badge-danger">{len(jailed)} Quarantined Device{'s' if len(jailed) != 1 else ''}</span>
            </div>""", unsafe_allow_html=True)

            for device in jailed:
                dev_id = device.get("device_id", "Unknown")
                reason = device.get("reason", "Unknown")
                timestamp = device.get("timestamp", "Unknown")

                c_info, c_btn = st.columns([4, 1])
                with c_info:
                    st.markdown(f"""
                    <div class="jail-card">
                        <div class="jail-card-header">
                            <div>
                                <div class="jail-device-id">⛓ {dev_id}</div>
                                <div style="font-size:11px; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:2px;">Quarantined: {timestamp}</div>
                            </div>
                            <span class="badge badge-danger">BLOCKED</span>
                        </div>
                        <div style="font-size:12px; color:#94a3b8; font-family:'Inter',sans-serif;"><b>Reason:</b> {reason}</div>
                    </div>""", unsafe_allow_html=True)
                with c_btn:
                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                    if st.button("Pardon", key=f"release_{dev_id}"):
                        rel_resp = requests.post(f"{API_BASE_URL}/quarantine/release/{dev_id}", timeout=5)
                        if rel_resp.status_code == 200:
                            st.toast(f"Device {dev_id} has been pardoned. Network access restored.", icon="🔓")
                            st.rerun()
                        else:
                            st.error("Failed to release device from quarantine.")
        else:
            st.markdown("""<div class="alert-warning">Could not reach the quarantine API endpoint.</div>""", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""<div class="alert-warning">Error loading Cyber Jail data: <code>{e}</code></div>""", unsafe_allow_html=True)


# ============================================================================
# Render: System Logs
# ============================================================================
def render_system_logs():
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📜</div>
        <div>
            <div class="section-header-text">System Log Viewer</div>
            <div class="section-header-sub">Real-time internal application audit trail</div>
        </div>
    </div>""", unsafe_allow_html=True)

    log_file_path = Path("logs/system.log")
    if not log_file_path.exists():
        st.markdown("""<div class="alert-info">System log file not found. Logs will appear once the system starts generating events.</div>""", unsafe_allow_html=True)
        return

    try:
        with open(log_file_path, "r") as f:
            logs = f.readlines()

        if not logs:
            st.markdown("""<div class="alert-info">System log file is currently empty.</div>""", unsafe_allow_html=True)
            return

        parsed_logs = []
        for line in reversed(logs[-2000:]):
            parts = line.strip().split(' - ', 3)
            if len(parts) == 4:
                parsed_logs.append({"Timestamp": parts[0], "Component": parts[1], "Severity": parts[2], "Message": parts[3]})
            else:
                parsed_logs.append({"Timestamp": "", "Component": "", "Severity": "UNKNOWN", "Message": line.strip()})

        df_logs = pd.DataFrame(parsed_logs)

        c1, c2 = st.columns([2, 1])
        with c1:
            search_term = st.text_input("Search logs...", "", placeholder="Filter by keyword...")
        with c2:
            severity_filter = st.multiselect(
                "Severity",
                ["INFO", "WARNING", "ERROR", "CRITICAL", "UNKNOWN"],
                default=["INFO", "WARNING", "ERROR", "CRITICAL"],
            )

        if search_term:
            df_logs = df_logs[df_logs["Message"].str.contains(search_term, case=False, na=False)]
        if severity_filter:
            df_logs = df_logs[df_logs["Severity"].isin(severity_filter)]

        def style_logs(row):
            sev = row["Severity"]
            if sev in ("ERROR", "CRITICAL"):
                return ['background-color: rgba(239,68,68,0.12); color:#fca5a5;'] * len(row)
            elif sev == "WARNING":
                return ['background-color: rgba(245,158,11,0.1); color:#fde68a;'] * len(row)
            return ['color:#94a3b8;'] * len(row)

        st.dataframe(df_logs.style.apply(style_logs, axis=1), use_container_width=True, height=560)

    except Exception as e:
        st.markdown(f"""<div class="alert-warning">Failed to read system logs: <code>{e}</code></div>""", unsafe_allow_html=True)


# ============================================================================
# Render: Alert History
# ============================================================================
def render_alert_history(df: Optional[pd.DataFrame]):
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🚨</div>
        <div>
            <div class="section-header-text">Anomaly Alert History</div>
            <div class="section-header-sub">Complete historical record of all detected anomaly events</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if df is None or df.empty:
        st.markdown("""<div class="alert-info">No prediction data available yet.</div>""", unsafe_allow_html=True)
        return

    anomalies = df[df['prediction'] == 'Anomaly'].copy()

    if anomalies.empty:
        st.markdown("""<div class="alert-success">No anomalies detected — the system is operating perfectly.</div>""", unsafe_allow_html=True)
        return

    display_df = _prepare_display_dataframe(anomalies)
    if 'ID' in display_df.columns:
        display_df = display_df.sort_values(by='ID', ascending=False)

    def style_alerts(row):
        return ['background-color: rgba(239,68,68,0.1); color:#fca5a5;'] * len(row)

    st.dataframe(display_df.style.apply(style_alerts, axis=1), use_container_width=True, height=560)


# ============================================================================
# Render: Offline / Fallback
# ============================================================================
def render_fallback_dashboard():
    st.markdown(f"""
    <div class="alert-info" style="margin-bottom:1.5rem;">
        <div style="font-weight:700; font-size:13px; font-family:'Inter',sans-serif; color:#38bdf8; margin-bottom:6px;">Quick Setup Guide</div>
        <div style="font-size:12px; color:#64748b; font-family:'JetBrains Mono',monospace; line-height:1.8;">
            <b style="color:#94a3b8;"># Terminal 1</b><br>
            python -m uvicorn backend.main:app --host {API_HOST} --port {API_PORT} --reload<br><br>
            <b style="color:#94a3b8;"># Terminal 2</b><br>
            streamlit run dashboard/app.py<br><br>
            <b style="color:#94a3b8;"># Terminal 3</b><br>
            python simulator/live_simulator.py
        </div>
    </div>
    <div class="metric-strip">
        <div class="metric-strip-item"><div class="metric-strip-label">API Host</div><div class="metric-strip-value">{API_HOST}</div></div>
        <div class="metric-strip-item"><div class="metric-strip-label">API Port</div><div class="metric-strip-value">{API_PORT}</div></div>
        <div class="metric-strip-item"><div class="metric-strip-label">Protocol</div><div class="metric-strip-value">{API_PROTOCOL.upper()}</div></div>
        <div class="metric-strip-item"><div class="metric-strip-label">Timeout</div><div class="metric-strip-value">{API_TIMEOUT}s</div></div>
        <div class="metric-strip-item"><div class="metric-strip-label">Refresh</div><div class="metric-strip-value">{REFRESH_INTERVAL}ms</div></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================
def _render_refresh_sidebar() -> Tuple[int, str]:
    """Render the refresh-control and navigation panel in the sidebar.
    Returns (interval_seconds, active_page)."""
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem; font-family:'Inter',sans-serif;">
            <div style="font-size:11px; font-weight:700; text-transform:uppercase;
                        letter-spacing:1.5px; color:#94a3b8; margin-bottom:0.75rem;">🧭 Navigation</div>
        </div>
        """, unsafe_allow_html=True)

        active_page = st.radio(
            "Go to",
            ["⚡ Overview", "🌍 Threat Map", "⛓️ Cyber Jail", "📜 System Logs", "🚨 Alert History"],
            label_visibility="collapsed",
            key="nav_radio"
        )

        st.markdown("<hr style='border-color:#e2e8f0; margin:1rem 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:0; font-family:'Inter',sans-serif;">
            <div style="font-size:11px; font-weight:700; text-transform:uppercase;
                        letter-spacing:1.5px; color:#94a3b8; margin-bottom:0.75rem;">⚙️ Refresh Control</div>
        </div>
        """, unsafe_allow_html=True)

        auto_refresh = st.toggle("Auto-refresh", value=True, key="auto_refresh_toggle")

        # Determine if we should force pause the refresh based on user activity
        force_pause = False
        pause_reason = ""
        
        if active_page in ["🌍 Threat Map", "⛓️ Cyber Jail"]:
            force_pause = True
            pause_reason = f"Paused on {active_page.split()[-1]}"
        elif st.session_state.get("ai_report_active"):
            force_pause = True
            pause_reason = "Paused for AI Analysis"

        interval_seconds = 30
        if auto_refresh and not force_pause:
            interval_seconds = st.select_slider(
                "Interval",
                options=[10, 15, 30, 60, 120, 300],
                value=30,
                format_func=lambda s: f"{s}s" if s < 60 else f"{s//60}m",
                key="refresh_interval_slider",
            )
        else:
            if force_pause:
                st.info(f"⏸️ {pause_reason}")
            interval_seconds = 0

        if st.button("🔄 Refresh Now", use_container_width=True, key="manual_refresh_btn"):
            check_api_health.clear()
            fetch_predictions_data.clear()
            st.session_state.ai_report_active = False # Reset on manual refresh
            st.rerun()

        # Last-updated timestamp
        last_updated = st.session_state.get("last_data_fetch_ts")
        if last_updated:
            elapsed = int(time.time() - last_updated)
            st.markdown(
                f"<div style='font-size:11px; color:#94a3b8; font-family:monospace; margin-top:0.5rem;'>"
                f"Last updated: {elapsed}s ago</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#e2e8f0; margin:1rem 0;'>", unsafe_allow_html=True)

    return interval_seconds, active_page


def main():
    st.set_page_config(
        page_title="Self Federated Cloud Anomaly Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"Get Help": None, "Report a bug": None, "About": "Federated IDS Platform v2.0"},
    )

    if 'ai_report_active' not in st.session_state:
        st.session_state.ai_report_active = False

    # ── Smooth polling with conditional pause ──
    interval_seconds, active_page = _render_refresh_sidebar()

    now_ts = time.time()
    last_fetch_ts = st.session_state.get("last_data_fetch_ts", 0)
    needs_refresh = (interval_seconds > 0) and ((now_ts - last_fetch_ts) >= interval_seconds)

    if needs_refresh:
        # Clear cache so the next call actually hits the API
        check_api_health.clear()
        fetch_predictions_data.clear()
        st.session_state["last_data_fetch_ts"] = now_ts
    elif "last_data_fetch_ts" not in st.session_state:
        # Very first run — record the timestamp
        st.session_state["last_data_fetch_ts"] = now_ts

    # Schedule the next wake-up only when auto-refresh is on
    if interval_seconds > 0:
        remaining = max(1, interval_seconds - int(now_ts - st.session_state["last_data_fetch_ts"]))
        # Use a hidden JS meta-refresh equivalent via session_state flag so
        # we don't block the UI thread — we'll rely on Streamlit's own
        # script runner being re-invoked after the page is idle.
        # A lightweight approach: schedule via time.sleep only if very close.
        if remaining <= 2:
            time.sleep(remaining)
            st.rerun()

    # Fetch data (served from cache unless we just cleared it above)
    api_healthy, health_message = check_api_health()
    df = None
    if api_healthy:
        df = fetch_predictions_data()

    # Header
    render_page_header(df)

    # Page Routing
    if active_page == "⚡ Overview":
        if not api_healthy:
            st.markdown(f"""
            <div class="alert-critical">
                <div>
                    <div style="font-weight:700; font-size:13px; font-family:'Inter',sans-serif; color:#f87171;">API Offline — Cannot Connect</div>
                    <div style="font-size:11px; color:#94a3b8; font-family:'JetBrains Mono',monospace; margin-top:4px;">{health_message}</div>
                    <div style="font-size:11px; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:2px;">Target: {API_BASE_URL}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            render_fallback_dashboard()

        elif df is None or df.empty:
            st.markdown("""
            <div class="alert-info">
                <div style="font-weight:600; font-size:13px; font-family:'Inter',sans-serif; color:#38bdf8;">Connected — Awaiting Data</div>
                <div style="font-size:12px; color:#475569; font-family:'JetBrains Mono',monospace; margin-top:4px;">The inference engine is active. Start the live simulator to generate telemetry.</div>
            </div>""", unsafe_allow_html=True)
            st.code("python simulator/live_simulator.py", language="bash")

        else:
            try:
                logger.info(f"Processing {len(df)} records from API")

                # New anomaly toast alerts
                # Initialize last_seen_id to 0 so first refresh surfaces recent anomalies.
                # We track the max ID we've already alerted on in session_state.
                if 'last_seen_id' not in st.session_state:
                    st.session_state.last_seen_id = 0
                    st.session_state.last_alert_df_hash = None

                # Only search for new records when data actually changed (cache busted)
                current_max_id = int(df['id'].max()) if not df.empty else 0
                df_hash = current_max_id  # proxy for "did the data change"

                if df_hash != st.session_state.get('last_alert_df_hash'):
                    st.session_state.last_alert_df_hash = df_hash
                    new_records = df[df['id'] > st.session_state.last_seen_id]
                    if not new_records.empty:
                        anomaly_new = new_records[new_records['prediction'] == 'Anomaly']
                        # Cap toasts at 5 to avoid flooding the UI
                        for _, row in anomaly_new.tail(5).iterrows():
                            device = row.get('device_id', 'Unknown')
                            cause = row.get('cause', 'Unknown pattern')
                            st.toast(f"🚨 Anomaly — {device}: {cause}", icon="🚨")
                    # Always advance the cursor so we don't re-alert
                    st.session_state.last_seen_id = current_max_id

                # Compute scores
                total_records = len(df)
                anomaly_count = int((df["prediction"] == "Anomaly").sum())
                anomaly_ratio = anomaly_count / total_records if total_records > 0 else 0
                stability_score, confidence_score = calculate_metrics(total_records, anomaly_count)

                # Status banner
                render_status_banner(anomaly_ratio)

                # KPI strip
                render_kpi_strip(df)
                st.markdown("<hr>", unsafe_allow_html=True)

                # Main 2-column layout
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    render_analysis_charts(df)

                with col_right:
                    st.markdown("""
                    <div class="section-header">
                        <div class="section-header-icon">🎯</div>
                        <div>
                            <div class="section-header-text">AI Stability Score</div>
                            <div class="section-header-sub">ML engine confidence</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    render_stability_gauge(stability_score)
                    st.metric("Prediction Confidence", f"{confidence_score:.1f}%",
                              delta=f"{confidence_score - 50:.1f}% vs baseline")
                    st.metric("Total Events Analysed", f"{total_records:,}")
                    st.metric("Anomaly Rate", f"{anomaly_ratio*100:.2f}%",
                              delta=f"{anomaly_count} anomalies detected",
                              delta_color="inverse")

                st.markdown("<hr>", unsafe_allow_html=True)

                # Countermeasures + Prognosis
                c_a, c_b = st.columns(2)
                with c_a:
                    render_countermeasures_status(df)
                with c_b:
                    render_ai_prognosis_panel(df)

                st.markdown("<hr>", unsafe_allow_html=True)

                # Recent records + expanders
                render_recent_records(df)

                logger.info("Dashboard rendered successfully")

            except Exception as e:
                logger.error(f"Dashboard rendering error: {e}", exc_info=True)
                st.markdown(f"""<div class="alert-critical"><b>Rendering error:</b> {str(e)}</div>""", unsafe_allow_html=True)

    elif active_page == "🌍 Threat Map":
        if df is not None and not df.empty:
            render_geographical_threat_map(df)
        else:
            st.markdown("""<div class="alert-info">Connect to the API and start the simulator to populate the threat map.</div>""", unsafe_allow_html=True)

    elif active_page == "⛓️ Cyber Jail":
        render_cyber_jail()

    elif active_page == "📜 System Logs":
        render_system_logs()

    elif active_page == "🚨 Alert History":
        render_alert_history(df)


if __name__ == "__main__":
    main()