import os
import json
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timezone
import sys
from pathlib import Path

# Navigate up 2 levels (/frontend/src -> /frontend -> ROOT) then down into /backend/src
BACKEND_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "src"

if str(BACKEND_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC_DIR))

# Now Python can find graph_visualizer in /backend/src
from graph_visualizer import render_fraud_topology


# ==================================================================
# PAGE CONFIGURATION
# ==================================================================
st.set_page_config(
    page_title="GUCE Customs Fraud Control Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================================
# CONFIGURATION ENV VARS
# ==================================================================
# Drift Report
def _resolve_drift_report_path():
    env_override = os.environ.get("DRIFT_REPORT_PATH")
    if env_override:
        return env_override, [env_override]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "..", "..", "backend", "data", "reports", "monitoring", "data_drift_latest.html"),
        os.path.join(this_dir, "..", "..", "backend", "data", "reports", "monitoring", "data_drift_latest.html"),
        r"\\wsl.localhost\Ubuntu\home\mouhe\MLopsPipeline\backend\data\reports\monitoring\data_drift_latest.html",
        "/home/mouhe/MLopsPipeline/backend/data/reports/monitoring/data_drift_latest.html",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path, candidates
    return None, candidates

DRIFT_REPORT_PATH, _DRIFT_PATH_CANDIDATES = _resolve_drift_report_path()

# Grafana
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000").rstrip("/")
GRAFANA_DASHBOARD_NAME = os.environ.get("GRAFANA_DASHBOARD_NAME", "GNN Dashboard")
GRAFANA_API_KEY = os.environ.get("GRAFANA_API_KEY", "") 
GRAFANA_ORG_ID = os.environ.get("GRAFANA_ORG_ID", "1")

# Airflow Integration
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080").rstrip("/")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID", "mlops_fraud_pipeline_v1")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "H7pyvvUTWX4PPPft")

# Predictions path
PREDICTIONS_PATH = "../../backend/data/final_fraud_predictions.csv"


# ==================================================================
# AIRFLOW 3.X HELPERS
# ==================================================================
def trigger_airflow_dag(dag_id: str):
    """Trigger an Apache Airflow 3.x DAG via the REST API v2 using JWT Auth."""
    token_endpoint = f"{AIRFLOW_URL}/auth/token"
    try:
        auth_resp = requests.post(
            token_endpoint,
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if auth_resp.status_code == 401:
            return False, f"Airflow 3 Auth Failed (HTTP 401): Invalid credentials for user '{AIRFLOW_USERNAME}'."
        elif auth_resp.status_code not in (200, 201):
            return False, f"Airflow 3 Auth Error (HTTP {auth_resp.status_code}): {auth_resp.text}"
            
        token = auth_resp.json().get("access_token")
        if not token:
            return False, "Failed to retrieve 'access_token' from Airflow token response."
            
    except requests.exceptions.RequestException as e:
        return False, f"Could not reach Airflow Auth Server ({token_endpoint}): {e}"

    dag_run_endpoint = f"{AIRFLOW_URL}/api/v2/dags/{dag_id}/dagRuns"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "logical_date": datetime.now(timezone.utc).isoformat(),
        "conf": {}
    }

    try:
        response = requests.post(
            dag_run_endpoint,
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code in (200, 201):
            run_data = response.json()
            run_id = run_data.get('dag_run_id') or run_data.get('run_id', 'Triggered')
            return True, f"Successfully triggered DAG '{dag_id}' (Run ID: {run_id})"
        elif response.status_code == 404:
            return False, f"HTTP 404: DAG '{dag_id}' not found on Airflow at {AIRFLOW_URL}. Ensure the DAG is unpaused and active."
        else:
            return False, f"DAG Trigger Failed (HTTP {response.status_code}): {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Network error calling Airflow API v2: {e}"


def get_dag_run_tasks(dag_id: str, dag_run_id: str):
    """Fetch all task instances for a specific DAG run in Airflow 3."""
    token_endpoint = f"{AIRFLOW_URL}/auth/token"
    try:
        auth_resp = requests.post(
            token_endpoint,
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            timeout=5
        )
        token = auth_resp.json().get("access_token")
        
        endpoint = f"{AIRFLOW_URL}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        resp = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("task_instances", [])
    except Exception as e:
        st.error(f"Failed to fetch task instances: {e}")
    return []


def get_task_logs(dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1):
    """Fetch terminal output logs for a specific task instance node in Airflow 3."""
    token_endpoint = f"{AIRFLOW_URL}/auth/token"
    try:
        auth_resp = requests.post(
            token_endpoint,
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            timeout=5
        )
        if auth_resp.status_code not in (200, 201):
            return f"Auth Error ({auth_resp.status_code}): {auth_resp.text}"
            
        token = auth_resp.json().get("access_token")

        endpoint = f"{AIRFLOW_URL}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}"
        resp = requests.get(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json"
            },
            timeout=10
        )
        
        if resp.status_code == 200:
            log_data = resp.json()
            if isinstance(log_data, dict):
                if "content" in log_data:
                    return log_data["content"]
                elif "lines" in log_data:
                    return "\n".join(log_data["lines"])
                elif "log" in log_data:
                    return log_data["log"]
                return json.dumps(log_data, indent=2)
            else:
                return str(log_data)
        else:
            return f"Log endpoint returned HTTP {resp.status_code}: {resp.text}"
            
    except Exception as e:
        return f"Error connecting to log endpoint: {e}"


# ==================================================================
# GRAFANA HELPERS
# ==================================================================
def _grafana_headers():
    headers = {"Accept": "application/json"}
    if GRAFANA_API_KEY:
        headers["Authorization"] = f"Bearer {GRAFANA_API_KEY}"
    return headers

@st.cache_data(ttl=300, show_spinner=False)
def resolve_grafana_dashboard(name: str):
    """Look up a dashboard by title and return (uid, url_slug, error_message)."""
    try:
        resp = requests.get(
            f"{GRAFANA_URL}/api/search",
            params={"query": name, "type": "dash-db"},
            headers=_grafana_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None, None, f"No dashboard found matching '{name}'."
        match = next((r for r in results if r.get("title", "").lower() == name.lower()), results[0])
        return match.get("uid"), match.get("url"), None
    except requests.exceptions.RequestException as e:
        return None, None, f"Could not reach Grafana at {GRAFANA_URL} ({e})."

@st.cache_data(ttl=60, show_spinner=False)
def get_grafana_panels(uid: str):
    """Fetch panel metadata (id + title) for a dashboard, flattening row groups."""
    try:
        resp = requests.get(
            f"{GRAFANA_URL}/api/dashboards/uid/{uid}",
            headers=_grafana_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        panels_raw = resp.json().get("dashboard", {}).get("panels", [])
    except requests.exceptions.RequestException as e:
        return [], f"Could not fetch panel list: {e}"

    panels = []
    for p in panels_raw:
        if p.get("type") == "row":
            panels.extend(p.get("panels", []))
        else:
            panels.append(p)
    panels = [{"id": p.get("id"), "title": p.get("title") or f"Panel {p.get('id')}"} for p in panels if p.get("id") is not None]
    return panels, None

def build_panel_embed_url(dashboard_url_path: str, panel_id: int, refresh: str, time_range: str):
    """Build a d-solo (single panel) embed URL from the dashboard's own /d/<uid>/<slug> path."""
    solo_path = dashboard_url_path.replace("/d/", "/d-solo/", 1)
    from_to = {
        "Last 15m": ("now-15m", "now"),
        "Last 1h": ("now-1h", "now"),
        "Last 6h": ("now-6h", "now"),
        "Last 24h": ("now-24h", "now"),
    }.get(time_range, ("now-6h", "now"))
    url = (
        f"{GRAFANA_URL}{solo_path}"
        f"?orgId={GRAFANA_ORG_ID}&panelId={panel_id}"
        f"&refresh={refresh}&from={from_to[0]}&to={from_to[1]}&theme=light"
    )
    if GRAFANA_API_KEY:
        url += f"&auth_token={GRAFANA_API_KEY}"
    return url


# ==================================================================
# CHART.JS EMBED HELPER
# ==================================================================
def render_chartjs_panel(canvas_id: str, chart_type: str, labels: list, datasets: list, height: int = 300):
    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": len(datasets) > 1, "position": "bottom"}},
            "scales": {
                "x": {"grid": {"display": False, "color": "#e2e8f0"}, "ticks": {"color": "#64748b"}},
                "y": {"grid": {"color": "#e2e8f0"}, "ticks": {"color": "#64748b"}, "beginAtZero": True},
            },
        },
    }
    html = f"""
    <div style="height:{height}px;">
        <canvas id="{canvas_id}"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
    <script>
        const ctx_{canvas_id} = document.getElementById('{canvas_id}').getContext('2d');
        new Chart(ctx_{canvas_id}, {json.dumps(config)});
    </script>
    """
    components.html(html, height=height + 10)


# ==================================================================
# LEDGER CARD-LIST RENDERER
# ==================================================================
def _pick_ledger_columns(df, max_cols=5):
    priority_keywords = [
        "file_id", "id", "dossier", "reference", "numero",
        "emetteur", "exporter", "pays", "country",
        "valeur", "value", "montant", "amount", "fob",
        "score", "prob", "flag", "audit", "fraud",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    picked = []
    for kw in priority_keywords:
        for lc, orig in lower_map.items():
            if kw in lc and orig not in picked:
                picked.append(orig)
        if len(picked) >= max_cols:
            break
    if len(picked) < 3:
        picked = list(df.columns)[:max_cols]
    return picked[:max_cols]

def _format_ledger_value(col_name, value):
    lc = col_name.lower()
    try:
        if any(k in lc for k in ("score", "prob")):
            val_float = float(value)
            badge_class = 'high' if val_float >= 0.75 else ('medium' if val_float >= 0.4 else 'low')
            return f"<span class='risk-badge {badge_class}'>{val_float:.2f}</span>"
        if any(k in lc for k in ("valeur", "value", "montant", "amount", "fob")):
            return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        pass
    return str(value)

def render_ledger_cards(df, max_rows=40):
    if df.empty:
        st.markdown("<div class='empty-panel-note'>No records match the current filter.</div>", unsafe_allow_html=True)
        return

    columns = _pick_ledger_columns(df)
    col_template = " ".join(["1fr"] * len(columns))

    header_cells = "".join(
        f"<div class='ledger-col-label'>{c.replace('_', ' ').title()}</div>" for c in columns
    )
    rows_html = ""
    for _, row in df[columns].head(max_rows).iterrows():
        cells = "".join(
            f"<div class='ledger-col-value mono'>{_format_ledger_value(c, row[c])}</div>" for c in columns
        )
        rows_html += f"<div class='ledger-row' style='grid-template-columns:{col_template};'>{cells}</div>"

    st.markdown(f"""
        <div class='ledger-list'>
            <div class='ledger-row header-row' style='grid-template-columns:{col_template};'>{header_cells}</div>
            {rows_html}
        </div>
    """, unsafe_allow_html=True)

    if len(df) > max_rows:
        st.caption(f"Showing top {max_rows} of {len(df):,} matching records — download the full manifest below for the complete list.")


# ==================================================================
# LIGHT THEME CSS — PRO DASHBOARD STYLING
# ==================================================================
st.markdown("""
    <style>
    /* -- GLOBAL LIGHT THEME -- */
    .stApp {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 100% !important;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
    }
    p, div, span, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* -- TOP NAV BAR -- */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1.5rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin: -1rem -4rem 1.5rem -4rem;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .nav-brand-icon {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #16a34a, #22c55e);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: #fff; font-size: 14px;
    }
    .nav-brand-title {
        font-size: 14px; font-weight: 700; color: #0f172a;
    }
    .nav-brand-sub {
        font-size: 11px; color: #64748b;
    }
    .nav-status {
        display: flex; gap: 12px; align-items: center;
    }
    .live-badge {
        background: #dcfce7; color: #166534;
        padding: 4px 12px; border-radius: 20px;
        font-size: 11px; font-weight: 700;
    }
    
    /* -- SECTION META LABELS -- */
    .section-meta {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        font-weight: 700;
    }
    
    /* -- CARDS & PANELS -- */
    .light-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .light-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .light-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
    }
    .light-card-sub {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    
    /* -- KPI CARDS -- */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .kpi-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .kpi-delta {
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .kpi-delta.up { color: #16a34a; }
    .kpi-delta.down { color: #dc2626; }
    .kpi-delta.flat { color: #64748b; }
    
    /* -- STATUS BANNERS -- */
    .status-banner-red {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        color: #7f1d1d !important;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #fecaca;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.08);
    }
    .status-banner-amber {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        color: #78350f !important;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #fde68a;
        box-shadow: 0 2px 8px rgba(217, 119, 6, 0.08);
    }
    .status-banner-green {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        color: #14532d !important;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #bbf7d0;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.08);
    }
    .status-meta {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
        margin-bottom: 0.25rem;
    }
    .status-class {
        font-size: 1.35rem;
        font-weight: 800;
    }
    
    /* -- THREAT FEED CARDS -- */
    .threat-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .threat-card:hover {
        transform: translateX(4px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .threat-card.red {
        background: #fef2f2;
        border: 1px solid #fecaca;
    }
    .threat-card.amber {
        background: #fffbeb;
        border: 1px solid #fde68a;
    }
    .threat-card.green {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .threat-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .threat-dot.red { background: #dc2626; }
    .threat-dot.amber { background: #d97706; }
    .threat-dot.green { background: #16a34a; }
    .threat-title {
        font-size: 0.8rem;
        font-weight: 700;
    }
    .threat-title.red { color: #991b1b; }
    .threat-title.amber { color: #92400e; }
    .threat-title.green { color: #166534; }
    .threat-meta {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 1px;
    }
    .threat-score {
        font-size: 0.85rem;
        font-weight: 800;
        margin-left: auto;
        padding-left: 8px;
    }
    
    /* -- LEDGER TABLE -- */
    .ledger-list { display: flex; flex-direction: column; gap: 0.5rem; }
    .ledger-row {
        display: grid;
        gap: 0.75rem;
        align-items: center;
        background-color: #f8fafc;
        border: 1px solid #f1f5f9;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .ledger-row:hover {
        background-color: #f0fdf4;
        border-color: #bbf7d0;
    }
    .ledger-row.header-row {
        background: transparent;
        border: none;
        padding: 0 1.1rem;
        margin-bottom: -0.1rem;
    }
    .ledger-col-label {
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        font-weight: 700;
    }
    .ledger-col-value {
        font-size: 0.82rem;
        color: #334155;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .ledger-col-value.mono { font-variant-numeric: tabular-nums; }
    .ledger-col-value.bold { color: #0f172a; font-weight: 600; }
    
    /* -- RISK BADGES -- */
    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .risk-badge.high { background-color: #fef2f2; color: #991b1b; }
    .risk-badge.medium { background-color: #fffbeb; color: #92400e; }
    .risk-badge.low { background-color: #f0fdf4; color: #166534; }
    
    /* -- STATUS PILL -- */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .status-pill.red { background: #dc2626; color: #fff; }
    .status-pill.amber { background: #d97706; color: #fff; }
    .status-pill.green { background: #16a34a; color: #fff; }
    
    /* -- METRICS OVERRIDE -- */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        font-variant-numeric: tabular-nums !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #94a3b8 !important;
        font-weight: 700 !important;
    }
    
    /* -- STREAMLIT COMPONENT OVERRIDES -- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #ffffff;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #f1f5f9 !important;
        color: #0f172a !important;
    }
    .stButton > button {
        background: #16a34a !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #15803d !important;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: #f1f5f9 !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #e2e8f0 !important;
        color: #0f172a !important;
    }
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
        border-color: #16a34a !important;
        box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.12) !important;
    }
    .stSelectbox > div > div > div {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
    }
    .stSlider > div > div > div {
        color: #16a34a !important;
    }
    .stForm {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    .stFormSubmitButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    }
    .stToggle > div > div > div {
        background: #e2e8f0 !important;
    }
    .stToggle > div > div > div[data-checked="true"] {
        background: #16a34a !important;
    }
    .stExpander {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stCodeBlock {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button {
        background: #f1f5f9 !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button:hover {
        background: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    hr {
        border-color: #e2e8f0 !important;
        margin: 1.5rem 0 !important;
    }
    .stSpinner > div {
        border-color: #16a34a !important;
    }
    .stSpinner p {
        color: #16a34a !important;
        font-weight: 600 !important;
    }

    /* -- TAB TRANSITIONS / LOADING MOTION -- */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-baseweb="tab-panel"] {
        animation: fadeSlideIn 0.35s ease-out;
    }
    [data-baseweb="tab-list"] button {
        transition: color 0.2s ease, background-color 0.2s ease;
    }
    [data-baseweb="tab-highlight"] {
        background-color: #16a34a !important;
        transition: left 0.25s cubic-bezier(0.4, 0, 0.2, 1), width 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .kpi-card, .light-card, [data-testid="stVerticalBlockBorderWrapper"] {
        animation: fadeSlideIn 0.4s ease-out;
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.25); }
        50% { box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); }
    }
    .live-badge {
        animation: pulseGlow 2s infinite;
    }

    /* -- EMPTY STATE -- */
    .empty-state {
        border: 2px dashed #e2e8f0;
        padding: 5rem 2rem;
        text-align: center;
        border-radius: 12px;
        color: #94a3b8;
        background: #ffffff;
    }
    
    /* -- ALERT BOXES -- */
    .alert-box {
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .alert-box.error {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    .alert-box.warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }
    .alert-box.info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
    }
    
    /* -- DELIBERATION LOG CARDS -- */
    .delib-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .delib-card:hover {
        border-color: #cbd5e1;
    }
    .delib-meta {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .delib-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .delib-body {
        font-size: 0.85rem;
        line-height: 1.6;
        color: #64748b;
    }
    
    /* -- DECISION PANELS -- */
    .decision-panel {
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .decision-panel.blue {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
    }
    .decision-panel.amber {
        background: #fffbeb;
        border-left: 4px solid #d97706;
    }
    .decision-panel strong {
        font-size: 0.9rem;
        font-weight: 700;
    }
    .decision-panel p {
        margin-top: 0.5rem;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    /* -- PIPELINE NODE GRID -- */
    .node-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
    }
    .node-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .node-label {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
    }
    .node-state {
        font-size: 0.8rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)


# ==================================================================
# TOP NAVIGATION BAR
# ==================================================================
st.markdown("""
    <div class="top-nav">
        <div class="nav-brand">
            <div class="nav-brand-icon">G</div>
            <div>
                <div class="nav-brand-title">GUCE Customs Fraud Control</div>
                <div class="nav-brand-sub">Risk Orchestration Framework v3.2</div>
            </div>
        </div>
        <div class="nav-status">
            <span class="live-badge">LIVE</span>
            <span style="color:#94a3b8;font-size:12px;">{} UTC</span>
        </div>
    </div>
""".format(datetime.now(timezone.utc).strftime("%H:%M")), unsafe_allow_html=True)


# ==================================================================
# MAIN TABS — 4 WORKSPACES
# ==================================================================
tab_audit, tab_pipeline, tab_triage, tab_telemetry = st.tabs([
    "Audit Ledger",
    "Pipeline & Execution",
    "Live Triage",
    "Telemetry & Drift"
])


# ==================================================================
# TAB 1: AUDIT LEDGER (PRIMARY VIEW)
# ==================================================================
with tab_audit:
    
    st.markdown("<p class='section-meta'>Targeted Enforcement Queue</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:0; margin-bottom:1.25rem;'>High-Risk Auditing Ledger</h2>", unsafe_allow_html=True)
    
    if os.path.exists(PREDICTIONS_PATH):
        try:
            df_preds = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
            
            flag_cols = [c for c in df_preds.columns if 'flag' in c.lower() or 'audit' in c.lower() or 'is_fraud' in c.lower()]
            score_cols = [c for c in df_preds.columns if 'score' in c.lower() or 'prob' in c.lower()]
            
            if flag_cols:
                df_flagged = df_preds[df_preds[flag_cols[0]].isin([1, 1.0, True, '1', 'True'])]
            elif score_cols:
                df_flagged = df_preds.sort_values(by=score_cols[0], ascending=False).head(514)
            else:
                df_flagged = df_preds
            
            total_records = len(df_preds)
            flagged_records = len(df_flagged)
            cutoff_rate = (flagged_records / total_records * 100) if total_records else 0.0
            
            fourth_label, fourth_value, fourth_dir = None, None, "flat"
            if score_cols:
                avg_score = df_flagged[score_cols[0]].mean()
                fourth_label, fourth_value = "Avg. Risk Weight", f"{avg_score:.2f}"
            else:
                exporter_cols = [c for c in df_preds.columns if c.lower() in ("emetteur", "exporter", "exportateur")]
                if exporter_cols:
                    fourth_label = "Unique Exporters Flagged"
                    fourth_value = f"{df_flagged[exporter_cols[0]].nunique():,}"
            
            kpi_cols = st.columns(4 if fourth_label else 3)
            kpi_data = [
                ("Total Batch Records", f"{total_records:,}", None, "flat"),
                ("Isolated Audit Queue", f"{flagged_records:,}", None, "flat"),
                ("Anomalous Cutoff Rate", f"{cutoff_rate:.2f}%", None, "up" if cutoff_rate >= 10 else "down"),
            ]
            if fourth_label:
                kpi_data.append((fourth_label, fourth_value, None, "flat"))
            
            for col, (label, value, _, direction) in zip(kpi_cols, kpi_data):
                arrow = {"up": "+", "down": "-", "flat": ""}[direction]
                delta_html = (
                    f"<div class='kpi-delta {direction}'>{arrow} {'above threshold' if direction == 'up' else 'below threshold'}</div>"
                    if direction != "flat" else ""
                )
                with col:
                    st.markdown(f"""
                        <div class='kpi-card'>
                            <div class='kpi-label'>{label}</div>
                            <div class='kpi-value'>{value}</div>
                            {delta_html}
                        </div>
                    """, unsafe_allow_html=True)
            
            st.write("")
            
            chart_col1, chart_col2 = st.columns(2)
            
            date_cols = [c for c in df_preds.columns if any(
                k in c.lower() for k in ("date", "timestamp", "datetime", "horodatage"))]
            
            with chart_col1:
                with st.container(border=True):
                    st.markdown("<div class='light-card-header'><span class='light-card-title'>Flagged Volume Trend</span><span class='light-card-sub'>Last 14 records</span></div>", unsafe_allow_html=True)
                    if date_cols:
                        try:
                            ts = pd.to_datetime(df_flagged[date_cols[0]], errors="coerce").dropna()
                            daily_counts = ts.dt.floor("D").value_counts().sort_index().tail(14)
                            labels = [d.strftime("%b %d") for d in daily_counts.index]
                            data_points = [int(v) for v in daily_counts.values]
                            render_chartjs_panel(
                                "trendChart", "line", labels,
                                [{
                                    "label": "Flagged",
                                    "data": data_points,
                                    "borderColor": "#16a34a",
                                    "backgroundColor": "rgba(22, 163, 74, 0.06)",
                                    "fill": True,
                                    "tension": 0.35,
                                    "pointRadius": 3,
                                    "pointBackgroundColor": "#16a34a",
                                }],
                                height=260,
                            )
                        except Exception:
                            st.markdown("<div class='empty-state' style='padding:2rem;'>Could not parse date column for trending.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='empty-state' style='padding:2rem;'>No date column found - add one (e.g. <code>declaration_date</code>).</div>", unsafe_allow_html=True)
            
            category_cols = [c for c in df_preds.columns if c.lower() in
                             ("emetteur", "exporter", "exportateur", "pays", "country", "importer", "declarant")]
            
            with chart_col2:
                with st.container(border=True):
                    st.markdown("<div class='light-card-header'><span class='light-card-title'>Top Flagged Categories</span><span class='light-card-sub'>By volume</span></div>", unsafe_allow_html=True)
                    if category_cols:
                        counts = df_flagged[category_cols[0]].astype(str).value_counts().head(7)
                        render_chartjs_panel(
                            "categoryChart", "bar",
                            list(counts.index), [{
                                "label": "Flagged",
                                "data": [int(v) for v in counts.values],
                                "backgroundColor": "#16a34a",
                                "borderRadius": 6,
                            }],
                            height=260,
                        )
                    else:
                        st.markdown("<div class='empty-state' style='padding:2rem;'>No exporter/country column found.</div>", unsafe_allow_html=True)
            
            st.write("")
            with st.container(border=True):
                st.markdown("""
                    <div class='light-card-header'>
                        <span class='light-card-title'>Pipeline Flagged Anomalies</span>
                        <span class='light-card-sub'>Airflow - final_fraud_fusion</span>
                    </div>
                """, unsafe_allow_html=True)
                
                search_query = st.text_input(
                    "Filter by Exporter ID, File ID, or Customs Reference", ""
                )
                if search_query:
                    mask = df_flagged.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
                    df_display = df_flagged[mask]
                else:
                    df_display = df_flagged
                
                render_ledger_cards(df_display)
                
                st.write("")
                csv_buffer = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Audit Manifest (CSV)",
                    data=csv_buffer,
                    file_name=f"guce_fraud_audit_manifest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
    else:
        st.markdown(f"""
            <div class="alert-box error">
                <strong>Predictions File Missing</strong><br>
                Could not locate: <code>{PREDICTIONS_PATH}</code><br>
                Verify the Airflow task <code>final_fraud_fusion</code> has run, or trigger the DAG in Command Center.
            </div>
        """, unsafe_allow_html=True)

# ==================================================================
# TAB 2: PIPELINE & EXECUTION
# ==================================================================
with tab_pipeline:

    st.markdown("<p class='section-meta'>Orchestration</p>", unsafe_allow_html=True)

    with st.container(border=False):
        exec_col1, exec_col2 = st.columns([4, 1])
        with exec_col1:
            st.markdown(f"<h3 style='margin-top:0;'>Execute DAG: <code style='color:#16a34a;'>{AIRFLOW_DAG_ID}</code></h3>", unsafe_allow_html=True)
            st.caption("Trigger the end-to-end pipeline: extraction, graph embedding, evaluation")
        with exec_col2:
            st.write("")
            st.write("")
            if st.button("Trigger Execution", use_container_width=True, type="primary"):
                with st.spinner("Dispatching to Airflow..."):
                    success, message = trigger_airflow_dag(AIRFLOW_DAG_ID)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    st.divider()

    st.markdown("<p class='section-meta'>Pipeline Diagnostics</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Node Inspector & Log Viewer</h3>", unsafe_allow_html=True)
    
    if "last_dag_run_id" not in st.session_state:
        st.session_state["last_dag_run_id"] = "manual__2026-07-20T17:29:02.110651+00:00"
    
    dag_col1, dag_col2 = st.columns([3, 1])
    with dag_col1:
        dag_run_id_input = st.text_input(
            "Active DAG Run ID",
            value=st.session_state.get("last_dag_run_id", ""),
            help="Enter the DAG Run ID to inspect node logs."
        )
    with dag_col2:
        st.write("")
        st.write("")
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()
    
    if dag_run_id_input:
        tasks = get_dag_run_tasks(AIRFLOW_DAG_ID, dag_run_id_input)
        
        if tasks:
            st.markdown("<p class='section-meta'>Node Status Overview</p>", unsafe_allow_html=True)
            
            node_cols = st.columns(min(len(tasks), 5))
            for idx, task in enumerate(tasks):
                t_id = task.get("task_id")
                state = task.get("state", "queued")
                
                state_color = {
                    "success": "#16a34a", "running": "#d97706", "failed": "#dc2626",
                    "queued": "#94a3b8", "skipped": "#94a3b8", "up_for_retry": "#d97706"
                }.get(state, "#94a3b8")
                
                with node_cols[idx % len(node_cols)]:
                    st.markdown(f"""
                        <div class='node-item'>
                            <div class='node-state' style='color:{state_color};font-size:1.2rem;margin-bottom:0.25rem;'>
                                {"OK" if state=="success" else "RUN" if state in ["running","up_for_retry"] else "ERR" if state=="failed" else "..."}
                            </div>
                            <div class='node-label'>{t_id}</div>
                            <div class='node-state' style='color:{state_color};'>{state.upper() if state else "PENDING"}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<p class='section-meta' style='margin-top:1.5rem;'>Log Viewer</p>", unsafe_allow_html=True)
            
            task_ids = [t.get("task_id") for t in tasks]
            selected_task_id = st.selectbox("Select Pipeline Node", options=task_ids)
            
            if selected_task_id:
                selected_task = next((t for t in tasks if t.get("task_id") == selected_task_id), {})
                try_number = selected_task.get("try_number", 1) or 1
                
                with st.spinner(f"Fetching logs for '{selected_task_id}'..."):
                    logs = get_task_logs(AIRFLOW_DAG_ID, dag_run_id_input, selected_task_id, try_number)
                    
                st.code(logs, language="text", line_numbers=True)
        else:
            st.info("No task instances found. The DAG may still be initializing.")


# ==================================================================
# TAB 3: LIVE TRIAGE ENGINE
# ==================================================================
with tab_triage:

    
    st.markdown("<p class='section-meta'>Data Entry Gateway</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Declaration Inputs</h3>", unsafe_allow_html=True)
    
    with st.form("input_form"):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            file_id = st.number_input("Structural Node ID (file_id)", min_value=0, value=10016, step=1)
        with f_col2:
            numero_dossier = st.text_input("Customs File Number", value="EP052469EX01DM")
        with f_col3:
            emetteur = st.text_input("Exporter Identification", value="OLAMCAM")
        with f_col4:
            valeur_fob = st.number_input("Declared FOB Value (CFA)", min_value=0.0, value=439227.84, step=1000.0)
        
        is_anomaly_ip_bool = st.toggle("IP Anomaly Detected", value=True)
        is_anomaly_ip = 1.0 if is_anomaly_ip_bool else 0.0
        
        submit_button = st.form_submit_button("Execute Multi-Agent Evaluation", use_container_width=True)
    
    st.write("")
    
    if submit_button:
        payload = {
            "file_id": file_id,
            "numero_dossier": numero_dossier,
            "emetteur": emetteur,
            "valeur_fob": valeur_fob,
            "is_anomaly_ip": is_anomaly_ip
        }
        api_url = "http://127.0.0.1:8000/api/v1/triage/evaluate"
        
        try:
            with st.spinner("Processing network topologies..."):
                response = requests.post(api_url, json=payload)
                
            if response.status_code == 200:
                dossier = response.json()
                verdict = dossier.get('verdict', {})
                
                classification = verdict.get('triage_classification', 'UNKNOWN').upper()
                score_global = verdict.get('multi_modal_score', 0.0)
                reasoning = verdict.get('executive_reasoning_summary', '')
                action = verdict.get('operational_directive', '')
                
                m_col1, m_col2 = st.columns([3, 1])
                with m_col1:
                    if "TIER 1" in classification or "RED" in classification or "SEIZURE" in classification:
                        st.markdown(f"""
                            <div class='status-banner-red'>
                                <div class='status-meta'>System Action Classification</div>
                                <div class='status-class'>{classification}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    elif "TIER 2" in classification or "TIER 5" in classification or "AMBER" in classification:
                        st.markdown(f"""
                            <div class='status-banner-amber'>
                                <div class='status-meta'>System Action Classification</div>
                                <div class='status-class'>{classification}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='status-banner-green'>
                                <div class='status-meta'>System Action Classification</div>
                                <div class='status-class'>{classification}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                with m_col2:
                    with st.container(border=True):
                        st.metric(label="Fused Confidence", value=f"{score_global * 100:.1f}%")
                
                st.write("")
                
                left_panel, right_panel = st.columns([1.1, 0.9])
                
                with left_panel:
                    st.markdown("<p class='section-meta'>Interactive Evidence</p>", unsafe_allow_html=True)
                    workspace_tabs = st.tabs(["Network Map", "Agent Ledger"])
                    
                    with workspace_tabs[0]:
                        with st.container(border=True):
                            real_xai_links = dossier.get("xai_graph_links", [])
                            real_raw_links = dossier.get("raw_neighborhood_links", [])
                            render_fraud_topology(
                                target_file=file_id, 
                                xai_links=real_xai_links,
                                raw_relations=real_raw_links
                            )
                            
                    with workspace_tabs[1]:
                        for log in dossier.get('deliberation_logs', []):
                            agent_name = log.get('agent', 'Analytical Node')
                            raw_findings = log.get('findings', 'No telemetry delivered.')
                            node_type = "Statistical Core" if "Tabular" in agent_name else "Relational Topology"
                            
                            st.markdown(f"""
                                <div class='delib-card'>
                                    <div class='delib-meta'>Pipeline Target: {node_type}</div>
                                    <div class='delib-title'>{agent_name}</div>
                                    <div class='delib-body'>{raw_findings}</div>
                                </div>
                            """, unsafe_allow_html=True)
                
                with right_panel:
                    st.markdown("<p class='section-meta'>Consolidated Decision</p>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='decision-panel blue'>
                            <strong style='color:#1e40af;'>Compliance Analysis</strong>
                            <p style='color:#1e40af;'>{reasoning}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class='decision-panel amber'>
                            <strong style='color:#92400e;'>Enforcement Directive</strong>
                            <p style='color:#92400e;'>{action}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                with st.expander("System Metadata (JSON Schema)"):
                    st.json(dossier)
                    
        except requests.exceptions.ConnectionError:
            st.error("Connection Failure: Fast API gateway offline (Port 8000 unreachable).")
    else:
        st.markdown("""
            <div class='empty-state'>
                <div style='font-size:2rem;margin-bottom:0.5rem;color:#cbd5e1;'>TARGET</div>
                <div style='font-weight:600;color:#64748b;'>Awaiting Evaluation</div>
                <div style='font-size:0.85rem;color:#94a3b8;margin-top:0.25rem;'>Complete input parameters and submit</div>
            </div>
        """, unsafe_allow_html=True)


# ==================================================================
# TAB 4: TELEMETRY & DRIFT
# ==================================================================
with tab_telemetry:

    st.markdown("<p class='section-meta'>Live Telemetry</p>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0;'>Grafana - {GRAFANA_DASHBOARD_NAME}</h3>", unsafe_allow_html=True)
    st.caption("Panels embedded via d-solo share links with auto-refresh")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        refresh_choice = st.selectbox("Refresh", ["5s", "10s", "30s", "1m"], index=0)
    with ctrl_col2:
        time_range_choice = st.selectbox("Time Range", ["Last 15m", "Last 1h", "Last 6h", "Last 24h"], index=2)
    with ctrl_col3:
        cols_per_row = st.slider("Panels per row", min_value=1, max_value=3, value=2)
    
    uid, dash_url_path, err = resolve_grafana_dashboard(GRAFANA_DASHBOARD_NAME)
    
    if err or not uid:
        st.markdown(f"""
            <div class="alert-box error">
                <strong>Could not load Grafana dashboard</strong><br>
                {err or f"Dashboard '{GRAFANA_DASHBOARD_NAME}' not found."}<br><br>
                <em>Checklist:</em>
                <ul style='margin-top:0.5rem;'>
                    <li>Grafana reachable at <code>{GRAFANA_URL}</code></li>
                    <li>Dashboard titled <code>{GRAFANA_DASHBOARD_NAME}</code> exists</li>
                    <li>Valid API key or anonymous access enabled</li>
                    <li><code>allow_embedding = true</code> in grafana.ini</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        panels, panel_err = get_grafana_panels(uid)
        if panel_err:
            st.error(panel_err)
        elif not panels:
            st.warning("Dashboard found but has no panels.")
        else:
            for row_start in range(0, len(panels), cols_per_row):
                row_panels = panels[row_start: row_start + cols_per_row]
                cols = st.columns(len(row_panels))
                for col, panel in zip(cols, row_panels):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"<div class='section-meta'>{panel['title']}</div>", unsafe_allow_html=True)
                            embed_url = build_panel_embed_url(
                                dash_url_path, panel["id"], refresh_choice, time_range_choice
                            )
                            components.iframe(embed_url, height=340, scrolling=True)
    
    st.divider()
    
    st.markdown("<p class='section-meta'>Statistical Quality Assurance</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Wasserstein Data Drift Matrix</h3>", unsafe_allow_html=True)
    st.caption("Monitors concept drift between pipeline distributions and active baseline")
    
    if DRIFT_REPORT_PATH and os.path.exists(DRIFT_REPORT_PATH):
        with open(DRIFT_REPORT_PATH, 'r', encoding='utf-8') as f:
            html_report_content = f.read()
        components.html(html_report_content, height=1200, scrolling=True)
    else:
        tried_list = "".join(f"<li><code>{p}</code></li>" for p in _DRIFT_PATH_CANDIDATES)
        st.markdown(f"""
            <div class="alert-box error">
                <strong>Monitoring Telemetry Offline</strong><br>
                Checked locations:<ul style='margin-top:0.5rem;'>{tried_list}</ul>
                <em>Fix:</em> Trigger the DAG above or set <code>DRIFT_REPORT_PATH</code>.
            </div>
        """, unsafe_allow_html=True)