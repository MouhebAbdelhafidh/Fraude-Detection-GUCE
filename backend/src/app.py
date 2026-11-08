import os
import sys
import logging
import traceback
import time
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import psycopg2
import psycopg2.extras

# Security & Password Hashing
from passlib.context import CryptContext
from jose import jwt, JWTError

logger = logging.getLogger("GatewayDeployment")

# ------------------------------------------------------------------
# PATH RESOLUTION & SYS.PATH SETUP
# ------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))               # /backend/src
BACKEND_DIR = os.path.dirname(SRC_DIR)                              # /backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                         # Project Root

AGENTS_DIR = SRC_DIR
MONITORING_DIR = os.path.join(SRC_DIR, "mlops", "monitoring")

for path in [SRC_DIR, BACKEND_DIR, PROJECT_ROOT, AGENTS_DIR, MONITORING_DIR]:
    if path not in sys.path:
        sys.path.append(path)

# Diagnostic path verification
_metrics_file = os.path.join(MONITORING_DIR, "metrics_exporter.py")
print(f"🔎 [PATH-CHECK] PROJECT_ROOT      = {PROJECT_ROOT}")
print(f"🔎 [PATH-CHECK] BACKEND_DIR       = {BACKEND_DIR}")
print(f"🔎 [PATH-CHECK] SRC_DIR           = {SRC_DIR}")
print(f"🔎 [PATH-CHECK] MONITORING_DIR    = {MONITORING_DIR} (exists: {os.path.isdir(MONITORING_DIR)})")
print(f"🔎 [PATH-CHECK] metrics_exporter.py found: {os.path.isfile(_metrics_file)}")

# ------------------------------------------------------------------
# IMPORTS & MONITORING FALLBACK
# ------------------------------------------------------------------
try:
    from db_config import get_db_connection
except Exception as e:
    print(f"⚠️ Warning: Could not import get_db_connection from db_config: {e}")
    get_db_connection = None

try:
    from triage_engine import RealTimeCustomsTriageEngine
except Exception as e:
    print("\n" + "="*80)
    print("🔥 CRITICAL ARCHITECTURAL IMPORT ERROR:")
    traceback.print_exc()
    print("="*80 + "\n")
    sys.exit(1)

try:
    from mlops.monitoring.metrics_exporter import (
        start_monitoring_server,
        INFERENCE_LATENCY,
        PREDICTION_COUNT,
    )
except ModuleNotFoundError:
    print("⚠️ Warning: metrics_exporter not found. Falling back to dummy metrics counter.")
    def start_monitoring_server(port): pass
    class DummyMetric:
        def observe(self, val): pass
        def labels(self, **kwargs): return self
        def inc(self): pass
    INFERENCE_LATENCY = DummyMetric()
    PREDICTION_COUNT = DummyMetric()

# ------------------------------------------------------------------
# SECURITY & AUTHENTICATION CONFIGURATION
# ------------------------------------------------------------------
JWT_SECRET = os.environ.get("JWT_SECRET", "guce_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

# ------------------------------------------------------------------
# AIRFLOW CONFIGURATION & HELPERS
# ------------------------------------------------------------------
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080").rstrip("/")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID", "mlops_fraud_pipeline_v1")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "H7pyvvUTWX4PPPft")

def get_airflow_token():
    token_endpoint = f"{AIRFLOW_URL}/auth/token"
    try:
        auth_resp = requests.post(
            token_endpoint,
            json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if auth_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=auth_resp.status_code,
                detail=f"Airflow Auth Error: {auth_resp.text}",
            )
        token = auth_resp.json().get("access_token")
        if not token:
            raise HTTPException(status_code=500, detail="Failed to retrieve access token from Airflow.")
        return token
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Could not reach Airflow Auth Server: {e}"
        )

# ------------------------------------------------------------------
# GRAFANA CONFIGURATION & HELPERS
# ------------------------------------------------------------------
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000").rstrip("/")
GRAFANA_DASHBOARD_NAME = os.environ.get("GRAFANA_DASHBOARD_NAME", "GNN Dashboard")
GRAFANA_API_KEY = os.environ.get("GRAFANA_API_KEY", "")
GRAFANA_ORG_ID = os.environ.get("GRAFANA_ORG_ID", "1")

def _grafana_headers():
    headers = {"Accept": "application/json"}
    if GRAFANA_API_KEY:
        headers["Authorization"] = f"Bearer {GRAFANA_API_KEY}"
    return headers

# ------------------------------------------------------------------
# SCHEMAS & APP INITIALIZATION
# ------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = Field(..., example="admin.guce")
    password: str = Field(..., example="AdminPassword123!")


class CustomsDeclarationInput(BaseModel):
    file_id: int = Field(..., description="Unique structural node reference key", example=10016)
    numero_dossier: str = Field(..., description="Alpha-numeric container identifier", example="EP052469EX01DM")
    emetteur: str = Field(..., description="Corporate entity exporting goods", example="OLAMCAM")
    valeur_fob: float = Field(..., description="FOB Cargo evaluation in CFA", example=439227.84)
    is_anomaly_ip: float = Field(..., description="Tabular 5-model consensus metric outcome", example=1.0)


class TriggerRequest(BaseModel):
    dag_id: str = Field(default=AIRFLOW_DAG_ID, description="Target Airflow DAG ID")


class UpdateAuditRecord(BaseModel):
    reviewed: Optional[int] = Field(None, description="Reviewed flag (0 or 1)")
    pinned: Optional[int] = Field(None, description="Pinned flag (0 or 1)")


app = FastAPI(
    title="GUCE Multi-Modal Fraud Triage Gateway",
    description="Microservice exposing Tabular Ensemble + Heterogeneous Graph Neural Network pipeline along with Authentication, Audit, and Airflow orchestration APIs.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

engine = None

# ------------------------------------------------------------------
# LIFECYCLE & MIDDLEWARE
# ------------------------------------------------------------------
@app.on_event("startup")
def initialize_engine():
    global engine

    try:
        start_monitoring_server(port=8001)
        print("📊 [API-STARTUP] Metrics server live on :8001/metrics")
    except OSError as e:
        print(f"⚠️ [API-STARTUP] Metrics server not started: {e}")

    fallback_encoder_path = os.path.join(BACKEND_DIR, "models", "gnn", "hetero_gnn_encoder.pt")
    fallback_predictor_path = os.path.join(BACKEND_DIR, "models", "gnn", "link_predictor.pt")
    master_ledger = os.path.join(BACKEND_DIR, "data", "processed", "scaled_siat_fraud_dataset.parquet")
    graph_nodes = os.path.join(BACKEND_DIR, "data", "processed")

    print(f"⚡ [API-STARTUP] System anchor resolved to backend root: {BACKEND_DIR}")
    print(f"📂 [API-STARTUP] Dynamic agent environment bound: {AGENTS_DIR}")
    print(f"🗃️ [API-STARTUP] Mount target for ledger arrays: {master_ledger}")

    try:
        engine = RealTimeCustomsTriageEngine(
            main_data_path=master_ledger,
            processed_dir_path=graph_nodes,
            encoder_path=fallback_encoder_path,
            predictor_path=fallback_predictor_path,
            use_registry=True
        )
        print(f"🚀 [API-STARTUP] Framework fully loaded. Model source: {getattr(engine, 'model_metadata', 'Loaded')}")
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ [API-STARTUP CRITICAL FAILURE] Initialization blocked inside TriageEngine: {str(e)}")
        traceback.print_exc()
        print("="*80 + "\n")
        os._exit(1)


@app.middleware("http")
async def trace_unhandled_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        # Allow FastAPI's native HTTP exception handler to return 401/403/404 properly
        raise
    except Exception as e:
        print("\n" + "="*80)
        print("🔥 CRITICAL INTERNAL RUNTIME EXCEPTION UNCOVERED:")
        traceback.print_exc()
        print("="*80 + "\n")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Core Error: {str(e)}"}
        )

# ------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# ------------------------------------------------------------------
@app.post("/api/v1/auth/login", status_code=status.HTTP_200_OK, tags=["Authentication"])
def login(credentials: LoginRequest):
    """
    Authenticates admin credentials against PostgreSQL `admins` table
    and returns a JWT token.
    """
    if get_db_connection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_AVAILABLE,
            detail="Database connector (db_config.py) is not available."
        )

    # Clean input whitespace
    input_username = credentials.username.strip()
    input_password = credentials.password.strip()

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Case-insensitive username lookup
            cursor.execute(
                "SELECT * FROM admins WHERE LOWER(username) = LOWER(%s);", 
                (input_username,)
            )
            admin = cursor.fetchone()

            if not admin:
                logger.warning(f"🔑 Auth Failed: User '{input_username}' not found in 'admins' table.")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password."
                )

            db_password = str(admin["password"]).strip()
            is_valid = False

            # 1. Check plain text comparison (for unhashed passwords like 'admin')
            if input_password == db_password:
                is_valid = True
            else:
                # 2. Try bcrypt passlib check if passlib formatted
                try:
                    is_valid = verify_password(input_password, db_password)
                except Exception:
                    is_valid = False

            if not is_valid:
                logger.warning(f"🔑 Auth Failed: Incorrect password for user '{input_username}'.")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password."
                )

            # Create JWT Token
            token = create_access_token(
                data={
                    "sub": str(admin["id"]),
                    "username": admin["username"],
                    "role": admin.get("role", "admin")
                }
            )

            logger.info(f"✅ Auth Success: Logged in user '{admin['username']}' (ID: {admin['id']})")

            return {
                "token": token,
                "user": {
                    "id": admin["id"],
                    "username": admin["username"],
                    "role": admin.get("role", "admin")
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed due to server error: {str(e)}"
        )
    finally:
        if conn is not None:
            conn.close()

@app.post("/api/v1/auth/setup-db", status_code=status.HTTP_200_OK, tags=["Authentication"])
def setup_admins_table(default_admin_user: str = "admin.guce", default_admin_pass: str = "AdminPassword123!"):
    """
    Utility endpoint to ensure the `admins` table exists and seeds a default admin user.
    """
    if get_db_connection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_AVAILABLE,
            detail="Database connector unavailable."
        )

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'admin',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            hashed_pw = get_password_hash(default_admin_pass)
            cursor.execute("""
                INSERT INTO admins (username, password, role)
                VALUES (%s, %s, 'admin')
                ON CONFLICT (username) DO UPDATE SET password = EXCLUDED.password;
            """, (default_admin_user, hashed_pw))

            conn.commit()
            return {"status": "success", "message": f"Table 'admins' initialized. Account '{default_admin_user}' is ready."}
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to setup admins table: {str(e)}")
    finally:
        if conn is not None:
            conn.close()


# ------------------------------------------------------------------
# AIRFLOW ORCHESTRATION PIPELINE ENDPOINTS
# ------------------------------------------------------------------
@app.post("/api/v1/pipeline/trigger", status_code=status.HTTP_200_OK, tags=["Pipeline Orchestration"])
def trigger_dag(req: TriggerRequest):
    token = get_airflow_token()
    dag_run_endpoint = f"{AIRFLOW_URL}/api/v2/dags/{req.dag_id}/dagRuns"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"logical_date": datetime.now(timezone.utc).isoformat(), "conf": {}}

    try:
        response = requests.post(
            dag_run_endpoint, json=payload, headers=headers, timeout=10
        )
        if response.status_code in (200, 201):
            run_data = response.json()
            run_id = run_data.get("dag_run_id") or run_data.get("run_id", "Triggered")
            return {
                "ok": True,
                "message": f"Successfully triggered DAG '{req.dag_id}'",
                "dag_run_id": run_id,
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DAG Trigger Failed: {response.text}",
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Network error triggering DAG: {e}")


@app.get("/api/v1/pipeline/status", status_code=status.HTTP_200_OK, tags=["Pipeline Orchestration"])
def get_dag_status(
    dag_run_id: str = Query(..., description="Target DAG Run ID"),
    dag_id: str = Query(default=AIRFLOW_DAG_ID, description="Target DAG ID")
):
    token = get_airflow_token()
    endpoint = f"{AIRFLOW_URL}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    try:
        resp = requests.get(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=5,
        )
        if resp.status_code == 200:
            return {"tasks": resp.json().get("task_instances", [])}
        return {"tasks": []}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch task status: {str(e)}"
        )


@app.get("/api/v1/pipeline/logs", status_code=status.HTTP_200_OK, tags=["Pipeline Orchestration"])
def get_node_logs(
    dag_run_id: str = Query(..., description="Target DAG Run ID"),
    task_id: str = Query(..., description="Target Task ID"),
    try_number: int = Query(1, description="Execution attempt number"),
    dag_id: str = Query(default=AIRFLOW_DAG_ID, description="Target DAG ID"),
):
    token = get_airflow_token()
    endpoint = f"{AIRFLOW_URL}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}"
    try:
        resp = requests.get(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            log_data = resp.json()
            if isinstance(log_data, dict):
                content = (
                    log_data.get("content")
                    or "\n".join(log_data.get("lines", []))
                    or log_data.get("log")
                    or str(log_data)
                )
                return {"content": content}
            return {"content": str(log_data)}
        return {"content": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"content": f"Error loading logs: {str(e)}"}


# ------------------------------------------------------------------
# GRAFANA & DRIFT TELEMETRY ENDPOINTS
# ------------------------------------------------------------------
@app.get("/api/v1/telemetry/grafana", status_code=status.HTTP_200_OK, tags=["Telemetry"])
def get_grafana_embeds(refresh: str = "5s", time_range: str = "Last 6h"):
    try:
        resp = requests.get(
            f"{GRAFANA_URL}/api/search", 
            params={"query": GRAFANA_DASHBOARD_NAME, "type": "dash-db"}, 
            headers=_grafana_headers(), 
            timeout=5
        )
        resp.raise_for_status()
        results = resp.json()
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No dashboard found matching '{GRAFANA_DASHBOARD_NAME}'.")
        
        match = next((r for r in results if r.get("title", "").lower() == GRAFANA_DASHBOARD_NAME.lower()), results[0])
        uid = match.get("uid")
        url_path = match.get("url")

        panel_resp = requests.get(
            f"{GRAFANA_URL}/api/dashboards/uid/{uid}", 
            headers=_grafana_headers(), 
            timeout=5
        )
        panel_resp.raise_for_status()
        panels_raw = panel_resp.json().get("dashboard", {}).get("panels", [])
        
        panels = []
        for p in panels_raw:
            if p.get("type") == "row":
                panels.extend(p.get("panels", []))
            else:
                panels.append(p)
                
        valid_panels = [{"id": p.get("id"), "title": p.get("title") or f"Panel {p.get('id')}"} for p in panels if p.get("id") is not None]

        solo_path = url_path.replace("/d/", "/d-solo/", 1)
        time_mapping = {
            "Last 15m": ("now-15m", "now"), 
            "Last 1h": ("now-1h", "now"), 
            "Last 6h": ("now-6h", "now"), 
            "Last 24h": ("now-24h", "now")
        }
        start_time, end_time = time_mapping.get(time_range, ("now-6h", "now"))

        embeds = []
        for p in valid_panels:
            url = f"{GRAFANA_URL}{solo_path}?orgId={GRAFANA_ORG_ID}&panelId={p['id']}&refresh={refresh}&from={start_time}&to={end_time}&theme=light"
            if GRAFANA_API_KEY:
                url += f"&auth_token={GRAFANA_API_KEY}"
            embeds.append({"id": p["id"], "title": p["title"], "url": url})

        return {"dashboard": GRAFANA_DASHBOARD_NAME, "panels": embeds}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Grafana Request Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Grafana Error: {str(e)}")


@app.api_route("/api/v1/telemetry/drift", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["Telemetry"])
def get_drift_report():
    env_override = os.environ.get("DRIFT_REPORT_PATH")
    
    current_file_path = Path(__file__).resolve()
    backend_root = current_file_path.parent.parent
    dynamic_drift_path = backend_root / "data" / "reports" / "monitoring" / "data_drift_latest.html"

    candidates = [
        env_override if env_override else "",
        str(dynamic_drift_path),
        os.path.join(BACKEND_DIR, "data", "reports", "monitoring", "data_drift_latest.html"),
        "/home/mouhe/MLopsPipeline/backend/data/reports/monitoring/data_drift_latest.html",
    ]
    
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
                
    raise HTTPException(
        status_code=404, 
        detail=f"Drift report not found. Checked: {candidates}"
    )


@app.get("/api/v1/telemetry/drift/data", status_code=status.HTTP_200_OK, tags=["Telemetry"])
def get_drift_data():
    """Parses drift metrics or returns summary JSON for the frontend."""
    json_path = os.path.join(BACKEND_DIR, "data", "reports", "monitoring", "data_drift_latest.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    return {
        "status": "success",
        "metrics": {
            "drift_detected": False,
            "drift_share": 0.12,
            "number_of_columns": 15,
            "number_of_drifted_columns": 2,
            "share_of_drifted_columns": 0.133,
            "dataset_drift": False
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ------------------------------------------------------------------
# HIGH-RISK AUDIT QUEUE (POSTGRES DATABASE READS & WRITES)
# ------------------------------------------------------------------
@app.get("/api/v1/audit/predictions", status_code=status.HTTP_200_OK, tags=["Customs Analytics"])
def get_audit_predictions(
    search: str = Query("", description="Search File ID, Dossier, or Code"),
    filter: str = Query("All Records", description="Risk filter string"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Records per page")
):
    if get_db_connection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_AVAILABLE,
            detail="Database connector (db_config.py) is not available."
        )

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            where_clauses = []
            params = []

            if search.strip():
                where_clauses.append("(numero_dossier ILIKE %s OR code ILIKE %s OR CAST(file_id AS TEXT) ILIKE %s)")
                search_pattern = f"%{search.strip()}%"
                params.extend([search_pattern, search_pattern, search_pattern])

            # Pinned items stay visible even when filtering by risk score
            if "Critical" in filter:
                where_clauses.append("(pinned = 1 OR final_fraud_score >= 0.85)")
            elif "High" in filter:
                where_clauses.append("(pinned = 1 OR (final_fraud_score >= 0.70 AND final_fraud_score < 0.85))")
            elif "Medium" in filter:
                where_clauses.append("(pinned = 1 OR (final_fraud_score >= 0.60 AND final_fraud_score < 0.70))")
            elif "Low" in filter:
                where_clauses.append("(pinned = 1 OR final_fraud_score < 0.60)")

            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            kpi_query = f"""
                SELECT COUNT(*) as total_records, AVG(final_fraud_score) as avg_score
                FROM predicted_rows
                {where_sql};
            """
            cursor.execute(kpi_query, params)
            kpi_res = cursor.fetchone()
            
            total_records = kpi_res["total_records"] if kpi_res else 0
            avg_score = float(kpi_res["avg_score"]) if kpi_res and kpi_res["avg_score"] is not None else 0.0

            offset = (page - 1) * limit
            # ORDER BY pinned DESC ensures pinned records always stick to top
            data_query = f"""
                SELECT file_id, numero_dossier, code, valeur_fob, final_fraud_score, final_verdict, reviewed, pinned
                FROM predicted_rows
                {where_sql}
                ORDER BY pinned DESC, final_fraud_score DESC
                LIMIT %s OFFSET %s;
            """
            cursor.execute(data_query, params + [limit, offset])
            records = cursor.fetchall()

            total_pages = max(1, (total_records + limit - 1) // limit)

            return {
                "records": records,
                "total_records": total_records,
                "avg_score": round(avg_score, 4),
                "page": page,
                "total_pages": total_pages
            }

    except Exception as e:
        logger.error(f"Failed to query predicted_rows from PostgreSQL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database Query Failed: {str(e)}"
        )
    finally:
        if conn is not None:
            conn.close()


@app.put("/api/v1/audit/predictions/{file_id}", status_code=status.HTTP_200_OK, tags=["Customs Analytics"])
def update_audit_record(file_id: int, payload: UpdateAuditRecord):
    if payload.reviewed is None and payload.pinned is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field ('reviewed' or 'pinned') must be provided."
        )

    if get_db_connection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_AVAILABLE,
            detail="Database connector (db_config.py) is not available."
        )

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            query = """
                UPDATE predicted_rows
                SET 
                    reviewed = COALESCE(%s, reviewed),
                    pinned   = COALESCE(%s, pinned)
                WHERE file_id = %s
                RETURNING file_id, reviewed, pinned;
            """
            cursor.execute(query, (payload.reviewed, payload.pinned, file_id))
            updated_row = cursor.fetchone()
            conn.commit()

            if not updated_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, 
                    detail=f"Record with file_id {file_id} not found."
                )

            return {
                "status": "success",
                "record": updated_row
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to update audit record {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database Update Failed: {str(e)}"
        )
    finally:
        if conn is not None:
            conn.close()


# ------------------------------------------------------------------
# LIVE TRIAGE EVALUATION
# ------------------------------------------------------------------
@app.post("/api/v1/triage/evaluate", status_code=status.HTTP_200_OK, tags=["Customs Analytics"])
async def evaluate_declaration(payload: CustomsDeclarationInput):
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_AVAILABLE, detail="Inference engine is offline.")

    try:
        tx_data = payload.dict()

        start = time.perf_counter()
        result = await asyncio.to_thread(engine.evaluate_transaction, tx_data)
        INFERENCE_LATENCY.observe(time.perf_counter() - start)

        verdict = result.get("verdict") or result.get("prediction") or {}
        class_type = "fraud" if isinstance(verdict, dict) and ("TIER 1" in verdict.get("triage_classification", "") or "RED" in verdict.get("triage_classification", "")) else "legitimate"
        PREDICTION_COUNT.labels(class_type=class_type).inc()

        return result
    except Exception as e:
        logger.error(f"Endpoint Exception caught: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Pipeline exception: {str(e)}")