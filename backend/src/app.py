import os
import sys
import logging
import traceback
import time
import asyncio
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("GatewayDeployment")

# ------------------------------------------------------------------
# PATH RESOLUTION & SYS.PATH SETUP
# ------------------------------------------------------------------
# Location of this file: /backend/src/app.py
SRC_DIR = os.path.dirname(os.path.abspath(__file__))               # /backend/src
BACKEND_DIR = os.path.dirname(SRC_DIR)                              # /backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                             # Project Root

# Set up directory references
AGENTS_DIR = SRC_DIR                                                # /backend/src
MONITORING_DIR = os.path.join(SRC_DIR, "mlops")                     # /backend/src/mlops

# Ensure required modules are discoverable in sys.path
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
# SCHEMAS & APP INITIALIZATION
# ------------------------------------------------------------------
class CustomsDeclarationInput(BaseModel):
    file_id: int = Field(..., description="Unique structural node reference key", example=10016)
    numero_dossier: str = Field(..., description="Alpha-numeric container identifier", example="EP052469EX01DM")
    emetteur: str = Field(..., description="Corporate entity exporting goods", example="OLAMCAM")
    valeur_fob: float = Field(..., description="FOB Cargo evaluation in CFA", example=439227.84)
    is_anomaly_ip: float = Field(..., description="Tabular 5-model consensus metric outcome", example=1.0)

app = FastAPI(
    title="GUCE Multi-Modal Fraud Triage Gateway",
    description="Microservice exposing your Tabular Ensemble + Heterogeneous Graph Neural Network multi-agent pipeline.",
    version="1.0.0"
)

engine = None

# ------------------------------------------------------------------
# LIFECYCLE & ENDPOINTS
# ------------------------------------------------------------------
@app.on_event("startup")
def initialize_engine():
    global engine

    try:
        start_monitoring_server(port=8001)
        print("📊 [API-STARTUP] Metrics server live on :8001/metrics")
    except OSError as e:
        print(f"⚠️ [API-STARTUP] Metrics server not started: {e}")

    # Paths resolved relative to BACKEND_DIR (/backend)
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
    except Exception as e:
        print("\n" + "="*80)
        print("🔥 CRITICAL INTERNAL RUNTIME EXCEPTION UNCOVERED:")
        traceback.print_exc()
        print("="*80 + "\n")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Core Error: {str(e)}"}
        )

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Infrastructure"])
def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Analytics layer initializing or corrupted.")
    return {"status": "healthy", "engine_loaded": True, "model_metadata": getattr(engine, "model_metadata", {})}

@app.post("/api/v1/triage/evaluate", status_code=status.HTTP_200_OK, tags=["Customs Analytics"])
async def evaluate_declaration(payload: CustomsDeclarationInput):
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_AVAILABLE, detail="Inference engine is offline.")

    try:
        tx_data = payload.dict()

        start = time.perf_counter()
        # Offload blocking multi-agent / GNN inference to a thread
        result = await asyncio.to_thread(engine.evaluate_transaction, tx_data)
        INFERENCE_LATENCY.observe(time.perf_counter() - start)

        verdict = result.get("verdict") or result.get("prediction") or {}
        class_type = "fraud" if isinstance(verdict, dict) and ("TIER 1" in verdict.get("triage_classification", "") or "RED" in verdict.get("triage_classification", "")) else "legitimate"
        PREDICTION_COUNT.labels(class_type=class_type).inc()

        return result
    except Exception as e:
        logger.error(f"Endpoint Exception caught: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Pipeline exception: {str(e)}")