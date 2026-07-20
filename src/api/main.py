# src/api/main.py

import os
import yaml
import pandas as pd
import time
import hashlib
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.responses import Response
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from dotenv import load_dotenv
load_dotenv()

# Database persistence layers
from src.database.service import init_db, save_inference

# Architectural components
from src.rag.agents import IntelliLoanAgent
from src.api.schemas import LoanApplicationRequest, AppraisalResponse
from config.logging import get_logger

# Initialize corporate logger
logger = get_logger(__name__)

# Global variable tracking
root_dir = Path(__file__).resolve().parents[2]
state = {"agent": None, "config": None, "modeling_data": None}

# --------------------------------------------------------------------------
# LIFESPAN MANAGEMENT (Startup & Shutdown)
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook: Bootstraps models, DB tables, and Gold datasets."""
    try:
        logger.info("[API Startup] Initializing Corporate Intelligence Runtime...")
        
        # 1. Load Configurations
        config_path = root_dir / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        state["config"] = config

        # 2. Initialize PostgreSQL Table from src/database/service.py
        init_db()
        
        # 3. Load ML & RAG Engines
        state["agent"] = IntelliLoanAgent(config)
        
        # 4. Cache Modeling Matrix for the Analyst Endpoint
        data_path = root_dir / config["data_paths"]["processed"] / config["tables"]["modeling_matrix"]
        state["modeling_data"] = pd.read_parquet(data_path)
        
        logger.info(f"[API Startup] System ready. Database synced and {len(state['modeling_data'])} records cached.")
        yield
    finally:
        state.clear()
        logger.info("[API Shutdown] Releasing resources.")

app = FastAPI(
    title="IntelliLoan Agentic API",
    description="Observable Decision Engine with Integrated Inference Store.",
    version="0.3.0",
    lifespan=lifespan
)

# --------------------------------------------------------------------------
# OBSERVABILITY: PROMETHEUS METRICS
# --------------------------------------------------------------------------
# Reset registry to avoid duplicate metrics on reload
for collector in list(REGISTRY._collector_to_names.keys()):
    REGISTRY.unregister(collector)

APPRAISAL_COUNT = Counter(
    "loan_appraisals_total", "Total count of processed loans", ["strategy", "verdict"]
)
APPRAISAL_LATENCY = Histogram(
    "loan_appraisal_seconds", "Full agentic pipeline execution time"
)
DB_SAVE_COUNT = Counter("db_persistence_total", "Status of SQL archiving", ["status"])

# --------------------------------------------------------------------------
# SECURITY & A/B ROUTING
# --------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-KEY")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Enforces API Key security."""
    if api_key != os.getenv("API_KEY_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid API Key")

def get_ab_strategy(client_id: int) -> str:
    """Deterministically routes clients based on ID hashing."""
    hash_val = int(hashlib.md5(str(client_id).encode()).hexdigest(), 16)
    return "B" if (hash_val % 2 == 0) else "A"

# --------------------------------------------------------------------------
# CORE PRODUCTION ROUTES
# --------------------------------------------------------------------------

@app.get("/metrics", tags=["System"])
def metrics():
    """Prometheus technical metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/v1/health", tags=["System"])
def health():
    """Liveness probe for Cloud Infrastructure."""
    return {"status": "online", "model_version": "0.3.0"}

@app.post("/v1/appraise/id/{client_id}", response_model=AppraisalResponse, tags=["Analyst View"])
async def appraise_by_id(
    client_id: int, 
    background_tasks: BackgroundTasks, # For non-blocking DB save
    _=Depends(verify_api_key)
):
    """
    PRIMARY ANALYST ENDPOINT:
    Executes the full reasoning loop and archives the result in the Inference Store.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # 1. Data Retrieval
        df = state["modeling_data"]
        main_key = state["config"]["fusion"]["main_key"]
        client_row = df[df[main_key] == client_id]

        if client_row.empty:
            raise HTTPException(status_code=404, detail=f"Client ID {client_id} not found.")

        # 2. A/B Strategy Assignment
        strategy = get_ab_strategy(client_id)
        
        # 3. Agent Execution
        features = client_row.drop(columns=['TARGET'], errors='ignore')
        result = state["agent"].run_inference(client_id, features, strategy=strategy)

        # 4. Prepare Persistence Payload
        inference_record = {
            "request_id": request_id,
            "client_id": client_id,
            "credit_score": result["ml_results"]["credit_score"],
            "verdict": result["ml_results"]["verdict"],
            "risk_level": result["ml_results"]["risk_level"],
            "probability_of_default": result["ml_results"]["probability_of_default"],
            "strategy_used": strategy,
            "final_memo": result["final_memo"],
            "top_risk_drivers": result["shap_reasons"]
        }

        # 5. BACKGROUND TASK: Save to Neon database
        # We don't wait for the DB to answer, we send the result to the analyst immediately.
        background_tasks.add_task(save_inference, inference_record)

        # 6. Observability
        duration = time.time() - start_time
        APPRAISAL_LATENCY.observe(duration)
        APPRAISAL_COUNT.labels(strategy=strategy, verdict=result["ml_results"]["verdict"]).inc()
        
        return AppraisalResponse(**inference_record)

    except HTTPException: raise
    except Exception as e:
        logger.error(f"Inference failure for {client_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal decision engine failure.")

@app.post("/v1/demo/validate", response_model=AppraisalResponse, tags=["Compliance Demo"])
async def tech_demo_validation(payload: LoanApplicationRequest, _=Depends(verify_api_key)):
    """Validates the input Data Contract before running the model."""
    try:
        df_row = pd.DataFrame([payload.features])
        strategy = get_ab_strategy(payload.SK_ID_CURR)
        result = state["agent"].run_inference(payload.SK_ID_CURR, df_row, strategy=strategy)
        
        return AppraisalResponse(client_id=payload.SK_ID_CURR, **result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data Contract Violation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)