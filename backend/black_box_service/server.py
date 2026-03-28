import os
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

# Import adapter functions
from .adapter import (
    run_blackbox_build,
    run_blackbox_inference,
    read_job,
    safe_workspace,
    JOBS_DIR,
)
from inference import Inference

# ----------------------------------------------------------
# NEW: import the MCP server instance from mcp_tools
# ----------------------------------------------------------
from .mcp_tools import mcp as black_box_mcp

app = FastAPI(title="Black-Box Integration Service")

# Shared workspace — all build outputs, job files, embeddings live here
WORKSPACE_DIR = os.path.abspath("./backend/storage/shared_workspace")

# ----------------------------------------------------------
# NEW: mount MCP server at /mcp
# Exposes SSE endpoint at  GET  /mcp/sse
# Message endpoint at      POST /mcp/messages/
# The existing REST endpoints below are completely unaffected.
# ----------------------------------------------------------
mcp_app = black_box_mcp.http_app(transport='sse')
app.mount("/mcp", mcp_app)


# ---------------------------------------------------------
# SINGLETON STATE (Heavy model held in RAM after startup)
# ---------------------------------------------------------
class AppState:
    inference_engine: Inference = None
    is_loaded: bool = False

state = AppState()


# ---------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------
class BuildRequest(BaseModel):
    dataset_name: str = "CELEBA"
    model_name: str = "RESNET"
    similarity: str = "COSINE"
    explainer: str = "LIME"


class InferRequest(BaseModel):
    query_image_path: str


# ---------------------------------------------------------
# STARTUP — Create directories and load model into RAM
# Unchanged from original.
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    jobs_abs = os.path.join(WORKSPACE_DIR, JOBS_DIR)
    os.makedirs(jobs_abs, exist_ok=True)
    print(f"[Server 8001] Jobs directory ready: {jobs_abs}")

    print("[Server 8001] Loading models into RAM...")
    with safe_workspace(WORKSPACE_DIR):
        try:
            state.inference_engine = Inference(
                dataset_name="CELEBA",
                model_name="RESNET",
                similarity="COSINE",
                top_k=5,
            )
            state.inference_engine.load_system()
            state.is_loaded = True
            print("[Server 8001] Models loaded successfully.")
        except Exception as e:
            print(
                f"[Server 8001] Could not load models on startup "
                f"(run /build first): {e}"
            )

    print("[Server 8001] MCP server mounted at /mcp  (SSE: GET /mcp/sse)")


# ---------------------------------------------------------
# ENDPOINT: BUILD  POST /build
# Kept intact — useful for direct testing without an MCP client.
# ---------------------------------------------------------
@app.post("/build")
async def trigger_build(req: BuildRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    print(f"[Server 8001] Build requested via REST — job_id={job_id}")

    background_tasks.add_task(
        run_blackbox_build,
        workspace_dir=WORKSPACE_DIR,
        job_id=job_id,
        dataset_name=req.dataset_name,
        model_name=req.model_name,
        similarity=req.similarity,
        explainer=req.explainer,
    )

    return {
        "status":  "processing",
        "job_id":  job_id,
        "message": (
            "Build started in the background. "
            f"Poll GET /status/{job_id} to track progress."
        ),
    }


# ---------------------------------------------------------
# ENDPOINT: STATUS  GET /status/{job_id}
# Kept intact — useful for direct testing without an MCP client.
# ---------------------------------------------------------
@app.get("/status/{job_id}")
async def get_build_status(job_id: str):
    job_data = read_job(WORKSPACE_DIR, job_id)

    if job_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job found with id '{job_id}'.",
        )

    return job_data


# ---------------------------------------------------------
# ENDPOINT: INFER  POST /infer
# Unchanged from original.
# ---------------------------------------------------------
@app.post("/infer")
async def trigger_inference(req: InferRequest):
    if not state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not loaded. "
                "Run POST /build first, then restart the server."
            ),
        )

    if not os.path.exists(req.query_image_path):
        raise HTTPException(
            status_code=400,
            detail=f"Image not found at path: {req.query_image_path}",
        )

    with safe_workspace(WORKSPACE_DIR):
        results = state.inference_engine.predict(req.query_image_path)

    return {
        "status":  "success",
        "matches": results,
    }


# ---------------------------------------------------------
# Run independently:
# uvicorn backend.black_box_service.server:app --port 8001 --reload
# ---------------------------------------------------------