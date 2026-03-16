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

app = FastAPI(title="Black-Box Integration Service")

# Shared workspace — all build outputs, job files, embeddings live here
WORKSPACE_DIR = os.path.abspath("./backend/storage/shared_workspace")


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
    explainer: str = "LIME"          # Was missing in original — added to match adapter


class InferRequest(BaseModel):
    query_image_path: str


# ---------------------------------------------------------
# STARTUP — Create directories and load model into RAM
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    # Ensure the jobs directory exists before any request comes in.
    # This is safe to call repeatedly.
    jobs_abs = os.path.join(WORKSPACE_DIR, JOBS_DIR)
    os.makedirs(jobs_abs, exist_ok=True)
    print(f"[Server 8001] Jobs directory ready: {jobs_abs}")

    # Load heavy model into RAM once
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
            # Non-fatal on startup — /build hasn't run yet on a fresh machine
            print(
                f"[Server 8001] Could not load models on startup "
                f"(run /build first): {e}"
            )


# ---------------------------------------------------------
# ENDPOINT: BUILD  POST /build
# Fires the heavy pipeline as a background task.
# Returns job_id immediately so the caller can poll /status.
# ---------------------------------------------------------
@app.post("/build")
async def trigger_build(req: BuildRequest, background_tasks: BackgroundTasks):
    """
    Kicks off the full build pipeline (embedding extraction, FAISS index,
    evaluation, LIME explanations) as a background task.

    Returns a job_id immediately. Poll GET /status/{job_id} to track progress.
    """
    job_id = str(uuid.uuid4())

    print(f"[Server 8001] Build requested — job_id={job_id} | "
          f"dataset={req.dataset_name} model={req.model_name} "
          f"similarity={req.similarity} explainer={req.explainer}")

    # Fire the background task — passes job_id so adapter can write status files
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
        "status": "processing",
        "job_id": job_id,
        "message": (
            "Build started in the background. "
            f"Poll GET /status/{job_id} to track progress."
        ),
    }


# ---------------------------------------------------------
# ENDPOINT: STATUS  GET /status/{job_id}
# Reads the job file written by the adapter and returns it.
# This is the only endpoint the LangGraph node polls.
# ---------------------------------------------------------
@app.get("/status/{job_id}")
async def get_build_status(job_id: str):
    """
    Returns the current status of a build job.

    Possible status values:
        processing  — pipeline is still running
        completed   — pipeline finished; metrics and image URLs are included
        failed      — pipeline crashed; error message is included

    Returns 404 if the job_id is not recognised.
    """
    job_data = read_job(WORKSPACE_DIR, job_id)

    if job_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job found with id '{job_id}'. "
                   "It may not have started yet or the id is incorrect.",
        )

    return job_data


# ---------------------------------------------------------
# ENDPOINT: INFER  POST /infer
# Unchanged from original — uses pre-loaded RAM model.
# ---------------------------------------------------------
@app.post("/infer")
async def trigger_inference(req: InferRequest):
    """
    Uses the pre-loaded model to search a query image against the FAISS index.
    Requires /build to have been run at least once before startup.
    """
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
        "status": "success",
        "matches": results,
    }


# ---------------------------------------------------------
# Run independently:
# uvicorn backend.black_box_service.server:app --port 8001 --reload
# ---------------------------------------------------------