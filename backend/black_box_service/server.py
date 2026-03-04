import os
import uuid #for generating unique job IDs
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

# Import our Adapter 
from .adapter import run_blackbox_build, run_blackbox_inference, safe_workspace
from inference import Inference

app = FastAPI(title="Black-Box Integration Service")

# We define the shared workspace we established
WORKSPACE_DIR = os.path.abspath("./backend/storage/shared_workspace")

# ---------------------------------------------------------
# SINGLETON STATE (To hold the heavy model in RAM)
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

class InferRequest(BaseModel):
    query_image_path: str  # Absolute path to the uploaded image

# ---------------------------------------------------------
# STARTUP EVENT (The Memory Fix)
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    """
    Loads the heavy model and FAISS index into memory exactly ONCE 
    when the FastAPI server starts.
    """
    print("🚀 Starting Black-Box API: Loading heavy models into RAM...")
    
    # We must run this inside the safe_workspace so her inference code 
    # can find the 'embeddings' and 'faiss_indexes' folders!
    with safe_workspace(WORKSPACE_DIR):
        try:
            # Note: We are hardcoding the expected inputs for now. 
            # In the future, this could be dynamic.
            state.inference_engine = Inference(
                dataset_name="CELEBA",
                model_name="RESNET",
                similarity="COSINE",
                top_k=5
            )
            state.inference_engine.load_system()
            state.is_loaded = True
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load models on startup. Did you run /build yet? Error: {e}")

# ---------------------------------------------------------
# ENDPOINT: BUILD (Asynchronous)
# ---------------------------------------------------------
@app.post("/build")
async def trigger_build(req: BuildRequest, background_tasks: BackgroundTasks):
    """
    Kicks off the heavy dataset processing in the background so 
    the LangGraph API request doesn't timeout.
    """
    job_id = str(uuid.uuid4())
    
    # Run the wrapper function we built in Phase 1 in the background
    background_tasks.add_task(
        run_blackbox_build,
        workspace_dir=WORKSPACE_DIR,
        dataset_name=req.dataset_name,
        model_name=req.model_name,
        similarity=req.similarity
    )
    
    return {
        "status": "processing",
        "job_id": job_id,
        "message": "Build started in the background. This may take a while."
    }

# ---------------------------------------------------------
# ENDPOINT: INFER (Synchronous & Fast)
# ---------------------------------------------------------
@app.post("/infer")
async def trigger_inference(req: InferRequest):
    """
    Uses the pre-loaded RAM model to instantly search the query image.
    """
    if not state.is_loaded:
        raise HTTPException(
            status_code=500, 
            detail="Models are not loaded. You may need to run /build first."
        )
        
    if not os.path.exists(req.query_image_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Image not found at path: {req.query_image_path}"
        )

    # Run inference securely inside our workspace context
    with safe_workspace(WORKSPACE_DIR):
        results = state.inference_engine.predict(req.query_image_path)

    return {
        "status": "success",
        "matches": results
    }

# To run this server independently:
# uvicorn backend.black_box_service.server:app --port 8001 --reload