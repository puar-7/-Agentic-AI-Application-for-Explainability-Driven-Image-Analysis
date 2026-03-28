# backend/black_box_service/mcp_tools.py
#
# Exposes the black-box pipeline as three MCP tools.
# This is a pure interface layer — no ML pipeline code lives here.
# All heavy lifting stays in adapter.py and runner.py, untouched.
#
# Tools:
#   run_build       — fire the full pipeline as a background thread,
#                     return job_id immediately
#   get_job_status  — read the job status file written by adapter.py
#   run_inference   — single-image retrieval against a built FAISS index
#
# The MCP server object (mcp) is imported by server.py and mounted
# at /mcp, exposing an SSE endpoint at /mcp/sse.

import os
import uuid
import threading

from fastmcp import FastMCP

from .adapter import (
    run_blackbox_build,
    run_blackbox_inference,
    read_job,
)

# Must match WORKSPACE_DIR in server.py — both resolve relative to CWD
# which is the project root when uvicorn is launched from there.
WORKSPACE_DIR = os.path.abspath("./backend/storage/shared_workspace")

# ----------------------------------------------------------
# MCP server instance
# ----------------------------------------------------------
mcp = FastMCP(
    name="black-box-pipeline",
    instructions=(
        "Provides tools to run the XAI black-box face recognition pipeline. "
        "Use run_build to start a pipeline job and receive a job_id. "
        "Poll get_job_status until status is 'completed' or 'failed'. "
        "Use run_inference for single-image retrieval after a build."
    ),
)


# ----------------------------------------------------------
# Tool 1 — run_build
# ----------------------------------------------------------

@mcp.tool()
def run_build(
    dataset_name: str,
    model_name: str,
    similarity: str,
    explainer: str,
) -> dict:
    """
    Start the full black-box build pipeline as a background job.

    Steps fired in the background:
        embedding extraction → FAISS indexing → evaluation → LIME explanations

    Returns a job_id immediately. The pipeline may take several minutes.
    Poll get_job_status with this job_id to track progress.

    NOTE: run_blackbox_build uses os.chdir() internally (safe_workspace).
    Do not fire two simultaneous builds — CWD is process-wide.
    """
    job_id = str(uuid.uuid4())

    print(
        f"[MCP:run_build] Starting job_id={job_id} | "
        f"dataset={dataset_name} model={model_name} "
        f"similarity={similarity} explainer={explainer}"
    )

    thread = threading.Thread(
        target=run_blackbox_build,
        kwargs={
            "workspace_dir": WORKSPACE_DIR,
            "job_id":        job_id,
            "dataset_name":  dataset_name,
            "model_name":    model_name,
            "similarity":    similarity,
            "explainer":     explainer,
        },
        daemon=True,   # thread dies with the process — no orphan jobs
    )
    thread.start()

    return {
        "status":  "processing",
        "job_id":  job_id,
        "message": (
            "Build started in background. "
            f"Poll get_job_status with job_id='{job_id}'."
        ),
    }


# ----------------------------------------------------------
# Tool 2 — get_job_status
# ----------------------------------------------------------

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Return the current status of a build job.

    Possible status values:
        processing  — pipeline is still running
        completed   — pipeline finished; metrics and image URLs included
        failed      — pipeline crashed; error message included
        not_found   — job file does not exist yet (race on startup) or
                      the job_id is wrong
    """
    job_data = read_job(WORKSPACE_DIR, job_id)

    if job_data is None:
        return {
            "status":  "not_found",
            "job_id":  job_id,
            "message": (
                "Job file not found. "
                "It may not have been written yet if polled immediately after run_build."
            ),
        }

    return job_data


# ----------------------------------------------------------
# Tool 3 — run_inference
# ----------------------------------------------------------

@mcp.tool()
def run_inference(query_image_path: str) -> dict:
    """
    Run face retrieval inference for a single query image.

    Searches the FAISS index built by a previous run_build call.
    Requires the build to have completed successfully at least once.

    Returns top-k matches with similarity scores and matched image paths.
    """
    if not os.path.exists(query_image_path):
        return {
            "status": "failed",
            "error":  f"Image not found at path: {query_image_path}",
        }

    try:
        result = run_blackbox_inference(
            workspace_dir=WORKSPACE_DIR,
            query_image_path=query_image_path,
        )
        return result

    except FileNotFoundError as e:
        return {
            "status": "failed",
            "error":  f"Index or model not found — run build first: {e}",
        }

    except Exception as e:
        return {
            "status": "failed",
            "error":  f"{type(e).__name__}: {e}",
        }