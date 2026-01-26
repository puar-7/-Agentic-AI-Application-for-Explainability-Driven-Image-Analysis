from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi import Request

from backend.graph.workflow_graph import WorkflowGraph
from backend.graph.state import GraphState
from datetime import datetime, timezone
from backend.schemas.execution_result import ExecutionResult

from backend.persistence.snapshot_builder import build_execution_snapshot
from backend.persistence.snapshot_validator import validate_snapshot
router = APIRouter()


class WorkflowRequest(BaseModel):
    config_text: str


def make_json_safe(obj):
    """
    Recursively convert ExecutionResult (and nested structures)
    into JSON-serializable Python primitives.
    """
    if isinstance(obj, ExecutionResult):
        return obj.dict()

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    return obj


@router.post("/")
def run_workflow(request: WorkflowRequest, http_request: Request):
    unified_graph = http_request.app.state.unified_graph
    if not request.config_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Workflow configuration cannot be empty."
        )

    state = GraphState(
        mode="workflow",
        user_message=request.config_text
    )



    result = unified_graph.run(state)
    if isinstance(result, GraphState):
         # If run() returns GraphState object, convert to dict for your existing logic
         result = result.dict()

    # --------------------------------------------------
    # Persistence Slot 
    # --------------------------------------------------
    try:
        # Persist ONLY successful executions
        if result.get("error") is None:
            serializable_result = make_json_safe(result)
            snapshot = build_execution_snapshot(
                raw_input_text=request.config_text,
                parsed_input=result.get("parsed_input"),
                final_state=serializable_result,
                execution_timestamp=datetime.now(timezone.utc).isoformat()

            )

            # Validate snapshot before DB insert
            validate_snapshot(snapshot)

            # Insert into MongoDB (best-effort)
            mongodb = http_request.app.state.mongodb
            if mongodb is not None and mongodb.db is not None:
                mongodb.db.workflow_executions.insert_one(snapshot)

    except Exception as e:
        # Persistence must NEVER break workflow execution
        print(f"[Persistence] Snapshot insert failed (non-fatal): {e}")


    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail=result["error"]
        )

    return result
