from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.graph.state import GraphState
from backend.schemas.execution_result import ExecutionResult
from backend.persistence.snapshot_builder import build_execution_snapshot
from backend.persistence.snapshot_validator import validate_snapshot

router = APIRouter()


class WorkflowRequest(BaseModel):
    config_text: str


def make_json_safe(obj: Any) -> Any:
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
async def run_workflow(request: WorkflowRequest, http_request: Request):
    """
    Async workflow endpoint.

    unified_graph.run() is now async and uses LangGraph's ainvoke()
    internally, so BlackBoxNode's asyncio.sleep polling works correctly.
    No run_in_executor needed — the entire call chain is properly async.
    """

    if not request.config_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Workflow configuration cannot be empty.",
        )

    unified_graph = http_request.app.state.unified_graph

    state = GraphState(
        mode="workflow",
        user_message=request.config_text,
    )

    # Direct await — no thread executor needed
    result = await unified_graph.run(state)

    # graph.run() returns a GraphState object — convert to dict
    if isinstance(result, GraphState):
        result = result.dict()

    # ----------------------------------------------------------
    # MongoDB persistence — identical to original
    # ----------------------------------------------------------
    try:
        if result.get("error") is None:
            serializable_result = make_json_safe(result)
            snapshot = build_execution_snapshot(
                raw_input_text=request.config_text,
                parsed_input=result.get("parsed_input"),
                final_state=serializable_result,
                execution_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            validate_snapshot(snapshot)

            mongodb = http_request.app.state.mongodb
            if mongodb is not None and mongodb.db is not None:
                mongodb.db.workflow_executions.insert_one(snapshot)

    except Exception as e:
        print(f"[Persistence] Snapshot insert failed (non-fatal): {e}")

    # ----------------------------------------------------------
    # Error response — identical to original
    # ----------------------------------------------------------
    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail=result["error"],
        )

    return result