from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.graph.workflow_graph import WorkflowGraph
from backend.graph.state import GraphState

router = APIRouter()


class WorkflowRequest(BaseModel):
    config_text: str


@router.post("/")
def run_workflow(request: WorkflowRequest):
    if not request.config_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Workflow configuration cannot be empty."
        )

    state = GraphState(
        mode="workflow",
        user_message=request.config_text
    )

    graph = WorkflowGraph()
    result = graph.run(state)

    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail=result["error"]
        )

    return result
