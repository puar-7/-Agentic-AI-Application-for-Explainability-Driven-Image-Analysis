from typing import Dict
from backend.graph.state import GraphState
from backend.schemas.execution_result import ExecutionResult


class WhiteBoxNode:
    def __call__(self, state: GraphState) -> Dict:
        if not state.workflow_input:
            return {
                "error": "Workflow input missing for white-box execution."
            }

        result = ExecutionResult(
            method="white_box",
            status="success",
            summary="Placeholder white-box analysis executed successfully.",
            raw_output={
                "dataset_path": state.workflow_input.dataset_path,
                "model_path": state.workflow_input.model_path,
                "target_variable": state.workflow_input.target_variable,
                "spurious_attribute": state.workflow_input.spurious_attribute,
                "note": "This is a dummy white-box result."
            }
        )

        return {
            "white_box_result": result
        }
