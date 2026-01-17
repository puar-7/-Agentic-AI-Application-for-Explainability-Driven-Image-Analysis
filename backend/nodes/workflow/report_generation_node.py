from typing import Dict
from backend.graph.state import GraphState


class ReportGenerationNode:
    """
    Generates a consolidated workflow report
    from execution results and user inputs.
    """

    def __call__(self, state: GraphState) -> Dict:
        report = {
            "summary": "Workflow execution report",
            "inputs": state.workflow_input.model_dump()
            if state.workflow_input else None,
            "white_box_result": state.white_box_result,
            "black_box_result": state.black_box_result,
            "error": state.error,
        }

        return {
            "report": report
        }
