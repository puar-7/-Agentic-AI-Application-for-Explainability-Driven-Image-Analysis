from typing import Dict
from backend.graph.state import GraphState


class EvaluationNode:
    """
    Performs a lightweight evaluation of the workflow execution.
    """

    def __call__(self, state: GraphState) -> Dict:
        if state.error:
            evaluation = {
                "status": "failed",
                "reason": state.error,
                "confidence": "low"
            }
        else:
            evaluation = {
                "status": "completed",
                "confidence": "medium",
                "notes": (
                    "Results are placeholders. "
                    "Real evaluation metrics can be added later."
                )
            }

        return {
            "evaluation": evaluation
        }
