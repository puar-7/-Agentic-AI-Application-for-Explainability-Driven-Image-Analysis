# Used for conditional routing based on workflow_input

from backend.graph.state import GraphState


class RouterNode:
    """
    Determines which workflow execution path to take
    based on user-provided execution_mode.
    """

    def __call__(self, state: GraphState) -> str:
        if not state.workflow_input:
            # This should not happen if parser is correct,
            # but we guard anyway.
            return {
                "error": "Workflow input missing during routing."
            }

        mode = state.workflow_input.execution_mode

        if mode == "white":
            route ="white_path"

        elif mode == "black":
            route = "black_path"

        elif mode == "both":
            route = "both_path"
        else:
            return {
                "error": f"Unknown execution mode: {mode}"
            }

        return {
            "route": route
        }
