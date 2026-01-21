# backend/persistence/snapshot_builder.py

from typing import Dict, Any, Optional


def build_execution_snapshot(
    *,
    raw_input_text: str,
    parsed_input: Optional[Dict[str, Any]],
    final_state: Dict[str, Any],
    execution_timestamp: str
) -> Dict[str, Any]:
    """
    Build an immutable execution snapshot from the final workflow state.
    Pure function: no side effects, no I/O.
    """

    snapshot = {
        "metadata": {
            "execution_timestamp": execution_timestamp
        },
        "input": {
            "raw_text": raw_input_text,
            "parsed": parsed_input
        },
        "nodes": {
            "router": final_state.get("router"),
            "white_box": final_state.get("white_box"),
            "black_box": final_state.get("black_box"),
            "report": final_state.get("report"),
            "evaluation": final_state.get("evaluation"),
        },
        "status": {
            "state": "failed" if final_state.get("error") else "success",
            "error": final_state.get("error")
        }
    }

    return snapshot
