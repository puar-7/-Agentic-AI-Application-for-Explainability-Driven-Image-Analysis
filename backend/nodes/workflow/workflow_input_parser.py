# Entry node of the workflow graph
# Reads workflow_input from GraphState and writes parsed config back to GraphState
#  need to handle chat response of error 
import re
from typing import Dict

from backend.graph.state import GraphState
from backend.schemas.workflow_input import WorkflowInput
import os

class WorkflowInputParserNode:
    """
    Parses structured workflow configuration from chat input
    using strict, rule-based extraction.
    """

    FIELD_PATTERNS = {
        "dataset_path": r"dataset path\s*-\s*(.+)",
        "model_path": r"model path\s*-\s*(.+)",
        "target_variable": r"target variable\s*-\s*(.+)",
        "spurious_attribute": r"spurious attribute\s*-\s*(.+)",
        "execution_mode": r"execution mode\s*-\s*(.+)",
    }

    ALLOWED_EXECUTION_MODES = {"white", "black", "both"}

    def __call__(self, state: GraphState) -> Dict:
        if not state.user_message:
            msg = "No workflow configuration found in the message."
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        extracted = {}

        for field, pattern in self.FIELD_PATTERNS.items():
            match = re.search(pattern, state.user_message, re.IGNORECASE)
            if not match:
                msg = f"Missing or invalid field: {field.replace('_', ' ')}"
                return {
                    "error": msg,
                    "chat_response": f"Workflow configuration error: {msg}"
                }
            extracted[field] = match.group(1).strip()


        # Check if dataset file exists
        if not os.path.exists(extracted["dataset_path"]):
            msg = f"Dataset file not found at: {extracted['dataset_path']}"
            return {
                "error": msg,
                "chat_response": f"File Error: {msg}"
            }

        # Check if model file exists
        if not os.path.exists(extracted["model_path"]):
            msg = f"Model file not found at: {extracted['model_path']}"
            return {
                "error": msg,
                "chat_response": f"File Error: {msg}"
            }    

        execution_mode = extracted["execution_mode"].lower()
        if execution_mode not in self.ALLOWED_EXECUTION_MODES:
            msg=(f"Invalid execution mode: {execution_mode}. "
                  f"Allowed values: {self.ALLOWED_EXECUTION_MODES}")
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        workflow_input = WorkflowInput(
            dataset_path=extracted["dataset_path"],
            model_path=extracted["model_path"],
            target_variable=extracted["target_variable"],
            spurious_attribute=extracted["spurious_attribute"],
                execution_mode=execution_mode,
            )

        return {
            "workflow_input": workflow_input
        }
