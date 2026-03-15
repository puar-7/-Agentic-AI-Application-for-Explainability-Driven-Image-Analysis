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

    REQUIRED_PATTERNS = {
        "dataset_name":    r"dataset name\s*-\s*(.+)",
        "model_name":      r"model name\s*-\s*(.+)",
        "execution_mode":  r"execution mode\s*-\s*(.+)",
    }

    OPTIONAL_PATTERNS = {
        "target_variable":   r"target variable\s*-\s*(.+)",
        "spurious_attribute": r"spurious attribute\s*-\s*(.+)",
        "similarity":        r"similarity\s*-\s*(.+)",
        "explainer":         r"explainer\s*-\s*(.+)",
    }

    ALLOWED_EXECUTION_MODES = {"white", "black", "both"}
    ALLOWED_DATASETS = {"CELEBA", "VGGFACE2", "DIGIFACE"}
    ALLOWED_MODELS = {"RESNET", "FACENET"}
    ALLOWED_SIMILARITIES = {"COSINE", "EUCLIDEAN"}
    ALLOWED_EXPLAINERS = {"LIME", "SHAP", "RISE"}

    def __call__(self, state: GraphState) -> Dict:
        if not state.user_message:
            msg = "No workflow configuration found in the message."
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        extracted = {}

        # Required fields 
        for field, pattern in self.REQUIRED_PATTERNS.items():
            match = re.search(pattern, state.user_message, re.IGNORECASE)
            if not match:
                msg = f"Missing or invalid field: {field.replace('_', ' ')}"
                return {
                    "error": msg,
                    "chat_response": f"Workflow configuration error: {msg}"
                }
            extracted[field] = match.group(1).strip()


        # --- Optional fields ---
        for field, pattern in self.OPTIONAL_PATTERNS.items():
            match = re.search(pattern, state.user_message, re.IGNORECASE)
            if match:
                extracted[field] = match.group(1).strip()

        # --- Validate dataset_name ---
        dataset_name = extracted["dataset_name"].upper()
        if dataset_name not in self.ALLOWED_DATASETS:
            msg = (f"Invalid dataset name: '{dataset_name}'. "
                   f"Allowed: {self.ALLOWED_DATASETS}")
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        # --- Validate model_name ---
        model_name = extracted["model_name"].upper()
        if model_name not in self.ALLOWED_MODELS:
            msg = (f"Invalid model name: '{model_name}'. "
                   f"Allowed: {self.ALLOWED_MODELS}")
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        # --- Validate execution_mode ---
        execution_mode = extracted["execution_mode"].lower()
        if execution_mode not in self.ALLOWED_EXECUTION_MODES:
            msg = (f"Invalid execution mode: '{execution_mode}'. "
                   f"Allowed: {self.ALLOWED_EXECUTION_MODES}")
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        # --- Validate optional similarity ---
        similarity = extracted.get("similarity", "COSINE").upper()
        if similarity not in self.ALLOWED_SIMILARITIES:
            msg = (f"Invalid similarity: '{similarity}'. "
                   f"Allowed: {self.ALLOWED_SIMILARITIES}")
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        # --- Validate optional explainer ---
        explainer = extracted.get("explainer", "LIME").upper()
        if explainer not in self.ALLOWED_EXPLAINERS:
            msg = (f"Invalid explainer: '{explainer}'. "
                   f"Allowed: {self.ALLOWED_EXPLAINERS}")
            return {
                "error": msg,
                "chat_response": f"Workflow configuration error: {msg}"
            }

        workflow_input = WorkflowInput(
            dataset_name=dataset_name,
            model_name=model_name,
            target_variable=extracted.get("target_variable"),
            spurious_attribute=extracted.get("spurious_attribute"),
            execution_mode=execution_mode,
            similarity=similarity,
            explainer=explainer,
        )

        return {
            "workflow_input": workflow_input
        }