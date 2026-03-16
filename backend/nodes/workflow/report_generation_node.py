from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import GraphState
from backend.llm.hf_client import get_chat_llm


class ReportGenerationNode:
    """
    Generates a structured workflow execution report using an LLM.

    Runs AFTER execution nodes (white-box / black-box)
    and BEFORE evaluation.

    The context passed to the LLM is built entirely from real pipeline
    outputs — no invented data.
    """

    def __init__(self):
        self.llm = get_chat_llm()

    # ----------------------------------------------------------
    # Context builder
    # ----------------------------------------------------------

    def _build_context(self, state: GraphState) -> str:
        workflow_input = state.workflow_input

        context = f"""
WORKFLOW CONFIGURATION (RAW):
{state.user_message}

PARSED INPUTS:
{workflow_input.model_dump() if workflow_input else "Not available"}

EXECUTION MODE:
{workflow_input.execution_mode if workflow_input else "Not available"}

ROUTER DECISION:
{state.route if state.route else "Not applicable"}

WHITE-BOX RESULT:
{state.white_box_result if state.white_box_result else "Not executed"}

BLACK-BOX RESULT:
{state.black_box_result if state.black_box_result else "Not executed"}
        """.strip()

        # ----------------------------------------------------------
        # Build-mode black-box context
        # Replaces the old api_matches / infer-mode block entirely.
        # Raw output now contains: metrics, images, dataset_name,
        # model_name, similarity, explainer — all from the build pipeline.
        # ----------------------------------------------------------
        if state.black_box_result and state.black_box_result.raw_output:
            raw = state.black_box_result.raw_output

            # ---- Configuration ----
            config_lines = ["\nBLACK-BOX BUILD CONFIGURATION:"]
            for key in ("dataset_name", "model_name", "similarity", "explainer"):
                if key in raw:
                    config_lines.append(f"  {key}: {raw[key]}")
            context += "\n" + "\n".join(config_lines)

            # ---- Retrieval metrics ----
            metrics_block = raw.get("metrics", {})
            # Handle both flat {"top1_accuracy": 0.9} and
            # nested {"metrics": {"top1_accuracy": 0.9}} formats
            metric_values = (
                metrics_block.get("metrics", metrics_block)
                if isinstance(metrics_block, dict)
                else {}
            )

            if metric_values:
                metric_lines = ["\nRETRIEVAL METRICS:"]
                for k, v in metric_values.items():
                    if isinstance(v, float):
                        metric_lines.append(f"  {k}: {v:.4f}")
                    else:
                        metric_lines.append(f"  {k}: {v}")
                context += "\n" + "\n".join(metric_lines)
            else:
                context += "\nRETRIEVAL METRICS: Not available"

            # ---- LIME explanation images ----
            images = raw.get("images", [])
            if images:
                image_lines = [f"\nLIME EXPLANATION IMAGES ({len(images)} generated):"]
                for img in images:
                    image_lines.append(
                        f"  image_id: {img.get('image_id', 'unknown')}  "
                        f"overlay: {img.get('overlay_url', '')}  "
                        f"heatmap: {img.get('heatmap_url', '')}"
                    )
                context += "\n" + "\n".join(image_lines)
            else:
                context += "\nLIME EXPLANATION IMAGES: None generated"

        return context

    # ----------------------------------------------------------
    # Node entry point
    # ----------------------------------------------------------

    def __call__(self, state: GraphState) -> Dict[str, Any]:

        context = self._build_context(state)

        system_prompt = (
            "You are a technical reporting assistant for an AI explainability workflow system.\n"
            "Generate a clear, neutral, and factual execution report strictly "
            "based on the provided workflow context.\n\n"
            "Do NOT invent results.\n"
            "Do NOT assume missing data.\n"
            "If outputs are placeholders or incomplete, explicitly state this."
        )

        human_prompt = (
            "Using the workflow context below, generate a structured execution report "
            "with the following sections:\n\n"
            "1. Overview\n"
            "2. Workflow Configuration Summary\n"
            "3. Execution Path Taken\n"
            "4. Analysis Results\n"
            "   - White-box: Summarise if executed, otherwise state not executed.\n"
            "   - Black-box: Using the retrieval metrics provided, report the exact "
            "     metric values (top-1 accuracy, top-5 accuracy, MRR where available). "
            "     Comment on what the similarity gap and top-2 score indicate about the "
            "     quality of the index. State how many LIME explanation images were "
            "     generated and what their presence indicates about the explainability "
            "     pipeline completing successfully.\n"
            "5. Limitations and Assumptions\n"
            "6. Conclusion\n\n"
            f"Workflow Context:\n{context}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(messages)

        state.report = {
            "human_readable": response.content
        }

        return state