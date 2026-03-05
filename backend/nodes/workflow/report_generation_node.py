from typing import Dict, Any



from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import GraphState
from backend.llm.hf_client import get_chat_llm
import os 

class ReportGenerationNode:
    """
    Generates a structured workflow execution report using an LLM.

    Runs AFTER execution nodes (router / white-box / black-box)
    and BEFORE evaluation.
    """

    def __init__(self):
        self.llm = get_chat_llm()

    def _build_context(self, state: GraphState) -> str:
        workflow_input = state.workflow_input

        # Basic context (keep existing)
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

        # Add detailed match information if available
        if state.black_box_result and state.black_box_result.raw_output:
            matches = state.black_box_result.raw_output.get("api_matches", [])
            if matches:
                lines = ["\nBLACK‑BOX MATCH DETAILS (top 5):"]
                for m in matches:
                    path = m.get("matched_path", "unknown")
                    score = m.get("score", 0.0)
                    label = m.get("label", "N/A")
                    # Shorten path for readability (just filename)
                    short_path = os.path.basename(path)
                    lines.append(f"  - {short_path}  (score: {score:.4f}, label: {label})")
                context += "\n" + "\n".join(lines)

        return context


    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute report generation and update GraphState.
        """

        context = self._build_context(state)

        system_prompt = (
            "You are a technical reporting assistant for an AI workflow system.\n"
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
    "   - White-box: (summarize if executed)\n"
    "   - Black-box: Based on the retrieved image matches provided below, "
    "     analyze the results. Mention the number of matches, the range of similarity scores, "
    "     any patterns in the filenames or labels (e.g., same identity, different poses), "
    "     and whether the top match appears plausible.\n"
            "5. Limitations and Assumptions\n"
            "6. Conclusion\n\n"
            f"Workflow Context:\n{context}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(messages)

        # For now: store full text as human-readable
        # Sections can be parsed later when structure stabilizes
        state.report = {
            "human_readable": response.content
        }

        return state
