from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from backend.graph.state import GraphState
from backend.llm.hf_client import get_chat_llm


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

        return f"""
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
            "5. Limitations and Assumptions\n"
            "6. Conclusion\n\n"
            "Then also produce a single, coherent human-readable report that combines "
            "all sections into a continuous narrative.\n\n"
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
            "human_readable": response.content,
            "sections": {
                "overview": "",
                "configuration_summary": "",
                "execution_path": "",
                "analysis_results": "",
                "limitations": "",
                "conclusion": "",
            },
        }

        return state
