from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict

from backend.schemas.workflow_input import WorkflowInput
from backend.schemas.execution_result import ExecutionResult


class GraphState(BaseModel):
    """
    Global state object passed between LangGraph nodes.
    This is the single source of truth for the entire system.
    """

    # --- Mode control ---
    mode: Literal["chat", "workflow"] = Field(
        ...,
        description="Current operating mode of the system"
    )

    # --- Chat-related state ---
    user_message: Optional[str] = Field(
        default=None,
        description="Latest user message from chat"
    )

    chat_history: List[Any] = Field(
        default_factory=list,
        description="Conversation history"
    )

    uploaded_docs: List[Any] = Field(
        default_factory=list,
        description="Uploaded local documents"
    )

    retrieved_docs: Optional[List[Any]] = Field(
        default=None,
        description="Documents retrieved for chat-based QA"
    )

    chat_response: Optional[str] = Field(
        default=None,
        description="LLM-generated chat response"
    )

    # ------------------------------------------------------------------
    # CRAG fields
    # ------------------------------------------------------------------

    retrieval_grade: Optional[Literal["correct", "incorrect", "ambiguous"]] = Field(
        default=None,
        description=(
            "Relevance grade assigned by RetrievalGraderNode. "
            "correct → use docs only. "
            "incorrect → discard docs, use web only. "
            "ambiguous → merge docs + web."
        )
    )

    web_search_results: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Normalized web search results. Each item is a dict with "
            "keys: content, metadata (source_type, title, url)."
        )
    )

    context_source: Optional[Literal["documents", "web", "hybrid"]] = Field(
        default=None,
        description=(
            "Tells ChatLLMNode where the context came from so it can "
            "adjust the system prompt accordingly."
        )
    )

    # ------------------------------------------------------------------

    # --- Workflow-related state ---
    workflow_input: Optional[WorkflowInput] = Field(
        default=None,
        description="Parsed workflow configuration"
    )

    route: Optional[str] = Field(
        default=None,
        description="Routing decision produced by RouterNode"
    )

    white_box_result: Optional[ExecutionResult] = Field(
        default=None,
        description="Result from white-box analysis"
    )

    black_box_result: Optional[ExecutionResult] = Field(
        default=None,
        description="Result from black-box analysis"
    )

    # --- Post-processing ---
    report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured workflow report"
    )

    evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured evaluation summary"
    )

    # --- Error handling ---
    error: Optional[str] = Field(
        default=None,
        description="Error message, if any node fails"
    )