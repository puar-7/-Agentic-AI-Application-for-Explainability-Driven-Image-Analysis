from langgraph.graph import StateGraph, END
from backend.graph.state import GraphState

from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.nodes.chat.retrieval_grader_node import RetrievalGraderNode
from backend.nodes.chat.web_search_node import WebSearchNode
from backend.nodes.workflow.workflow_input_parser import WorkflowInputParserNode
from backend.nodes.workflow.router_node import RouterNode
from backend.nodes.workflow.white_box_node import WhiteBoxNode
from backend.nodes.workflow.black_box_node import BlackBoxNode
from backend.nodes.workflow.parallel_node import ParallelExecutionNode
from backend.nodes.workflow.report_generation_node import ReportGenerationNode
from backend.nodes.workflow.evaluation_node import EvaluationNode


def _grade_router(state: GraphState) -> str:
    """
    Routes after RetrievalGraderNode based on the grade.

        correct   → skip web search, go straight to generation
        incorrect → run web search (docs discarded by WebSearchNode)
        ambiguous → run web search (docs kept, merged with web results)
    """
    grade = state.retrieval_grade or "ambiguous"
    if grade == "correct":
        return "correct"
    return "needs_web"   # covers both incorrect and ambiguous


class UnifiedGraph:
    """
    A single graph that handles both Chat and Workflow modes.

    Chat path (with CRAG):
        local_retrieval
            → retrieval_grader
                → [correct]    chat_generation
                → [needs_web]  web_search → chat_generation

    Workflow path: unchanged.
    """

    def __init__(
        self,
        retriever_node: LocalRetrieverNode,
        chat_llm_node: ChatLLMNode,
    ):
        self.retriever_node  = retriever_node
        self.chat_llm_node   = chat_llm_node

        # CRAG nodes
        self.grader_node     = RetrievalGraderNode()
        self.web_search_node = WebSearchNode(max_results=5)

        # Workflow nodes
        self.wf_parser = WorkflowInputParserNode()
        self.wf_router = RouterNode()
        self.white_box = WhiteBoxNode()
        self.black_box = BlackBoxNode()
        self.parallel  = ParallelExecutionNode()
        self.report    = ReportGenerationNode()
        self.eval      = EvaluationNode()

        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        # ── Chat nodes ──────────────────────────────────────────────
        graph.add_node("local_retrieval",  self.retriever_node)
        graph.add_node("retrieval_grader", self.grader_node)
        graph.add_node("web_search",       self.web_search_node)
        graph.add_node("chat_generation",  self.chat_llm_node)

        # ── Workflow nodes ───────────────────────────────────────────
        graph.add_node("parse_input",  self.wf_parser)
        graph.add_node("route",        self.wf_router)
        graph.add_node("parallel",     self.parallel)
        graph.add_node("white_box",    self.white_box)
        graph.add_node("black_box",    self.black_box)
        graph.add_node("report",       self.report)
        graph.add_node("evaluation",   self.eval)

        # ── Entry point ──────────────────────────────────────────────
        graph.set_conditional_entry_point(
            lambda state: state.mode,
            {
                "chat":     "local_retrieval",
                "workflow": "parse_input",
            },
        )

        # ── Chat path ────────────────────────────────────────────────
        graph.add_edge("local_retrieval", "retrieval_grader")

        graph.add_conditional_edges(
            "retrieval_grader",
            _grade_router,
            {
                "correct":    "chat_generation",  # docs good → skip web
                "needs_web":  "web_search",        # incorrect or ambiguous
            },
        )

        graph.add_edge("web_search",      "chat_generation")
        graph.add_edge("chat_generation", END)

        # ── Workflow path ─────────────────────────────────────────────
        graph.add_conditional_edges(
            "parse_input",
            lambda state: "error" if state.error else "ok",
            {"error": END, "ok": "route"},
        )

        graph.add_conditional_edges(
            "route",
            lambda state: state.route,
            {
                "white_path": "white_box",
                "black_path": "black_box",
                "both_path":  "parallel",
            },
        )

        graph.add_edge("parallel",   "white_box")
        graph.add_edge("parallel",   "black_box")
        graph.add_edge("white_box",  "report")
        graph.add_edge("black_box",  "report")
        graph.add_edge("report",     "evaluation")
        graph.add_edge("evaluation", END)

        return graph.compile()

    async def run(self, state: GraphState) -> GraphState:
        result = await self.graph.ainvoke(state)
        return GraphState(**result)