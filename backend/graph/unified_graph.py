from langgraph.graph import StateGraph, END
from backend.graph.state import GraphState

from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.nodes.workflow.workflow_input_parser import WorkflowInputParserNode
from backend.nodes.workflow.router_node import RouterNode
from backend.nodes.workflow.white_box_node import WhiteBoxNode
from backend.nodes.workflow.black_box_node import BlackBoxNode
from backend.nodes.workflow.parallel_node import ParallelExecutionNode
from backend.nodes.workflow.report_generation_node import ReportGenerationNode
from backend.nodes.workflow.evaluation_node import EvaluationNode


class UnifiedGraph:
    """
    A single graph that handles both Chat and Workflow modes.
    Routing is determined by state.mode.

    run() is async because BlackBoxNode.__call__ is async.
    LangGraph's ainvoke() properly awaits async nodes.
    """

    def __init__(
        self,
        retriever_node: LocalRetrieverNode,
        chat_llm_node: ChatLLMNode,
    ):
        self.retriever_node = retriever_node
        self.chat_llm_node = chat_llm_node

        # Workflow nodes are stateless — instantiate here
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

        # ---- Register nodes ----
        graph.add_node("local_retrieval", self.retriever_node)
        graph.add_node("chat_generation", self.chat_llm_node)
        graph.add_node("parse_input",     self.wf_parser)
        graph.add_node("route",           self.wf_router)
        graph.add_node("parallel",        self.parallel)
        graph.add_node("white_box",       self.white_box)
        graph.add_node("black_box",       self.black_box)
        graph.add_node("report",          self.report)
        graph.add_node("evaluation",      self.eval)

        # ---- Entry point ----
        graph.set_conditional_entry_point(
            lambda state: state.mode,
            {
                "chat":     "local_retrieval",
                "workflow": "parse_input",
            },
        )

        # ---- Chat path ----
        graph.add_edge("local_retrieval", "chat_generation")
        graph.add_edge("chat_generation", END)

        # ---- Workflow path ----
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

    # ----------------------------------------------------------
    # async run — uses ainvoke so async nodes are properly awaited
    # ----------------------------------------------------------
    async def run(self, state: GraphState) -> GraphState:
        """
        Invoke the graph asynchronously.

        ainvoke() is LangGraph's async counterpart to invoke().
        It correctly awaits any node whose __call__ is async def,
        including BlackBoxNode which polls port 8001 with asyncio.sleep.
        """
        result = await self.graph.ainvoke(state)
        return GraphState(**result)