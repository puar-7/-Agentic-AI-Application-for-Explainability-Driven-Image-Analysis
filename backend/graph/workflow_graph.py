from langgraph.graph import StateGraph, END

from backend.graph.state import GraphState
from backend.nodes.workflow.workflow_input_parser import WorkflowInputParserNode
from backend.nodes.workflow.router_node import RouterNode
from backend.nodes.workflow.white_box_node import WhiteBoxNode
from backend.nodes.workflow.black_box_node import BlackBoxNode
from backend.nodes.workflow.parallel_node import ParallelExecutionNode
from backend.nodes.workflow.report_generation_node import ReportGenerationNode
from backend.nodes.workflow.evaluation_node import EvaluationNode


class WorkflowGraph:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        # Nodes
        graph.add_node("parse_input", WorkflowInputParserNode())
        graph.add_node("route", RouterNode())
        graph.add_node("parallel", ParallelExecutionNode())
        graph.add_node("white_box", WhiteBoxNode())
        graph.add_node("black_box", BlackBoxNode())
        graph.add_node("report", ReportGenerationNode())
        graph.add_node("evaluation", EvaluationNode())

        # Entry
        graph.set_entry_point("parse_input")

        # Parse failure → END
        graph.add_conditional_edges(
            "parse_input",
            lambda state: "error" if state.error else "ok",
            {
                "error": END,
                "ok": "route",
            }
        )

        # Routing 
        graph.add_conditional_edges(
            "route",
            lambda state: state.route,
            {
                "white_path": "white_box",
                "black_path": "black_box",
                "both_path": "parallel",
                
            }
        )

        # Fan-out for parallel execution
        graph.add_edge("parallel", "white_box")
        graph.add_edge("parallel", "black_box")

        # Merge
        graph.add_edge("white_box", "report")
        graph.add_edge("black_box", "report")

        # Finalization
        graph.add_edge("report", "evaluation")
        graph.add_edge("evaluation", END)

        return graph.compile()

    def run(self, state: GraphState):
        return self.graph.invoke(state)
