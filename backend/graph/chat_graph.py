#Responsible for:
# Graph construction
# Node wiring
# Graph compilation
# Graph execution helper

from langgraph.graph import StateGraph, END

from backend.graph.state import GraphState
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode


class ChatGraph:
    """
    LangGraph-based chat graph.

    Flow:
        START → LocalRetrieverNode → ChatLLMNode → END
    """

    def __init__(
        self,
        retriever_node: LocalRetrieverNode,
        chat_llm_node: ChatLLMNode,
    ):
        self.retriever_node = retriever_node
        self.chat_llm_node = chat_llm_node

        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(GraphState)

        # Register nodes
        graph_builder.add_node(
            "local_retrieval",
            self.retriever_node
        )
        graph_builder.add_node(
            "chat_generation",
            self.chat_llm_node
        )

        # Wire edges
        graph_builder.set_entry_point("local_retrieval")
        graph_builder.add_edge("local_retrieval", "chat_generation")
        graph_builder.add_edge("chat_generation", END)

        return graph_builder.compile()

    def run(self, state: GraphState) -> GraphState:
        """
        Execute the chat graph with an initial GraphState.
        """
        result = self.graph.invoke(state)
        return GraphState(**result)
