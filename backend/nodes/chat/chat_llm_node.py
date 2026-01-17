from typing import Dict, List

from backend.graph.state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatHuggingFace


class ChatLLMNode:
    """
    LangGraph node responsible for generating chat responses
    using retrieved local document context only.
    """

    def __init__(self, llm: ChatHuggingFace):
        self.llm = llm

    def _build_context(self, retrieved_docs: List) -> str:
        """
        Concatenate retrieved document chunks into a single context string.
        """
        return "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

    def __call__(self, state: GraphState) -> Dict:
        """
        Executes grounded chat generation.

        Reads:
            - state.user_message
            - state.retrieved_docs

        Writes:
            - state.chat_response
        """

        if not state.user_message:
            raise ValueError("No user_message found in GraphState.")

        if not state.retrieved_docs:
            print("[ChatLLMNode] No retrieved documents found.")
            return {
                "chat_response": "No relevant local context found to answer your question."
            }

        print("[ChatLLMNode] Generating response from local context")

        context = self._build_context(state.retrieved_docs)

        system_prompt = (
            "You are an assistant that answers questions strictly "
            "using the provided context. "
            "If the answer is not present in the context, say you do not know."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question:\n{state.user_message}"
                )
            )
        ]

        response = self.llm.invoke(messages)

        return {
            "chat_response": response.content
        }
