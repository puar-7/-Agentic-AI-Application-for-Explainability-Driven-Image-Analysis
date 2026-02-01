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

        context = ""
        if state.retrieved_docs:
            context = self._build_context(state.retrieved_docs)

        # 2. Flexible System Prompt
        # This tells the LLM: "Here is some info, but ignore it if the user is just saying hi."
        system_prompt = (
            "You are a helpful AI assistant designed to answer questions about the user's documents. "
            "You have been provided with context information below. "
            "\n\n"
            "INSTRUCTIONS:\n"
            "1. If the user's message is a greeting (e.g., 'Hello', 'Hi'), respond naturally and politely without using the context.\n"
            "2. If the user's message matches the context, use the context to answer accurately.\n"
            "3. If the answer is NOT in the context, strictly state that you do not know based on the documents. "
            "Do not hallucinate technical details."
        )

        # 3. Construct the message payload
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"User Message:\n{state.user_message}"
                )
            )
        ]

        # 4. Invoke LLM
        response = self.llm.invoke(messages)

        return {
            "chat_response": response.content
        }
