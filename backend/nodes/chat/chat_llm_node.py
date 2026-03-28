from typing import Dict, List

from backend.graph.state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models import ChatHuggingFace


class ChatLLMNode:
    """
    LangGraph node responsible for generating chat responses.

    Handles three context sources:
        "documents" — answer from local uploaded docs only
        "web"       — answer from web search results only
        "hybrid"    — answer from both docs and web results merged
    """

    def __init__(self, llm):
        self.llm = llm
        self.MAX_INPUT_CHARS = 24000

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_doc_context(self, retrieved_docs: List) -> str:
        return "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

    def _build_web_context(self, web_results: List) -> str:
        lines = []
        for i, result in enumerate(web_results, 1):
            title = result.get("metadata", {}).get("title", "Untitled")
            url   = result.get("metadata", {}).get("url", "")
            content = result.get("content", "")
            lines.append(
                f"[Web Source {i} — {title} ({url})]:\n{content}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    def _get_system_prompt(self, context_source: str) -> str:

        if context_source == "documents":
            return """You are a helpful AI assistant operating in a document-grounded chat system.

You have access to TWO data sources:
1. Conversation History — prior user and assistant messages.
2. Document Context — content retrieved from the user's uploaded documents.

### INSTRUCTIONS
- Greetings, preferences, or acknowledgements: respond naturally. Do NOT use Document Context.
- Document questions: answer strictly using the provided Document Context.
  Do not use external knowledge or make up facts.
- Conversation references (e.g. "what did I say earlier?"): use Conversation History.
- If information is in neither source, clearly state you do not know.

Do NOT proactively introduce document information unless explicitly asked."""

        elif context_source == "web":
            return """You are a helpful AI assistant. The user's question was not answerable
from their uploaded documents, so web search results have been retrieved instead.

### INSTRUCTIONS
- Answer using ONLY the provided Web Search Results.
- Always make clear your answer is based on web search, not the user's documents.
- If the web results do not contain enough information, say so clearly.
- Do not invent facts not present in the web results.
- If relevant, mention the source title or website naturally in your answer."""

        else:  # hybrid
            return """You are a helpful AI assistant. Both the user's uploaded documents
and web search results are available to answer this question.

### INSTRUCTIONS
- Use BOTH Document Context and Web Search Results to form your answer.
- Clearly distinguish when you are drawing from documents vs web results.
  For example: "According to your uploaded documents..." or "Web search results indicate..."
- Do not invent facts not present in either source.
- If the sources conflict, mention the discrepancy rather than picking one silently."""

    # ------------------------------------------------------------------
    # History filter
    # ------------------------------------------------------------------

    def _filter_history(self, history: List[Dict], available_chars: int) -> List[BaseMessage]:
        selected_messages = []
        current_chars = 0

        for msg in reversed(history):
            content = msg.get("content", "")
            role = msg.get("role")
            msg_len = len(content)

            if current_chars + msg_len > available_chars:
                break

            if role == "user":
                selected_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                selected_messages.append(AIMessage(content=content))

            current_chars += msg_len

        return list(reversed(selected_messages))

    # ------------------------------------------------------------------
    # Node entry point
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState) -> Dict:
        """
        Builds context from the appropriate source(s) and generates
        a response using the correct system prompt.
        """

        context_source = state.context_source or "documents"

        # ----------------------------------------------------------
        # Build context string
        # ----------------------------------------------------------
        context_parts = []

        if context_source in ("documents", "hybrid") and state.retrieved_docs:
            doc_context = self._build_doc_context(state.retrieved_docs)
            context_parts.append(f"--- Document Context ---\n{doc_context}")

        if context_source in ("web", "hybrid") and state.web_search_results:
            web_context = self._build_web_context(state.web_search_results)
            context_parts.append(f"--- Web Search Results ---\n{web_context}")

        full_context = "\n\n".join(context_parts)

        # ----------------------------------------------------------
        # Build prompts
        # ----------------------------------------------------------
        system_prompt_text = self._get_system_prompt(context_source)

        user_message_text = (
            f"Context:\n{full_context}\n\n"
            f"User Message:\n{state.user_message}"
        )

        # ----------------------------------------------------------
        # Trim history before building message list.
        #
        # The frontend appends the current user message to chat_history
        # BEFORE sending the request, so state.chat_history always ends
        # with the current user message. If we include it in the history
        # block AND append it again as the final HumanMessage, APIs that
        # enforce strict user/assistant alternation (e.g. Sarvam) reject
        # the request with a 400.
        #
        # Fix: drop any trailing user messages from history — they will
        # be represented by the explicit HumanMessage appended below.
        # This is safe for all providers: Llama on HF was silently
        # tolerating the duplicate; this makes the behaviour correct
        # and consistent everywhere.
        # ----------------------------------------------------------
        trimmed_history = list(state.chat_history or [])
        while trimmed_history and trimmed_history[-1].get("role") == "user":
            trimmed_history.pop()

        # ----------------------------------------------------------
        # History budget
        # ----------------------------------------------------------
        fixed_cost = len(system_prompt_text) + len(user_message_text)
        remaining_chars = self.MAX_INPUT_CHARS - fixed_cost

        messages = [SystemMessage(content=system_prompt_text)]

        if trimmed_history and remaining_chars > 500:
            history_messages = self._filter_history(trimmed_history, remaining_chars)
            messages.extend(history_messages)

        messages.append(HumanMessage(content=user_message_text))

        # ----------------------------------------------------------
        # Generate
        # ----------------------------------------------------------
        response = self.llm.invoke(messages)

        return {"chat_response": response.content}