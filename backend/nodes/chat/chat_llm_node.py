from typing import Dict, List

from backend.graph.state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models import ChatHuggingFace


class ChatLLMNode:
    """
    LangGraph node responsible for generating chat responses
    using retrieved local document context only.
    """

    def __init__(self, llm: ChatHuggingFace):
        self.llm = llm
        # Safety Limit for Llama-3-8B (8192 tokens total context).
        # We reserve ~2000 tokens for generation/overhead, leaving ~6000 for input.
        # Approx 4 chars per token -> 6000 * 4 = 24,000 chars.
        self.MAX_INPUT_CHARS = 24000

    def _build_context(self, retrieved_docs: List) -> str:
        """
        Concatenate retrieved document chunks into a single context string.
        """
        return "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

    def _filter_history(self, history: List[Dict[str, str]], available_chars: int) -> List[BaseMessage]:
        """
        Selects the most recent messages that fit within the character budget.
        """
        selected_messages = []
        current_chars = 0

        # Iterate backwards (Newest -> Oldest) to keep most recent context
        for msg in reversed(history):
            content = msg.get("content", "")
            role = msg.get("role")
            msg_len = len(content)

            # Stop if adding this message exceeds budget
            if current_chars + msg_len > available_chars:
                break

            # Convert to LangChain message objects
            if role == "user":
                selected_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                selected_messages.append(AIMessage(content=content))
            
            current_chars += msg_len

        # Reverse back to chronological order (Oldest -> Newest)
        return list(reversed(selected_messages))

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

        
        # This tells the LLM: "Here is some info, but ignore it if the user is just saying hi."
        system_prompt_text = ( """
            
    You are a helpful AI assistant operating in a document-grounded chat system.

You have access to TWO data sources:
1. Conversation History — prior user and assistant messages in the current session.
2. Document Context — content retrieved from the user’s uploaded documents.

### DATA SOURCE USAGE RULES
- Use **Conversation History** for:
  - Names, preferences, or facts explicitly mentioned by the user
  - Follow-up questions about earlier messages
  - Meta questions about the conversation itself

- Use **Document Context** ONLY when the user is clearly asking
  for information, explanation, analysis, or summaries related to the uploaded documents.

### INSTRUCTIONS
- **Greetings, Preferences, Acknowledgements, or Instructions**  
  (e.g., “Hi”, “Okay”, “I want concise answers”, “Thanks”):  
  Respond naturally and politely.  
  **Do NOT use Document Context for these messages.**

- **Document Questions**  
  (e.g., “Explain the paper”, “What does mHC do?”, “Summarize the results”):  
  Answer strictly using the provided Document Context.  
  Do not use external knowledge or make up facts.

- **Conversation References**  
  (e.g., “What is my name?”, “What did I say earlier?”):  
  Answer using Conversation History.

- **Summarization Requests**  
  If the user explicitly asks for a summary or overview of a document,  
  you may synthesize information across multiple parts of the Document Context.

- **Unknowns**  
  If the required information is present in NEITHER Conversation History  
  nor Document Context, clearly state that you do not know.

### IMPORTANT
Do NOT proactively introduce information from the documents unless the user
explicitly asks a document-related question.
"""
)
        

        user_message_text = (
            f"Context:\n{context}\n\n"
            f"User Message:\n{state.user_message}"
        )

        # 3. Calculate Budget for History
        fixed_cost = len(system_prompt_text) + len(user_message_text)
        remaining_chars = self.MAX_INPUT_CHARS - fixed_cost

        # 4. Construct Message Payload
        messages = [SystemMessage(content=system_prompt_text)]

        # Add History (only if we have space)
        if state.chat_history and remaining_chars > 500:
            history_messages = self._filter_history(state.chat_history, remaining_chars)
            messages.extend(history_messages)

        # Add User Message (Always included)
        messages.append(HumanMessage(content=user_message_text))

        # 5. Invoke LLM
        response = self.llm.invoke(messages)

        return {
            "chat_response": response.content
        }
