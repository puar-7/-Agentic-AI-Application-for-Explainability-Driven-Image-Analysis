import re
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from backend.llm.hf_client import get_chat_llm


MAX_CONTEXT_CHARS = 2000  # Reduced — grader only needs a taste, not full chunks


class RetrievalGraderNode:
    """
    LangGraph node that evaluates whether retrieved documents
    are relevant to the user's query.

    Two-stage evaluation:
        Stage 1 — Fast keyword pre-check (no LLM needed)
            Checks if key terms from the query appear in the chunks.
            If zero overlap → immediately "incorrect", skip LLM call entirely.

        Stage 2 — LLM grader (only runs if Stage 1 passes)
            Stricter prompt with explicit examples to prevent Llama-3-8B
            from being too lenient.

    Output → state.retrieval_grade:
        "correct"   — docs are relevant, use them directly
        "incorrect" — docs are irrelevant, discard and use web search
        "ambiguous" — partially relevant, merge docs + web search
    """

    SYSTEM_PROMPT = """You are a strict retrieval quality evaluator.

Your job: decide if document chunks actually contain information
that answers the user's query.

Rules:
- "correct"   → the chunks DIRECTLY address the query topic with relevant facts
- "incorrect" → the chunks are about a DIFFERENT topic entirely
- "ambiguous" → the chunks touch the topic but lack enough detail

IMPORTANT:
- If the query asks about a PERSON and the chunks don't mention that person → incorrect
- If the query asks about a CONCEPT and the chunks discuss unrelated concepts → incorrect
- Do NOT give "correct" just because chunks exist. Judge actual content match.

Examples:
  Query: "Tell me about Albert Einstein"
  Chunks: [text about neural networks and deep learning]
  Answer: incorrect

  Query: "What is CRAG?"
  Chunks: [text explaining Corrective RAG methodology and retrieval evaluation]
  Answer: correct

  Query: "How does attention work in transformers?"
  Chunks: [text about transformers mentioning attention briefly, mostly other topics]
  Answer: ambiguous

Respond with EXACTLY ONE WORD: correct, incorrect, or ambiguous.
No explanation. No punctuation. One word only."""

    def __init__(self):
        self.llm = get_chat_llm(
            max_new_tokens=15,
            temperature=0.0,
        )

    def _keyword_precheck(self, query: str, retrieved_docs) -> bool:
        """
        Stage 1: Fast keyword overlap check — no LLM involved.

        Returns False (→ immediate "incorrect") if less than 30% of
        meaningful query terms appear anywhere in the retrieved chunks.
        This catches obvious mismatches like "Andrej Karpathy" vs mHC paper.
        """
        STOP_WORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "me", "my", "we", "our", "you", "your", "he", "his", "she",
            "her", "it", "its", "they", "their", "what", "which", "who",
            "whom", "this", "that", "these", "those", "am", "about",
            "tell", "explain", "describe", "how", "why", "when", "where",
            "give", "show", "get", "make", "much", "many", "more", "most",
        }

        query_terms = set(
            word.lower()
            for word in re.findall(r'\b\w+\b', query)
            if len(word) > 2 and word.lower() not in STOP_WORDS
        )

        if not query_terms:
            return True  # Cannot pre-check — let LLM decide

        combined_text = " ".join(
            (doc.page_content if hasattr(doc, "page_content") else doc.get("content", ""))
            for doc in retrieved_docs
        ).lower()

        matches = sum(1 for term in query_terms if term in combined_text)
        match_ratio = matches / len(query_terms)

        print(
            f"[RetrievalGraderNode] Keyword pre-check: "
            f"{matches}/{len(query_terms)} terms matched "
            f"({match_ratio:.0%}) — terms: {query_terms}"
        )

        return match_ratio >= 0.30

    def _build_context_snippet(self, retrieved_docs) -> str:
        snippets = []
        total_chars = 0

        for i, doc in enumerate(retrieved_docs[:4]):  # Max 4 chunks for speed
            content = (
                doc.page_content
                if hasattr(doc, "page_content")
                else doc.get("content", "")
            )

            if total_chars + len(content) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - total_chars
                snippets.append(f"[Chunk {i+1}]: {content[:remaining]}...")
                break

            snippets.append(f"[Chunk {i+1}]: {content}")
            total_chars += len(content)

        return "\n\n".join(snippets)

    def _parse_grade(self, raw_response: str) -> str:
        """
        Checks "incorrect" BEFORE "correct" — otherwise the substring
        "correct" inside "incorrect" would match first and give wrong result.
        """
        cleaned = raw_response.strip().lower()

        if "incorrect" in cleaned:
            return "incorrect"
        if "ambiguous" in cleaned:
            return "ambiguous"
        if "correct" in cleaned:
            return "correct"

        print(
            f"[RetrievalGraderNode] Unparseable response: '{raw_response}'. "
            "Defaulting to 'ambiguous'."
        )
        return "ambiguous"

    def __call__(self, state) -> Dict:

        # No docs at all
        if not state.retrieved_docs:
            print("[RetrievalGraderNode] No docs retrieved → incorrect.")
            return {
                "retrieval_grade": "incorrect",
                "context_source": "web",
            }

        # Stage 1 — keyword pre-check
        terms_overlap = self._keyword_precheck(
            state.user_message,
            state.retrieved_docs
        )

        if not terms_overlap:
            print(
                "[RetrievalGraderNode] Keyword pre-check failed → "
                "incorrect (skipping LLM grader)."
            )
            return {
                "retrieval_grade": "incorrect",
                "context_source": "web",
            }

        # Stage 2 — LLM grader
        context_snippet = self._build_context_snippet(state.retrieved_docs)

        human_prompt = (
            f"User query: {state.user_message}\n\n"
            f"Retrieved document chunks:\n{context_snippet}\n\n"
            f"Do these chunks contain information that directly answers "
            f"the query about '{state.user_message[:80]}'?\n"
            "Answer with ONE WORD only: correct, incorrect, or ambiguous."
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            grade = self._parse_grade(response.content)
        except Exception as e:
            print(f"[RetrievalGraderNode] LLM call failed: {e}. Defaulting to ambiguous.")
            grade = "ambiguous"

        print(
            f"[RetrievalGraderNode] Grade: '{grade}' "
            f"for query: '{state.user_message[:60]}'"
        )

        context_source = "documents" if grade == "correct" else None

        return {
            "retrieval_grade": grade,
            "context_source": context_source,
        }