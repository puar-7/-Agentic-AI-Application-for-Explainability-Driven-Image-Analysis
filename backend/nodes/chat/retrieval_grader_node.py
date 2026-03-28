import os
import re
from typing import Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from backend.llm.hf_client import get_grader_llm


MAX_CONTEXT_CHARS = 2000


class RetrievalGraderNode:
    """
    LangGraph node that evaluates whether retrieved documents
    are relevant to the user's query.

    Three-stage evaluation (in order):

        Stage 0 — Document-intent short-circuit (no LLM, no keyword check)
            If the user explicitly references their uploaded documents,
            force grade = "correct" immediately and attempt to filter
            retrieved_docs to only chunks from the named document.

        Stage 1 — Fast keyword pre-check (no LLM needed)
            Checks if key content terms from the query appear in chunks.
            If below 50% overlap → immediately "incorrect", skip LLM.

        Stage 2 — LLM grader (only runs if Stage 1 passes)
            Stricter prompt with explicit examples.

    Output → state.retrieval_grade:
        "correct"   — docs are relevant, use them directly
        "incorrect" — docs are irrelevant, discard and use web search
        "ambiguous" — partially relevant, merge docs + web search
    """

    # ----------------------------------------------------------
    # Stage 0 — Document-intent trigger phrases
    # ----------------------------------------------------------
    DOCUMENT_INTENT_PHRASES = [
        "uploaded document",
        "uploaded file",
        "uploaded paper",
        "the document",
        "the paper",
        "the file",
        "my document",
        "my paper",
        "my file",
        "summarize the",
        "summarise the",
        "summary of the",
        "explain the document",
        "explain the paper",
        "explain the file",
        "in the document",
        "in the paper",
        "in the file",
        "from the document",
        "from the paper",
        "from the file",
        "according to the document",
        "according to the paper",
        "i uploaded",
        "i have uploaded",
        "you have the document",
        "you have the paper",
        ".pdf",
        ".txt",
    ]

    # Words that are never part of a document name — used when
    # extracting the filename hint from a document-intent query.
    FILENAME_STOP_WORDS = {
        "can", "you", "could", "please", "help", "summarize", "summarise",
        "explain", "describe", "tell", "give", "show", "find", "get",
        "what", "which", "who", "how", "why", "when", "where",
        "the", "a", "an", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "may", "might", "shall",
        "uploaded", "upload", "document", "file", "paper", "pdf", "txt",
        "about", "of", "in", "on", "at", "to", "for", "with", "by",
        "from", "and", "or", "but", "not", "also", "just",
        "me", "my", "our", "your", "its",
        "summary", "idea", "concept", "general", "main", "key",
        "information", "content", "topic", "subject",
    }

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
        self.llm = get_grader_llm(
            
        )

    # ----------------------------------------------------------
    # Stage 0a — Document-intent detection
    # ----------------------------------------------------------

    def _is_document_intent_query(self, query: str) -> bool:
        """
        Returns True if the query contains explicit signals that the user
        is asking about their locally uploaded documents.
        """
        query_lower = query.lower()
        for phrase in self.DOCUMENT_INTENT_PHRASES:
            if phrase in query_lower:
                print(
                    f"[RetrievalGraderNode] Document-intent detected "
                    f"(matched: '{phrase}') → forcing correct."
                )
                return True
        return False

    # ----------------------------------------------------------
    # Stage 0b — Filename-targeted doc filtering
    # ----------------------------------------------------------

    def _extract_filename_tokens(self, query: str) -> List[str]:
        """
        Extracts tokens from the query that are likely part of a
        document name — i.e. everything left after removing intent
        phrases and filename stop words.

        Example:
            "Can you summarize the uploaded document deepsek mhc"
            → removes "uploaded document", stop words
            → returns ["deepsek", "mhc"]
        """
        # Remove intent phrases first so their words don't pollute tokens
        text = query.lower()
        for phrase in self.DOCUMENT_INTENT_PHRASES:
            text = text.replace(phrase, " ")

        tokens = [
            w for w in re.findall(r'\b\w+\b', text)
            if len(w) > 1 and w not in self.FILENAME_STOP_WORDS
        ]
        return tokens

    def _filter_docs_by_filename(
        self,
        retrieved_docs: List,
        filename_tokens: List[str],
    ) -> Optional[List]:
        """
        Filters retrieved_docs to only those whose source filename
        contains at least one of the filename_tokens.

        Matching is against the basename of metadata["source"] so that
        full OS paths don't interfere.

        Returns:
            Filtered list if at least one matching chunk was found.
            None if no chunks matched (caller should fall back to all docs).
        """
        if not filename_tokens:
            return None

        matched = []
        for doc in retrieved_docs:
            source_path = ""
            if hasattr(doc, "metadata"):
                source_path = doc.metadata.get("source", "")
            elif isinstance(doc, dict):
                source_path = doc.get("metadata", {}).get("source", "")

            # Compare against the filename only, not the full path
            filename = os.path.basename(source_path).lower()

            if any(token in filename for token in filename_tokens):
                matched.append(doc)

        if matched:
            print(
                f"[RetrievalGraderNode] Filename filter: "
                f"{len(matched)}/{len(retrieved_docs)} chunks matched "
                f"tokens {filename_tokens}"
            )
            return matched

        # No match — tokens may not correspond to any indexed filename.
        # Log it and signal fallback.
        print(
            f"[RetrievalGraderNode] Filename filter: no chunks matched "
            f"tokens {filename_tokens} — using all retrieved docs."
        )
        return None

    # ----------------------------------------------------------
    # Stage 1 — Keyword pre-check
    # ----------------------------------------------------------

    def _keyword_precheck(self, query: str, retrieved_docs) -> bool:
        """
        Returns False (→ immediate "incorrect") if less than 50% of
        meaningful query terms appear anywhere in the retrieved chunks.
        """
        STOP_WORDS = {
            # Standard English function words
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "me", "my", "we", "our", "you", "your", "he", "his", "she",
            "her", "it", "its", "they", "their", "what", "which", "who",
            "whom", "this", "that", "these", "those", "am", "about",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "up", "down", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "then", "once", "here", "there", "when",
            "where", "why", "all", "both", "few", "more", "most",
            "other", "some", "such", "no", "not", "only", "same",
            "so", "than", "too", "very", "just", "because", "if",
            "while", "although", "though", "or", "but", "and", "nor",
            "yet", "either", "neither", "each", "every", "any",
            "its", "own", "whose", "also",
            # Query meta-words
            "tell", "explain", "describe", "how", "why", "give",
            "show", "get", "make", "much", "many", "summarise",
            "summarize", "discuss", "define", "list", "provide",
            "can", "could", "please", "help", "want", "need",
            "find", "look", "search", "question", "answer",
            "understand", "know", "think", "believe", "say", "said",
            # Generic academic vocabulary
            "term", "terms", "meant", "meaning", "means",
            "concept", "concepts", "idea", "ideas",
            "general", "specific", "particular", "certain",
            "type", "types", "kind", "kinds",
            "use", "used", "using", "based",
            "way", "ways", "case", "cases",
            "called", "known", "refer", "refers", "reference",
            "include", "includes", "including",
            "provide", "provides", "provided",
            "paper", "document", "study", "research",
            "method", "approach", "system", "model",
            "result", "results", "data", "set",
            "two", "one", "three", "four", "five",
            "new", "different", "various", "several",
            "between", "among", "across",
            "information", "context", "example",
            # Upload / document meta-words
            "uploaded", "upload", "file", "attached", "attachment",
        }

        query_terms = set(
            word.lower()
            for word in re.findall(r'\b\w+\b', query)
            if len(word) > 2 and word.lower() not in STOP_WORDS
        )

        if not query_terms:
            return True

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

        return match_ratio >= 0.50

    # ----------------------------------------------------------
    # Stage 2 helpers
    # ----------------------------------------------------------

    def _build_context_snippet(self, retrieved_docs) -> str:
        snippets = []
        total_chars = 0

        for i, doc in enumerate(retrieved_docs[:4]):
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

    # ----------------------------------------------------------
    # Node entry point
    # ----------------------------------------------------------

    def __call__(self, state) -> Dict:

        # No docs at all
        if not state.retrieved_docs:
            print("[RetrievalGraderNode] No docs retrieved → incorrect.")
            return {
                "retrieval_grade": "incorrect",
                "context_source": "web",
            }

        # ----------------------------------------------------------
        # Stage 0 — Document-intent short-circuit
        #
        # When the user explicitly names their uploaded content:
        #   0a. Detect intent → force correct, skip keyword + LLM
        #   0b. Extract filename tokens from query and filter
        #       retrieved_docs to only chunks from the named file.
        #       Falls back to all docs if no filename match found,
        #       so the system degrades gracefully rather than breaking.
        # ----------------------------------------------------------
        if self._is_document_intent_query(state.user_message):
            filename_tokens = self._extract_filename_tokens(state.user_message)
            filtered = self._filter_docs_by_filename(
                state.retrieved_docs, filename_tokens
            )
            # Use filtered subset if available, otherwise all retrieved docs
            docs_to_use = filtered if filtered is not None else state.retrieved_docs

            return {
                "retrieval_grade": "correct",
                "context_source": "documents",
                "retrieved_docs": docs_to_use,
            }

        # ----------------------------------------------------------
        # Stage 1 — Keyword pre-check
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Stage 2 — LLM grader
        # ----------------------------------------------------------
        context_snippet = self._build_context_snippet(state.retrieved_docs)

        human_prompt = (
            f"User query: {state.user_message}\n\n"
            f"Retrieved document chunks:\n{context_snippet}\n\n"
            f"Do these chunks contain information that directly answers "
            f"the query about '{state.user_message}'?\n"
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