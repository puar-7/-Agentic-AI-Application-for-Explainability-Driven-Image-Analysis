import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class WebSearchNode:
    """
    LangGraph node that performs a web search using Tavily
    and normalizes results into the same shape as local document chunks.

    Normalized output shape (mirrors LangChain Document metadata style):
        {
            "content":  "snippet text ...",
            "metadata": {
                "source_type": "web",
                "title":       "Page title from Tavily",
                "url":         "https://...",
            }
        }

    This means ChatLLMNode and _group_sources() in the frontend
    can treat web results and local docs identically — they just
    check metadata["source_type"] to know which is which.

    Writes:
        state.web_search_results  — list of normalized dicts
        state.context_source      — "web" or "hybrid"
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._client = None

    # ------------------------------------------------------------------
    # Lazy Tavily client — only imported/created on first use
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python is not installed. "
                "Run: pip install tavily-python"
            )

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY not found. "
                "Add it to your .env file."
            )

        self._client = TavilyClient(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, raw_results: List[dict]) -> List[dict]:
        """
        Converts Tavily result dicts into normalized doc-like dicts.

        Tavily returns:
            {"url": "...", "content": "...", "title": "..."}

        We produce:
            {
                "content": "...",
                "metadata": {
                    "source_type": "web",
                    "title": "...",
                    "url": "..."
                }
            }
        """
        normalized = []

        for result in raw_results:
            content = result.get("content", "").strip()
            if not content:
                continue  # skip empty snippets

            normalized.append({
                "content": content,
                "metadata": {
                    "source_type": "web",
                    "title": result.get("title", "Untitled"),
                    "url":   result.get("url", ""),
                }
            })

        return normalized

    # ------------------------------------------------------------------
    # Node entry point
    # ------------------------------------------------------------------

    def __call__(self, state) -> Dict:
        """
        Runs a Tavily web search for state.user_message.

        If retrieval_grade is "ambiguous", we keep the existing
        retrieved_docs AND add web results → context_source = "hybrid".

        If retrieval_grade is "incorrect", we discard docs entirely
        → context_source = "web".
        """

        print(f"[WebSearchNode] Searching for: {state.user_message}")

        try:
            client = self._get_client()
            raw = client.search(
                query=state.user_message,
                max_results=self.max_results,
                search_depth="basic",   # "basic" is faster; "advanced" is more thorough
            )
            results = raw.get("results", [])

        except Exception as e:
            print(f"[WebSearchNode] Search failed: {e}")
            # Fail gracefully — fall back to whatever docs we had
            return {
                "web_search_results": [],
                "context_source": (
                    "documents"
                    if state.retrieved_docs
                    else "web"
                ),
            }

        normalized = self._normalize(results)
        print(f"[WebSearchNode] Got {len(normalized)} web results.")

        # Determine context_source based on grade
        if state.retrieval_grade == "ambiguous" and state.retrieved_docs:
            context_source = "hybrid"
        else:
            context_source = "web"

        return {
            "web_search_results": normalized,
            "context_source": context_source,
        }