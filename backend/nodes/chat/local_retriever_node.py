from typing import Dict

from backend.graph.state import GraphState
from backend.services.document_store import DocumentStore


class LocalRetrieverNode:
    """
    LangGraph node responsible for retrieving relevant local documents
    using hybrid retrieval (dense + sparse).
    """

    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    def __call__(self, state: GraphState) -> Dict:
        """
        Executes local document retrieval.

        Reads:
            - state.user_message

        Writes:
            - state.retrieved_docs
        """

        if not state.user_message:
            raise ValueError("No user_message found in GraphState.")

        query = state.user_message

        print("[LocalRetrieverNode] Retrieving documents for query:")
        print(f"  → {query}")

        if (
        self.document_store.vector_store is None
        or self.document_store.bm25 is None
        ):
            print("[LocalRetrieverNode] No indexes found. Skipping retrieval.")
            return {
                "retrieved_docs": []
            }
        retrieved_docs = self.document_store.hybrid_retrieve(query)

        print(f"[LocalRetrieverNode] Retrieved {len(retrieved_docs)} document chunks")

        return {
            "retrieved_docs": retrieved_docs
        }
