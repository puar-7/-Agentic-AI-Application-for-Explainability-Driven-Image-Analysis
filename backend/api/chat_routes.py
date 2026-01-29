from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from fastapi import Request
from backend.services.document_store import DocumentStore
from backend.graph.chat_graph import ChatGraph
from backend.graph.state import GraphState
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.llm.hf_client import get_chat_llm

# ✅ THIS LINE WAS MISSING
router = APIRouter()

INDEX_PATH = "backend/storage/index/index.pkl"


class ChatRequest(BaseModel):
    query: str


@router.post("/")
@router.post("/")
def chat(request: ChatRequest, http_request: Request): # Add http_request
    # Use the global unified graph
    unified_graph = http_request.app.state.unified_graph

    state = GraphState(
        mode="chat", # This triggers the "chat" path in the unified graph
        user_message=request.query
    )

    final_state = unified_graph.run(state)

    return {
    "answer": final_state.chat_response,
    "sources": [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in (final_state.retrieved_docs or [])
    ]
}