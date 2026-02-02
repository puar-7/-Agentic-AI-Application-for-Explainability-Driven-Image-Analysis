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
from typing import List, Dict, Any
router = APIRouter()

INDEX_PATH = "backend/storage/index/index.pkl"


class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = [] # <--- Added History Field with default empty list


@router.post("/")
def chat(request: ChatRequest, http_request: Request): # Add http_request
    # Use the global unified graph
    unified_graph = http_request.app.state.unified_graph

    state = GraphState(
        mode="chat", # This triggers the "chat" path in the unified graph
        user_message=request.query,
        chat_history=request.history  # Pass chat history to the state
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