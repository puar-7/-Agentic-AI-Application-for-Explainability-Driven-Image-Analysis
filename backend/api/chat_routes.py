from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from fastapi import Request
from backend.graph.state import GraphState
from typing import List, Dict, Any

router = APIRouter()

INDEX_PATH = "backend/storage/index/index.pkl"


class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []


@router.post("/")
async def chat(request: ChatRequest, http_request: Request):
    unified_graph = http_request.app.state.unified_graph

    state = GraphState(
        mode="chat",
        user_message=request.query,
        chat_history=request.history
    )

    final_state = await unified_graph.run(state)

    # ------------------------------------------------------------------
    # Build sources list — merge local docs + web results
    # Both are normalized to {content, metadata} shape, so the frontend
    # can handle them identically via metadata["source_type"].
    # ------------------------------------------------------------------

    sources = []

    # Local document chunks
    for doc in (final_state.retrieved_docs or []):
        # Only include docs that were actually used
        # (if grade was "incorrect", context_source = "web" —
        #  we still return them so the frontend can see what was
        #  retrieved, but ChatLLMNode ignored them)
        if final_state.context_source != "web":
            sources.append({
                "content":  doc.page_content,
                "metadata": doc.metadata,
            })

    # Web search results
    for result in (final_state.web_search_results or []):
        sources.append({
            "content":  result.get("content", ""),
            "metadata": result.get("metadata", {}),
        })

    return {
        "answer":         final_state.chat_response,
        "context_source": final_state.context_source,
        "retrieval_grade": final_state.retrieval_grade,
        "sources":        sources,
    }