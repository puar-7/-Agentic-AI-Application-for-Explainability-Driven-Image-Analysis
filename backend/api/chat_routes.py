from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.graph.chat_graph import ChatGraph
from backend.graph.state import GraphState
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.llm.hf_client import get_chat_llm
from backend.services.document_store import DocumentStore

router = APIRouter()

# ⚠️ Phase-1 only: temporary in-memory store
document_store = None


class ChatRequest(BaseModel):
    query: str


@router.post("/")
def chat(request: ChatRequest):
    global document_store

    if document_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload step not implemented in Phase 1."
        )

    retriever = LocalRetrieverNode(document_store)
    llm = get_chat_llm()
    chat_node = ChatLLMNode(llm)

    chat_graph = ChatGraph(
        retriever_node=retriever,
        chat_llm_node=chat_node
    )

    state = GraphState(
        mode="chat",
        user_message=request.query
    )

    final_state = chat_graph.run(state)

    return {
        "answer": final_state.chat_response
    }
