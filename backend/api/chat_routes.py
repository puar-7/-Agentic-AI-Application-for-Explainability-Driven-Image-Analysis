from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

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
def chat(request: ChatRequest):
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(
            status_code=400,
            detail="No index found. Please upload documents first."
        )

    store = DocumentStore.load(INDEX_PATH)

    retriever = LocalRetrieverNode(store)
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

    return {"answer": final_state.chat_response}
