#Registers chat & workflow APIs

from fastapi import FastAPI
from backend.api.chat_routes import router as chat_router
from backend.api.workflow_routes import router as workflow_router
from backend.api.upload_routes import router as upload_router
from backend.api.clear_routes import router as clear_router
from backend.db.mongo import MongoDB

from backend.services.document_store import DocumentStore
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.llm.hf_client import get_chat_llm
from backend.graph.unified_graph import UnifiedGraph # Import the new graph
import os

INDEX_PATH = "backend/storage/index/index.pkl"

app = FastAPI(
    title="Agentic Framework Backend",
    version="0.1.0"
)

mongodb = MongoDB(
    uri="mongodb://localhost:27017",
    db_name="agentic_framework"
)
@app.on_event("startup")
def startup_event():
    mongodb.connect()
    app.state.mongodb = mongodb

    # 2. Load Index ONCE (Performance Fix)
    if os.path.exists(INDEX_PATH):
        print("Loading Document Store...")
        store = DocumentStore.load(INDEX_PATH)
    else:
        print("No index found. Initializing empty store.")
        store = DocumentStore() # You might need to handle empty store logic
    
    # 3. Initialize Nodes
    retriever = LocalRetrieverNode(store)
    llm = get_chat_llm()
    chat_node = ChatLLMNode(llm)

    # 4. Initialize Unified Graph ONCE
    app.state.unified_graph = UnifiedGraph(
        retriever_node=retriever,
        chat_llm_node=chat_node
    )
    print("Unified Graph initialized successfully.")

# Register routes

app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(workflow_router, prefix="/workflow", tags=["Workflow"])
app.include_router(upload_router, tags=["Documents"])
app.include_router(clear_router, tags=["Documents"])

@app.on_event("shutdown")
def shutdown_event():
    mongodb.close()
    
@app.get("/health")
def health_check():
    return {"status": "ok"}
