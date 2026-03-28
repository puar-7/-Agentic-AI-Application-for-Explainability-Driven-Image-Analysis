# Registers chat & workflow APIs

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api.chat_routes import router as chat_router
from backend.api.workflow_routes import router as workflow_router
from backend.api.upload_routes import router as upload_router
from backend.api.clear_routes import router as clear_router
from backend.db.mongo import MongoDB

from backend.services.document_store import DocumentStore
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.llm.hf_client import get_llm, validate_llm_providers
from backend.graph.unified_graph import UnifiedGraph
import os

INDEX_PATH = "backend/storage/index/index.pkl"

SHARED_OUTPUTS_DIR = os.path.abspath(
    "./backend/storage/shared_workspace/outputs"
)

app = FastAPI(
    title="Agentic Framework Backend",
    version="0.1.0"
)

os.makedirs(SHARED_OUTPUTS_DIR, exist_ok=True)
app.mount(
    "/outputs",
    StaticFiles(directory=SHARED_OUTPUTS_DIR, html=False),
    name="outputs",
)

mongodb = MongoDB(
    uri="mongodb://localhost:27017",
    db_name="agentic_framework"
)

@app.on_event("startup")
def startup_event():
    # ------------------------------------------------------------------
    # Step 1 — Validate LLM provider config before anything else.
    # This surfaces model availability failures at boot time rather than
    # on the first user request. Falls back gracefully with a warning if
    # Sarvam-M is configured but unavailable.
    # ------------------------------------------------------------------
    validate_llm_providers()

    # Step 2 — MongoDB
    mongodb.connect()
    app.state.mongodb = mongodb

    # Step 3 — Document store
    if os.path.exists(INDEX_PATH):
        print("Loading Document Store...")
        store = DocumentStore.load(INDEX_PATH)
    else:
        print("No index found. Initializing empty store.")
        store = DocumentStore()

    app.state.document_store = store

    # Step 4 — Nodes and graph
    # get_llm() is called here via validate_llm_providers() already,
    # but we call it again explicitly so the node gets a fresh instance
    # bound to its own reference (not shared with the probe call).
    retriever = LocalRetrieverNode(store)
    llm = get_llm()
    chat_node = ChatLLMNode(llm)

    app.state.unified_graph = UnifiedGraph(
        retriever_node=retriever,
        chat_llm_node=chat_node,
    )
    print("Unified Graph initialized successfully.")
    print(f"Static outputs mounted from: {SHARED_OUTPUTS_DIR}")


# Register routes
app.include_router(chat_router,     prefix="/chat",     tags=["Chat"])
app.include_router(workflow_router, prefix="/workflow",  tags=["Workflow"])
app.include_router(upload_router,                        tags=["Documents"])
app.include_router(clear_router,                         tags=["Documents"])


@app.on_event("shutdown")
def shutdown_event():
    mongodb.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}