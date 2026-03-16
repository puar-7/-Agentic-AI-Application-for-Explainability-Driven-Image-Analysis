# Registers chat & workflow APIs

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles                    # Phase 3 — added

from backend.api.chat_routes import router as chat_router
from backend.api.workflow_routes import router as workflow_router
from backend.api.upload_routes import router as upload_router
from backend.api.clear_routes import router as clear_router
from backend.db.mongo import MongoDB

from backend.services.document_store import DocumentStore
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.llm.hf_client import get_chat_llm
from backend.graph.unified_graph import UnifiedGraph
import os

INDEX_PATH = "backend/storage/index/index.pkl"

# Absolute path to the shared workspace outputs directory.
# Port 8001 writes heatmaps/overlays here.
# Port 8000 serves them as static files so the frontend can render them.
SHARED_OUTPUTS_DIR = os.path.abspath(
    "./backend/storage/shared_workspace/outputs"
)

app = FastAPI(
    title="Agentic Framework Backend",
    version="0.1.0"
)

# ---------------------------------------------------------
# Phase 3 — Static file serving
#
# Any file under shared_workspace/outputs/ is now reachable at:
#     http://localhost:8000/outputs/<relative_path>
#
# Example:
#     /outputs/heatmaps/overlays/000001_overlay.jpg
#     /outputs/heatmaps/heatmap_images/000001_heatmap.jpg
#
# The adapter writes URL paths in exactly this format, so the
# frontend can render them without any path translation.
#
# html=False: we are serving images, not an HTML app.
# check_dir=False: directory may not exist yet on a fresh machine
#                  (created by the first /build run). Without this,
#                  FastAPI would raise an error on startup.
# ---------------------------------------------------------
os.makedirs(SHARED_OUTPUTS_DIR, exist_ok=True)          # ensure it exists at startup
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
    mongodb.connect()
    app.state.mongodb = mongodb

    # Load document index
    if os.path.exists(INDEX_PATH):
        print("Loading Document Store...")
        store = DocumentStore.load(INDEX_PATH)
    else:
        print("No index found. Initializing empty store.")
        store = DocumentStore()

    app.state.document_store = store

    # Initialize nodes
    retriever = LocalRetrieverNode(store)
    llm = get_chat_llm()
    chat_node = ChatLLMNode(llm)

    # Initialize unified graph
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