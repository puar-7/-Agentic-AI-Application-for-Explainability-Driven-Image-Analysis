#Registers chat & workflow APIs

from fastapi import FastAPI
from backend.api.chat_routes import router as chat_router
from backend.api.workflow_routes import router as workflow_router
from backend.api.upload_routes import router as upload_router
from backend.api.clear_routes import router as clear_router


app = FastAPI(
    title="Agentic Framework Backend",
    version="0.1.0"
)


# Register routes

app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(workflow_router, prefix="/workflow", tags=["Workflow"])
app.include_router(upload_router, tags=["Documents"])
app.include_router(clear_router, tags=["Documents"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
