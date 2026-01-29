from fastapi import APIRouter, Request
import shutil
import os

router = APIRouter()

@router.post("/clear")
def clear_all(request: Request):
    if os.path.exists("backend/storage/uploads"):
        shutil.rmtree("backend/storage/uploads")
        os.makedirs("backend/storage/uploads")

    if os.path.exists("backend/storage/index"):
        shutil.rmtree("backend/storage/index")
        os.makedirs("backend/storage/index")
    # This ensures the chatbot forgets the documents IMMEDIATELY
    if hasattr(request.app.state, "document_store"):
        request.app.state.document_store.reset()
    return {"message": "All documents and index cleared"}
