from fastapi import APIRouter, UploadFile, File
import os
import shutil
from typing import List
from backend.services.document_store import DocumentStore

router = APIRouter()

UPLOAD_DIR = "backend/storage/uploads"
INDEX_PATH = "backend/storage/index/index.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("backend/storage/index", exist_ok=True)

@router.post("/upload-docs")
def upload_docs(files: List[UploadFile] = File(...)):
    file_paths = []

    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(save_path)

    store = DocumentStore()
    docs = store.load_documents(file_paths)
    store.build_indexes(docs)

    store.save(INDEX_PATH)

    return {
        "message": "Documents uploaded and indexed successfully",
        "documents": [f.filename for f in files]
    }
