from fastapi import APIRouter, UploadFile, File
import os
import shutil
from typing import List
from backend.services.document_store import DocumentStore
import json
from backend.services.utils import compute_file_hash

router = APIRouter()
METADATA_PATH = "backend/storage/index/index_metadata.json"

UPLOAD_DIR = "backend/storage/uploads"
INDEX_PATH = "backend/storage/index/index.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("backend/storage/index", exist_ok=True)

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {"indexed_files": []}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

def save_metadata(metadata: dict):
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

@router.post("/upload-docs")
def upload_docs(files: List[UploadFile] = File(...)):
    # ---------- Load metadata ----------
    metadata = load_metadata()
    indexed = {
        (item["filename"], item["hash"])
        for item in metadata["indexed_files"]
    }

    new_file_paths = []
    new_metadata_entries = []

    # ---------- Save files + compute hashes ----------
    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_hash = compute_file_hash(save_path)

        key = (file.filename, file_hash)
        if key in indexed:
            continue  # already indexed

        new_file_paths.append(save_path)
        new_metadata_entries.append({
            "filename": file.filename,
            "hash": file_hash
        })

    if not new_file_paths:
        return {
            "message": "No new documents to index.",
            "indexed_files": metadata["indexed_files"]
        }

    # ---------- Index logic ----------
    if os.path.exists(INDEX_PATH):
        # Incremental append
        store = DocumentStore.load(INDEX_PATH)
        new_docs = store.load_documents(new_file_paths)
        store.add_documents(new_docs)
    else:
        # Cold start
        store = DocumentStore()
        docs = store.load_documents(new_file_paths)
        store.build_indexes(docs)

    # ---------- Persist ----------
    store.save(INDEX_PATH)
    metadata["indexed_files"].extend(new_metadata_entries)
    save_metadata(metadata)

    return {
        "message": "Documents indexed successfully",
        "new_files": [f["filename"] for f in new_metadata_entries],
        "total_indexed": len(metadata["indexed_files"])
    }