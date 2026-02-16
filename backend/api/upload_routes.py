from fastapi import APIRouter, HTTPException, UploadFile, File, Request
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

# --- NEW ENDPOINT: LIST DOCUMENTS (For Tasks 3 & 5) ---
@router.get("/documents")
def get_documents():
    """
    Returns a list of all currently indexed filenames.
    Used by the frontend to prevent duplicate uploads and show the sidebar list.
    """
    metadata = load_metadata()
    # Extract just the filenames for the UI
    filenames = [item["filename"] for item in metadata.get("indexed_files", [])]
    return {"documents": filenames}        

@router.post("/upload-docs")
def upload_docs(request: Request, files: List[UploadFile] = File(...)):
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

        #Save the uploaded file to disk
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_hash = compute_file_hash(save_path)

        key = (file.filename, file_hash)
        if key in indexed:
            # If duplicate, we don't index
            # For now, we just skip adding it to the processing list.
            # Clean up the duplicate upload to save space
            if os.path.exists(save_path):
                os.remove(save_path)
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
    store = request.app.state.document_store

    try:
        # Load and Validate Documents
        # This will raise ValueError if file type is unsupported (e.g. .docx)
        new_docs = store.load_documents(new_file_paths)

        # update vector store
        if store.vector_store is None:
            # Cold start for the live object
            store.build_indexes(new_docs)
        else:
            # Incremental append
            store.add_documents(new_docs)


        
        # ---------- Persist ----------
        store.save(INDEX_PATH)
        metadata["indexed_files"].extend(new_metadata_entries)
        save_metadata(metadata)

        return {
            "message": "Documents indexed successfully",
            "new_files": [f["filename"] for f in new_metadata_entries],
            "total_indexed": len(metadata["indexed_files"])
        }
    
    except ValueError as e:
        # ERROR HANDLING: 400 Bad Request for unsupported files
        # Cleanup: Remove the invalid files from disk
        print(f"Validation Error: {e}")
        for path in new_file_paths:
            if os.path.exists(path):
                os.remove(path)
        
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # ERROR HANDLING: 500 Internal Server Error for anything else
        # Cleanup: Remove files to prevent 'corrupted' state
        print(f"Indexing Error: {e}")
        for path in new_file_paths:
            if os.path.exists(path):
                os.remove(path)
                
        raise HTTPException(status_code=500, detail=f"Server error during indexing: {str(e)}")