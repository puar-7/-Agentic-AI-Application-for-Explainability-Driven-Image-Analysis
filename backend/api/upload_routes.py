import os
import shutil
import zipfile
import tempfile
import json
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Request

from backend.services.document_store import DocumentStore
from backend.services.file_dispatcher import (
    load_file,
    UnsupportedFileTypeError,
    FileLoadError,
    SUPPORTED_EXTENSIONS,
)
from backend.services.utils import compute_file_hash

router = APIRouter()

METADATA_PATH = "backend/storage/index/index_metadata.json"
UPLOAD_DIR    = "backend/storage/uploads"
INDEX_PATH    = "backend/storage/index/index.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("backend/storage/index", exist_ok=True)


# ------------------------------------------------------------------
# Metadata helpers — unchanged from original
# ------------------------------------------------------------------

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {"indexed_files": []}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def save_metadata(metadata: dict):
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


# ------------------------------------------------------------------
# GET /documents — unchanged from original
# ------------------------------------------------------------------

@router.get("/documents")
def get_documents():
    """
    Returns a list of all currently indexed filenames.
    Used by the frontend to prevent duplicate uploads and show
    the sidebar list.
    """
    metadata = load_metadata()
    filenames = [item["filename"] for item in metadata.get("indexed_files", [])]
    return {"documents": filenames}


# ------------------------------------------------------------------
# POST /upload-docs
# ------------------------------------------------------------------

@router.post("/upload-docs")
def upload_docs(request: Request, files: List[UploadFile] = File(...)):
    """
    Handles both direct file uploads and ZIP folder uploads.

    Direct upload (.pdf, .txt, .docx, .xlsx, .pptx):
        Behaves identically to original — hash-based deduplication,
        load via dispatcher, index, persist.

    ZIP upload (.zip):
        Extracts to a temp directory (cleaned up automatically),
        walks the tree, processes each file individually.
        Partial success is supported — one bad file does not abort
        the entire ZIP.

    Response schema (both paths return this shape):
        {
            "source":               "direct" | "zip",
            "indexed":              [filenames successfully indexed],
            "skipped_duplicates":   [filenames already in index],
            "skipped_unsupported":  [filenames with unsupported extension],
            "failed":               [{"file": name, "reason": str}],
            "total_indexed":        int
        }
    """

    # Separate ZIPs from direct files
    zip_files    = [f for f in files if f.filename.lower().endswith(".zip")]
    direct_files = [f for f in files if not f.filename.lower().endswith(".zip")]

    # We process at most one ZIP per request — if multiple ZIPs are
    # uploaded, process the first and treat the rest as unsupported.
    # This keeps the logic simple and the UX predictable.
    extra_zips = zip_files[1:] if len(zip_files) > 1 else []

    results = {
        "source": "zip" if zip_files else "direct",
        "indexed": [],
        "skipped_duplicates": [],
        "skipped_unsupported": [f.filename for f in extra_zips],
        "failed": [],
        "total_indexed": 0,
    }

    store = request.app.state.document_store

    # ------------------------------------------------------------------
    # Path A — Direct file uploads
    # ------------------------------------------------------------------
    if direct_files:
        _handle_direct_uploads(direct_files, store, results)

    # ------------------------------------------------------------------
    # Path B — ZIP upload
    # ------------------------------------------------------------------
    if zip_files:
        _handle_zip_upload(zip_files[0], store, results)

    # Persist the FAISS + BM25 index to disk if anything was indexed.
    # Metadata is already persisted inside each handler with real hashes —
    # _handle_direct_uploads and _process_extracted_directory both call
    # save_metadata() internally. We only need store.save() here.
    if results["indexed"]:
        store.save(INDEX_PATH)

    results["total_indexed"] = len(results["indexed"])
    return results


# ------------------------------------------------------------------
# Path A — Direct upload handler
# ------------------------------------------------------------------

def _handle_direct_uploads(
    files: List[UploadFile],
    store: DocumentStore,
    results: dict,
):
    """
    Processes each directly uploaded file.
    Maintains the original hash-based deduplication behaviour.
    Populates results dict in place.
    """
    metadata = load_metadata()
    indexed_hashes = {
        item["hash"] for item in metadata["indexed_files"]
        if item.get("hash")
    }

    all_new_docs = []
    new_metadata_entries = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()

        # Check extension before saving to disk — fast rejection
        if ext not in SUPPORTED_EXTENSIONS:
            results["skipped_unsupported"].append(file.filename)
            continue

        save_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_hash = compute_file_hash(save_path)

        if file_hash in indexed_hashes:
            # Already indexed — clean up the file we just saved
            os.remove(save_path)
            results["skipped_duplicates"].append(file.filename)
            continue

        # Attempt to load via dispatcher
        try:
            docs = load_file(save_path)

            if not docs:
                # File loaded but produced no text (e.g. scanned PDF)
                results["failed"].append({
                    "file": file.filename,
                    "reason": "No extractable text found in file."
                })
                os.remove(save_path)
                continue

            all_new_docs.extend(docs)
            new_metadata_entries.append({
                "filename": file.filename,
                "hash": file_hash,
            })
            results["indexed"].append(file.filename)

        except (UnsupportedFileTypeError, FileLoadError) as e:
            results["failed"].append({
                "file": file.filename,
                "reason": str(e),
            })
            if os.path.exists(save_path):
                os.remove(save_path)

        except Exception as e:
            results["failed"].append({
                "file": file.filename,
                "reason": f"Unexpected error: {type(e).__name__}: {e}",
            })
            if os.path.exists(save_path):
                os.remove(save_path)

    if not all_new_docs:
        return

    # Index all new documents together
    try:
        if store.vector_store is None:
            store.build_indexes(all_new_docs)
        else:
            store.add_documents(all_new_docs)

        # Commit metadata for successfully indexed files
        metadata["indexed_files"].extend(new_metadata_entries)
        save_metadata(metadata)

    except Exception as e:
        # Indexing itself failed — move these back to failed
        for entry in new_metadata_entries:
            results["indexed"].remove(entry["filename"])
            results["failed"].append({
                "file": entry["filename"],
                "reason": f"Indexing failed: {type(e).__name__}: {e}",
            })


# ------------------------------------------------------------------
# Path B — ZIP upload handler
# ------------------------------------------------------------------

def _handle_zip_upload(
    zip_file: UploadFile,
    store: DocumentStore,
    results: dict,
):
    """
    Extracts a ZIP, walks the contents, and processes each supported
    file individually. Partial success is fully supported.

    Uses tempfile.TemporaryDirectory as a context manager to guarantee
    cleanup even if an exception occurs partway through.

    Populates results dict in place.
    """

    # Save the ZIP to disk first so zipfile can open it
    zip_save_path = os.path.join(UPLOAD_DIR, zip_file.filename)

    try:
        with open(zip_save_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)

    except Exception as e:
        results["failed"].append({
            "file": zip_file.filename,
            "reason": f"Could not save uploaded ZIP: {e}",
        })
        return

    # Validate it's actually a ZIP (not a renamed file)
    if not zipfile.is_zipfile(zip_save_path):
        results["failed"].append({
            "file": zip_file.filename,
            "reason": "File is not a valid ZIP archive.",
        })
        os.remove(zip_save_path)
        return

    # ------------------------------------------------------------------
    # Extract to a temporary directory
    # TemporaryDirectory cleans up automatically when the with block exits,
    # even if an exception is raised inside it.
    # ------------------------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # --- ZIP Slip protection ---
            # Validate every member path BEFORE extracting anything.
            # A malicious ZIP can contain paths like "../../etc/passwd"
            # which would escape tmp_dir if extracted naively.
            try:
                with zipfile.ZipFile(zip_save_path, "r") as zf:
                    _validate_zip_members(zf, tmp_dir)
                    zf.extractall(tmp_dir)

            except zipfile.BadZipFile:
                results["failed"].append({
                    "file": zip_file.filename,
                    "reason": "ZIP archive is corrupted or unreadable.",
                })
                return

            except RuntimeError as e:
                # RuntimeError is raised by zipfile for password-protected archives
                results["failed"].append({
                    "file": zip_file.filename,
                    "reason": "ZIP is password-protected and cannot be extracted.",
                })
                return

            except ValueError as e:
                # Raised by our own _validate_zip_members on ZIP Slip detection
                results["failed"].append({
                    "file": zip_file.filename,
                    "reason": str(e),
                })
                return

            # --- Walk the extracted tree ---
            _process_extracted_directory(tmp_dir, zip_file.filename, store, results)

        # TemporaryDirectory deleted here automatically

    except Exception as e:
        results["failed"].append({
            "file": zip_file.filename,
            "reason": f"Unexpected error during ZIP processing: {type(e).__name__}: {e}",
        })

    finally:
        # Always clean up the saved ZIP file itself
        if os.path.exists(zip_save_path):
            os.remove(zip_save_path)


# ------------------------------------------------------------------
# ZIP Slip validation
# ------------------------------------------------------------------

def _validate_zip_members(zf: zipfile.ZipFile, extract_dir: str) -> None:
    """
    Validates that no ZIP member would extract outside extract_dir.

    This prevents the ZIP Slip vulnerability where a crafted ZIP
    contains members with paths like "../../etc/passwd" that escape
    the intended extraction directory.

    Raises:
        ValueError: If any member path escapes extract_dir.
    """
    # Resolve the extraction directory to an absolute path
    # so comparison works correctly regardless of CWD
    abs_extract_dir = os.path.realpath(extract_dir)

    for member in zf.namelist():
        # Resolve where this member would land after extraction
        member_path = os.path.realpath(
            os.path.join(abs_extract_dir, member)
        )

        # It must start with the extraction directory path
        if not member_path.startswith(abs_extract_dir + os.sep):
            raise ValueError(
                f"ZIP Slip detected: member '{member}' would extract "
                f"outside the target directory. Upload rejected."
            )


# ------------------------------------------------------------------
# Walk extracted directory and process each file
# ------------------------------------------------------------------

def _process_extracted_directory(
    tmp_dir: str,
    zip_filename: str,
    store: DocumentStore,
    results: dict,
):
    """
    Walks the extracted ZIP directory, attempts to load each file,
    and indexes them collectively if any succeed.

    Key behaviours:
        - Nested ZIPs (.zip inside .zip) are added to skipped_unsupported
        - Files with unsupported extensions are added to skipped_unsupported
        - Files that fail to load are added to failed with reason
        - Empty files (no text) are added to failed with explanation
        - Duplicate files (same hash as already-indexed) go to skipped_duplicates
        - Display names use ZIP-relative paths, not temp filesystem paths

    Populates results dict in place.
    """
    metadata = load_metadata()
    indexed_hashes = {
        item["hash"] for item in metadata["indexed_files"]
        if item.get("hash")
    }

    all_new_docs = []
    newly_indexed_names = []
    # Tracks real hash per indexed name so metadata stores it correctly.
    # This is what makes deduplication work on subsequent ZIP uploads —
    # previously we stored hash: "" which was filtered out by the
    # indexed_hashes set builder, making every re-upload bypass dedup.
    name_to_hash: dict = {}

    for root, dirs, files in os.walk(tmp_dir):
        # Sort for deterministic processing order
        dirs.sort()
        files.sort()

        for filename in files:
            abs_path = os.path.join(root, filename)

            # Compute display name as path relative to tmp_dir
            # e.g. "subfolder/paper.pdf" rather than a temp OS path
            rel_path = os.path.relpath(abs_path, tmp_dir)
            display_name = rel_path.replace("\\", "/")  # normalise on Windows

            ext = os.path.splitext(filename)[1].lower()

            # Nested ZIPs — skip, don't recurse
            if ext == ".zip":
                results["skipped_unsupported"].append(
                    f"{display_name} (nested ZIP — not extracted)"
                )
                continue

            # Unsupported extension
            if ext not in SUPPORTED_EXTENSIONS:
                results["skipped_unsupported"].append(display_name)
                continue

            # Deduplication by hash
            try:
                file_hash = compute_file_hash(abs_path)
            except Exception as e:
                results["failed"].append({
                    "file": display_name,
                    "reason": f"Could not hash file: {e}",
                })
                continue

            if file_hash in indexed_hashes:
                results["skipped_duplicates"].append(display_name)
                continue

            # Attempt to load via dispatcher
            try:
                docs = load_file(abs_path)

                if not docs:
                    results["failed"].append({
                        "file": display_name,
                        "reason": "No extractable text found in file.",
                    })
                    continue

                # Rewrite the source metadata to use the ZIP-relative path
                # instead of the temp directory path, which will not exist
                # after this request completes.
                for doc in docs:
                    doc.metadata["source"] = display_name

                all_new_docs.extend(docs)
                newly_indexed_names.append(display_name)
                # Store real hash — used in metadata save below AND
                # prevents re-indexing of duplicates within the same ZIP
                indexed_hashes.add(file_hash)
                name_to_hash[display_name] = file_hash

            except UnsupportedFileTypeError:
                results["skipped_unsupported"].append(display_name)

            except FileLoadError as e:
                results["failed"].append({
                    "file": display_name,
                    "reason": e.reason,
                })

            except Exception as e:
                results["failed"].append({
                    "file": display_name,
                    "reason": f"Unexpected error: {type(e).__name__}: {e}",
                })

    if not all_new_docs:
        # Nothing was indexable in this ZIP
        if not results["failed"] and not results["skipped_unsupported"]:
            results["failed"].append({
                "file": zip_filename,
                "reason": "ZIP contained no files with indexable content.",
            })
        return

    # Index all documents from the ZIP together
    try:
        if store.vector_store is None:
            store.build_indexes(all_new_docs)
        else:
            store.add_documents(all_new_docs)

        results["indexed"].extend(newly_indexed_names)

        # Store real file hashes in metadata.
        # Previously this stored hash: "" which caused every subsequent
        # upload of the same ZIP to bypass deduplication entirely
        # (empty strings are falsy and were filtered from indexed_hashes).
        for name in newly_indexed_names:
            metadata["indexed_files"].append({
                "filename": name,
                "hash": name_to_hash.get(name, ""),
            })

        save_metadata(metadata)

    except Exception as e:
        # Indexing failed — none of the ZIP files get added
        for name in newly_indexed_names:
            results["failed"].append({
                "file": name,
                "reason": f"Indexing failed: {type(e).__name__}: {e}",
            })