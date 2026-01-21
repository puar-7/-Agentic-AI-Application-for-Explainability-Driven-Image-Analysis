from fastapi import APIRouter
import shutil
import os

router = APIRouter()

@router.post("/clear")
def clear_all():
    if os.path.exists("backend/storage/uploads"):
        shutil.rmtree("backend/storage/uploads")
        os.makedirs("backend/storage/uploads")

    if os.path.exists("backend/storage/index"):
        shutil.rmtree("backend/storage/index")
        os.makedirs("backend/storage/index")

    return {"message": "All documents and index cleared"}
