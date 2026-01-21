import requests

# Backend base URL
API_BASE = "http://127.0.0.1:8000"


def post_json(endpoint: str, payload: dict):
    """
    Send a JSON POST request to the backend.
    Used by chat and workflow UIs.
    """
    return requests.post(
        f"{API_BASE}{endpoint}",
        json=payload,
        timeout=300
    )


def post_files(endpoint: str, files):
    """
    Send multipart/form-data POST request.
    Used for document uploads.
    """
    return requests.post(
        f"{API_BASE}{endpoint}",
        files=files,
        timeout=300
    )
