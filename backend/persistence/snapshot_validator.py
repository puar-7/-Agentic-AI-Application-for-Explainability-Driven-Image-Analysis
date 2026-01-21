import json
from typing import Dict, Any


class SnapshotValidationError(Exception):
    """Raised when execution snapshot is not JSON-serializable."""
    pass


def validate_snapshot(snapshot: Dict[str, Any]) -> None:
    """
    Validate that the execution snapshot is JSON-serializable.
    Raises SnapshotValidationError if validation fails.
    """
    try:
        json.dumps(snapshot)
    except (TypeError, ValueError) as e:
        raise SnapshotValidationError(
            f"Execution snapshot is not JSON-serializable: {e}"
        ) from e