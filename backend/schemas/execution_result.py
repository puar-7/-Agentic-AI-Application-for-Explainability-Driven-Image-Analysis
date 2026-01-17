#White & black box nodes return identical structure

from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any


class ExecutionResult(BaseModel):
    """
    Standardized output from white-box or black-box execution nodes.
    """

    method: Literal["white_box", "black_box"] = Field(
        ...,
        description="Type of analysis performed"
    )

    status: Literal["success", "failure"] = Field(
        ...,
        description="Execution status"
    )

    summary: str = Field(
        ...,
        description="Human-readable summary of the execution result"
    )

    raw_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured raw output for further processing"
    )
