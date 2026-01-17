# No defaults → user must provide everything

# Literal enforces strict routing

# This schema becomes the contract between UI → router → nodes


from pydantic import BaseModel, Field
from typing import Literal


class WorkflowInput(BaseModel):
    """
    Structured workflow configuration provided by the user in workflow mode.
    This is assumed to be explicitly and correctly formatted (v1).
    """

    dataset_path: str = Field(
        ...,
        description="Path to the dataset used for analysis"
    )

    model_path: str = Field(
        ...,
        description="Path to the trained model"
    )

    target_variable: str = Field(
        ...,
        description="Target variable for prediction"
    )

    spurious_attribute: str = Field(
        ...,
        description="Spurious attribute to analyze bias or correlation"
    )

    execution_mode: Literal["white", "black", "both"] = Field(
        ...,
        description="Execution mode for analysis"
    )
