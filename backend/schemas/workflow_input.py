from pydantic import BaseModel, Field
from typing import Literal, Optional


class WorkflowInput(BaseModel):
    """
    Structured workflow configuration provided by the user in workflow mode.
    """

    dataset_name: Literal["CELEBA", "VGGFACE2", "DIGIFACE"] = Field(
        ...,
        description="Supported datasets: CELEBA, VGGFACE2, DIGIFACE"
    )

    model_name: Literal["RESNET", "FACENET"] = Field(
        ...,
        description="Supported models: RESNET, FACENET"
    )

    target_variable: Optional[str] = Field(
        default=None,
        description="Target variable for prediction (reserved for future use)"
    )

    spurious_attribute: Optional[str] = Field(
        default=None,
        description="Spurious attribute for bias analysis (reserved for future use)"
    )

    execution_mode: Literal["white", "black", "both"] = Field(
        ...,
        description="Execution mode for analysis"
    )

    similarity: Literal["COSINE", "EUCLIDEAN"] = Field(
        default="COSINE",
        description="Similarity measure for retrieval"
    )

    explainer: Literal["LIME", "SHAP", "RISE"] = Field(
        default="LIME",
        description="Explainability method"
    )