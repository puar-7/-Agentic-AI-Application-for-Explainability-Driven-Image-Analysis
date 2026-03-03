import torch
import torch.nn as nn
from torchvision import models


class ModelLoader:
    """
    Loads face embedding models in inference mode.

    Supported Models:
        - RESNET50 (ImageNet pretrained)
        - FACENET (InceptionResnetV1 pretrained on VGGFace2)

    Returns:
        model (ready for embedding extraction)
    """

    SUPPORTED_MODELS = ["RESNET", "FACENET"]

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name.upper()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        print(f"🔹 Loading model: {self.model_name}")
        print(f"🔹 Using device: {self.device}")

        self.model = self._load_model()
        self.embedding_dim = self._get_embedding_dim()

    # ----------------------------------------------------------
    # Model Loader
    # ----------------------------------------------------------

    def _load_model(self):
        if self.model_name == "RESNET":
            return self._load_resnet()

        elif self.model_name == "FACENET":
            return self._load_facenet()

    # ----------------------------------------------------------
    # RESNET50 Loader (ImageNet pretrained)
    # ----------------------------------------------------------

    def _load_resnet(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove classification head → output embeddings
        model.fc = nn.Identity()

        model.eval()
        model.to(self.device)

        return model

    # ----------------------------------------------------------
    # FACENET Loader (VGGFace2 pretrained)
    # ----------------------------------------------------------

    def _load_facenet(self):
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is not installed.\n"
                "Install it using: pip install facenet-pytorch"
            )

        model = InceptionResnetV1(pretrained="vggface2")
        model.eval()
        model.to(self.device)

        return model

    # ----------------------------------------------------------
    # Embedding Dimension
    # ----------------------------------------------------------

    def _get_embedding_dim(self):
        if self.model_name == "RESNET":
            return 2048
        elif self.model_name == "FACENET":
            return 512

    # ----------------------------------------------------------
    # Public Getter
    # ----------------------------------------------------------

    def get_model(self):
        return self.model

    def get_embedding_dim(self):
        return self.embedding_dim
