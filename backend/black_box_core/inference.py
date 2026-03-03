# inference.py

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from core.model_loader import ModelLoader
from core.faiss_index import FaissIndex


class Inference:
    """
    Real-world retrieval system.

    Steps:
        - Load trained embedding model
        - Load FAISS index
        - Load metadata (paths + labels)
        - Extract query embedding
        - Perform top-k search
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        similarity: str,
        top_k: int = 5,
        device: str = None,
    ):
        self.dataset_name = dataset_name.upper()
        self.model_name = model_name.upper()
        self.similarity = similarity.upper()
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_prefix = f"{self.model_name.lower()}_{self.dataset_name.lower()}"
        self.embedding_file = os.path.join(
            "embeddings",
            f"{self.embedding_prefix}_final.npz"
        )
        self.index_file = os.path.join(
            "faiss_indexes",
            f"{self.embedding_prefix}_{self.similarity.lower()}.index"
        )

        self.model = None
        self.faiss_index = None
        self.embeddings = None
        self.labels = None
        self.paths = None

        self.transform = self._get_transform()

    # ----------------------------------------------------------
    # Load Components
    # ----------------------------------------------------------

    def load_system(self):

        # ---------- Load Model ----------
        print("🔄 Loading model...")
        model_loader = ModelLoader(self.model_name)
        self.model = model_loader.get_model()
        self.model.eval()

        # ---------- Load Embeddings Metadata ----------
        if not os.path.exists(self.embedding_file):
            raise FileNotFoundError("Embeddings file not found.")

        print("📦 Loading embeddings metadata...")
        data = np.load(self.embedding_file, allow_pickle=True)

        self.embeddings = data["embeddings"].astype("float32")
        self.labels = data["labels"]
        self.paths = data["paths"]

        embedding_dim = self.embeddings.shape[1]

        # ---------- Load FAISS Index ----------
        if not os.path.exists(self.index_file):
            raise FileNotFoundError("FAISS index file not found.")

        print("🔄 Loading FAISS index...")
        self.faiss_index = FaissIndex(
            dim=embedding_dim,
            similarity=self.similarity,
            use_gpu=False
        )
        self.faiss_index.load(self.index_file)

        print("✅ System loaded successfully.\n")

    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------

    def _get_transform(self):

        if self.model_name == "FACENET":
            return transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

        elif self.model_name == "RESNET":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:
            raise ValueError("Unsupported model.")

    # ----------------------------------------------------------
    # Extract Embedding From Query Image
    # ----------------------------------------------------------

    @torch.no_grad()
    def _extract_query_embedding(self, image_path: str):

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        embedding = self.model(image_tensor)
        embedding = embedding.detach().cpu().numpy().astype("float32")

        return embedding

    # ----------------------------------------------------------
    # Run Inference
    # ----------------------------------------------------------

    def predict(self, image_path: str):

        if self.model is None or self.faiss_index is None:
            raise RuntimeError("Call load_system() before prediction.")

        print(f"\n🔍 Query Image: {image_path}")

        query_embedding = self._extract_query_embedding(image_path)

        distances, indices = self.faiss_index.search(
            query_embedding,
            k=self.top_k
        )

        results = []

        for rank in range(self.top_k):
            idx = indices[0][rank]
            score = distances[0][rank]

            result = {
                "rank": rank + 1,
                "matched_path": self.paths[idx],
                "label": int(self.labels[idx]) if self.labels is not None else None,
                "score": float(score),
            }

            results.append(result)

        return results
