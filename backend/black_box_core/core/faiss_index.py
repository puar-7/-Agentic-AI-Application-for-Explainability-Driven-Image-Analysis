# faiss_index.py

import os
import faiss
import numpy as np


class FaissIndex:
    """
    Production-ready FAISS index manager.

    Supports:
        - COSINE similarity
        - EUCLIDEAN similarity
        - GPU (optional)
        - Search capability
        - Metadata storage (labels + paths)
    """

    SUPPORTED_SIMILARITIES = ["COSINE", "EUCLIDEAN"]

    def __init__(self, dim: int, similarity: str = "COSINE", use_gpu: bool = False):
        self.dim = dim
        self.similarity = similarity.upper()
        self.use_gpu = use_gpu
        self.index = None

        if self.similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(
                f"Similarity must be one of {self.SUPPORTED_SIMILARITIES}"
            )

        self._create_index()

    # ----------------------------------------------------------
    # Index Creation
    # ----------------------------------------------------------

    def _create_index(self):
        if self.similarity == "COSINE":
            # Cosine = Inner Product with normalized vectors
            index = faiss.IndexFlatIP(self.dim)
        else:
            index = faiss.IndexFlatL2(self.dim)

        if self.use_gpu:
            if not faiss.get_num_gpus():
                raise RuntimeError("GPU requested but no FAISS GPU available.")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.index = index

    # ----------------------------------------------------------
    # Build from NPZ (embeddings + metadata)
    # ----------------------------------------------------------

    def build_from_npz(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"{npz_path} not found")

        data = np.load(npz_path, allow_pickle=True)

        embeddings = data["embeddings"].astype("float32")
        self.labels = data.get("labels")
        self.paths = data.get("paths")

        print(f"📦 Loaded embeddings shape: {embeddings.shape}")

        if self.similarity == "COSINE":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        print(f"✅ FAISS index built with {self.index.ntotal} vectors")

    # ----------------------------------------------------------
    # Add embeddings manually
    # ----------------------------------------------------------

    def add(self, embeddings: np.ndarray):
        embeddings = embeddings.astype("float32")

        if self.similarity == "COSINE":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    # ----------------------------------------------------------
    # Search
    # ----------------------------------------------------------

    def search(self, query_embedding: np.ndarray, k: int = 1):
        """
        Returns:
            distances: (N, k)
            indices: (N, k)
        """

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = query_embedding.astype("float32")

        if self.similarity == "COSINE":
            faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        return distances, indices

    # ----------------------------------------------------------
    # Search + Return Metadata
    # ----------------------------------------------------------

    def search_with_metadata(self, query_embedding: np.ndarray, k: int = 1):
        distances, indices = self.search(query_embedding, k)

        results = []

        for i in range(len(indices)):
            batch_result = []
            for j in range(k):
                idx = indices[i][j]

                result = {
                    "distance": float(distances[i][j]),
                    "index": int(idx),
                }

                if hasattr(self, "labels") and self.labels is not None:
                    result["label"] = int(self.labels[idx])

                if hasattr(self, "paths") and self.paths is not None:
                    result["path"] = str(self.paths[idx])

                batch_result.append(result)

            results.append(batch_result)

        return results

    # ----------------------------------------------------------
    # Save / Load Index
    # ----------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Move back to CPU before saving if GPU used
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)

        print(f"💾 FAISS index saved to: {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")

        self.index = faiss.read_index(path)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        print(f"✅ FAISS index loaded from: {path}")
