# similarity_wrapper.py

import torch
import numpy as np
import faiss


class SimilarityWrapper:
    """
    Converts embedding model into scalar similarity function.

    This is required for:
        - LIME
        - SHAP
        - RISE
        - Any explainability method

    Input:
        image (numpy H,W,3 or batch N,H,W,3)

    Output:
        similarity score (scalar or batch of scalars)
    """

    SUPPORTED_SIMILARITIES = ["COSINE", "EUCLIDEAN"]

    def __init__(
        self,
        model,
        target_embedding: np.ndarray,
        similarity: str = "COSINE",
        device: str = None,
        transform=None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity = similarity.upper()
        self.transform = transform  # Optional preprocessing pipeline

        if self.similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(
                f"Similarity must be one of {self.SUPPORTED_SIMILARITIES}"
            )

        self.model.eval()

        # Store target embedding properly
        self.target_embedding = target_embedding.astype("float32").reshape(1, -1)

        if self.similarity == "COSINE":
            faiss.normalize_L2(self.target_embedding)

    # ----------------------------------------------------------
    # Forward Call
    # ----------------------------------------------------------

    @torch.no_grad()
    def __call__(self, image_numpy: np.ndarray):
        """
        Input:
            image_numpy:
                - (H, W, 3)
                OR
                - (N, H, W, 3)

        Returns:
            scalar similarity (if single image)
            OR
            numpy array of similarities (if batch)
        """

        # Ensure batch dimension
        if image_numpy.ndim == 3:
            image_numpy = np.expand_dims(image_numpy, axis=0)

        batch_size = image_numpy.shape[0]

        # Convert to tensor
        images = []

        for img in image_numpy:
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

            images.append(img)

        image_tensor = torch.stack(images).to(self.device)

        # Extract embeddings
        embeddings = self.model(image_tensor)
        embeddings = embeddings.detach().cpu().numpy().astype("float32")

        # Similarity computation
        if self.similarity == "COSINE":
            faiss.normalize_L2(embeddings)
            scores = np.dot(embeddings, self.target_embedding.T).squeeze()

        else:  # EUCLIDEAN
            distances = np.linalg.norm(
                embeddings - self.target_embedding, axis=1
            )
            scores = -distances  # Negative because higher is better

        # Return scalar if single input
        if batch_size == 1:
            return float(scores)

        return scores
