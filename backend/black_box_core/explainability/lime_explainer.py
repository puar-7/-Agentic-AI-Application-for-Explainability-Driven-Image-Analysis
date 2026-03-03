import os
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class LimeSimilarityExplainer:
    """
    Quantitative-Ready LIME Explainer for Retrieval Systems

    ✔ Structured directory saving
    ✔ Saves overlay visualization
    ✔ Saves segmentation mask (.npy)
    ✔ Saves numerical importance map (.npy)  <-- REQUIRED FOR XAI METRICS
    ✔ Saves heatmap image for visualization
    ✔ Resume-safe (checks importance file)
    ✔ CPU optimized
    ✔ Stable normalization
    ✔ Thesis-level clean structure
    """

    # ----------------------------------------------------------
    # INIT
    # ----------------------------------------------------------

    def __init__(self, model, model_name="FACENET"):

        self.device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())

        self.model = model.to(self.device)
        self.model.eval()

        self.model_name = model_name.upper()
        self.explainer = lime_image.LimeImageExplainer()
        self.transform = self._build_transform()

        # ------------------------------------------------------
        # STRUCTURED OUTPUT DIRECTORIES
        # ------------------------------------------------------

        self.base_dir = os.path.join("outputs", "heatmaps")

        self.overlay_dir = os.path.join(self.base_dir, "overlays")
        self.mask_dir = os.path.join(self.base_dir, "masks")
        self.importance_dir = os.path.join(self.base_dir, "importance_maps")
        self.heatmap_img_dir = os.path.join(self.base_dir, "heatmap_images")

        for directory in [
            self.overlay_dir,
            self.mask_dir,
            self.importance_dir,
            self.heatmap_img_dir
        ]:
            os.makedirs(directory, exist_ok=True)

    # ----------------------------------------------------------
    # TRANSFORM BUILDER
    # ----------------------------------------------------------

    def _build_transform(self):

        if self.model_name == "FACENET":
            size = (160, 160)
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        elif self.model_name == "RESNET":
            size = (224, 224)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        else:
            raise ValueError("Unsupported model name")

        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # ----------------------------------------------------------
    # EMBEDDING NORMALIZATION
    # ----------------------------------------------------------

    @staticmethod
    def _normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return embeddings / norms

    # ----------------------------------------------------------
    # COSINE SIMILARITY
    # ----------------------------------------------------------

    @staticmethod
    def _cosine_similarity(emb1, emb2):
        return F.cosine_similarity(emb1, emb2, dim=1)

    # ----------------------------------------------------------
    # LIME PREDICTION FUNCTION
    # ----------------------------------------------------------

    def _similarity_predict(self, images, target_embedding):

        tensors = []

        for img in images:
            img_pil = Image.fromarray(img.astype(np.uint8))
            tensors.append(self.transform(img_pil))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)

        similarities = self._cosine_similarity(
            embeddings,
            target_embedding.expand(embeddings.size(0), -1)
        )

        return similarities.cpu().numpy().reshape(-1, 1)

    # ----------------------------------------------------------
    # SINGLE IMAGE EXPLANATION
    # ----------------------------------------------------------

    def explain_image(
        self,
        image_path,
        target_embedding,
        num_samples=200,
        num_features=6
    ):

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        target_embedding = target_embedding.to(self.device)

        explanation = self.explainer.explain_instance(
            image_np,
            lambda imgs: self._similarity_predict(imgs, target_embedding),
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )

        # NOTE: positive_only=False keeps negative contributions too
        highlighted, segmentation_mask = explanation.get_image_and_mask(
            label=0,
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )

        filename = os.path.splitext(os.path.basename(image_path))[0]

        # ------------------------------------------------------
        # SAVE OVERLAY
        # ------------------------------------------------------
        overlay_path = os.path.join(self.overlay_dir, f"{filename}_overlay.jpg")
        Image.fromarray(highlighted).save(overlay_path)

        # ------------------------------------------------------
        # SAVE SEGMENTATION MASK
        # ------------------------------------------------------
        mask_path = os.path.join(self.mask_dir, f"{filename}_mask.npy")
        np.save(mask_path, segmentation_mask.astype(np.int32))

        # ------------------------------------------------------
        # BUILD PIXEL-WISE IMPORTANCE MAP
        # ------------------------------------------------------
        heatmap_dict = explanation.local_exp[0]

        importance_map = np.zeros(
            segmentation_mask.shape,
            dtype=np.float32
        )

        for superpixel, weight in heatmap_dict:
            importance_map[segmentation_mask == superpixel] = weight

        # Stable normalization to [0, 1]
        min_val = importance_map.min()
        max_val = importance_map.max()

        if max_val - min_val > 1e-8:
            importance_map = (importance_map - min_val) / (max_val - min_val)
        else:
            importance_map = np.zeros_like(importance_map)

        # ------------------------------------------------------
        # SAVE NUMERICAL IMPORTANCE MAP (CRITICAL FOR METRICS)
        # ------------------------------------------------------
        importance_path = os.path.join(
            self.importance_dir,
            f"{filename}_importance.npy"
        )
        np.save(importance_path, importance_map)

        # ------------------------------------------------------
        # SAVE VISUAL HEATMAP (FOR HUMANS)
        # ------------------------------------------------------
        plt.figure()
        plt.imshow(importance_map, cmap="jet")
        plt.axis("off")

        heatmap_path = os.path.join(
            self.heatmap_img_dir,
            f"{filename}_heatmap.jpg"
        )
        plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    # ----------------------------------------------------------
    # DATASET EXPLANATION
    # ----------------------------------------------------------

    def explain_dataset(
        self,
        image_paths,
        embeddings,
        index,
        max_images=1000,
        num_samples=200,
        num_features=6
    ):

        total_images = min(len(image_paths), max_images)

        print(f"Generating LIME explanations for {total_images} images...")

        embeddings = self._normalize_embeddings(embeddings)

        for idx in tqdm(range(total_images)):

            image_path = image_paths[idx]
            filename = os.path.splitext(os.path.basename(image_path))[0]

            # Resume-safe: check importance file
            importance_path = os.path.join(
                self.importance_dir,
                f"{filename}_importance.npy"
            )

            if os.path.exists(importance_path):
                continue

            query_embedding = embeddings[idx].reshape(1, -1).astype(np.float32)

            # FAISS nearest neighbor
            D, I = index.search(query_embedding, 1)
            top_index = I[0][0]

            target_embedding = torch.tensor(
                embeddings[top_index],
                dtype=torch.float32
            ).unsqueeze(0)

            self.explain_image(
                image_path=image_path,
                target_embedding=target_embedding,
                num_samples=num_samples,
                num_features=num_features
            )

        print("All structured LIME outputs saved successfully.")
