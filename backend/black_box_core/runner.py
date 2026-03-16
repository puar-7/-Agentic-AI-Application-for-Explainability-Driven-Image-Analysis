# runner.py

import os
import numpy as np
import torch

from core.dataset_loader import DatasetLoader
from core.model_loader import ModelLoader
from core.embedding_extractor import EmbeddingExtractor
from core.faiss_index import FaissIndex
from evaluation.metrics_report import evaluate_retrieval, save_results
from generate_metadata import MetadataGenerator
from explainability.lime_explainer import LimeSimilarityExplainer


class Runner:
    """
    Main pipeline orchestrator.
    """

    def __init__(self):
        self.dataset_name = None
        self.model_name = None
        self.similarity_measure = None
        self.explain_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------
    # INPUT HANDLING
    # ----------------------------------------------------------

    def _get_choice(self, title, options):
        print(f"\n{title}")
        for key, value in options.items():
            print(f"{key}. {value}")

        while True:
            choice = input("ENTER YOUR CHOICE: ").strip()
            if choice in options:
                print(f"✅ SELECTED: {options[choice]}\n")
                return options[choice]
            else:
                print("❌ Invalid choice. Try again.")

    def inputs(self):
        self.dataset_name = self._get_choice(
            "SELECT DATASET",
            {"1": "CELEBA", "2": "VGGFACE2", "3": "DIGIFACE"}
        )

        self.model_name = self._get_choice(
            "SELECT MODEL",
            {"1": "RESNET", "2": "FACENET"}
        )

        self.similarity_measure = self._get_choice(
            "SELECT SIMILARITY",
            {"1": "COSINE", "2": "EUCLIDEAN"}
        )

        self.explain_model = self._get_choice(
            "SELECT EXPLAINABILITY METHOD",
            {"1": "LIME", "2": "SHAP", "3": "RISE"}
        )

    # ----------------------------------------------------------
    # MAIN PIPELINE
    # ----------------------------------------------------------

    def run(self):

        os.makedirs("embeddings", exist_ok=True)
        os.makedirs("faiss_indexes", exist_ok=True)

        embedding_prefix = f"{self.model_name.lower()}_{self.dataset_name.lower()}"
        final_embedding_file = os.path.join(
            "embeddings",
            f"{embedding_prefix}_final.npz"
        )

        # ------------------------------------------------------
        # STEP 1 — EMBEDDING EXTRACTION
        # ------------------------------------------------------

        if not os.path.exists(final_embedding_file):

            print("🚀 Embeddings not found. Starting extraction...")

            dataset_loader = DatasetLoader(
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                batch_size=8
            )

            dataloader, _ = dataset_loader.get_loader()

            model_loader = ModelLoader(self.model_name)
            model = model_loader.get_model()

            extractor = EmbeddingExtractor(
                model=model,
                dataloader=dataloader,
                device=self.device,
                output_dir="embeddings",
                file_prefix=embedding_prefix
            )

            extractor.extract_and_save()
            extractor.merge_batches(
                output_filename=f"{embedding_prefix}_final.npz"
            )

        else:
            print("✅ Final embeddings already exist. Skipping extraction.")

        # ------------------------------------------------------
        # STEP 2 — LOAD EMBEDDINGS
        # ------------------------------------------------------

        print("📦 Loading embeddings...")
        data = np.load(final_embedding_file, allow_pickle=True)

        embeddings = data["embeddings"].astype("float32")
        paths = data["paths"]

        embedding_dim = embeddings.shape[1]
        print(f"✅ Embeddings loaded. Shape: {embeddings.shape}")

        # ------------------------------------------------------
        # STEP 3 — BUILD / LOAD FAISS INDEX
        # ------------------------------------------------------

        index_path = os.path.join(
            "faiss_indexes",
            f"{embedding_prefix}_{self.similarity_measure.lower()}.index"
        )

        faiss_idx = FaissIndex(
            dim=embedding_dim,
            similarity=self.similarity_measure,
            use_gpu=False
        )

        if not os.path.exists(index_path):

            print("🚀 Building FAISS index...")
            faiss_idx.add(embeddings)
            faiss_idx.save(index_path)

        else:
            print("✅ FAISS index already exists. Loading...")
            faiss_idx.load(index_path)

        # ------------------------------------------------------
        # STEP 4 — SANITY RETRIEVAL
        # ------------------------------------------------------

        print("\n🔍 Performing sanity retrieval test...")

        query_idx = np.random.randint(0, len(embeddings))
        query_embedding = embeddings[query_idx].reshape(1, -1)

        distances, indices = faiss_idx.search(query_embedding, k=1)
        matched_idx = indices[0][0]

        print(f"🔎 Query Image  : {paths[query_idx]}")
        print(f"🎯 Matched Image: {paths[matched_idx]}")
        print(f"📏 Similarity Score: {distances[0][0]}")

        print("\n✅ Retrieval pipeline completed successfully.")

        # ------------------------------------------------------
        # STEP 4.5 — GENERATE METADATA
        # ------------------------------------------------------

        print("\n📁 Checking / Generating Metadata...")

        dataset_path = os.path.join("data", self.dataset_name.lower())

        metadata_generator = MetadataGenerator(
            dataset_path=dataset_path,
            dataset_name=self.dataset_name
        )

        metadata_path = metadata_generator.generate()

        # ------------------------------------------------------
        # STEP 5 — RETRIEVAL EVALUATION (10K RANDOM)
        # ------------------------------------------------------

        print("\n📊 Running Retrieval Evaluation (10,000 RANDOM queries)...")

        results, total_samples = evaluate_retrieval(
            index_path=index_path,
            embeddings_path=final_embedding_file,
            metadata_path=metadata_path,
            similarity=self.similarity_measure,
            top_k=5,
            num_eval_queries=10 # 10000
        )

        print("\n===== 📊 Evaluation Results =====")
        for key, value in results.items():
            print(f"{key}: {value:.6f}")

        save_results(
            results=results,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            similarity=self.similarity_measure,
            total_samples=total_samples,
            top_k=5
        )

        print("✅ Evaluation reports saved inside outputs/reports/metric_reports/")

        # ------------------------------------------------------
        # STEP 6 — EXPLAINABILITY (LIME)
        # ------------------------------------------------------

        if self.explain_model == "LIME":

            print("\n🔥 Starting LIME Explainability...")

            model_loader = ModelLoader(self.model_name)
            model = model_loader.get_model()

            explainer = LimeSimilarityExplainer(
                model=model,
                model_name=self.model_name
            )

            explainer.explain_dataset(
                image_paths=paths,
                embeddings=embeddings,
                index=faiss_idx,
                max_images=10, #200
                num_samples=50, #150
                num_features=6
            )

            print("\n✅ LIME explainability completed.")
            print("📂 Heatmaps saved inside: outputs/heatmaps/")

        else:
            print("\n⚠ Selected explainability method not implemented yet.")

        print("\n🎯 Pipeline fully completed.")
