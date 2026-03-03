import os
import glob
import numpy as np
import torch
from tqdm import tqdm


class EmbeddingExtractor:
    """
    Enterprise-level batch-safe embedding extractor.

    Features:
        - Saves each batch separately (.npz)
        - Resume-safe
        - Scalable
        - Final merge utility
    """

    def __init__(
        self,
        model,
        dataloader,
        device=None,
        output_dir="embeddings",
        file_prefix="dataset"
    ):
        self.model = model
        self.dataloader = dataloader  # MUST be only DataLoader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.file_prefix = file_prefix

        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Extract + Save Each Batch Separately
    # ----------------------------------------------------------

    @torch.no_grad()
    def extract_and_save(self):
        self.model.eval()
        self.model.to(self.device)

        print("🚀 Starting enterprise batch extraction...")

        existing_batches = self._get_existing_batch_numbers()
        start_batch_idx = max(existing_batches) + 1 if existing_batches else 0

        for batch_idx, batch in enumerate(
            tqdm(self.dataloader, desc="🔍 Extracting"),
            start=start_batch_idx
        ):

            # Safe unpacking
            if len(batch) == 3:
                images, batch_labels, batch_paths = batch
            else:
                raise RuntimeError(
                    f"Unexpected batch structure. Expected 3 items, got {len(batch)}"
                )

            batch_file = self._get_batch_filename(batch_idx)

            # Resume-safe skip
            if os.path.exists(batch_file):
                print(f"⚠️ Skipping existing batch {batch_idx}")
                continue

            images = images.to(self.device, non_blocking=True)

            feats = self.model(images)
            feats = feats.detach().cpu().numpy().astype("float32")

            batch_labels = batch_labels.cpu().numpy()
            batch_paths = np.array(batch_paths)

            np.savez_compressed(
                batch_file,
                embeddings=feats,
                labels=batch_labels,
                paths=batch_paths
            )

        print("✅ Extraction completed safely.")

    # ----------------------------------------------------------
    # Merge All Batches Into Single File
    # ----------------------------------------------------------

    def merge_batches(self, output_filename="final_embeddings.npz"):
        print("🔄 Merging batch files...")

        batch_files = sorted(
            glob.glob(os.path.join(self.output_dir, f"{self.file_prefix}_batch_*.npz"))
        )

        if not batch_files:
            raise RuntimeError("No batch files found to merge.")

        all_embeddings = []
        all_labels = []
        all_paths = []

        for file in tqdm(batch_files, desc="📦 Loading batches"):
            data = np.load(file, allow_pickle=True)
            all_embeddings.append(data["embeddings"])
            all_labels.append(data["labels"])
            all_paths.append(data["paths"])

        embeddings = np.vstack(all_embeddings).astype("float32")
        labels = np.concatenate(all_labels)
        paths = np.concatenate(all_paths)

        final_path = os.path.join(self.output_dir, output_filename)

        np.savez_compressed(
            final_path,
            embeddings=embeddings,
            labels=labels,
            paths=paths
        )

        print(f"✅ Final merged file saved at: {final_path}")

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _get_batch_filename(self, batch_idx):
        return os.path.join(
            self.output_dir,
            f"{self.file_prefix}_batch_{batch_idx:06d}.npz"
        )

    def _get_existing_batch_numbers(self):
        batch_files = glob.glob(
            os.path.join(self.output_dir, f"{self.file_prefix}_batch_*.npz")
        )

        batch_numbers = []
        for file in batch_files:
            name = os.path.basename(file)
            number = name.split("_batch_")[-1].split(".")[0]
            batch_numbers.append(int(number))

        return sorted(batch_numbers)
