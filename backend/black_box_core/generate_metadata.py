import os
import pickle


class MetadataGenerator:
    """
    Generic metadata generator for ANY dataset structure.

    Works for:
        - Flat datasets (CelebA style)
        - Identity folder datasets (VGGFace2 style)

    All metadata files are saved inside:
        C:/Users/abdul/Desktop/XAI Project/metadata
    """

    def __init__(self, dataset_path, dataset_name):

        self.dataset_path = os.path.abspath(dataset_path)
        self.dataset_name = dataset_name.lower()

        # ✅ Fixed global metadata directory
        self.output_dir = os.path.abspath("metadata")

        os.makedirs(self.output_dir, exist_ok=True)

        self.output_path = os.path.join(
            self.output_dir,
            f"{self.dataset_name}_metadata.pkl"
        )

    # ----------------------------------------------------------
    # GENERATE METADATA
    # ----------------------------------------------------------

    def generate(self, overwrite=False):

        # ✅ Resume-safe check
        if os.path.exists(self.output_path) and not overwrite:
            print(f"✅ Metadata already exists at: {self.output_path}")
            print("⏩ Skipping generation (use overwrite=True to regenerate).")
            return self.output_path

        print(f"\n🔄 Generating metadata for dataset: {self.dataset_name}")
        print(f"📂 Scanning dataset path: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"❌ Dataset path not found: {self.dataset_path}"
            )

        paths = []
        labels = []

        label_counter = 0

        subfolders = [
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ]

        # ------------------------------------------------------
        # CASE 1 — Identity folder dataset
        # ------------------------------------------------------
        if subfolders:

            print("🔎 Detected identity-folder dataset structure.")

            for identity in sorted(subfolders):

                identity_path = os.path.join(self.dataset_path, identity)

                image_files = [
                    f for f in os.listdir(identity_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                for img in sorted(image_files):
                    full_path = os.path.abspath(
                        os.path.join(identity_path, img)
                    )
                    paths.append(full_path)
                    labels.append(label_counter)

                label_counter += 1

        # ------------------------------------------------------
        # CASE 2 — Flat dataset
        # ------------------------------------------------------
        else:

            print("🔎 Detected flat dataset structure.")

            image_files = [
                f for f in os.listdir(self.dataset_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for idx, img in enumerate(sorted(image_files)):
                full_path = os.path.abspath(
                    os.path.join(self.dataset_path, img)
                )
                paths.append(full_path)
                labels.append(idx)

        if len(paths) == 0:
            raise ValueError("❌ No images found in dataset.")

        metadata = {
            "paths": paths,
            "labels": labels
        }

        with open(self.output_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"\n✅ Metadata successfully saved!")
        print(f"📂 Location: {self.output_path}")
        print(f"🖼️ Total images indexed: {len(paths)}")

        return self.output_path
