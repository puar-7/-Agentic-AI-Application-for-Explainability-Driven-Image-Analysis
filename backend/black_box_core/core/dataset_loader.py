import os
from PIL import Image # used to open .jpg files
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class FaceDataset(Dataset):
    """
    Hybrid Face Dataset Loader

    Automatically detects dataset structure:

    1️⃣ Flat Structure (e.g., CELEBA)
        data/dataset_name/
            img1.jpg
            img2.jpg

    2️⃣ Identity Structure (e.g., VGGFACE2, DIGIFACE)
        data/dataset_name/
            person_1/
                img1.jpg
                img2.jpg
            person_2/
                img3.jpg

    Returns:
        image_tensor, label, image_path
    """

    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []            # [(img_path, label)]
        self.label_to_index = {}
        self.index_to_label = {}

        self._load_dataset()

    # ----------------------------------------------------------
    # Auto-detect structure
    # ----------------------------------------------------------

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Dataset path does not exist: {self.root_dir}")

        print(f"📂 Loading dataset from: {self.root_dir}")

        entries = sorted(os.listdir(self.root_dir))

        subfolders = [
            d for d in entries
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        image_files = [
            f for f in entries
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ]

        # ------------------------------------------------------
        # CASE 1: Identity-based dataset (VGGFACE2 / DIGIFACE)
        # ------------------------------------------------------
        if len(subfolders) > 0:

            print("🔎 Detected identity-based dataset structure.")

            for label_id, identity in enumerate(subfolders):
                identity_path = os.path.join(self.root_dir, identity)

                self.label_to_index[identity] = label_id
                self.index_to_label[label_id] = identity

                for file in os.listdir(identity_path):
                    if file.lower().endswith(self.SUPPORTED_EXTENSIONS):
                        img_path = os.path.join(identity_path, file)
                        self.samples.append((img_path, label_id))

            print(f"🖼️ Total identities: {len(subfolders)}")

        # ------------------------------------------------------
        # CASE 2: Flat dataset (CELEBA)
        # ------------------------------------------------------
        elif len(image_files) > 0:

            print("🔎 Detected flat dataset structure.")

            for idx, file in enumerate(image_files):
                img_path = os.path.join(self.root_dir, file)

                # Each image treated as unique identity
                self.samples.append((img_path, idx))
                self.label_to_index[file] = idx
                self.index_to_label[idx] = file

        else:
            raise RuntimeError("No images found in dataset.")

        print(f"🖼️ Total images: {len(self.samples)}")

    # ----------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


# ==============================================================
# Dataset Loader Wrapper
# ==============================================================

class DatasetLoader:

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = None
    ):
        self.dataset_name = dataset_name.upper()
        self.model_name = model_name.upper()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers or min(4, os.cpu_count())

        self.dataset_path = self._resolve_dataset_path()
        self.transform = self._get_transform()

    # ----------------------------------------------------------

    def _resolve_dataset_path(self):
       # Instead, we just look for the "data" folder in the current working directory!
        data_root = os.path.abspath("data")

        dataset_map = {
            "CELEBA": "celeba",
            "VGGFACE2": "vggface2",
            "DIGIFACE": "digiface"
        }

        if self.dataset_name not in dataset_map:
            raise ValueError("Unsupported dataset name.")

        dataset_path = os.path.join(data_root, dataset_map[self.dataset_name])

        if not os.path.exists(dataset_path):
            raise RuntimeError(f"Dataset folder not found: {dataset_path}")

        return dataset_path

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
            raise ValueError("Unsupported model name.")

    # ----------------------------------------------------------

    def get_loader(self):

        dataset = FaceDataset(
            root_dir=self.dataset_path,
            transform=self.transform
        )

        dataset.samples = dataset.samples[:50] # Limit to 50 images for testing

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

        return dataloader, dataset
