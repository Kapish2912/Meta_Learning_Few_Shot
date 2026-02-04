# src/dataset_loader.py

# src/dataset_loader.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = ["benign", "malignant"]


class HAMDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row["image_id"]
        dx = row["dx"]

        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")

        # ✅ Binary labels
        label = 0 if dx in ["nv", "bkl"] else 1

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# --------------------------------------------------
# PH2 DATASET (Binary, ROBUST)
# --------------------------------------------------

class PH2Dataset(Dataset):
    """
    PH2 Binary Dataset
    Benign: Any type of nevus
    Malignant: Melanoma
    """

    def __init__(self, ph2_root, labels_path, transform=None):
        self.ph2_root = ph2_root
        self.transform = transform

        # Load Excel without headers
        df = pd.read_excel(labels_path, header=None)

        records = []

        for _, row in df.iterrows():
            row_str = row.astype(str)

            # Detect image ID (IMDxxx)
            image_cells = row_str[row_str.str.startswith("IMD")]
            if len(image_cells) == 0:
                continue

            img_id = image_cells.iloc[0].strip()

            # Detect diagnosis
            diagnosis_cells = row_str[row_str.str.contains(
                "nevus|melanoma", case=False, na=False
            )]
            if len(diagnosis_cells) == 0:
                continue

            diagnosis = diagnosis_cells.iloc[0].strip().lower()
            records.append((img_id, diagnosis))

        self.labels = pd.DataFrame(
            records, columns=["image", "diagnosis"]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        img_id = row["image"]
        diagnosis = row["diagnosis"]

        img_path = os.path.join(
            self.ph2_root,
            img_id,
            f"{img_id}_Dermoscopic_Image",
            f"{img_id}.bmp"
        )

        image = Image.open(img_path).convert("RGB")

        if "melanoma" in diagnosis:
            label = 1
        elif "nevus" in diagnosis:
            label = 0
        else:
            raise ValueError(f"Unknown diagnosis: {diagnosis}")

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def map_binary_label(dx):
        if dx in ["mel", "bcc", "akiec"]:
            return 1  # Malignant
        else:
            return 0  # Benign
        
import os
import torch
from PIL import Image


def load_fewshot_batch(df, image_dir, transform, device):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_id"] + ".jpg")
        img = Image.open(img_path).convert("RGB")

        if transform:
            img = transform(img)

        # ✅ FIXED GLOBAL LABELS
        if row["dx"] in ["nv", "bkl"]:
            label = 0  # benign
        else:
            label = 1  # malignant

        images.append(img)
        labels.append(label)

    return (
        torch.stack(images).to(device),
        torch.tensor(labels, dtype=torch.long).to(device),
    )