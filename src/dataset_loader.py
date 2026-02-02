import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------
# HAM10000 DATASET (Binary)
# --------------------------------------------------

BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]

LABEL_MAP = {cls: 0 for cls in BENIGN_CLASSES}
LABEL_MAP.update({cls: 1 for cls in MALIGNANT_CLASSES})

CLASS_NAMES = ["Benign", "Malignant"]


class HAMDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image_id"]
        label_str = row["dx"]

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = LABEL_MAP[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


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