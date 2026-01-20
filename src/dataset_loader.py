import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset

class HAMDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        image_path = os.path.join(self.img_dir, img_name + ".jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label