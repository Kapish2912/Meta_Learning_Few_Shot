import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset_loader import HAMDataset, CLASS_NAMES
from src.baseline_cnn import SimpleCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    meta = pd.read_csv("data/HAM10000_metadata.csv")
    keep_classes = ["nv", "bkl", "mel", "bcc", "akiec"]
    meta = meta[meta["dx"].isin(keep_classes)].reset_index(drop=True)

    train_df, test_df = train_test_split(
        meta,
        test_size=0.2,
        stratify=meta["dx"],
        random_state=42,
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_data = HAMDataset(train_df, "data/images", transform)
    test_data = HAMDataset(test_df, "data/images", transform)

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0,      # ✅ Windows safe
        pin_memory=False,   # ✅ CPU safe
    )

    test_loader = DataLoader(
        test_data,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ✅ Sanity check
    images, labels = next(iter(train_loader))
    print("✅ Unique labels:", labels.unique())

    model = SimpleCNN(num_classes=2).to(device)

    train_df["binary"] = train_df["dx"].apply(
        lambda x: 0 if x in ["nv", "bkl"] else 1
    )

    counts = train_df["binary"].value_counts().sort_index()
    weights = torch.tensor((1.0 / counts).values, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        print(f"\n🚀 Starting Epoch {epoch+1}/{EPOCHS}")
        model.train()

        correct, total = 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            progress.set_postfix(
                acc=f"{100*correct/total:.2f}%",
                loss=f"{loss.item():.4f}",
            )

    print("✅ Training complete")

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()