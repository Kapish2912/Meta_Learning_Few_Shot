import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset_loader import HAMDataset, CLASS_NAMES
from src.baseline_cnn import SimpleCNN


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# Load & FILTER HAM10000 metadata
# --------------------------------------------------
meta = pd.read_csv("data/HAM10000_metadata.csv")

# ✅ KEEP ONLY THESE CLASSES
# Benign: nv, bkl
# Malignant: mel, bcc, akiec
keep_classes = ["nv", "bkl", "mel", "bcc", "akiec"]
meta = meta[meta["dx"].isin(keep_classes)].reset_index(drop=True)

print("Class distribution after filtering:")
print(meta["dx"].value_counts())


# --------------------------------------------------
# Train / Test split (stratified)
# --------------------------------------------------
train_df, test_df = train_test_split(
    meta,
    test_size=0.2,
    stratify=meta["dx"],
    random_state=42,
)


# --------------------------------------------------
# Transforms
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Datasets & Loaders
# --------------------------------------------------
train_data = HAMDataset(train_df, "data/images", transform)
test_data = HAMDataset(test_df, "data/images", transform)

train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_data, batch_size=16, shuffle=False, num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------
model = SimpleCNN(num_classes=2).to(device)

# ✅ Class-weighted loss (important)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 2.5]).to(device)
)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


# --------------------------------------------------
# Training
# --------------------------------------------------
EPOCHS = 5
train_acc = []

for epoch in range(EPOCHS):
    print(f"\n🚀 Starting Epoch {epoch+1}/{EPOCHS}")
    model.train()

    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    train_acc.append(acc)

    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] Training Accuracy: {acc:.2f}%")


# --------------------------------------------------
# Save model & accuracy plot
# --------------------------------------------------
torch.save(model.state_dict(), "outputs/ham_filtered_model.pth")

plt.plot(train_acc)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.tight_layout()
plt.savefig("outputs/ham_filtered_accuracy.png")
plt.close()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n=== HAM10000 Filtered Binary Classification Report ===")
print(
    classification_report(
        y_true, y_pred, target_names=CLASS_NAMES
    )
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/ham_filtered_confusion_matrix.png")
plt.close()

print("Filtered HAM10000 training & evaluation complete.")