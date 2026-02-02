import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset_loader import PH2Dataset
from src.baseline_cnn import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Paths
# --------------------------------------------------
PH2_ROOT = "data/PH2"
PH2_LABELS = "data/PH2_dataset.xlsx"

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
# Dataset
# --------------------------------------------------
dataset = PH2Dataset(
    ph2_root=PH2_ROOT,
    labels_path=PH2_LABELS,
    transform=transform,
)

labels = [dataset[i][1] for i in range(len(dataset))]

train_idx, test_idx = train_test_split(
    np.arange(len(labels)),
    test_size=0.2,
    stratify=labels,
    random_state=42,
)

train_data = torch.utils.data.Subset(dataset, train_idx)
test_data = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = SimpleCNN(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------------------------------
# Training
# --------------------------------------------------
EPOCHS = 6
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
# Save model
# --------------------------------------------------
torch.save(model.state_dict(), "outputs/ph2_model_weights.pth")

plt.plot(train_acc, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/ph2_accuracy_plot.png")
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

print("\n=== PH2 Classification Report ===")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"],
    )
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.title("PH2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/ph2_confusion_matrix.png")
plt.close()

print("PH2 training and evaluation complete.")