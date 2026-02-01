import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from src.dataset_loader import HAMDataset
from src.baseline_cnn import SimpleCNN
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
meta = pd.read_csv("data/HAM10000_metadata.csv")
selected_classes = ['mel', 'nv', 'bkl', 'df']
meta = meta[meta['dx'].isin(selected_classes)]
le = LabelEncoder()
meta['label'] = le.fit_transform(meta['dx'])

train_df, test_df = train_test_split(meta[['image_id', 'label']], test_size=0.2, stratify=meta['label'], random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = HAMDataset(train_df, "data/images", transform)
test_data = HAMDataset(test_df, "data/images", transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = SimpleCNN(num_classes=len(selected_classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc = []
val_acc = []

for epoch in range(5):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    train_acc.append(accuracy)
    print(f"Epoch [{epoch+1}/5], Training Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "outputs/model_weights.pth")

plt.plot(train_acc, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("outputs/accuracy_plot.png")
print("Training complete. Graph saved to outputs/accuracy_plot.png")


# EVALUATION
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 1. Classification Report
print("\n=== Classification Report (Baseline CNN) ===")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# 2. Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues",
            fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Baseline CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# 3. Visualize sample predictions
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(12,6))
for i in range(6):
    img = images[i].cpu().permute(1,2,0).numpy()
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Real: {le.inverse_transform([labels[i].cpu()])[0]} \nPred: {le.inverse_transform([preds[i].cpu()])[0]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/week_b_analysis/confusion_matrix_v1.png")
plt.close()

print("Evaluation complete. Results saved in outputs folder.")