import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.proto_network import ProtoNet, euclidean_dist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# CONFIGURATION (BINARY FEW-SHOT)
# ---------------------------
n_way = 2          # Binary
k_shot = 5
q_query = 10
n_episodes = 300
lr = 1e-3

# ---------------------------
# BINARY LABEL MAP
# ---------------------------
def map_binary_label(dx):
    if dx in ["mel", "bcc", "akiec"]:
        return 1  # malignant
    else:
        return 0  # benign

# ---------------------------
# LOAD METADATA
# ---------------------------
meta = pd.read_csv("data/HAM10000_metadata.csv")

benign_df = meta[~meta["dx"].isin(["mel", "bcc", "akiec"])].reset_index(drop=True)
malignant_df = meta[meta["dx"].isin(["mel", "bcc", "akiec"])].reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# MODEL
# ---------------------------
model = ProtoNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_history = []
acc_history = []

print("\n[INFO] Starting Binary ProtoNet Training...\n")

# ---------------------------
# TRAINING LOOP
# ---------------------------
for episode in range(1, n_episodes + 1):

    # ---------------------------
    # SAMPLE BINARY EPISODE (SAFE)
    # ---------------------------
    support = []
    query = []

    benign_support = benign_df.sample(k_shot)
    malignant_support = malignant_df.sample(k_shot)

    benign_query = benign_df.sample(q_query)
    malignant_query = malignant_df.sample(q_query)

    for _, r in benign_support.iterrows():
        support.append((r["image_id"], r["dx"]))
    for _, r in malignant_support.iterrows():
        support.append((r["image_id"], r["dx"]))

    for _, r in benign_query.iterrows():
        query.append((r["image_id"], r["dx"]))
    for _, r in malignant_query.iterrows():
        query.append((r["image_id"], r["dx"]))

    random.shuffle(support)
    random.shuffle(query)

    # ---------------------------
    # LOAD IMAGES
    # ---------------------------
    def load_batch(samples):
        imgs, labels = [], []
        for img_id, dx in samples:
            img_path = os.path.join("data/images", img_id + ".jpg")
            img = Image.open(img_path).convert("RGB")
            imgs.append(transform(img))
            labels.append(map_binary_label(dx))
        return torch.stack(imgs), torch.tensor(labels)

    support_imgs, support_labels = load_batch(support)
    query_imgs, query_labels = load_batch(query)

    support_imgs = support_imgs.to(device)
    query_imgs = query_imgs.to(device)
    support_labels = support_labels.to(device)
    query_labels = query_labels.to(device)

    # ---------------------------
    # ENCODE
    # ---------------------------
    support_embeddings = model(support_imgs)
    query_embeddings = model(query_imgs)

    # ---------------------------
    # COMPUTE PROTOTYPES (SAFE)
    # ---------------------------
    prototypes = []
    skip_episode = False

    for c in range(n_way):
        class_embeddings = support_embeddings[support_labels == c]

        if class_embeddings.size(0) == 0:
            skip_episode = True
            break

        prototypes.append(class_embeddings.mean(0))

    if skip_episode:
        continue   # ✅ SAFE: INSIDE EPISODE LOOP

    prototypes = torch.stack(prototypes)

    # ---------------------------
    # LOSS + ACCURACY
    # ---------------------------
    dist = euclidean_dist(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dist, dim=1)

    loss = F.nll_loss(log_p_y, query_labels)

    if torch.isnan(loss):
        continue

    preds = log_p_y.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()

    loss_history.append(loss.item())
    acc_history.append(acc)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 20 == 0:
        print(
            f"[Episode {episode}/{n_episodes}] "
            f"Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%"
        )

# ---------------------------
# SAVE MODEL
# ---------------------------
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/protonet_binary_fewshot.pth")

# ---------------------------
# PLOTS
# ---------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, color="red")
plt.title("Loss per Episode")

plt.subplot(1, 2, 2)
plt.plot(acc_history, color="green")
plt.title("Accuracy per Episode")

plt.tight_layout()
plt.savefig("outputs/protonet_binary_training_curves.png")
plt.show()

print("\n✅ Binary ProtoNet training complete.\n")