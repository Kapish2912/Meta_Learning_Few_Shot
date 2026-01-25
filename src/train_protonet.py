import os, random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.proto_network import ProtoNet, euclidean_dist
from src.fewshot_sampler import FewShotDataset
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# CONFIGURATION
# ---------------------------
n_way = 4         # classes per episode
k_shot = 5        # support samples per class
q_query = 5       # query samples per class
n_episodes = 200  # total episodes
lr = 1e-3

# ---------------------------
# DATASET
# ---------------------------
meta = pd.read_csv("data/HAM10000_metadata.csv")
selected = ['mel', 'nv', 'bkl', 'df']
meta = meta[meta['dx'].isin(selected)]
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

dataset = FewShotDataset(meta, "data/images", n_way, k_shot, q_query, transform)

# ---------------------------
# MODEL + OPTIMIZER
# ---------------------------
model = ProtoNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
acc_history, loss_history = [], []

print("\n[INFO] Starting Prototypical Network training...\n")

# ---------------------------
# TRAINING LOOP
# ---------------------------
for episode in range(1, n_episodes + 1):
    support, query = dataset.create_episode()

    # Load images and labels
    def load_data(samples):
        imgs, labels = [], []
        for img_id, cls in samples:
            path = os.path.join("data/images", img_id + ".jpg")
            img = Image.open(path).convert("RGB")
            imgs.append(transform(img))
            labels.append(cls)
        return torch.stack(imgs), labels

    support_imgs, support_labels = load_data(support)
    query_imgs, query_labels = load_data(query)

    support_imgs, query_imgs = support_imgs.to(device), query_imgs.to(device)

    # Encode
    support_embeddings = model(support_imgs)
    query_embeddings = model(query_imgs)

    # Convert label strings to numeric ids
    classes = sorted(list(set(support_labels)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    support_y = torch.tensor([class_to_idx[c] for c in support_labels]).to(device)
    query_y = torch.tensor([class_to_idx[c] for c in query_labels]).to(device)

    # Compute class prototypes
    prototypes = []
    for c in range(n_way):
        prototypes.append(support_embeddings[support_y == c].mean(0))
    prototypes = torch.stack(prototypes)

    # Compute distances and loss
    dist = euclidean_dist(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dist, dim=1)
    loss = F.nll_loss(log_p_y, query_y)

    # Accuracy
    y_hat = log_p_y.argmax(1)
    acc = (y_hat == query_y).float().mean().item()
    acc_history.append(acc)
    loss_history.append(loss.item())

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 20 == 0:
        print(f"[Episode {episode}/{n_episodes}] Loss: {loss.item():.4f} Acc: {acc*100:.2f}%")

# ---------------------------
# SAVE MODEL
# ---------------------------
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/protonet_fewshot.pth")
print("\nModel saved as outputs/protonet_fewshot.pth\n")

# ---------------------------
# PLOTS
# ---------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history, color='red')
plt.title("Loss per Episode")
plt.xlabel("Episodes"); plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(acc_history, color='green')
plt.title("Accuracy per Episode")
plt.xlabel("Episodes"); plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("outputs/protonet_training_curves.png")
plt.show()
print("\nTraining curves saved at outputs/protonet_training_curves.png\n")