import os, torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from src.proto_network import ProtoNet, euclidean_dist
from src.fewshot_sampler import FewShotDataset
from src.report_generator import generate_metrics_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CONFIG
# ----------------------------
n_way = 4
k_shot = 5
q_query = 5
test_episodes = 100
save_dir = "outputs/fewshot_results"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# LOAD DATASET & MODEL
# ----------------------------
meta = pd.read_csv("data/HAM10000_metadata.csv")
selected = ['mel', 'nv', 'bkl', 'df']
meta = meta[meta['dx'].isin(selected)]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = FewShotDataset(meta, "data/images", n_way, k_shot, q_query, transform)

model = ProtoNet().to(device)
model.load_state_dict(torch.load("outputs/protonet_fewshot.pth", map_location=device))
model.eval()

print("\n[INFO] Evaluating ProtoNet on new few‑shot tasks...\n")

accuracies = []
all_true, all_pred = [], []

for ep in range(test_episodes):
    support, query = dataset.create_episode()

    # Load episode data
    def load_data(list_):
        imgs, labels = [], []
        for img_id, cls in list_:
            path = os.path.join("data/images", img_id + ".jpg")
            imgs.append(transform(Image.open(path).convert("RGB")))
            labels.append(cls)
        return torch.stack(imgs), labels

    support_imgs, support_labels = load_data(support)
    query_imgs, query_labels = load_data(query)

    with torch.no_grad():
        support_emb = model(support_imgs.to(device))
        query_emb = model(query_imgs.to(device))

    classes = sorted(list(set(support_labels)))
    mapping = {c: i for i, c in enumerate(classes)}
    support_y = torch.tensor([mapping[c] for c in support_labels]).to(device)
    query_y = torch.tensor([mapping[c] for c in query_labels]).to(device)

    prototypes = torch.stack([support_emb[support_y == c].mean(0) for c in range(n_way)])
    dist = euclidean_dist(query_emb, prototypes)
    log_p_y = F.log_softmax(-dist, dim=1)
    preds = log_p_y.argmax(1)
    acc = (preds == query_y).float().mean().item()
    accuracies.append(acc)

    # Store for overall report
    for i in range(len(query_y)):
        all_true.append(classes[query_y[i].item()])
        all_pred.append(classes[preds[i].item()])

# ----------------------------
# OVERALL PERFORMANCE
# ----------------------------
avg_acc = np.mean(accuracies) * 100
std_acc = np.std(accuracies) * 100
print(f"Average Few‑Shot Accuracy over {test_episodes} episodes: {avg_acc:.2f}% ± {std_acc:.2f}%")

# ----------------------------
# GENERATE METRICS REPORT
# ----------------------------
# Dynamically include only labels actually present in evaluation
unique_labels = sorted(list(set(all_true + all_pred)))

generate_metrics_report(
    y_true=all_true,
    y_pred=all_pred,
    labels=unique_labels,
    model_name="ProtoNet Few‑Shot",
    save_dir=save_dir
)