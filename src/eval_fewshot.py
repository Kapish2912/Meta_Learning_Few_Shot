import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from src.protonet_encoder import ProtoEncoder
from src.dataset_loader import load_fewshot_batch
from src.fewshot_utils import sample_episode


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv("data/HAM10000_metadata.csv")
    meta = meta[meta["dx"].isin(["nv", "bkl", "mel", "bcc", "akiec"])]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = ProtoEncoder(128).to(device)
    model.load_state_dict(
        torch.load("outputs/ham_fewshot_model.pth", map_location=device)
    )
    model.eval()

    accs = []

    with torch.no_grad():
        for _ in tqdm(range(200), desc="Few-Shot Evaluation"):
            support, query = sample_episode(meta)

            s_img, s_lbl = load_fewshot_batch(
                support, "data/images", transform, device
            )
            q_img, q_lbl = load_fewshot_batch(
                query, "data/images", transform, device
            )

            emb_s = model(s_img)
            emb_q = model(q_img)

            proto0 = emb_s[s_lbl == 0].mean(0)
            proto1 = emb_s[s_lbl == 1].mean(0)
            prototypes = torch.stack([proto0, proto1])

            sims = torch.mm(emb_q, prototypes.T)
            preds = sims.argmax(dim=1)

            accs.append((preds == q_lbl).float().mean().item())

    accs = np.array(accs)
    print("\n✅ FINAL FEW-SHOT RESULTS")
    print(f"5-shot Accuracy: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%")


if __name__ == "__main__":
    main()