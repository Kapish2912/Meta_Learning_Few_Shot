import torch
import pandas as pd
from torchvision import transforms
from tqdm import tqdm

from src.protonet_encoder import ProtoEncoder
from src.dataset_loader import load_fewshot_batch
from src.fewshot_utils import prototypical_loss, sample_episode


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv("data/HAM10000_metadata.csv")
    meta = meta[meta["dx"].isin(["nv", "bkl", "mel", "bcc", "akiec"])]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])

    model = ProtoEncoder(embedding_dim=128).to(device)
    model.load_state_dict(
        torch.load("outputs/protonet_binary_fewshot.pth", map_location=device)
    )

    for p in model.encoder["conv1"].parameters():
        p.requires_grad = False
    for p in model.encoder["conv2"].parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    EPISODES = 300

    model.train()
    for _ in tqdm(range(EPISODES), desc="Few-Shot Training"):
        support, query = sample_episode(meta)

        s_img, s_lbl = load_fewshot_batch(
            support, "data/images", transform, device
        )
        q_img, q_lbl = load_fewshot_batch(
            query, "data/images", transform, device
        )

        emb_s = model(s_img)
        emb_q = model(q_img)

        loss = prototypical_loss(
            torch.cat([emb_s, emb_q]),
            torch.cat([s_lbl, q_lbl]),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "outputs/ham_fewshot_model.pth")


if __name__ == "__main__":
    main()