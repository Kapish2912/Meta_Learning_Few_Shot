import torch
import torch.nn.functional as F
import pandas as pd


def prototypical_loss(embeddings, labels):
    classes = torch.unique(labels)
    prototypes = []

    for c in classes:
        prototypes.append(embeddings[labels == c].mean(0))

    prototypes = torch.stack(prototypes)

    logits = torch.mm(embeddings, prototypes.T)

    targets = torch.tensor(
        [torch.where(classes == y)[0][0] for y in labels],
        device=labels.device,
    )

    return F.cross_entropy(logits, targets)


def sample_episode(df, n_way=2, k_shot=7, q_query=15):
    benign_df = df[df["dx"].isin(["nv", "bkl"])]
    malignant_df = df[df["dx"].isin(["mel", "bcc", "akiec"])]

    benign_samples = benign_df.sample(k_shot + q_query)
    malignant_samples = malignant_df.sample(k_shot + q_query)

    support = pd.concat([
        benign_samples.iloc[:k_shot],
        malignant_samples.iloc[:k_shot],
    ])

    query = pd.concat([
        benign_samples.iloc[k_shot:],
        malignant_samples.iloc[k_shot:],
    ])

    return support, query