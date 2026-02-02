import random, os
from torch.utils.data import Dataset

# Your sampler used by the Prototypical Network
class FewShotDataset:
    def __init__(self, meta, img_dir, n_way, k_shot, q_query, transform):
        self.meta = meta
        self.img_dir = img_dir
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform = transform

        self.benign = meta[~meta["dx"].isin(["mel", "bcc", "akiec"])]
        self.malignant = meta[meta["dx"].isin(["mel", "bcc", "akiec"])]

    def create_episode(self):
        support, query = [], []

        benign_support = self.benign.sample(self.k_shot)
        malignant_support = self.malignant.sample(self.k_shot)

        benign_query = self.benign.sample(self.q_query)
        malignant_query = self.malignant.sample(self.q_query)

        for _, row in benign_support.iterrows():
            support.append((row["image_id"], row["dx"]))
        for _, row in malignant_support.iterrows():
            support.append((row["image_id"], row["dx"]))

        for _, row in benign_query.iterrows():
            query.append((row["image_id"], row["dx"]))
        for _, row in malignant_query.iterrows():
            query.append((row["image_id"], row["dx"]))

        random.shuffle(support)
        random.shuffle(query)

        return support, query

    def __len__(self):
        # dummy value; usual training uses create_episode() directly
        return 1000