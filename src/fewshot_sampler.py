import random, os
from torch.utils.data import Dataset

# Your sampler used by the Prototypical Network
class FewShotDataset(Dataset):
    """
    Creates N‑way K‑shot episodes from a dataframe (image_id, dx)
    """
    def __init__(self, df, img_dir,
                 n_way=4, k_shot=5, q_query=5, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.classes = list(df['dx'].unique())
        self.class_to_images = {
            c: df[df['dx'] == c]['image_id'].tolist() for c in self.classes
        }

    def create_episode(self):
        """Return support and query lists for one episode"""
        chosen_classes = random.sample(self.classes, self.n_way)
        support, query = [], []
        for c in chosen_classes:
            imgs = random.sample(self.class_to_images[c],
                                 self.k_shot + self.q_query)
            support += [(i, c) for i in imgs[:self.k_shot]]
            query  += [(i, c) for i in imgs[self.k_shot:]]
        return support, query

    def __len__(self):
        # dummy value; usual training uses create_episode() directly
        return 1000