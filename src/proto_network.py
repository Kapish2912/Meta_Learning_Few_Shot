import torch
import torch.nn as nn
import torch.nn.functional as F
from src.baseline_cnn import SimpleCNN

# ---------------------------------
# Simple CNN encoder for ProtoNet
# ---------------------------------
class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super(SimpleCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x  # 128-dimensional embedding vector


# ---------------------------------
# Prototypical Network main class
# ---------------------------------
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.encoder = SimpleCNNEncoder()

    def forward(self, x):
        embeddings = self.encoder(x)
        return embeddings


# ---------------------------------
# Distance function
# ---------------------------------
def euclidean_dist(x, y):
    """
    Compute pairwise Euclidean distances between x and y.
    Args:
        x: [n, d]
        y: [m, d]
    Returns:
        dist: [n, m]
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist