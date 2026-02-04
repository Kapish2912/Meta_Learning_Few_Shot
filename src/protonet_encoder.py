import torch.nn as nn
import torch.nn.functional as F


class ProtoEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.encoder = nn.ModuleDict({
            "conv1": nn.Conv2d(3, 16, kernel_size=3, padding=1),
            "conv2": nn.Conv2d(16, 32, kernel_size=3, padding=1),
            "fc1": nn.Linear(32 * 56 * 56, embedding_dim),
        })

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.encoder["conv1"](x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.encoder["conv2"](x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.encoder["fc1"](x)
        return F.normalize(x, dim=1)