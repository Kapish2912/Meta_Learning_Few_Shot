import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """
    ResNet18-based binary classifier
    CPU-friendly and fast
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)