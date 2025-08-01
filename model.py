# model.py
import torch.nn as nn

class FDTRegressor(nn.Module):
    def __init__(self, in_features: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)