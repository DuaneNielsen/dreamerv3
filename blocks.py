from torch import nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.SiLU()
        )

    def forward(self, x):
        return self.mlp(x)
