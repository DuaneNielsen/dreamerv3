from torch import nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1.)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.SiLU()
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)


class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, bias=False, padding=1, padding_mode='replicate'),
            nn.LayerNorm([out_channels, in_height // 2, in_width // 2]),
            nn.SiLU(inplace=True))
        self.apply(init_weights)

    def forward(self, x):
        return self.block(x)


class ModernDecoderConvBlock(nn.Module):
    def __init__(self, out_channels, out_h, out_w, in_channels, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=bias),
            nn.LayerNorm([out_channels, out_h, out_w]),
            nn.SiLU(inplace=True))

        self.apply(init_weights)

    def forward(self, x):
        return self.block(x)


class DecoderConvBlock(nn.Module):
    def __init__(self, out_channels, out_h, out_w, in_channels, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.LayerNorm([out_channels, out_h, out_w]),
            nn.SiLU(inplace=True))
        self.apply(init_weights)

    def forward(self, x):
        return self.block(x)


