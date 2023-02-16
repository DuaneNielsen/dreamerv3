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


class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.LayerNorm([out_channels, in_height // 2, in_width // 2]),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ModernDecoderConvBlock(nn.Module):
    def __init__(self, out_channels, out_h, out_w, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LayerNorm([out_channels, out_h, out_w]),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DecoderConvBlock(nn.Module):
    def __init__(self, out_channels, out_h, out_w, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([out_channels, out_h, out_w]),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Embedder(nn.Module):
    def __init__(self, in_channels=3, cnn_multi=32):
        super().__init__()
        self.embedder = nn.Sequential(
            EncoderConvBlock(in_channels, 64, 64, cnn_multi),
            EncoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            EncoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            EncoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            nn.Flatten()
        )

    def forward(self, x):
        """
        param: x: [T, N, C, H, W] observation in standard form
        """
        T, N, C, H, W = x.shape
        return self.embedder(x.flatten(start_dim=0, end_dim=1)).unflatten(0, (T, N))
