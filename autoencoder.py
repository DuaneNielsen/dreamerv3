import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
from torch.distributions import Normal, OneHotCategorical
from dists import OneHotCategoricalStraightThru


class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.LayerNorm([out_channels, in_height // 2, in_width // 2]),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


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
    def __init__(self, cnn_multi=32):
        super().__init__()
        self.embedder = nn.Sequential(
            EncoderConvBlock(1, 64, 64, cnn_multi),
            EncoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            EncoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            EncoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            nn.Flatten()
        )

    def forward(self, x):
        return self.embedder(x)


class Encoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.encoder = nn.Sequential(
            MLPBlock(4 * 4 * cnn_multi * 2 ** 3 + h_size, mlp_hidden),
            *[MLPBlock(mlp_hidden, mlp_hidden) for _ in range(mlp_layers - 1)],
            nn.Linear(mlp_hidden, 32 * 32),
            nn.Unflatten(1, (32, 32))
        )

    def forward(self, h, e):
        return self.encoder(torch.cat([h, e], dim=-1))


class Decoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32 * 32 + h_size, mlp_hidden),
            *[MLPBlock(mlp_hidden, mlp_hidden) for _ in range(mlp_layers - 1)],
            MLPBlock(mlp_hidden, 4 * 4 * cnn_multi * 2 ** 3),
            nn.Unflatten(-1, (cnn_multi * 2 ** 3, 4, 4)),
            DecoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            DecoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            DecoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            DecoderConvBlock(1, 64, 64, cnn_multi),
        )

    def forward(self, h, z):
        hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
        x = self.decoder(hz_flat)
        return Normal(loc=x, scale=1.)


class Autoencoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.embedder = Embedder(cnn_multi)
        self.encoder = Encoder(cnn_multi, mlp_layers, mlp_hidden, h_size)
        self.decoder = Decoder(cnn_multi, mlp_layers, mlp_hidden, h_size)

    def forward(self, x, h):
        e = self.embedder(x)
        z_logits = self.encoder(h, e)
        z_dist = OneHotCategoricalStraightThru(logits=z_logits)
        z = z_dist.sample()
        x_dist = self.decoder(h, z)
        return x_dist, z, z_dist


if __name__ == '__main__':

    h_size = 512

    img_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,), inplace=True)
    ])

    dataset = MNIST('/mnt/data/data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = Autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # visualize
    plt.ion()
    fig, ax = plt.subplots(ncols=5)
    gt_ax = ax[0].imshow(torch.randn(64, 64))
    sampled_ax = ax[1].imshow(torch.randn(64, 64))
    mean_ax = ax[2].imshow(torch.randn(64, 64))
    z_sample_ax = ax[3].imshow(torch.randn(32, 32))
    z_dist_ax = ax[4].imshow(torch.randn(32, 32))
    plt.show()

    for epoch in range(1000):
        for img, label in dataloader:
            img = img.cuda()
            h = torch.zeros(img.size(0), h_size).cuda()

            dist, z, z_dist = model(img, h)
            out_img = dist.sample()
            loss = - dist.log_prob(img).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize
            gt_ax.set_data(img[0, 0].detach().cpu())
            sampled_ax.set_data(out_img[0, 0].detach().cpu())
            mean_ax.set_data(dist.mean[0, 0].detach().cpu())
            z_sample_ax.set_data(z[0].detach().cpu())
            z_dist_ax.set_data(z_dist.probs[0].detach().cpu())
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        if epoch % 10 == 0:
            pass
