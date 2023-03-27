import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10
from matplotlib import pyplot as plt
from torch.distributions import Normal, OneHotCategorical
from dists import OneHotCategoricalStraightThru, ProcessedDist
from rssm import ModernDecoder, LinearEncoder, Decoder, Embedder
from torchvision.transforms.functional import normalize
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import wandb
from argparse import ArgumentParser
from torchvision.utils import make_grid


def prepro(obs):
    return obs


class Autoencoder(nn.Module):
    def __init__(self, channels, cnn_multi=32, mlp_hidden=512, h_size=512, decoder=None):
        super().__init__()
        self.embedder = Embedder(cnn_multi=cnn_multi, in_channels=channels)
        self.encoder = LinearEncoder(cnn_multi, mlp_hidden, h_size)
        if decoder is None:
            self.decoder = Decoder(out_channels=channels, cnn_multi=cnn_multi, mlp_hidden=mlp_hidden, h_size=h_size, )
        else:
            self.decoder = decoder

    def forward(self, x, h):
        e = self.embedder(x)
        z_logits = self.encoder(h, e)
        z_dist = OneHotCategoricalStraightThru(logits=z_logits)
        z = z_dist.sample()
        x_dist = self.decoder(h, z)
        return x_dist, z, z_dist


class PreproAutoEncoder(nn.Module):
    def __init__(self, autoencoder, prepro, postpro):
        super().__init__()
        self.autoencoder = autoencoder
        self.prepro = prepro
        self.postpro = postpro

    def forward(self, x, h):
        x = self.prepro(x)
        x_dist, z, z_dist = self.autoencoder(x, h)
        return ProcessedDist(x_dist, self.prepro, self.postpro), z, z_dist


class EMANormalize:
    def __init__(self):
        self.mean = None
        self.std = None
        self.target_samples = 10000
        self.samples = 0

    def __call__(self, x):
        self.samples += 1
        if self.samples < self.target_samples:
            mean = x.mean((0, -2, -1))
            std = x.std((0, -2, -1))
            coeff = 1 / self.samples
            if self.mean is None:
                self.mean = mean
                self.std = std
            self.means = self.mean * (1 - coeff) + coeff * mean
            self.std = self.std * (1 - coeff) + coeff * std
        return (x - self.means[None, :, None, None]) / self.std[None, :, None, None]

    def invert(self, x):
        return (x * self.std[None, :, None, None].to(x.device)) + self.mean[None, :, None, None].to(x.device)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        super().__init__()
        with Path(path).open('rb') as f:
            self.buff = pickle.load(f)
        if transforms is None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

    def __getitem__(self, item):
        img = Image.fromarray(self.buff[item].observation)
        return self.transforms(img), np.array([0])

    def __len__(self):
        return len(self.buff)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='runs/dreamerv3-ALE-Boxing-v5/run_567/buff.pk')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--h_size', type=int, default=1024)
    parser.add_argument('--normalizer', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--modern', action='store_true')
    args = parser.parse_args()

    wandb.init(project='dreamerv3-autoencoder')
    wandb.config.update(args)

    dataset = ImageDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Autoencoder(3, h_size=args.h_size).cuda(args.device)
    normalizer = EMANormalize()
    model = PreproAutoEncoder(model, normalizer.__call__, normalizer.invert)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # visualize
    plt.ion()
    fig, ax = plt.subplots(ncols=5)
    gt_ax = ax[0].imshow(torch.zeros(64, 64, 3))
    sampled_ax = ax[1].imshow(torch.zeros(64, 64, 3))
    mean_ax = ax[2].imshow(torch.zeros(64, 64, 3))
    z_sample_ax = ax[3].imshow(torch.zeros(32, 32))
    z_dist_ax = ax[4].imshow(torch.zeros(32, 32))
    plt.show()

    steps = 0

    for epoch in range(1000):
        for img, label in dataloader:
            img = img.cuda(args.device)
            h = torch.zeros(img.size(0), args.h_size).cuda(args.device)

            dist, z, z_dist = model(img, h)
            out_img = dist.mean
            loss = - dist.log_prob(img).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss.item()}, step=steps)


            def inv_prepro(obs):
                if normalizer is not None:
                    obs = normalizer.invert(obs)
                return (obs.permute(1, 2, 0) * 255).to(dtype=torch.uint8).detach().cpu().numpy()


            if steps % 200 == 0:
                gt_ax.set_data(img[0].detach().cpu().permute(1, 2, 0))
                sampled_ax.set_data(out_img[0].detach().cpu().permute(1, 2, 0))
                mean_ax.set_data(dist.mean[0].detach().cpu().permute(1, 2, 0))
                z_sample_ax.set_data(z[0].detach().cpu())
                z_dist_ax.set_data(z_dist.probs[0].detach().cpu())
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)

            if steps % args.log_every_n_steps == 0:
                wandb.log({
                    'batch_inputs': wandb.Image(make_grid(img)),
                    'batch_outputs': wandb.Image(make_grid(out_img))
                }, step=steps)

            steps += 1
