import torch
import torch.nn as nn
from torch.optim import Adam
from replay import simple_trajectory, ReplayBuffer
from env import Env
from matplotlib import pyplot as plt
from torch.distributions import OneHotCategorical
from utils import sample_one_hot, sample_one_hot_log
from tqdm import tqdm
from collections import deque


if __name__ == '__main__':

    XD = Env.state_size
    XC = Env.state_classes
    AD = Env.action_size
    AC = Env.action_classes
    ZD = 1  # latent_size
    ZC = 10  # latent classes

    T = 12  # Time Horizon
    N = 128  # batch_size
    H = 32  # Gru recurrent units - a.k.a. "hidden size"
    num_layers = 3
    device = "cpu"

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(H + XD * XC, ZD * ZC)

        def forward(self, h, x):
            hx_flat = torch.cat([h, x.flatten(-2)], dim=1)
            return self.encoder(hx_flat).unflatten(-1, (ZD, ZC))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(H + ZD * ZC, XD * XC)

        def forward(self, h, z):
            hz = torch.cat([h, z.flatten(-2)], dim=2)
            x_decoded_flat = self.decoder(hz)
            x_decoded = x_decoded_flat.unflatten(-1, (XD, XC))
            return OneHotCategorical(logits=x_decoded)

    class RSSM(nn.Module):

        def __init__(self):
            super().__init__()
            # encoder: z ~ q ( z | h, x )
            self.encoder = Encoder()

            # sequence model: h = f ( h, z, a )
            self.gru = nn.GRUCell(ZD * ZC + AD * AC, H)

            # decoder: x ~ p ( x | h, z )
            self.decoder = Decoder()

            # dynamics predictor: z ~ p ( z | h )
            self.dynamics = nn.Linear(H, ZD * ZC)

            # reward predictor: r ~ p ( r | h, z )
            # continue predictor: c ~ p( c | h, z )

        def forward(self, x, h0):
            """

            :param x: [T, N, XD, XC] state tensor
            :param h0: [N, H] start state tensor
            :return: x_decoded: prediction of next state
            """
            # recurrent encoding
            h_list = [h0]
            embed_list = [self.encoder(h_list[0], x[0])]
            z_list = [sample_one_hot(embed_list[0])]

            for i in range(0, x.size(0) - 1):
                za_flat = torch.cat([z_list[i].flatten(-2), a[i].flatten(-2)], dim=-1)
                h_list += [self.gru(za_flat, h_list[i])]
                embed_list += [self.encoder(h_list[i], x[i])]
                z_list += [sample_one_hot(embed_list[i])]

            h = torch.stack(h_list)
            embed = torch.stack(embed_list)
            z = torch.stack(z_list)

            # decode
            x_dist = self.decoder(h, embed)

            # predict
            z_pred = self.dynamics(h)

            return x_dist, z, h


    rssm = RSSM()

    # optimizer
    opt = Adam(rssm.parameters(), lr=1e-3)

    # dataset
    buffer = ReplayBuffer()
    for i in range(100):
        buffer += simple_trajectory([Env.right] * 8)
        buffer += simple_trajectory([Env.left])

    # visualize
    plt.ion()
    fig, ax = plt.subplots(6)
    scatter = ax[0].scatter([0, 10], [0, 10])
    loss_hist = deque(maxlen=200)

    for batch in tqdm(range(5000)):
        x, a, r, c, next_x, mask = buffer.sample_batch(T, N)
        h0 = torch.zeros(N, H)
        x_dist, z, h = rssm(x, h0)
        loss = - x_dist.log_prob(x) * mask.unsqueeze(-1)
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # visualize
        loss_hist.append(loss.item())

        if batch % 10 == 0:
            x_argmax, x_dist_argmax = x.argmax(-1).flatten(), x_dist.probs.argmax(-1).flatten()
            [ax[i].cla() for i in range(6)]
            ax[0].scatter(x_argmax[mask.flatten()], x_dist_argmax[mask.flatten()])
            ax[1].hist(x_argmax[mask.flatten()], bins=8)
            ax[2].hist(x_dist_argmax[mask.flatten()], bins=8)
            ax[3].hist(z.argmax(-1).flatten()[mask.flatten()], bins=10)
            ax[4].hist(h.detach().flatten(), bins=10)
            ax[5].plot(loss_hist)
            fig.canvas.draw()
            fig.canvas.flush_events()
