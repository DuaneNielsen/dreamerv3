"""
 RSSM equations

 Sequence Model:       h = f( h, z, a )
 Encoder:              z ~ q( z | h, x)
 Dynamics predictor:   z ~ p( z | h )
 Reward   predictor:   r ~ p( z | h )
 Continue predictor:   c ~ p( z | h )
 Decoder:              x ~ p( x | h, z )


 lets refactor to ...

 Sequence Model:       h = f( h, z, a )
 Embedder:             e = q( x )
 Encoder:              zprior ~ q ( zprior | h, e )
 Dynamics predictor    zpost ~ p ( zpost | h )
 Reward predictor:     r ~ p( z | h )
 Continue predictor:   c ~ p( z | h )
 Decoder:              x ~ p( x | h, z )

 During training z = zprior
 During prediction z = zpost

"""

import torch
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.distributions import OneHotCategorical, Normal, Bernoulli
from dists import sample_one_hot, OneHotCategoricalStraightThru
from env import Env
from replay import ReplayBuffer, simple_trajectory
from collections import deque


x_size, x_cls = 1, 10  # input image dims
e_size = 10  # size of flattened embedding
z_size, z_cls = 1, 10  # size of latent space
h_size = 32  # size of hidden space
a_size, a_cls = 1, 10  # action space size and classes (discrete)
batch_size = 64
T = 8


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(h_size + e_size, z_size * z_cls)

    def forward(self, h, e):
        he_flat = torch.cat([h, e], dim=-1)
        z_flat = self.encoder(he_flat)
        return OneHotCategoricalStraightThru(logits=z_flat.unflatten(-1, (z_size, z_cls)))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(h_size + z_size * z_cls, x_size * x_cls)

    def forward(self, h, z):
        """
        Decoder:              x ~ p( x | h, z )
        """
        hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
        x_flat = self.decoder(hz_flat)
        return OneHotCategorical(logits=x_flat.unflatten(-1, (x_size, x_cls)))


class DynamicsPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics_predictor = nn.Linear(h_size, z_size * z_cls)

    def forward(self, h):
        z_flat = self.dynamics_predictor(h)
        return OneHotCategoricalStraightThru(logits=z_flat.unflatten(-1, (z_size, z_cls)))


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.reward_predictor = nn.Linear(h_size, 1, bias=False)

    def forward(self, h):
        return Normal(loc=self.reward_predictor(h), scale=1.)


class ContinuePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.continue_predictor = nn.Linear(h_size, 1)

    def forward(self, h):
        return Bernoulli(logits=self.continue_predictor(h))


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        """
         Sequence Model:       h = f( h, z, a )
         Embedder:             e = q( x )
         Encoder:              zprior ~ q ( zprior | h, e )
         Dynamics predictor    zpost ~ p ( zpost | h )
         Reward predictor:     r ~ p( z | h )
         Continue predictor:   c ~ p( z | h )
         Decoder:              x ~ p( x | h, z )
        
        """

        self.embedder = nn.Flatten(-2)
        self.encoder = Encoder()
        self.seq_model = nn.GRUCell(z_size * z_cls + a_size * a_cls, h_size)
        self.decoder = Decoder()
        self.dynamics_pred = DynamicsPredictor()
        self.reward_pred = RewardPredictor()
        self.continue_pred = ContinuePredictor()

    def forward(self, x, a, h0):
        """

        :param x: [T, N, x_size, x_cls]
        :param a: [T, N, a_size, a_cls]
        :param h0: [N, h0]
        :return: x, h, z

        ^                          ┌────┐
        x0 ◄───────────────────────┤dec │
                                   └────┘
                                    ▲  ▲
            ┌───┐     ┌───┐         │  │   ┌─────────┐
        x0─►│emb├─►e─►│enc├─►zprior─┴──┼──►│Sequence │    x1─►
            └───┘     └───┘            │   │Model    │
                        ▲              │   │         │
        h0──┬───────────┴──────────────┴──►│         ├───►h1─►
            │                              │         │
            │  ┌───┐                       │         │
            ├─►│dyn├─►zpost                │         │
            │  └───┘                       │         │
            │                              │         │
            │  ┌───┐                       │         │
            ├─►│rew├─►reward0              │         │
            │  └───┘                       │         │
            │                              │         │
            │  ┌───┐                       │         │
            └─►│con├─►cont0                │         │
               └───┘                       │         │
                                           │         │
        a0────────────────────────────────►│         │    a1─►
                                           └─────────┘

        """

        h_list = [h0]
        e = self.embedder(x)
        z_list = [self.encoder(h0, e[0]).sample()]

        for t in range(1, x.size(0)):
            za_flat = torch.cat([z_list[t-1].flatten(-2), a[t-1].flatten(-2)], dim=1)
            h_list += [self.seq_model(za_flat, h_list[t - 1])]
            z_list += [self.encoder(h_list[t], e[t]).sample()]

        h = torch.stack(h_list)
        z = torch.stack(z_list)

        x_dist = self.decoder(h, z)
        z_post = self.dynamics_pred(h).sample()
        r_dist = self.reward_pred(h)
        c_dist = self.continue_pred(h)

        return x_dist, r_dist, c_dist, z, z_post


if __name__ == '__main__':

    # visualize
    plt.ion()
    fig, ax = plt.subplots(3, 4)
    loss_buff, loss_x_buff, loss_r_buff, loss_c_buff = \
        deque(maxlen=400), deque(maxlen=400), deque(maxlen=400), deque(maxlen=400)
    plt.show()

    # dataset
    buff = ReplayBuffer()
    buff += simple_trajectory([Env.right] * 8)
    buff += simple_trajectory([Env.left])

    rssm = RSSM()
    opt = Adam(rssm.parameters(), lr=1e-3)

    for batch in range(5000):
        x, a, r, c, next_x, mask = buff.sample_batch(T, batch_size)
        mask = mask.unsqueeze(-1)

        h0 = torch.zeros(batch_size, h_size)
        x_dist, r_dist, c_dist, z_prior, z_post = rssm(x, a, h0)
        loss_x = - x_dist.log_prob(x) * mask
        loss_r = - r_dist.log_prob(r) * mask
        loss_c = - c_dist.log_prob(c) * mask
        loss = (loss_x + loss_r + loss_c).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_buff += [loss.item()]
        loss_x_buff += [loss_x.mean().item()]
        loss_r_buff += [loss_r.mean().item()]
        loss_c_buff += [loss_c.mean().item()]

        with torch.no_grad():
            if batch % 10 == 0:
                for row in ax:
                    for subplot in row:
                        subplot.clear()

                # x predictor
                x, x_est = x.argmax(-1)[mask].flatten().detach().cpu(), x_dist.logits.argmax(-1)[mask].flatten().detach().cpu()
                bins = []
                for i in range(1, 9):
                    bins += [x_est[x == i]]
                ax[0, 0].boxplot(bins)
                ax[0, 1].hist(x, bins=8)
                ax[0, 2].hist(x_est, bins=8)
                ax[0, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[0, 3].plot(list(range(batch, batch + len(loss_x_buff))), loss_x_buff)

                # r predictor
                r, r_est = r[mask].flatten(), r_dist.mean[mask].flatten()
                bins = []
                for i in [0.0, 1.0]:
                    bins += [r_est[r == i]]
                ax[1, 0].boxplot(bins)
                ax[1, 1].hist(r)
                ax[1, 2].hist(r_est)
                ax[1, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[1, 3].plot(list(range(batch, batch + len(loss_r_buff))), loss_r_buff)

                # r predictor
                c, c_est = c[mask].flatten(), c_dist.probs[mask].flatten()
                bins = []
                for i in [0.0, 1.0]:
                    bins += [c_est[c == i]]
                ax[2, 0].boxplot(bins)
                ax[2, 1].hist(c)
                ax[2, 2].hist(c_est)
                ax[2, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[2, 3].plot(list(range(batch, batch + len(loss_c_buff))), loss_c_buff)

                fig.canvas.draw()
                plt.pause(0.01)