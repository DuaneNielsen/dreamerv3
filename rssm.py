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


 Dataflow Diagram (partial)

    X0               a                      X1
    │                │                      │
    ▼                │                      ▼
┌──────────┐         │                  ┌──────────┐
│ Embedder │         │                  │ Embedder │
└──────────┘         │                  └──────────┘
    e                │                      e
    │                ▼                      │
    ▼             ┌────────────┐            ▼
┌───────────┐     │            │        ┌───────────┐
│ Encoder   │     │            │ h1┌───►│ Encoder   │
└──┬────────┘     │            │  ─┤    └───┬───────┘
   │     ▲        │ Sequence   │   │        │
   ▼     │        │            │   │        ▼
 zprior ─┼───────►│ Predictor  │   │      zprior
   │     │        │            │   │        │
   │     h0 ─────►│            │   │        │
   │     │        └────────────┘   │        │
   ▼     ▼                         ▼        ▼
┌───────────┐                     ┌───────────┐
│ Decoder   │                     │ Decoder   │
└───┬───────┘                     └─────────┬─┘
    │                                       │
    ▼                                       ▼

    X0                                      X1

"""

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import OneHotCategorical
from utils import sample_one_hot
from env import Env
from replay import ReplayBuffer, simple_trajectory


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
        return z_flat.unflatten(-1, (z_size, z_cls))


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
        return x_flat.unflatten(-1, (x_size, x_cls))


# models
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
        self.seq_pred = nn.GRUCell(z_size * z_cls + a_size * a_cls, h_size)
        self.decoder = Decoder()

    def forward(self, x, a, h0):
        """

        :param x: [T, N, x_size, x_cls]
        :param a: [T, N, a_size, a_cls]
        :param h0: [N, h0]
        :return: x, h, z
        """

        h_list = [h0]
        e0 = self.embedder(x[0])
        z_prior_list = [self.encoder(h0, e0)]

        for t in range(1, x.size(0)-1):
            za_flat = torch.cat([z_prior_list[t-1].flatten(-2), a[t-1].flatten(-2)], dim=1)
            h_list += [self.seq_pred(za_flat, h_list[t-1])]
            e_t = self.embedder(x[t])
            z_prior_list += [self.encoder(h_list[t], e_t)]

        h = torch.stack(h_list)
        z_prior = torch.stack(z_prior_list)
        x_ = self.decoder(h, z_prior)

        return x_, h, z_prior


if __name__ == '__main__':

    # dataset
    buff = ReplayBuffer()
    buff += simple_trajectory([Env.right] * 8)
    buff += simple_trajectory([Env.left])

    rssm = RSSM()

    for batch in range(5000):
        x, a, r, c, next_x, mask = buff.sample_batch(T, batch_size)
        h0 = torch.zeros(batch_size, h_size)
        x_, h, z_prior = rssm(x, a, h0)

