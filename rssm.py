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
from torch.distributions import OneHotCategorical, Normal, Bernoulli, kl_divergence
from dists import OneHotCategoricalStraightThru
from symlog import symlog, symexp


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
        T, N, C, H, W = x.shape
        return self.embedder(x.flatten(start_dim=0, end_dim=1)).unflatten(0, (T, N))


class Encoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.encoder = nn.Sequential(
            MLPBlock(4 * 4 * cnn_multi * 2 ** 3 + h_size, mlp_hidden),
            *[MLPBlock(mlp_hidden, mlp_hidden) for _ in range(mlp_layers - 1)],
            nn.Linear(mlp_hidden, 32 * 32, bias=False),
            nn.Unflatten(1, (32, 32))
        )

    def forward(self, h, e):
        return self.encoder(torch.cat([h, e], dim=-1))


class Decoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32 * 32 + h_size, mlp_hidden, bias=False),
            *[MLPBlock(mlp_hidden, mlp_hidden) for _ in range(mlp_layers - 1)],
            MLPBlock(mlp_hidden, 4 * 4 * cnn_multi * 2 ** 3),
            nn.Unflatten(-1, (cnn_multi * 2 ** 3, 4, 4)),
            DecoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            DecoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            DecoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            DecoderConvBlock(1, 64, 64, cnn_multi),
        )

    def forward(self, h, z):
        T, N, D = h.shape
        hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
        hz_flat = hz_flat.flatten(start_dim=0, end_dim=1)
        x = self.decoder(hz_flat).unflatten(0, (T, N))
        return Normal(loc=x, scale=1.)


class DynamicsPredictor(nn.Module):
    def __init__(self, h_size=512, mlp_size=512, mlp_layers=2, z_size=32, z_cls=32):
        super().__init__()
        self.dynamics_predictor = nn.Sequential(
            MLPBlock(h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers-1)],
            nn.Linear(mlp_size, z_size * z_cls, bias=False)
        )
        self.z_size = z_size
        self.z_cls = z_cls

    def forward(self, h):
        z_flat = self.dynamics_predictor(h)
        return z_flat.unflatten(-1, (self.z_size, self.z_cls))


class RewardPredictor(nn.Module):
    def __init__(self, h_size=512, mlp_size=512, mlp_layers=2):
        super().__init__()
        self.reward_predictor = nn.Sequential(
            MLPBlock(h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers-1)],
            nn.Linear(mlp_size, 1, bias=False)
        )

    def forward(self, h):
        return Normal(loc=self.reward_predictor(h), scale=1.)


class ContinuePredictor(nn.Module):
    def __init__(self,  h_size=512, mlp_size=512, mlp_layers=2):
        super().__init__()
        self.continue_predictor = nn.Sequential(
            MLPBlock(h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers-1)],
            nn.Linear(mlp_size, 1, bias=False)
        )

    def forward(self, h):
        return Bernoulli(logits=self.continue_predictor(h))


class SequenceModel(nn.Module):
    def __init__(self, a_cls, a_size=1, h_size=512, z_size=32, z_cls=32):
        super().__init__()
        self.seq_model = nn.GRUCell(z_size * z_cls + a_size * a_cls, h_size)

    def forward(self, z, a, h):
        za_flat = torch.cat([z.flatten(-2), a.flatten(-2)], dim=1)
        return self.seq_model(za_flat, h)


class RSSM(nn.Module):
    def __init__(self, sequence_model, embedder, encoder, decoder, dynamics_pred, reward_pred, continue_pred, h_size=512):
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

        self.h_size = h_size
        self.seq_model = sequence_model
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.dynamics_pred = dynamics_pred
        self.reward_pred = reward_pred
        self.continue_pred = continue_pred


        # prediction state
        self.h = None
        self.z = None

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
        z_logit_list = [self.encoder(h0, e[0])]
        z_list = [OneHotCategoricalStraightThru(logits=z_logit_list[0]).sample()]

        for t in range(1, x.size(0)):
            h_list += [self.seq_model(z_list[t - 1], a[t - 1], h_list[t - 1])]
            z_logit_list += [self.encoder(h_list[t], e[t])]
            z_list += [OneHotCategoricalStraightThru(logits=z_logit_list[t]).sample()]

        h = torch.stack(h_list)
        z_prior_logits = torch.stack(z_logit_list)
        z = torch.stack(z_list)

        x_dist = self.decoder(h, z)
        z_post_logits = self.dynamics_pred(h)
        r_dist = self.reward_pred(h)
        c_dist = self.continue_pred(h)

        return x_dist, r_dist, c_dist, z_prior_logits, z_post_logits

    def reset(self, N=1, h0=None):
        """
        Reset the environment to start state
        :param N : batch size for simulation, default 1, ignored if h0 is used
        :param h0: [N, h_size ] : hidden variable to initialize environment, zero if None
        """
        self.h = h0 if h0 is not None else torch.zeros(N, self.h_size)
        self.z = OneHotCategorical(logits=self.dynamics_pred(self.h)).sample()
        return self.decoder(self.h, self.z).sample()

    def step(self, a):
        """
        Runs a batched step on the environment
        :param a: [N, a_size, a_cls ]
        :return: x, r, c : observation, reward, continue


        ^                          ┌────┐
        x0 ◄───────────────────────┤dec │
                                   └────┘
                                    ▲  ▲
                                    │  │   ┌─────────┐
                                    │  │   │Sequence │    
                                    │  │   │Model    │
                                    │  │   │         │
        h0──┬───────────────────────┼──┴──►│         ├───►h1─►
            │                       │      │         │
            │  ┌───┐                │      │         │
            ├─►│dyn├─►zpost ────────┴────► │         │
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

        self.h = self.seq_model(self.z, a, self.h)
        self.z = OneHotCategorical(logits=self.dynamics_pred(self.h)).sample()
        x_ = self.decoder(self.h, self.z).sample()
        r_ = self.reward_pred(self.h).sample()
        c_ = self.continue_pred(self.h).sample()
        return x_, symexp(r_), c_


class RSSMLoss:
    def __init__(self):
        self.loss_x = None
        self.loss_r = None
        self.loss_c = None
        self.loss_dyn = None
        self.loss_rep = None

    def __call__(self, x, r, c, mask, x_dist, r_dist, c_dist, z_prior_logits, z_post_logits):
        self.loss_x = - x_dist.log_prob(x).flatten(start_dim=2) * mask
        self.loss_r = - r_dist.log_prob(symlog(r)) * mask
        self.loss_c = - c_dist.log_prob(c) * mask
        self.loss_dyn = kl_divergence(
            OneHotCategorical(logits=z_prior_logits.detach()),
            OneHotCategorical(logits=z_post_logits)
        ).clamp(max=1.) * mask
        self.loss_rep = kl_divergence(
            OneHotCategorical(logits=z_prior_logits),
            OneHotCategorical(logits=z_post_logits.detach())
        ).clamp(max=1.) * mask
        return self.loss_x.mean() + self.loss_r.mean() + self.loss_c.mean() + 0.5 * self.loss_dyn.mean() + 0.1 * self.loss_rep.mean()


def make_small(action_classes, in_channels=3):
    """
    Small as per Appendix B of the Mastering Diverse Domains through World Models paper
    :param action_classes:
    :return:
    """
    return RSSM(
        sequence_model=SequenceModel(action_classes),
        embedder=Embedder(in_channels=in_channels),
        encoder=Encoder(),
        decoder=Decoder(),
        dynamics_pred=DynamicsPredictor(),
        reward_pred=RewardPredictor(),
        continue_pred=ContinuePredictor(),
        h_size=512
    )