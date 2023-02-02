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


class Encoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.encoder = nn.Linear(conf.h_size + conf.e_size, conf.z_size * conf.z_cls)
        self.conf = conf

    def forward(self, h, e):
        he_flat = torch.cat([h, e], dim=-1)
        z_flat = self.encoder(he_flat)
        return z_flat.unflatten(-1, (self.conf.z_size, self.conf.z_cls))


class Decoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.decoder = nn.Linear(conf.h_size + conf.z_size * conf.z_cls, conf.x_size * conf.x_cls)
        self.conf = conf

    def forward(self, h, z):
        """
        Decoder:              x ~ p( x | h, z )
        """
        hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
        x_flat = self.decoder(hz_flat)
        return OneHotCategorical(logits=x_flat.unflatten(-1, (self.conf.x_size, self.conf.x_cls)))


class DynamicsPredictor(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dynamics_predictor = nn.Linear(conf.h_size, conf.z_size * conf.z_cls)
        self.conf = conf

    def forward(self, h):
        z_flat = self.dynamics_predictor(h)
        return z_flat.unflatten(-1, (self.conf.z_size, self.conf.z_cls))


class RewardPredictor(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.reward_predictor = nn.Linear(conf.h_size, 1, bias=False)

    def forward(self, h):
        return Normal(loc=self.reward_predictor(h), scale=1.)


class ContinuePredictor(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.continue_predictor = nn.Linear(conf.h_size, 1)

    def forward(self, h):
        return Bernoulli(logits=self.continue_predictor(h))


class SequenceModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.seq_model = nn.GRUCell(conf.z_size * conf.z_cls + conf.a_size * conf.a_cls, conf.h_size)

    def forward(self, z, a, h):
        za_flat = torch.cat([z.flatten(-2), a.flatten(-2)], dim=1)
        return self.seq_model(za_flat, h)


class RSSM(nn.Module):
    def __init__(self, sequence_model, embedder, encoder, decoder, dynamics_pred, reward_pred, continue_pred, config):
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

        self.conf = config
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
        self.h = h0 if h0 is not None else torch.zeros(N, self.conf.h_size)
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
        self.loss_x = - x_dist.log_prob(x) * mask
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
        return (self.loss_x + self.loss_r + self.loss_c + 0.5 * self.loss_dyn + 0.1 * self.loss_rep).mean()



