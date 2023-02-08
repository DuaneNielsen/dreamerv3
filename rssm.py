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
from torch.distributions import OneHotCategorical, Normal, Bernoulli
from dists import OneHotCategoricalStraightThru, categorical_kl_divergence_clamped
from blocks import MLPBlock


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
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
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
    def __init__(self, out_channels=3, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32 * 32 + h_size, mlp_hidden, bias=False),
            *[MLPBlock(mlp_hidden, mlp_hidden) for _ in range(mlp_layers - 1)],
            MLPBlock(mlp_hidden, 4 * 4 * cnn_multi * 2 ** 3),
            nn.Unflatten(-1, (cnn_multi * 2 ** 3, 4, 4)),
            ModernDecoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            ModernDecoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            ModernDecoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            ModernDecoderConvBlock(out_channels, 64, 64, cnn_multi),
        )

    def forward(self, h, z):
        if len(h.shape) == 3:
            T, N, D = h.shape
            hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
            hz_flat = hz_flat.flatten(start_dim=0, end_dim=1)
            x = self.decoder(hz_flat).unflatten(0, (T, N))
        else:
            hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
            x = self.decoder(hz_flat)

        return Normal(loc=x, scale=1.)


class DynamicsPredictor(nn.Module):
    def __init__(self, h_size=512, mlp_size=512, mlp_layers=2, z_size=32, z_cls=32):
        super().__init__()
        self.dynamics_predictor = nn.Sequential(
            MLPBlock(h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers - 1)],
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
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers - 1)],
            nn.Linear(mlp_size, 1, bias=False)
        )

    def forward(self, h):
        return Normal(loc=self.reward_predictor(h), scale=1.)


class ContinuePredictor(nn.Module):
    def __init__(self,  h_size=512, mlp_size=512, mlp_layers=2):
        super().__init__()
        self.continue_predictor = nn.Sequential(
            MLPBlock(h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers - 1)],
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
        self.dummy = nn.Parameter(torch.empty(0))

        # prediction state
        self.h = None
        self.z = None

    @property
    def device(self):
        return self.dummy.device

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
        embedding = self.embedder(x)
        z_logit_list = [self.encoder(h0, embedding[0])]
        z_list = [OneHotCategoricalStraightThru(logits=z_logit_list[0]).sample()]

        for t in range(1, x.size(0)):
            h_list += [self.seq_model(z_list[t - 1], a[t - 1], h_list[t - 1])]
            z_logit_list += [self.encoder(h_list[t], embedding[t])]
            z_list += [OneHotCategoricalStraightThru(logits=z_logit_list[t]).sample()]

        h = torch.stack(h_list)
        z_prior_logits = torch.stack(z_logit_list)
        z = torch.stack(z_list)

        x_dist = self.decoder(h, z)
        z_post_logits = self.dynamics_pred(h)
        reward_symlog_dist = self.reward_pred(h)
        continue_dist = self.continue_pred(h)

        return x_dist, reward_symlog_dist, continue_dist, z_prior_logits, z_post_logits

    def reset(self, N=1, h0=None, x=None):
        """
        Reset the environment to start state
        :param N : batch size for simulation, default 1, ignored if h0 is used
        :param h0: [N, h_size ] : hidden variable to initialize environment, zero if None
        """
        self.h = h0 if h0 is not None else torch.zeros(N, self.h_size, device=self.device)
        if x is not None:
            embed = self.embedder(x)
            self.z = self.encoder(h0, embed[0])
        else:
            self.z = OneHotCategorical(logits=self.dynamics_pred(self.h)).sample()
        return self.h, self.z

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
        reward = self.reward_pred(self.h).mean
        cont = self.continue_pred(self.h).probs
        return self.h, self.z, reward, cont

    def imagine(self, h0, obs, reward, cont, policy, imagination_horizon=15):
        """
        runs a policy on the model from the provided start states
        :param h0: [N, hdim]
        :param obs: [N, C, H, W] starting observation
        :param reward: [N, 1] rewards
        :param cont: [N, 1] continue
        :param policy: a = policy(h, z)
        :param imagination_horizon: steps to project using the model, default 15
        :return: h [H, N, h_size], z [H, N, z_size, z_classes], a [H, N, a_size, a_classes],
        reward [H, N, 1], continue [H, N, 1]
        where H = imagination_horizon + 1
        """

        (h, z), r, c = self.reset(h0=h0, x=obs[0]), reward[0], cont[0]
        a = policy(h, z)

        h_list, z_list, a_list, reward_list, cont_list = [h], [z], [a], [r], [c]

        for t in range(imagination_horizon):
            h, z, reward, cont = self.step(a)
            a = policy(h, z)

            h_list += [h]
            z_list += [z]
            a_list += [a]
            reward_list += [reward]
            cont_list += [cont]

        h = torch.stack(h_list)
        z = torch.stack(z_list)
        a = torch.stack(a_list)
        reward = torch.stack(reward_list)
        cont = torch.stack(cont_list)
        return h, z, a, reward, cont


class RSSMLoss:
    def __init__(self):
        self.loss_obs = None
        self.loss_reward = None
        self.loss_cont = None
        self.loss_dyn = None
        self.loss_rep = None
        self.loss = None

    def __call__(self, obs, reward, cont, mask, obs_dist, reward_dist, cont_dist, z_prior_logits, z_post_logits):
        self.loss_obs = - obs_dist.log_prob(obs).flatten(start_dim=2) * mask
        self.loss_reward = - reward_dist.log_prob(reward) * mask
        self.loss_cont = - cont_dist.log_prob(cont) * mask
        self.loss_dyn = 0.5 * categorical_kl_divergence_clamped(z_prior_logits.detach(), z_post_logits) * mask
        self.loss_rep = 0.1 * categorical_kl_divergence_clamped(z_prior_logits, z_post_logits.detach()) * mask
        self.loss_obs = self.loss_obs.mean()
        self.loss_reward = self.loss_reward.mean()
        self.loss_cont = self.loss_cont.mean()
        self.loss_dyn = self.loss_dyn.mean()
        self.loss_rep = self.loss_rep.mean()
        self.loss = self.loss_obs + self.loss_reward + self.loss_cont + self.loss_dyn + self.loss_rep
        return self.loss

    def loss_dict(self):
        return {
            'loss': self.loss.item(),
            'loss_pred': self.loss_obs.item() + self.loss_reward.item() + self.loss_cont.item(),
            'loss_dyn': self.loss_dyn.item(),
            'loss_rep': self.loss_rep.item()
        }


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
        decoder=Decoder(out_channels=in_channels),
        dynamics_pred=DynamicsPredictor(),
        reward_pred=RewardPredictor(),
        continue_pred=ContinuePredictor(),
        h_size=512
    )