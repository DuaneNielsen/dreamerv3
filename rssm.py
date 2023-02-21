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
from blocks import MLPBlock, ModernDecoderConvBlock, Embedder, DecoderConvBlock
from torch.nn.functional import cross_entropy
from encoding import decode_onehot, encode_onehot

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
        """
        Encodes a single timestep.

        Note that since we require h to encode, we cannot encode more than 1 step at a time,
        so the input does not take a T dimension
        param: h : [N, h_size]
        param: e :  [N, embed_size]
        """
        return self.encoder(torch.cat([h, e], dim=-1))


class Decoder(nn.Module):
    def __init__(self, out_channels=3, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(32 * 32 + h_size, mlp_hidden, bias=False),
            *[MLPBlock(mlp_hidden, mlp_hidden) for _ in range(mlp_layers - 1)],
            MLPBlock(mlp_hidden, 4 * 4 * cnn_multi * 2 ** 3),
            nn.Unflatten(-1, (cnn_multi * 2 ** 3, 4, 4)),
            DecoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            DecoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            DecoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            DecoderConvBlock(out_channels, 64, 64, cnn_multi),
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

        output = nn.Linear(mlp_size, 256, bias=False)
        # output.weight.data.zero_()
        # output.bias.data.zero_()
        # output.bias.data[127] = 1e-8

        self.reward_predictor = nn.Sequential(
            MLPBlock(h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers - 1)],
            output
        )

    def forward(self, h):
        return self.reward_predictor(h)


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
         Decoder:              x ~ p( x | h, z )\ 
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
        # self.h = None
        # self.z = None

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
        reward_probs = self.reward_pred(h)
        continue_dist = self.continue_pred(h)

        return x_dist, reward_probs, continue_dist, z_prior_logits, z_post_logits

    def new_hidden0(self, batch_size):
        """
        :param batch_size: batch size
        return: a tensor with the correct humber of hidden dims for the rssm
        """
        return torch.zeros(batch_size, self.h_size, device=self.device)

    def encode_observation(self, h, obs):
        """
        :param h : [N, h_size ]
        :param obs : [N, C, H, W]
        :return z_dist : [N, 32, 32] OneHotCategoricalStraightThru
        """
        embed = self.embedder(obs.unsqueeze(0)).squeeze(0)
        return OneHotCategoricalStraightThru(logits=self.encoder(h, embed))

    def step_reality(self, h, obs, a):
        """
        Runs a batched step on the environment,
        using either an observation or an imagined latent
        :param h: [N, h_size]
        :param a: [N, a_size, a_cls ]
        :param z: [N, z_size, z_classes]
        :return: h, z : hidden, z_prior


        """

        z = self.encode_observation(h, obs).mode
        h = self.seq_model(z, a, h)
        return h, z

    def step_imagine(self, h, z, a):
        """
        Runs a batched step on the environment,
        using either an observation or an imagined latent
        :param h: [N, h_size]
        :param a: [N, a_size, a_cls ]
        :param z: [N, z_size, z_classes]
        :return: h, z, r, c : hidden, z, reward, continue


        """

        h = self.seq_model(z, a, h)
        z_imagine = OneHotCategorical(logits=self.dynamics_pred(h)).sample()
        reward = self.reward_pred(h)
        cont = self.continue_pred(h).probs
        return h, z_imagine, reward, cont

    def imagine(self, h0, obs, reward, cont, policy, imagination_horizon=15):
        """
        runs a policy on the model from the provided start states
        :param h0: [N, hdim]
        :param obs: [N, C, H, W] starting observation
        :param reward: [N, enc_size] rewards in encoded format
        :param cont: [N, 1] continue
        :param policy: a = policy(h, z)
        :param imagination_horizon: steps to project using the model, default 15
        :return: h [H, N, h_size], z [H, N, z_size, z_classes], a [H, N, a_size, a_classes],
        reward [H, N, 1], continue [H, N, 1]
        where H = imagination_horizon + 1
        """

        z = self.encode_observation(h0, obs).sample()
        a = policy.sample_action(h0, z)

        h_list, z_list, a_list, reward_enc_list, cont_list = [h0], [z], [a], [reward], [cont]

        for t in range(imagination_horizon):
            h, z, reward, cont = self.step_imagine(h_list[-1], z_list[-1], a_list[-1])
            a = policy.sample_action(h, z)

            h_list += [h]
            z_list += [z]
            a_list += [a]
            reward_enc_list += [reward]
            cont_list += [cont]

        h = torch.stack(h_list)
        z = torch.stack(z_list)
        a = torch.stack(a_list)
        reward = torch.stack(reward_enc_list)
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

    def __call__(self, obs, reward_encoded, cont, mask, obs_dist, reward_logits, cont_dist, z_prior_logits, z_post_logits):
        self.loss_obs = - obs_dist.log_prob(obs).flatten(start_dim=2) * mask
        reward_onehot_encoded = reward_encoded * mask
        reward_logits = reward_logits * mask
        self.loss_reward = cross_entropy(reward_logits.flatten(0, 1), reward_onehot_encoded.flatten(0, 1))
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


if __name__ == '__main__':

    """
    Load a saved RSSM and manually inspect the model quality
    """

    import utils
    from torch.optim import Adam
    from argparse import ArgumentParser
    from matplotlib import pyplot as plt
    from torch.distributions import OneHotCategorical
    from matplotlib.widgets import Button
    from torch.nn.functional import one_hot

    with torch.no_grad():
        parser = ArgumentParser()
        parser.add_argument('--resume', type=str, required=True)
        parser.add_argument('--action_classes', type=int, required=True)
        args = parser.parse_args()

        rssm = make_small(action_classes=args.action_classes, in_channels=3)
        steps, _ = utils.load(args.resume, rssm)


        class ImaginedEnv:
            def __init__(self):

                self.h = None
                self.z = None

            def reset(self):
                self.h = rssm.new_hidden0(batch_size=1)
                z_logits = rssm.dynamics_pred(self.h)
                self.z = OneHotCategorical(logits=z_logits).mode
                return rssm.decoder(self.h, self.z).mean

            def step(self, action):
                if self.h is None:
                    raise Exception('call env.reset() before step')
                self.h, self.z, r, c = rssm.step_imagine(self.h, self.z, action)
                obs = rssm.decoder(self.h, self.z).mean
                return obs, r, c

        def press_button(action):
            action = one_hot(torch.tensor([[action]]), args.action_classes)
            obs, r, c = env.step(action)
            obs_plt.set_data(obs[0].permute(1, 2, 0).clamp(0, 1))
            fig.canvas.draw()
            fig.canvas.flush_events()

        def press_zero(event):
            press_button(0)

        def press_one(event):
            press_button(1)

        def press_two(event):
            press_button(2)

        def press_three(event):
            press_button(3)


        env = ImaginedEnv()
        obs = env.reset()

        fig, ax = plt.subplots()
        obs_plt = ax.imshow(obs[0].permute(1, 2, 0).detach().clamp(0, 1))
        fig.canvas.draw()
        fig.canvas.flush_events()

        axzero = fig.add_axes([0.2, 0.05, 0.1, 0.075])
        bzero = Button(axzero, '0')
        bzero.on_clicked(press_zero)

        axone = fig.add_axes([0.3, 0.05, 0.1, 0.075])
        bone = Button(axone, '1')
        bone.on_clicked(press_one)

        axone = fig.add_axes([0.4, 0.05, 0.1, 0.075])
        btwo = Button(axone, '2')
        btwo.on_clicked(press_two)

        axthree = fig.add_axes([0.5, 0.05, 0.1, 0.075])
        bthree = Button(axthree, '3')
        bthree.on_clicked(press_three)


        plt.show()
