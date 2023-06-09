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
from torch import nn as nn
from torch.distributions import Bernoulli, Categorical, Independent

import symlog
from dists import OneHotCategoricalStraightThru, categorical_kl_divergence, TwoHotSymlog, \
    OneHotCategoricalUnimix, ImageNormalDist, ImageMSEDist, ProcessedDist
from blocks import MLPBlock, ModernDecoderBatchNormConvBlock, DecoderConvBlock, init_weights, EncoderBatchNormConvBlock


class Embedder(nn.Module):
    def __init__(self, in_channels=3, cnn_multi=32):
        super().__init__()
        self.embedder = nn.Sequential(
            EncoderBatchNormConvBlock(in_channels, 64, 64, cnn_multi),
            EncoderBatchNormConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            EncoderBatchNormConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            EncoderBatchNormConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            nn.Flatten()
        )

    def forward(self, x):
        """
        param: x: [T, N, C, H, W] observation in standard form
        """
        batch_shape = x.shape[0:-3]
        # T, N, C, H, W = x.shape
        return self.embedder(x.flatten(start_dim=0, end_dim=len(batch_shape)-1)).unflatten(0, batch_shape)


class Encoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_hidden=640, h_size=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4 * 4 * cnn_multi * 2 ** 3 + h_size, mlp_hidden, bias=False),
            nn.LayerNorm([mlp_hidden]),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 32 * 32, bias=True),
            nn.Unflatten(1, (32, 32))
        )
        self.apply(init_weights)

    def forward(self, h, e):
        """
        Encodes a single timestep.

        Note that since we require h to encode, we cannot encode more than 1 step at a time,
        so the input does not take a T dimension
        param: h : [N, h_size]
        param: e :  [N, embed_size]
        """
        return self.encoder(torch.cat([h, e], dim=-1))


class LinearEncoder(nn.Module):
    def __init__(self, cnn_multi=32, mlp_hidden=640, h_size=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4 * 4 * cnn_multi * 2 ** 3 + h_size, 32 * 32, bias=False),
            nn.Unflatten(1, (32, 32))
        )
        self.apply(init_weights)

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
    def __init__(self, out_channels=3, cnn_multi=32, mlp_hidden=640, h_size=512, z_size=32, z_cls=32):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_cls * z_size + h_size, mlp_hidden, bias=False),
            # nn.LayerNorm([mlp_hidden]),
            # nn.SiLU(),
            nn.Linear(mlp_hidden, 4 * 4 * cnn_multi * 2 ** 3, bias=True),
            nn.Unflatten(-1, (cnn_multi * 2 ** 3, 4, 4)),
            DecoderConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            DecoderConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            DecoderConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            DecoderConvBlock(out_channels, 64, 64, cnn_multi, bias=True),
        )
        self.apply(init_weights)

        # self.register_full_backward_hook(lambda module, grad_input, grad_output: print(module, grad_input))
        # self.register_backward_hook(lambda grad: print(grad))

    def forward(self, h, z):
        if len(h.shape) == 3:
            T, N, D = h.shape
            hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
            hz_flat = hz_flat.flatten(start_dim=0, end_dim=1)
            x = self.decoder(hz_flat).unflatten(0, (T, N))
        else:
            hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
            x = self.decoder(hz_flat)

        return ImageMSEDist(mode=x, dims=3)


class ModernDecoder(nn.Module):
    def __init__(self, out_channels=3, cnn_multi=32, mlp_layers=2, mlp_hidden=512, h_size=512, z_size=32, z_cls=32):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_cls * z_size + h_size, 4 * 4 * cnn_multi * 2 ** 3, bias=True),
            nn.Unflatten(-1, (cnn_multi * 2 ** 3, 4, 4)),
            ModernDecoderBatchNormConvBlock(cnn_multi * 2 ** 2, 8, 8, cnn_multi * 2 ** 3),
            ModernDecoderBatchNormConvBlock(cnn_multi * 2 ** 1, 16, 16, cnn_multi * 2 ** 2),
            ModernDecoderBatchNormConvBlock(cnn_multi * 2 ** 0, 32, 32, cnn_multi * 2 ** 1),
            ModernDecoderBatchNormConvBlock(out_channels, 64, 64, cnn_multi, bias=True),
        )
        self.apply(init_weights)

    def forward(self, h, z):
        if len(h.shape) == 3:
            T, N, D = h.shape
            hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
            hz_flat = hz_flat.flatten(start_dim=0, end_dim=1)
            x = self.decoder(hz_flat).unflatten(0, (T, N))
        else:
            hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
            x = self.decoder(hz_flat)

        return ImageMSEDist(mode=x, dims=3)


class DynamicsPredictor(nn.Module):
    def __init__(self, h_size=512, mlp_size=640, z_size=32, z_cls=32):
        super().__init__()
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(h_size, mlp_size, bias=False),
            nn.LayerNorm([mlp_size]),
            nn.SiLU(),
            nn.Linear(mlp_size, z_size * z_cls, bias=True),
        )
        self.z_size = z_size
        self.z_cls = z_cls
        self.apply(init_weights)

    def forward(self, h):
        z_flat = self.dynamics_predictor(h)
        return z_flat.unflatten(-1, (self.z_size, self.z_cls))


class LinearDynamicsPredictor(nn.Module):
    def __init__(self, h_size=512, mlp_size=640, z_size=32, z_cls=32):
        super().__init__()
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(h_size, z_size * z_cls, bias=True),
        )
        self.z_size = z_size
        self.z_cls = z_cls
        self.apply(init_weights)

    def forward(self, h):
        z_flat = self.dynamics_predictor(h)
        return z_flat.unflatten(-1, (self.z_size, self.z_cls))



class RewardPredictor(nn.Module):
    def __init__(self, h_size=512, z_size=32, z_cls=32, mlp_size=512, mlp_layers=2):
        super().__init__()

        self.output = nn.Linear(mlp_size, 256, bias=False)

        self.reward_predictor = nn.Sequential(
            MLPBlock(z_cls * z_size + h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers - 1)],
            self.output
        )
        self.apply(init_weights)
        self.output.weight.data.zero_()

    def forward(self, h, z):
        hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
        return TwoHotSymlog(self.reward_predictor(hz_flat))


class ContinuePredictor(nn.Module):
    def __init__(self, h_size=512, z_size=32, z_cls=32, mlp_size=512, mlp_layers=2):
        super().__init__()
        self.continue_predictor = nn.Sequential(
            MLPBlock(z_cls * z_size + h_size, mlp_size),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers - 1)],
            nn.Linear(mlp_size, 1, bias=False)
        )
        self.apply(init_weights)

    def forward(self, h, z):
        hz_flat = torch.cat([h, z.flatten(-2)], dim=-1)
        return Bernoulli(logits=self.continue_predictor(hz_flat))


class SequenceModel(nn.Module):
    def __init__(self, a_cls, a_size=1, h_size=512, z_size=32, z_cls=32):
        super().__init__()
        self.seq_model = nn.GRUCell(z_size * z_cls + a_size * a_cls, h_size)
        self.apply(init_weights)

    def forward(self, z, a, h):
        za_flat = torch.cat([z.flatten(-2), a.flatten(-2)], dim=1)
        return self.seq_model(za_flat, h)


class EmbedderPrepro(nn.Module):
    def __init__(self, embedder, prepro):
        super().__init__()
        self.embedder = embedder
        self.prepro = prepro

    def forward(self, x):
        x = self.prepro(x)
        return self.embedder(x)


class DecoderProcessed(nn.Module):
    def __init__(self, decoder, prepro, postpro):
        super().__init__()
        self.decoder = decoder
        self.prepro = prepro
        self.postpro = postpro

    def forward(self, h, z):
        obs_dist = self.decoder(h, z)
        return ProcessedDist(obs_dist, self.prepro, self.postpro)


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
        z_post_logit_list = [self.encoder(h0, embedding[0])]
        z_post_list = [OneHotCategoricalStraightThru(logits=z_post_logit_list[0]).sample()]

        for t in range(1, x.size(0)):
            h_list += [self.seq_model(z_post_list[t - 1], a[t - 1], h_list[t - 1])]
            z_post_logit_list += [self.encoder(h_list[t], embedding[t])]
            z_post_list += [OneHotCategoricalStraightThru(logits=z_post_logit_list[t]).sample()]

        h = torch.stack(h_list)
        z_post_logits = torch.stack(z_post_logit_list)
        z_post = torch.stack(z_post_list)

        x_dist = self.decoder(h, z_post)
        z_prior_logits = self.dynamics_pred(h)
        reward_probs = self.reward_pred(h, z_post)
        continue_dist = self.continue_pred(h, z_post)

        return x_dist, reward_probs, continue_dist, z_prior_logits, z_post_logits, z_post

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
        r = self.reward_pred(h, z)
        c = self.continue_pred(h, z)
        decode_obs = self.decoder(h, z)
        return h, z, r, c, decode_obs

    def single_step_imagine(self, h, z, a):
        """
        Runs a batched step on the environment,
        using either an observation or an imagined latent
        :param h: [N, h_size]
        :param a: [N, a_size, a_cls ]
        :param z: [N, z_size, z_classes]
        :return: h, z, r, c : hidden, z, reward, continue


        """

        h = self.seq_model(z, a, h)
        z_imagine = OneHotCategoricalUnimix(logits=self.dynamics_pred(h)).sample()
        reward = self.reward_pred(h, z_imagine).mean.unsqueeze(-1)
        cont = self.continue_pred(h, z_imagine).probs
        return h, z_imagine, reward, cont

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
        z_imagine = OneHotCategoricalStraightThru(logits=self.dynamics_pred(h)).sample()
        return h, z_imagine

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
        a = policy(h0, z).sample()

        h_list, z_list, a_list, reward_enc_list, cont_list = [h0], [z], [a], [reward], [cont]

        for t in range(imagination_horizon):
            h, z = self.step_imagine(h_list[-1], z_list[-1], a_list[-1])
            a = policy(h, z).sample()

            h_list += [h]
            z_list += [z]
            a_list += [a]

        h = torch.stack(h_list)
        z = torch.stack(z_list)
        a = torch.stack(a_list)

        reward = self.reward_pred(h, z).mean.unsqueeze(-1)
        cont = self.continue_pred(h, z).probs

        return h.detach(), z.detach(), a.detach(), reward.detach(), cont.detach()


class RSSMLoss:
    def __init__(self):
        self.loss_obs = None
        self.loss_reward = None
        self.loss_cont = None
        self.loss_dyn = None
        self.loss_rep = None
        self.loss = None

    def __call__(self, obs, rewards, cont, obs_dist, reward_dist, cont_dist, z_prior_logits, z_post_logits):
        self.loss_obs = - obs_dist.log_prob(obs)
        self.loss_reward = - reward_dist.log_prob(rewards.squeeze(-1))
        self.loss_cont = - cont_dist.log_prob(cont).squeeze(-1)
        self.loss_dyn = categorical_kl_divergence(z_post_logits.detach(), z_prior_logits)
        self.loss_rep = categorical_kl_divergence(z_post_logits, z_prior_logits.detach())
        self.loss = self.loss_obs + self.loss_reward + self.loss_cont + 0.5 * self.loss_dyn + 0.1 * self.loss_rep
        self.loss = self.loss.mean()
        return self.loss

    def loss_dict(self):
        return {
            'wm_loss': self.loss.item(),
            'wm_loss_obs': self.loss_obs.mean().item(),
            'wm_loss_reward': self.loss_reward.mean().item(),
            'wm_loss_cont': self.loss_cont.mean().item(),
            'wm_loss_dyn': self.loss_dyn.mean().item(),
            'wm_loss_rep': self.loss_rep.mean().item()
        }


if __name__ == '__main__':

    """
    Load a saved RSSM and manually inspect the model quality
    """

    from argparse import ArgumentParser
    from matplotlib import pyplot as plt
    from torch.distributions import OneHotCategorical
    from matplotlib.widgets import Button
    from torch.nn.functional import one_hot
    from config import make

    with torch.no_grad():
        parser = ArgumentParser()
        parser.add_argument('--resume', type=str, required=True)
        parser.add_argument('--action_classes', type=int, required=True)
        args = parser.parse_args()

        checkpoint = torch.load(args.resume)
        step, run_args = checkpoint['step'], checkpoint['args']
        rssm, actor, critic = make(model_size='medium',action_size=1, action_classes=args.action_classes, in_channels=3)
        rssm.load_state_dict(checkpoint['world_model_state_dict']['rssm_state_dict'])

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
                self.h, self.z, r, c = rssm.single_step_imagine(self.h, self.z, action)
                obs = rssm.decoder(self.h, self.z).mean
                return obs, r, c


        def press_button(action):
            action = one_hot(torch.tensor([[action]]), args.action_classes)
            obs, r, c = env.step(action)
            obs_plt.set_data(normalize(obs)[0].clamp(0, 255).to(torch.uint8).detach().numpy())
            tx.set_text(f'reward {r} continue: {c}')
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

        def press_four(event):
            press_button(4)


        env = ImaginedEnv()
        obs = env.reset()

        def normalize(rgb_image):
            return (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255

        fig, ax = plt.subplots()
        obs_plt = ax.imshow(normalize(obs)[0].detach().clamp(0, 255).to(torch.uint8).detach().numpy())

        tx = ax.text(3, 8, 'reward_zero', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
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

        axfour = fig.add_axes([0.6, 0.05, 0.1, 0.075])
        bfour = Button(axfour, '4')
        bfour.on_clicked(press_four)


        plt.show()
