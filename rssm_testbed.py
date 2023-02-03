import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import OneHotCategorical, Bernoulli, Normal
from symlog import symlog, symexp
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from env import Env
from replay import Step, sample_batch, rollout_open_loop_policy
from collections import deque
from rssm import RSSM, RSSMLoss


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


def make_linear_rssm(config):
    return RSSM(
        sequence_model=SequenceModel(config),
        embedder=nn.Flatten(-2),
        encoder=Encoder(config),
        decoder=Decoder(config),
        dynamics_pred=DynamicsPredictor(config),
        reward_pred=RewardPredictor(config),
        continue_pred=ContinuePredictor(config),
        h_size=config.h_size
    )


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--x_size', type=int, default=1, required=False)
    parser.add_argument('--x_cls', type=int, default=10, required=False)
    parser.add_argument('--e_size', type=int, default=10, required=False)
    parser.add_argument('--z_size', type=int, default=1, required=False)
    parser.add_argument('--z_cls', type=int, default=10, required=False)
    parser.add_argument('--h_size', type=int, default=32, required=False)
    parser.add_argument('--a_size', type=int, default=1, required=False)
    parser.add_argument('--a_cls', type=int, default=10, required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    parser.add_argument('--t_horizon', type=int, default=8, required=False)
    conf = parser.parse_args()

    # visualize
    plt.ion()
    fig, ax = plt.subplots(4, 5)
    loss_buff, loss_x_buff, loss_r_buff, loss_c_buff, loss_dyn_buff, loss_rep_buff = \
        deque(maxlen=400), deque(maxlen=400), deque(maxlen=400), deque(maxlen=400), deque(maxlen=400), deque(maxlen=400)
    plt.show()

    # dataset
    buff = deque(maxlen=1000000)
    env = Env()
    for _ in range(20):
        for step in rollout_open_loop_policy(env, [Env.right] * 8):
            buff.append(step)
        for step in rollout_open_loop_policy(env, [Env.left]):
            buff.append(step)

    rssm = make_linear_rssm(conf)
    opt = Adam(rssm.parameters(), lr=1e-3)
    criterion = RSSMLoss()

    for batch in range(5000):
        x, a, r, c, mask = sample_batch(buff, conf.t_horizon, conf.batch_size, Env.pad_state, Env.pad_action)

        h0 = torch.zeros(conf.batch_size, conf.h_size)
        x_dist, r_dist, c_dist, z_prior, z_post = rssm(x, a, h0)
        loss = criterion(x, r, c, mask, x_dist, r_dist, c_dist, z_prior, z_post)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_buff += [loss.item()]
        loss_x_buff += [criterion.loss_x.item()]
        loss_r_buff += [criterion.loss_r.item()]
        loss_c_buff += [criterion.loss_c.item()]
        loss_dyn_buff += [criterion.loss_dyn.item()]
        loss_rep_buff += [criterion.loss_rep.item()]

        # visualize training
        with torch.no_grad():
            if batch % 50 == 0:
                for row in ax:
                    for subplot in row:
                        subplot.clear()

                # x predictor
                x, x_est = x.argmax(-1)[mask].flatten().detach().cpu(), x_dist.logits.argmax(-1)[
                    mask].flatten().detach().cpu()
                bins = []
                for i in range(1, 9):
                    bins += [x_est[x == i]]
                ax[0, 0].set_title("joint dist: gt and training")
                ax[0, 0].set_ylabel("observations")
                ax[0, 0].boxplot(bins)
                ax[0, 1].set_title("ground truth marginals")
                ax[0, 1].hist(x, bins=8)
                ax[0, 2].set_title("training marginals")
                ax[0, 2].hist(x_est, bins=8)
                ax[0, 3].set_title("losses")
                ax[0, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[0, 3].plot(list(range(batch, batch + len(loss_x_buff))), loss_x_buff)

                # r predictor
                r, r_est = r[mask].flatten(), symexp(r_dist.mean[mask].flatten())
                bins = []
                for i in [0.0, 1.0]:
                    bins += [r_est[r == i]]
                ax[1, 0].set_ylabel("rewards")
                ax[1, 0].boxplot(bins)
                ax[1, 1].hist(r, bins=2)
                ax[1, 2].hist(r_est, bins=2)
                ax[1, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[1, 3].plot(list(range(batch, batch + len(loss_r_buff))), loss_r_buff)

                # c predictor
                c, c_est = c[mask].flatten(), c_dist.probs[mask].flatten()
                bins = []
                for i in [0.0, 1.0]:
                    bins += [c_est[c == i]]
                ax[2, 0].set_ylabel("continue")
                ax[2, 0].boxplot(bins)
                ax[2, 1].hist(c, bins=2)
                ax[2, 2].hist(c_est, bins=2)
                ax[2, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[2, 3].plot(list(range(batch, batch + len(loss_c_buff))), loss_c_buff)

                # r predictor
                z_prior_prob = torch.argmax(z_prior, -1)[mask].flatten()
                z_post_prob = torch.argmax(z_post, -1)[mask].flatten()
                joint_hist = torch.zeros((10, 10))
                for prior, post in zip(z_prior_prob, z_post_prob):
                    joint_hist[prior.item(), post.item()] += 1
                ax[3, 0].set_ylabel("z_prior/z_post")
                ax[3, 0].imshow(joint_hist)
                ax[3, 1].hist(z_prior_prob, bins=10)
                ax[3, 2].hist(z_post_prob, bins=10)
                ax[3, 3].plot(list(range(batch, batch + len(loss_buff))), loss_buff)
                ax[3, 3].plot(list(range(batch, batch + len(loss_dyn_buff))), loss_dyn_buff)
                ax[3, 3].plot(list(range(batch, batch + len(loss_rep_buff))), loss_rep_buff)

                # run the policies on the model
                buffer = deque()

                go_right = [Env.right] * 8
                go_left = [Env.left]

                def rollout(policy):
                    x, r, c = rssm.reset(), torch.tensor([[0.0]]), torch.tensor([[1.0]])
                    for a in policy:
                        a = a.unsqueeze(0)
                        yield Step(x[0], a[0], r[0], c[0])
                        x, r, c = rssm.step(a)
                    yield Step(x[0], None, r[0], c[0])

                for n in range(10):
                    for step in rollout(go_right):
                        buffer.append(step)
                    for step in rollout(go_left):
                        buffer.append(step)

                x_, a, r_, c_, mask = sample_batch(buffer, 8, 12, Env.pad_state, Env.pad_action)
                x_ = x_.argmax(-1)[mask].flatten()
                ax[0, 4].set_title("inference marginals")
                ax[0, 4].hist(x_, bins=8)
                r_ = r_[mask].flatten()
                ax[1, 4].hist(r_, bins=2)
                c_ = c_[mask].flatten()
                ax[2, 4].hist(c_, bins=2)
                a = a.argmax(-1)[mask].flatten()
                ax[3, 4].set_title("action dist")
                ax[3, 4].hist(a, bins=2)

                fig.canvas.draw()
                plt.pause(0.01)

