import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from minigrid.wrappers import RGBImgObsWrapper
from replay import sample_batch, Step
from collections import deque
from torch.nn.functional import one_hot
from torchvision.transforms.functional import resize
from rssm import make_small, RSSMLoss
import torch
from torch.optim import Adam
from argparse import ArgumentParser
from viz import PlotLosses, PlotJointAndMarginals
from symlog import symlog, symexp


def prepro(x):
    x = torch.from_numpy(x['image']).permute(2, 0, 1).float() / 255.0
    return resize(x, [64, 64])


class RepeatOpenLoopPolicy:
    def __init__(self, actions):
        self.i = 0
        self.actions = actions

    def __call__(self, x):
        a = self.actions[self.i]
        self.i = (self.i + 1) % len(self.actions)
        return one_hot(torch.tensor([a]), 4)


def random_policy(x):
    return one_hot(torch.randint(0, 4, [1]), 4)


def rollout(env, policy, seed=42):
    (x, _), r, c = env.reset(seed=seed), 0, 1.0
    while True:
        a = policy(x)
        yield Step(prepro(x), a, r, c)
        x, r, terminated, truncated, _ = env.step(a.argmax().item())
        c = 0.0 if terminated else 1.0
        if terminated or truncated:
            yield Step(prepro(x), None, r, c)
            (x, _), r, c = env.reset(seed=seed), 0, 1.0


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_length', type=int, default=64)
    parser.add_argument('--replay_capacity', type=int, default=10 ** 6)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--gradient_clipping', type=float, default=1000.)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    buff = deque(maxlen=args.replay_capacity)

    """
    Environment action space
    +------+---------+-------------------+
    | Num  |  Name   |      Action       |
    +------+---------+-------------------+
    |    0 | left    | Turn left         |
    |    1 | right   | Turn right        |
    |    2 | forward | Move forward      |
    |    3 | pickup  | Pick up an object |
    +------+---------+-------------------+    
    """

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = RGBImgObsWrapper(env, tile_size=13)
    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), 4)

    rssm = make_small(action_classes=4)
    opt = Adam(rssm.parameters(), lr=args.learning_rate)
    criterion = RSSMLoss()

    # viz
    fig = plt.figure(constrained_layout=True)
    # ax = fig.add_gridspec(nrows=4, ncols=2, top=0.75, right=0.75).subplots()

    gs = GridSpec(nrows=4, ncols=4, figure=fig)
    x_gt_img_ax = fig.add_subplot(gs[0, 0])
    x_mean_img_ax = fig.add_subplot(gs[0, 1])
    joint_r_ax = fig.add_subplot(gs[1, 0:2])
    joint_c_ax = fig.add_subplot(gs[2, 0:2])
    joint_z_ax = fig.add_subplot(gs[3, 0:2])
    loss_x_ax = fig.add_subplot(gs[0, 3])
    loss_r_ax = fig.add_subplot(gs[1, 3])
    loss_c_ax = fig.add_subplot(gs[2, 3])
    loss_z_ax = fig.add_subplot(gs[3, 3])

    joint_r_plt = PlotJointAndMarginals(joint_r_ax, title='joint distri', ylabel='reward')
    joint_c_plt = PlotJointAndMarginals(joint_c_ax, ylabel='continue')
    joint_z_plt = PlotJointAndMarginals(joint_z_ax, ylabel='z - latent')
    loss_x_plt = PlotLosses(loss_x_ax, num_losses=2)
    loss_r_plt = PlotLosses(loss_r_ax, num_losses=2)
    loss_c_plt = PlotLosses(loss_c_ax, num_losses=2)
    loss_z_plt = PlotLosses(loss_z_ax, num_losses=3)

    gen = rollout(env, RepeatOpenLoopPolicy([2, 2, 1, 2, 2]))
    for i in range(10):
        buff += [next(gen)]

    for steps in range(10000):
        buff += [next(gen)]
        x, a, r, c, mask = sample_batch(buff, args.batch_length, args.batch_size, pad_state, pad_action)
        x = symlog(x)
        r = symlog(r)

        h0 = torch.zeros(args.batch_size, rssm.h_size)
        x_dist, r_dist, c_dist, z_prior, z_post = rssm(x, a, h0)
        loss = criterion(x, r, c, mask, x_dist, r_dist, c_dist, z_prior, z_post)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            def to_img(x):
                x = x.permute(1, 2, 0) * 255
                x = x.to(dtype=torch.uint8).cpu().numpy()
                return x

            x_gt_img_ax.imshow(to_img(symexp(x[0, 0])))
            x_mean_img_ax.imshow(to_img(symexp(x_dist.mean[0, 0])))
            joint_r_plt.scatter_hist(symexp(r[mask]).flatten().cpu().numpy(), symexp(r_dist.mean[mask]).flatten().cpu().numpy())
            joint_c_plt.scatter_hist(c[mask].flatten().cpu().numpy(), c_dist.probs[mask].flatten().cpu().numpy())
            joint_z_plt.scatter_hist(z_prior.argmax(-1).flatten().cpu().numpy(), z_post.argmax(-1).flatten().cpu().numpy())
            loss_x_plt.plot(loss.item(), criterion.loss_x.item())
            loss_r_plt.plot(loss.item(), criterion.loss_r.item())
            loss_c_plt.plot(loss.item(), criterion.loss_c.item())
            loss_z_plt.plot(loss.item(), criterion.loss_dyn.item(), criterion.loss_rep.item())



