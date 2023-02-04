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
from viz import PlotLosses, PlotJointAndMarginals, PlotImage
from symlog import symlog, symexp
from time import time
import utils


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


class Dashboard:
    def __init__(self):
        # viz
        self.fig = plt.figure(constrained_layout=True)

        self.grid_spec = GridSpec(nrows=4, ncols=4, figure=self.fig)
        self.xt_img_ax = self.fig.add_subplot(self.grid_spec[0, 0])
        self.x_mean_img_ax = self.fig.add_subplot(self.grid_spec[0, 1])
        self.x_mean_dist_ax = self.fig.add_subplot(self.grid_spec[0, 2])
        self.joint_r_ax = self.fig.add_subplot(self.grid_spec[1, 0:2])
        self.joint_c_ax = self.fig.add_subplot(self.grid_spec[2, 0:2])
        self.joint_z_ax = self.fig.add_subplot(self.grid_spec[3, 0:2])
        self.loss_x_ax = self.fig.add_subplot(self.grid_spec[0, 3])
        self.loss_r_ax = self.fig.add_subplot(self.grid_spec[1, 3])
        self.loss_c_ax = self.fig.add_subplot(self.grid_spec[2, 3])
        self.loss_z_ax = self.fig.add_subplot(self.grid_spec[3, 3])

        self.x_gt_img_plt = PlotImage(self.xt_img_ax)
        self.x_mean_img_plt = PlotImage(self.x_mean_img_ax)
        self.joint_r_plt = PlotJointAndMarginals(self.joint_r_ax, title='joint distri', ylabel='reward')
        self.joint_c_plt = PlotJointAndMarginals(self.joint_c_ax, ylabel='continue')
        self.joint_z_plt = PlotJointAndMarginals(self.joint_z_ax, ylabel='z - latent')
        self.loss_x_plt = PlotLosses(self.loss_x_ax, num_losses=2)
        self.loss_r_plt = PlotLosses(self.loss_r_ax, num_losses=2)
        self.loss_c_plt = PlotLosses(self.loss_c_ax, num_losses=2)
        self.loss_z_plt = PlotLosses(self.loss_z_ax, num_losses=3)
        plt.pause(0.1)

    def update_loss(self, loss, criterion):
        self.loss_x_plt.update(loss.item(), criterion.loss_x.item())
        self.loss_r_plt.update(loss.item(), criterion.loss_r.item())
        self.loss_c_plt.update(loss.item(), criterion.loss_c.item())
        self.loss_z_plt.update(loss.item(), criterion.loss_dyn.item(), criterion.loss_rep.item())

    def plot(self, x, r, c, mask, x_dist, r_dist, c_dist, z_prior, z_post):
        def to_img(x):
            x = x.permute(1, 2, 0) * 255
            x = x.to(dtype=torch.uint8).clamp(0, 255).cpu().numpy()
            return x

        self.x_gt_img_plt.imshow(to_img(symexp(x[0, 0])))
        self.x_mean_img_plt.imshow(to_img(symexp(x_dist.mean[0, 0])))
        x_hist = torch.masked_select(symexp(x_dist.mean).flatten(start_dim=2), mask)
        sample = torch.rand_like(x_hist)
        x_hist = x_hist[sample < 0.01] * 255
        self.x_mean_dist_ax.cla()
        self.x_mean_dist_ax.hist(x_hist.to(dtype=torch.uint8).cpu())
        self.joint_r_plt.scatter_hist(symexp(r[mask]).flatten().cpu().numpy(), symexp(r_dist.mean[mask]).flatten().cpu().numpy())
        self.joint_c_plt.scatter_hist(c[mask].flatten().cpu().numpy(), c_dist.probs[mask].flatten().cpu().numpy())
        self.joint_z_plt.scatter_hist(z_prior.argmax(-1).flatten().cpu().numpy(), z_post.argmax(-1).flatten().cpu().numpy())
        self.loss_x_plt.plot()
        self.loss_r_plt.plot()
        self.loss_c_plt.plot()
        self.loss_z_plt.plot()
        plt.pause(0.05)

    def save(self, filename):
        plt.savefig(filename)


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
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    dashboard = Dashboard()

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

    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = RGBImgObsWrapper(env, tile_size=13)
    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), 4)

    buff = deque(maxlen=args.replay_capacity)
    gen = rollout(env, RepeatOpenLoopPolicy([2, 2, 1, 2, 2]))
    for i in range(10):
        buff += [next(gen)]

    rssm = make_small(action_classes=4).to(args.device)
    opt = Adam(rssm.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    for p in rssm.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -args.gradient_clipping, args.gradient_clipping))
    criterion = RSSMLoss()

    steps = 0
    if args.resume:
        rssm, opt, steps, resume_args, loss = utils.load(args.resume, rssm, opt)
        print(f'resuming from step {steps} of {args.resume} with {resume_args}')

    start_t = time()
    while True:
        buff += [next(gen)]
        obs, act, reward, cont, mask = sample_batch(buff, args.batch_length, args.batch_size, pad_state, pad_action)
        obs, act, reward, cont, mask = obs.to(args.device), act.to(args.device), reward.to(args.device), cont.to(args.device), mask.to(args.device)
        obs = symlog(obs)
        reward = symlog(reward)

        h0 = torch.zeros(args.batch_size, rssm.h_size, device=args.device)
        obs_dist, rew_dist, cont_dist, z_prior, z_post = rssm(obs, act, h0)
        loss = criterion(obs, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post)

        opt.zero_grad()
        loss.backward()
        opt.step()

        dashboard.update_loss(loss, criterion)
        steps += 1

        if steps % 300 == 0:
            end_train = time()
            with torch.no_grad():
                dashboard.plot(obs, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post)
            utils.save(utils.run.rundir, rssm, opt, args, steps, loss.item())
            dashboard.save(utils.run.rundir + '/dashboard.png')
            end_plot = time()

            print(f'train time: {end_train - start_t} plot time: {end_plot - end_train}')
            start_t = time()

