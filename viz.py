import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from PIL import Image, ImageDraw
from torch.nn.functional import conv1d
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

from replay import stack_trajectory
from symlog import symexp

"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
"""


class PlotJointAndMarginals:
    def __init__(self, ax, title=None, ylabel=None, xlabel=None):
        self.ax = ax
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel

    def scatter_hist(self, x, y, binwidth=0.25):
        self.ax.cla()
        self.ax.set(aspect=1)
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlabel(self.xlabel)

        # Create marginal axes, which have 25% of the size of the main axes.  Note that
        # the inset axes are positioned *outside* (on the right and the top) of the
        # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
        # less than 0 would likewise specify positions on the left and the bottom of
        # the main axes.
        self.ax_histx = self.ax.inset_axes([0, 1.05, 1, 0.25], sharex=self.ax)
        self.ax_histy = self.ax.inset_axes([1.05, 0, 0.25, 1], sharey=self.ax)

        # no labels
        self.ax_histx.tick_params(axis="x", labelbottom=False)
        self.ax_histy.tick_params(axis="y", labelleft=False)

        # now determine nice limits by hand:
        xymax = max(np.max(x), np.max(y))
        xymin = min(np.min(x), np.min(y))

        bins = np.arange(xymin, xymax + binwidth, binwidth)
        self.ax_histx.hist(x, bins=bins)
        self.ax_histy.hist(y, bins=bins, orientation='horizontal')

        cells_x = np.digitize(x, bins=bins)
        cells_y = np.digitize(y, bins=bins)
        cells_joint = np.zeros((len(bins), len(bins)), dtype=int)

        # this could be faster if vectorized
        for x, y in zip(cells_x, cells_y):
            assert y-1 < cells_joint.shape[1], f'{x} {y} {cells_joint.shape}'
            cells_joint[y-1, x-1] += 1

        # the scatter plot:
        self.ax.imshow(cells_joint, origin='lower', interpolation='none', extent=[xymin, xymax+binwidth, xymin, xymax + binwidth])


class PlotLosses:
    def __init__(self, ax, num_losses, maxlen=1000, downsample=10):
        self.steps = 0
        self.history = [deque(maxlen=maxlen) for _ in range(num_losses)]
        self.ax = ax
        self.downsample = downsample
        self.weights = torch.ones(1, 1, downsample) / downsample

    def update(self, *args):
        assert len(args) == len(self.history), f"expected num_losses = {len(self.history)} losses"
        for i in range(len(self.history)):
            self.history[i] += [args[i]]
        self.steps += 1

    def plot(self):
        with torch.no_grad():
            self.ax.cla()
            for i in range(len(self.history)):
                loss_value = torch.tensor([self.history[i]]).float()
                loss_value = conv1d(loss_value, weight=self.weights, stride=self.downsample).squeeze()
                x = np.arange(self.steps - len(loss_value) * self.downsample, self.steps, step=self.downsample)
                self.ax.plot(x, loss_value)


class PlotImage:
    def __init__(self, ax):
        self.ax = ax
        self.image = None

    def imshow(self, image):
        if self.image is None:
            self.image = self.ax.imshow(image)
        else:
            self.image.set_data(image)


def make_caption(caption, width, height):
    img = Image.new('RGB', (width, height))
    d = ImageDraw.Draw(img)
    d.text((1, 5), caption)
    return to_tensor(img)


def add_caption_to_observation(obs, action, reward, cont, action_table):
    if action_table:
        top_caption = make_caption(f'{action_table[action.argmax().item()]}', 64, 16).to(obs.device)
    else:
        top_caption = make_caption(f'{action.argmax().item()}', 64, 16).to(obs.device)
    bottom_caption = make_caption(f'{reward.item():.2f} {cont.item():.2f}', 64, 16).to(obs.device)
    return torch.cat([top_caption, obs, bottom_caption], dim=1)


def make_trajectory_panel(trajectory, pad_action, symexp_on=True, action_table=None):
    obs, action, reward, cont = stack_trajectory(trajectory, pad_action=pad_action)
    if symexp_on:
        obs = symexp(obs)
        reward = symexp(reward)

    panel = [add_caption_to_observation(*step, action_table=action_table) for step in zip(obs, action, reward, cont)]
    panel = torch.stack(panel)
    return make_grid(panel)


def make_panel(obs, action, reward, cont, mask, filter_mask=None, sym_exp_on=True, action_table=None):
    """
    obs: observations [..., C, H, W]
    action: discrete action [..., AD, AC]
    reward: rewards [..., 1]
    cont: continue [..., 1]
    filter_mask: booleans, filter results in panel [...]
    sym_exp_on: obs and rewards are read out in sym_exp
    action_table: {0: "action1", 1:{action2}... etc}
    """

    if filter_mask is None:
        filter_mask = torch.full_like(mask.unsqueeze(-1), True, dtype=torch.bool)

    obs_sample = obs[filter_mask] * mask[filter_mask][..., None, None]
    action_sample = action[filter_mask] * mask[filter_mask][..., None]
    reward_sample = reward[filter_mask] * mask[filter_mask]
    cont_sample = cont[filter_mask] * mask[filter_mask]

    if sym_exp_on:
        obs_sample = symexp(obs_sample)
        reward_sample = symexp(reward_sample)

    panel = [add_caption_to_observation(*step, action_table) for step in zip(obs_sample, action_sample, reward_sample, cont_sample)]
    return make_grid(torch.stack(panel))


def make_gt_pred_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, filter_mask=None, sym_exp_on=True, action_table=None):
    gt_panel = make_panel(obs, action, reward, cont, mask, filter_mask, sym_exp_on, action_table)
    pred_panel = make_panel(obs_pred, action, reward_pred, cont_pred, mask, filter_mask, sym_exp_on, action_table)
    return torch.cat((gt_panel, pred_panel), dim=-1)


def make_batch_panels(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, sym_exp_on=True, action_table=None):

    batch_filter_mask = torch.full_like(mask[:, :, 0], False, dtype=torch.bool)
    batch_filter_mask[0:8, 0:8] = True
    batch_panel = make_gt_pred_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, batch_filter_mask, sym_exp_on, action_table)

    rewards_filter = reward.squeeze() != 0.0
    rewards_panel = make_gt_pred_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, rewards_filter, sym_exp_on, action_table)

    terminal_filter = (cont.squeeze() == 0.0) & mask.squeeze()
    terminal_panel = make_gt_pred_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, terminal_filter, sym_exp_on, action_table)

    return batch_panel, rewards_panel, terminal_panel


if __name__ == '__main__':
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # some random data
    x = np.random.randn(1000) + 2.0
    y = np.random.randn(1000) - 1.0

    loss = list(range(200))
    loss2 = list(range(20, 220))

    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(constrained_layout=True)

    # Create the main axes, leaving 25% of the figure space at the top and on the
    # right to position marginals.
    ax = fig.add_gridspec(nrows=2, ncols=2, top=0.75, right=0.75).subplots()

    joint1_plot = PlotJointAndMarginals(ax[0, 0], 'title', 'ylabel')
    joint2_plot = PlotJointAndMarginals(ax[1, 0], ylabel='ylabel')
    # Draw the scatter plot and marginals.
    joint1_plot.scatter_hist(x, y)
    joint2_plot.scatter_hist(x, y)
    loss1_plot = PlotLosses(ax[0, 1], 2)
    for loss, loss2 in zip(loss, loss2):
        loss1_plot.update(loss, loss2)
    loss1_plot.plot()

    plt.show()
