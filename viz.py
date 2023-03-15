import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn.functional import conv1d
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

import replay
import viz
from replay import stack_trajectory
from math import floor

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
            assert y - 1 < cells_joint.shape[1], f'{x} {y} {cells_joint.shape}'
            cells_joint[y - 1, x - 1] += 1

        # the scatter plot:
        self.ax.imshow(cells_joint, origin='lower', interpolation='none',
                       extent=[xymin, xymax + binwidth, xymin, xymax + binwidth])


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


def make_caption(caption, color=(255, 255, 255, 255), fontsize=8, width=64, height=16, anchor=None):
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    unicode_font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    draw.text((1, 5), caption, fill=color, font=unicode_font, anchor=anchor)
    return np.array(img)


def visualize_imagined_trajectories(obs_dec, action, rewards_dec, cont, value_preds, visualizer,
                                    inverse_obs_transform=None):
    T, N = obs_dec.shape[0:2]
    imagine_buff = replay.unstack_batch(obs_dec, action, rewards_dec, cont, value=value_preds, return_type='list',
                                        inverse_obs_transform=inverse_obs_transform)
    imagine_viz = []
    for trajectory in imagine_buff:
        imagine_viz += [viz.visualize_buff(trajectory, visualizer=visualizer)]
    video = []
    for t in range(T):
        frame = [images[t] for images in imagine_viz]
        frame = make_grid(make_grid(torch.from_numpy(np.stack(frame))))
        video += [frame]

    return torch.stack(video)


def make_trajectory_panel(trajectory, action_meanings=None):
    panel = visualize_buff(trajectory, visualizer=VizStep(action_meanings=action_meanings))
    panel_tensor = torch.from_numpy(panel)
    return make_grid(panel_tensor)


def viz_reward(reward):
    reward_caption = f'rew: {reward.item():.5f}'

    if - 0.1 < reward < 0.1:
        reward_color = (255, 255, 255, 255)
    elif reward < -0.1:
        reward_color = (255, 0, 0, 255)
    elif reward > 0.1:
        reward_color = (0, 255, 0, 255)
    else:
        reward_color = (255, 255, 0, 255)
    return make_caption(reward_caption, reward_color)


def viz_cont(cont):
    cont_caption = f'con: {cont.item():.5f}'
    cont_color = (floor(255 * (1 - cont.item())), 180, floor(255 * cont.item()), 255)
    return make_caption(cont_caption, cont_color)


class VizAction:
    def __init__(self, action_meanings):
        self.action_meanings = action_meanings

    def __call__(self, action):
        if self.action_meanings is not None:
            action_caption = f'{self.action_meanings[action.argmax().item()]}'
        else:
            action_caption = f'action: {action.argmax().item()}'
        return make_caption(action_caption)


def normalized_image(observation):
    obs = observation / np.max(observation) * 255
    return obs.astype(np.uint8)


class VizStep:
    def __init__(self, action_meanings=None, hw=None):
        self.viz_action = VizAction(action_meanings)
        self.hw = hw
        self.hooks = []

    def observation(self, step):
        return normalized_image(step.observation)

    def action(self, step):
        return self.viz_action(step.action)

    def reward(self, step):
        return viz_reward(step.reward)

    def cont(self, step):
        return viz_cont(step.cont)

    def add_hook(self, viz_f):
        self.hooks += [viz_f]

    def __call__(self, step):
        images = [self.observation(step), self.action(step), self.reward(step), self.cont(step)]
        for viz_f in self.hooks:
            images += [viz_f(step)]
        image = np.concatenate(images, 0)
        if self.hw is not None:
            h, w, c = image.shape
            image = np.pad(image, ((0, self.hw[0] - h), (0, self.hw[1] - w), (0, 0)))
        return image.transpose((2, 0, 1))


class ValueHook:
    def __init__(self):
        self.color = (255, 255, 200)

    def __call__(self, step):
        if 'value' in step.info:
            val = step.info['value']
            return make_caption(f'value: {val}', color=self.color)
        else:
            return make_caption(f'no value', color=self.color)


def visualize_buff(buff, visualizer):
    return np.stack([visualizer(step) for step in buff])


def make_panel(obs, action, reward, cont, value=None, action_meanings=None, nrow=8):
    """
    obs: observations [..., C, H, W]
    action: discrete action [..., AD, AC]
    reward: rewards [..., 1]
    cont: continue [..., 1]
    filter_mask: booleans, filter results in panel [...]
    action_table: {0: "action1", 1:{action2}... etc}
    """

    buff = replay.unstack_batch(obs, action, reward, cont, value=value)
    panel = visualize_buff(buff, visualizer=VizStep(action_meanings=action_meanings))
    panel = torch.from_numpy(panel)

    if len(panel) == 0:
        return make_grid(torch.zeros(1, 3, 64, 64))

    return make_grid(panel, nrow)