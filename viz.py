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


def add_caption_to_observation(obs, caption_below_list):
    bottom_captions = [make_caption(*caption) for caption in caption_below_list]
    bottom_captions = np.concatenate(bottom_captions, 0)
    return np.concatenate([obs, bottom_captions], 0)


def visualize_imagined_trajectories(obs_dec, action, rewards_dec, cont, value_preds, action_meanings):
    T, N = obs_dec.shape[0:2]
    imagine_buff = replay.unstack_batch(obs_dec, action, rewards_dec, cont, value=value_preds, return_type='list')
    imagine_viz = []
    for trajectory in imagine_buff:
        imagine_viz += [viz.visualize_buff(trajectory, action_meanings=action_meanings)]
    video = []
    for t in range(T):
        frame = [images[t] for images in imagine_viz]
        frame = make_grid(make_grid(torch.from_numpy(np.stack(frame))))
        video += [frame]

    return torch.stack(video)


def make_trajectory_panel(trajectory, action_meanings=None):
    panel = visualize_buff(trajectory, action_meanings=action_meanings)
    panel_tensor = torch.from_numpy(panel)
    return make_grid(panel_tensor)


def visualize_step(step, action_meanings=None, info_keys=None, image_hw=None):
    if action_meanings is not None:
        action_caption = f'{action_meanings[step.action.argmax().item()]}'
    else:
        action_caption = f'action: {step.action.argmax().item()}'

    reward_caption = f'rew: {step.reward.item():.5f}'

    if - 0.1 < step.reward < 0.1:
        reward_color = (255, 255, 255, 255)
    elif step.reward < -0.1:
        reward_color = (255, 0, 0, 255)
    elif step.reward > 0.1:
        reward_color = (0, 255, 0, 255)
    else:
        reward_color = (255, 255, 0, 255)

    cont_caption = f'con: {step.cont.item():.5f}'
    cont_color = (floor(255 * (1 - step.cont.item())), 180, floor(255 * step.cont.item()), 255)

    caption_below_list = [(action_caption,), (reward_caption, reward_color), (cont_caption, cont_color)]

    if info_keys is not None:
        for key in info_keys:
            if key in step.info:
                caption = f'{key}: {step.info[key].item()}'
                caption_below_list += [(caption,)]
            else:
                raise Exception(f'key {key} was not found in step')

    obs = step.observation / np.max(step.observation) * 255
    obs = obs.astype(np.uint8)

    captioned_obs = add_caption_to_observation(
        obs=obs,
        caption_below_list=caption_below_list
    )

    if image_hw is not None:
        h, w, c = captioned_obs.shape
        captioned_obs = np.pad(captioned_obs, ((0, image_hw[0] - h), (0, image_hw[1] - w), (0, 0)))

    return captioned_obs.transpose((2, 0, 1))


def visualize_buff(buff, image_hw=None, action_meanings=None, info_keys=None):
    return np.stack([visualize_step(step, action_meanings, info_keys, image_hw) for step in buff])


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
    panel = visualize_buff(buff, action_meanings=action_meanings)
    panel = torch.from_numpy(panel)

    if len(panel) == 0:
        return make_grid(torch.zeros(1, 3, 64, 64))

    return make_grid(panel, nrow)


def decode_trajectory(rssm, latest_trajectory, critic):
    observations, actions, rewards, cont, value = [], [], [], [], []
    h = rssm.new_hidden0(batch_size=1)
    obs = torch.from_numpy(latest_trajectory[0].observation).permute(2, 0, 1).unsqueeze(0).to(rssm.device) / 255.
    action = torch.from_numpy(latest_trajectory[0].action).unsqueeze(0).to(rssm.device)
    z = rssm.encode_observation(h, obs).mode

    observations += [rssm.decoder(h, z).mean]
    actions += [action]
    rewards += [rssm.reward_pred(h, z).mean]
    cont += [rssm.continue_pred(h, z).mean]
    value += [critic(h, z).mean.unsqueeze(-1)]

    for step in list(latest_trajectory)[1:]:
        h, z = rssm.step_reality(h, obs, action)
        observations += [rssm.decoder(h, z).mean]
        rewards += [rssm.reward_pred(h, z).mean]
        cont += [rssm.continue_pred(h, z).mean]
        value += [critic(h, z).mean.unsqueeze(-1)]

        obs = torch.from_numpy(step.observation).permute(2, 0, 1).unsqueeze(0).to(rssm.device) / 255.
        action = torch.from_numpy(step.action).unsqueeze(0).to(rssm.device)
        actions += [action]

        if step.is_terminal:
            break

    return torch.stack(observations), torch.stack(actions), torch.stack(rewards), torch.stack(cont), torch.stack(value)
