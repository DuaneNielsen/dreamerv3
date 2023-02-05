import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

import wandb
from replay import stack_trajectory
from symlog import symexp


def log_loss(loss, criterion):
    wandb.log({
        'loss': loss.item(),
        'loss_pred': criterion.loss_obs.item() + criterion.loss_reward.item() + criterion.loss_cont.item(),
        'loss_dyn': criterion.loss_dyn.item(),
        'loss_rep': criterion.loss_rep.item()
    })


def log_confusion_and_hist_from_scalars(name, ground_truth, pred, min, max, num_bins):
    bins = np.linspace(min, max, num_bins)
    ground_truth_digital = np.digitize(ground_truth, bins=bins, right=True)
    pred_digital = np.digitize(pred, bins=bins)
    labels = [f'{b:.2f}' for b in bins] + [f'>{bins[-1]:.2f}']
    wandb.log({
        f'{name}_gt': wandb.Histogram(ground_truth, num_bins=num_bins),
        f'{name}_pred': wandb.Histogram(pred, num_bins=num_bins),
        f'{name}_confusion': wandb.plot.confusion_matrix(y_true=ground_truth_digital, preds=pred_digital, class_names=labels),
    })


def log_confusion_and_hist(name, ground_truth, pred):
    wandb.log({
        f'{name}_gt': wandb.Histogram(ground_truth),
        f'{name}_pred': wandb.Histogram(pred),
        f'{name}_confusion': wandb.plot.confusion_matrix(y_true=ground_truth, preds=pred),
    })


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


def log_trajectory(trajectory, pad_action, symexp_on=True, action_table=None):
    with torch.no_grad():
        obs, action, reward, cont = stack_trajectory(trajectory, pad_action=pad_action)
        if symexp_on:
            obs = symexp(obs)
            reward = symexp(reward)

        panel = [add_caption_to_observation(*step, action_table=action_table) for step in zip(obs, action, reward, cont)]
        panel = torch.stack(panel)
        panel = make_grid(panel)

        wandb.log({
            'imagined_obs': wandb.Image(panel)
        })


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


def log_nonzero_rewards_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, sym_exp_on=True, action_table=None):
    filter = reward.squeeze() != 0.0
    gt_panel = make_panel(obs, action, reward, cont, mask, filter, sym_exp_on, action_table)
    pred_panel = make_panel(obs_pred, action, reward_pred, cont_pred, mask, filter, sym_exp_on, action_table)
    panel = torch.cat((gt_panel, pred_panel), dim=-1)
    wandb.log({
        'nonzero_rewards_panel': wandb.Image(panel)
    })


def log_terminal_state_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, sym_exp_on=True, action_table=None):
    filter = (cont.squeeze() == 0.0) & mask.squeeze()
    gt_panel = make_panel(obs, action, reward, cont, mask, filter, sym_exp_on, action_table)
    pred_panel = make_panel(obs_pred, action, reward_pred, cont_pred, mask, filter, sym_exp_on, action_table)
    panel = torch.cat((gt_panel, pred_panel), dim=-1)
    wandb.log({
        'terminal_state_panel': wandb.Image(panel)
    })


def log_training_panel(obs, action, reward, cont, obs_pred, reward_pred, cont_pred, mask, sym_exp_on=True, action_table=None):

    filter_mask = torch.full_like(mask[:, :, 0], False, dtype=torch.bool)
    filter_mask[0:8, 0:8] = True

    gt_panel = make_panel(obs, action, reward, cont, mask, filter_mask, sym_exp_on, action_table)
    pred_panel = make_panel(obs_pred, action, reward_pred, cont_pred, mask, filter_mask, sym_exp_on, action_table)

    panel = torch.cat((gt_panel, pred_panel), dim=-1)

    wandb.log({
        'obs_panel': wandb.Image(panel),
    })