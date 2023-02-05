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


def log_trajectory(trajectory, pad_action):
    with torch.no_grad():
        obs, action, reward, cont = stack_trajectory(trajectory, pad_action=pad_action)
        obs = symexp(obs)

        panel = []
        for i, o in enumerate(obs.unbind(0)):
            caption = make_caption(f'{symexp(reward[i]).item():.2f} {cont[i].item():.2f}', 64, 16)
            panel += [torch.cat([o, caption.to(o.device)], dim=1)]
        panel = torch.stack(panel)

        panel = make_grid(panel)

        wandb.log({
            'imagined_obs': wandb.Image(panel)
        })
