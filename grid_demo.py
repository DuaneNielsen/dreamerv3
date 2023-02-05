import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper
from replay import sample_batch, Step, get_trajectories, stack_trajectory
from collections import deque
from torch.nn.functional import one_hot
from torchvision.transforms.functional import resize, to_tensor
from rssm import make_small, RSSMLoss
import torch
from torch.optim import Adam
from argparse import ArgumentParser
from symlog import symlog, symexp
from time import time
import utils
import wandb
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


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


def log(obs, r, c, mask, obs_dist, r_dist, c_dist, z_prior, z_post):
    with torch.no_grad():
        reward_gt = symexp(r[mask]).flatten().cpu().numpy()
        reward_mean = symexp(r_dist.mean[mask]).flatten().cpu().numpy()
        cont_gt = c[mask].flatten().cpu().numpy()
        cont_probs = c_dist.probs[mask].flatten().cpu().numpy()
        z_prior_argmax, z_post_argmax = z_prior.argmax(-1).flatten().cpu().numpy(), z_post.argmax(
            -1).flatten().cpu().numpy()

        obs_grid = make_grid(symexp(obs[0:8, 0:8] * mask[:, 0:8, None, None]).flatten(0, 1))
        obs_pred_mean_grid = make_grid(symexp(obs_dist.mean[0:8, 0:8] * mask[0:8, 0:8, None, None]).flatten(0, 1))
        obs_panel = torch.cat((obs_grid, obs_pred_mean_grid), dim=-1)

        wandb.log({
            'obs_panel': wandb.Image(obs_panel),
        })
        log_confusion_and_hist_from_scalars('reward', reward_gt, reward_mean, 0.0, 1.0, 5)
        log_confusion_and_hist_from_scalars('continue', cont_gt, cont_probs, 0.0, 1.0, 5)
        z_labels = [f'{l:02d}' for l in list(range(32))]
        wandb.log({
            f'z_post': wandb.Histogram(z_post_argmax),
            f'z_prior': wandb.Histogram(z_prior_argmax),
            f'z_prior_z_post_confusion': wandb.plot.confusion_matrix(y_true=z_prior_argmax, preds=z_post_argmax, class_names=z_labels),
        })


def log_confusion_and_hist(name, ground_truth, pred):
    wandb.log({
        f'{name}_gt': wandb.Histogram(ground_truth),
        f'{name}_pred': wandb.Histogram(pred),
        f'{name}_confusion': wandb.plot.confusion_matrix(y_true=ground_truth, preds=pred),
    })


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


def rollout_on_world_model(env, policy):
    (h, z), r, c = env.reset(), [0], [1.0]
    x = env.decoder(h, z).mean
    while True:
        a = policy(x).unsqueeze(0).to(env.device)
        yield Step(x[0], a[0], r[0], c[0])
        h, z, r, c = env.step(a)
        x = env.decoder(h, z).mean
        if c[0] < 0.5:
            yield Step(x[0], None, r[0], c[0])
            x, r, c = env.reset(), [0], [1.0]


def make_caption(caption, width, height):
    img = Image.new('RGB', (width, height))
    d = ImageDraw.Draw(img)
    # font = ImageFont.truetype(font_manager.findSystemFonts(fontext='ttf')[0])
    d.text((1, 5), caption)
    return to_tensor(img)


def log_trajectory(trajectory):
    with torch.no_grad():
        obs, action, reward, cont = stack_trajectory(trajectory, pad_action=pad_action.to(rssm.device))

        panel = []
        for i, o in enumerate(obs.unbind(0)):
            caption = make_caption(f'{reward[i].item():.2f} {cont[i].item():.2f}', 64, 16)
            panel += [torch.cat([o, caption.to(o.device)], dim=1)]
        panel = torch.stack(panel)

        panel = make_grid(panel)

        wandb.log({
            'imagined_obs': wandb.Image(panel)
        })


def generate_and_log_trajectory_on_world_model(rssm, policy):
    imagination_gen = rollout_on_world_model(rssm, policy)
    imagine_buffer = [next(imagination_gen) for step in range(16)]
    trajectories = get_trajectories(imagine_buffer)
    log_trajectory(trajectories[0])


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

    wandb.init(project="dreamerv3-minigrid-demo")
    wandb.config.update(args)

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
        obs, act, reward, cont, mask = obs.to(args.device), act.to(args.device), reward.to(args.device), cont.to(
            args.device), mask.to(args.device)
        obs = symlog(obs)
        reward = symlog(reward)

        h0 = torch.zeros(args.batch_size, rssm.h_size, device=args.device)
        obs_dist, rew_dist, cont_dist, z_prior, z_post = rssm(obs, act, h0)
        loss = criterion(obs, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post)

        opt.zero_grad()
        loss.backward()
        opt.step()

        log_loss(loss, criterion)
        steps += 1

        if steps % 200 == 0:
            end_train = time()
            log(obs, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post)
            utils.save(utils.run.rundir, rssm, opt, args, steps, loss.item())

            generate_and_log_trajectory_on_world_model(rssm, RepeatOpenLoopPolicy([2, 2, 1, 2, 2]))

            end_plot = time()
            print(f'train time: {end_train - start_t} plot time: {end_plot - end_train}')
            start_t = time()
