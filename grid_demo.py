import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import RewardWrapper
from minigrid.wrappers import RGBImgObsWrapper

import replay
from replay import BatchLoader, Step
from collections import deque
from torch.nn.functional import one_hot, mse_loss, cross_entropy
from torchvision.transforms.functional import resize
from rssm import make_small, RSSMLoss
from critic import Critic, td_lambda, polyak_update, Actor, ActorLoss
import torch
from torch.optim import Adam
from argparse import ArgumentParser
import copy
import utils
import wandb
from viz import make_batch_panels
from wandb.data_types import Video
import viz
import envs.gridworld as gridworld
import numpy as np



def log_decoded_trajectory(latest_trajectory, steps):
    decoded_trajectory = []
    h = rssm.new_hidden0(batch_size=1)
    z = rssm.encode_observation(h, latest_trajectory[0].obs.unsqueeze(0).to(args.device)).mode
    for step in latest_trajectory:
        decoded_trajectory += [rssm.decoder(h, z).mean]
        if step.act is None:
            break
        h, z = rssm.step_reality(h, step.obs.unsqueeze(0).to(args.device), step.act.unsqueeze(0).to(args.device))

    decoded_trajectory = torch.stack(decoded_trajectory, dim=1)
    decoded_trajectory = (decoded_trajectory * 255).to(dtype=torch.uint8, device='cpu').numpy()
    wandb.log({'decoded_trajectory': wandb.Video(decoded_trajectory)}, step=steps)


def rollout(env, actor, rssm):
    rssm.eval()
    actor.eval()
    obs, info = env.reset()
    h = rssm.new_hidden0(batch_size=1)
    obs = obs.to(rssm.device).unsqueeze(0)
    z = rssm.encode_observation(h, obs).mode
    reward, terminated, truncated, cont = 0.0, False, False, 1.

    while True:
        action = actor.sample_action(h, z)
        rssm.train()
        rssm.train()
        yield Step(obs[0].cpu(), action[0].cpu(), reward, cont)
        rssm.eval()
        actor.eval()

        obs, reward, terminated, truncated, info = env.step(action[0].cpu())
        obs = obs.to(rssm.device).unsqueeze(0)
        h, z = rssm.step_reality(h, obs, action)

        cont = 0. if terminated else 1.

        if terminated or truncated:
            rssm.train()
            actor.train()
            yield Step(obs[0].cpu(), None, reward, cont)
            rssm.eval()
            actor.eval()

            obs, info = env.reset()
            obs = obs.to(rssm.device).unsqueeze(0)
            h = rssm.new_hidden0(batch_size=1)
            z = rssm.encode_observation(h, obs).mode
            reward, terminated, truncated, cont = 0.0, False, False, 1.


def train_world_model(buff, rssm, rssm_opt, rssm_criterion, step=None):
    obs, action, reward, cont, mask = loader.sample(buff, args.batch_length, args.batch_size)

    h0 = rssm.new_hidden0(args.batch_size)
    obs_dist, reward_dist, cont_dist, z_prior, z_post = rssm(obs, action, h0)
    rssm_loss = rssm_criterion(obs, reward, cont, mask, obs_dist, reward_dist, cont_dist, z_prior, z_post)

    rssm_opt.zero_grad()
    rssm_loss.backward()
    rssm_opt.step()

    with torch.no_grad():
        wandb.log(rssm_criterion.loss_dict(), step=step)
        if step % args.log_every_n_steps == 0:
            with torch.no_grad():
                batch_panel, rewards_panel, terminal_panel = \
                    make_batch_panels(obs, action, reward, cont, obs_dist.mean,
                                      reward_dist.mean.unsqueeze(-1), cont_dist.probs, mask,
                                      action_table=action_table)

                wandb.log({
                    'batch_panel': wandb.Image(batch_panel),
                    'nonzero_rewards_panel': wandb.Image(rewards_panel),
                    'terminal_state_panel': wandb.Image(terminal_panel),
                    'reward_gt': wandb.Histogram(reward[mask].flatten().cpu(), num_bins=256),
                    'reward_pred': wandb.Histogram(reward_dist.mean.unsqueeze(-1)[mask].flatten().cpu(), num_bins=256),
                    'cont_gt': wandb.Histogram(cont[mask].flatten().cpu(), num_bins=5),
                    'cont_probs': wandb.Histogram(cont_dist.probs[mask].flatten().cpu(), num_bins=5),
                    'z_prior': wandb.Histogram(z_prior.argmax(-1).flatten().cpu().numpy(), num_bins=32),
                    'z_post': wandb.Histogram(z_post.argmax(-1).flatten().cpu().numpy()),
                }, step=step)


def train_actor_critic(buff, rssm, critic, critic_opt, ema_critic, actor, actor_opt, actor_criterion, step=None):
    with torch.no_grad():
        obs, act, reward, cont, mask = loader.sample(buff, batch_length=1, batch_size=args.batch_size)
        h0 = rssm.new_hidden0(args.batch_size)
        h, z, a, rewards, cont = \
            rssm.imagine(h0, obs[0], reward[0], cont[0], actor, imagination_horizon=args.imagination_horizon)

        values_ema_dist = ema_critic(h, z)
        value_ema = values_ema_dist.mean.unsqueeze(-1)
        value_targets = td_lambda(rewards, cont, value_ema)

    value_preds_dist = critic(h, z)
    critic_loss = - value_preds_dist.log_prob(value_targets).mean()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    polyak_update(critic, ema_critic)

    actor_logits = actor(h, z)
    actor_loss = actor_criterion(actor_logits, a, value_targets)

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    with torch.no_grad():

        wandb.log(actor_criterion.loss_dict(), step=step)
        wandb.log({'critic_loss': critic_loss.item()}, step=step)

        if replay.is_trajectory_end(buff[-1]):
            latest_trajectory = replay.get_tail_trajectory(buff)
            trajectory_reward = replay.total_reward(latest_trajectory)
            trajectory_viz = viz.visualize_trajectory(latest_trajectory, pad_action, action_table=action_table)
            trajectory_viz = (trajectory_viz * 255).to(dtype=torch.uint8).numpy()
            wandb.log({
                'trajectory_reward': trajectory_reward,
                'trajectory_length': len(latest_trajectory),
                'trajectory_viz': Video(trajectory_viz)
            }, step=step)

            log_decoded_trajectory(latest_trajectory, step)

            print(f'trajectory end: reward {trajectory_reward} len: {len(latest_trajectory)}')

        if step % args.log_every_n_steps == 0:

            obs_imagined = rssm.decoder(h, z).mean
            mask = mask.repeat(1 + args.imagination_horizon, 1, 1)

            wandb.log({
                'rewards_dec': wandb.Histogram(rewards.cpu(), num_bins=256),
                'value_ema_dec': wandb.Histogram(value_ema.cpu(), num_bins=256),
                'value_targets_dec': wandb.Histogram(value_targets.cpu(), num_bins=256),
                'value_preds_dec': wandb.Histogram(value_preds_dist.mean.cpu(), num_bins=256),
            }, step=step)

            imagined_trajectory_viz = viz.visualize_imagined_trajectory(
                obs_imagined[:, :], a[:, :], rewards[:, :], cont[:, :], mask[:, :],
                value_targets[:, :], action_table=action_table)
            imagined_trajectory_viz = (imagined_trajectory_viz * 255).to(dtype=torch.uint8).cpu().numpy()
            wandb.log({'imagined_trajectory': wandb.Video(imagined_trajectory_viz)}, step=step)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--train_ratio', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_length', type=int, default=64)
    parser.add_argument('--replay_capacity', type=int, default=10 ** 6)
    parser.add_argument('--world_model_learning_rate', type=float, default=1e-4)
    parser.add_argument('--world_model_adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--world_model_gradient_clipping', type=float, default=1000.)
    parser.add_argument('--actor_critic_learning_rate', type=float, default=3e-5)
    parser.add_argument('--actor_critic_adam_epsilon', type=float, default=1e-5)
    parser.add_argument('--actor_critic_gradient_clipping', type=float, default=100.)
    parser.add_argument('--imagination_horizon', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_every_n_steps', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=8 * 10 ** 4)
    parser.add_argument('--env_action_classes', type=int, default=3)
    parser.add_argument('--env_action_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    +------+---------+-------------------+    
    """
    action_table = {0: 'left', 1: 'right', 2: 'forward'}

    env = gridworld.make('dont_look_back')
    env = gridworld.MaxStepsWrapper(env, max_steps=100)
    env = gridworld.RGBImageWrapper(env)
    env = gridworld.TensorObsWrapper(env)
    env = gridworld.OneHotTensorActionWrapper(env)

    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), args.env_action_classes)

    rssm = make_small(action_classes=args.env_action_classes).to(args.device)
    rssm_opt = Adam(rssm.parameters(), lr=args.world_model_learning_rate, eps=args.world_model_adam_epsilon)


    def register_gradient_clamp(nn_module, gradient_min_max):
        for p in nn_module.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -gradient_min_max, gradient_min_max))


    register_gradient_clamp(rssm, args.world_model_gradient_clipping)
    rssm_criterion = RSSMLoss()

    critic = Critic()
    ema_critic = copy.deepcopy(critic)
    actor = Actor(action_size=args.env_action_size, action_classes=args.env_action_classes)
    critic, ema_critic, actor = critic.to(args.device), ema_critic.to(args.device), actor.to(args.device)
    critic_opt = Adam(critic.parameters(), lr=args.actor_critic_learning_rate, eps=args.actor_critic_adam_epsilon)
    actor_opt = Adam(actor.parameters(), lr=args.actor_critic_learning_rate, eps=args.actor_critic_adam_epsilon)
    register_gradient_clamp(critic, args.actor_critic_gradient_clipping)
    register_gradient_clamp(actor, args.actor_critic_gradient_clipping)
    actor_criterion = ActorLoss()

    buff = deque(maxlen=args.replay_capacity)
    gen = rollout(env, actor, rssm)
    for i in range(20):
        buff += [next(gen)]

    if args.resume:
        rssm, rssm_optim, critic, critic_optim, actor, actor_optim, steps, resume_args = \
            utils.load(args.resume, rssm, rssm_opt, critic, critic_opt, actor, actor_opt)
        print(f'resuming from step {steps} of {args.resume} with {resume_args}')

    loader = BatchLoader(pad_observation=pad_state, pad_action=pad_action, device=args.device)

    for step in range(args.max_steps):

        buff += [next(gen)]
        train_world_model(buff, rssm, rssm_opt, rssm_criterion, step)
        train_actor_critic(buff, rssm, critic, critic_opt, ema_critic, actor, actor_opt, actor_criterion, step)

        if step % args.log_every_n_steps == 0:
            utils.save(utils.run.rundir, rssm, rssm_opt, critic, critic_opt, actor, actor_opt, args, step)
            print(f'saved model at step {step}')
