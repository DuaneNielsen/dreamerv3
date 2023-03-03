import gymnasium
from gymnasium.core import WrapperActType, ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box
from envs.simpler_gridworld import PartialRGBObservationWrapper

import replay
from replay import BatchLoader, Step
from collections import deque
from torchvision.transforms.functional import to_tensor
from rssm import make_small, RSSMLoss
from actor_critic import score, CriticLoss, ActorLoss, traj_weight, polyak_update, Actor, Critic
import torch
from torch.optim import Adam
from argparse import ArgumentParser
import copy
import utils
import wandb
from utils import register_gradient_clamp
from viz import make_batch_panels
from wandb.data_types import Video
import viz
import envs.gridworld as gridworld
import numpy as np
from copy import deepcopy
from time import time


def rollout(env, actor, rssm, pad_action):
    obs, info = env.reset()
    h = rssm.new_hidden0(batch_size=1)
    obs_tensor = to_tensor(obs).to(rssm.device).unsqueeze(0)
    z = rssm.encode_observation(h, obs_tensor).mode
    reward, terminated, truncated = 0.0, False, False

    while True:
        action = actor.sample_action(h, z)
        yield Step(obs, action[0].detach().cpu().numpy(), reward, terminated, truncated)

        obs, reward, terminated, truncated, info = env.step(action[0].cpu())
        obs_tensor = to_tensor(obs).to(rssm.device).unsqueeze(0)
        h, z = rssm.step_reality(h, obs_tensor, action)

        if terminated or truncated:
            yield Step(obs, pad_action, reward, terminated, truncated)

            obs, info = env.reset()
            obs_tensor = to_tensor(obs).to(rssm.device).unsqueeze(0)
            h = rssm.new_hidden0(batch_size=1)
            z = rssm.encode_observation(h, obs_tensor).mode
            reward, terminated, truncated = 0.0, False, False


def train_world_model(buff, rssm, rssm_opt, rssm_criterion, step=None):
    obs, action, reward, cont = loader.sample(buff, args.batch_length, args.batch_size)

    h0 = rssm.new_hidden0(args.batch_size)
    obs_dist, reward_dist, cont_dist, z_prior, z_post = rssm(obs, action, h0)
    rssm_loss = rssm_criterion(obs, reward, cont, obs_dist, reward_dist, cont_dist, z_prior, z_post)

    rssm_opt.zero_grad()
    rssm_loss.backward()
    rssm_opt.step()

    with torch.no_grad():
        wandb.log(rssm_criterion.loss_dict(), step=step)

        if step % args.log_every_n_steps == 0:
            with torch.no_grad():
                batch_panel, rewards_panel, terminal_panel = \
                    make_batch_panels(obs, action, reward, cont, obs_dist.mean,
                                      reward_dist.mean.unsqueeze(-1), cont_dist.probs,
                                      action_table=action_table)

                wandb.log({
                    'wm_batch_panel': wandb.Image(batch_panel),
                    'wm_nonzero_rewards_panel': wandb.Image(rewards_panel),
                    'wm_terminal_state_panel': wandb.Image(terminal_panel),
                    'wm_reward_gt': wandb.Histogram(reward.flatten().cpu(), num_bins=256),
                    'wm_reward_pred': wandb.Histogram(reward_dist.mean.unsqueeze(-1).flatten().cpu(), num_bins=256),
                    'wm_cont_gt': wandb.Histogram(cont.flatten().cpu(), num_bins=5),
                    'wm_cont_probs': wandb.Histogram(cont_dist.probs.flatten().cpu(), num_bins=5),
                    'wm_z_prior': wandb.Histogram(z_prior.argmax(-1).flatten().cpu().numpy(), num_bins=32),
                    'wm_z_post': wandb.Histogram(z_post.argmax(-1).flatten().cpu().numpy()),
                }, step=step)


class ActorCriticTrainer:
    def __init__(self, actor, critic, lr, adam_eps, grad_clip, device):
        self.actor_criterion = ActorLoss()
        self.critic_criterion = CriticLoss()
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.ema_critic = deepcopy(critic).to(device)
        self.critic_opt = Adam(critic.parameters(), lr=lr, eps=adam_eps)
        self.actor_opt = Adam(actor.parameters(), lr=lr, eps=adam_eps)
        register_gradient_clamp(critic, grad_clip)
        register_gradient_clamp(actor, grad_clip)

    def train(self, h, z, action, rewards, cont):
        self.h = h
        self.z = z
        self.action = action
        self.rewards = rewards
        self.cont = cont

        with torch.no_grad():
            critic_dist = self.critic(h.detach(), z.detach())
            self.returns, self.values = score(rewards, cont, critic_dist.mean.unsqueeze(-1))
            tw = traj_weight(cont)

        self.critic_dist = critic(h.detach()[:-1], z.detach()[:-1])
        self.ema_critic_dist = self.ema_critic(h.detach()[:-1], z.detach()[:-1])
        crit_loss = self.critic_criterion(self.returns, self.critic_dist, self.ema_critic_dist, tw[:-1])

        self.critic_opt.zero_grad()
        crit_loss.backward()
        self.critic_opt.step()
        polyak_update(self.critic, self.ema_critic)

        self.action_dist = actor.train_action(h.detach(), z.detach())
        act_loss = self.actor_criterion(self.action_dist, action, self.returns, self.values, cont)

        self.actor_opt.zero_grad()
        act_loss.backward()
        self.actor_opt.step()

    def state_dicts(self):
        self.actor.state_dict(), self.actor_opt.state_dict(), self.critic.state_dict(), self.critic_opt.state_dict()

    def load_state_dicts(self, actor_state_dict, actor_opt_state_dict, critic_state_dict, critic_opt_state_dict):
        self.actor.load_state_dict(actor_state_dict)
        self.actor_opt.load_state_dict(actor_opt_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        self.critic_opt.load_state_dict(critic_opt_state_dict)

    def log_scalars(self):
        with torch.no_grad():
            actor_criterion_log = self.actor_criterion.log_dict()
            critic_criterion_log = self.critic_criterion.log_dict()
            return {**actor_criterion_log, **critic_criterion_log}

    def log_distributions(self):
        return {
            'ac_rewards': self.rewards.detach().cpu(),
            'ac_critic_mean': self.critic_dist.mean.detach().cpu(),
            'ac_ema_critic_mean': self.ema_critic_dist.mean.detach().cpu(),
            'ac_returns': self.returns.detach().cpu(),
            'ac_values': self.values.detach().cpu(),
            'ac_actions': self.action.detach().cpu(),
            'ac_action_dist_mean': self.action_dist.mean.detach().cpu(),
            'ac_cont': self.cont.detach().cpu()
        }


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
    parser.add_argument('--log_every_n_steps', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=8 * 10 ** 4)
    parser.add_argument('--env', type=str, default='SimplerGridWorld-grab_em_all-v0')
    parser.add_argument('--env_action_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--decoder', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wandb.init(project=f"dreamerv3-{args.env.replace('/', '-')}")
    wandb.config.update(args)


    class OneHotActionWrapper(gymnasium.ActionWrapper):

        def __init__(self, env):
            super().__init__(env)

        def action(self, action: WrapperActType) -> ActType:
            return action.argmax(-1).item()


    class MaxCombineObservations(gymnasium.ObservationWrapper):

        def __init__(self, env):
            super().__init__(env)
            obs_shape = self.observation_space.shape[1:]
            self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

        def observation(self, observation: ObsType) -> WrapperObsType:
            return np.maximum(observation[0], observation[1])


    if 'ALE' in args.env:
        env = gymnasium.make(args.env)
        env = OneHotActionWrapper(env)
        env = gymnasium.wrappers.FrameStack(env, 2)
        env = MaxCombineObservations(env)
        env = gymnasium.wrappers.ResizeObservation(env, (64, 64))

    if 'SimpleGridworld' in args.env:
        env = gymnasium.make(args.env, max_episode_steps=100)
        env = gridworld.RGBImageWrapper(env)
        env = gridworld.OneHotTensorActionWrapper(env)

    if 'SimplerGridWorld' in args.env:
        env = gymnasium.make(args.env)
        env = PartialRGBObservationWrapper(env)
        env = gridworld.OneHotTensorActionWrapper(env)

    pad_action = np.zeros((1, env.action_space.n.item()))
    action_table = env.get_action_meanings()

    rssm = make_small(action_classes=env.action_space.n.item(), decoder=args.decoder).to(args.device)
    rssm_opt = Adam(rssm.parameters(), lr=args.world_model_learning_rate, eps=args.world_model_adam_epsilon)
    register_gradient_clamp(rssm, args.world_model_gradient_clipping)
    rssm_criterion = RSSMLoss()

    critic = Critic()
    actor = Actor(action_size=1, action_classes=env.action_space.n.item())
    actor_critic_trainer = ActorCriticTrainer(
        actor=actor,
        critic=critic,
        lr=args.actor_critic_learning_rate,
        adam_eps=args.actor_critic_adam_epsilon,
        grad_clip=args.actor_critic_gradient_clipping,
        device=args.device
    )

    buff = deque(maxlen=args.replay_capacity)
    gen = rollout(env, actor, rssm, pad_action)
    for i in range(400):
        buff += [next(gen)]

    # if args.resume:
    #     rssm, rssm_optim, critic, critic_optim, actor, actor_optim, steps, resume_args = \
    #         utils.load(args.resume, rssm, rssm_opt, critic, critic_opt, actor, actor_opt)
    #     print(f'resuming from step {steps} of {args.resume} with {resume_args}')

    loader = BatchLoader(device=args.device, observation_transform=replay.transform_rgb_image)

    for step in range(args.max_steps):

        buff += [next(gen)]
        train_world_model(buff, rssm, rssm_opt, rssm_criterion, step)

        obs, act, reward, cont = loader.sample(buff, batch_length=1, batch_size=args.batch_size)
        h0 = rssm.new_hidden0(args.batch_size)
        imag_h, imag_z, imag_action, imag_rewards, imag_cont = \
            rssm.imagine(h0, obs[0], reward[0], cont[0], actor, imagination_horizon=args.imagination_horizon)

        actor_critic_trainer.train(imag_h, imag_z, imag_action, imag_rewards, imag_cont)

        wandb.log(actor_critic_trainer.log_scalars(), step=step)
        wandb.log({k: wandb.Histogram(v) for k, v in actor_critic_trainer.log_distributions().items()}, step=step)

        # if step % args.log_every_n_steps == 0:
        # actor, actor_opt, critic, critic_opt = actor_critic_trainer.state_dicts()
        # utils.save(utils.run.rundir, rssm, rssm_opt, critic, critic_opt, actor,
        #            actor_opt, args, step)
        # print(f'saved model at step {step}')

        if buff[-1].is_terminal:
            latest_trajectory = replay.get_tail_trajectory(buff)
            trajectory_reward = replay.total_reward(latest_trajectory)
            trajectory_viz = viz.visualize_trajectory(latest_trajectory, action_table=action_table)
            trajectory_viz = (trajectory_viz * 255).to(dtype=torch.uint8).numpy()
            dec_trajectory = viz.decode_trajectory(rssm, latest_trajectory)
            wandb.log({
                'trajectory_reward': trajectory_reward,
                'trajectory_length': len(latest_trajectory),
                'trajectory_viz': Video(trajectory_viz),
                'trajectory_dec': Video(dec_trajectory)
            }, step=step)

            print(f'trajectory end: reward {trajectory_reward} len: {len(latest_trajectory)}')



        if step % args.log_every_n_steps == 0:
            imag_obs = rssm.decoder(imag_h, imag_z).mean
            imag_returns = actor_critic_trainer.log_distributions()['ac_returns'].to(args.device)
            imagined_trajectory_viz = viz.visualize_imagined_trajectory(
                imag_obs[:-1, :], imag_action[:-1, :], imag_rewards[:-1, :], imag_cont[:-1, :],
                imag_returns[:, :], action_table=action_table)
            imagined_trajectory_viz = (imagined_trajectory_viz * 255).to(dtype=torch.uint8).cpu().numpy()
            wandb.log({'ac_imagined_trajectory': wandb.Video(imagined_trajectory_viz)}, step=step)
