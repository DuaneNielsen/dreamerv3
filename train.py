import gymnasium
from torch.nn.functional import one_hot
from gridworlds.simpler_gridworld import PartialRGBObservationWrapper

import replay
from gridworlds.wrappers import OneHotActionWrapper, MaxCombineObservations
from replay import BatchLoader, Step
from collections import deque

from rssm import RSSMLoss
from actor_critic import Actor, Critic, ActorCriticTrainer
import torch
import torch.nn as nn
from torch.optim import Adam
from argparse import ArgumentParser
import utils
import wandb

from train_grid import viz_reward_pred, viz_cont_pred
from utils import register_gradient_clamp
from config import make
from viz import visualize_buff, VizStep, ValueHook
from wandb.data_types import Video
import viz
import gridworlds.gridworld as gridworld
import numpy as np
from torchvision.utils import make_grid
from copy import deepcopy
from math import ceil
import pathlib


class WorldModelTrainer:
    def __init__(self, rssm, lr, adam_eps, grad_clip, action_meanings=None):
        super().__init__()
        self.rssm = rssm
        self.opt = Adam(rssm.parameters(), lr=lr, eps=adam_eps)
        self.rssm_criterion = RSSMLoss()
        register_gradient_clamp(self.rssm, grad_clip)
        self.action_meanings = action_meanings

    def train_model(self, obs, action, reward, cont):
        self.obs, self.action, self.reward, self.cont = obs, action, reward, cont

        h0 = rssm.new_hidden0(args.batch_size)
        self.obs_dist, self.reward_dist, self.cont_dist, self.z_prior, self.z_post = rssm(obs, action, h0)
        rssm_loss = self.rssm_criterion(self.obs, self.reward, self.cont, self.obs_dist, self.reward_dist, self.cont_dist, self.z_prior, self.z_post)

        self.opt.zero_grad()
        rssm_loss.backward()
        self.opt.step()

    def log_scalars(self):
        return {**self.rssm_criterion.loss_dict()}

    def log_images(self, vizualiser):
        with torch.no_grad():
            viz_batch_gt_buff = replay.unstack_batch(self.obs[0:8, 0:8], self.action[0:8, 0:8], self.reward[0:8, 0:8], self.cont[0:8, 0:8])
            viz_batch_pred_buff = replay.unstack_batch(self.obs_dist.mean[0:8, 0:8], self.action[0:8, 0:8], self.reward_dist.mean[0:8, 0:8],
                                                       self.cont_dist.mean[0:8, 0:8])
            viz_batch_gt = visualize_buff(viz_batch_gt_buff, vizualiser)
            viz_batch_pred = visualize_buff(viz_batch_pred_buff, vizualiser)
            batch_panel = torch.cat((make_grid(torch.from_numpy(viz_batch_gt)), make_grid(torch.from_numpy(viz_batch_pred))),
                                    dim=2).numpy().transpose(1, 2, 0)
            viz_gt_buff = replay.unstack_batch(self.obs, self.action, self.reward, self.cont)
            viz_pred_buff = replay.unstack_batch(self.obs_dist.mean, self.action, self.reward_dist.mean, self.cont_dist.mean)

            def side_by_side(left_buff, right_buff, filter_lam):
                side_by_side = []
                for gt, pred in zip(left_buff, right_buff):
                    if filter_lam(gt) or filter_lam(pred):
                        side_by_side += [
                            np.concatenate(
                                (vizualiser(gt),
                                 vizualiser(pred)), axis=1)
                        ]
                if len(side_by_side) == 0:
                    return np.zeros((64, 64, 3), dtype=np.uint8)
                side_by_side = np.stack(side_by_side)
                return make_grid(torch.from_numpy(side_by_side)).permute(1, 2, 0).numpy()

            rewards_panel = side_by_side(viz_gt_buff, viz_pred_buff, lambda st: abs(st.reward) > 0.1)
            terminal_panel = side_by_side(viz_gt_buff, viz_pred_buff, lambda st: st.cont < 0.05)

            return {
                'wm_image_batch': batch_panel,
                'wm_image_reward': rewards_panel,
                'wm_image_terminal': terminal_panel
            }

    def log_distributions(self):
        return {
            'wm_reward_gt': self.reward.flatten().detach().cpu().numpy(),
            'wm_reward_pred': self.reward_dist.mean.flatten().detach().cpu().numpy(),
            'wm_cont_gt': self.cont.flatten().detach().cpu().numpy(),
            'wm_cont_probs': self.cont_dist.probs.flatten().detach().cpu().numpy(),
            'wm_z_prior': self.z_prior.argmax(-1).flatten().detach().cpu().numpy(),
            'wm_z_post': self.z_prior.argmax(-1).flatten().detach().cpu().numpy(),
        }

    def state_dict(self):
        return {
            'rssm_state_dict': self.rssm.state_dict(),
            'opt_state_dict': self.opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.rssm.load_state_dict(state_dict['rssm_state_dict'])
        self.opt.load_state_dict(state_dict['opt_state_dict'])


class Rollout:
    def __init__(self, env, actor, rssm, pad_action):
        self.actor = actor
        self.rssm = rssm
        self.env = env
        self.pad_action = pad_action
        self.h = None
        self.terminated, self.truncated = True, False
        self.action = None
        self.obs = None

    def step(self, action=None):
        with torch.no_grad():
            if self.terminated or self.truncated:
                self.obs, info = self.env.reset()
                self.h = self.rssm.new_hidden0(batch_size=1)
                z = self.rssm.encode_observation(self.h, torch.from_numpy(self.obs).unsqueeze(0)).mode
                self.action = action if action is not None else self.actor(self.h, z).sample()
                self.terminated, self.truncated = False, False
                r_pred = self.rssm.reward_pred(self.h, z)
                c_pred = self.rssm.continue_pred(self.h, z)
                obs_pred = self.rssm.decoder(self.h, z)
                return Step(self.obs, self.action[0].detach().cpu().numpy(), 0., self.terminated, self.truncated,
                            reward_pred=r_pred.mean.detach().cpu().numpy(),
                            cont_pred=c_pred.probs.detach().cpu().numpy(),
                            obs_pred=obs_pred.mean[0].detach().cpu().numpy())

            next_obs, reward, self.terminated, self.truncated, info = self.env.step(self.action[0].cpu())
            self.h, z, r_pred, c_pred, obs_pred = self.rssm.step_reality(self.h, torch.from_numpy(self.obs).unsqueeze(0), self.action)
            self.action = action if action is not None else self.actor(self.h, z).sample()
            self.obs = next_obs

            return Step(self.obs, self.action[0].detach().cpu().numpy(), reward, self.terminated, self.truncated,
                        reward_pred=r_pred.mean.detach().cpu().numpy(), cont_pred=c_pred.probs.detach().cpu().numpy(),
                        obs_pred=obs_pred.mean[0].detach().cpu().numpy())


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_size', type=str, choices=['extra_small', 'small', 'medium'], default='medium')
    parser.add_argument('--train_ratio', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_length', type=int, default=64)
    parser.add_argument('--imagination_horizon', type=int, default=15)
    parser.add_argument('--imagination_batch_size', type=int, default=32)
    parser.add_argument('--replay_capacity', type=int, default=10 ** 6)
    parser.add_argument('--world_model_learning_rate', type=float, default=1e-4)
    parser.add_argument('--world_model_adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--world_model_gradient_clipping', type=float, default=1000.)
    parser.add_argument('--actor_critic_learning_rate', type=float, default=3e-5)
    parser.add_argument('--actor_critic_adam_epsilon', type=float, default=1e-5)
    parser.add_argument('--actor_critic_gradient_clipping', type=float, default=100.)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_every_n_steps', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=8 * 10 ** 4)
    parser.add_argument('--env', type=str, default='SimplerGridWorld-grab_em_all-v0')
    parser.add_argument('--env_action_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--decoder', type=str, default=None)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--full_obs', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    project_name = f"dreamerv3-{args.env.replace('/', '-')}"
    project_name = project_name + '-dev' if args.dev else project_name
    wandb.init(project=project_name)
    run_dir = 'runs/' + project_name + '/' + utils.next_run()
    pathlib.Path(run_dir).mkdir(exist_ok=True, parents=True)
    vars(args)['run_dir'] = run_dir
    wandb.config.update(args)

    if 'ALE' in args.env:
        env = gymnasium.make(args.env, frameskip=4)
        env = OneHotActionWrapper(env)
        env = gymnasium.wrappers.FrameStack(env, 2)
        env = MaxCombineObservations(env)
        env = gymnasium.wrappers.ResizeObservation(env, (64, 64))

    if 'SimpleGridworld' in args.env:
        env = gymnasium.make(args.env, max_episode_steps=100)
        env = gridworld.RGBImageWrapper(env)
        env = gridworld.OneHotTensorActionWrapper(env)

    if 'SimplerGridWorld' in args.env and args.full_obs:
        env = gymnasium.make(args.env)
        import gridworlds.simpler_gridworld

        env = gridworlds.simpler_gridworld.RGBObservationWrapper(env)
        env = gridworld.OneHotTensorActionWrapper(env)
    elif 'SimplerGridWorld' in args.env:
        env = gymnasium.make(args.env)
        env = PartialRGBObservationWrapper(env)
        env = gridworld.OneHotTensorActionWrapper(env)

    pad_action = np.zeros((1, env.action_space.n.item()))
    pad_action[:, 0] = 1.
    action_table = env.get_action_meanings()
    visualizer_traj = VizStep(action_meanings=action_table)


    def vizualize_obs_pred(step):
        return viz.normalized_image(step.info['obs_pred'])


    visualizer_traj.add_hook(vizualize_obs_pred)
    visualizer_traj.add_hook(viz_reward_pred)
    visualizer_traj.add_hook(viz_cont_pred)
    visualizer_traj.add_hook(ValueHook())

    vizualizer = VizStep(action_meanings=action_table)
    vizualizer.add_hook(ValueHook())

    rssm, actor, critic = make(args.model_size, action_size=1, action_classes=env.action_space.n.item(), decoder=args.decoder)
    rssm, actor, critic = rssm.to(args.device), actor.to(args.device), critic.to(args.device)

    world_model_trainer = WorldModelTrainer(rssm,
                                            lr=args.world_model_learning_rate,
                                            adam_eps=args.world_model_adam_epsilon,
                                            grad_clip=args.world_model_gradient_clipping)

    rssm_policy = deepcopy(rssm).cpu()
    actor_policy = deepcopy(actor).cpu()

    actor_critic_trainer = ActorCriticTrainer(
        actor=actor,
        critic=critic,
        lr=args.actor_critic_learning_rate,
        adam_eps=args.actor_critic_adam_epsilon,
        grad_clip=args.actor_critic_gradient_clipping,
        device=args.device
    )

    buff = deque(maxlen=args.replay_capacity)
    gen = Rollout(env, actor_policy, rssm_policy, pad_action)

    for i in range(args.batch_length * args.batch_size):
        random_action = one_hot(torch.tensor([env.action_space.sample()]), env.action_space.n).unsqueeze(0)
        buff += [gen.step(random_action)]

    if args.resume:
        checkpoint = torch.load(args.resume)
        world_model_trainer.load_state_dict(checkpoint['world_model_state_dict'])
        actor_critic_trainer.load_state_dict(checkpoint['actor_critic_trainer_state_dict'])
        step, run_args = checkpoint['step'], checkpoint['args']
        print(f'resuming from step {step} of {args.resume} with {run_args}')

    loader = BatchLoader(device=args.device, observation_transform=replay.symlog_rgb_image)

    for step in range(args.max_steps):

        buff += [gen.step()]
        if buff[-1].is_terminal:
            latest_trajectory = replay.get_tail_trajectory(buff)
            trajectory_reward = replay.total_reward(latest_trajectory)
            trajectory_reward_pred = replay.sum_key(latest_trajectory, 'reward_pred')
            trajectory_reward_mse = replay.reward_mse(latest_trajectory)
            trajectory_cont_mse = replay.cont_mse(latest_trajectory)
            trajectory_obs_mse = replay.observation_mse(latest_trajectory)

            trajectory_viz = viz.visualize_buff(latest_trajectory, visualizer=visualizer_traj)

            wandb.log({
                'trajectory_obs_mse': trajectory_obs_mse,
                'trajectory_cont_mse': trajectory_cont_mse,
                'trajectory_reward_mse': trajectory_reward_mse,
                'trajectory_reward_pred': trajectory_reward_pred,
                'trajectory_reward': trajectory_reward,
                'trajectory_length': len(latest_trajectory),
                'trajectory_viz': Video(trajectory_viz)
            }, step=step)

            print(f'trajectory end: reward {trajectory_reward} reward pred: {trajectory_reward_pred} len: {len(latest_trajectory)}')

        if step % ceil(args.imagination_horizon * args.imagination_batch_size / args.train_ratio) != 0:
            continue

        obs, action, reward, cont = loader.sample(buff, args.batch_length, args.batch_size)
        world_model_trainer.train_model(obs, action, reward, cont)

        obs, act, reward, cont = loader.sample(buff, batch_length=1, batch_size=args.imagination_batch_size)
        h0 = rssm.new_hidden0(args.imagination_batch_size)
        imag_h, imag_z, imag_action, imag_rewards, imag_cont = \
            rssm.imagine(h0, obs[0], reward[0], cont[0], actor, imagination_horizon=args.imagination_horizon)

        actor_critic_trainer.train_step(imag_h, imag_z, imag_action, imag_rewards, imag_cont)

        rssm_policy.load_state_dict(rssm.state_dict())
        actor_policy.load_state_dict(actor.state_dict())

        wandb.log({**actor_critic_trainer.log_scalars(), **world_model_trainer.log_scalars()}, step=step)
        wandb.log({k: wandb.Histogram(v) for k, v in actor_critic_trainer.log_distributions().items()}, step=step)
        wandb.log({k: wandb.Histogram(v) for k, v in world_model_trainer.log_distributions().items()}, step=step)

        if step % args.log_every_n_steps == 0:
            print('logging step')
            with torch.no_grad():
                vizualizer = VizStep(action_meanings=action_table)
                wandb.log({k: wandb.Image(v) for k, v in world_model_trainer.log_images(vizualizer).items()}, step=step)

                imag_obs = rssm.decoder(imag_h, imag_z).mean
                imag_returns = actor_critic_trainer.log_distributions()['ac_returns'].to(args.device)
                imagined_trajectory_viz = viz.visualize_imagined_trajectories(
                    imag_obs[:-1, :], imag_action[:-1, :], imag_rewards[:-1, :], imag_cont[:-1, :],
                    imag_returns[:, :], visualizer=vizualizer)

                wandb.log({'ac_imagined_trajectory': wandb.Video(imagined_trajectory_viz)}, step=step)

                torch.save({
                    'args': args,
                    'step': step,
                    'actor_critic_trainer_state_dict': actor_critic_trainer.state_dict(),
                    'world_model_state_dict': world_model_trainer.state_dict(),
                }, run_dir + '/checkpoint.pt')
