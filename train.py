import gymnasium
from envs.simpler_gridworld import PartialRGBObservationWrapper

import replay
from envs.wrappers import OneHotActionWrapper, MaxCombineObservations
from replay import BatchLoader, Step
from collections import deque
from torchvision.transforms.functional import to_tensor
from rssm import make_small, RSSMLoss
from actor_critic import Actor, Critic, ActorCriticTrainer
import torch
from torch.optim import Adam
from argparse import ArgumentParser
import utils
import wandb
from utils import register_gradient_clamp
from viz import make_batch_panels
from wandb.data_types import Video
import viz
import envs.gridworld as gridworld
import numpy as np


def rollout(env, actor, rssm, pad_action):
    obs, info = env.reset()
    h = rssm.new_hidden0(batch_size=1)
    obs_tensor = to_tensor(obs).to(rssm.device).unsqueeze(0)
    z = rssm.encode_observation(h, obs_tensor).mode
    reward, terminated, truncated = 0.0, False, False

    while True:
        action = actor(h, z).sample()
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
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    project_name = f"dreamerv3-{args.env.replace('/', '-')}"
    project_name = project_name + '-dev' if args.dev else project_name
    wandb.init(project=project_name)
    run_dir = utils.next_run()
    wandb.config.update(args)

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

    if args.resume:
        checkpoint = torch.load(args.resume)
        rssm.load_state_dict(checkpoint['rssm_state_dict'])
        rssm_opt.load_state_dict(checkpoint['rssm_opt_state_dict'])
        actor_critic_trainer.load_state_dict(checkpoint['actor_critic_trainer_state_dict'])
        step, run_args = checkpoint['step'], checkpoint['args']
        print(f'resuming from step {step} of {args.resume} with {run_args}')

    loader = BatchLoader(device=args.device, observation_transform=replay.transform_rgb_image)

    for step in range(args.max_steps):

        buff += [next(gen)]
        train_world_model(buff, rssm, rssm_opt, rssm_criterion, step)

        imag_batch_size = int(args.train_ratio / args.imagination_horizon)
        obs, act, reward, cont = loader.sample(buff, batch_length=1, batch_size=imag_batch_size)
        h0 = rssm.new_hidden0(imag_batch_size)
        imag_h, imag_z, imag_action, imag_rewards, imag_cont = \
            rssm.imagine(h0, obs[0], reward[0], cont[0], actor, imagination_horizon=args.imagination_horizon)

        actor_critic_trainer.train_step(imag_h, imag_z, imag_action, imag_rewards, imag_cont)

        wandb.log(actor_critic_trainer.log_scalars(), step=step)
        wandb.log({k: wandb.Histogram(v) for k, v in actor_critic_trainer.log_distributions().items()}, step=step)

        if buff[-1].is_terminal:
            latest_trajectory = replay.get_tail_trajectory(buff)

            trajectory_reward = replay.total_reward(latest_trajectory)
            trajectory_viz = viz.visualize_buff(latest_trajectory, action_meanings=action_table)
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

            torch.save({
                'args': args,
                'step': step,
                'actor_critic_trainer_state_dict': actor_critic_trainer.state_dict(),
                'rssm_state_dict': rssm.state_dict(),
                'rssm_opt_state_dict': rssm_opt.state_dict()
            }, run_dir + '/checkpoint.pt')
