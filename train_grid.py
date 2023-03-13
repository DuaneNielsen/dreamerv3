import gymnasium

import replay
from gridworlds.wrappers import OneHotActionWrapper, MaxCombineObservations
from replay import BatchLoader, Step
from collections import deque
from torchvision.transforms.functional import to_tensor
from rssm import make_small, RSSMLoss, make_small_gridworld
from actor_critic import Actor, Critic, ActorCriticTrainer
import torch
import torch.nn as nn
from torch.optim import Adam
from argparse import ArgumentParser
import utils
import wandb
from utils import register_gradient_clamp
from viz import visualize_buff, VizStep, ValueHook
from wandb.data_types import Video
import viz
import numpy as np
from torchvision.utils import make_grid
import gridworlds.simpler_gridworld as gridworld
from gridworlds.gridworld import OneHotTensorActionWrapper


def identity(obs):
    return obs


def rollout(env, actor, rssm, pad_action):
    obs, info = env.reset()
    h = rssm.new_hidden0(batch_size=1)
    obs_tensor = torch.from_numpy(obs[0]).to(rssm.device).unsqueeze(0)
    z = rssm.encode_observation(h, obs_tensor).mode
    r_pred, c_pred, obs_pred = rssm.reward_pred(h, z), rssm.continue_pred(h, z), rssm.decoder(h, z)
    reward, terminated, truncated = 0.0, False, False

    while True:
        action = actor(h, z).sample()
        yield Step(obs[0], action[0].detach().cpu().numpy(), reward, terminated, truncated,
                   reward_pred=r_pred.mean.detach().cpu().numpy(), cont_pred=c_pred.probs.detach().cpu().numpy(),
                   obs_pred=obs_pred.mode[0].detach().cpu().numpy())

        obs, reward, terminated, truncated, info = env.step(action[0].cpu())
        obs_tensor = torch.from_numpy(obs[0]).to(rssm.device).unsqueeze(0)
        h, z, r_pred, c_pred, obs_pred = rssm.step_reality(h, obs_tensor, action)

        if terminated or truncated:
            yield Step(obs[0], pad_action, reward, terminated, truncated,
                       reward_pred=r_pred.mean.detach().cpu().numpy(), cont_pred=c_pred.probs.detach().cpu().numpy(),
                       obs_pred=obs_pred.mode[0].detach().cpu().numpy())

            obs, info = env.reset()
            obs_tensor = torch.from_numpy(obs[0]).to(rssm.device).unsqueeze(0)
            h = rssm.new_hidden0(batch_size=1)
            z = rssm.encode_observation(h, obs_tensor).mode
            reward, terminated, truncated = 0.0, False, False


class WorldModelTrainer(nn.Module):
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
        rssm_loss = self.rssm_criterion(self.obs, self.reward, self.cont,
                                        self.obs_dist, self.reward_dist, self.cont_dist, self.z_prior, self.z_post)

        self.opt.zero_grad()
        rssm_loss.backward()
        self.opt.step()

    def log_scalars(self):
        return {**self.rssm_criterion.loss_dict()}

    def log_images(self, vizualiser):
        with torch.no_grad():
            viz_batch_gt_buff = replay.unstack_batch(self.obs[0:8, 0:8], self.action[0:8, 0:8], self.reward[0:8, 0:8],
                                                     self.cont[0:8, 0:8], inverse_obs_transform=identity)
            viz_batch_pred_buff = replay.unstack_batch(self.obs_dist.mode[0:8, 0:8], self.action[0:8, 0:8],
                                                       self.reward_dist.mean[0:8, 0:8], self.cont_dist.mean[0:8, 0:8],
                                                       inverse_obs_transform=identity)
            viz_batch_gt = visualize_buff(viz_batch_gt_buff, vizualiser)
            viz_batch_pred = visualize_buff(viz_batch_pred_buff, vizualiser)
            batch_panel = torch.cat(
                (make_grid(torch.from_numpy(viz_batch_gt)), make_grid(torch.from_numpy(viz_batch_pred))),
                dim=2).numpy().transpose(1, 2, 0)
            viz_gt_buff = replay.unstack_batch(self.obs, self.action, self.reward, self.cont,
                                               inverse_obs_transform=identity)
            viz_pred_buff = replay.unstack_batch(self.obs_dist.mode, self.action, self.reward_dist.mean,
                                                 self.cont_dist.mean, inverse_obs_transform=identity)

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


class VizGridPosStep(VizStep):
    def __init__(self, action_meanings, env_name):
        super().__init__(action_meanings=action_meanings, hw=(64, 64))
        self.env_name = env_name
        self.grid, self.grid_steps, _, _ = gridworld.make_grid(self.env_name)

    def observation(self, step):
        obs = np.clip(step.observation, a_min=0, a_max=15), self.grid, gridworld.yellow

        return gridworld.render(obs, (64, 64))

    def __call__(self, step):
        images = [self.observation(step), self.action(step), self.reward(step), self.cont(step)]
        for viz_f in self.hooks:
            images += [viz_f(step)]
        image = np.concatenate(images, 0)
        return image.transpose((2, 0, 1))


class ObsPredHook:
    def __init__(self, env_name):
        self.env_name = env_name
        self.grid, self.grid_steps, _, _ = gridworld.make_grid(self.env_name)

    def __call__(self, step):
        if 'obs_pred' in step.info:
            obs = np.clip(step.info['obs_pred'], a_min=0, a_max=15), self.grid, gridworld.yellow
            image = gridworld.render(obs, (64, 64))
        else:
            image = np.zeros((1, 64, 3), dtype=np.uint8)
        return image


def viz_reward_pred(step):
    if 'reward_pred' in step.info:
        return viz.viz_reward(step.info['reward_pred'])
    else:
        return np.zeros((1, 64, 3), dtype=np.uint8)


def viz_cont_pred(step):
    if 'cont_pred' in step.info:
        return viz.viz_reward(step.info['cont_pred'])
    else:
        return np.zeros((1, 64, 3), dtype=np.uint8)


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
    project_name = f"dreamerv3-{args.env.replace('/', '-')}-pos_grid"
    project_name = project_name + '-dev' if args.dev else project_name
    wandb.init(project=project_name)
    run_dir = utils.next_run()
    wandb.config.update(args)

    env = gymnasium.make(args.env)
    env = OneHotTensorActionWrapper(env)

    action_table = env.get_action_meanings()
    vizualizer = VizGridPosStep(action_meanings=env.get_action_meanings(), env_name='grab_em_all')
    visualizer_traj = VizGridPosStep(action_meanings=action_table, env_name='grab_em_all')
    visualizer_traj.add_hook(ObsPredHook(env_name='grab_em_all'))
    visualizer_traj.add_hook(viz_reward_pred)
    visualizer_traj.add_hook(viz_cont_pred)
    visualizer_traj.add_hook(ValueHook())

    pad_action = np.zeros((1, env.action_space.n.item()))
    pad_action[:, 0] = 1.

    rssm = make_small_gridworld(action_classes=env.action_space.n.item()).to(args.device)

    world_model_trainer = WorldModelTrainer(rssm,
                                            lr=args.world_model_learning_rate,
                                            adam_eps=args.world_model_adam_epsilon,
                                            grad_clip=args.world_model_gradient_clipping)
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
        world_model_trainer.load_state_dict(checkpoint['world_model_state_dict'])
        actor_critic_trainer.load_state_dict(checkpoint['actor_critic_trainer_state_dict'])
        step, run_args = checkpoint['step'], checkpoint['args']
        print(f'resuming from step {step} of {args.resume} with {run_args}')

    loader = BatchLoader(device=args.device, observation_transform=None)

    for step in range(args.max_steps):

        buff += [next(gen)]
        obs, action, reward, cont = loader.sample(buff, args.batch_length, args.batch_size)
        world_model_trainer.train_model(obs, action, reward, cont)

        imag_batch_size = int(args.train_ratio / args.imagination_horizon)
        obs, act, reward, cont = loader.sample(buff, batch_length=1, batch_size=imag_batch_size)
        h0 = rssm.new_hidden0(imag_batch_size)
        imag_h, imag_z, imag_action, imag_rewards, imag_cont = \
            rssm.imagine(h0, obs[0], reward[0], cont[0], actor, imagination_horizon=args.imagination_horizon)

        actor_critic_trainer.train_step(imag_h, imag_z, imag_action, imag_rewards, imag_cont)

        wandb.log({**actor_critic_trainer.log_scalars(), **world_model_trainer.log_scalars()}, step=step)
        wandb.log({k: wandb.Histogram(v) for k, v in actor_critic_trainer.log_distributions().items()}, step=step)
        wandb.log({k: wandb.Histogram(v) for k, v in world_model_trainer.log_distributions().items()}, step=step)


        def prepro_pos(obs):
            return torch.from_numpy(obs)


        if buff[-1].is_terminal:
            latest_trajectory = replay.get_tail_trajectory(buff)
            trajectory_reward = replay.total_reward(latest_trajectory)
            trajectory_reward_pred = replay.sum_key(latest_trajectory, 'reward_pred')
            trajectory_viz = viz.visualize_buff(latest_trajectory, visualizer=visualizer_traj)

            wandb.log({
                'trajectory_reward': trajectory_reward,
                'trajectory_reward_pred': trajectory_reward_pred,
                'trajectory_length': len(latest_trajectory),
                'trajectory_viz': Video(trajectory_viz)
            }, step=step)

            print(
                f'trajectory end:  len: {len(latest_trajectory)} reward {trajectory_reward} reward_pred: {trajectory_reward_pred}')

    if step % args.log_every_n_steps == 0:
        wandb.log({k: wandb.Image(v) for k, v in world_model_trainer.log_images(vizualizer).items()}, step=step)

        imag_obs = rssm.decoder(imag_h, imag_z).mode
        imag_returns = actor_critic_trainer.log_distributions()['ac_returns'].to(args.device)
        imagined_trajectory_viz = viz.visualize_imagined_trajectories(
            imag_obs[:-1, :], imag_action[:-1, :], imag_rewards[:-1, :], imag_cont[:-1, :],
            imag_returns[:, :], visualizer=visualizer_traj, inverse_obs_transform=identity)

        wandb.log({'ac_imagined_trajectory': wandb.Video(imagined_trajectory_viz)}, step=step)

        torch.save({
            'args': args,
            'step': step,
            'actor_critic_trainer_state_dict': actor_critic_trainer.state_dict(),
            'world_model_state_dict': world_model_trainer.state_dict(),
        }, run_dir + '/checkpoint.pt')
