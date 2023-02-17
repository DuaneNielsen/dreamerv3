import gymnasium as gym
from gymnasium import RewardWrapper
from minigrid.wrappers import RGBImgObsWrapper

import replay
import viz
from replay import sample_batch, Step
from collections import deque
from torch.nn.functional import one_hot, mse_loss
from torchvision.transforms.functional import resize
from rssm import make_small, RSSMLoss
from critic import Critic, td_lambda, polyak_update, Actor, ActorLoss
import torch
from torch.optim import Adam
from argparse import ArgumentParser
from symlog import symlog, symexp
import copy
import utils
import wandb
from viz import make_batch_panels, make_panel
from wandb.data_types import Video


def log(obs, action, r, c, mask, obs_dist, r_dist, c_dist, z_prior, z_post, step):
    with torch.no_grad():
        batch_panel, rewards_panel, terminal_panel = \
            make_batch_panels(obs, action, reward, cont, obs_dist.mean,
                              r_dist.mean, c_dist.probs, mask,
                              symexp_on=True,
                              action_table=action_table)

        class_labels = utils.bin_labels(0., 1., num_bins=5)
        reward_gt = symexp(r[mask]).flatten().cpu()
        reward_mean = symexp(r_dist.mean[mask]).flatten().cpu()
        reward_gt_binned = utils.bin_values(reward_gt, 0., 1., num_bins=5).numpy()
        reward_mean_binned = utils.bin_values(reward_mean, 0., 1., num_bins=5).numpy()
        cont_gt = c[mask].flatten().cpu()
        cont_probs = c_dist.probs[mask].flatten().cpu()
        cont_gt_binned = utils.bin_values(cont_gt, 0., 1., 5).numpy()
        cont_probs_binned = utils.bin_values(cont_probs, 0., 1., 5).numpy()

        z_prior_argmax, z_post_argmax = z_prior.argmax(-1).flatten().cpu().numpy(), z_post.argmax(
            -1).flatten().cpu().numpy()
        z_labels = [f'{l:02d}' for l in list(range(32))]

        wandb.log({
            'batch_panel': wandb.Image(batch_panel),
            'nonzero_rewards_panel': wandb.Image(rewards_panel),
            'terminal_state_panel': wandb.Image(terminal_panel),
            'reward_gt': wandb.Histogram(reward_gt, num_bins=5),
            'reward_pred': wandb.Histogram(reward_mean, num_bins=5),
            'reward_confusion': wandb.plot.confusion_matrix(y_true=reward_gt_binned,
                                                            preds=reward_mean_binned,
                                                            class_names=class_labels),
            'cont_gt': wandb.Histogram(cont_gt, num_bins=5),
            'cont_probs': wandb.Histogram(cont_probs, num_bins=5),
            'cont_confusion': wandb.plot.confusion_matrix(y_true=cont_gt_binned,
                                                          preds=cont_probs_binned,
                                                          class_names=class_labels),
            'z_post': wandb.Histogram(z_post_argmax),
            'z_prior': wandb.Histogram(z_prior_argmax),
            'z_prior_z_post_confusion': wandb.plot.confusion_matrix(y_true=z_prior_argmax, preds=z_post_argmax,
                                                                    class_names=z_labels),
        }, step=step)


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


def random_policy(h, z):
    return one_hot(torch.randint(0, 4, [1]), 4)


def random_policy_for_world_model(h, z):
    return one_hot(torch.randint(0, 4, [z.shape[0]]), 4).to(z.device).unsqueeze(1)


def rollout(env, policy, rssm, seed=42):
    terminated, truncated = True, True
    while True:
        if terminated or truncated:
            (obs, _), r, c = env.reset(seed=seed), 0, 1.0
            obs_pre = prepro(obs).to(args.device).unsqueeze(0)
            h = rssm.new_hidden0(batch_size=1)
            z = rssm.encode_observation(h, obs_pre).mode

        a = policy.sample_action(h, z)
        yield Step(obs_pre[0].detach().cpu(), a[0].detach().cpu(), r, c)
        obs, r, terminated, truncated, _ = env.step(a[0].argmax().item())
        obs_pre = prepro(obs).to(args.device).unsqueeze(0)
        h, z = rssm.step_reality(h, obs_pre, a)
        c = 0.0 if terminated else 1.0
        if terminated or truncated:
            yield Step(obs_pre[0].detach().cpu(), None, r, c)


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
    action_table = {0: 'left', 1: 'right', 2: 'forward', 3: 'pickup'}
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = RGBImgObsWrapper(env, tile_size=13)


    class RewardOneOrZeroOnlyWrapper(RewardWrapper):
        def __init__(self, env):
            super().__init__(env)

        def reward(self, reward):
            return 0. if reward == 0. else 1.


    env = RewardOneOrZeroOnlyWrapper(env)

    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), 4)

    rssm = make_small(action_classes=4).to(args.device)
    rssm_opt = Adam(rssm.parameters(), lr=args.world_model_learning_rate, eps=args.world_model_adam_epsilon)
    for p in rssm.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -args.world_model_gradient_clipping, args.world_model_gradient_clipping))
    criterion = RSSMLoss()

    critic = Critic()
    ema_critic = copy.deepcopy(critic)
    actor = Actor(action_size=1, action_classes=4)
    critic, ema_critic, actor = critic.to(args.device), ema_critic.to(args.device), actor.to(args.device)
    critic_opt = Adam(critic.parameters(), lr=args.actor_critic_learning_rate, eps=args.actor_critic_adam_epsilon)
    actor_opt = Adam(actor.parameters(), lr=args.actor_critic_learning_rate, eps=args.actor_critic_adam_epsilon)
    for p in critic.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -args.actor_critic_gradient_clipping, args.actor_critic_gradient_clipping))
    for p in actor.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -args.actor_critic_gradient_clipping, args.actor_critic_gradient_clipping))
    actor_criterion = ActorLoss()

    buff = deque(maxlen=args.replay_capacity)
    gen = rollout(env, actor, rssm)
    for i in range(20):
        buff += [next(gen)]

    steps = 0
    training_time = utils.StopWatch()
    logging_time = utils.StopWatch()

    if args.resume:
        rssm, rssm_opt, steps, resume_args, loss = utils.load(args.resume, rssm, rssm_opt)
        print(f'resuming from step {steps} of {args.resume} with {resume_args}')

    while True:

        training_time.go()

        # sample batch from replay buffer
        buff += [next(gen)]

        obs, act, reward, cont, mask = sample_batch(buff, args.batch_length, args.batch_size, pad_state, pad_action)
        obs, act, reward, cont, mask = obs.to(args.device), act.to(args.device), reward.to(args.device), cont.to(
            args.device), mask.to(args.device)
        obs = symlog(obs)
        reward = symlog(reward)

        # train world model
        h0 = torch.zeros(args.batch_size, rssm.h_size, device=args.device)
        obs_dist, rew_dist, cont_dist, z_prior, z_post = rssm(obs, act, h0)
        loss = criterion(obs, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post)

        rssm_opt.zero_grad()
        loss.backward()
        rssm_opt.step()

        training_time.pause()

        with torch.no_grad():
            wandb.log(criterion.loss_dict(), step=steps)

            if steps % args.log_every_n_steps == 0:
                logging_time.go()
                log(obs, act, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post, steps)
                utils.save(utils.run.rundir, rssm, rssm_opt, args, steps, loss.item())
                logging_time.pause()

        training_time.go()

        # sample 1 step from replay buffer and imagine trajectory under policy
        obs, act, reward, cont, mask = sample_batch(buff, 1, args.batch_size, pad_state, pad_action)
        obs, reward, cont, mask = obs.to(args.device), reward.to(args.device), cont.to(args.device), mask.to(
            args.device)
        h0 = rssm.new_hidden0(args.batch_size)
        h, z, a, reward_symlog, cont = rssm.imagine(h0, obs[0], reward[0], cont[0], actor,
                                                    imagination_horizon=args.imagination_horizon, epsilon=0.01)

        h, z = h.detach(), z.detach()

        # compute values based on imagined trajectory and update critic
        values = ema_critic(h, z)
        value_targets = td_lambda(reward_symlog, cont, values)
        value_preds = critic(h, z)
        critic_loss = mse_loss(value_preds, value_targets)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        polyak_update(critic, ema_critic)

        actor_logits = actor(h, z)
        actor_loss = actor_criterion(actor_logits, a, value_preds)

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        training_time.pause()

        with torch.no_grad():
            wandb.log(actor_criterion.loss_dict(), step=steps)
            wandb.log({'critic_loss': critic_loss.item()}, step=steps)

            if replay.is_trajectory_end(buff[-1]):
                latest_trajectory = replay.get_tail_trajectory(buff)
                trajectory_reward = replay.total_reward(latest_trajectory)
                trajectory_viz = viz.visualize_trajectory(latest_trajectory, pad_action, symexp_on=False, action_table=action_table)
                trajectory_viz = (trajectory_viz * 255).to(dtype=torch.uint8).numpy()
                wandb.log({
                    'trajectory_reward': trajectory_reward,
                    'trajectory_length': len(latest_trajectory),
                    'trajectory_viz': Video(trajectory_viz)
                }, step=steps)
                print(f'trajectory end: reward {trajectory_reward} len: {len(latest_trajectory)}')

            if steps % args.log_every_n_steps == 0:
                logging_time.go()

                decoded_obs = rssm.decoder(h, z).mean
                mask = mask.repeat(1 + args.imagination_horizon, 1, 1)
                imagination_panel = make_panel(decoded_obs[:, 0:2], a[:, 0:2], reward_symlog[:, 0:2], cont[:, 0:2],
                                               mask[:, 0:2], value_preds[:, 0:2], action_table=action_table, nrow=16)
                wandb.log({'imagination_panel': wandb.Image(imagination_panel)}, step=steps)

                logging_time.pause()
                print(
                    f'logged step {steps} training_time {training_time.total_time} logging_time {logging_time.total_time}')
                training_time.reset()
                logging_time.reset()

        steps += 1
