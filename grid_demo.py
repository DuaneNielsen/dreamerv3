import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from replay import sample_batch, Step, get_trajectories
from collections import deque
from torch.nn.functional import one_hot
from torchvision.transforms.functional import resize
from rssm import make_small, RSSMLoss
import torch
from torch.optim import Adam
from argparse import ArgumentParser
from symlog import symlog, symexp
from time import time
import utils
import wandb
from viz import make_trajectory_panel, make_batch_panels


def log(obs, action, r, c, mask, obs_dist, r_dist, c_dist, z_prior, z_post, step):
    with torch.no_grad():
        batch_panel, rewards_panel, terminal_panel = \
            make_batch_panels(obs, action, reward, cont, obs_dist.mean,
                              r_dist.mean, c_dist.probs, mask,
                              sym_exp_on=True,
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


def generate_and_log_trajectory_on_world_model(rssm, policy, step):
    imagination_gen = rollout_on_world_model(rssm, policy)
    imagine_buffer = [next(imagination_gen) for step in range(16)]
    trajectories = get_trajectories(imagine_buffer)
    panel = make_trajectory_panel(trajectories[0], pad_action.to(rssm.device), action_table=action_table)
    wandb.log({
        'imagined_obs': wandb.Image(panel)
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
    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), 4)

    policy = random_policy
    buff = deque(maxlen=args.replay_capacity)
    gen = rollout(env, policy)
    for i in range(500):
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

        with torch.no_grad():
            wandb.log(criterion.loss_dict(), step=steps)
            steps += 1

            if steps % args.log_every_n_steps == 0:
                end_train = time()
                log(obs, act, reward, cont, mask, obs_dist, rew_dist, cont_dist, z_prior, z_post, steps)
                utils.save(utils.run.rundir, rssm, opt, args, steps, loss.item())

                generate_and_log_trajectory_on_world_model(rssm, policy, steps)

                end_plot = time()
                print(f'step: {steps} train time: {end_train - start_t} plot time: {end_plot - end_train}')
                start_t = time()
