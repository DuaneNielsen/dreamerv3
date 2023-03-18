import pathlib

import torch
import torch.nn as nn
from torch.optim import Adam
from dists import TwoHotSymlog, OneHotCategoricalStraightThru
from blocks import MLPBlock
from utils import register_gradient_clamp
from copy import deepcopy


class Critic(nn.Module):
    def __init__(self, h_size=512, z_size=32, z_classes=32, mlp_size=512, mlp_layers=2):
        super().__init__()

        self.out_layer = nn.Linear(mlp_size, 256, bias=False)
        self.out_layer.weight.data.zero_()
        self.critic = torch.nn.Sequential(
            nn.Linear(h_size + z_size * z_classes, mlp_size, bias=False),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers)],
            self.out_layer
        )

    def forward(self, h, z):
        h_z_flat = torch.cat((h, z.flatten(-2)), dim=-1)
        return TwoHotSymlog(self.critic(h_z_flat))


class Actor(nn.Module):
    def __init__(self, action_size, action_classes, h_size=512, z_size=32, z_classes=32, mlp_size=512, mlp_layers=2):
        super().__init__()
        self.action_size = action_size
        self.action_classes = action_classes
        self.output = nn.Linear(mlp_size, action_size * action_classes, bias=False)
        self.critic = torch.nn.Sequential(
            nn.Linear(h_size + z_size * z_classes, mlp_size, bias=False),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers)],
            self.output
        )

    def forward(self, h, z):
        h_z_flat = torch.cat((h, z.flatten(-2)), dim=-1)
        action_flat = self.critic(h_z_flat)
        action_logits = action_flat.unflatten(-1, (self.action_size, self.action_classes))
        return OneHotCategoricalStraightThru(logits=action_logits)


def score(reward, cont, value, discount=0.997, lamb_da=0.95):
    """
    params: reward [T, N, 1]
    params: cont [T,N,1]
    params: value[T,N,1]
    returns
        reward[T-1, N, 1]
        ret[T-1,N, 1]
        value[T-1,N, 1]

                              initial                 terminal
           observation        obs0    obs1    obs2    obs3
           act = policy(obs)  act0    act1    act2    act3
           reward              0.0    rew1    rew2    rew3
           cont                1.0    con1    con2    con3
           val = critic(obs)  val0    val1    val2    val3

                                     │
                                     ▼

                              initial                                                             terminal

           observation        obs0                   obs1                  obs2                   obs3
           act                act0                   act1                  act2                   act3
        r  reward[1:]         rew1                   rew2                  rew3
        c  cont[1:] * disc    con1 * disc            con2 * disc           con3 * disc
        v  val[1:]            val1                   val2                  val3

        t  temp_diff          r1 + c1 * v1 * (1-l)   r1 + c1 * v1 * (1-l)  r1 + c1 * v1 * (1-l)

        m  monte_carlo        t0 + c1 * l *  ────►   t1 + c2 * l *  ────►  t2 + c3 * l *  ────►   val3

                            ┌────┐                 ┌────┐                ┌────┐
           target           │ m0 │                 │ m1 │                │ m2 │
                            └────┘                 └────┘                └────┘

           l -> lambda (0.95) disc -> discount (0.997)
    """

    reward = reward[1:]
    cont = cont[1:]
    assert len(reward) == len(value) - 1
    disc = cont * discount
    vals = [value[-1]]

    # interm = reward + disc * value[1:] * (1 - lamb_da)
    # for t in reversed(range(len(disc))):
    #     vals.append(interm[t] + disc[t] * lamb_da * vals[-1])
    #
    # interm = reward + disc * value[1:] * (1 - lamb_da)
    for t in reversed(range(len(disc))):
        vals.append(reward[t] + disc[t] * ((1 - lamb_da) * value[t + 1] + lamb_da * vals[-1]))
    ret = torch.stack(list(reversed(vals))[:-1])
    return ret, value[:-1]


def traj_weight(cont, discount=0.997):
    return (torch.cumprod(discount * cont, 0) / discount).detach()


class CriticLoss:
    def __init__(self):
        pass

    def __call__(self, returns, critic_dist, slow_critic_dist, cont):
        """
        note: returns len should be 1 shorter than critic_dist
        """
        trj_wgt = traj_weight(cont)
        self.loss_critic = -critic_dist.log_prob(returns.squeeze(-1).detach())
        self.loss_reg = -critic_dist.log_prob(slow_critic_dist.mean.squeeze(-1).detach())
        self.loss = self.loss_critic + self.loss_reg
        self.loss = self.loss * trj_wgt.squeeze(-1).detach()
        return self.loss.mean()

    def log_dict(self):
        return {
            'ac_critic_loss_critic': self.loss_critic.mean().detach().cpu(),
            'ac_critic_loss_reg': self.loss_reg.mean().detach().cpu(),
            'ac_critic_loss': self.loss.mean().detach().cpu()
        }


class Moment:
    def __init__(self, decay):
        self.value = None
        self.decay = decay

    def update(self, value):
        if self.value is None:
            self.value = value
        self.value * self.decay + value * (1. - self.decay)


class ActorLoss:
    def __init__(self):
        self.perc_5 = Moment(decay=0.99)
        self.perc_95 = Moment(decay=0.99)
        self.offset = 0.
        self.invscale = 1.

    def __call__(self, action_dist, action, returns, base, cont, entropy_scale=3e-4):
        self.perc_5.update(torch.quantile(returns, 0.05, interpolation='higher'))
        self.perc_95.update(torch.quantile(returns, 0.95, interpolation='lower'))
        traj_wgt = traj_weight(cont)

        self.offset, self.invscale = self.perc_5.value, max(self.perc_95.value - self.perc_5.value, 1.)
        self.normed_ret = (returns - self.offset) / self.invscale
        self.normed_base = (base - self.offset) / self.invscale
        self.adv = self.normed_ret - self.normed_base
        self.logpi = action_dist.log_prob(action.detach())[:-1]
        self.loss_policy_grad = -self.logpi * self.adv.detach()
        self.ent = action_dist.entropy()[:-1]
        self.loss_ent = - entropy_scale * self.ent
        self.loss = self.loss_policy_grad + self.loss_ent
        self.loss *= traj_wgt.detach()[:-1]
        self.loss = self.loss.mean()
        return self.loss

    def log_scalars(self):
        return {
            'ac_actor_loss_perc_5': self.perc_5.value,
            'ac_actor_loss_perc_95': self.perc_95.value,
            'ac_actor_loss_offset': self.offset,
            'ac_actor_loss_invscale': self.invscale,
            'ac_actor_loss_normed_ret': self.normed_ret.mean().detach().cpu(),
            'ac_actor_loss_normed_base': self.normed_base.mean().detach().cpu(),
            'ac_actor_loss_adv': self.adv.mean().detach().cpu(),
            'ac_actor_loss_logpi': self.logpi.mean().detach().cpu(),
            'ac_actor_loss_policy_grad': self.loss_policy_grad.mean().detach().cpu(),
            'ac_actor_loss_entropy': self.ent.mean().detach().cpu(),
            'ac_actor_loss_loss_ent': self.loss_ent.mean().detach().cpu(),
            'ac_actor_loss': self.loss.detach().cpu(),
        }

    def log_distributions(self):
        return {
            'ac_actor_loss_advantage': self.adv.detach().cpu().numpy(),
            'ac_actor_loss_entropy_dist': self.ent.detach().cpu().numpy(),
        }


def polyak_update(critic, ema_critic, critic_ema_decay=0.98):

    """

    :param critic: critic to source the weights from (ie the critic we are training with grad descent)
    :param ema_critic: critic that will be used to compute target values
    :param critic_ema_decay: smoothing factor, default 0.98
    :return: None
    """
    with torch.no_grad():
        for critic_params, ema_critic_params in zip(critic.parameters(), ema_critic.parameters()):
            ema_critic_params.data = ema_critic_params * critic_ema_decay + (1 - critic_ema_decay) * critic_params


class ActorCriticTrainer:
    def __init__(self, actor, critic, lr=3e-5, adam_eps=1e-5, grad_clip=100., device='cpu'):
        super().__init__()
        self.actor_criterion = ActorLoss()
        self.critic_criterion = CriticLoss()
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.ema_critic = deepcopy(critic).to(device)
        self.critic_opt = Adam(critic.parameters(), lr=lr, eps=adam_eps)
        self.actor_opt = Adam(actor.parameters(), lr=lr, eps=adam_eps)
        register_gradient_clamp(critic, grad_clip)
        register_gradient_clamp(actor, grad_clip)

    def train_step(self, h, z, action, rewards, cont):
        self.h = h
        self.z = z
        self.action = action
        self.rewards = rewards
        self.cont = cont

        with torch.no_grad():
            critic_dist = self.critic(h.detach(), z.detach())
            self.returns, self.values = score(rewards, cont, critic_dist.mean.unsqueeze(-1))
            tw = traj_weight(cont)

        self.critic_dist = self.critic(h.detach()[:-1], z.detach()[:-1])
        self.ema_critic_dist = self.ema_critic(h.detach()[:-1], z.detach()[:-1])
        crit_loss = self.critic_criterion(self.returns, self.critic_dist, self.ema_critic_dist, tw[:-1])

        self.critic_opt.zero_grad()
        crit_loss.backward()
        self.critic_opt.step()
        polyak_update(self.critic, self.ema_critic)

        self.action_dist = self.actor(h.detach(), z.detach())
        act_loss = self.actor_criterion(self.action_dist, action, self.returns, self.values, cont)

        self.actor_opt.zero_grad()
        act_loss.backward()
        self.actor_opt.step()

    def log_scalars(self):
        with torch.no_grad():
            actor_criterion_log = self.actor_criterion.log_scalars()
            critic_criterion_log = self.critic_criterion.log_dict()
            return {**actor_criterion_log, **critic_criterion_log}

    def log_distributions(self):
        actor_criterion_dist = self.actor_criterion.log_distributions()
        return {
            'ac_rewards': self.rewards.detach().cpu(),
            'ac_critic_mean': self.critic_dist.mean.detach().cpu(),
            'ac_ema_critic_mean': self.ema_critic_dist.mean.detach().cpu(),
            'ac_returns': self.returns.detach().cpu(),
            'ac_values': self.values.detach().cpu(),
            'ac_actions': self.action.argmax(-1).detach().cpu(),
            'ac_action_dist_argmax': self.action_dist.logits.argmax(-1).detach().cpu(),
            'ac_cont': self.cont.detach().cpu(),
            **actor_criterion_dist
        }

    def state_dict(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor_state_dict'])
        self.critic.load_state_dict(state_dict['critic_state_dict'])
        self.actor_opt.load_state_dict(state_dict['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(state_dict['critic_opt_state_dict'])


if __name__ == '__main__':
    config_horizon = 333
    discount = 1 - 1 / config_horizon

    import time
    from replay import Step
    import replay
    from torch.optim import Adam
    from torch import nn
    from copy import deepcopy
    from torch.distributions import Categorical, Normal
    import gymnasium
    from statistics import mean, stdev
    import numpy as np
    from matplotlib import pyplot as plt
    import wandb
    from multiprocessing import Process
    from torch.linalg import norm

    plt.ion()
    torch.manual_seed(0)
    np.random.seed(0)

    env_name = 'SimplerGridWorld-grab_em_all-v0'
    wandb.init(project=f'alt-critic-{env_name}')
    env = gymnasium.make(env_name, render_mode=None)


    class CriticTable(nn.Module):
        def __init__(self, obs_shape):
            super().__init__()
            self.table = nn.Parameter(torch.zeros(*obs_shape))

        def forward(self, observation, *args):
            L, N, _ = observation.shape
            observation = observation.flatten(0, 1)
            x, y = observation[:, 0], observation[:, 1]
            return Normal(loc=self.table[x, y].unflatten(0, (L, N)), scale=1.)


    class ActorTable(nn.Module):
        def __init__(self, obs_shape, action_shape):
            super().__init__()
            shape = obs_shape + (action_shape,)
            self.table = nn.Parameter(torch.zeros(*shape))

        def forward(self, observation, *args):
            L, N, _ = observation.shape
            observation = observation.flatten(0, 1)
            x, y = observation[:, 0], observation[:, 1]
            return Categorical(logits=self.table[x, y].unflatten(0, (L, N)).unsqueeze(-2))


    def prepro(obs):
        return torch.tensor([obs[0][0], obs[0][1]], dtype=torch.long)


    def rollout(roll_env, roll_actor, pad_action):

        obs, info = roll_env.reset()
        obs_prepro = prepro(obs)
        reward, terminated, truncated = 0.0, False, False

        while True:
            action = roll_actor(obs_prepro.unsqueeze(0).unsqueeze(0)).sample().item()

            yield Step(obs_prepro, np.array([action]), reward, terminated, truncated)

            obs, reward, terminated, truncated, info = roll_env.step(action)
            obs_prepro = prepro(obs)

            if terminated or truncated:
                yield Step(obs_prepro, pad_action, reward, terminated, truncated)

                obs, info = roll_env.reset()
                obs_prepro = prepro(obs)
                reward, terminated, truncated = 0.0, False, False


    pad_action = np.array([0])

    critic = CriticTable(obs_shape=env.observation_space.shape)
    actor = ActorTable(obs_shape=env.observation_space.shape, action_shape=env.action_space.n.item())
    display_actor = deepcopy(actor)
    trainer = ActorCriticTrainer(actor, critic, lr=3e-3)

    batch_size = 128
    batch_length = 16


    def on_policy(rollout):
        buff = []
        lengths = []
        rewards = []

        for _ in range(10 * batch_size):
            buff += [next(rollout)]
            if buff[-1].is_terminal:
                trajectory = replay.get_tail_trajectory(buff)
                lengths += [len(trajectory)]
                rewards += [sum([step.reward.item() for step in trajectory])]
        return buff, lengths, rewards


    total_steps = 0

    loader = replay.BatchLoader()

    fig, ax = plt.subplots()
    plt.show()


    def run_display(process_name):
        display_env = gymnasium.make(env_name)
        display_actor = ActorTable(obs_shape=env.observation_space.shape, action_shape=env.action_space.n.item())
        if pathlib.Path('actor.pt').exists():
            display_actor.load_state_dict(torch.load('actor.pt'))
        runner = rollout(display_env, display_actor, pad_action)
        while True:
            step = next(runner)
            time.sleep(0.1)
            if step.is_terminal:
                if pathlib.Path('actor.pt').exists():
                    display_actor.load_state_dict(torch.load('actor.pt'))


    p = Process(target=run_display, args=('bob',))
    p.start()

    for step in range(20000):

        buff, tj_len, tj_rewards = on_policy(rollout(env, actor, pad_action))
        print(
            f'rewards mean: {mean(tj_rewards)}, stdev: {stdev(tj_rewards)} length mean: {mean(tj_len)}, stdev: {stdev(tj_len)}')
        wandb.log({
            'tj_rewards': mean(tj_rewards),
            'tj_len': mean(tj_len)
        })
        total_steps += len(buff)

        obs, act, reward, cont = loader.sample(buff, batch_length, batch_size)
        obs = obs.long()

        trainer.train_step(obs, obs, act, reward, cont)

        direction = actor.table.argmax(-1)
        arrows = env.get_action_meanings()

        # ascii plot
        for y in range(actor.table.shape[1]):
            row = ''
            for x in range(actor.table.shape[0]):
                row += arrows[direction[x, y]]
            print(row)

        obs_shape = env.observation_space.shape
        x, y = torch.meshgrid(torch.arange(obs_shape[0]), torch.arange(obs_shape[1]))
        x, y = x.flatten().float(), y.flatten().float()
        actor_probs = Categorical(logits=actor.table).probs

        ax.cla()
        ax.set_xlim(-1, obs_shape[0])
        ax.set_ylim(-1, obs_shape[1])
        plt.gca().invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.pcolormesh(np.arange(obs_shape[0] + 1) - 0.5, np.arange(obs_shape[1] + 1) - 0.5, critic.table.T.detach(),
                      shading='flat', vmin=critic.table.min(), vmax=critic.table.max(), cmap='gray')

        import matplotlib.cm

        max_direction = actor_probs.argmax(-1).flatten()
        colormap = matplotlib.cm.bwr

        color = colormap((max_direction == 1) * 1.)
        u, v = torch.zeros_like(x), -actor_probs[:, :, 1].detach().flatten()
        mag = (u ** 2 + v ** 2) ** 0.5
        u, v = u/mag/4, v / mag/4
        ax.quiver(x, y, u, v, angles='xy', color=color)

        color = colormap((max_direction == 3) * 1.)
        u, v = torch.zeros_like(x), actor_probs[:, :, 3].detach().flatten()
        mag = (u ** 2 + v ** 2) ** 0.5
        u, v = u/mag/4, v / mag/4
        ax.quiver(x, y, u, v, angles='xy', color=color)

        color = colormap((max_direction == 2) * 1.)
        u, v = actor_probs[:, :, 2].detach().flatten(), torch.zeros_like(y)
        mag = (u ** 2 + v ** 2) ** 0.5
        u, v = u/mag/4, v / mag/4
        ax.quiver(x, y, u, v, angles='xy', color=color)

        color = colormap((max_direction == 4) * 1.)
        u, v = - actor_probs[:, :, 4].detach().flatten(), torch.zeros_like(y)
        mag = (u ** 2 + v ** 2) ** 0.5
        u, v = u/mag/4, v / mag/4
        ax.quiver(x, y, u, v, angles='xy', color=color)

        plt.pause(0.1)

        torch.save(critic, 'critic.pt')
        torch.save(actor.state_dict(), 'actor.pt')
