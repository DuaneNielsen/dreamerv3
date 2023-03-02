import torch
from multiprocessing import Process

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
        v  val[:-1]           val1                   val2                  val3

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
    interm = reward + disc * value[1:] * (1 - lamb_da)
    for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * lamb_da * vals[-1])
    ret = torch.stack(list(reversed(vals))[:-1])
    return ret, value[:-1]


def traj_weight(cont, discount=0.997):
    return (torch.cumprod(discount * cont, 0) / discount).detach()


def critic_loss(returns, critic_dist, slow_critic_dist, traj_weight):
    loss = -critic_dist.log_prob(returns.detach())
    reg = -critic_dist.log_prob(slow_critic_dist.mean.detach())
    loss += reg
    return (loss * traj_weight.detach()).mean()


class Moment:
    def __init__(self, decay):
        self.value = None
        self.decay = decay

    def update(self, value):
        if self.value is None:
            self.value = value
        self.value * self.decay + value * (1. - self.decay)


def actor_loss(ret, perc_5, perc_95, base, action_dist, action, traj_weight, entropy_scale=-3e-4):
    offset, invscale = perc_5, max(perc_95 - perc_5, 1.)
    normed_ret = (ret - offset) / invscale
    normed_base = (base - offset) / invscale
    adv = normed_ret - normed_base
    logpi = action_dist.log_prob(action.detach())[:-1]
    loss = -logpi * adv.detach()
    ent = action_dist.entropy()[:-1]
    loss -= entropy_scale * ent
    loss *= traj_weight.detach()[:-1]
    return loss.mean()


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


if __name__ == '__main__':
    config_horizon = 333
    discount = 1 - 1 / config_horizon

    from replay import Step
    import replay
    from torch.optim import SGD, Adam
    from torch import nn
    from torch.nn.functional import one_hot
    from copy import deepcopy
    from torch.distributions import Categorical, Normal
    import gymnasium
    import envs
    from statistics import mean, stdev
    import numpy as np
    from matplotlib import pyplot as plt
    import wandb

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

        def forward(self, observation):
            L, N, _ = observation.shape
            observation = observation.flatten(0, 1)
            x, y = observation[:, 0], observation[:, 1]
            return Normal(loc=self.table[x, y].unflatten(0, (L, N, 1)), scale=1.)


    class ActorTable(nn.Module):
        def __init__(self, obs_shape, action_shape):
            super().__init__()
            shape = obs_shape + (action_shape,)
            self.table = nn.Parameter(torch.zeros(*shape))

        def forward(self, observation):
            L, N, _ = observation.shape
            observation = observation.flatten(0, 1)
            x, y = observation[:, 0], observation[:, 1]
            return Categorical(logits=self.table[x, y].unflatten(0, (L, N)).unsqueeze(-2))


    def prepro(obs):
        return torch.tensor([obs[0][0], obs[0][1]], dtype=torch.long)


    def rollout(env, actor, pad_action):

        obs, info = env.reset()
        obs_prepro = prepro(obs)
        reward, terminated, truncated = 0.0, False, False

        while True:
            action = actor(obs_prepro.unsqueeze(0).unsqueeze(0)).sample().item()

            yield Step(obs_prepro, np.array([action]), reward, terminated, truncated)

            obs, reward, terminated, truncated, info = env.step(action)
            obs_prepro = prepro(obs)

            if terminated or truncated:
                yield Step(obs_prepro, pad_action, reward, terminated, truncated)

                obs, info = env.reset()
                obs_prepro = prepro(obs)
                reward, terminated, truncated = 0.0, False, False


    pad_action = np.array([0])

    critic = CriticTable(obs_shape=env.observation_space.shape)
    ema_critic = deepcopy(critic)
    opt = Adam(critic.parameters(), lr=1e-2)

    actor = ActorTable(obs_shape=env.observation_space.shape, action_shape=env.action_space.n.item())
    opt_actor = Adam(actor.parameters(), lr=1e-2)
    display_actor = deepcopy(actor)

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

    returns_5th = Moment(decay=0.99)
    returns_95th = Moment(decay=0.99)

    loader = replay.BatchLoader()

    fig, ax = plt.subplots()
    plt.show()

    def run_display(process_name):
        display_env = gymnasium.make(env_name)
        display_actor = ActorTable(obs_shape=env.observation_space.shape, action_shape=env.action_space.n.item())
        display_actor.load_state_dict(torch.load('actor.pt'))
        runner = rollout(display_env, display_actor, pad_action)
        while True:
            step = next(runner)
            if step.is_terminal:
                display_actor.load_state_dict(torch.load('actor.pt'))


    p = Process(target=run_display, args=('bob',))
    p.start()

    for step in range(20000):

        buff, tj_len, tj_rewards = on_policy(rollout(env, actor, pad_action))
        print(f'rewards mean: {mean(tj_rewards)}, stdev: {stdev(tj_rewards)} length mean: {mean(tj_len)}, stdev: {stdev(tj_len)}')
        wandb.log({
            'tj_rewards': mean(tj_rewards),
            'tj_len': mean(tj_len)
        })
        total_steps += len(buff)

        obs, act, reward, cont = loader.sample(buff, batch_length, batch_size)
        obs = obs.long()

        critic_dist = critic(obs)
        returns, values = score(reward, cont, critic_dist.mean)
        returns_5th.update(torch.quantile(returns, 0.05, interpolation='higher'))
        returns_95th.update(torch.quantile(returns, 0.95, interpolation='lower'))
        tw = traj_weight(cont)

        critic_dist = critic(obs[:-1])
        ema_critic_dist = ema_critic(obs[:-1])
        crit_loss = critic_loss(returns, critic_dist, ema_critic_dist, tw[:-1])

        opt.zero_grad()
        crit_loss.backward()
        opt.step()
        polyak_update(critic, ema_critic)

        action_dist = actor(obs)

        act_loss = actor_loss(returns, returns_5th.value, returns_95th.value, values, action_dist, act, tw)
        opt_actor.zero_grad()
        act_loss.backward()
        opt_actor.step()

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

        color = colormap((max_direction == 0) * 1.)
        u, v = torch.zeros_like(x), -actor_probs[:, :, 0].detach().flatten()
        ax.quiver(x, y, u, v, angles='xy', color=color)

        color = colormap((max_direction == 2) * 1.)
        u, v = torch.zeros_like(x), actor_probs[:, :, 2].detach().flatten()
        ax.quiver(x, y, u, v, angles='xy', color=color)

        color = colormap((max_direction == 1) * 1.)
        u, v = actor_probs[:, :, 1].detach().flatten(), torch.zeros_like(y)
        ax.quiver(x, y, u, v, angles='xy', color=color)

        color = colormap((max_direction == 3) * 1.)
        u, v = - actor_probs[:, :, 3].detach().flatten(), torch.zeros_like(y)
        ax.quiver(x, y, u, v, angles='xy', color=color)

        plt.pause(0.1)

        torch.save(critic, 'critic.pt')
        torch.save(actor.state_dict(), 'actor.pt')


