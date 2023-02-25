import torch
from torch import nn as nn
from torch.nn.functional import one_hot
from copy import deepcopy
from blocks import MLPBlock, Embedder
from torch.distributions import OneHotCategorical
from dists import OneHotCategoricalStraightThru, TwoHotSymlog, OneHotCategoricalUnimix


class Critic(nn.Module):
    def __init__(self,  h_size=512, z_size=32, z_classes=32, mlp_size=512, mlp_layers=2):
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


def monte_carlo(rewards, discount=0.997):
    value_t_plus_one = 0.
    value = torch.zeros_like(rewards)
    for t in reversed(range(rewards.size(0))):
        value[t] = rewards[t] + discount * value_t_plus_one
        value_t_plus_one = value[t]
    return value


def td(rewards, cont, values, discount=0.997):
    target_values = torch.zeros_like(values)
    target_values[-1] = values[-1]
    for t in reversed(range(values.size(0)-1)):
        target_values[t] = cont[t] * discount * target_values[t+1] + rewards[t]
    return target_values.detach()


def td_lambda(rewards, cont, values_pred, discount=0.997, lam=0.95):
    """
    target values for the value function
    use an ema critic to calculate targets, don't use the training critic!
    :param rewards: [..., 1] rewards
    :param cont: [..., 1]
    :param values_pred: [..., 1]
    :param discount: discount factor default 0.997
    :param lam: lambda factor as in td lambda algorithm, default 0.95
    :return: target values for value function
    """
    target_values = torch.zeros_like(values_pred)
    target_values[-1] = values_pred[-1]
    for t in reversed(range(values_pred.size(0) - 1)):
        lam_da = (1 - lam) * values_pred[t + 1] + lam * target_values[t + 1]
        target_values[t] = cont[t] * discount * lam_da + rewards[t + 1]
    return target_values.detach()


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
        return action_flat.unflatten(-1, (self.action_size, self.action_classes))

    def sample_action(self, h, z):
        action_logits = self.forward(h, z)
        return OneHotCategoricalStraightThru(logits=action_logits).sample()

    def exploit_action(self, h, z):
        action_logits = self.forward(h, z)
        return OneHotCategorical(logits=action_logits).mode


class ActorLoss:
    def __init__(self, return_normalization_limit=1.0, return_normalization_decay=0.99):
        self.reinforce_loss = None
        self.entropy_loss = None
        self.actor_loss = None
        self.return_scale_running = None
        self.return_normalization_limit = return_normalization_limit
        self.return_normalization_decay = return_normalization_decay
        self.applied_scale = 1.
        self.value_preds_min = None
        self.value_preds_5th_percentile = None
        self.value_preds_95th_percentile = None
        self.value_preds_max = None

    def __call__(self, actor_logits, actions, critic_values, mask=None):
        """
        Policy gradient loss for actor, value based Reinforce + a fixed entropy bonus
        :param actor_logits: (L, N, action_size, action_classes) - raw scores from the policy drawn from
        states
        :param actions: (L, N, action_size, action_classes) - actions taken by the policy during the trajectory
        :param : (L, N, 1) values of states in the trajectory from the critic
        :param mask: optional (L, N, 1) -  Boolean Tensor indicating values to not be included in the loss

        to use:

        actor_logits = actor(obs)
        actor_loss = actor_criterion(actor_logits, act, critic_values, mask)
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()


        """

        # critic values are shifted right, last action in trajectory is discarded
        # if the last action is the terminal, its going to be tha pad action anyway
        with torch.no_grad():
            self.value_preds_min = critic_values.min()
            self.value_preds_max = critic_values.max()
            self.value_preds_95th_percentile = torch.quantile(critic_values, 0.95, interpolation='lower')
            self.value_preds_5th_percentile = torch.quantile(critic_values, 0.05, interpolation='higher')
            return_scale_step = self.value_preds_95th_percentile - self.value_preds_5th_percentile
            if self.return_scale_running is None:
                self.return_scale_running = return_scale_step
            self.return_scale_running = self.return_scale_running * self.return_normalization_decay + return_scale_step * (1. - self.return_normalization_decay)
            self.applied_scale = max(self.return_scale_running.item(), 1.)

        actor_dist = OneHotCategoricalUnimix(logits=actor_logits)
        self.reinforce_loss = - critic_values.detach() * actor_dist.log_prob(actions) / self.applied_scale
        self.entropy_loss = - 3e-4 * actor_dist.entropy()
        if mask is not None:
            self.reinforce_loss = self.reinforce_loss * mask
            self.entropy_loss = self.entropy_loss * mask
        self.reinforce_loss = self.reinforce_loss.mean()
        self.entropy_loss = self.entropy_loss.mean()
        self.actor_loss = self.reinforce_loss + self.entropy_loss
        return self.actor_loss

    def loss_dict(self):
        return {
            'actor_reinforce_loss': self.reinforce_loss.item(),
            'actor_entropy_loss': self.entropy_loss.item(),
            'actor_returns_applied_scale:': self.applied_scale,
            'actor_returns_min': self.value_preds_min.item(),
            'actor_returns_5th_percentile': self.value_preds_5th_percentile.item(),
            'actor_returns_95th_percentile': self.value_preds_95th_percentile.item(),
            'actor_returns_max': self.value_preds_max.item(),
            'actor_loss': self.actor_loss.item()
        }


if __name__ == '__main__':

    from replay import sample_batch, Step
    import replay
    from torch.optim import Adam
    from envs import gridworld

    env = gridworld.make("3x3", render_mode='human')

    class CriticTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.zeros(3, 3, 4))

        def forward(self, observation):
            L, N, _ = observation.shape
            observation = observation.flatten(0, 1)
            x, y, d = observation[:, 0], observation[:, 1], observation[:, 2]
            return self.table[x, y, d].unflatten(0, (L, N, 1))


    class ActorTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.zeros(3, 3, 4, 4))

        def forward(self, observation):
            L, N, _ = observation.shape
            observation = observation.flatten(0, 1)
            x, y, d = observation[:, 0], observation[:, 1], observation[:, 2]
            return self.table[x, y, d].unflatten(0, (L, N)).unsqueeze(-2)

        def sample_action(self, observation):
            logits = self.forward(observation)
            return OneHotCategorical(logits=logits).sample()

    def prepro(obs):
        return torch.tensor([obs.pos.x, obs.pos.y, obs.direction], dtype=torch.long)

    def prepro_action(a):
        return one_hot(torch.tensor([a]), 4)


    def rollout(env, actor):

        obs = env.reset()
        obs_prepro = prepro(obs)
        reward, terminated, truncated, cont = 0.0, False, False, 1.

        while True:
            action = actor.sample_action(obs_prepro.unsqueeze(0).unsqueeze(0))
            action_env = action.argmax(-1).item()

            yield Step(obs_prepro, action[0, 0], reward, cont)

            obs, reward, terminated, truncated, info = env.step(action_env)
            obs_prepro = prepro(obs)
            cont = 0. if terminated else 1.

            if terminated or truncated:

                yield Step(obs_prepro, None, reward, cont)

                obs = env.reset()
                obs_prepro = prepro(obs)
                reward, terminated, truncated, cont = 0.0, False, False, 1.


    pad_action = one_hot(torch.tensor([0]), 4)
    pad_state = torch.tensor([0, 0, 0], dtype=torch.long)

    critic = CriticTable()
    ema_critic = deepcopy(critic)
    opt = Adam(critic.parameters(), lr=1e-2)

    actor = ActorTable()
    opt_actor = Adam(actor.parameters(), lr=1e-2)
    actor_criterion = ActorLoss()

    batch_size = 8

    def on_policy(rollout):
        buff = []

        for _ in range(10 * batch_size):
            buff += [next(rollout)]
            if replay.is_trajectory_end(buff[-1]):
                trajectory = replay.get_tail_trajectory(buff)
                print(f'traj end reward: {replay.total_reward(trajectory)} len {len(trajectory)}')
        return buff

    total_steps = 0

    for step in range(20000):

        buff = on_policy(rollout(env, actor))
        total_steps += len(buff)

        obs, act, reward, cont, mask = sample_batch(buff, 10, batch_size, pad_state, pad_action)

        ema_values = ema_critic(obs)
        value_targets = td_lambda(reward, cont, ema_values)
        critic_values = critic(obs)
        loss = 0.5 * (critic_values - value_targets) ** 2
        loss = loss * mask
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        polyak_update(critic, ema_critic)

        actor_logits = actor(obs)
        actor_loss = actor_criterion(actor_logits, act, critic_values, mask)
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        if step % 10 == 0:
            for i in range(critic.table.shape[2]):
                print(critic.table[:, :, i])

            for i in range(actor.table.shape[2]):
                print(actor.table[:, :, i].argmax(-1))

            print(total_steps)
