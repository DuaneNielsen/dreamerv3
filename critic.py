import torch
import torch.nn as nn
from torch.nn.functional import one_hot, mse_loss
from torch.optim import Adam
from copy import deepcopy
from blocks import MLPBlock


class Critic(nn.Module):
    def __init__(self,  h_size=512, z_size=32, z_classes=32, mlp_size=512, mlp_layers=2):
        super().__init__()
        self.output = nn.Linear(mlp_size, 1, bias=False)
        self.critic = torch.nn.Sequential(
            nn.Linear(h_size + z_size * z_classes, mlp_size, bias=False),
            *[MLPBlock(mlp_size, mlp_size) for _ in range(mlp_layers)],
            self.output
        )
        self.output.weight.data.zero_()

    def forward(self, h, z):
        h_z_flat = torch.cat((h, z.flatten(-2)), dim=-1)
        return self.critic(h_z_flat)


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


def td_lambda(rewards, cont, values, discount=0.997, lam=0.95):
    """
    target values for the value function
    use an ema critic to calculate targets, don't use the training critic!
    :param rewards: [..., 1] rewards
    :param cont: [..., 1]
    :param values: [..., 1]
    :param discount: discount factor default 0.997
    :param lam: lamda factor as in td lambda algorithm, default 0.95
    :return: target values for value function
    """
    target_values = torch.zeros_like(values)
    target_values[-1] = values[-1]
    for t in reversed(range(values.size(0)-1)):
        lam_da = (1 - lam) * values[t + 1] + lam * target_values[t + 1]
        target_values[t] = cont[t] * discount * lam_da + rewards[t]
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


if __name__ == '__main__':

    from replay import sample_batch, Step
    from env import Env

    env = Env()
    buff = []

    def rollout_open_loop_policy(env, actions):
        x, r, c = env.reset(), 0, 1.0
        for a in actions:
            yield Step(x, a, r, c)
            x, r, done, _ = env.step(a)
            c = 0.0 if done else 1.0
        yield Step(x, None, r, c)


    go_right = [Env.right] * 8
    go_left = [Env.left]

    for step in rollout_open_loop_policy(env, go_right):
        buff.append(step)

    critic = Critic(h_size=10)
    ema_critic = deepcopy(critic)

    random_noise = torch.randn(1, 10)
    assert torch.allclose(critic(random_noise), ema_critic(random_noise))

    opt = Adam(critic.parameters(), lr=1e-2)

    for step in range(2000):
        obs, act, reward, cont, mask = sample_batch(buff, 10, 4, Env.pad_state, Env.pad_action)
        obs = obs[:, 0]
        act = act[:, 0]
        reward = reward[:, 0]
        cont = cont[:, 0]

        values = ema_critic(obs)
        value_targets = td_lambda(reward, cont, values)
        loss = mse_loss(critic(obs), value_targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        polyak_update(critic, ema_critic)

        if step % 10 == 0:
            value = monte_carlo(reward)
            print(value.T)
            print(critic(obs).T)
