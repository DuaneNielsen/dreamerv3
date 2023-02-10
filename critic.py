import torch
import torch.nn as nn
from torch.nn.functional import one_hot, mse_loss
from torch.optim import Adam
from copy import deepcopy
from blocks import MLPBlock, Embedder
from torchvision.transforms.functional import resize
from viz import make_panel
from matplotlib import pyplot as plt


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
    import gymnasium as gym
    from gymnasium import RewardWrapper
    import random
    from minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper
    # env = Env()
    buff = []

    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = RGBImgObsWrapper(env)

    class RewardOneOrZeroOnlyWrapper(RewardWrapper):
        def __init__(self, env):
            super().__init__(env)

        def reward(self, reward):
            return 0. if reward == 0. else 1.

    env = RewardOneOrZeroOnlyWrapper(env)

    def rollout_open_loop_policy(env, actions):
        x, r, c = env.reset(), 0, 1.0
        for a in actions:
            yield Step(x, a, r, c)
            x, r, done, _ = env.step(a)
            c = 0.0 if done else 1.0
        yield Step(x, None, r, c)

    def prepro(x):
        x = torch.from_numpy(x['image']).permute(2, 0, 1).float() / 255.0
        return resize(x, [64, 64])

    def prepro_action(a):
        return one_hot(torch.tensor([a]), 4)

    def rollout_random_policy(env):
        (x, _), r, c = env.reset(), 0, 1.0
        truncated = False
        while c == 1.0 and not truncated:
            a = random.randint(0, 3)
            yield Step(prepro(x), prepro_action(a), r, c)
            x, r, done, truncated, _ = env.step(a)
            c = 0.0 if done else 1.0
        yield Step(prepro(x), None, r, c)


    for _ in range(500):
        for step in rollout_random_policy(env):
            buff.append(step)

    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), 4)

    embedder = Embedder()
    critic = Critic(h_size=4096, z_size=1, z_classes=1)
    ema_critic = deepcopy(critic)

    # random_noise = torch.randn(1, 10)
    # assert torch.allclose(critic(random_noise), ema_critic(random_noise))

    opt = Adam(critic.parameters(), lr=1e-2)

    drawn = None
    for step in range(20000):
        obs, act, reward, cont, mask = sample_batch(buff, 10, 4, pad_state, pad_action)
        z = torch.zeros(obs.shape[0], obs.shape[1], 1, 1)

        embed = embedder(obs)
        ema_values = ema_critic(embed, z)
        value_targets = td_lambda(reward, cont, ema_values)
        critic_values = critic(embed, z)
        loss = 0.5 * ( critic_values - value_targets) ** 2
        loss = loss * mask
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        polyak_update(critic, ema_critic)

        if step % 10 == 0:
            panel = make_panel(obs, act, reward, cont, mask, critic_values, symexp_on=False)
            if drawn:
                drawn.set_data(panel.permute(1, 2, 0))
            else:
                drawn = plt.imshow(panel.permute(1, 2, 0))
            plt.pause(0.01)


        # if step % 10 == 0:
        #     value = monte_carlo(reward)
        #     print(value.T)
        #     print(critic(obs, z).T)
