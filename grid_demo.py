import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from replay import sample_batch, Step
from collections import deque
from torch.nn.functional import one_hot
from torchvision.transforms.functional import resize
from rssm import make_small, RSSMLoss
import torch
from torch.optim import Adam
from time import sleep
from argparse import ArgumentParser


def prepro(x):
    x = torch.from_numpy(x['image']).permute(2, 0, 1).float()
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


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_length', type=int, default=64)
    parser.add_argument('--replay_capacity', type=int, default=10 ** 6)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--gradient_clipping', type=float, default=1000.)
    args = parser.parse_args()

    buff = deque(maxlen=args.replay_capacity)

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = RGBImgObsWrapper(env)
    pad_state = torch.zeros(3, 64, 64, dtype=torch.uint8)
    pad_action = one_hot(torch.tensor([0]), 4)

    """
    +------+---------+-------------------+
    | Num  |  Name   |      Action       |
    +------+---------+-------------------+
    |    0 | left    | Turn left         |
    |    1 | right   | Turn right        |
    |    2 | forward | Move forward      |
    |    3 | pickup  | Pick up an object |
    +------+---------+-------------------+    
    """

    rssm = make_small(action_classes=4)
    opt = Adam(rssm.parameters(), lr=args.learning_rate)
    criterion = RSSMLoss()

    gen = rollout(env, RepeatOpenLoopPolicy([2, 2, 1, 2, 2]))
    for i in range(10):
        sleep(0.2)
        buff += [next(gen)]

    for steps in range(10000):
        buff += [next(gen)]
        print(buff)
        x, a, r, c, mask = sample_batch(buff, args.batch_length, args.batch_size, pad_state, pad_action)

        h0 = torch.zeros(args.batch_size, rssm.h_size)
        x_dist, r_dist, c_dist, z_prior, z_post = rssm(x, a, h0)
        loss = criterion(x, r, c, mask, x_dist, r_dist, c_dist, z_prior, z_post)

        opt.zero_grad()
        loss.backward()
        opt.step()



