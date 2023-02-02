import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from replay import sample_batch, Step
from collections import deque
from torch.nn.functional import one_hot
from torchvision.transforms.functional import resize
from rssm import RSSM
import torch
from time import sleep


def prepro(x):
    x = torch.from_numpy(x['image']).permute(2, 0, 1)
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

    buff = deque(maxlen=10 ** 6)

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = RGBImgObsWrapper(env)

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

    gen = rollout(env, RepeatOpenLoopPolicy([2, 2, 1, 2, 2]))
    for i in range(10):
        sleep(0.2)
        buff += [next(gen)]
