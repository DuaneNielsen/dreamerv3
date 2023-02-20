from PIL import Image
from gymnasium.core import ObsType, WrapperObsType, ActType, WrapperActType
from matplotlib import pyplot as plt
from collections import namedtuple
import importlib.resources
from copy import copy
from gymnasium import ObservationWrapper, ActionWrapper
from torchvision.transforms.functional import to_tensor, resize


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    def __ge__(self, other):
        return self.x >= other.x and self.y >= other.y

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __le__(self, other):
        return self.x <= other.x and self.y <= other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f'{(self.x, self.y)}'


State = namedtuple("State", ["pos", "direction", "grid"])

start_location = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
empty_tile = {'e'}.union(start_location)
goal_tile = {'g'}
lava_tile = {'l'}
wall_tile = {'w'}
terminal_tile = goal_tile.union(lava_tile)
reward_tile = {'g': 1.0, 'l': -1.0}
blocking_tile = wall_tile


class Renderer:
    def __init__(self, env):
        self.env = env

        with importlib.resources.path("envs", "empty_tile.png") as empty_tile:
            self.empty_tile = Image.open(empty_tile)
        with importlib.resources.path("envs", "goal_tile.png") as goal_tile:
            self.goal_tile = Image.open(goal_tile)
        with importlib.resources.path("envs", "lava_tile.png") as lava_tile:
            self.lava_tile = Image.open(lava_tile)
        with importlib.resources.path("envs", "wall_tile.png") as wall_tile:
            self.wall_tile = Image.open(wall_tile)
        self.arrow = []
        for direction in range(0, 4):
            with importlib.resources.path("envs", f'arrow_{direction}.png') as arrow:
                self.arrow += [Image.open(arrow)]

    def _background(self, grid):
        tiles = []
        for x in range(len(grid[0])):
            tiles += [[]]
            for y in range(len(grid)):
                if grid[y][x] in goal_tile:
                    tiles[x] += [self.goal_tile]
                if grid[y][x] in empty_tile:
                    tiles[x] += [self.empty_tile]
                if grid[y][x] in wall_tile:
                    tiles[x] += [self.wall_tile]
                if grid[y][x] in lava_tile:
                    tiles[x] += [self.lava_tile]

        return tiles

    def draw(self, pos, direction, grid):
        tiles = self._background(grid)
        tiles[pos.x][pos.y] = Image.alpha_composite(tiles[pos.x][pos.y], self.arrow[direction])
        image = Image.new('RGB', (len(grid[0]) * 21, len(grid) * 21))
        for x, row in enumerate(tiles):
            for y, tile in enumerate(row):
                image.paste(tile, (x * 21, y * 21))
        return image


class RGBImageWrapper:
    def __init__(self, env):
        self.env = env
        self.renderer = Renderer(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        image = self.renderer.draw(obs.pos, obs.direction, obs.grid)
        return image, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        image = self.renderer.draw(obs.pos, obs.direction, obs.grid)
        return image, reward, terminated, truncated, info


class MaxStepsWrapper:
    def __init__(self, env, max_steps):
        self.env = env
        self.steps = 0
        self.max_steps = max_steps

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.steps == self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class TensorObsWrapper(ObservationWrapper):

    def __init__(self, env, width=64, height=64):
        super().__init__(env)
        self.width = width
        self.height = height

    def observation(self, observation: ObsType) -> WrapperObsType:
        observation = resize(observation, [self.height, self.width])
        return to_tensor(observation)


class OneHotTensorActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action: WrapperActType) -> ActType:
        return action.argmax(-1)


class SimpleGridWorld:
    DIRECTION = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]

    def __init__(self, grid, render_mode=None):
        """
        creates a simple gridworld

        param: grid:

            to create a 3x3 gridworld provide a list of strings
            grid = [
              'Uee',
              'eee',
              'eeg'
            ]

            U, D, L, R -> starting position
            e -> empty tile
            g -> goal tile
            l -> lava tile
            w -> wall tile
        """

        self.grid = grid

        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                if tile in start_location:
                    self.start_pos = Vector2(x, y)
                    self.start_direction = start_location[tile]

        self.pos = copy(self.start_pos)
        self.direction = self.start_direction

        self.max = Vector2(len(self.grid[0]) - 1, len(self.grid) - 1)
        self.min = Vector2(0, 0)
        self.render_mode = render_mode
        self.render = Renderer(self)
        if render_mode == 'human':
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax_image = self.ax.imshow(self.render.draw(self.pos, self.direction, self.grid))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def draw(self):
        if self.render_mode == 'human':
            self.ax_image.set_data(self.render.draw(self.pos, self.direction, self.grid))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def reset(self, seed=None, **kwargs):
        self.pos = copy(self.start_pos)
        self.direction = self.start_direction
        self.draw()
        return State(self.pos, self.direction, self.grid)

    def step(self, action):
        """
        steps the environment
        :param action: 0 -> Turn left, 1 -> turn right, 2 -> forward
        """

        reward = 0.
        terminated = False
        truncated = False

        if action == 0:  # left
            self.direction = (self.direction - 1) % 4
        if action == 1:  # right
            self.direction = (self.direction + 1) % 4
        if action == 2:  # forward
            lookahead = self.pos + self.DIRECTION[self.direction]
            if self.max >= lookahead >= self.min:
                tile = self.grid[lookahead.y][lookahead.x]
                if tile not in blocking_tile:
                    self.pos += self.DIRECTION[self.direction]
                reward = reward_tile[tile] if tile in reward_tile else 0.
                terminated = tile in terminal_tile

        self.draw()

        return State(self.pos, self.direction, self.grid), reward, terminated, truncated, {}


env_configs = {
    '3x3': [
        'Uee',
        'eee',
        'eeg'
    ],
    'bandit': [
        'Rg'
    ],
    'corridor': [
        'lReeeg'
    ],
    'turn_around': [
        'lLeeg'
    ],
    'the_choice': [
        'lUg'
    ],
    'go_around': [
        'eee',
        'Rwg'
    ],
    'dont_fall': [
        'lll',
        'Reg',
        'lll'
    ]

}


def make(env_name, render_mode=None):
    return SimpleGridWorld(env_configs[env_name], render_mode=render_mode)


if __name__ == '__main__':

    from torch.nn.functional import one_hot
    import torch

    plt.ion()
    fig, ax = plt.subplots()

    env = make('corridor', render_mode='human')
    env = MaxStepsWrapper(env, 100)
    env = RGBImageWrapper(env)
    env = TensorObsWrapper(env)
    env = OneHotTensorActionWrapper(env)
    obs, info = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        try:
            action = input('press 0, 1, 2, q')
            if action == 'q':
                break
            action = int(action)
            if 0 <= action <= 2:
                action = one_hot(torch.tensor([action]), 3)
                obs, reward, terminated, truncated, info = env.step(action)
                print(reward, terminated)
                ax.imshow(obs.permute(1, 2, 0))
                fig.canvas.draw()
                fig.canvas.flush_events()
        except ValueError:
            continue
