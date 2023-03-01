from __future__ import annotations

from typing import SupportsFloat, Any, Tuple, Dict

import gymnasium
from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.spaces import Discrete, Box
import numpy as np
from copy import deepcopy
import cv2


class Tile:
    def __init__(self, color, traversable=True, terminal=False, reward=0.):
        self.color = color
        self.traversable = traversable
        self.terminal = terminal
        self.reward = reward
        self.stack = []


class SimplerGridWorld(gymnasium.Env):

    def __init__(self, world_name, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        self.grid, self.grid_size = make_grid(world_name)
        self.observation_space = Box(low=0, high=max(self.grid_size), shape=self.grid_size, dtype=np.int64)
        self.action_space = Discrete(4)
        self.start_pos = np.array([0, 0], dtype=np.int64)
        self.pos = deepcopy(self.start_pos)

        self.moves = np.array([
            [0, -1],  # N
            [1, 0],  # E
            [0, 1],  # S
            [-1, 0]  # W
        ], dtype=np.int64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        self.pos = deepcopy(self.start_pos)

        if self.render_mode == 'human':
            self.render()

        return (self.pos, self.grid), {}

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        next_pos = self.pos + self.moves[action]
        next_pos[0] = np.clip(next_pos[0], 0, self.grid_size[0] - 1)
        next_pos[1] = np.clip(next_pos[1], 0, self.grid_size[1] - 1)
        next_tile = self.grid[next_pos[0]][next_pos[1]]

        if next_tile.traversable:
            self.pos = next_pos
        reward = next_tile.reward
        terminated = next_tile.terminal

        if self.render_mode == 'human':
            self.render()
        return (self.pos, self.grid), reward, terminated, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        base_size = self.grid_size + (3,)
        image = np.zeros(base_size, dtype=np.uint8)
        for k, row in enumerate(self.grid):
            for j, column in enumerate(row):
                for i, color in enumerate(column.color):
                    image[k, j, i] = color

        image[self.pos[0].item(), self.pos[1].item(), 2] = 255
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('SimplerGridWorld', image)
        cv2.waitKey(100)
        return image


tiles = {
    'e': Tile(color=(0, 0, 0)),
    'g': Tile(color=(0, 255, 0), terminal=True, reward=1.)
}

worlds = {
    'empty': [
        'eee',
        'eee',
        'eeg'
    ]
}


def world_hw(world):
    return len(world[0]), len(world)


def make_grid(world_name):
    world = worlds[world_name]
    grid = []
    for r, row in enumerate(world):
        grid += [[]]
        for i, column in enumerate(row):
            grid[r] += [tiles[column]]
    return grid, world_hw(world)


if __name__ == '__main__':

    import envs
    env = gymnasium.make('SimplerGridWorld-empty-v0')
    obs, info = env.reset()
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
