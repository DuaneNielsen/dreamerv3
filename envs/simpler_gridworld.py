from __future__ import annotations

from typing import SupportsFloat, Any, Tuple, Dict

import gymnasium
from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.spaces import Discrete, Box
import numpy as np
from copy import deepcopy
import cv2


class Item:
    def __init__(self, color, reward):
        self.color = color
        self.reward = reward


class Tile:
    def __init__(self, color, start_pos=False, traversable=True, terminal=False, reward=0., stack=None):
        self._color = color
        self.traversable = traversable
        self.terminal = terminal
        self.reward = reward
        self.start_pos = start_pos
        self.stack = [] if stack is None else stack

    @property
    def has_item(self):
        return len(self.stack) > 0

    @property
    def color(self):
        if self.has_item:
            return self.stack[-1].color
        else:
            return self._color


class SimplerGridWorld(gymnasium.Env):

    def __init__(self, world_name, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        self.start_grid, self.grid_size, self.start_pos = make_grid(world_name)
        self.observation_space = Box(low=0, high=max(self.grid_size), shape=self.grid_size, dtype=np.int64)
        self.action_space = Discrete(4)
        self.pos = deepcopy(self.start_pos)
        self.grid = deepcopy(self.start_grid)
        self.render_speed = 1

        self.moves = np.array([
            [0, -1],  # N
            [1, 0],  # E
            [0, 1],  # S
            [-1, 0]  # W
        ], dtype=np.int64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        self.pos = deepcopy(self.start_pos)
        self.grid = deepcopy(self.start_grid)

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

        if next_tile.has_item:
            reward += next_tile.stack[-1].reward
            next_tile.stack.pop()

        if self.render_mode == 'human':
            self.render()
        return (self.pos, self.grid), reward, terminated, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        base_size = self.grid_size + (3,)
        image = np.zeros(base_size, dtype=np.uint8)
        for k in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for i, color in enumerate(self.grid[k][j].color):
                    image[k, j, i] = color

        image[self.pos[0].item(), self.pos[1].item(), 2] = 255
        image = image.swapaxes(1, 0)
        image = cv2.resize(image, (image.shape[1] * 128, image.shape[0] * 128), interpolation=cv2.INTER_AREA)
        cv2.imshow('SimplerGridWorld', image)
        cv2.waitKey(self.render_speed)
        return image

    def get_action_meanings(self):
        return ['\u2191', '\u2192', '\u2193', '\u2190']


items = {
    'reward_pack': Item(color=(20., 190, 20), reward=1.0)
}

tiles = {
    's': Tile(color=(0, 0, 0), start_pos=True),
    'e': Tile(color=(0, 0, 0)),
    'g': Tile(color=(0, 255, 0), terminal=True, reward=1.),
    'l': Tile(color=(70, 100, 200), terminal=True, reward=-1.),
    'w': Tile(color=(255, 255, 255), traversable=False),
    'p': Tile(color=(0, 0, 0), stack=[items['reward_pack']]),
}

worlds = {
    'empty': [
        'see',
        'eee',
        'eeg'
    ],

    'the_choice': [
        'lsg'
    ],

    'corridor': [
        'seeeeeeg',
    ],

    'go_around': [
        'see',
        'ewe',
        'eeg'
    ],

    'dont_turn_back': [
        'weeee',
        'lsweg'
    ],

    'grab_em_all': [
        'eeeeeeeeeeeeeeeepee',
        'eeeeeeeepeeeeeeeeee',
        'eeepeeeeeeeeeepeeee'
        'eeeeeeeeeeeeeeeeeee',
        'eeeeeeeeeseeeeeeeee',
        'eeeeeeeeeeeeeeeeeee'
        'eeepeeeeeeeeeeeeeee',
        'eeeeeeeeeeeeeepeeee',
        'epeeeeeeeeeeeeeeeee'
    ],

    'frozen_lake': [
        'leeegeeeee',
        'eeeleeelee',
        'eeeelleeee',
        'eeleeeelee',
        'eeeeseeeee'
    ]
}


def world_wh(world):
    return len(world[0]), len(world)


def make_grid(world_name):
    world = worlds[world_name]
    grid = []
    start_pos = np.array([0, 0], dtype=np.int64)
    w, h = world_wh(world)
    for x in range(w):
        grid += [[]]
        for y in range(h):
            tile = deepcopy(tiles[world[y][x]])
            grid[x] += [tile]
            if tile.start_pos:
                start_pos = np.array([x, y], dtype=np.int64)

    return grid, world_wh(world), start_pos


if __name__ == '__main__':

    import envs

    env = gymnasium.make('SimplerGridWorld-frozen_lake-v0')
    env.unwrapped.render_speed = 1000
    obs, info = env.reset()
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
