from __future__ import annotations

from typing import SupportsFloat, Any, Tuple, Dict

import cv2
import gymnasium
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.spaces import MultiBinary, MultiDiscrete


class ButtonWorld(gymnasium.Env):

    def __init__(self, target, render_mode='human'):
        super().__init__()
        self.grid = None
        self.target = target
        self.observation_space = MultiBinary(self.target.shape)
        self.action_space = MultiDiscrete(self.target.shape)
        self.render_mode = render_mode
        self.reset()

    def render_human(self):
        rgb_array = self.render()
        cv2.imshow('buttonworld', rgb_array)
        cv2.waitKey(1000)

    def reset(self, seed=None, **options):
        self.grid = np.zeros(self.target.shape)
        return self.grid

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.grid[action[0], action[1]] = np.logical_not(self.grid[action[0], action[1]])
        reward = np.logical_and(self.grid, self.target).sum()
        terminated = np.allclose(self.grid, self.target)
        if self.render_mode == 'human':
            self.render_human()
        return self.grid, reward, terminated, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        gray = self.grid.astype(np.uint8) * 255
        gray = cv2.resize(gray, dsize=(300, 300))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class GameOfLife(ButtonWorld):
    def __init__(self, target, render_mode='human'):
        super().__init__(target, render_mode)
        self.kernel = np.array([
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.]
        ])

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.grid[action[0], action[1]] = np.logical_not(self.grid[action[0], action[1]])
        scores = cv2.filter2D(src=self.grid, ddepth=-1, kernel=self.kernel)
        self.grid = np.logical_or(scores >= 3.0, np.logical_and(scores == 2, self.grid == 1.0)) * 1.0
        reward = np.logical_and(self.grid, self.target).sum()
        terminated = np.allclose(self.grid, self.target)
        if self.render_mode == 'human':
            self.render_human()
        return self.grid, reward, terminated, False, {}



if __name__ == '__main__':

    target = np.ones((100, 100))
    env = GameOfLife(target)
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
