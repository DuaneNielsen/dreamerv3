from __future__ import annotations

from typing import SupportsFloat, Any, Tuple, Dict, List

import gymnasium
from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import cv2


class Teams:
    def __init__(self):
        self.teams = [[], []]

    def view(self, my_team):
        return self.teams[my_team % 2], self.teams[my_team + 1 % 2]


class Body:
    def __init__(self):
        self.health = 0
        self.speed = 0
        self.abilities = []
        self.team = 0

    def is_dead(self):
        return self.health <= 0


def roll_dice(amount, sides):
    return np.floor(np.random.uniform(size=amount, low=1., high=sides)).sum()


class Ability:

    def apply(self, teams, user, target):
        pass


class Sword(Ability):
    def apply(self, teams, user, target):
        friendly_team, enemy_team = teams.view(user.team)
        target_body = enemy_team[target]
        target_body.health -= roll_dice(1, 8)


class Heal(Ability):
    def apply(self, teams, user, target):
        friendly_team, enemy_team = teams.view(user.team)
        target_body = friendly_team[target]
        target_body.health += roll_dice(1, 6)


class Player(Body):
    def __init__(self):
        super().__init__()
        self.health = 30
        self.speed = 5
        self.abilities = [Sword(), Heal()]


class Enemy(Body):
    def __init__(self):
        super().__init__()
        self.health = 30
        self.speed = 3
        self.abilities = [Sword()]
        self.policy = None


class Dungeon(gymnasium.Env):

    def __init__(self, render_mode='human'):
        self.render_mode = render_mode
        self.slots = []
        self.observation_space = Box(low=0., high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete((1, 2), dtype=np.int64)
        self.player_slots = []
        self.enemy_slots = []
        self.observation = None
        self.reset()

    def reset(self, *args, seed=None):
        self.player_slots = [Player()]
        self.enemy_slots = [Enemy()]
        self.update_observation()
        return self.observation

    def background(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def update_observation(self):
        image = self.background()
        image = cv2.rectangle(image, (0, 0), (32, 64), (0, 0, 255), thickness=-1)
        image = cv2.rectangle(image, (32, 0), (64, 64), (255, 0, 0), thickness=-1)
        self.observation = image

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.player_turn(self.player_slots[0], action)
        self.enemy_turn(self.enemy_slots[0], action)
        self.update_observation()
        reward = 0.
        reward += -1. if self.player_slots[0].is_dead() else 0.
        reward += 1. if self.enemy_slots[0].is_dead() else 0.
        terminated = self.enemy_slots[0].is_dead() or self.player_slots[0].is_dead()
        return self.observation, reward, terminated, False, {}

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        cv2.imshow(self.observation)
        cv2.waitKey(0)
        return self.observation


if __name__ == '__main__':

    env = Dungeon()
    observation = env.reset()
    terminated = False
    while not terminated:
        env.step(env.action_space.sample())