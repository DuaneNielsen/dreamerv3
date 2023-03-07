from __future__ import annotations

from typing import SupportsFloat, Any, Tuple, Dict

import gymnasium
from gymnasium.core import RenderFrame, ActType, ObsType, WrapperObsType
from gymnasium.spaces import Discrete, Box
import numpy as np
from copy import deepcopy
import cv2
from uuid import uuid4
from numpy.linalg import norm
from numpy.random import randint

yellow = (0, 255, 255)
red = (0, 0, 255)
pink = (255, 0, 255)
violet = (138, 43, 226)


class Ghost:
    def __init__(self):
        self.color = red
        self.start_pos = None
        self.prev_reverse_action = None
        self.counter = 0
        self.mode = 'chase'
        self.id = uuid4()
        self.effects = {}

    def reset(self):
        self.pos = deepcopy(self.start_pos)
        self.prev_reverse_action = None
        self.counter = 0
        self.mode = 'scatter'

    def tick_status(self):
        expired = []
        for effect in self.effects:
            self.effects[effect] -= 1
            if self.effects[effect] == 0:
                expired += [effect]
        for effect in expired:
            del self.effects[effect]
        return expired

    def get_target(self, mode, env):
        if mode == 'chase':
            return env.pos
        else:
            return self.start_pos

    def step(self, env):
        now_tile = env.grid[self.pos[0]][self.pos[1]]
        choices = [env.lookahead(self.pos, action) for action in range(env.action_space.n.item())]

        if 'frightened' in self.effects:
            if self.counter < 70:
                self.mode = 'scatter'
            else:
                self.mode = 'chase'

        target = self.get_target(self.mode, env)

        preferred_choice = []

        for (tile, pos), action in zip(choices, range(env.action_space.n.item())):
            if tile.traversable:
                d = norm(target - pos)
                preferred_choice += [(action, d, pos, tile)]

        preferred_choice = sorted(preferred_choice, key=lambda choice: choice[1])

        if self.prev_reverse_action is not None:
            if preferred_choice[0][0] == self.prev_reverse_action:
                preferred_choice = [preferred_choice[1]]

        if 'frightened' in self.effects and len(preferred_choice) > 2:
            randdir = randint(0, len(preferred_choice))
            selected_action, _, selected_pos, selected_tile = preferred_choice[randdir]
        else:
            selected_action, _, selected_pos, selected_tile = preferred_choice[0]

        self.pos = selected_pos
        self.prev_reverse_action = reverse_action(selected_action)
        del now_tile.monster_stack[self.id]
        selected_tile.monster_stack[self.id] = self

        if 'frightened' in self.effects:
            self.counter += 1
            self.counter = self.counter % 270


class Blinky(Ghost):
    def __init__(self):
        super().__init__()
        self.color = (0, 0, 255)


class Pinky(Ghost):
    def __init__(self):
        super().__init__()
        self.color = pink

    def get_target(self, mode, env):
        if mode == 'chase':
            return env.pos + SimplerGridWorld.moves[env.action] * 4
        else:
            return self.start_pos


class Clyde(Ghost):
    def __init__(self):
        super().__init__()
        self.color = (0, 165, 255)

    def get_target(self, mode, env):
        if mode == 'chase':
            if norm(self.pos - env.pos) < 8.:
                return self.start_pos
            else:
                return env.pos
        else:
            return self.start_pos


class Inky(Ghost):
    def __init__(self):
        super().__init__()
        self.color = (255, 0, 0)

    def get_target(self, mode, env):
        if mode == 'chase':
            vec = (env.pos + SimplerGridWorld.moves[env.action] * 2) - self.pos
            return self.pos + vec * 2
        else:
            return self.start_pos


def reverse_action(action):
    return (action + 2) % 4


class Item:
    def __init__(self, color, reward):
        self.color = color
        self.reward = reward

    def effect(self, env):
        pass

    def effect_ghost(self, ghost):
        pass


class PowerPill(Item):
    def __init__(self, color, reward):
        super().__init__(color, reward)

    def effect(self, env):
        env.energised = 30

    def effect_ghost(self, ghost):
        ghost.effects['frightened'] = 30


class Tile:
    def __init__(self, color, start_pos=False, monster=None, traversable=True, terminal=False, reward=0., stack=None):
        self._color = color
        self.traversable = traversable
        self.terminal = terminal
        self.reward = reward
        self.start_pos = start_pos
        self.monster_stack = {monster.id: monster} if monster else {}
        self.stack = [] if stack is None else stack

    @property
    def has_item(self):
        return len(self.stack) > 0

    @property
    def has_monster(self):
        return len(self.monster_stack) > 0

    @property
    def color(self):
        if self.has_monster:
            for monster in self.monster_stack:
                return self.monster_stack[monster].color
        elif self.has_item:
            return self.stack[-1].color
        else:
            return self._color


class SimplerGridWorld(gymnasium.Env):
    moves = np.array([
        [0, -1],  # N
        [1, 0],  # E
        [0, 1],  # S
        [-1, 0]  # W
    ], dtype=np.int64)

    def __init__(self, world_name, render_mode='human'):
        super().__init__()
        self.color = yellow
        self.world_name = world_name
        self.render_mode = render_mode
        self.grid, self.grid_size, self.start_pos, self.monsters = make_grid(world_name)
        self.observation_space = Box(low=0, high=max(self.grid_size), shape=self.grid_size, dtype=np.int64)
        self.action_space = Discrete(4)
        self.render_speed = 1
        self.uuid = str(uuid4())
        self.reset()
        self.action = 0
        self.energised = 0

    def lookahead(self, pos, action):
        next_pos = pos + self.moves[action]
        next_pos[0] = np.clip(next_pos[0], 0, self.grid_size[0] - 1)
        next_pos[1] = np.clip(next_pos[1], 0, self.grid_size[1] - 1)
        next_tile = self.grid[next_pos[0]][next_pos[1]]
        return next_tile, next_pos

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        self.grid, self.grid_size, self.pos, self.monsters = make_grid(self.world_name)

        for monster in self.monsters.values():
            monster.reset()

        if self.render_mode == 'human':
            self.render()

        return (self.pos, self.grid, self.color), {}

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        self.action = action
        reward = 0.

        now_tile = self.grid[self.pos[0]][self.pos[1]]
        if now_tile.has_item:
            now_tile.stack[-1].effect(self)
            for ghost in self.monsters.values():
                now_tile.stack[-1].effect_ghost(ghost)
            now_tile.stack.pop()

        if self.energised > 0:
            self.color = violet
        else:
            self.color = yellow
        self.energised -= 1

        if now_tile.has_monster:
            if self.energised > 0:
                monster_id = [monster for monster in now_tile.monster_stack]
                for monster in monster_id:
                    del self.monsters[monster]
                    del now_tile.monster_stack[monster]
            else:
                return (self.pos, self.grid, self.color), -1.0, True, False, {}

        next_tile, next_pos = self.lookahead(self.pos, action)

        if next_tile.traversable:
            self.pos = next_pos

        reward += next_tile.reward
        terminated = next_tile.terminal

        if next_tile.has_item:
            reward += next_tile.stack[-1].reward

        if next_tile.has_monster:
            if self.energised > 0:
                monster_id = [monster for monster in next_tile.monster_stack]
                for monster in monster_id:
                    del self.monsters[monster]
                    del next_tile.monster_stack[monster]
                reward += 1
            else:
                reward += -1.
                terminated = True

        for monster in self.monsters.values():
            monster.step(self)

        if self.render_mode == 'human':
            self.render()

        return (self.pos, self.grid, self.color), reward, terminated, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        base_size = self.grid_size + (3,)
        image = np.zeros(base_size, dtype=np.uint8)
        for k in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                image[k, j, :] = np.array(self.grid[k][j].color)

        image[self.pos[0].item(), self.pos[1].item(), 0:3] = np.array(list(self.color))
        image = image.swapaxes(1, 0)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow(f'SimplerGridWorld - {self.uuid[0:5]}', image)
        cv2.waitKey(self.render_speed)
        return image

    def get_action_meanings(self):
        return ['\u2191', '\u2192', '\u2193', '\u2190']


class RGBObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env, h=64, w=64):
        super().__init__(env)
        self.h, self.w = h, w

    def observation(self, observation: ObsType) -> WrapperObsType:
        pos, grid, color = observation
        base_size = self.grid_size + (3,)
        image = np.zeros(base_size, dtype=np.uint8)
        for k in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                image[k, j, :] = np.array(grid[k][j].color)

        image[self.obs_dist, self.obs_dist, :] = np.array(color)
        image = image.swapaxes(1, 0)
        image = cv2.resize(image, (self.h, self.w), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class PartialRGBObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env, h=64, w=64, obs_dist=5):
        super().__init__(env)
        self.h, self.w = h, w
        self.obs_dist = obs_dist
        self.observation_space = Box(low=0, high=255, shape=(obs_dist, obs_dist))

    def observation(self, observation: ObsType) -> WrapperObsType:
        (x, y), grid, color = observation
        base_size = self.grid_size + (3,)
        image = np.zeros(base_size, dtype=np.uint8)
        for k in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                image[k, j, :] = np.array(grid[k][j].color)

        image = np.pad(image, pad_width=[(self.obs_dist, self.obs_dist), (self.obs_dist, self.obs_dist), (0, 0)],
                       constant_values=255)
        image = image[x:x + self.obs_dist * 2 + 1, y:y + self.obs_dist * 2 + 1]

        image[self.obs_dist, self.obs_dist, :] = np.array(color)
        image = image.swapaxes(1, 0)
        image = cv2.resize(image, (self.h, self.w), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


items = {
    'reward_pack': Item(color=(20, 150, 20), reward=1.0),
    'small_pill': Item(color=(110, 80, 40), reward=0.01),
    'power_pill': PowerPill(color=(20, 150, 20), reward=1.0),
}

tiles = {
    's': Tile(color=(0, 0, 0), start_pos=True),
    'e': Tile(color=(0, 0, 0)),
    'g': Tile(color=(0, 255, 0), terminal=True, reward=1.),
    'l': Tile(color=(70, 100, 200), terminal=True, reward=-1.),
    'w': Tile(color=(255, 255, 255), traversable=False, reward=-0.01),
    'p': Tile(color=(0, 0, 0), stack=[items['reward_pack']]),
    'z': Tile(color=(0, 0, 0), stack=[items['power_pill']]),
    'd': Tile(color=(0, 0, 0), stack=[items['small_pill']]),
    'b': Tile(color=(0, 0, 0), stack=[items['small_pill']], monster=Blinky()),
    'n': Tile(color=(0, 0, 0), stack=[items['small_pill']], monster=Pinky()),
    'i': Tile(color=(0, 0, 0), stack=[items['small_pill']], monster=Inky()),
    'c': Tile(color=(0, 0, 0), stack=[items['small_pill']], monster=Clyde()),
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
        'wwwwwwwwwwwwwwww',
        'weepeeeeeeeeeeew',
        'weeeeeeeeeeepeew',
        'weeeeeeeeeeeeeew',
        'wepeeeeeeeeeeeew',
        'weeeeeeeeeeeeeew',
        'weeeeeeeeeepeeew',
        'weeeeeeeeeeeeeew',
        'weeeeeeeseeeeeew',
        'weeeeeeeeeeeeeew',
        'wepeeeeeeeeeeeew',
        'weeeeeeeeeeepeew',
        'weeeeeeeeeeeeeew',
        'weeeepeeeeeeeeew',
        'weeeeeeeeeeeeeew',
        'wwwwwwwwwwwwwwww'
    ],

    'maze': [
        'wwwwwwwwwwwwwwww',
        'weepeeeeeeeeeeew',
        'wewwwwwweeeepeew',
        'weeeeepweeeeeeew',
        'wewwwwewewwwwwww',
        'wepeeweweweeeeew',
        'weewewewewepeeew',
        'weewewewewwwweww',
        'weeeeeeeseeeeeew',
        'wwwwwwwwewwwwwew',
        'wepeeeeweweeewew',
        'weeeeeeweeeepwpw',
        'weeeeeeeeweeewew',
        'wewwwpwwewwwewew',
        'wpeeeeeeeeeeeeew',
        'wwwwwwwwwwwwwwww'
    ],

    'frozen_lake': [
        'leeegeeeee',
        'eeeleeelee',
        'eeeelleeee',
        'eeleeeelee',
        'eeeeseeeee'
    ],

    'pacman': [
        'wwwwwwwwwwwwwwwwwww',
        'wdbddddddwddddddidw',
        'wzwwdwwwdwdwwwdwwzw',
        'wdddddddddddddddddw',
        'wdwwdwdwwwwwdwdwwdw',
        'wddddwdddwdddwddddw',
        'wwwwdwwwdwdwwwdwwww',
        'wwwwdwdddddddwdwwww',
        'wwwwdddwwwwwdddwwww',
        'wwwwdwdwwwwwdwdwwww',
        'wwwwdwdddzdddwdwwww',
        'wwwwdwdwwwwwdwdwwww',
        'wddddddddwddddddddw',
        'wdwwdwwwdwdwwwdwwdw',
        'wzdwddddzszddddwdzw',
        'wwdwdwdwwwwwdwdwdww',
        'wddddwdddwdddwddddw',
        'wdwwwwwwdwdwwwwwwdw',
        'wdcdddddddddddndddw',
        'wwwwwwwwwwwwwwwwwww'
    ]
}

world_config = {
    'pacman': {
        'max_episode_steps': 1000,
    }
}


def world_wh(world):
    return len(world[0]), len(world)


def make_grid(world_name):
    world = worlds[world_name]
    grid = []
    start_pos = np.array([0, 0], dtype=np.int64)
    monsters = {}
    w, h = world_wh(world)
    for x in range(w):
        grid += [[]]
        for y in range(h):
            tile = deepcopy(tiles[world[y][x]])
            grid[x] += [tile]
            if tile.start_pos:
                start_pos = np.array([x, y], dtype=np.int64)
            if tile.has_monster:
                for monster in tile.monster_stack:
                    tile.monster_stack[monster].start_pos = np.array([x, y], dtype=np.int64)
                    monsters[monster] = tile.monster_stack[monster]

    return grid, world_wh(world), start_pos, monsters


if __name__ == '__main__':

    import envs

    env = gymnasium.make('SimplerGridWorld-pacman-v0', render_mode='human')
    env.unwrapped.render_speed = 200
    env = PartialRGBObservationWrapper(env)
    obs, info = env.reset()
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        obs = cv2.resize(obs, (256, 256))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        cv2.imshow('observation', obs)
        cv2.waitKey(100)
