from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple


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


State = namedtuple("State", ["pos", "direction", "goal"])


class Renderer:
    def __init__(self, env):
        self.env = env
        self.empty_tile = Image.open('empty_tile.bmp')
        self.goal_tile = Image.open('goal_tile.bmp')
        self.arrow = [Image.open(f'arrow_{direction}.bmp') for direction in range(0, 4)]

    def _background(self):
        self.grid = []
        for x in range(self.env.min.x, self.env.max.x + 1):
            self.grid += [[]]
            for y in range(self.env.min.y, self.env.max.y + 1):
                if x == self.env.goal.x and y == self.env.goal.y:
                    self.grid[x] += [self.goal_tile]
                else:
                    self.grid[x] += [self.empty_tile]

    def _concatentate_tiles(self):
        rows = []
        for row in self.grid:
            rows += [np.concatenate(row, axis=0)]
        return np.concatenate(rows, axis=1)

    def draw(self, pos, direction):
        self._background()
        self.grid[pos.x][pos.y] = np.add(self.arrow[direction], self.grid[pos.x][pos.y])
        return self._concatentate_tiles()


class RGBImageWrapper:
    def __init__(self, env):
        self.env = env
        self.renderer = Renderer(env)

    def reset(self):
        obs = self.env.reset()
        return self.renderer.draw(obs.pos, obs.direction)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.renderer.draw(obs.pos, obs.direction), reward, terminated, truncated, info


class SimpleGridWorld:
    DIRECTION = [Vector2(0, -1), Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0)]

    def __init__(self, start_pos, start_direction, width, height, goal_pos, render_mode=None):
        """
        creates a simple grid
        :param start_pos: Vector2(x, y) starting position
        :param start_direction: 0 -> up, 1 -> right, 2 -> down, 3 -> left
        :param width: width of grid
        :param height: height of grid
        :param goal_pos: goal_pos in grid
        """
        self.pos = start_pos
        self.direction = start_direction
        self.max = Vector2(width-1, height-1)
        self.min = Vector2(0, 0)
        self.goal = goal_pos
        self.render_mode = render_mode
        self.render = Renderer(self)
        if render_mode == 'human':
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax_image = self.ax.imshow(self.render.draw(self.pos, self.direction))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def draw(self):
        if self.render_mode == 'human':
            self.ax_image.set_data(self.render.draw(self.pos, self.direction))
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def reset(self):
        self.draw()
        return State(self.pos, self.direction, self.goal)

    def step(self, action):
        """
        steps the environment
        :param action: 0 -> Turn left, 1 -> turn right, 2 -> forward
        """
        if action == 0:  # left
            self.direction = (self.direction - 1) % 4
        if action == 1:  # right
            self.direction = (self.direction + 1) % 4
        if action == 2:  # forward
            lookahead = self.pos + self.DIRECTION[self.direction]
            if self.max >= lookahead >= self.min:
                self.pos += self.DIRECTION[self.direction]
        reward = 1.0 if self.pos == self.goal else 0.0
        terminated = True if self.pos == self.goal else False
        truncated = False

        self.draw()

        return State(self.pos, self.direction, self.goal), reward, terminated, truncated, {}


GridConf = namedtuple('env_config', ['start_pos', 'start_direction', 'width', 'height', 'goal'])

env_configs = {
    '9x9': GridConf(start_pos=Vector2(0, 0), start_direction=0, width=3, height=3, goal=Vector2(2, 2)),
    'bandit': GridConf(start_pos=Vector2(0, 0), start_direction=1, width=2, height=1, goal=Vector2(1, 0)),
    'corridor': GridConf(start_pos=Vector2(0, 0), start_direction=1, width=5, height=1, goal=Vector2(4, 0)),
}


def make(env_name, render_mode=None):
    return SimpleGridWorld(*env_configs[env_name], render_mode=render_mode)


if __name__ == '__main__':

    env = make('9x9', render_mode='human')
    env = RGBImageWrapper(env)
    obs = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        try:
            action = input('press 0, 1, 2, q')
            if action == 'q':
                break
            action = int(action)
            if 0 <= action <= 2:
                obs, reward, terminated, truncated, info = env.step(action)
        except ValueError:
            continue
