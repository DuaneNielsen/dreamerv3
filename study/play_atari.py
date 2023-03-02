import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np

play(gym.make("ALE/Breakout-v5",
              render_mode="rgb_array"),
     keys_to_action={
         "a": 3,
         "d": 2,
     }, noop=0)
