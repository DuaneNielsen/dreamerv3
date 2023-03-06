import gymnasium
import numpy as np
from gymnasium.core import WrapperActType, ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box


class OneHotActionWrapper(gymnasium.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

    def action(self, action: WrapperActType) -> ActType:
        return action.argmax(-1).item()


class MaxCombineObservations(gymnasium.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[1:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return np.maximum(observation[0], observation[1])
