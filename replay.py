from envs.env import Env
import torch
from collections import deque
import numpy as np


class Step:
    def __init__(self, observation, action, reward=0., terminated=None, truncated=None, cont=None, **kwargs):
        """

        :param observation:
        :param action: set to all zeros if no action
        :param reward:
        :param terminated:
        :param truncated:
        """
        self.observation = observation
        self.action = action
        self.reward = np.array([reward], dtype=np.float32)
        assert (terminated is None) != (cont is None), 'must specify terminated or cont but not both'
        if terminated is not None:
            self.cont = np.logical_not(np.array([terminated])) * 1.
        else:
            self.cont = cont
        self.terminated = terminated
        self.truncated = truncated
        self.info = kwargs

    def as_tuple(self):
        return self.observation, self.action, self.reward, self.cont

    def __repr__(self):
        return str(self.as_tuple())

    @property
    def is_terminal(self):
        return self.terminated or self.truncated


def get_trajectory(buff, offset, truncate_len=None):
    step = offset
    trajectory = []
    while step < len(buff):
        if truncate_len is not None:
            if len(trajectory) == truncate_len:
                break
        trajectory.append(buff[step])
        if buff[step].is_terminal:
            break
        step += 1
    return trajectory


def stack_trajectory(trajectory):
    arrays = [[], [], [], []]
    for step in trajectory:
        for i, array in enumerate(arrays):
            array += [step.as_tuple()[i]]
    for i, array in enumerate(arrays):
        arrays[i] = np.stack(arrays[i])
    return arrays


def stack_batch(stacked_trajectories):
    arrays = [[], [], [], []]
    for stacked_trajectory in stacked_trajectories:
        for i, array in enumerate(arrays):
            array += [stacked_trajectory[i]]
    for i, array in enumerate(arrays):
        arrays[i] = np.stack(array, axis=1)
    return arrays


def sample_batch(buffer, length, batch_size, max_rejects=100):
    trajectories = []
    rejects = 0
    while len(trajectories) < batch_size:
        offset = torch.randint(0, len(buffer) - 1, (1,)).item()
        trajectory = get_trajectory(buffer, offset, length)
        if len(trajectory) == length:
            trajectories += [stack_trajectory(trajectory)]
        else:
            rejects += 1
            if rejects > max_rejects:
                raise Exception(f'sampled {max_rejects} trajectories shorter than batch_length, shorten the batch_length')
    observations, actions, rewards, cont = tuple(stack_batch(trajectories))
    return observations, actions, rewards, cont


def transform_rgb_image(rgb_array):
    return torch.from_numpy(rgb_array).permute(0, 1, 4, 2, 3) / 255.0


class BatchLoader:
    def __init__(self, device='cpu', observation_transform=None):
        self.device = device
        self.observation_transform = observation_transform

    def sample(self, replay_buffer, batch_length, batch_size):
        """
        Samples with replacement, ignores trajectories that are shorter than batch length
        :param replay_buffer:
        :param batch_length:
        :param batch_size:
        :return: overvation, action, reward, cont
        """

        observation, action, reward, cont = sample_batch(replay_buffer, batch_length, batch_size)
        if self.observation_transform is not None:
            observation = self.observation_transform(observation).to(device=self.device, dtype=torch.float32)
        else:
            observation = torch.from_numpy(observation).to(device=self.device, dtype=torch.float32)

        action = torch.from_numpy(action).to(device=self.device, dtype=torch.float32)
        reward = torch.from_numpy(reward).to(device=self.device, dtype=torch.float32)
        cont = torch.from_numpy(cont).to(device=self.device, dtype=torch.float32)

        return observation, action, reward, cont


def get_trajectories(buff, max_trajectories=None):
    offset = 0
    trajectories = []
    while offset < len(buff):
        if max_trajectories is not None:
            if len(trajectories) < max_trajectories:
                break
        trajectories += [get_trajectory(buff, offset)]
        offset += len(trajectories[-1])
    return trajectories


def get_tail_trajectory(buff):
    """
    returns the most recent trajectory in the buffer
    """
    trajectory = deque()
    for i in reversed(range(0, len(buff))):
        trajectory.appendleft(buff[i])
        if i == 0:
            return trajectory
        if buff[i-1].is_terminal:
            return trajectory


def total_reward(trajectory):
    t_reward = 0
    for step in trajectory:
        t_reward += step.reward
    return t_reward[0]


def unroll(tensor):
    return [trajectory.aslist() for trajectory in tensor.unbind(1)]


def unstack_batch(observation, action, reward, cont, inverse_obs_transform=None, **kwargs):
    T, N = observation.shape[0:2]
    if inverse_obs_transform is not None:
        observation = inverse_obs_transform(observation)
    buff = []
    for n in range(N):
        for t in range(T):
            obs_step = observation[t, n].detach().cpu().numpy()
            action_step = action[t, n].detach().cpu().numpy()
            reward_step = reward[t, n].detach().cpu().numpy()
            cont_step = cont[t, n].detach().cpu().numpy()
            arg_step = {arg: kwargs[arg][t, n].detach().cpu().numpy() for arg in kwargs}
            buff += [Step(obs_step, action_step, reward_step, cont=cont_step, **arg_step)]
    return buff


if __name__ == '__main__':

    buff = deque()
    env = Env()

    def rollout_open_loop_policy(env, actions):
        state, reward, terminated, truncated = env.reset(), 0., False, False
        for action in actions:
            yield Step(state, action, reward, terminated, truncated)
            state, reward, terminated, truncated, _ = env.step(action)
        yield Step(state, Env.pad_action, reward, terminated, truncated)


    go_right = [Env.right] * 8
    go_left = [Env.left]

    for step in rollout_open_loop_policy(env, go_left):
        buff.append(step)

    for step in rollout_open_loop_policy(env, go_left):
        buff.append(step)

    for step in rollout_open_loop_policy(env, go_left):
        buff.append(step)

    trajectory_1 = get_trajectory(buff, 0)
    trajectory_stacked_1 = stack_trajectory(trajectory_1)

    batch = stack_batch([stack_trajectory(get_trajectory(buff, i)) for i in [0, 2, 4]])

    for step in rollout_open_loop_policy(env, go_left):
        buff.append(step)
        batch = sample_batch(buff, length=2, batch_size=4)

    for step in rollout_open_loop_policy(env, go_right):
        buff.append(step)
        sample_batch(buff, length=2, batch_size=4)

    loader = BatchLoader()
    observation, action, reward, cont = loader.sample(buff, 2, 10)

    total_reward(trajectory_1)
    values = torch.zeros_like(reward)
    buff = unstack_batch(observation, action, reward, cont, values=values)

    print(buff)