import symlog
from gridworlds.env import Env
import torch
from collections import deque
import numpy as np
import statistics
from PIL import Image
import io


class Step:
    def __init__(self, observation, action, reward=0., terminated=None, truncated=None, cont=None, pred_step=None, **kwargs):
        """

        :param observation:
        :param action: set to all zeros if no action
        :param reward:
        :param terminated:
        :param truncated:
        """
        self._observation = observation
        self.action = action
        self.reward = np.array([reward], dtype=np.float32)
        assert (terminated is None) != (cont is None), 'must specify terminated or cont but not both'
        if terminated is not None:
            self.cont = np.logical_not(np.array([terminated])) * 1.
        else:
            self.cont = cont
        self.terminated = terminated
        self.truncated = truncated
        self.pred_step = pred_step
        self.info = kwargs

    @property
    def observation(self):
        return self._observation

    def as_tuple(self):
        return self.observation, self.action, self.reward, self.cont

    def __repr__(self):
        return str(self.as_tuple())

    @property
    def is_terminal(self):
        return self.terminated or self.truncated


class PNGStep(Step):
    def __init__(self, observation, action, reward=0., terminated=None, truncated=None, cont=None, **kwargs):

        observation = Image.fromarray(observation)
        file = io.BytesIO()
        observation.save(file, format='PNG')
        file = file.getvalue()
        super().__init__(file, action, reward=reward, terminated=terminated, truncated=truncated, cont=cont, **kwargs)

    @property
    def observation(self):
        pic = Image.open(io.BytesIO(self._observation))
        return np.array(pic)


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


def symlog_rgb_image(rgb_array):
    return symlog.symlog(torch.from_numpy(rgb_array).permute(0, 1, 4, 2, 3).float())


def invert_symlog(symlog_rgb_tensor):
    lead_dims = symlog_rgb_tensor.shape[:-3]
    image = symlog_rgb_tensor.flatten(0, len(lead_dims)-1)
    image = image.permute(0, 2, 3, 1)
    image = image.unflatten(0, lead_dims)
    return symlog.symexp(image).to(dtype=torch.uint8)


def invert_to_tensor(image):
    lead_dims = image.shape[:-3]
    image = image.flatten(0, len(lead_dims)-1)
    image = image.permute(0, 2, 3, 1)
    image = image.unflatten(0, lead_dims)
    return (image * 255).to(dtype=torch.uint8)


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

        observation = torch.from_numpy(observation).to(device=self.device, dtype=torch.float)
        action = torch.from_numpy(action).to(device=self.device, dtype=torch.float)
        reward = torch.from_numpy(reward).to(device=self.device, dtype=torch.float)
        cont = torch.from_numpy(cont).to(device=self.device, dtype=torch.float)

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


def sum_key(trajectory, key):
    sum = 0
    for step in trajectory:
        if key in step.info:
            sum += step.info[key]
    return sum[0]


def pred_reward(trajectory):
    t_reward = 0
    for step in trajectory:
        t_reward += step.pred_step.reward
    return t_reward[0]


def reward_mse(trajectory):
    error = []
    for step in trajectory:
        error += [(step.reward.item() - step.pred_step.reward.item()) ** 2]
    return statistics.mean(error)


def cont_mse(trajectory):
    error = []
    for step in trajectory:
        error += [(step.cont.item() - step.pred_step.cont.item()) ** 2]
    return statistics.mean(error)


def observation_mse(trajectory):
    error = []
    for step in trajectory:
        error += [np.mean((step.observation - step.pred_step.observation) ** 2)]
    return statistics.mean(error)


def unroll(tensor):
    return [trajectory.aslist() for trajectory in tensor.unbind(1)]


def unstack_batch(observation, action, reward, cont, return_type=None, **kwargs):
    """
    input in T, N, ... format
    """
    T, N = observation.shape[0:2]
    buff = []

    observation = observation.detach().cpu().numpy()
    action = action.detach().cpu().numpy()
    reward = reward.detach().cpu().numpy()
    cont = cont.detach().cpu().numpy()
    kwargs = {arg: kwargs[arg].detach().cpu().numpy() for arg in kwargs}

    if return_type is None:
        for n in range(N):
            for t in range(T):
                args_step = {arg: kwargs[arg][t, n] for arg in kwargs}
                buff += [Step(observation[t, n], action[t, n], reward[t, n], cont=cont[t, n], **args_step)]
        return buff

    if return_type == 'list':
        for n in range(N):
            trajectory = []
            for t in range(T):
                args_step = {arg: kwargs[arg][t, n] for arg in kwargs}
                trajectory += [Step(observation[t, n], action[t, n], reward[t, n], cont=cont[t, n], **args_step)]
            buff += [trajectory]
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