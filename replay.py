import encoding
from envs.env import Env
import torch
from collections import deque, namedtuple
from symlog import symlog
from encoding import encode_onehot, decode_onehot

"""
x -> state or observation
a -> action
r -> reward
c -> continue, 0.0 means terminal state
"""
Step = namedtuple("step", ['obs', 'act', 'reward', 'cont'])


def is_trajectory_end(step):
    """
    End of trajectory

    NOTE:  End of trajectory is not the same as terminal state!

    Terminal state means that the environment ended the trajectory, ie: agent is unable to get any future reward

    End of trajectory means that is all we know about the trajectory, usually because the agent stopped
    interacting with the environment

    Thus we set the action to None to signify that the agent took no action in this state.

    If the state was terminal, then the action must be None.  As the agent cannot take an action in the terminal
    state.

    However, we will pad this None with a dummy action when sampling the batch.  This won't affect the world model
    as the RSSM algorithm does not use the last action in a treajectory for any purpose.
    It only uses x, r and c on the final step.  (a is only used to predict the next step)

    We will need to keep an eye out if this padding scheme will effect the value function.
    It may if the value function is of the x, a, next_x variety

    :param step: Step tuple
    :return: True if this state is terminal
    """
    return step.act is None


def get_trajectory(buff, offset, truncate_len=None):
    step = offset
    trajectory = []
    while step < len(buff):
        if truncate_len is not None:
            if len(trajectory) == truncate_len:
                break
        trajectory.append(buff[step])
        if is_trajectory_end(buff[step]):
            break
        step += 1
    return trajectory


def pad_trajectory(trajectory, length, pad_state, pad_action):
    pad = Step(pad_state, pad_action, 0.0, 0.0)
    orig_len = len(trajectory)
    pad_length = length - len(trajectory)
    trajectory = trajectory + [pad] * pad_length
    mask = [True] * orig_len + [False] * pad_length
    return trajectory, mask


def stack_trajectory(trajectory, pad_action):
    obs, action, reward, cont = [], [], [], []
    for step in trajectory:
        obs += [step.obs]
        if is_trajectory_end(step):
            action += [pad_action]
        else:
            action += [step.act]
        reward += [torch.tensor([step.reward])]
        cont += [torch.tensor([step.cont])]
    return torch.stack(obs), torch.stack(action), torch.stack(reward), torch.stack(cont)


def make_batch_elegant(buff, offsets, length, pad_state, pad_action):
    obs_b, action_b, reward_b, cont_b, mask_b = [], [], [], [], []
    for offset in offsets:
        trajectory = get_trajectory(buff, offset, length)
        trajectory, mask = pad_trajectory(trajectory, length, pad_state, pad_action)
        obs, action, reward, cont = stack_trajectory(trajectory, pad_action)
        mask = torch.tensor(mask)
        obs_b += [obs]
        action_b += [action]
        reward_b += [reward]
        cont_b += [cont]
        mask_b += [mask]
    obs_b = torch.stack(obs_b, dim=1)
    action_b = torch.stack(action_b, dim=1)
    reward_b = torch.stack(reward_b, dim=1)
    cont_b = torch.stack(cont_b, dim=1)
    mask_b = torch.stack(mask_b, dim=1).unsqueeze(-1)
    return obs_b, action_b, reward_b, cont_b, mask_b


def make_batch(buffer, offsets, length, pad_state, pad_action):
    """
    Make a batch from the replay buffer
    :param offsets: list of offsets of length batch size to sample
    :param buffer: replay buffer to sample from
    :param length: number of timesteps to sample T
    :return: x -> [T, N, ... ], a -> [T, N, ...], r -> [T, N, 1], c -> [T, N, 1], m -> [T, N, 1]
    """

    pad = [False] * len(offsets)

    x, a, r, c, mask = [], [], [], [], []

    for t in range(length):
        x_i, a_i, r_i, c_i, m_i = [], [], [], [], []
        for n, o in enumerate(offsets):
            if pad[n] or o + t >= len(buffer):
                x_i += [pad_state]
                a_i += [pad_action]
                r_i += [torch.zeros(1)]
                c_i += [torch.zeros(1)]
                m_i += [torch.tensor([False])]
            else:
                x_i += [buffer[o + t].obs]
                if is_trajectory_end(buffer[o + t]):
                    a_i += [pad_action]
                    pad[n] = True
                else:
                    a_i += [buffer[o + t].act]
                r_i += [torch.tensor([buffer[o + t].reward])]
                c_i += [torch.tensor([buffer[o + t].cont])]
                m_i += [torch.tensor([True])]

        x += [torch.stack(x_i)]
        a += [torch.stack(a_i)]
        r += [torch.stack(r_i)]
        c += [torch.stack(c_i)]
        mask += [torch.stack(m_i)]

    return torch.stack(x), torch.stack(a), torch.stack(r), torch.stack(c), torch.stack(mask)


def sample_batch(buffer, length, batch_size, pad_state, pad_action):
    """
    Sample a batch from the replay buffer
    :param buffer: replay buffer to sample from
    :param length: number of timesteps to sample T
    :param batch_size: batch size N
    :return: x -> [T, N, ... ], a -> [T, N, ...], r -> [T, N, 1], c -> [T, N, 1], m -> [T, N, 1]
    """
    offsets = torch.randint(0, len(buffer)-1, (batch_size,)).tolist()
    return make_batch_elegant(buffer, offsets, length, pad_state, pad_action)


class BatchLoader:
    def __init__(self, pad_observation, pad_action, obs_codec=None, reward_codec=None, device='cpu'):
        self.pad_observation = pad_observation
        self.pad_action = pad_action
        self.obs_codec = obs_codec
        self.codec = reward_codec
        self.device = device

    def sample(self, replay_buffer, batch_length, batch_size):

        observation, action, reward, cont, mask = \
            sample_batch(replay_buffer, batch_length, batch_size, self.pad_observation, self.pad_action)

        if self.obs_codec:
            observation = self.obs_codec.encode(observation)

        if self.codec:
            reward = self.codec.encode(reward)

        observation = observation.to(self.device).detach()
        action = action.to(self.device).detach()
        reward = reward.to(self.device).detach()
        cont = cont.to(self.device).detach()
        mask = mask.to(self.device).detach()

        return observation, action, reward, cont, mask


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
        if is_trajectory_end(buff[i-1]):
            return trajectory


def total_reward(trajectory):
    t_reward = 0
    for step in trajectory:
        t_reward += step.reward
    return t_reward


if __name__ == '__main__':

    buff = deque()
    env = Env()

    def rollout_open_loop_policy(env, actions):
        x, r, c = env.reset(), 0, 1.0
        for a in actions:
            yield Step(x, a, r, c)
            x, r, done, _ = env.step(a)
            c = 0.0 if done else 1.0
        yield Step(x, None, r, c)


    go_right = [Env.right] * 8
    go_left = [Env.left]

    for step in rollout_open_loop_policy(env, go_left):
        buff.append(step)
        sample_batch(buff, 10, 4, Env.pad_state, Env.pad_action)
    for step in rollout_open_loop_policy(env, go_right):
        buff.append(step)
        sample_batch(buff, 10, 4, Env.pad_state, Env.pad_action)

    print(buff)
    print([is_trajectory_end(s) for s in buff])

    x, a, r, c, m = sample_batch(buff, 10, 4, Env.pad_state, Env.pad_action)

    assert x.shape == (10, 4, 1, 10)
    assert a.shape == (10, 4, 1, 10)
    assert c.shape == (10, 4, 1)
    assert r.shape == (10, 4, 1)
    assert m.shape == (10, 4, 1)

    offsets = [2, 3]
    obs, act, rew, cont, mask = make_batch_elegant(buff, offsets, 4, Env.pad_state, Env.pad_action)
    obs1, act, rew, cont, mask = make_batch(buff, offsets, 4, Env.pad_state, Env.pad_action)
    assert torch.allclose(obs, obs1)

    trajectories = get_trajectories(buff)
    assert len(trajectories) == 2
    assert len(trajectories[0]) == 2
    assert len(trajectories[1]) == 9
    assert is_trajectory_end(trajectories[0][-1])
    assert trajectories[0][-1].reward == 0.
    assert trajectories[0][-1].cont == 0.
    assert is_trajectory_end(trajectories[1][-1])
    assert is_trajectory_end(trajectories[0][-2]) == False
    assert is_trajectory_end(trajectories[0][-2]) == False

    assert trajectories[1][-1].reward == 1.
    assert trajectories[1][-1].cont == 0.
    assert trajectories[1][-2].cont == 1.

    print(trajectories)