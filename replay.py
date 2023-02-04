from env import Env
import torch
from collections import deque, namedtuple

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


def sample_batch(buffer, length, batch_size, pad_state, pad_action):
    """
    Sample a batch from the replay buffer
    :param buffer: replay buffer to sample from
    :param length: number of timesteps to sample T
    :param batch_size: batch size N
    :return: x -> [T, N, ... ], a -> [T, N, ...], r -> [T, N, 1], c -> [T, N, 1], m -> [T, N, 1]
    """
    offsets = torch.randint(0, len(buffer), (batch_size,))
    pad = [False] * batch_size

    x, a, r, c, mask = [], [], [], [], []

    for t in range(length):
        x_i, a_i, r_i, c_i, m_i = [], [], [], [], []
        for n, o in enumerate(offsets):
            o = o.item()
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


def rollout_open_loop_policy(env, actions):
    x, r, c = env.reset(), 0, 1.0
    for a in actions:
        yield Step(x, a, r, c)
        x, r, done, _ = env.step(a)
        c = 0.0 if done else 1.0
    yield Step(x, None, r, c)


if __name__ == '__main__':

    buff = deque()
    env = Env()

    go_right = [Env.right] * 8
    go_left = [Env.left]

    for step in rollout_open_loop_policy(env, go_left):
        buff.append(step)
        sample_batch(buff, 10, 4)
    for step in rollout_open_loop_policy(env, go_right):
        buff.append(step)
        sample_batch(buff, 10, 4)

    print(buff)
    print([is_trajectory_end(s) for s in buff])

    x, a, r, c, m = sample_batch(buff, 10, 4)

    assert x.shape == (10, 4, 1, 10)
    assert a.shape == (10, 4, 1, 10)
    assert c.shape == (10, 4, 1)
    assert r.shape == (10, 4, 1)
    assert m.shape == (10, 4, 1)

