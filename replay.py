import torch
from env import Env, reward, cont
from collections import namedtuple, deque


Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'next_s', 'mask'])
state_size = 10
EMPTY = Transition(torch.zeros(state_size, 1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(state_size, 1), torch.zeros(1))


def trajectory_len(trj):
    """
    Returns the number of transitions in a trajectory
    :param trj: a trajectory in the form [state, reward, continue, s, r, c .... s, r, c, s]
    :return: the number of transitions
    """
    return int((len(trj) - 1) / 4)


def trajectory_gen(trj, start_i=0):
    """
    generator function to iterate over trajectory

    to use

    >>> gen = trajectory_gen([s, r, c, s])
    >>> next(gen)

    Transition(s, r, c, next_s)

    or more usefully, to generate a added list of transitions

    >>> gen = trajectory_gen([s, r, c, s])
    >>> [next(gen) for _ in range(10)]

    :param trj: a trajectory in the form [state, reward, continue, s, r, c .... s, r, c, s]
    :param start_i: the Transition to start with, defaults to 0 (the first)
    :return: a generator, than when called using next(gen) will return an infinite list of transitions
    once the terminal transition is reached, the empty transition will returned thereafter
    """
    i = start_i * 4
    while True:
        if i + 4 < len(trj):
            yield Transition(trj[i], trj[i+1], trj[i+2], trj[i+3], trj[i+4], torch.ones(1))
        else:
            yield EMPTY
        i += 4


def trajectory_batch(trj_list, length, start=None):
    """
    pads a list of trajectories ready to load into GRU
    :param trj_list: [[Transition, Transition ...], [Transition, Transition...]]  can be of varying length
    :param length: the length of trajectories in the batch we want to return, trajectories will be padded to length with zeros
    :param start: a list of offsets for each trajectory indicating which transition tos start from
    :return: states [L, N, C, D], actions [L, N, AD] rewards [L, N, 1], continues [L, N, 1], next_states [L, N, C, D]
        L -> length of trajectory,
        N -> number of trajectories in batch,
        AD -> dimension of each action
        C -> number of classes in categorical variables
        D -> number of categorical variables
    """

    padded = []
    start = [0]*len(trj_list) if start is None else start
    for i, trj in enumerate(trj_list):
        gen = trajectory_gen(trj, start[i])
        padded.append([next(gen) for _ in range(length)])

    s, a, r, c, next_s, mask = [], [], [], [], [], []
    for trj in padded:
        s.append(torch.stack([t.s for t in trj]))
        a.append(torch.stack([t.a for t in trj]))
        r.append(torch.stack([t.r for t in trj]))
        c.append(torch.stack([t.c for t in trj]))
        next_s.append(torch.stack([t.next_s for t in trj]))
        mask.append(torch.stack([t.mask for t in trj]))
    s = torch.stack(s, dim=1)
    a = torch.stack(a, dim=1)
    r = torch.stack(r, dim=1)
    c = torch.stack(c, dim=1)
    next_s = torch.stack(next_s, dim=1)
    mask = torch.stack(mask, dim=1)
    return s, a, r, c, next_s, mask


class ReplayBuffer:
    def __init__(self, max_trajectories=None):
        self.max_trajectories = max_trajectories
        self.buffer = deque(maxlen=max_trajectories)
        self._len = 0
        self.trj_idx = deque()
        self.offset_idx = deque()
        self.dropped = 0
        self.next_trajectory_idx = 0

    def sample_batch(self, length, batch_size):
        """ samples with replacement as this is much faster and simpler in pytorch,
        and replay buffer will be massive anyway, so chance of drawing same transition is basically zero """
        batch = torch.randint(low=0, high=len(self), size=(batch_size, ))
        trj_list = [self.buffer[self._to_trj(sample_idx)] for sample_idx in batch]
        offset = [self.offset_idx[sample_idx] for sample_idx in batch]
        return trajectory_batch(trj_list, length, offset)

    # def __iadd__(self, trajectory):
    #     self.buffer.append(trajectory)
    #     self._len = 0
    #     for t, trj in enumerate(self.buffer):
    #         for o in range(trajectory_len(trj)):
    #             self.trj_idx += [t]
    #             self.offset_idx += [o]
    #             self._len += 1
    #     return self

    def _to_trj(self, transition):
        return self.trj_idx[transition] - self.dropped

    def __iadd__(self, trj):
        if len(self.buffer) == self.max_trajectories:
            pop_len = trajectory_len(self.buffer[0])
            self._len -= pop_len
            self.dropped += 1
            for _ in range(pop_len):
                self.trj_idx.popleft()
                self.offset_idx.popleft()

        tlen = trajectory_len(trj)
        self._len += tlen
        self.buffer.append(trj)
        self.trj_idx += [self.next_trajectory_idx] * tlen
        self.next_trajectory_idx += 1
        self.offset_idx += list(range(tlen))
        return self

    def __len__(self):
        return self._len


def simple_trajectory(actions=None):
    actions = [Env.right] * 8 if actions is None else actions
    env = Env(state_size, reward, cont)
    trj = [env.reset()]
    for a in actions:
        s, r, d, _ = env.step(a)
        trj += [a, r, d, s]
    return trj


if __name__ == '__main__':

    with torch.no_grad():

        trj = simple_trajectory()
        assert(len(trj) == 1 + 8 * 4)

        trj_len = trajectory_len(trj)
        assert trj_len == 8

        replay_buffer = ReplayBuffer()
        replay_buffer += trj
        replay_buffer += trj
        assert len(replay_buffer) == 16

        s, a, r, c, next_s, mask = trajectory_batch([trj, trj], 10)
        assert s.shape == (10, 2, 10, 1)
        assert a.shape == (10, 2, 1)
        assert r.shape == (10, 2, 1)
        assert c.shape == (10, 2, 1)
        assert next_s.shape == (10, 2, 10, 1)

        replay_buffer = ReplayBuffer(max_trajectories=10)
        bins = torch.zeros(10, dtype=torch.int)
        for i in range(1000):
            replay_buffer += trj
            s, a, r, c, next_s, mask = replay_buffer.sample_batch(5, 10)
            bins[next_s[0].argmax(-2).flatten()] += 1
            print(bins)