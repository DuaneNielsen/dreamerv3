import torch
from env import Env, reward, cont
from collections import deque

offsets = {'s': 0, 'a': 1, 'r': 2, 'c': 3, 'next_s': 4, 'mask': 5}
pads = {'s': torch.zeros(Env.state_classes, Env.state_size),
        'a': torch.zeros(Env.action_classes, Env.action_size),
        'r': torch.zeros(1),
        'c': torch.zeros(1),
        'next_s': torch.zeros(Env.state_classes, Env.state_size)}


def trajectory_len(trj):
    """
    Returns the number of transitions in a trajectory
    :param trj: a trajectory in the form [state, reward, continue, s, r, c .... s, r, c, s]
    :return: the number of transitions
    """
    return int((len(trj) - 1) / 4)


def trajectory_batch_mask_gen(trj_list, offset_list):
    offset_list = [s * 4 for s in offset_list]
    while True:
        batch = []
        for trj, i in zip(trj_list, offset_list):
            if i + 4 < len(trj):
                batch += torch.tensor([True])
            else:
                batch += torch.tensor([False])

        yield torch.stack(batch)
        offset_list = [s + 4 for s in offset_list]


def trajectory_batch_gen(key, trj_list, offset_list, padding):
    offset_list = [s * 4 for s in offset_list]
    while True:
        batch = []
        for trj, i in zip(trj_list, offset_list):
            if i + 4 < len(trj):
                batch += [trj[i + offsets[key]]]
            else:
                batch += [padding]

        yield torch.stack(batch)
        offset_list = [s + 4 for s in offset_list]


def trajectory_batch(key, length, trj_list, offset_list, padding=None):
    """
    Given a list of trajectories and transition offsets returns a tensor for the required field
    that is suitable for loading into a GRU
    :param key: the field to generate the tensor for, valid keys are
        s -> state
        a -> action
        r -> reward
        c -> continue
        next_s -> next state
        mask -> a bool Tensor indicating padding
    :param length: the length to take from each tensor
    :param trj_list: a list of trajectories of varying length
    :param offset_list: a list of offsets into the trajectories
    :return: (L, N, ...) Tensor where L is the number of
    """

    offset_list = [0] * len(trj_list) if offset_list is None else offset_list

    if key != 'mask':
        padding = pads[key] if padding is None else padding
        batch_gen = trajectory_batch_gen(key, trj_list, offset_list, padding)
    else:
        batch_gen = trajectory_batch_mask_gen(trj_list, offset_list)
    return torch.stack([next(batch_gen) for _ in range(length)])


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
        batch = torch.randint(low=0, high=len(self), size=(batch_size,))
        trj_list = [self.buffer[self._to_trj(sample_idx)] for sample_idx in batch]
        offset_list = [self.offset_idx[sample_idx] for sample_idx in batch]
        return tuple([trajectory_batch(key, length, trj_list, offset_list) for key in offsets])

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


def simple_trajectory(actions):
    env = Env(reward, cont)
    trj = [env.reset()]
    for a in actions:
        s, r, d, _ = env.step(a)
        trj += [a, r, d, s]
    return trj


if __name__ == '__main__':

    with torch.no_grad():

        trj = simple_trajectory([Env.right] * 8)
        assert (len(trj) == 1 + 8 * 4)

        trj_len = trajectory_len(trj)
        assert trj_len == 8

        replay_buffer = ReplayBuffer()
        replay_buffer += trj
        replay_buffer += trj
        assert len(replay_buffer) == 16

        s = trajectory_batch('s', 10, [trj, trj], [0, 0])
        assert s.shape == (10, 2, 10, 1)

        a = trajectory_batch('a', 10, [trj, trj], [0, 0])
        assert a.shape == (10, 2, 10, 1)

        r = trajectory_batch('r', 10, [trj, trj], [0, 0])
        assert r.shape == (10, 2, 1)

        c = trajectory_batch('c', 10, [trj, trj], [0, 0])
        assert c.shape == (10, 2, 1)

        next_s = trajectory_batch('next_s', 10, [trj, trj], [0, 0])
        assert next_s.shape == (10, 2, 10, 1)

        replay_buffer = ReplayBuffer(max_trajectories=10)
        bins = torch.zeros(10, dtype=torch.int)
        for i in range(1000):
            replay_buffer += trj
            s, a, r, c, next_s, mask = replay_buffer.sample_batch(length=5, batch_size=10)
            bins[next_s[0].argmax(-2).flatten()] += 1
            print(bins)
