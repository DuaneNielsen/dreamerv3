import numpy as np


class Env:
    left = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    right = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    state_classes = 10
    state_size = 1
    action_classes = 10
    action_size = 1
    one_hot_targets = np.eye(10, dtype=np.float32)

    pad_state = one_hot_targets[np.array([0])]
    pad_action = left

    def __init__(self):
        self.state = np.array([1])

    def reset(self):
        self.state = np.array([1])
        return Env.one_hot_targets[self.state]

    def step(self, action):
        self.state += action.argmax() - 1
        self.state = self.state.clip(0, 9)
        reward = (self.state == 9) * 1.
        terminated = self.state == 9 or self.state == 0

        return Env.one_hot_targets[self.state], reward[0], terminated[0], False, {"state": self.state}


if __name__ == "__main__":
    env = Env()

    # test reset
    s = env.reset()
    assert s.argmax() == 1

    # test left
    state = env.reset()
    state, reward, terminated, truncated, info = env.step(Env.left)
    assert state.argmax() == 0
    assert state.dtype == np.float32
    assert reward == 0.
    assert terminated
    assert not truncated

    # test right
    state = env.reset()
    state, reward, terminated, truncated, info = env.step(Env.right)
    assert state.argmax() == 2
    assert state.dtype == np.float32
    assert reward == 0.
    assert not terminated
    assert not truncated

    # test go right to end
    s = env.reset()
    for t in range(7):
        state, reward, terminated, truncated, info = env.step(Env.right)
        assert state.argmax() == t + 2
        assert reward == 0.
        assert not terminated
        assert not truncated

    state, reward, terminated, truncated, info = env.step(Env.right)
    assert state.argmax() == 9
    assert state.dtype == np.float32
    assert reward == 1.
    assert terminated
    assert not truncated
