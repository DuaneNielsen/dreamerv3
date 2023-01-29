import torch
import torch.nn.functional as nf


def reward(state):
    return (state == 9) * 1.


def cont(state):
    return (0. < state < 9.) * 1.


class Env:
    left = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    right = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    state_classes = 10
    state_size = 1
    action_classes = 10
    action_size = 1

    def __init__(self, reward_f, cont_f):
        self.state = torch.tensor([1], requires_grad=False)
        self.reward_f = reward_f
        self.done_f = cont_f

    def reset(self):
        self.state = torch.tensor([1], requires_grad=False)
        return nf.one_hot(self.state, Env.state_classes).float()

    def step(self, action):
        self.state += action.argmax() - 1
        self.state = self.state.clamp(0, 9)
        return nf.one_hot(self.state, Env.state_classes).float(), self.reward_f(self.state), self.done_f(self.state), {"state": self.state}


if __name__ == "__main__":
    env = Env(reward, cont)

    # test reset
    s = env.reset()
    assert s.argmax() == 1

    # test left
    s = env.reset()
    s, r, c, i = env.step(Env.left)
    assert s.argmax() == 0
    assert s.dtype == torch.float32
    assert r == 0.
    assert c == 0.

    # test right
    s = env.reset()
    s, r, c, i = env.step(Env.right)
    assert s.argmax() == 2
    assert r == 0.
    assert c == 1.

    # test go right to end
    s = env.reset()
    for t in range(7):
        s, r, c, i = env.step(Env.right)
        assert s.argmax() == t + 2
        assert r == 0.
        assert c == 1.
    s, r, c, i = env.step(Env.right)
    assert s.argmax() == 9
    assert r == 1.
    assert c == 0.

