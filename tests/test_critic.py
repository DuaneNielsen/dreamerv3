from critic import td_lambda
import torch


def test_lambda():
    state = torch.tensor([0, 1, 2, 4])
    action = torch.tensor([2, 2, 2, 2])
    values = torch.tensor([0., 0., 0., 0.0])
    reward = torch.tensor([0., 0., 0., 1.0])
    cont = torch.tensor([1., 1., 1., 0.0])

    discount = 0.9
    lam = 0.8
    targets = td_lambda(reward, cont, values, discount=discount, lam=lam)
    target_1 = lam * discount ** 1
    target_2 = (target_1 * lam) * discount
    expected = torch.tensor([target_2, target_1, 1.0, 0.0])
    assert torch.allclose(targets, expected)