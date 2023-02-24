from critic import td_lambda
import torch


def test_lambda():
    state = torch.tensor([0, 1, 2, 4])
    action = torch.tensor([2, 2, 2, 2])
    values = torch.tensor([0., 0., 0., 0.0])
    reward = torch.tensor([0., 0., 0., 1.0])
    cont = torch.tensor([1., 1., 1., 0.0])

    targets = td_lambda(reward, cont, values)
    print('')
    print(targets)