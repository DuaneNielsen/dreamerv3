import torch
from matplotlib import pyplot as plt


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.)


if __name__ == '__main__':

    x = torch.linspace(-12., 12., 100)

    plt.plot(x, x, label='identity')
    plt.plot(x, symlog(x), label='symlog')
    plt.plot(x, torch.log(x), label='log')
    plt.legend()
    plt.show()

    assert symexp(symlog(x)).allclose(x)
