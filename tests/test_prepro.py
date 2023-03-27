import config
import torch


def test_prepro():
    prepro = config.PrePostProcessing()
    x = torch.rand(8, 10, 3, 64, 64)
    y = prepro.prepro(x)
    x_ = prepro.postpro(y)
    assert torch.allclose(x, x_)