import torch
from torch.distributions import OneHotCategorical

if __name__ == '__main__':

    z = OneHotCategorical(logits=torch.rand(1, 10))