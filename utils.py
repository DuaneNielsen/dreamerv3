import torch
from torch.nn.functional import softmax, log_softmax
from torch.distributions import OneHotCategorical


def sample_one_hot(logits, epsilon=0.01):
    """  sample from a categorical using the straight-thru method  """
    uniform = torch.ones_like(logits)
    probs = softmax(logits, -1)
    probs = (1 - epsilon) * probs + epsilon * uniform
    dist = OneHotCategorical(probs=probs)
    return dist.sample() + probs - probs.detach()


def sample_one_hot_log(logits, epsilon=0.01):
    """  sample from a categorical using the straight-thru method  """
    uniform = torch.ones_like(logits)
    logits = log_softmax(logits, -1)
    # probs = (1 - epsilon) * probs + epsilon * uniform
    # logits = probs.log()
    dist = OneHotCategorical(logits=logits)
    return dist.sample() + logits - logits.detach(), dist