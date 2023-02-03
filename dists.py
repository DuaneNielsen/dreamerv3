import torch
from torch.nn.functional import softmax
from torch.distributions import OneHotCategorical, kl_divergence


def sample_one_hot(logits, epsilon=0.01):
    """  sample from a categorical using the straight-thru method  """
    uniform = torch.ones_like(logits)
    probs = softmax(logits, -1)
    probs = (1 - epsilon) * probs + epsilon * uniform
    dist = OneHotCategorical(probs=probs)
    return dist.sample() + probs - probs.detach()


class OneHotCategoricalStraightThru(OneHotCategorical):
    def __init__(self, probs=None, logits=None, epsilon=0.01):
        probs = probs if probs is not None else softmax(logits, -1)
        uniform = torch.ones_like(probs)
        probs = (1 - epsilon) * probs + epsilon * uniform
        super().__init__(probs=probs)

    def sample(self, epsilon=0.01):
        return super().sample() + self.probs - self.probs.detach()


def categorical_kl_divergence_clamped(logits_left, logits_right, clamp=1.):
    return kl_divergence(
        OneHotCategorical(logits=logits_left),
        OneHotCategorical(logits=logits_right)
    ).clamp(max=1.)
