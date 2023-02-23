import torch
from torch.nn.functional import softmax
from torch.distributions import OneHotCategorical, kl_divergence
from torch.nn.functional import one_hot
from symlog import symlog, symexp


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
        super().__init__(probs=probs, validate_args=False)

    def sample(self, sample_shape=None):
        if sample_shape:
            raise NotImplementedError('sample_shape is not implemented')
        return super().sample() + self.probs - self.probs.detach()


def categorical_kl_divergence_clamped(logits_left, logits_right, clamp=1.):
    return kl_divergence(
        OneHotCategorical(logits=logits_left),
        OneHotCategorical(logits=logits_right)
    ).clamp(max=clamp)


class EncodeTwoHot:
    def __init__(self, low, high, num_bins, device='cpu'):
        self.bins = torch.linspace(low, high, num_bins).to(device)
        self.num_bins = num_bins

    def __call__(self, value):
        below = (self.bins <= value[..., None]).to(dtype=torch.long).sum(-1) - 1
        above = self.num_bins - (self.bins > value[..., None]).to(dtype=torch.long).sum(-1)
        below = below.clamp(0, self.num_bins - 1)
        above = above.clamp(0, self.num_bins - 1)
        equal = below == above
        dist_to_below = torch.where(equal, 1., torch.abs(self.bins[below] - value))
        dist_to_above = torch.where(equal, 1., torch.abs(self.bins[above] - value))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        return one_hot(below, self.num_bins) * weight_below[..., None] + one_hot(above, self.num_bins) * weight_above[..., None]


class TwoHot:
    """
    Interprets logits as a two hot distribution
    param: logits: [..., bins] logits
    param: low: the lowest value represented
    param: high: the highest value represented
    """
    def __init__(self, logits, low=-20, high=20.):
        self.logits = logits
        self.encode_two_hot = EncodeTwoHot(low, high, logits.shape[-1], device=logits.device)

    @property
    def mean(self):
        return (torch.softmax(self.logits, -1) * self.encode_two_hot.bins).sum(-1)

    def log_prob(self, value):
        """
        param: value: [...] values
        """
        target = self.encode_two_hot(value)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_pred).sum(-1)


class TwoHotSymlog(TwoHot):
    """
    Interprets logits as a two hot distribution in symlog space
    param: logits
    """
    def __init__(self, logits):
        super().__init__(logits)

    @property
    def mean(self):
        return symexp(super().mean)

    def log_prob(self, value):
        value = symlog(value)
        return super().log_prob(value)


if __name__ == '__main__':

    for i in range(10):
        OneHotCategoricalStraightThru(logits=torch.randn(10, 1, 4)).log_prob(
            OneHotCategoricalStraightThru(logits=torch.randn(10, 1, 4)).sample())

    x = torch.linspace(0, 10, 10).unsqueeze(0)
    encode_two_hot = EncodeTwoHot(0, 10, 5)
    print(encode_two_hot(x))

    dist_values = torch.linspace(-20, 20, 200)
    encode_two_hot = EncodeTwoHot(-20, 20, 256)
    logits = encode_two_hot(dist_values)
    value = torch.zeros(200)
    dist = TwoHot(logits, low=-20, high=20)
    logprobs = dist.log_prob(value)
    print(dist.mean)

    from torch.nn import Linear
    from torch.optim import SGD

    net = Linear(1, 256)
    optim = SGD(net.parameters(), lr=1e-1)

    for _ in range(4000):
        logits = net(dist_values.unsqueeze(1))
        twohot_dist = TwoHot(logits, low=-20, high=20)
        loss = - twohot_dist.log_prob(dist_values).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(twohot_dist.mean)
