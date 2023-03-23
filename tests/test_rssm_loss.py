from rssm import RSSMLoss
from dists import categorical_kl_divergence, ImageMSEDist, TwoHotSymlog
import torch
from config import prepro, inv_prepro
from torch.distributions import Bernoulli

def test_dyn_loss():
    g_cpu = torch.Generator()
    g_cpu = torch.manual_seed(0)
    z_post_logits = torch.randn(1, 32, 32)
    z_prior_logits = torch.randn(1, 32, 32)
    loss = categorical_kl_divergence(z_post_logits, z_prior_logits)
    assert abs(loss.mean().item() - 32.1388) < 0.001

    z_prior_logits = torch.ones(1, 32, 32)
    loss = categorical_kl_divergence(z_post_logits, z_prior_logits)
    assert abs(loss.mean().item() - 17.3173) < 0.001


def test_the_rssm_loss():
    g_cpu = torch.Generator()
    g_cpu = torch.manual_seed(0)
    z_post_logits = torch.randn(1, 1, 32, 32)
    z_prior_logits = torch.randn(1, 1, 32, 32)
    obs = torch.randn(1, 1, 3, 64, 64)
    rewards = torch.randn(1, 1, 1)
    cont = torch.tensor([1.])
    identity = lambda x: x
    obs_dist = ImageMSEDist(mode=torch.randn(1, 1, 3, 64, 64), dims=3, prepro=identity, inv_prepro=identity)
    reward_dist = TwoHotSymlog(logits=torch.randn(1, 1, 255))
    cont_dist = Bernoulli(logits=torch.randn(1, 1, 1))

    criterion = RSSMLoss()
    loss = criterion(obs, rewards, cont, obs_dist, reward_dist, cont_dist, z_prior_logits, z_post_logits)
    print(criterion.loss_dict())