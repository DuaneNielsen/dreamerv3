import torch
import torch.nn as nn
from dists import TwoHot, TwoHotSymlog
from torch.optim import Adam

net = nn.Linear(10, 256)
optim = Adam(net.parameters(), lr=1e-2)

input = torch.eye(10, requires_grad=True)
gt = torch.tensor([0.]*8 + [1.]*2)
gt = torch.tensor([0.]*9 + [1.]*1)

for step in range(10000):
    pred = net(input)
    pred_dist = TwoHotSymlog(logits=pred)
    loss = - pred_dist.log_prob(gt)
    loss = loss.mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f'{loss.item()} {pred_dist.mean[0]} {pred_dist.mean[9]}')
