import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import one_hot, cross_entropy, nll_loss
from matplotlib import pyplot as plt

label = torch.tensor([0, 1, 2, 3, 4, 4, 4, 4, 4, 4])

inp = [one_hot(a, 5) for a in label]
batch = torch.stack(inp).float()

# net = nn.Linear(5, 5)
net = nn.Sequential(nn.Linear(5, 5), nn.LayerNorm(5), nn.SiLU(), nn.Linear(5, 5))
opt = Adam(net.parameters(), lr=1e-3)

for step in range(5000):
    pred = net(batch)
    loss = cross_entropy(pred, batch)
    # loss = nll_loss(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(pred.argmax(-1))
