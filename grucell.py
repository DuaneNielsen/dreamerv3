import torch
import torch.nn as nn
from torch.optim import Adam
from replay import simple_trajectory, ReplayBuffer
from env import Env
from matplotlib import pyplot as plt
from torch.distributions import OneHotCategorical

if __name__ == '__main__':

    XD = Env.state_size
    XC = Env.state_classes
    AD = Env.action_size
    AC = Env.action_classes
    ZD = 1  # latent_size
    ZC = 32  # latent classes

    T = 12  # Time Horizon
    N = 10  # batch_size
    H = 32  # Gru recurrent units - a.k.a. "hidden size"
    num_layers = 3
    device = "cpu"


    class RSSM(nn.Module):

        def __init__(self):
            super().__init__()
            # encoder: z ~ q ( z | h, x )
            self.encoder = nn.Linear(H + XD * XC, ZD * ZC)

            # sequence model: h = f ( h, z, a )
            self.gru = nn.GRUCell(ZD * ZC + AD * AC, H)

            # decoder: x ~ p ( x | h, z )
            self.decoder = nn.Linear(H + ZD * ZC, XD * XC)

        def forward(self, x, h0):
            """

            :param x: [T, N, XD, XC] state tensor
            :param h0: [N, H] start state tensor
            :return: x_decoded: prediction of next state
            """
            # recurrent encoding
            h_list = [h0]
            hs_flat = torch.cat((h_list[0], x[0].flatten(-2)), dim=1)
            z_list_flat = [self.encoder(hs_flat)]

            for i in range(0, T - 1):
                za_flat = torch.cat([z_list_flat[i], a[i].flatten(-2)], dim=-1)
                h_list += [self.gru(za_flat, h_list[i])]
                hs_flat = torch.cat([h_list[i], x[i].flatten(-2)], dim=1)
                z_list_flat += [self.encoder(hs_flat)]
            h = torch.stack(h_list)
            z = torch.stack(z_list_flat).unflatten(-1, (ZD, ZC))

            # decode
            hz = torch.cat([h, z.flatten(-2)], dim=2)
            x_decoded_flat = self.decoder(hz)
            x_decoded = x_decoded_flat.unflatten(-1, (XD, XC))
            return OneHotCategorical(logits=x_decoded)


    rssm = RSSM()

    # optimizer
    opt = Adam(rssm.parameters(), lr=1e-3)

    # dataset
    buffer = ReplayBuffer()
    for i in range(100):
        buffer += simple_trajectory([Env.right] * 8)
        buffer += simple_trajectory([Env.left])

    # visualize
    plt.ion()
    fig, ax = plt.subplots(1)
    scatter = ax.scatter([0, 10], [0, 10])

    for epoch in range(2000):
        x, a, r, c, next_x, mask = buffer.sample_batch(T, N)
        h0 = torch.zeros(N, H)
        x_dist = rssm(x, h0)
        loss = - x_dist.log_prob(x) * mask.unsqueeze(-1)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()

        # visualize
        matrix = torch.stack([x.argmax(-1).flatten(), x_dist.mean.argmax(-1).flatten()], dim=1)
        matrix = matrix[mask.flatten()]
        scatter.set_offsets(matrix.detach())
        fig.canvas.draw()
        fig.canvas.flush_events()
