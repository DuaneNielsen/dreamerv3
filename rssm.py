import torch
import torch.nn as nn
from torch.nn.functional import kl_div, softmax
from torch.optim import Adam
from replay import simple_trajectory, ReplayBuffer
from env import Env
from matplotlib import pyplot as plt
from torch.distributions import OneHotCategorical, Normal




if __name__ == '__main__':

    D = Env.state_size
    C = Env.state_classes

    AD = Env.action_size
    AC = Env.action_classes

    ZD = 1  # latent_size
    ZC = 32  # latent classes

    L = 12  #
    N = 10  # batch_size
    hidden_size = 32
    num_layers = 3
    device = "cpu"

    # setup networks
    encoder = nn.Linear(D * C, ZD * ZC).to(device)
    decoder = nn.Linear(ZD * ZC + hidden_size, D * C).to(device)
    gru = nn.GRU(ZD * ZC + AD * AC, hidden_size, num_layers).to(device)
    dynamics_predictor = nn.Linear(hidden_size, ZD * ZC).to(device)
    opt = Adam(gru.parameters(), lr=1e-3)

    # dataset
    buffer = ReplayBuffer()
    for i in range(100):
        buffer += simple_trajectory([Env.right] * 8)
        buffer += simple_trajectory([Env.left])

    # visualize
    plt.ion()
    fig, ax = plt.subplots(1)
    scatter = ax.scatter([], [])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.draw()

    # train
    for epoch in range(2000):
        s, a, r, c, next_s, mask = buffer.sample_batch(L, N)

        # encoder: z ~ q ( z | h, x )
        z_enc_logits = encoder(s.flatten(-2)).unflatten(-1, (ZD, ZC))
        z_enc = sample_one_hot(z_enc_logits)

        # gru: h =  f (h | h, z, a)
        za = torch.cat((z_enc.flatten(-2), a.flatten(-2)), -1)
        h, _ = gru(za)

        # decoder
        s_dec = decoder(torch.cat((z_enc.flatten(-2), h), -1)).unflatten(-1, (D, C))
        s_dec_dist = Normal(s_dec, 1.)

        # predictor
        z_pred = dynamics_predictor(h).unflatten(-1, (ZD, ZC))
        z_pred = sample_one_hot(logits=z_pred)

        loss = - s_dec_dist.log_prob(s).mean()
        # loss += kl_div(input=z.flatten(start_dim=0, end_dim=1).permute(0, 2, 1).log() * mask.reshape(N * L, 1, 1),
        #               target=next_s.flatten(start_dim=0, end_dim=1).permute(0, 2, 1), reduction='batchmean')
        opt.zero_grad()
        loss.backward()
        opt.step()

        # visualize
        with torch.no_grad():
            output, _ = gru(sa.flatten(-2))
            output = dynamics_predictor(output).unflatten(-1, (ZD, ZC))
            transitions = []
            for l in range(L):
                for n in range(N):
                    s_, o = s[l, n, 0].argmax().item(), output[l, n, 0].argmax().item()
                    if o - s_ > 1. and mask[l, n] == 1.:
                        transitions += [f"{s_} -> {o}"]
            print(transitions)
            t0 = s.argmax(-1).flatten()
            t1 = output.argmax(-1).flatten()
            conn_matrix = torch.stack((t0, t1), dim=1)
            conn_matrix = conn_matrix[mask.flatten() == 1.]
            scatter.set_offsets(conn_matrix)
            fig.canvas.draw()
            fig.canvas.flush_events()
