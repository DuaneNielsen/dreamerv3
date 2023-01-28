import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from replay import simple_trajectory, ReplayBuffer
from env import Env
from matplotlib import pyplot as plt

if __name__ == '__main__':

    L = 12
    N = 10
    C = 10
    D = 1
    hidden_size = 20
    num_layers = 3
    device = "cpu"

    # setup GRU
    gru = nn.GRU(C * D, hidden_size, num_layers).to(device)
    hidden_to_part = nn.Linear(hidden_size, C * D)
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
    for epoch in range(1000):
        s, a, r, c, next_s, mask = buffer.sample_batch(L, N)

        output, _ = gru(s.flatten(-2))
        output = hidden_to_part(output).unflatten(-1, (C, D))
        output = output.softmax(2)
        loss = cross_entropy(input=output.flatten(start_dim=0, end_dim=1) * mask.reshape(N * L, 1, 1),
                             target=next_s.flatten(start_dim=0, end_dim=1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        # test
        with torch.no_grad():
            output, _ = gru(s.flatten(-2))
            output = hidden_to_part(output).unflatten(-1, (C, D))
            mse = ((output[:, :, 0].argmax() - s[:, :, 0].argmax(-1) - 1.) ** 2).std()
            transitions = []
            for l in range(L):
                for n in range(N):
                    s_, o = s[l, n].argmax(0).item(), output[l, n].argmax(0).item()
                    if o - s_ > 1. and mask[l, n] == 1.:
                        transitions += [f"{s_} -> {o}"]
            print(transitions)
            t0 = s.argmax(2).flatten()
            t1 = output.argmax(2).flatten()
            conn_matrix = torch.stack((t0, t1), dim=1)
            conn_matrix = conn_matrix[mask.flatten() == 1.]
            scatter.set_offsets(conn_matrix)
            fig.canvas.draw()
            fig.canvas.flush_events()
