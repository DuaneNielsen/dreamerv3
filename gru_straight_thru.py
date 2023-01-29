import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax
from torch.optim import Adam
from replay import simple_trajectory, ReplayBuffer
from env import Env
from matplotlib import pyplot as plt
from torch.distributions import OneHotCategorical


if __name__ == '__main__':

    C = Env.state_classes
    D = Env.state_size
    AC = Env.action_classes
    AD = Env.action_size

    L = 12  #
    N = 10  # batch_size
    hidden_size = 32
    num_layers = 3
    device = "cpu"

    # setup GRU
    gru = nn.GRU(C * D + AC * AD, hidden_size, num_layers).to(device)
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

    def sample_one_hot(logits, epsilon=0.01):
        " sample from a categorical using the straight-thru method"
        uniform = torch.ones_like(logits)
        probs = torch.softmax(logits, -1)
        probs = (1 - epsilon) * probs + epsilon * uniform
        dist = OneHotCategorical(probs=probs)
        return dist.sample() + dist.probs - dist.probs.detach()


    # train
    for epoch in range(2000):
        s, a, r, c, next_s, mask = buffer.sample_batch(L, N)

        sa = torch.cat((s, a), -1)
        output, _ = gru(sa.flatten(-2))
        logits = hidden_to_part(output).unflatten(-1, (D, C))
        sample = sample_one_hot(logits)
        loss = cross_entropy(input=sample.flatten(start_dim=0, end_dim=1).permute(0, 2, 1) * mask.reshape(N * L, 1, 1),
                             target=next_s.flatten(start_dim=0, end_dim=1).permute(0, 2, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        # test
        with torch.no_grad():
            output, _ = gru(sa.flatten(-2))
            output = hidden_to_part(output).unflatten(-1, (D, C))
            transitions = []
            for l in range(L):
                for n in range(N):
                    s_, o = s[l, n, 0].argmax().item(), output[l, n, 0].argmax().item()
                    if abs(o - s_) > 1. and mask[l, n] == 1.:
                        transitions += [f"{s_} -> {o}"]
            print(transitions)
            t0 = s.argmax(-1).flatten()
            t1 = output.argmax(-1).flatten()
            conn_matrix = torch.stack((t0, t1), dim=1)
            conn_matrix = conn_matrix[mask.flatten() == 1.]
            scatter.set_offsets(conn_matrix)
            fig.canvas.draw()
            fig.canvas.flush_events()
