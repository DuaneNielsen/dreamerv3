import pickle
from pathlib import Path
import torch
from time import time, perf_counter


class Run:
    def __init__(self):
        self.run_id = -1
        self.rundir = None

    def next_run(self):
        self.run_id += 1
        dir = Path(f'runs/run_{self.run_id}')
        dir.mkdir(exist_ok=True)
        self.rundir = str(dir)
        return self


Path('runs').mkdir(exist_ok=True)

if Path('runs/.runs').exists():
    with Path('runs/.runs').open('rb') as f:
        run = pickle.load(f)
else:
    run = Run()

run.next_run()
with Path('runs/.runs').open('wb') as f:
    pickle.dump(run, f)
print(f"starting run {run.run_id}")


def save(rundir, rssm, rssm_optim, critic, critic_optim, actor, actor_optim, args, steps):
    torch.save({
        'rssm_state_dict': rssm.state_dict(),
        'rssm_optim_state_dict': rssm_optim.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'critic_optim_state_dict': critic_optim.state_dict(),
        'actor_state_dict': actor.state_dict(),
        'actor_optim_state_dict': actor_optim.state_dict(),
        'args': args,
        'steps': steps
    }, rundir + '/model_opt.pt')


def load(rundir, rssm=None, rssm_optim=None, critic=None, critic_optim=None, actor=None, actor_optim=None):
    checkpoint = torch.load(rundir + '/model_opt.pt')
    if rssm is not None:
        rssm.load_state_dict(checkpoint['rssm_state_dict'])
    if rssm_optim is not None:
        rssm_optim.load_state_dict(checkpoint['rssm_optim_state_dict'])
    if critic is not None:
        critic.load_state_dict(checkpoint['critic_state_dict'])
    if critic_optim is not None:
        critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
    if actor is not None:
        actor.load_state_dict(checkpoint['actor_state_dict'])
    if actor_optim is not None:
        actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
    args = checkpoint['args']
    steps = checkpoint['steps']
    return steps, args


def bin_values(values, min, max, num_bins):
    values = values.clamp(min=min, max=max) - min
    values = values / (max - min)
    values = values * (num_bins - 1)
    return values.round().long()


def bin_labels(min, max, num_bins):
    bins = torch.linspace(min, max, num_bins+1)
    labels = []
    for i in range(1, num_bins+1):
        labels += [f'{bins[i-1]:.2f}-{bins[i]:.2f}']
    return labels


class ProgramTimer:
    def __init__(self):
        self.elapsed = 0

    def __enter__(self, *args):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop = perf_counter()
        self.elapsed = self.stop - self.start
        return False


def register_gradient_clamp(nn_module, gradient_min_max):
    for p in nn_module.parameters():
        p.register_hook(
            lambda grad: torch.clamp(grad, -gradient_min_max, gradient_min_max))


if __name__ == '__main__':

    def fibonacci(n):
        f1 = 1
        f2 = 1
        for i in range(n - 1):
            f1, f2 = f2, f1 + f2

        return f1

    with ProgramTimer() as timer:
        for _ in range(10000):
            fibonacci(1000)

    print(timer.elapsed)
    print(timer)

