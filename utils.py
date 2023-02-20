import pickle
from pathlib import Path
import torch
from time import time


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


def load(rundir, rssm, rssm_optim, critic, critic_optim, actor, actor_optim):
    checkpoint = torch.load(rundir + '/model_opt.pt')
    rssm.load_state_dict(checkpoint['rssm_state_dict'])
    rssm_optim.load_state_dict(checkpoint['rssm_optim_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
    args = checkpoint['args']
    steps = checkpoint['steps']
    return rssm, rssm_optim, critic, critic_optim, actor, actor_optim, steps, args


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


class StopWatch:
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.total_time = 0
        self.running = False

    def reset(self):
        self.total_time = 0
        self.running = False

    def go(self):
        self.start_time = time()
        self.running = True

    def pause(self):
        if not self.running:
            raise Exception('Call go() before calling pause()')
        self.total_time += time() - self.start_time
        self.running = False


if __name__ == '__main__':

    from rssm import make_small
    from torch.optim import Adam
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--hello', default='hello')
    args = parser.parse_args()

    rssm = make_small(3)
    opt = Adam(rssm.parameters())

    save(run.rundir, rssm, opt, args, 100, 1.32)
    rssm, opt, step, args, loss = load(run.rundir, rssm, opt)
    print(args)
