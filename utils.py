import pickle
from pathlib import Path
import torch


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


def save(rundir, rssm, optimizer, args, steps, loss):
    torch.save({
        'rssm_state_dict': rssm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'steps': steps,
        'loss': loss
    }, rundir + '/model_opt.pt')


def load(rundir, rssm, optimizer):
    checkpoint = torch.load(rundir + '/model_opt.pt')
    rssm.load_state_dict(checkpoint['rssm_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    args = checkpoint['args']
    steps = checkpoint['steps']
    loss = checkpoint['loss']
    return rssm, optimizer, steps, args, loss


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
