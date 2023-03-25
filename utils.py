from pathlib import Path
import torch
from time import perf_counter


def next_run():
    run_id_file = Path(f'runs/.runs')
    if not run_id_file.exists():
        run_id = 0
        with run_id_file.open('w') as f:
            f.write(str(run_id))
    with run_id_file.open('r') as f:
        run_id= int(f.readline().rstrip())
        run_id += 1
    with run_id_file.open('w') as f:
        f.write(str(run_id))
    return f'run_{run_id}'


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


