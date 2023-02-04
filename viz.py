import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch.nn.functional import conv1d

"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
"""


class PlotJointAndMarginals:
    def __init__(self, ax, title=None, ylabel=None, xlabel=None):
        self.ax = ax
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel

    def scatter_hist(self, x, y):
        self.ax.cla()
        self.ax.set(aspect=1)
        self.ax.set_title(self.title)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlabel(self.xlabel)

        # Create marginal axes, which have 25% of the size of the main axes.  Note that
        # the inset axes are positioned *outside* (on the right and the top) of the
        # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
        # less than 0 would likewise specify positions on the left and the bottom of
        # the main axes.
        self.ax_histx = self.ax.inset_axes([0, 1.05, 1, 0.25], sharex=self.ax)
        self.ax_histy = self.ax.inset_axes([1.05, 0, 0.25, 1], sharey=self.ax)

        # no labels
        self.ax_histx.tick_params(axis="x", labelbottom=False)
        self.ax_histy.tick_params(axis="y", labelleft=False)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(x), np.max(y))
        xymin = min(np.min(x), np.min(y))

        bins = np.arange(xymin, xymax + binwidth, binwidth)
        self.ax_histx.hist(x, bins=bins)
        self.ax_histy.hist(y, bins=bins, orientation='horizontal')

        cells_x = np.digitize(x, bins=bins)
        cells_y = np.digitize(y, bins=bins)
        cells_joint = np.zeros((len(bins), len(bins)), dtype=int)

        # this could be faster if vectorized
        for x, y in zip(cells_x, cells_y):
            assert y-1 < cells_joint.shape[1], f'{x} {y} {cells_joint.shape}'
            cells_joint[y-1, x-1] += 1

        # the scatter plot:
        self.ax.imshow(cells_joint, origin='lower', interpolation='none', extent=[xymin, xymax+binwidth, xymin, xymax + binwidth])


class PlotLosses:
    def __init__(self, ax, num_losses, maxlen=1000, downsample=10):
        self.steps = 0
        self.history = [deque(maxlen=maxlen) for _ in range(num_losses)]
        self.ax = ax
        self.downsample = downsample
        self.weights = torch.ones(1, 1, downsample) / downsample

    def update(self, *args):
        assert len(args) == len(self.history), f"expected num_losses = {len(self.history)} losses"
        for i in range(len(self.history)):
            self.history[i] += [args[i]]
        self.steps += 1

    def plot(self):
        with torch.no_grad():
            self.ax.cla()
            for i in range(len(self.history)):
                loss_value = torch.tensor([self.history[i]]).float()
                loss_value = conv1d(loss_value, weight=self.weights, stride=self.downsample).squeeze()
                x = np.arange(self.steps - len(loss_value) * self.downsample, self.steps, step=self.downsample)
                self.ax.plot(x, loss_value)


class PlotImage:
    def __init__(self, ax):
        self.ax = ax
        self.image = None

    def imshow(self, image):
        if self.image is None:
            self.image = self.ax.imshow(image)
        else:
            self.image.set_data(image)


if __name__ == '__main__':
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # some random data
    x = np.random.randn(1000) + 2.0
    y = np.random.randn(1000) - 1.0

    loss = list(range(200))
    loss2 = list(range(20, 220))

    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(constrained_layout=True)

    # Create the main axes, leaving 25% of the figure space at the top and on the
    # right to position marginals.
    ax = fig.add_gridspec(nrows=2, ncols=2, top=0.75, right=0.75).subplots()

    joint1_plot = PlotJointAndMarginals(ax[0, 0], 'title', 'ylabel')
    joint2_plot = PlotJointAndMarginals(ax[1, 0], ylabel='ylabel')
    # Draw the scatter plot and marginals.
    joint1_plot.scatter_hist(x, y)
    joint2_plot.scatter_hist(x, y)
    loss1_plot = PlotLosses(ax[0, 1], 2)
    for loss, loss2 in zip(loss, loss2):
        loss1_plot.update(loss, loss2)
    loss1_plot.plot()

    plt.show()
