import seaborn as sns
from matplotlib import pyplot as plt
from subprocess import run

sns.set_style('ticks')
colors = sns.color_palette()

from msmbuilder.dataset2 import load
import numpy as np

meta, ttrajs = load('meta.pandas.pickl', 'ttrajs')
txx = np.concatenate(list(ttrajs.values()))


def plot_heatmap(ax):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap=sns.cubehelix_palette(as_cmap=True),
              mincnt=1,
              bins='log'
              )
    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)


fig, ax = plt.subplots(figsize=(7, 5))
plot_heatmap(ax)
fig.tight_layout()
fig.savefig('tica-heatmap.pdf')
run(['xdg-open', 'tica-heatmap.pdf'])
