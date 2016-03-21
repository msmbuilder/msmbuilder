from subprocess import run

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.dataset2 import load_trajs

sns.set_style('ticks')
colors = sns.color_palette()

meta, ftrajs = load_trajs('diheds')
vmxx = np.concatenate(list(ftrajs.values()))


def plot_box(ax):
    n_feats_plot = min(vmxx.shape[1], 100)
    ax.boxplot(vmxx[::100, :100],
               boxprops={'color': colors[0]},
               whiskerprops={'color': colors[0]},
               capprops={'color': colors[0]},
               medianprops={'color': colors[2]},
               )

    if vmxx.shape[1] > 100:
        ax.annotate("(Only showing the first 100 features)",
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    fontsize=14,
                    va='top',
                    )

    ax.set_xlabel("Feature Index", fontsize=16)
    xx = np.arange(0, n_feats_plot, 10)
    ax.set_xticks(xx)
    ax.set_xticklabels([str(x) for x in xx])
    ax.set_xlim((0, n_feats_plot + 1))
    ax.set_ylabel("Feature Value", fontsize=16)


fig, ax = plt.subplots(figsize=(15, 5))
plot_box(ax)
fig.tight_layout()
fig.savefig("ftrajs-box.pdf")
run(['xdg-open', 'ftrajs-box.pdf'])
