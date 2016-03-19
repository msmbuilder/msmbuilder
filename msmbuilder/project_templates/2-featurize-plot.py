import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('ticks')
colors = sns.color_palette()

from msmbuilder.dataset2 import load
import numpy as np

meta, vmtrajs = load('meta.pandas.pickl', 'vmtrajs')
vmxx = np.concatenate(list(vmtrajs.values()))


def plot_box(ax):
    ax.boxplot(vmxx[::100, :100],
               boxprops={'color': colors[0]},
               whiskerprops={'color': colors[0]},
               capprops={'color': colors[0]},
               medianprops={'color': colors[2]},
               )
    ax.set_xlabel("Feature Index", fontsize=16)
    xx = np.arange(0, 100, 10)
    ax.set_xticks(xx)
    ax.set_xticklabels([str(x) for x in xx])
    ax.set_ylabel("Feature Value", fontsize=16)


fig, ax = plt.subplots(figsize=(15, 5))
plot_box(ax)
fig.tight_layout()
fig.savefig("vmtrajs-box.pdf")
