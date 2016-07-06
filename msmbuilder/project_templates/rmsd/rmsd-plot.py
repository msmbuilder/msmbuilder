"""Plot RMSD results

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.io import load_trajs

sns.set_style('ticks')
colors = sns.color_palette()

## Load
meta, rmsds = load_trajs('rmsds')


## Plot box plot
def plot_boxplot(ax):
    catted = np.concatenate([rmsds[k] for k in meta.index])
    sns.boxplot(catted * 10, ax=ax)
    ax.set_xlabel(r'RMSD / $\mathrm{\AA}$', fontsize=18)
    ax.set_yticks([])
    # ax.set_xticks(fontsize=16) #TODO: fontsize


## Report bad trajectories
def bad_trajs(cutoff=0.7):
    bad = {}
    for k in meta.index:
        arr = rmsds[k]
        wh = np.where(np.asarray(arr) > cutoff)[0]
        if len(wh) > 0:
            bad[k] = wh
    return bad


## Plot
fig, ax = plt.subplots(figsize=(6, 3))
plot_boxplot(ax)
fig.tight_layout()
fig.savefig("rmsd-boxplot.pdf")
# {{xdg_open('rmsd-boxplot.pdf')}}

## Bad trajectories
for k, frame_is in bad_trajs().items():
    print("Trajectory", k)
    print("Frames:", frame_is)
