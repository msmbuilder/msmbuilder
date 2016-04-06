"""Plot metadata info

{{header}}
"""
from subprocess import run

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.io import load_meta, render_meta

sns.set_style('ticks')
colors = sns.color_palette()

## Load
meta = load_meta()


## Plot logic
def plot_lengths(ax):
    lengths_ns = meta['nframes'] * (meta['step_ps'] / 1000)
    ax.hist(lengths_ns)
    ax.set_xlabel("Lenths / ns", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)

    total_label = ("Total length: {us:.2f}"
                   .format(us=np.sum(lengths_ns) / 1000))
    total_label += r" / $\mathrm{\mu s}$"
    ax.annotate(total_label,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=18,
                va='top',
                )


## Plot
fig, ax = plt.subplots(figsize=(7, 5))
plot_lengths(ax)
fig.tight_layout()
fig.savefig("lengths.pdf")
run(['xdg-open', 'lengths.pdf'])

## Save metadata as html table
render_meta(meta, 'meta.pandas.html')
