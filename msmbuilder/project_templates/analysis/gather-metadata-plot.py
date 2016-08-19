"""Plot metadata info

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.io import load_meta, render_meta

sns.set_style('ticks')
colors = sns.color_palette()

## Load
meta = load_meta()


## Histogram of trajectory lengths
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


## Pie graph
def plot_pie(ax):
    lengths_ns = meta['nframes'] * (meta['step_ps'] / 1000)
    sampling = lengths_ns.groupby(level=0).sum()

    ax.pie(sampling,
           shadow=True,
           labels=sampling.index,
           colors=sns.color_palette(),
           )
    ax.axis('equal')


## Box plot
def plot_boxplot(ax):
    meta2 = meta.copy()
    meta2['ns'] = meta['nframes'] * (meta['step_ps'] / 1000)
    sns.boxplot(
        x=meta2.index.names[0],
        y='ns',
        data=meta2.reset_index(),
        ax=ax,
    )


## Plot hist
fig, ax = plt.subplots(figsize=(7, 5))
plot_lengths(ax)
fig.tight_layout()
fig.savefig("lengths-hist.pdf")
# {{xdg_open('lengths-hist.pdf')}}

## Plot pie
fig, ax = plt.subplots(figsize=(7, 5))
plot_pie(ax)
fig.tight_layout()
fig.savefig("lengths-pie.pdf")
# {{xdg_open('lengths-pie.pdf')}}

## Plot box
fig, ax = plt.subplots(figsize=(7, 5))
plot_boxplot(ax)
fig.tight_layout()
fig.savefig("lengths-boxplot.pdf")
# {{xdg_open('lengths-boxplot.pdf')}}

## Save metadata as html table
render_meta(meta, 'meta.pandas.html')
