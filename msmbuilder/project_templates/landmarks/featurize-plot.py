"""Plot statistics from RMSD clustering

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
meta, ktrajs = load_trajs('ktrajs')
kxx = np.concatenate(list(ktrajs.values()))


## Scatter number of conformations in each state
def plot_num_per_state(ax):
    num_per_state = np.bincount(kxx)
    ax.scatter(np.arange(len(num_per_state)), num_per_state,
               s=40,
               c=colors[0],
               )
    ax.set_xlabel("State Index", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)


## Histogram number of conformations in each state
def plot_num_per_state_hist(ax):
    num_per_state = np.bincount(kxx)
    ax.hist(num_per_state)
    ax.set_xlabel("Conformations / State", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)


## Histogram cluster centroid distances
def plot_dist_histogram(ax):
    # This isn't really possible yet. We need to add something
    # to msmbuilder where we can transform a trajectory into an array
    # of minimum distances
    pass


## Plot 1
fig, ax = plt.subplots(figsize=(7, 5))
plot_num_per_state(ax)
fig.tight_layout()
fig.savefig('ktrajs-statecount.pdf')
# {{ xdg_open('ktrajs-statecount.pdf') }}

## Plot 2
fig, ax = plt.subplots(figsize=(7, 5))
plot_num_per_state_hist(ax)
fig.tight_layout()
fig.savefig('ktrajs-statehist.pdf')
# {{ xdg_open('ktrajs-statehist.pdf') }}
