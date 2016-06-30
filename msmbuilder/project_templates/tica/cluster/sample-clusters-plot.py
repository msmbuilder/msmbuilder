"""Plot the result of sampling clusters

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.io import load_trajs, load_generic

sns.set_style('ticks')
colors = sns.color_palette()

## Load
meta, ttrajs = load_trajs('ttrajs')
txx = np.concatenate(list(ttrajs.values()))
kmeans = load_generic('kmeans.pickl')

inds = load_generic("cluster-sample-inds.pickl")
coordinates = [
    np.asarray([ttrajs[traj_i][frame_i, :] for traj_i, frame_i in inds])
    for state_inds in inds
    ]


## Overlay sampled states on histogram
def plot_sampled_states(ax):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap='magma_r',
              mincnt=1,
              bins='log',
              alpha=0.8,
              )

    # Show sampled points as scatter
    # Annotate cluster index
    for i, coo in enumerate(coordinates):
        plt.scatter(coo[:, 0], coo[:1], c=colors[i % 8], s=40)
        ax.text(kmeans.cluster_centers_[i, 0],
                kmeans.cluster_centers_[i, 1],
                "{}".format(i),
                ha='center',
                va='center',
                size=16,
                bbox=dict(
                    boxstyle='round',
                    fc='w',
                    ec="0.5",
                    alpha=0.9,
                ),
                zorder=10,
                )

    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)


## Plot
fig, ax = plt.subplots(figsize=(7, 5))
plot_sampled_states(ax)
fig.tight_layout()
fig.savefig('tica-dimension-0-heatmap.pdf')
# {{xdg_open('tica-dimension-0-heatmap.pdf')}}
