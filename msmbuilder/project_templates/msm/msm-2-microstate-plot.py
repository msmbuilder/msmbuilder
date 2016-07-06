"""Plot populations and eigvectors from microstate MSM

{{header}}
Meta
----
depends:
 - kmeans.pickl
 - ../ttrajs
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
kmeans = load_generic('kmeans.pickl')
msm = load_generic('msm.pickl')
meta, ttrajs = load_trajs('ttrajs')
txx = np.concatenate(list(ttrajs.values()))


## Plot microstates
def plot_microstates(ax):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap='Greys',
              mincnt=1,
              bins='log',
              )

    scale = 100 / np.max(msm.populations_)
    add_a_bit = 5
    ax.scatter(kmeans.cluster_centers_[msm.state_labels_, 0],
               kmeans.cluster_centers_[msm.state_labels_, 1],
               s=scale * msm.populations_ + add_a_bit,
               c=msm.left_eigenvectors_[:, 1],
               cmap='RdBu'
               )
    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)
    # ax.colorbar(label='First Dynamical Eigenvector', fontsize=16)


## Plot
fig, ax = plt.subplots(figsize=(7, 5))
plot_microstates(ax)
fig.tight_layout()
fig.savefig('msm-microstates.pdf')
# {{xdg_open('msm-microstates.pdf')}}
