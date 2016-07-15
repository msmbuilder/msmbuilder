"""Plot implied timescales vs lagtime

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('ticks')
colors = sns.color_palette()

## Load
timescales = pd.read_pickle('timescales.pandas.pickl')
n_timescales = len([x for x in timescales.columns
                    if x.startswith('timescale_')])


## Implied timescales vs lagtime
def plot_timescales(ax):
    for i in range(n_timescales):
        ax.scatter(timescales['lag_time'],
                   timescales['timescale_{}'.format(i)],
                   s=50, c=colors[0],
                   label=None,  # pandas be interfering
                   )

    xmin, xmax = ax.get_xlim()
    xx = np.linspace(xmin, xmax)
    ax.plot(xx, xx, color=colors[2], label='$y=x$')
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('Lag Time / todo:units', fontsize=18)
    ax.set_ylabel('Implied Timescales / todo:units', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')

## Percent trimmed vs lagtime
def plot_trimmed(ax):
    ax.plot(timescales['lag_time'],
            timescales['percent_retained'],
            'o-',
            label=None,  # pandas be interfering
            )
    ax.axhline(100, color='k', ls='--', label='100%')
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('Lag Time / todo:units', fontsize=18)
    ax.set_ylabel('Retained / %', fontsize=18)
    ax.set_xscale('log')
    ax.set_ylim((0, 110))

## Plot timescales
fig, ax = plt.subplots(figsize=(7, 5))
plot_timescales(ax)
fig.tight_layout()
fig.savefig('implied-timescales.pdf')
# {{xdg_open('implied-timescales.pdf')}}

## Plot trimmed
fig, ax = plt.subplots(figsize=(7,5))
plot_trimmed(ax)
fig.tight_layout()
fig.savefig('percent-trimmed.pdf')
# {{xdg_open('percent-trimmed.pdf')}}
