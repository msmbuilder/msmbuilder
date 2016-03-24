"""Plot implied timescales vs lagtime

{{header}}
"""

from subprocess import run

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


## Plot
fig, ax = plt.subplots(figsize=(7, 5))
plot_timescales(ax)
fig.tight_layout()
fig.savefig('implied-timescales.pdf')
run(['xdg-open', 'implied-timescales.pdf'])
