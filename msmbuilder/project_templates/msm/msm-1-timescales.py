"""Calculate implied timescales vs. lagtime

{{header}}

Meta
----
depends:
 - meta.pandas.pickl
 - ktrajs
"""
from multiprocessing import Pool

import pandas as pd

from msmbuilder.io import load_trajs
from msmbuilder.msm import MarkovStateModel

## Load
meta, ktrajs = load_trajs('ktrajs')

## Parameters
lagtimes = [2 ** i for i in range(8)]


## Define what to do for parallel execution
def at_lagtime(lt):
    msm = MarkovStateModel(lag_time=lt, n_timescales=10, verbose=False)
    msm.fit(list(ktrajs.values()))
    ret = {
        'lag_time': lt,
        'percent_retained': msm.percent_retained_,
    }
    for i in range(msm.n_timescales):
        ret['timescale_{}'.format(i)] = msm.timescales_[i]
    return ret


## Do the calculation
with Pool() as p:
    results = p.map(at_lagtime, lagtimes)

lt_df = pd.DataFrame(results)

## Save
print(lt_df.head())
lt_df.to_pickle('timescales.pandas.pickl')
